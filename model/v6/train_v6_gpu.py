# -*- coding: utf-8 -*-
"""
v6 training on Kaggle **T4 x2** (`machine_shape: NvidiaTeslaT4`). Model rationale: ARCHITECTURE.md.

WHY GPU AND NOT TPU. Two full Kaggle TPU queue cycles produced zero training:
  1. `xmp.spawn(nprocs=8)` -- rejected under PJRT (TPU_NOTES #13). Fixed.
  2. `TPU initialization failed: Invalid --..._slice_builder_worker_addresses. Expected 8 worker
     addresses, got 1` -- TPU v5e multi-process init on a single Kaggle VM (TPU_NOTES #15).
Each cost hours of queue for a start-up traceback, and the TPU was measured **slower per core than a T4**
anyway [8]. This path is the v5 trainer's proven GPU setup, which trained v5 end-to-end without incident.

The three guards below are not boilerplate -- each is a lesson that cost real quota:
  * no CPU fallback: a GPU kernel that silently runs on CPU still burns the session (billed by wall-clock,
    not utilisation). Fail fast.
  * P100 probe: Kaggle's torch ships no sm_60 kernels, so a P100 assignment dies on the FIRST op with
    "no kernel image is available". A 1-second matmul probe turns an 8-hour waste into an instant error.
  * resumable checkpoint + time budget: a non-clean exit discards /kaggle/working entirely.

Run via the Kaggle kernel; locally `python model/v6/train_v6_gpu.py --epochs 1 --batch 8` needs a CUDA GPU.
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_v6 import V6Config, V6DataConfig, resolve_paths
from model_v6 import LincsV6
from data import LincsDataset, build_splits, collate
import losses as L


def probe_gpu():
    """Fail fast and loudly rather than burn a session. See the module docstring."""
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: no CUDA device. Refusing to train on CPU -- it would burn the whole GPU "
                         "session while using no GPU. Provision as T4x2 (machine_shape NvidiaTeslaT4).")
    try:
        _ = (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
    except RuntimeError as e:
        raise SystemExit(f"FATAL: GPU present but unusable ({e}). Almost certainly a P100 (sm_60), for "
                         f"which Kaggle's torch has no kernels. Re-provision as NvidiaTeslaT4.")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if any("P100" in n for n in names):
        raise SystemExit(f"FATAL: P100 assigned ({names}). Re-provision as NvidiaTeslaT4.")
    return names


def check_inputs(ds, sp):
    """Refuse to train on a silently degraded setup.

    `data.py` falls back to neutral defaults when an input file is absent instead of failing, so a dataset
    that is merely MISSING A FILE trains happily and produces numbers that look fine and mean something
    else. This actually happened (2026-08-14): rebuilding the Kaggle model-src dataset from a PAGINATED
    file listing silently dropped two files, and the run that followed had

      * no `scaffold_split.json` -> no compound holdout: test_colddrug/test_coldboth EMPTY and their
        signatures folded into train (235,628 vs 179,772) => the model trains on the very compounds later
        scored as "unseen". Leakage straight into the headline unseen-compound number.
      * no `sig_strength.npy`  -> strength and weight become ALL ONES: reliability weighting off, and the
        `strength >= 1` reproducible-stratum filter passes everything (rule #1 silently disabled).

    Neither raised anything. Eight and a half hours of GPU would have produced an invalid model."""
    problems = []
    for k in ("test_colddrug", "test_coldboth"):
        if len(sp[k]) == 0:
            problems.append(f"{k} is EMPTY -> scaffold_split.json was not found; there is no compound "
                            f"holdout and train has absorbed those signatures (leakage)")
    if float(ds.strength.min()) == 1.0 and float(ds.strength.max()) == 1.0:
        problems.append("per-signature strength is identically 1.0 -> sig_strength.npy was not found; "
                        "reliability weighting is OFF and the reproducible-stratum filter passes everything")
    if float(ds.weight.min()) == 1.0 and float(ds.weight.max()) == 1.0:
        problems.append("per-signature weight is identically 1.0 -> reliability weighting is not active")
    if problems:
        raise SystemExit("FATAL: input files are missing and data.py fell back to neutral defaults:\n  - "
                         + "\n  - ".join(problems)
                         + "\nRefusing to train: this would burn the session on an invalid model.")
    print(f"input check OK: strength {ds.strength.min():.2f}-{ds.strength.max():.2f}, "
          f"weight {ds.weight.min():.3f}-{ds.weight.max():.3f}, compound holdout present", flush=True)


@torch.no_grad()
def quick_eval(model, ds, idx, cfg, device, batch=64, max_n=3000):
    """Per-epoch progress signal on the REPRODUCIBLE stratum only (rule #1: ~75 % of LINCS is inert, and
    evaluating on everything once inverted a real effect's sign [6.2]). Deliberately small -- the real
    evaluation is eval_v6.py on the saved checkpoint."""
    if len(idx) == 0:
        return {}
    if len(idx) > max_n:
        idx = np.random.default_rng(0).choice(idx, max_n, replace=False)
    model.eval()
    yh, yt = [], []
    for s in range(0, len(idx), batch):
        b = collate([ds[i] for i in idx[s:s + batch]], cfg.max_atoms)
        bd = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        yh.append(model(bd).float().cpu()); yt.append(b["Y"])
    model.train()
    yh, yt = torch.cat(yh), torch.cat(yt)
    return {"pearson_median": float(L.pearson_per_row(yh, yt).median()),
            "r2_overall": L.r2_overall(yh, yt), "n": int(len(yt))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=48)       # split across both T4s by DataParallel
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--budget_h", type=float, default=8.5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    names = probe_gpu()
    torch.backends.cudnn.benchmark = True                  # static gene-token length -> autotuned kernels
    device = "cuda"; n_gpu = torch.cuda.device_count()
    print(f"GPUs: {names}", flush=True)

    dc = resolve_paths(V6DataConfig()); cfg = V6Config()
    dc.cell_fold = a.fold
    WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    CKPT = f"{WORK}/ckpt_v6_fold{a.fold}.pt"

    M = np.load(dc.m_reactome_path if os.path.isabs(dc.m_reactome_path)
                else os.path.join(dc.root, dc.m_reactome_path))
    core = LincsV6(cfg, M).to(device)                      # unwrapped: state_dict + cfg live here
    print(f"params {sum(p.numel() for p in core.parameters())/1e6:.2f}M", flush=True)
    # Use BOTH T4s. DataParallel splits the batch (48 -> 24+24), so len(dl) and therefore the OneCycle
    # step count are UNCHANGED -- a resumed schedule stays exact.
    model = torch.nn.DataParallel(core) if n_gpu > 1 else core
    if n_gpu > 1:
        print(f"DataParallel across {n_gpu} GPUs (per-GPU batch {a.batch // n_gpu})", flush=True)

    shared = LincsDataset.load_shared(dc)
    full = LincsDataset(dc, _shared=shared)
    sp = build_splits(full, dc)
    print(f"splits={ {k: len(v) for k, v in sp.items() if not k.startswith('_')} }", flush=True)
    check_inputs(full, sp)
    train_ds = LincsDataset(dc, indices=sp["train"], _shared=shared)
    # fixed_pad is a TPU requirement (static shapes); on GPU dynamic padding is correct AND faster, and
    # the two are proven to give identical predictions (test_v6: padding invariance, 2.7e-07).
    dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=True,
                    collate_fn=lambda s: collate(s, cfg.max_atoms))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * max(1, len(dl)),
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, hist = 0, []
    if os.path.exists(CKPT):                               # resume after a budget-triggered stop
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start_epoch = ck["epoch"] + 1
        hist = ck.get("hist", [])
        print(f"resumed from {CKPT} at epoch {start_epoch}", flush=True)

    rep = sp["test_coldcell"][full.strength[sp["test_coldcell"]] >= dc.eval_min_strength]
    t0 = time.time()
    model.train()
    for epoch in range(start_epoch, a.epochs):
        te = time.time(); run = 0.0; n = 0
        for it, b in enumerate(dl):
            bd = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = L.weighted_huber(model(bd), bd["Y"], w=bd.get("w"), delta=cfg.huber_delta)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss); n += 1
            if it % 200 == 0:
                print(f"  e{epoch} it{it}/{len(dl)} loss {float(loss):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        recd = {"epoch": epoch, "train_loss": run / max(n, 1), "steps": n,
                "sec": round(time.time() - te, 1), "elapsed_h": round((time.time() - t0) / 3600, 3),
                "cold_reproducible": quick_eval(model, full, rep, cfg, device)}
        hist.append(recd); print("EPOCH", json.dumps(recd), flush=True)
        torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "hist": hist, "cfg": vars(cfg)}, CKPT)
        json.dump(hist, open(f"{WORK}/metrics_v6_fold{a.fold}.json", "w"), indent=2)
        if (time.time() - t0) / 3600 > a.budget_h:
            print("time budget reached -> clean checkpoint + stop (resumable)", flush=True)
            break
    print("DONE", flush=True)   # full evaluation + ablation: eval_v6.py on the checkpoint, locally on CPU


if __name__ == "__main__":
    main()
