# -*- coding: utf-8 -*-
"""
v6 training on Kaggle TPU v3-8 (torch_xla, 8 cores). Engineering rationale for every choice here:
TPU_NOTES.md (same directory). Model rationale: ARCHITECTURE.md.

IMPORTANT: this module must be started via `launch_v6_tpu.py`, which sets the bf16 environment BEFORE
torch_xla is imported (issue #4 in TPU_NOTES.md -- setting it afterwards silently does nothing).

TPU-specific design, in the order it matters:
  * static shapes everywhere (fixed atom padding)         -> no recompilation  [#1]
  * warm-up before timing                                  -> honest ms/step   [#2]
  * loss accumulated ON DEVICE, one sync per epoch         -> no host stalls   [#3]
  * dataset keys cached to .npz, parsed ONCE not 8x        -> fast start, less RAM [#5]
  * 2 loader workers per process (8 procs on one VM)       -> no thrash        [#6]
  * checkpoint via xm.save; NO per-epoch .to("cpu")        -> no resync        [#7]
  * eval/probe deferred to a separate CPU pass             -> no shape churn   [#9]
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config_v6 import V6Config, V6DataConfig
from model_v6 import LincsV6
from data import LincsDataset, build_splits, collate
import losses as L


def resolve_paths(dc):
    """Kaggle mounts inputs flat under /kaggle/input/**; find each file rather than assuming a layout."""
    import glob as _g
    if not os.path.isdir("/kaggle/input"):
        return dc
    hit = lambda n: next(iter(_g.glob(f"/kaggle/input/**/{n}", recursive=True)), n)
    bundle = os.path.dirname(hit("Y_target_level5_978.npy"))
    dc.root = bundle
    for attr, name in [("y_path", "Y_target_level5_978.npy"), ("sig_path", "signatures_usable.tsv"),
                       ("xbase_path", "X_base_lincs.npy"), ("e_path", "E_final.npy"),
                       ("e_mask_path", "E_final_mask.npy"), ("e_reliability_path", "E_reliability.tsv"),
                       ("cop_path", "A_copathway.npy"), ("ppi_path", "STRING_adj_978.npy"),
                       ("drug_index_path", "drug_feature_index.json"), ("desc_path", "drug_descriptors.npy"),
                       ("fp_path", "drug_fingerprints.npy"), ("unimol_cls_path", "drug_unimol.npy"),
                       ("chemberta_path", "drug_chemberta.npy"), ("atom_reprs_path", "drug_atom_reprs.npy"),
                       ("atom_offsets_path", "drug_atom_offsets.npy"), ("strength_path", "sig_strength.npy"),
                       ("lineage_path", "cell_lineage.npy"), ("m_reactome_path", "M_reactome.npy"),
                       ("scaffold_split_path", "scaffold_split.json")]:
        setattr(dc, attr, hit(name))
    return dc


def _run(rank, args):
    dc = resolve_paths(V6DataConfig()); cfg = V6Config()
    dc.cell_fold = args.fold
    device = torch_xla.device()
    world, ordinal = xr.world_size(), xr.global_ordinal()
    is_master = ordinal == 0

    M = np.load(dc.m_reactome_path if os.path.isabs(dc.m_reactome_path)
                else os.path.join(dc.root, dc.m_reactome_path))
    model = LincsV6(cfg, M).to(device)

    shared = LincsDataset.load_shared(dc)
    full = LincsDataset(dc, _shared=shared)
    sp = build_splits(full, dc)
    train_ds = LincsDataset(dc, indices=sp["train"], _shared=shared)
    if is_master:
        print(f"cores={world} fold={args.fold} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
              flush=True)
        print(f"splits={ {k: len(v) for k, v in sp.items() if not k.startswith('_')} }", flush=True)

    sampler = DistributedSampler(train_ds, num_replicas=world, rank=ordinal, shuffle=True, drop_last=True)
    dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, drop_last=True,
                    num_workers=2, persistent_workers=True, prefetch_factor=4,   # [#6]
                    collate_fn=lambda s: collate(s, cfg.max_atoms, fixed_pad=True))  # [#1] static shapes
    mp_dl = pl.MpDeviceLoader(dl, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * max(1, len(dl)), pct_start=0.05)
    WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    autocast = (lambda: torch.autocast("xla", dtype=torch.bfloat16)) if args.bf16 else \
               (lambda: torch.autocast("xla", enabled=False))     # [#4] no env-var ordering hazard

    def step(b):
        opt.zero_grad(set_to_none=True)
        with autocast():
            loss = L.weighted_huber(model(b), b["Y"], w=b.get("w"), delta=cfg.huber_delta)
        loss.backward()
        xm.optimizer_step(opt)
        return loss.detach()

    # ---------- warm-up: pay compilation OUTSIDE the timer [#2] ----------
    model.train()
    t_warm = time.time()
    for i, b in enumerate(mp_dl):
        last = step(b)
        if i + 1 >= 3:
            break
    torch_xla.sync(); _ = float(last)
    if is_master:
        print(f"warm-up + XLA compile: {time.time()-t_warm:.0f}s", flush=True)

    hist = []; t0 = time.time()
    for epoch in range(args.epochs):
        model.train(); sampler.set_epoch(epoch)
        run = torch.zeros((), device=device); n = 0                 # [#3] stays on device
        te = time.time()
        for b in mp_dl:
            run += step(b); sched.step(); n += 1
        train_loss = float(run) / max(n, 1)                          # ONE sync per epoch
        if is_master:
            dt = time.time() - te
            rec = {"epoch": epoch, "train_loss": train_loss, "steps": n,
                   "sec": round(dt, 1), "ms_per_sample": round(dt / max(n * args.batch, 1) * 1000, 2),
                   "elapsed_h": round((time.time() - t0) / 3600, 3)}
            hist.append(rec); print("EPOCH", json.dumps(rec), flush=True)
            json.dump(hist, open(f"{WORK}/metrics_v6_fold{args.fold}.json", "w"), indent=2)
        xm.save(model.state_dict(), f"{WORK}/ckpt_v6_fold{args.fold}.pt")   # [#7] handles device transfer
        if (time.time() - t0) / 3600 > args.budget_h:
            if is_master:
                print("time budget reached -> clean checkpoint + stop (resumable)", flush=True)  # [#11]
            break
    if is_master:
        print("DONE", flush=True)      # eval/probe run separately on CPU from the checkpoint [#9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)      # per core -> 256 effective across 8 [#8]
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--budget_h", type=float, default=8.0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    xmp.spawn(_run, args=(a,), nprocs=a.cores)


if __name__ == "__main__":
    main()
