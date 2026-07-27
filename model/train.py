# -*- coding: utf-8 -*-
"""
Train LincsCrossAttn. Runs on Kaggle GPU (auto-detects the training-bundle mount) or locally.
Deterministic predictor of Y (978-gene differential). Reports R²/magnitude + Pearson on val and the
COLD-CELL test (MCF10A/NPC), vs Mean/Meancell/Meandrug baselines. Checkpoints to /kaggle/working
(resumable). DTI recall@k (learned atom->gene vs dti_reference) is a separate eval hook (eval.py).

Usage: python train.py            # auto-detects environment
"""
import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, DataConfig
from data import LincsDataset, build_splits, collate
from model import LincsCrossAttn
import losses as L


def find_bundle():
    """On Kaggle, locate the training-bundle dir (contains Y_target_level5_978.npy) under /kaggle/input."""
    if os.path.isdir("/kaggle/input"):
        hits = glob.glob("/kaggle/input/**/Y_target_level5_978.npy", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def make_dataconfig():
    bundle = find_bundle()
    if bundle is None:
        return DataConfig()                       # local layout
    # Kaggle: all inputs flat in the bundle dir; sig_strength may ship with model-src instead
    f = lambda n: n
    st = glob.glob("/kaggle/input/**/sig_strength.npy", recursive=True)
    strength = st[0] if st else "sig_strength.npy"          # absolute path wins over root join
    lg = glob.glob("/kaggle/input/**/cell_lineage.npy", recursive=True)
    lineage = lg[0] if lg else "cell_lineage.npy"           # ships with model-src, not the big bundle
    ss = glob.glob("/kaggle/input/**/scaffold_split.json", recursive=True)
    scaffold = ss[0] if ss else "scaffold_split.json"       # unseen-compound holdout (model-src)
    return DataConfig(
        strength_path=strength, lineage_path=lineage, scaffold_split_path=scaffold,
        root=bundle, y_path=f("Y_target_level5_978.npy"), sig_path=f("signatures_usable.tsv"),
        xbase_path=f("X_base_lincs.npy"), background_path=f("ccle_full_background.npy"),
        e_path=f("E_final.npy"), e_mask_path=f("E_final_mask.npy"), e_reliability_path=f("E_reliability.tsv"),
        cop_path=f("A_copathway.npy"), ppi_path=f("STRING_adj_978.npy"),
        drug_index_path=f("drug_feature_index.json"), desc_path=f("drug_descriptors.npy"),
        fp_path=f("drug_fingerprints.npy"), unimol_cls_path=f("drug_unimol.npy"),
        chemberta_path=f("drug_chemberta.npy"), atom_reprs_path=f("drug_atom_reprs.npy"),
        atom_offsets_path=f("drug_atom_offsets.npy"))


@torch.no_grad()
def evaluate(model, ds, idx, dc, device, batch=64, max_n=20000, ablate_epi=False, max_atoms=96):
    """R²/Pearson/MSE on a split (subsampled for speed), stratified by phase.
    ablate_epi=True zeroes E and r (gate -> 1) to measure how much epigenetics actually contributes.
    max_atoms is passed explicitly (not read off model.cfg) so this works when model is DataParallel-wrapped."""
    model.eval()
    if len(idx) > max_n:
        idx = np.random.default_rng(0).choice(idx, max_n, replace=False)
    sub = LincsDataset(dc, indices=idx, _shared=ds.__dict__)
    dl = DataLoader(sub, batch_size=batch, collate_fn=lambda s: collate(s, max_atoms),
                    num_workers=2)
    yh, yt = [], []
    for b in dl:
        bd = {k: v.to(device) for k, v in b.items()}
        if ablate_epi:
            bd["E"] = torch.zeros_like(bd["E"]); bd["r"] = torch.zeros_like(bd["r"])
        yh.append(model(bd).float().cpu()); yt.append(b["Y"])
    yh = torch.cat(yh); yt = torch.cat(yt)
    out = {"r2_overall": L.r2_overall(yh, yt),
           "r2_gene_median": float(L.r2_per_gene(yh, yt).median()),
           "pearson_median": float(L.pearson_per_row(yh, yt).median()),
           "mse": float(((yh - yt) ** 2).mean()), "n": int(len(yt))}
    ph = ds.phase[idx]                      # detect residual batch effects (we do NOT model plate)
    for p in np.unique(ph):
        m = torch.from_numpy(ph == p)
        if int(m.sum()) > 50:
            out[f"r2_phase_{p}"] = L.r2_overall(yh[m], yt[m])
    return out


@torch.no_grad()
def gate_diagnostic(model, ds, device, max_sig_per_cell=400):
    """Post-hoc test of the chromatin hypothesis WITHOUT constraining the model (we deliberately did
    NOT add an output gate): does the learned gate s_{c,g} track how much gene g actually moves in
    cell c? Reports median over epi-covered cells of corr(s[c,:], mean|Y|[c,:]) across the 978 genes."""
    model.eval()
    s = model.gate(torch.tensor(ds.E, device=device), torch.tensor(ds.r_cell, device=device)).cpu().numpy()
    rng = np.random.default_rng(0); cors = []
    for c in np.unique(ds.cell_row):
        if ds.r_cell[c].max() <= 0:          # no epigenetics -> gate is identically 1, uninformative
            continue
        rows = np.where(ds.cell_row == c)[0]
        if len(rows) < 50:
            continue
        if len(rows) > max_sig_per_cell:
            rows = rng.choice(rows, max_sig_per_cell, replace=False)
        absY = np.abs(np.asarray(ds.Y[ds.y_row[rows]], np.float32)).mean(0)     # [978]
        sc = s[c]
        if absY.std() > 0 and sc.std() > 0:
            cors.append(float(np.corrcoef(sc, absY)[0, 1]))
    return {"gate_vs_absY_median_corr": float(np.median(cors)) if cors else None,
            "n_cells": len(cors)}


def build_probe(ds, strength_min=1.6, n_cells=6, max_sig=6, n_drug=15):
    """Fixed strong-stratum balanced (drug,cell) design for the per-epoch interaction probe (built once)."""
    from collections import defaultdict
    idx = np.where(ds.strength >= strength_min)[0]
    keep = defaultdict(list)
    for j in idx:
        keep[(int(ds.drug_row[j]), int(ds.cell_row[j]))].append(int(j))
    cc = defaultdict(set)
    for (d, c) in keep:
        cc[c].add(d)
    top = [c for c, _ in sorted(cc.items(), key=lambda x: -len(x[1]))[:n_cells]]
    if len(top) < 3:
        return None
    common = sorted(set.intersection(*[cc[c] for c in top]))[:n_drug]
    if len(common) < 6:
        return None
    return (common, top, {(d, c): keep[(d, c)][:max_sig] for d in common for c in top})


@torch.no_grad()
def interaction_probe(model, ds, device, design, max_atoms=96):
    """Instrumentation: on a FIXED strong-stratum design, how much drug x cell interaction does the model
    express? std(predI)/std(trueI) ~1 = full, ~0 = shrunk to the drug-average. Watches the fix working."""
    model.eval()
    drugs, cells, keep = design
    P = np.zeros((len(drugs), len(cells), 978), np.float32); T = np.zeros_like(P)
    for di, d in enumerate(drugs):
        for cj, c in enumerate(cells):
            b = collate([ds[i] for i in keep[(d, c)]], max_atoms)
            P[di, cj] = model({k: v.to(device) for k, v in b.items()}).float().cpu().numpy().mean(0)
            T[di, cj] = b["Y"].numpy().mean(0)

    def I(Z):
        g = Z.mean((0, 1)); return Z - g[None, None] - (Z.mean(1) - g)[:, None] - (Z.mean(0) - g)[None, :]
    IT, IP = I(T), I(P)
    return {"interaction_std_ratio": float(IP.std() / (IT.std() + 1e-9)),
            "interaction_corr": float(np.corrcoef(IT.ravel(), IP.ravel())[0, 1])}


def main():
    EPOCHS = int(os.environ.get("EPOCHS", 15)); BATCH = int(os.environ.get("BATCH", 48))
    LR = float(os.environ.get("LR", 3e-4)); TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 8.5))
    # v4: correlation-loss weight (magnitude-invariant pattern term to un-shrink the drug x cell interaction;
    # 0 = off = v3 behavior). FINETUNE=1 loads model weights only (fresh optimizer/schedule) for a short
    # low-LR adaptation run from the v3 checkpoint.
    LAMBDA_CORR = float(os.environ.get("LAMBDA_CORR", 0.0)); FINETUNE = int(os.environ.get("FINETUNE", 0))
    WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    CKPT = f"{WORK}/ckpt.pt"; t0 = time.time()
    # NO CPU FALLBACK for training: a GPU-provisioned kernel that runs on CPU still burns GPU quota
    # (billed by session wall-clock, not utilization) while using no GPU. Fail fast instead. CPU-only
    # kernels (unit tests) are FREE and run elsewhere -- this guard is only for the GPU train path.
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: no CUDA device. Refusing to train on CPU (would waste a GPU session). "
                         "Provision the kernel as T4x2 (never P100: sm_60 is unusable with Kaggle torch).")
    device = "cuda"
    # Fail fast on an unusable P100 (sm_60): Kaggle torch has no sm_60 kernels, so the FIRST op throws
    # "no kernel image is available". Probe with a tiny matmul now so a bad GPU assignment costs ~1s, not
    # a full data-load. (Policy: never P100; always T4x2 via --accelerator NvidiaTeslaT4.)
    try:
        _ = (torch.randn(16, 16, device=device) @ torch.randn(16, 16, device=device)).sum().item()
    except Exception as e:
        raise SystemExit(f"FATAL: GPU cannot execute kernels ({type(e).__name__}: {e}). "
                         "Likely a P100 (sm_60). Re-run with --accelerator NvidiaTeslaT4.")
    torch.backends.cudnn.benchmark = True              # static gene-seq len -> autotuned kernels
    n_gpu = torch.cuda.device_count()
    print(f"device={device} n_gpu={n_gpu} epochs={EPOCHS} batch={BATCH} lr={LR}", flush=True)

    mc = ModelConfig(); dc = make_dataconfig()
    R = lambda p: os.path.join(dc.root, p)
    cop = np.load(R(dc.cop_path)); ppi = np.load(R(dc.ppi_path))
    core = LincsCrossAttn(mc, cop, ppi).to(device)     # unwrapped model: ckpt / cfg / .gate live here
    print(f"params: {sum(p.numel() for p in core.parameters())/1e6:.1f}M", flush=True)
    # Use BOTH T4s. DataParallel splits BATCH across GPUs (48 -> 24+24), so len(dl) and the OneCycle
    # schedule are UNCHANGED -> resuming the saved scheduler from the in-flight ckpt stays exact.
    # (A larger effective batch would change the step count and desync the resumed schedule.)
    model = torch.nn.DataParallel(core) if n_gpu > 1 else core
    if n_gpu > 1:
        print(f"DataParallel across {n_gpu} GPUs (per-GPU batch {BATCH // n_gpu})", flush=True)

    shared = LincsDataset.load_shared(dc)
    full = LincsDataset(dc, _shared=shared)
    sp = build_splits(full, dc)
    print("splits:", {k: len(v) for k, v in sp.items()}, flush=True)
    train_ds = LincsDataset(dc, indices=sp["train"], _shared=shared)
    dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True,
                    collate_fn=lambda s: collate(s, mc.max_atoms), persistent_workers=True)

    opt = torch.optim.AdamW(core.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch = 0
    resume = CKPT if os.path.exists(CKPT) else next(iter(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)), None)
    if resume:
        ck = torch.load(resume, map_location=device)
        core.load_state_dict(ck["model"])
        if FINETUNE:                                # load weights only -> fresh short low-LR adaptation
            print(f"FINETUNE: loaded weights from {resume}; fresh optimizer/schedule, LAMBDA_CORR={LAMBDA_CORR}", flush=True)
        else:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start_epoch = ck["epoch"] + 1; print(f"RESUMED from {resume} at epoch {start_epoch}", flush=True)

    probe = build_probe(full)
    if probe:
        print(f"interaction probe: {len(probe[0])} drugs x {len(probe[1])} cells (strong stratum)", flush=True)

    hist = []
    for epoch in range(start_epoch, EPOCHS):
        model.train(); run = 0.0
        for it, b in enumerate(dl):
            bd = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                yhat = model(bd)
                loss = L.weighted_huber(yhat, bd["Y"], w=bd.get("w"), delta=mc.huber_delta)
                if LAMBDA_CORR > 0:                 # magnitude-invariant pattern term (un-shrink interaction)
                    loss = loss + LAMBDA_CORR * L.correlation_loss(yhat, bd["Y"], w=bd.get("w"))
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss)
            if it % 200 == 0:
                print(f"  e{epoch} it{it}/{len(dl)} loss {loss:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        val = evaluate(model, full, sp["val"], dc, device, max_n=8000)
        cold = evaluate(model, full, sp["test_coldcell"], dc, device, max_n=8000)
        # the metrics that actually mean something: REPRODUCIBLE sigs on each generalization axis
        thr = getattr(dc, "eval_min_strength", 0.0)
        def rep(split_name, n=6000):
            idx = sp.get(split_name)
            if idx is None or len(idx) == 0:
                return {}
            ridx = idx[full.strength[idx] >= thr]
            return evaluate(model, full, ridx, dc, device, max_n=n) if len(ridx) > 500 else {}
        cold_rep = rep("test_coldcell", 8000)
        rec = {"epoch": epoch, "train_loss": run / len(dl), "val": val, "cold_cell": cold,
               "cold_reproducible": cold_rep,
               "unseen_compound_reproducible": rep("test_colddrug"),
               "unseen_both_reproducible": rep("test_coldboth")}
        if probe:                                  # in-between instrumentation: is the interaction un-shrinking?
            rec["interaction_probe"] = interaction_probe(model, full, device, probe, mc.max_atoms)
        hist.append(rec); print("EPOCH", json.dumps(rec), flush=True)
        torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "cfg": vars(mc)}, CKPT)
        json.dump(hist, open(f"{WORK}/metrics.json", "w"), indent=2)
        if (time.time() - t0) / 3600 > TIME_BUDGET_H:
            print("time budget reached -> checkpoint + stop (resumable)", flush=True); break

    # ---------- final diagnostics ----------
    ci, tr = sp["test_coldcell"], sp["train"]
    rng = np.random.default_rng(0)
    trs = np.sort(rng.choice(tr, min(40000, len(tr)), replace=False))
    cis = np.sort(rng.choice(ci, min(20000, len(ci)), replace=False))
    bl = L.naive_baselines(np.asarray(full.Y[full.y_row[trs]], np.float32), full.cell_row[trs],
                           full.drug_row[trs], np.asarray(full.Y[full.y_row[cis]], np.float32),
                           full.cell_row[cis], full.drug_row[cis])
    print("COLD-CELL baselines (MSE):", json.dumps(bl), flush=True)

    # does epigenetics actually contribute? (guards against multimodal collapse)
    cold = evaluate(model, full, ci, dc, device)
    cold_abl = evaluate(model, full, ci, dc, device, ablate_epi=True)
    epi = {"r2_with_epi": cold["r2_overall"], "r2_epi_ablated": cold_abl["r2_overall"],
           "delta_r2": cold["r2_overall"] - cold_abl["r2_overall"]}
    print("EPI ABLATION (cold-cell):", json.dumps(epi), flush=True)

    # chromatin hypothesis, measured not imposed (we deliberately have no output gate)
    gd = gate_diagnostic(core, full, device)
    print("GATE DIAGNOSTIC:", json.dumps(gd), flush=True)

    # per-cell cold-cell R2 -- the phase gap is a cell-composition artifact, so report per cell not lumped
    percell = {}
    row2cell = {v: k for k, v in full.cell_idx.items()}
    for cell_row in sp["_test_cells"]:
        ci_c = ci[full.cell_row[ci] == cell_row]
        if len(ci_c) < 200:
            continue
        m = evaluate(model, full, ci_c, dc, device, max_n=6000)
        percell[row2cell.get(cell_row, cell_row)] = round(m["r2_overall"], 3)
    print("PER-CELL cold R2:", json.dumps(percell), flush=True)

    json.dump({"history": hist, "coldcell": cold, "coldcell_baselines": bl, "epi_ablation": epi,
               "gate_diagnostic": gd, "per_cell_cold_r2": percell,
               "test_cells": sp["_test_cells"], "fold_load": sp["_fold_load"]},
              open(f"{WORK}/metrics.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
