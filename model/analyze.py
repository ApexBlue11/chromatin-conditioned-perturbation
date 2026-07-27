# -*- coding: utf-8 -*-
"""
analyze.py — WHY is the drug x cell interaction under-expressed (26.5% pred vs 47.9% truth)? Diagnose
the CAUSE before any retraining (per the methodological rule: understand the mechanism first).

Hypotheses:
  H_shrink (loss)  : MSE shrinkage under noise. The model correctly hedges toward the drug-average on
                     UNreliable signatures, suppressing cell-specificity. If so, interaction expression
                     should IMPROVE (std ratio -> 1, corr up) as signature strength/reproducibility rises.
                     Fix would be a correlation/rank loss (magnitude-shrinkage-immune) + reliability weighting.
  H_capacity (arch): the cell-conditioning pathway can't express the interaction at all -> expression stays
                     low and flat across strength. Fix would be architecture (deeper cell conditioning).
  H_dead (bug)     : predictions barely vary across cells for a fixed drug -> pathway dead / math mismatch.

Test: build a balanced drug x cell design WITHIN signature-strength bins; measure, per bin,
      std(pred_interaction)/std(true_interaction) and corr(pred_I, true_I). Also a liveness check:
      cross-cell variance of predictions vs truth for fixed drugs. Kaggle CPU. Writes analyze.json.
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
import numpy as np
import torch

from config import ModelConfig
from data import LincsDataset, collate
from model import LincsCrossAttn
from train import make_dataconfig

BINS = [(0.30, 0.50), (0.50, 0.80), (0.80, 1.10), (1.10, 1.60), (1.60, 9.9)]
N_CELLS = 6
MAX_SIG = 6


def interaction(Z):
    """Z [D,C,G] -> pure drug x cell x gene interaction (gene effect + main effects removed)."""
    g = Z.mean((0, 1))
    return Z - g[None, None, :] - (Z.mean(1) - g)[:, None, :] - (Z.mean(0) - g)[None, :, :]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = make_dataconfig(); mc = ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).to(device).eval()
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    model.load_state_dict(torch.load(ck[0], map_location=device)["model"])
    print("loaded", ck[0], flush=True)
    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    rng = np.random.default_rng(0)

    def predict(rows):
        out = []
        for a in range(0, len(rows), 64):
            b = collate([ds[i] for i in rows[a:a + 64]], mc.max_atoms)
            with torch.no_grad():
                out.append(model({k: v.to(device) for k, v in b.items()}).float().cpu().numpy())
        return np.concatenate(out)

    def truth(rows):
        return np.stack([np.asarray(ds.Y[ds.y_row[i]], np.float32) for i in rows])

    def balanced_design(idx):
        keep = defaultdict(list)
        for j in idx:
            keep[(int(ds.drug_row[j]), int(ds.cell_row[j]))].append(int(j))
        cellcount = defaultdict(set)
        for (d, c) in keep:
            cellcount[c].add(d)
        top = [c for c, _ in sorted(cellcount.items(), key=lambda x: -len(x[1]))[:N_CELLS]]
        if len(top) < 3:
            return None
        common = sorted(set.intersection(*[cellcount[c] for c in top]))
        if len(common) < 8:
            return None
        return keep, top, common

    # ---- A. interaction expression stratified by signature strength ----
    print("\n== A. interaction expression vs signature strength ==", flush=True)
    A = {}
    for lo, hi in BINS:
        idx = np.where((ds.strength >= lo) & (ds.strength < hi))[0]
        if len(idx) > 80000:
            idx = rng.choice(idx, 80000, replace=False)
        bd = balanced_design(idx)
        if bd is None:
            print(f"  strength[{lo},{hi}): too few common (drug,cell) groups -> skip", flush=True); continue
        keep, top, common = bd
        P = np.zeros((len(common), len(top), 978), np.float32); T = np.zeros_like(P)
        for di, d in enumerate(common):
            for cj, c in enumerate(top):
                rws = keep[(d, c)][:MAX_SIG]
                P[di, cj] = predict(rws).mean(0)
                T[di, cj] = truth(rws).mean(0)
        IT, IP = interaction(T), interaction(P)
        ratio = float(IP.std() / (IT.std() + 1e-9))
        cor = float(np.corrcoef(IT.ravel(), IP.ravel())[0, 1])
        # also overall main-effect (drug x gene) tracking for context
        A[f"{lo}-{hi}"] = {"n_drugs": len(common), "n_cells": len(top),
                           "std_ratio_predI_over_trueI": ratio, "corr_interaction": cor,
                           "true_I_std": float(IT.std()), "pred_I_std": float(IP.std())}
        print(f"  strength[{lo},{hi}): {len(common)}drugs x {len(top)}cells | "
              f"std(predI)/std(trueI)={ratio:.3f}  corr(I)={cor:.3f}", flush=True)

    # ---- B. liveness: does the prediction vary across cells for a fixed drug at all? ----
    print("\n== B. cross-cell liveness (fixed drug, reproducible sigs) ==", flush=True)
    idx = np.where(ds.strength >= 1.0)[0]
    bd = balanced_design(idx)
    live = {}
    if bd is not None:
        keep, top, common = bd
        P = np.zeros((len(common), len(top), 978), np.float32); T = np.zeros_like(P)
        for di, d in enumerate(common):
            for cj, c in enumerate(top):
                rws = keep[(d, c)][:MAX_SIG]
                P[di, cj] = predict(rws).mean(0); T[di, cj] = truth(rws).mean(0)
        # per fixed drug, std across cells (averaged over drugs, genes)
        pred_xcell = float(P.std(axis=1).mean()); true_xcell = float(T.std(axis=1).mean())
        live = {"pred_crosscell_std": pred_xcell, "true_crosscell_std": true_xcell,
                "ratio": pred_xcell / (true_xcell + 1e-9)}
        print(f"  cross-cell std: pred={pred_xcell:.3f} true={true_xcell:.3f} "
              f"ratio={live['ratio']:.3f}  (near 0 => pathway dead; ~1 => full expression)", flush=True)

    verdict = ("H_shrink (noise-driven) if std ratio & corr RISE with strength; "
               "H_capacity if flat & low; H_dead if liveness ratio ~0")
    out = {"A_interaction_vs_strength": A, "B_liveness": live, "reading_guide": verdict}
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    json.dump(out, open(f"{work}/analyze.json", "w"), indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
