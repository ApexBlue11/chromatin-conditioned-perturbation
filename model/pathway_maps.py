# -*- coding: utf-8 -*-
"""
Pathway-conductance interpretability maps (CPU, local).
c_{cell,gene} = 1 + tanh(MLP([E; x; avail])) in (0,2): how much gene g listens to its pathway neighbours
in cell c. c<1 = damped (closed chromatin), c>1 = amplified. This turns the mechanism into a READOUT.

Questions answered:
 1. Does c actually vary across cells (is it doing cell-specific work) or is it basically gene-only?
 2. Is c driven by CHROMATIN or is it just re-encoding baseline expression? (partial correlation)
 3. Which genes/cells get the most amplified vs damped pathway input -> inspectable biology.
 4. Does c align with the epigenetic marks in the expected direction (open/active => higher conductance)?

Run: python model/pathway_maps.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import ModelConfig, DataConfig
from data import LincsDataset
from model import LincsCrossAttn

HERE = os.path.dirname(os.path.abspath(__file__))


def partial_corr(a, b, ctrl):
    ra = a - np.polyval(np.polyfit(ctrl, a, 1), ctrl)
    rb = b - np.polyval(np.polyfit(ctrl, b, 1), ctrl)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    dc, mc = DataConfig(), ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).eval()
    model.load_state_dict(torch.load(os.path.join(HERE, "results", "v5_ckpt.pt"), map_location="cpu")["model"])
    if model.pathway_cond is None:
        print("model has no pathway_cond"); return
    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    name = {v: k for k, v in ds.cell_idx.items()}

    # conductance for every (cell, gene): gate the baseline exactly as forward() does
    E = torch.tensor(ds.E); r = torch.tensor(ds.r_cell); X = torch.tensor(ds.Xb)
    with torch.no_grad():
        s = model.gate(E, r)
        C = model.pathway_cond(E, X * s, r).numpy()          # [83,978]
    np.save(os.path.join(HERE, "results", "pathway_conductance.npy"), C)

    print(f"conductance C {C.shape}: mean {C.mean():.4f}  min {C.min():.4f}  max {C.max():.4f}")
    # 1. cell-specific vs gene-only: variance decomposition
    g = C.mean(0); c_ = C.mean(1); tot = C.var()
    var_gene = ((g - C.mean()) ** 2).mean(); var_cell = ((c_ - C.mean()) ** 2).mean()
    var_inter = tot - var_gene - var_cell
    print(f"\n1. variance split: gene {100*var_gene/tot:5.1f}% | cell {100*var_cell/tot:5.1f}% | "
          f"cell x gene {100*var_inter/tot:5.1f}%   (interaction => genuinely cell-specific)")

    # 2. chromatin-driven or just baseline? correlate C against each mark and X_base
    has = ds.Emask.any(1)                                     # [83,3]
    rows = [i for i in range(C.shape[0]) if has[i].any()]
    flatC = C[rows].ravel(); flatX = ds.Xb[rows].ravel()
    print(f"\n2. what drives conductance (epi-covered cells, n={len(rows)} cells x 978 genes):")
    print(f"   corr(C, X_base)          = {np.corrcoef(flatC, flatX)[0,1]:+.4f}")
    for k, mk in enumerate(["ATAC", "H3K27ac", "H3K27me3"]):
        rr = [i for i in rows if has[i, k]]
        if not rr:
            continue
        fc = C[rr].ravel(); fe = ds.E[rr][:, :, k].ravel(); fx = ds.Xb[rr].ravel()
        print(f"   corr(C, {mk:9s})     = {np.corrcoef(fc, fe)[0,1]:+.4f}   "
              f"partial |X_base = {partial_corr(fc, fe, fx):+.4f}   (n={len(rr)} cells)")
    print("   (expected: ATAC/H3K27ac positive = open chromatin conducts; H3K27me3 negative = repressed damps)")

    # 3. most amplified / damped genes (averaged over epi-covered cells)
    gm = C[rows].mean(0)
    order = np.argsort(gm)
    print(f"\n3. most DAMPED genes (c<1, pathway input suppressed): idx {order[:10].tolist()}  "
          f"c={np.round(gm[order[:10]],3).tolist()}")
    print(f"   most AMPLIFIED genes (c>1):                        idx {order[-10:].tolist()}  "
          f"c={np.round(gm[order[-10:]],3).tolist()}")
    cm = C[rows].mean(1)
    o2 = np.argsort(cm)
    print(f"\n   cells with LOWEST mean conductance : {[name.get(rows[i], rows[i]) for i in o2[:5]]}")
    print(f"   cells with HIGHEST mean conductance: {[name.get(rows[i], rows[i]) for i in o2[-5:]]}")

    json.dump({"mean": float(C.mean()), "min": float(C.min()), "max": float(C.max()),
               "var_gene_pct": float(100*var_gene/tot), "var_cell_pct": float(100*var_cell/tot),
               "var_interaction_pct": float(100*var_inter/tot),
               "corr_C_Xbase": float(np.corrcoef(flatC, flatX)[0, 1])},
              open(os.path.join(HERE, "results", "pathway_maps.json"), "w"), indent=2)
    print("\nwrote results/pathway_conductance.npy + pathway_maps.json")


if __name__ == "__main__":
    main()
