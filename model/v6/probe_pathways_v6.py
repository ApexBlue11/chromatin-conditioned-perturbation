# -*- coding: utf-8 -*-
"""
Validate v6's pathway readout against INDEPENDENT annotation (HANDOFF 2 step 4). Local CPU.

WHAT THE READOUT ACTUALLY IS -- verified, not assumed.
`aux["pathway_activations"]` is computed at step (4) of the forward pass, BEFORE the drug is introduced at
step (5). Measured on a live model: replacing the drug with a completely different molecule changes the
readout by EXACTLY 0.0, while baseline expression (2.54), chromatin (1.28) and dose/time (1.59) all move it.
So the readout is a CELL x DOSE/TIME quantity -- "which named Reactome pathways are chromatin-permitted in
this cell at this exposure" -- and it carries NO drug information whatsoever.

Consequence, stated plainly because the natural reading is wrong: v6 does NOT repair the falsified claim
[4.1a/4.1b] that atom->gene attention localises to drug targets. That was a DRUG-level claim; this is a
CELL-level readout. Testing it against drug->target annotation (dti_reference.tsv) is structurally
guaranteed to return a null that means nothing -- so this script does not do that.

THE TEST THAT IS VALID.
Independent annotation = the MEASURED LINCS response, which is never an input to the readout:
  model side    a_p(c)  = mean |pathway activation| for pathway p in cell c
  measured side m_p(c)  = mean over p's member genes of mean|Y| in cell c, REPRODUCIBLE signatures only
                          (~75 % of LINCS is inert; dilution once inverted the sign of a real effect [6.2])
  -> Spearman/Pearson of a_p(c) vs m_p(c) across the 360 named pathways, median over cells.
Question: is the readout organised along the Reactome axes it is built from, in a way that tracks what the
cell's pathways ACTUALLY do?

THREE CONTROLS, because a bare correlation here would not survive this project's own standards:
  1. gene-permutation null -- permute mean|Y| across the 978 genes and recompute m_p. Preserves every
     pathway's SIZE and the marginal distribution of gene movement, destroys only the gene<->pathway
     correspondence. Large pathways average more genes and would otherwise carry a size artefact.
  2. CELL-SHUFFLE null -- score cell c's activations against a DIFFERENT cell's measured movement. If this
     matches the within-cell correlation, the readout is a global pathway prior, not a cell-specific one.
     This is the decisive control: v5's chromatin effect was +0.089 in-distribution but ~0 on unseen cells
     [2.3], so "looks cell-specific but is not" is the established failure mode here.
  3. stratification -- TRAIN cells vs fold-0 TEST cells, and cells WITH vs WITHOUT chromatin data. A readout
     that only works on cells whose Y the model trained on is a memorisation readout.

Run:  python model/v6/probe_pathways_v6.py --ckpt model/results/ckpt_v6_fold0.pt
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

from config_v6 import V6Config, V6DataConfig
from data import LincsDataset, build_splits, collate
from eval_v6 import load_pathway_names, load_v6


def rank(x):
    """average-rank transform, so Pearson-on-ranks == Spearman (avoids a scipy dependency)."""
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x), np.float64); r[o] = np.arange(len(x), dtype=np.float64)
    # average ties
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman(a, b):
    return corr(rank(a), rank(b))


@torch.no_grad()
def activation_profile(model, ds, rows, cfg, batch=16):
    """a_p: mean |pathway activation| per named pathway over `rows` (all from ONE cell). -> [P]"""
    acc, n = None, 0
    for s in range(0, len(rows), batch):
        b = collate([ds[i] for i in rows[s:s + batch]], cfg.max_atoms)
        _, aux = model(b, return_interp=True)
        v = aux["pathway_activations"].abs().mean(-1).sum(0)      # [P]
        acc = v if acc is None else acc + v
        n += b["X"].shape[0]
    return (acc / max(n, 1)).numpy()


def movement_profile(ds, rows, M_norm):
    """m_p: how much pathway p's member genes ACTUALLY move in this cell (measured, never a model input).
    mean|Y| per gene over the cell's reproducible signatures, averaged over each pathway's members."""
    Y = np.abs(np.asarray(ds.Y[ds.y_row[rows]], np.float32)).mean(0)      # [978]
    return M_norm @ Y, Y                                                  # [P], [978]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_sig_per_cell", type=int, default=24,
                    help="reproducible signatures per cell fed to the model (the readout is drug-invariant, "
                         "so these differ only in dose/time -- a handful suffices)")
    ap.add_argument("--min_sig_per_cell", type=int, default=50,
                    help="minimum reproducible signatures for a cell's MEASURED profile to be stable")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--random_init", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    dc, cfg = V6DataConfig(), V6Config()
    dc.cell_fold = a.fold
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_reactome_path)).astype(np.float32)
    if M.shape[0] > M.shape[1]:
        M = M.T
    pathways = load_pathway_names(dc, M.astype(np.int8))
    print(f"pathway_info.tsv <-> M_reactome row order VERIFIED ({len(pathways)} named nodes)", flush=True)
    M_norm = M / np.maximum(M.sum(1, keepdims=True), 1)                   # [P,G] row-normalised membership
    sizes = M.sum(1).astype(int)

    model, ckpt_name = load_v6(dc, cfg, M.astype(np.int8), a.ckpt, a.fold, a.random_init)

    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    sp = build_splits(ds, dc)
    test_cells = set(int(c) for c in sp["_test_cells"])
    rep_mask = ds.strength >= dc.eval_min_strength
    rng = np.random.default_rng(0)

    # ---- per-cell profiles -------------------------------------------------------------------------
    cells, prof = [], {}
    for c in np.unique(ds.cell_row):
        rows = np.where((ds.cell_row == c) & rep_mask)[0]
        if len(rows) < a.min_sig_per_cell:
            continue
        use = rows if len(rows) <= a.max_sig_per_cell else rng.choice(rows, a.max_sig_per_cell, replace=False)
        a_p = activation_profile(model, ds, np.sort(use), cfg, a.batch)
        m_p, g = movement_profile(ds, rows, M_norm)
        prof[int(c)] = {"a": a_p, "m": m_p, "g": g, "n_rep": int(len(rows)),
                        "has_epi": bool(ds.r_cell[c].max() > 0), "is_test": int(c) in test_cells}
        cells.append(int(c))
    print(f"cells profiled: {len(cells)}  (test-fold {sum(prof[c]['is_test'] for c in cells)}, "
          f"with chromatin {sum(prof[c]['has_epi'] for c in cells)})", flush=True)
    if len(cells) < 5:
        print("too few cells -- aborting"); return

    # ---- within-cell correlation + gene-permutation null -------------------------------------------
    rows_out = []
    for c in cells:
        p = prof[c]
        sr, pr = spearman(p["a"], p["m"]), corr(p["a"], p["m"])
        null = np.empty(a.n_perm)
        for i in range(a.n_perm):                       # permute genes: sizes + marginals preserved
            null[i] = spearman(p["a"], M_norm @ p["g"][rng.permutation(len(p["g"]))])
        rows_out.append({"cell_row": c, "cell": next((k for k, v in ds.cell_idx.items() if v == c), str(c)),
                         "n_rep": p["n_rep"], "has_epi": p["has_epi"], "is_test": p["is_test"],
                         "spearman": sr, "pearson": pr,
                         "null_mean": float(null.mean()), "null_sd": float(null.std()),
                         "z_vs_null": float((sr - null.mean()) / (null.std() + 1e-9)),
                         "p_one_sided": float((null >= sr).mean())})

    # ---- cell-shuffle null: is the readout CELL-SPECIFIC at all? ----------------------------------
    cross = []
    for c in cells:
        for c2 in rng.choice([x for x in cells if x != c], min(8, len(cells) - 1), replace=False):
            cross.append(spearman(prof[c]["a"], prof[int(c2)]["m"]))
    cross = np.array(cross)

    def med(sel):
        v = [r["spearman"] for r in rows_out if sel(r)]
        return (float(np.median(v)), len(v)) if v else (float("nan"), 0)

    within, n_w = med(lambda r: True)
    strata = {"train_cells": med(lambda r: not r["is_test"]), "test_cells": med(lambda r: r["is_test"]),
              "with_chromatin": med(lambda r: r["has_epi"]), "without_chromatin": med(lambda r: not r["has_epi"])}

    print(f"\n{'='*82}")
    print(f"pathway readout vs MEASURED pathway movement -- Spearman across {len(pathways)} named pathways")
    print(f"{'='*82}")
    print(f"  within-cell  median rho = {within:+.4f}   (n={n_w} cells)")
    print(f"  CELL-SHUFFLE median rho = {float(np.median(cross)):+.4f}   (n={len(cross)} mismatched pairs)")
    print(f"  cell-specific gap       = {within - float(np.median(cross)):+.4f}"
          f"   <- if ~0, the readout is a GLOBAL pathway prior, not a cell-specific one")
    print(f"  gene-permutation null   = {float(np.mean([r['null_mean'] for r in rows_out])):+.4f}"
          f"   median z = {float(np.median([r['z_vs_null'] for r in rows_out])):+.2f}"
          f"   cells with p<0.05: {sum(1 for r in rows_out if r['p_one_sided'] < 0.05)}/{n_w}")
    for k, (v, n) in strata.items():
        print(f"    {k:20s} median rho = {v:+.4f}  (n={n})")
    print(f"\n  correlation of pathway activation with pathway SIZE: "
          f"{spearman(np.mean([prof[c]['a'] for c in cells], 0), sizes.astype(float)):+.4f}"
          f"   <- a large value means the readout is largely reporting pathway size")

    mean_a = np.mean([prof[c]["a"] for c in cells], 0)
    out = {"ckpt": ckpt_name, "fold": a.fold, "random_init": bool(a.random_init),
           "n_cells": n_w, "n_pathways": len(pathways), "n_perm": a.n_perm,
           "within_cell_median_spearman": within,
           "cell_shuffle_median_spearman": float(np.median(cross)),
           "cell_specific_gap": within - float(np.median(cross)),
           "gene_permutation_null_mean": float(np.mean([r["null_mean"] for r in rows_out])),
           "median_z_vs_permutation_null": float(np.median([r["z_vs_null"] for r in rows_out])),
           "n_cells_p_lt_0.05": sum(1 for r in rows_out if r["p_one_sided"] < 0.05),
           "activation_vs_pathway_size_spearman": spearman(mean_a, sizes.astype(float)),
           "strata_median_spearman": {k: {"rho": v, "n": n} for k, (v, n) in strata.items()},
           "per_cell": rows_out,
           "top20_pathways_by_mean_activation": [
               {"rank": i + 1, "pathway_id": pathways[p][0], "pathway_name": pathways[p][1],
                "n_genes": int(sizes[p]), "mean_abs_activation": float(mean_a[p])}
               for i, p in enumerate(np.argsort(-mean_a)[:20])]}
    dest = os.path.join(os.path.dirname(HERE), "results", f"probe_pathways_v6_fold{a.fold}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")
    print("A null here is a RESULT: it would mean the masked bottleneck buys accuracy-neutral structure "
          "without a readout that tracks biology, and it must be reported as prominently as a positive.",
          flush=True)


if __name__ == "__main__":
    main()
