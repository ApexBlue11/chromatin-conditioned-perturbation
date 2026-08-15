# -*- coding: utf-8 -*-
"""
Does chromatin (and the pathway layer) matter MORE for drugs that act on chromatin?

WHY THIS EXISTS. v6's aggregate ablation says chromatin contributes ~0 on unseen cells. But an aggregate
null does NOT exclude a strong subgroup effect: if chromatin only matters for the ~30 drugs that actually
target chromatin-modifying enzymes, averaging over ~10k drugs would wash it out completely. HDAC/DNMT
inhibitors are the obvious case -- their entire mechanism is chromatin. Testing this is cheap and the
aggregate result cannot stand as "chromatin does not help" until it is done.

Drug class (`drug/outputs/dti/epi_drug_pert_ids.json`, 30 drugs): union of
  * curated-ChEMBL edges whose TARGET is a chromatin modifier (HDAC/DNMT/EZH/KDM/KAT/BRD/SIRT/...), and
  * a canonical name list (vorinostat, trichostatin-a, panobinostat, azacitidine, ...).
STITCH-only edges are excluded on purpose -- they call glycerol and blebbistatin HDAC binders.

Compared on matched samples from the SAME split, with the ablation mean taken within each group (the
ablate-to-mean discipline: the mean must come from the signatures being scored).

Run:  python model/v6/probe_epidrug_v6.py --ckpt model/results/ckpt_v6_fold0.pt
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

from config_v6 import V6Config, V6DataConfig
from data import LincsDataset, build_splits, collate
from eval_v6 import load_v6, metrics, MeanAblate, substitute


@torch.no_grad()
def score(model, ds, idx, cfg, batch, mode, hooks):
    """full / mean_chromatin / mean_pathway on `idx`, with the ablation mean taken over `idx` itself."""
    # pass 1: collect the means this group needs
    accE, accn = None, 0
    for h in hooks.values():
        h.mode, h.sum, h.n = "collect", None, 0
    yh, yt = [], []
    for s in range(0, len(idx), batch):
        b = collate([ds[i] for i in idx[s:s + batch]], cfg.max_atoms)
        accE = b["E"].sum(0) if accE is None else accE + b["E"].sum(0)
        accn += b["X"].shape[0]
        yh.append(model(b).float()); yt.append(b["Y"])
    for h in hooks.values():
        h.mode = "off"
    base = metrics(torch.cat(yh), torch.cat(yt))
    if mode == "full":
        return base
    mu = {"E": accE / max(accn, 1)}
    yh2, yt2 = [], []
    for s in range(0, len(idx), batch):
        b = collate([ds[i] for i in idx[s:s + batch]], cfg.max_atoms)
        if mode == "mean_chromatin":
            b = substitute(b, "mean_chromatin", mu)
            yh2.append(model(b).float())
        else:
            hooks["mean_pathway"].mode = "ablate"
            yh2.append(model(b).float())
            hooks["mean_pathway"].mode = "off"
        yt2.append(b["Y"])
    m = metrics(torch.cat(yh2), torch.cat(yt2))
    m["delta_pearson"] = base["pearson_median"] - m["pearson_median"]
    m["delta_r2"] = base["r2_overall"] - m["r2_overall"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=800, help="signatures per group (epi and non-epi matched)")
    ap.add_argument("--splits", default="val,test_coldcell")
    ap.add_argument("--random_init", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    dc, cfg = V6DataConfig(), V6Config()
    dc.cell_fold = a.fold
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_reactome_path))
    model, ckpt_name = load_v6(dc, cfg, M, a.ckpt, a.fold, a.random_init)
    hooks = {"mean_pathway": MeanAblate(model.pathway, tuple_index=0)}

    ds = LincsDataset(dc); sp = build_splits(ds, dc)
    dindex = json.load(open(R(dc.drug_index_path)))
    epi_pids = json.load(open(R("drug/outputs/dti/epi_drug_pert_ids.json")))
    epi_rows = {dindex[p] for p in epi_pids if p in dindex}
    is_epi = np.isin(ds.drug_row, list(epi_rows))
    print(f"epigenetic drug class: {len(epi_rows)} drugs, {int(is_epi.sum()):,} signatures total", flush=True)

    rng = np.random.default_rng(0)
    out = {"ckpt": ckpt_name, "n_epi_drugs": len(epi_rows), "splits": {}}
    for sname in a.splits.split(","):
        idx = sp[sname]
        rep = idx[ds.strength[idx] >= dc.eval_min_strength]
        e = rep[is_epi[rep]]; o = rep[~is_epi[rep]]
        if len(e) < 100:
            print(f"\n{sname}: only {len(e)} reproducible epi-drug signatures -- skipped", flush=True); continue
        n = min(a.n, len(e), len(o))
        e = np.sort(rng.choice(e, n, replace=False)); o = np.sort(rng.choice(o, n, replace=False))
        print(f"\n{'='*76}\n=== {sname}  (n={n} per group, reproducible)\n{'='*76}", flush=True)
        rec = {}
        for gname, gidx in [("EPIGENETIC drugs", e), ("other drugs", o)]:
            full = score(model, ds, gidx, cfg, a.batch, "full", hooks)
            chrom = score(model, ds, gidx, cfg, a.batch, "mean_chromatin", hooks)
            path = score(model, ds, gidx, cfg, a.batch, "mean_pathway", hooks)
            rec[gname] = {"full": full, "mean_chromatin": chrom, "mean_pathway": path}
            print(f"  {gname:18s} full pearson={full['pearson_median']:.4f}  R2={full['r2_overall']:+.4f}")
            print(f"  {'':18s}   ablate chromatin -> d_pearson={chrom['delta_pearson']:+.4f}  "
                  f"d_R2={chrom['delta_r2']:+.4f}   |dY|max n/a")
            print(f"  {'':18s}   ablate pathway   -> d_pearson={path['delta_pearson']:+.4f}  "
                  f"d_R2={path['delta_r2']:+.4f}", flush=True)
        de = rec["EPIGENETIC drugs"]["mean_chromatin"]["delta_pearson"]
        do = rec["other drugs"]["mean_chromatin"]["delta_pearson"]
        print(f"\n  CHROMATIN effect, epi-drugs minus others = {de - do:+.4f}"
              f"   <- positive means chromatin matters MORE for drugs that act on chromatin")
        rec["chromatin_epi_minus_other"] = de - do
        out["splits"][sname] = rec

    dest = os.path.join(os.path.dirname(HERE), "results", f"probe_epidrug_v6_fold{a.fold}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
