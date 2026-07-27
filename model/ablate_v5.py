# -*- coding: utf-8 -*-
"""
Attribute v5's components on the v5 checkpoint (LOCAL CPU, no Kaggle needed).
v5 bundled 4 changes AND trained on 24% less data (scaffold drug holdout), so the headline cold-cell
number cannot tell us which component did what. This ablates each at INFERENCE on the SAME signatures:
  - lineage   : cell_ctx -> zeros (no lineage info)
  - pathway   : conductance forced to 1 (static priors, i.e. v3 behaviour)
  - epi       : E=0, r=0  (the FAIR test: reproducible sigs only, never all-signature)
Reported on reproducible (mean|Y|>=1) unseen-CELL and unseen-COMPOUND signatures.

Run: python model/ablate_v5.py  [--ckpt PATH] [--n 2000]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import ModelConfig, DataConfig
from data import LincsDataset, collate, build_splits
from model import LincsCrossAttn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    dc, mc = DataConfig(), ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).eval()
    ck = a.ckpt or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v5_ckpt.pt")
    sd = torch.load(ck, map_location="cpu")["model"]
    model.load_state_dict(sd)
    print(f"loaded {ck}", flush=True)

    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    sp = build_splits(ds, dc)
    rng = np.random.default_rng(0)

    def pick(name):
        idx = sp[name]; idx = idx[ds.strength[idx] >= 1.0]
        return rng.choice(idx, min(a.n, len(idx)), replace=False) if len(idx) else idx

    splits = {"unseen_cell": pick("test_coldcell"), "unseen_compound": pick("test_colddrug")}

    @torch.no_grad()
    def ev(idx, mode):
        keep = model.pathway_cond
        if mode == "no_pathway":
            model.pathway_cond = None          # conductance == 1 (static priors, v3 behaviour)
        yh, yt = [], []
        for s in range(0, len(idx), a.batch):
            b = collate([ds[i] for i in idx[s:s + a.batch]], mc.max_atoms)
            if mode == "no_lineage":
                b["cell_ctx"] = torch.zeros_like(b["cell_ctx"])
            if mode == "no_epi":
                b["E"] = torch.zeros_like(b["E"]); b["r"] = torch.zeros_like(b["r"])
            yh.append(model(b).float()); yt.append(b["Y"])
        model.pathway_cond = keep
        yh = torch.cat(yh); yt = torch.cat(yt)
        r2 = float(1 - ((yt - yh) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        yhc = yh - yh.mean(1, keepdim=True); ytc = yt - yt.mean(1, keepdim=True)
        pear = float(((yhc * ytc).sum(1) / torch.sqrt((yhc ** 2).sum(1) * (ytc ** 2).sum(1) + 1e-8)).median())
        return r2, pear

    out = {}
    for sname, idx in splits.items():
        if len(idx) == 0:
            continue
        print(f"\n=== {sname}  (n={len(idx)}, reproducible) ===", flush=True)
        b_r2, b_p = ev(idx, "full")
        print(f"  full          R2={b_r2:+.4f}  pearson={b_p:.4f}", flush=True)
        out[sname] = {"full": {"r2": b_r2, "pearson": b_p}}
        for mode in ["no_lineage", "no_pathway", "no_epi"]:
            r2, p = ev(idx, mode)
            out[sname][mode] = {"r2": r2, "pearson": p, "delta_r2": b_r2 - r2, "delta_pearson": b_p - p}
            print(f"  {mode:13s} R2={r2:+.4f} (d={b_r2-r2:+.4f})  pearson={p:.4f} (d={b_p-p:+.4f})", flush=True)

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ablate_v5.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}\n(positive delta => the component CONTRIBUTES)", flush=True)


if __name__ == "__main__":
    main()
