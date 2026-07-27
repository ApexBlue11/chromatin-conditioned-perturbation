# -*- coding: utf-8 -*-
"""
drug_ablation.py (Strategy C) — which drug-feature groups actually carry the signal? Zero each group at
inference on REPRODUCIBLE COLD-CELL signatures and measure ΔR²/Δpearson vs the un-ablated baseline.
Groups: the four global-feature blocks (UniMol CLS / ChemBERTa / RDKit descriptors / ECFP4 fingerprint)
concatenated in u_feats, plus the per-atom tokens (the atom->gene attention substrate). Kaggle CPU.

u_feats layout (from data.py): [UniMol CLS 0:512][ChemBERTa 512:896][descriptors 896:916][fingerprint 916:2964]
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import ModelConfig
from data import LincsDataset, collate, build_splits
from model import LincsCrossAttn
from train import make_dataconfig

GROUPS = {"unimol_cls": (0, 512), "chemberta": (512, 896),
          "descriptors": (896, 916), "fingerprint": (916, 2964), "atoms": "atoms"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = make_dataconfig(); mc = ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).to(device).eval()
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    model.load_state_dict(torch.load(ck[0], map_location=device)["model"])
    print("loaded", ck[0], flush=True)
    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)

    sp = build_splits(ds, dc)
    idx = sp["test_coldcell"][ds.strength[sp["test_coldcell"]] >= 1.0]     # reproducible cold-cell
    if len(idx) > 8000:
        idx = np.random.default_rng(0).choice(idx, 8000, replace=False)
    print(f"reproducible cold-cell sigs: {len(idx)}", flush=True)

    @torch.no_grad()
    def ev(group):
        yh, yt = [], []
        for a in range(0, len(idx), 64):
            b = collate([ds[i] for i in idx[a:a + 64]], mc.max_atoms)
            bd = {k: v.to(device) for k, v in b.items()}
            if group == "atoms":
                bd["atoms"] = torch.zeros_like(bd["atoms"]); bd["atom_mask"] = torch.zeros_like(bd["atom_mask"])
            elif group is not None:
                lo, hi = group; bd["u_feats"] = bd["u_feats"].clone(); bd["u_feats"][:, lo:hi] = 0
            yh.append(model(bd).float().cpu()); yt.append(b["Y"])
        yh = torch.cat(yh); yt = torch.cat(yt)
        r2 = float(1 - ((yt - yh) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        pear = float(np.median([np.corrcoef(yh[i], yt[i])[0, 1] for i in range(len(yt))]))
        return r2, pear

    br2, bp = ev(None)
    print(f"\nBASE (no ablation): R2={br2:.4f} pearson={bp:.4f}", flush=True)
    res = {"base": {"r2": br2, "pearson": bp}}
    for name, g in GROUPS.items():
        r2, p = ev(g)
        res[name] = {"r2": r2, "pearson": p, "delta_r2": br2 - r2, "delta_pearson": bp - p}
        print(f"  ablate {name:12s}: R2={r2:.4f} (dR2={br2 - r2:+.4f})  "
              f"pearson={p:.4f} (dP={bp - p:+.4f})", flush=True)
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    json.dump(res, open(f"{work}/drug_ablation.json", "w"), indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
