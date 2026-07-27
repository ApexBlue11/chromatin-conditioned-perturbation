# -*- coding: utf-8 -*-
"""
Report our model under BOTH metric conventions so readers can compare to published numbers.

The SOTA papers (e.g. the latent-diffusion model: unseen-cell PCC 0.743, unseen-compound 0.870) predict
ABSOLUTE perturbed expression with the basal profile supplied as conditioning, and correlate per-sample
across the 978 genes. Because perturbed ~= basal for most genes, that PCC is dominated by the basal
profile the model was GIVEN -- it largely measures "did you echo the baseline", not "did you predict the
drug effect". We predict the DIFFERENTIAL directly (baseline removed), which is strictly harder.

This script quantifies the gap on the SAME predictions:
  differential PCC : corr(Yhat, Y)                      <- our (strict) metric
  absolute-style   : corr(Xbase + Yhat, Xbase + Y)      <- their (inflated) convention

CAVEAT (must be reported): we have NO L1000 control profiles (only Level 5), so the basal anchor here is
the CCLE baseline -- a DIFFERENT platform and scale from the L1000 z-scored differential. The absolute
number is therefore ILLUSTRATIVE of the convention's inflation, NOT a claim of protocol equivalence.
The inflation magnitude depends on the basal:delta variance ratio, which we also report.

Run: python model/dual_metric.py [--n 1500]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import ModelConfig, DataConfig
from data import LincsDataset, collate, build_splits
from model import LincsCrossAttn


def pcc_rows(a, b):
    ac = a - a.mean(1, keepdim=True); bc = b - b.mean(1, keepdim=True)
    return (ac * bc).sum(1) / torch.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1) + 1e-8)


def r2(a, b):
    return float(1 - ((b - a) ** 2).sum() / ((b - b.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v5_ckpt.pt"))
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    dc, mc = DataConfig(), ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    sp = build_splits(ds, dc); rng = np.random.default_rng(0)

    out = {}
    for split in ["test_coldcell", "test_colddrug"]:
        idx = sp[split]; idx = idx[ds.strength[idx] >= 1.0]
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(a.n, len(idx)), replace=False)
        YH, YT, XB = [], [], []
        with torch.no_grad():
            for s in range(0, len(idx), a.batch):
                b = collate([ds[i] for i in idx[s:s + a.batch]], mc.max_atoms)
                YH.append(model(b).float()); YT.append(b["Y"]); XB.append(b["X"])
        yh = torch.cat(YH); yt = torch.cat(YT); xb = torch.cat(XB)

        d_p = float(pcc_rows(yh, yt).median()); d_r2 = r2(yh, yt)
        a_p = float(pcc_rows(xb + yh, xb + yt).median()); a_r2 = r2(xb + yh, xb + yt)
        var_ratio = float((xb.var() / yt.var()))
        label = {"test_coldcell": "unseen CELL", "test_colddrug": "unseen COMPOUND"}[split]
        print(f"\n=== {label}  (n={len(idx)}, reproducible) ===")
        print(f"  DIFFERENTIAL (ours, strict) : PCC {d_p:.3f}   R2 {d_r2:+.3f}")
        print(f"  ABSOLUTE-style (their conv.): PCC {a_p:.3f}   R2 {a_r2:+.3f}")
        print(f"  inflation from adding basal : PCC {a_p - d_p:+.3f}   (basal:delta var ratio {var_ratio:.1f})")
        out[split] = {"label": label, "n": int(len(idx)),
                      "differential": {"pcc": d_p, "r2": d_r2},
                      "absolute_style": {"pcc": a_p, "r2": a_r2},
                      "pcc_inflation": a_p - d_p, "basal_delta_var_ratio": var_ratio}

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dual_metric.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")
    print("NOTE: absolute-style uses the CCLE baseline as anchor (no L1000 controls exist in our data);")
    print("      it is ILLUSTRATIVE of the convention's inflation, not a protocol-equivalent comparison.")


if __name__ == "__main__":
    main()
