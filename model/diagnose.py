# -*- coding: utf-8 -*-
"""
Decisive model diagnostic: apply the SAME orthogonal three-way variance partition to the model's
PREDICTIONS and to the TRUTH, on a balanced drug x cell design of REPRODUCIBLE signatures.

Truth (measured): drug x gene 41.6% | cell x gene 9.5% | drug x cell x gene interaction 49.0%.
If the model's predictions are ~all drug x gene with ~0 interaction, it is predicting a drug-average
signature and ignoring cell-specificity -> that is the performance gap, not epigenetics.

Run inside a Kaggle kernel (needs model-src + train bundle + v2 checkpoint).
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

STRENGTH = 1.0        # reproducible threshold (replicate r ~0.37+ above this)
N_CELLS = 6


def partition(Ym):
    """Ym [D,C,G] -> orthogonal shares (drug x gene, cell x gene, interaction), gene effect removed."""
    D, C, G = Ym.shape
    g = Ym.mean((0, 1)); A = Ym.mean(1) - g; B = Ym.mean(0) - g
    I = Ym - g[None, None, :] - A[:, None, :] - B[None, :, :]
    tot = ((Ym - g[None, None, :]) ** 2).sum()
    return (100 * C * (A ** 2).sum() / tot, 100 * D * (B ** 2).sum() / tot, 100 * (I ** 2).sum() / tot)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = make_dataconfig(); mc = ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).to(device).eval()
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    sd = torch.load(ck[0], map_location=device)["model"]
    model.load_state_dict(sd); print("loaded checkpoint", ck[0], flush=True)

    shared = LincsDataset.load_shared(dc)
    ds = LincsDataset(dc, _shared=shared)
    Y = ds.Y

    # reproducible signatures only
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ds.y_row), 150000, replace=False)
    keep = defaultdict(list)
    for j in sub:
        y = np.asarray(Y[ds.y_row[j]], np.float32)
        if np.abs(y).mean() >= STRENGTH:
            keep[(ds.drug_row[j], ds.cell_row[j])].append(j)
    print(f"reproducible (drug,cell) groups: {len(keep):,}", flush=True)

    cellcount = defaultdict(set)
    for (d, c) in keep: cellcount[c].add(d)
    top = [c for c, _ in sorted(cellcount.items(), key=lambda x: -len(x[1]))[:N_CELLS]]
    common = sorted(set.intersection(*[cellcount[c] for c in top]))
    print(f"balanced design: {len(common)} drugs x {len(top)} cells", flush=True)
    if len(common) < 10:
        print("too few common drugs"); return

    # predict + truth for every (drug,cell)
    P = np.zeros((len(common), len(top), 978), np.float32)
    T = np.zeros_like(P)
    for di, d in enumerate(common):
        for cj, c in enumerate(top):
            idxs = keep[(d, c)][:8]
            batch = collate([ds[i] for i in idxs], mc.max_atoms)
            with torch.no_grad():
                pred = model({k: v.to(device) for k, v in batch.items()}).float().cpu().numpy()
            P[di, cj] = pred.mean(0)
            T[di, cj] = batch["Y"].numpy().mean(0)
        if di % 20 == 0: print(f"  {di}/{len(common)}", flush=True)

    # ---- THE FAIR EPIGENETICS TEST: ablation measured on REPRODUCIBLE signatures only ----
    rep = np.concatenate([np.array(v[:8]) for v in keep.values()])
    rep = rep[:12000]
    from data import collate as _col
    def _eval(ablate):
        yh, yt = [], []
        for a in range(0, len(rep), 64):
            b = _col([ds[i] for i in rep[a:a + 64]], mc.max_atoms)
            bd = {k: v.to(device) for k, v in b.items()}
            if ablate:
                bd["E"] = torch.zeros_like(bd["E"]); bd["r"] = torch.zeros_like(bd["r"])
            with torch.no_grad():
                yh.append(model(bd).float().cpu()); yt.append(b["Y"])
        yh = torch.cat(yh); yt = torch.cat(yt)
        ss_res = ((yt - yh) ** 2).sum(); ss_tot = ((yt - yt.mean()) ** 2).sum()
        pear = float(np.median([np.corrcoef(yh[i], yt[i])[0, 1] for i in range(len(yt))]))
        return float(1 - ss_res / ss_tot), pear
    r2_w, p_w = _eval(False); r2_a, p_a = _eval(True)
    print(f"\nFAIR EPI ABLATION (reproducible sigs, n={len(rep)}):")
    print(f"  with epi : R2={r2_w:+.4f}  pearson_median={p_w:.4f}")
    print(f"  ablated  : R2={r2_a:+.4f}  pearson_median={p_a:.4f}")
    print(f"  delta    : R2={r2_w-r2_a:+.5f}  pearson={p_w-p_a:+.5f}   (POSITIVE => epi contributes)")

    pt = partition(T); pp = partition(P)
    print("\n           drug x gene | cell x gene | DRUG x CELL interaction")
    print(f"  TRUTH:     {pt[0]:6.1f}%  |  {pt[1]:6.1f}%  |  {pt[2]:6.1f}%")
    print(f"  MODEL:     {pp[0]:6.1f}%  |  {pp[1]:6.1f}%  |  {pp[2]:6.1f}%")
    # how well does the model track the interaction component specifically?
    def inter(Z):
        g = Z.mean((0, 1)); return Z - g[None, None, :] - (Z.mean(1) - g)[:, None, :] - (Z.mean(0) - g)[None, :, :]
    it, ip = inter(T).ravel(), inter(P).ravel()
    print(f"\n  corr(TRUE interaction, PREDICTED interaction) = {np.corrcoef(it, ip)[0,1]:+.4f}")
    print(f"  corr(TRUE overall,     PREDICTED overall)     = {np.corrcoef(T.ravel(), P.ravel())[0,1]:+.4f}")
    json.dump({"truth": [float(x) for x in pt], "model": [float(x) for x in pp],
               "corr_interaction": float(np.corrcoef(it, ip)[0, 1]),
               "corr_overall": float(np.corrcoef(T.ravel(), P.ravel())[0, 1])},
              open("/kaggle/working/diagnose.json", "w"), indent=2)


if __name__ == "__main__":
    main()
