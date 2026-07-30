# -*- coding: utf-8 -*-
"""
RIDGE LINEAR BASELINE — the baseline we were missing (M.1 in results/CLAIMS.md §6c). Local CPU, minutes.

Why this exists. Ahlmann-Eltze, Huber & Anders (Nature Methods 2025) compared five foundation models and
two deep models for perturbation-effect prediction and **none beat deliberately simple baselines**. The
decisive baseline there is not a mean — it is a **ridge-style LINEAR model** mapping a perturbation
embedding (and cell context) to the response. We had only Mean / Meancell / Meandrug, so "beats all naive
baselines" was a weak claim. This fits the missing one.

Two variants, both closed-form ridge, both scored with the IDENTICAL protocol as the neural models
(`v6/eval_v6.py`): same splits, same reproducible stratum, same metric functions, same v5 subsampling caps.
  `ecfp_cell`    [ECFP4 2048 | descriptors 20 | lineage 16 | log-dose | time]  -> Y[978]   — the simple one
  `full_linear`  [**all** of u_feats 2580 = UniMol CLS + desc + ECFP4 | X_base 978 | lineage | dose | time]
`full_linear` is the honest competitor: it is given **the same global information the neural model gets**,
lacking only the per-atom Uni-Mol tokens and chromatin. If it matches v5/v6 on a split, the architecture is
not earning its complexity there, and that has to be reported rather than buried.

Method notes that matter for a fair fight:
  * fitted ONLY on the train split; lambda chosen on the SAME held-out `val` split the neural models use.
  * features standardised by train statistics; Y centred by the train mean and the offset added back, so
    the intercept is exact and the model can reproduce the Mean baseline as a special case.
  * one Gram matrix, refactorised per lambda -- so the lambda sweep is nearly free.
  * NO reliability weighting, deliberately: a baseline should be simple. (The neural models use it, which
    if anything favours them.)

Run:  python model/baseline_linear.py
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "v6"))

import numpy as np
import torch

from config_v6 import V6DataConfig
from data import LincsDataset, build_splits
from eval_v6 import metrics, SPLIT_KEY, V5_REF, STRATA
import losses as L

VARIANTS = ["ecfp_cell", "full_linear"]


def build_features(ds, variant):
    """Per-signature design matrix, assembled lazily per chunk to avoid materialising N x D.

    u_feats is [UniMol CLS 512 | descriptors 20 | ECFP4 2048] (data.py builds it in that order with
    ChemBERTa dropped), so the trailing slices below pick out ECFP4 and the descriptors."""
    fp = ds.u_feats[:, -2048:]
    desc = ds.u_feats[:, -2068:-2048]
    def rows(idx):
        d, c = ds.drug_row[idx], ds.cell_row[idx]
        cond = [ds.cell_ctx[c], ds.dose_n[idx][:, None], ds.time_n[idx][:, None]]
        if variant == "full_linear":
            blocks = [ds.u_feats[d], ds.Xb[c]] + cond
        else:
            blocks = [fp[d], desc[d]] + cond
        return np.concatenate(blocks, axis=1).astype(np.float64)
    return rows


def accumulate(rows, ds, idx, D, chunk=4096, stats=None):
    """Stream over `idx` accumulating (XtX, XtY, sums) -- or, if stats is given, standardise first."""
    XtX = np.zeros((D + 1, D + 1))                  # +1 for the intercept column
    XtY = np.zeros((D + 1, 978))
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        X = rows(sl)
        if stats is not None:
            X = (X - stats[0]) / stats[1]
        X = np.concatenate([X, np.ones((len(sl), 1))], axis=1)
        Y = np.asarray(ds.Y[ds.y_row[sl]], np.float64)
        XtX += X.T @ X
        XtY += X.T @ Y
    return XtX, XtY


def feature_stats(rows, ds, idx, D, chunk=4096):
    n = 0; s1 = np.zeros(D); s2 = np.zeros(D)
    for s in range(0, len(idx), chunk):
        X = rows(idx[s:s + chunk])
        n += len(X); s1 += X.sum(0); s2 += (X * X).sum(0)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    return mu, np.maximum(sd, 1e-6)                  # constant columns -> unit scale, ridge kills them


def predict(rows, ds, idx, W, stats, chunk=4096):
    out = np.empty((len(idx), 978), np.float32)
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        X = (rows(sl) - stats[0]) / stats[1]
        X = np.concatenate([X, np.ones((len(sl), 1))], axis=1)
        out[s:s + len(sl)] = (X @ W).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--lambdas", default="1e2,1e3,1e4,1e5,1e6")
    ap.add_argument("--strata_n", type=int, default=1500)
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    dc = V6DataConfig(); dc.cell_fold = a.fold
    ds = LincsDataset(dc)
    sp = build_splits(ds, dc)
    print(f"train={len(sp['train']):,}  val={len(sp['val']):,}", flush=True)

    out = {"fold": a.fold, "eval_min_strength": dc.eval_min_strength, "v5_reference": V5_REF,
           "note": "ridge linear baseline; identical splits/stratum/metrics to v6/eval_v6.py",
           "variants": {}}

    for variant in [v for v in a.variants.split(",") if v]:
        rows = build_features(ds, variant)
        D = rows(sp["train"][:2]).shape[1]
        print(f"\n{'='*78}\n=== {variant}: {D} features\n{'='*78}", flush=True)
        t0 = time.time()
        stats = feature_stats(rows, ds, sp["train"], D)
        XtX, XtY = accumulate(rows, ds, sp["train"], D, stats=stats)
        print(f"  Gram accumulated in {time.time()-t0:.0f}s", flush=True)

        # ---- lambda by validation, refactorising the SAME Gram (cheap) ----
        reg = np.eye(D + 1); reg[-1, -1] = 0.0            # never penalise the intercept
        val_idx = sp["val"]; Yval = np.asarray(ds.Y[ds.y_row[val_idx]], np.float32)
        best = None
        for lam in [float(x) for x in a.lambdas.split(",")]:
            W = np.linalg.solve(XtX + lam * reg, XtY)
            p = predict(rows, ds, val_idx, W, stats)
            mse = float(((p - Yval) ** 2).mean())
            print(f"  lambda={lam:>8.0e}  val MSE={mse:.4f}", flush=True)
            if best is None or mse < best[1]:
                best = (lam, mse, W)
        lam, val_mse, W = best
        print(f"  -> lambda={lam:.0e} (val MSE {val_mse:.4f})", flush=True)
        rec = {"n_features": int(D), "lambda": lam, "val_mse": val_mse, "splits": {}}

        for name, key in SPLIT_KEY.items():
            idx = sp[key]
            rep = idx[ds.strength[idx] >= dc.eval_min_strength]
            ref, cap = V5_REF[name], V5_REF[name]["cap"]
            if len(rep) < 500:
                continue
            hn = min(len(rep), cap)
            h_idx = np.random.default_rng(0).choice(rep, hn, replace=False) if hn < len(rep) else rep
            yh = torch.from_numpy(predict(rows, ds, h_idx, W, stats))
            yt = torch.from_numpy(np.asarray(ds.Y[ds.y_row[h_idx]], np.float32))
            m = metrics(yh, yt, ds.phase[h_idx])
            srec = {"headline": m, "strata": {}}
            print(f"\n  {name:16s} n={hn:<5} pearson={m['pearson_median']:.4f}  R2={m['r2_overall']:+.4f}"
                  f"  cDEG@100={m['common_degs@100']:.3f}"
                  f"   |  v5 neural: {ref['pearson']:.3f} / {ref['r2']:+.3f}"
                  f"   |  linear-minus-v5 = {m['pearson_median']-ref['pearson']:+.4f}", flush=True)

            for lo, hi in (STRATA if a.strata_n else []):
                band = idx[(ds.strength[idx] >= lo) & (ds.strength[idx] < hi)]
                if len(band) < 200:
                    continue
                sn = min(a.strata_n, len(band))
                s_idx = (np.random.default_rng(11).choice(band, sn, replace=False)
                         if sn < len(band) else band)
                s_idx = np.sort(s_idx)
                ph = torch.from_numpy(predict(rows, ds, s_idx, W, stats))
                pt = torch.from_numpy(np.asarray(ds.Y[ds.y_row[s_idx]], np.float32))
                sm = metrics(ph, pt)
                srec["strata"][f"{lo}-{hi}"] = {**sm, "n_available": int(len(band))}
                print(f"      mean|Y| {lo:>4}-{str(hi):<4} n={sn:<5} pearson={sm['pearson_median']:.4f}"
                      f"  R2={sm['r2_overall']:+.4f}  cDEG@100={sm['common_degs@100']:.3f}", flush=True)
            rec["splits"][name] = srec
        out["variants"][variant] = rec

    # ---- the mean baselines, on the SAME reproducible stratum and the SAME metrics --------------------
    # They were previously reported only as MSE on all cold-cell signatures, which is not comparable to
    # anything else we quote.
    out["mean_baselines"] = {}
    for name, key in SPLIT_KEY.items():
        idx = sp[key]
        rep = idx[ds.strength[idx] >= dc.eval_min_strength]
        if len(rep) < 500:
            continue
        cap = V5_REF[name]["cap"]; hn = min(len(rep), cap)
        h_idx = np.random.default_rng(0).choice(rep, hn, replace=False) if hn < len(rep) else rep
        Yt = np.asarray(ds.Y[ds.y_row[h_idx]], np.float32)
        Ytr = np.asarray(ds.Y[ds.y_row[sp["train"]]], np.float32)
        gmean = Ytr.mean(0)
        cmean = {c: Ytr[ds.cell_row[sp["train"]] == c].mean(0) for c in np.unique(ds.cell_row[h_idx])
                 if (ds.cell_row[sp["train"]] == c).any()}
        dmean = {d: Ytr[ds.drug_row[sp["train"]] == d].mean(0) for d in np.unique(ds.drug_row[h_idx])
                 if (ds.drug_row[sp["train"]] == d).any()}
        preds = {"Mean": np.broadcast_to(gmean, Yt.shape),
                 "Meancell": np.stack([cmean.get(c, gmean) for c in ds.cell_row[h_idx]]),
                 "Meandrug": np.stack([dmean.get(d, gmean) for d in ds.drug_row[h_idx]])}
        out["mean_baselines"][name] = {}
        for bn, p in preds.items():
            m = metrics(torch.from_numpy(np.ascontiguousarray(p)), torch.from_numpy(Yt))
            out["mean_baselines"][name][bn] = m
            print(f"  {name:16s} {bn:9s} pearson={m['pearson_median']:.4f}  R2={m['r2_overall']:+.4f}"
                  f"  cDEG@100={m['common_degs@100']:.3f}", flush=True)

    dest = os.path.join(HERE, "results", f"baseline_linear_fold{a.fold}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")
    print("If the linear model matches the neural model on a split, the architecture is not earning its "
          "complexity there -- report that, do not bury it.", flush=True)


if __name__ == "__main__":
    main()
