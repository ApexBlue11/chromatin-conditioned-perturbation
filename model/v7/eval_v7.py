# -*- coding: utf-8 -*-
"""
v7 evaluation, protocol-matched to v5/v6 so the three are directly comparable, and able to score the RAW
and the EMA weights separately -- the per-epoch number printed during training uses the raw weights, so
EMA (the whole point of which is that it generalises better) is otherwise never actually measured.

Everything about the protocol is inherited from eval_v6: same fold-0 splits, same reproducible stratum,
same metric functions, same v5 subsampling caps, same split-identity check.

Run:  python model/v7/eval_v7.py --ckpt model/results/ckpt_v7_fold0_seed0.pt --weights both
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v6"))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

from config_v7 import V7Config, V6DataConfig
from model_v7 import LincsV7
from data import LincsDataset, build_splits, collate
from eval_v6 import metrics, SPLIT_KEY, V5_REF, STRATA

V6_REF = {"unseen_cell": 0.4466, "unseen_compound": 0.4686, "unseen_both": 0.4663}


def load_v7(dc, cfg, M, ppi, ckpt, which="raw"):
    """which='raw' | 'ema'. A partial load is fatal: it would quietly produce a half-initialised model."""
    model = LincsV7(cfg, M, ppi).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    key = "ema" if which == "ema" else "model"
    if key not in sd:
        raise RuntimeError(f"checkpoint has no '{key}' weights (keys: {list(sd)[:6]})")
    w = {k[7:] if k.startswith("module.") else k: v for k, v in sd[key].items()}
    missing, unexpected = model.load_state_dict(w, strict=False)
    real_missing = [k for k in missing if "task_weights" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"{which}: checkpoint does not match the model: "
                           f"missing={real_missing[:6]} unexpected={list(unexpected)[:6]}")
    return model


@torch.no_grad()
def predict(model, ds, idx, cfg, batch=32):
    yh, yt = [], []
    for s in range(0, len(idx), batch):
        b = collate([ds[i] for i in idx[s:s + batch]], cfg.max_atoms)
        yh.append(model(b).float()); yt.append(b["Y"])
    return torch.cat(yh), torch.cat(yt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="both", choices=["raw", "ema", "both"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--m_reactome", default="M_reactome_ms5.npy")
    ap.add_argument("--strata_n", type=int, default=0, help="0 = skip the all-strata pass (headline only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    dc, cfg = V6DataConfig(), V7Config()
    dc.cell_fold = a.fold
    Rp = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(Rp(os.path.join("network/outputs", a.m_reactome)))
    ppi = np.load(Rp(dc.ppi_path))
    print(f"pathway matrix {a.m_reactome}: {M.shape[0]} nodes | STRING {int((ppi > 0).sum()//2)} edges",
          flush=True)

    ds = LincsDataset(dc); sp = build_splits(ds, dc)
    out = {"ckpt": os.path.basename(a.ckpt), "fold": a.fold, "n_pathways": int(M.shape[0]),
           "v5_reference": {k: v["pearson"] for k, v in V5_REF.items()}, "v6_reference": V6_REF,
           "weights": {}}

    for which in (["raw", "ema"] if a.weights == "both" else [a.weights]):
        model = load_v7(dc, cfg, M, ppi, a.ckpt, which)
        print(f"\n{'#'*78}\n### {which.upper()} weights\n{'#'*78}", flush=True)
        rec = {}
        for name, key in SPLIT_KEY.items():
            idx = sp[key]
            rep = idx[ds.strength[idx] >= dc.eval_min_strength]
            cap = V5_REF[name]["cap"]
            if len(rep) < 500:
                continue
            hn = min(len(rep), cap)
            h_idx = np.random.default_rng(0).choice(rep, hn, replace=False) if hn < len(rep) else rep
            t0 = time.time()
            yh, yt = predict(model, ds, h_idx, cfg, a.batch)
            m = metrics(yh, yt, ds.phase[h_idx])
            srec = {"headline": m, "strata": {}}
            print(f"  {name:17s} n={hn:<5} ({time.time()-t0:.0f}s)  pearson={m['pearson_median']:.4f}  "
                  f"R2={m['r2_overall']:+.4f}  cDEG@100={m['common_degs@100']:.3f}"
                  f"   | v5 {V5_REF[name]['pearson']:.3f}  v6 {V6_REF[name]:.4f}"
                  f"   | v7-v6 = {m['pearson_median']-V6_REF[name]:+.4f}", flush=True)
            for lo, hi in (STRATA if a.strata_n else []):
                band = idx[(ds.strength[idx] >= lo) & (ds.strength[idx] < hi)]
                if len(band) < 200:
                    continue
                sn = min(a.strata_n, len(band))
                s_idx = (np.sort(np.random.default_rng(11).choice(band, sn, replace=False))
                         if sn < len(band) else band)
                ph, pt = predict(model, ds, s_idx, cfg, a.batch)
                srec["strata"][f"{lo}-{hi}"] = metrics(ph, pt)
            rec[name] = srec
        out["weights"][which] = rec

    if a.weights == "both":
        print(f"\n{'='*78}\nEMA vs RAW (positive = EMA helped)")
        for name in out["weights"]["raw"]:
            r = out["weights"]["raw"][name]["headline"]["pearson_median"]
            e = out["weights"]["ema"][name]["headline"]["pearson_median"]
            print(f"  {name:17s} raw={r:.4f}  ema={e:.4f}  delta={e-r:+.4f}")

    dest = a.out or os.path.join(os.path.dirname(HERE), "results",
                                 f"eval_v7_{os.path.basename(a.ckpt).replace('.pt','')}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
