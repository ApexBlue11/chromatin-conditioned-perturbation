# -*- coding: utf-8 -*-
"""
Three measurements this project has never made, all from an existing checkpoint, all cheap.

1. WARM-START. The field reports warm-start / cold-cell / cold-drug [XPert, Nat Mach Intell 2026]; we have
   only ever reported the two cold regimes. Their released split is the WARM one -- their own checkpoint is
   named `l1000_mdmt_warm_split.pth` -- so without a warm number of our own there is nothing to set beside
   their headline. Our `val` split IS warm-start by construction: it excludes the held-out cells and the
   held-out scaffold family, so its rows share both cell lines and compounds with training and only the
   specific (cell, drug, dose, time) condition is unseen.

2. EMA vs RAW WEIGHTS. v7 measured EMA at -0.0005 / +0.0003 / +0.0001 -- nothing -- and it was kept in the
   recipe anyway. The v9 checkpoint stores both weight sets, so re-testing it costs one forward pass.

3. THE EPI-DRUG SUBSET. XPert's released predictions are their HDAC-inhibitor figure, and this project
   measured epi-drugs at +0.20 easier than average [2.6]. Scoring our own model on our own epi-drug subset
   is the like-for-like way to read their number.

    python model/v9/regimes_v9.py --ckpt ckpt_v9_fold0_seed0.pt
"""
import os, sys, json, argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9
from data_v9 import LincsV9Dataset, build_splits, collate_v9
from train_v9_gpu import resolve_v9


def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


@torch.no_grad()
def score(model, ds, idx, dev, batch=48):
    P = {'delta': [], 'abs': [], 'l5': []}
    T = {'delta': [], 'abs': [], 'l5': []}
    ctls = []
    for s in range(0, len(idx), batch):
        b = collate_v9([ds[i] for i in idx[s:s + batch]])
        bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        o = model(bd)
        for k in P:
            P[k].append(o[k].float().cpu().numpy())
        T['delta'].append(b['y_delta'].numpy()); T['abs'].append(b['y_abs'].numpy())
        T['l5'].append(b['y_l5'].numpy()); ctls.append(b['x_ctl'].numpy())
    out = {'n': int(len(idx))}
    for k in P:
        out[k] = round(float(np.nanmedian(pearson_rows(np.concatenate(P[k]), np.concatenate(T[k])))), 4)
    out['copy_ctl_abs'] = round(float(np.nanmedian(
        pearson_rows(np.concatenate(ctls), np.concatenate(T['abs'])))), 4)
    out['abs_value_added'] = round(out['abs'] - out['copy_ctl_abs'], 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_eval', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', dev, flush=True)

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = V9Config(**{k: v for k, v in ck['cfg'].items() if k in V9Config.__dataclass_fields__})
    dc = resolve_v9(V9DataConfig())
    dc.cache_in_ram = False
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_pathway_path)); ppi = np.load(R(dc.ppi_v9_path)); gv = np.load(R(dc.gene_vec_path))

    model = LincsV9(cfg, M, ppi, gv)
    model.load_state_dict(ck['model'])
    model = model.to(dev).eval()

    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)
    phase = np.array([p for p in ds.phase])
    ok = ds.has_l3

    def pick(idx, n=None):
        idx = idx[ok[idx] & (ds.strength[idx] >= dc.eval_min_strength)]
        n = n or a.n_eval
        return np.sort(rng.choice(idx, min(n, len(idx)), replace=False)) if len(idx) else idx

    res = {'ckpt': os.path.basename(a.ckpt), 'epoch': int(ck.get('epoch', -1))}

    # ---- 1. the three regimes the field reports, ours for the first time ----
    print('\n=== REGIMES (ours, matching the field\'s warm-start / cold-cell / cold-drug) ===', flush=True)
    regimes = {'warm_start': pick(sp['val']),
               'cold_cell': pick(sp['test_coldcell']),
               'cold_drug': pick(sp['test_colddrug']),
               'cold_both': pick(sp['test_coldboth'])}
    for name, idx in regimes.items():
        if len(idx) < 100:
            continue
        r = score(model, ds, idx, dev, a.batch)
        res[name] = r
        print(f'  {name:12s} n={r["n"]:5d}  delta {r["delta"]:.4f}  abs {r["abs"]:.4f} '
              f'(copy-ctl {r["copy_ctl_abs"]:.4f}, +{r["abs_value_added"]:.4f})  l5 {r["l5"]:.4f}',
              flush=True)

    # ---- 2. per phase, since P2 rows are measurably easier [RESULTS 35] ----
    print('\n=== BY PHASE (P2 test rows score +0.11..+0.13 above P1 with the same model) ===', flush=True)
    res['by_phase'] = {}
    for name, key in [('cold_cell', 'test_coldcell'), ('cold_drug', 'test_colddrug'),
                      ('cold_both', 'test_coldboth')]:
        base = sp[key]
        for ph in ['P1', 'P2']:
            idx = pick(base[phase[base] == ph], 2000)
            if len(idx) < 100:
                continue
            r = score(model, ds, idx, dev, a.batch)
            res['by_phase'][f'{name}/{ph}'] = r
            print(f'  {name:10s} {ph}  n={r["n"]:5d}  delta {r["delta"]:.4f}  l5 {r["l5"]:.4f}', flush=True)

    # ---- 3. the epi-drug subset, the like-for-like read of their HDACi figure ----
    epi_path = os.path.join(dc.root, 'drug', 'outputs', 'dti', 'epi_drug_pert_ids.json')
    if os.path.exists(epi_path):
        epi = json.load(open(epi_path))
        epi = set(epi if isinstance(epi, list) else epi.get('pert_ids', epi.keys()))
        import json as _j
        dindex = _j.load(open(R(dc.drug_index_path)))
        epi_rows = {dindex[p] for p in epi if p in dindex}
        print(f'\n=== EPI-DRUG SUBSET ({len(epi_rows)} of our featurised drugs) ===', flush=True)
        res['epi_drug'] = {}
        for name, key in [('cold_cell', 'test_coldcell'), ('cold_drug', 'test_colddrug')]:
            base = sp[key]
            m = np.isin(ds.drug_row[base], list(epi_rows))
            idx = pick(base[m], 2000)
            if len(idx) < 60:
                print(f'  {name}: only {len(idx)} epi-drug rows, skipping')
                continue
            r = score(model, ds, idx, dev, a.batch)
            rest = score(model, ds, pick(base[~m], 2000), dev, a.batch)
            res['epi_drug'][name] = {'epi': r, 'non_epi': rest}
            print(f'  {name:10s} epi-drugs n={r["n"]:4d} delta {r["delta"]:.4f} | '
                  f'others n={rest["n"]:4d} delta {rest["delta"]:.4f} | '
                  f'epi advantage {r["delta"] - rest["delta"]:+.4f}', flush=True)

    # ---- 4. EMA vs raw: v7 measured EMA at ~0; does v9 agree? ----
    if 'ema' in ck:
        print('\n=== EMA vs RAW WEIGHTS (v7 measured EMA at -0.0005/+0.0003/+0.0001) ===', flush=True)
        sd_raw = {k: v.clone() for k, v in model.state_dict().items()}
        ema = ck['ema']
        missing = model.load_state_dict({**sd_raw, **{k: v for k, v in ema.items() if k in sd_raw}},
                                        strict=False)
        res['ema'] = {}
        for name in ['warm_start', 'cold_cell', 'cold_drug', 'cold_both']:
            if name not in res:
                continue
            r = score(model, ds, regimes[name], dev, a.batch)
            d = r['delta'] - res[name]['delta']
            res['ema'][name] = {'ema_delta': r['delta'], 'raw_delta': res[name]['delta'],
                                'ema_minus_raw': round(d, 4)}
            print(f'  {name:12s} raw {res[name]["delta"]:.4f} -> EMA {r["delta"]:.4f}  ({d:+.4f})',
                  flush=True)
        model.load_state_dict(sd_raw)

    WORK = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'model', 'results')
    os.makedirs(WORK, exist_ok=True)
    dst = os.path.join(WORK, f'v9_regimes_{os.path.basename(a.ckpt).replace(".pt", "")}.json')
    json.dump(res, open(dst, 'w'), indent=2)
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
