# -*- coding: utf-8 -*-
"""RESULTS 93.2 item 1 as amended by 93.5 C4: the H2 learning curve (reported, never a gate). PI-written glue around
chromatin_funnel.run_t1; no new estimator.

    python model/v9/chromatin_h2.py --data_dir D --provenance P --out chromatin_h2_93.json

For k in {4, 6, 8} covered dev-train cells (5 subsets each, default_rng(9300 + k)) and k = all (11, one fit), T1 is refitted on
the subset (its LOCO and N1 over the same subset) and scored on the drug-known dev rows. Recorded per subset: FBC - N1 (the
read quantity) and FBC - FB (reported); the same subsets carry M2's planted P1 at 0.5 % (draw 0), to show the estimator's own
small-k behaviour. Prints no reading; read_chromatin93.py reads the JSON.
"""
import argparse
import json
import os
import time

import numpy as np

import chromatin_funnel as cf
import chromatin_power as cp

KS = (4, 6, 8)
N_SUB = 5
PI_PLANT = 0.005


def subsets(fit):
    fit = sorted(fit)
    out = {}
    for k in KS:
        rng = np.random.default_rng(9300 + k)
        out[str(k)] = [sorted(rng.choice(fit, size=k, replace=False).tolist()) for _ in range(N_SUB)]
    out['all'] = [fit]
    return out


def t1_deltas(ctx, y, B, fit):
    specs = {k: (B[k], ['full']) for k in ('FB', 'FBC', 'N1')}
    out, mu, level = cf.run_t1(ctx, y, specs, fit)
    known = level[ctx.dev_mask] <= 2
    s = {k: cf.score(ctx, y, out[(k, 'full')]['y_hat_dev'], known) for k in ('FB', 'FBC', 'N1')}
    return {'FBC_minus_N1': s['FBC']['all'] - s['N1']['all'], 'FBC_minus_FB': s['FBC']['all'] - s['FB']['all'],
            'FBC_minus_N1_per_cell': {c: s['FBC']['per_cell'][c] - s['N1']['per_cell'][c] for c in cf.DEV_CELLS},
            'kappa': {k: [out[(k, 'full')]['kappa'], out[(k, 'full')]['kappa_d']] for k in ('FB', 'FBC', 'N1')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t = time.time()
    ctx = cf.prepare(a.data_dir, a.provenance)
    fit_all = ctx.enc['rank_normal']['cov_dt']
    assert len(fit_all) == 11 and not set(fit_all) & set(cf.DEV_CELLS)
    S = subsets(fit_all)
    fstar, rho = cp.synthetic_feature(ctx, 0)
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    y_plant, alpha = cp.plant(ctx, ctx.y, mu_e, fstar, 'P1', PI_PLANT, 0)
    res = {'subsets': S, 'pi_plant': PI_PLANT, 'alpha_plant': alpha, 'rho_plant': rho, 'real': {}, 'planted_P1': {}}
    for k, subs in S.items():
        res['real'][k], res['planted_P1'][k] = [], []
        for fit in subs:
            t1 = time.time()
            Br = {kind: cf.feature_builder(ctx, 'rank_normal', kind, fit) for kind in ('FB', 'FBC', 'N1')}
            res['real'][k].append(t1_deltas(ctx, ctx.y, Br, fit))
            Bp = cp.builders(ctx, fstar, fit, 0)
            res['planted_P1'][k].append(t1_deltas(ctx, y_plant, Bp, fit))
            print('k=%s subset %d/%d done in %.0f s' % (k, len(res['real'][k]), len(subs), time.time() - t1), flush=True)
    res['inputs'] = ctx.inputs
    res['seconds'] = time.time() - t
    json.dump(cf.jsonable(res), open(a.out, 'w'), indent=1)
    print('wrote', a.out, 'seconds %.0f' % res['seconds'], flush=True)   # no reading is echoed


if __name__ == '__main__':
    main()
