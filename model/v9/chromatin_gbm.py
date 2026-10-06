# -*- coding: utf-8 -*-
"""RESULTS 93.2 item 2 as amended by 93.5 (review 050) and the PI amendment of 2026-10-06: H3, a gradient-boosted
(non-linear) form of the chromatin test, with its calibration. Written by W30 (agy, gemini-3.1-pro-high); PI fixes: fold
features built from each fold's training cells (W30 built N1 from all covered cells, the held-out fold included), draw-by-draw
calibration, file-sha1 marker with the full input sha1s, no process pool. Prints no reading; read_chromatin93.py reads it.

    python model/v9/chromatin_gbm.py --data_dir D --provenance P --out chromatin_gbm_93.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import lightgbm

import chromatin_funnel as cf
import chromatin_power as cp

def h3_reading(s_C, s_B, s_N1, bars):
    d = cf.paired(s_C, s_B)
    dn = cf.paired(s_C, s_N1)
    
    conjuncts = {
        'all_or_top': bool(d['all'] >= bars['t1'] or d['top'] >= bars['t1_top']),
        'cells_pos': bool(d['cells_pos'] >= 4),
        'centred_all': bool(d['centred_all'] >= bars['t1_centred']),
        'centred_cells_pos': bool(d['centred_cells_pos'] >= 4),
        'vs_N1_all': bool(dn['all'] >= bars['t1'] / 2.0),
        'vs_N1_cells_pos': bool(dn['cells_pos'] >= 4)
    }
    
    return {
        'delta': d,
        'vs_N1': dn,
        'conjuncts': conjuncts,
        'pass': bool(all(conjuncts.values()))
    }

def make_dataset(ctx, r_indices, g_indices, mu_matrix, F_dict):
    n_samples = len(r_indices)
    D = 0
    for v in F_dict.values():
        if v.size > 0:
            D = v.shape[1]
            break
            
    X = np.empty((n_samples, 2 + D), dtype=np.float32)
    
    r_uniq = np.unique(r_indices)
    mu_norm = np.zeros(len(ctx.y), dtype=np.float32)
    mu_norm[r_uniq] = np.linalg.norm(mu_matrix[r_uniq], axis=1)
    
    X[:, 0] = mu_matrix[r_indices, g_indices]
    X[:, 1] = mu_norm[r_indices]
    
    if D > 0:
        cells = ctx.rows['cell'][r_indices]
        for c, F_c in F_dict.items():
            mask = (cells == c)
            if not mask.any():
                continue
            X[mask, 2:] = F_c[g_indices[mask]]
        
    return X

def get_fitting_pairs(ctx, y, seed=9301):
    mu_full, level_full = cf.condition_means(ctx, y, ctx.fit_mask)
    
    cov_dt = ctx.enc['rank_normal']['cov_dt']
    is_cov = np.isin(ctx.rows['cell'], cov_dt)
    
    is_fitting_row = is_cov & ctx.fit_mask & (level_full <= 2)
    fit_rows = np.flatnonzero(is_fitting_row)
    
    G = ctx.y.shape[1]
    n_pairs = len(fit_rows) * G
    
    rng = np.random.default_rng(seed)
    sample_size = int(n_pairs * 0.05)
    sampled_indices = rng.choice(n_pairs, size=sample_size, replace=False)
    
    sampled_r = fit_rows[sampled_indices // G]
    sampled_g = sampled_indices % G
    
    return fit_rows, sampled_r, sampled_g

def get_folds(ctx):
    cov_dt = ctx.enc['rank_normal']['cov_dt']
    sorted_cells = sorted(cov_dt)
    rng = np.random.default_rng(9302)
    permuted = rng.permutation(sorted_cells)
    folds = [permuted[i::4] for i in range(4)]
    return folds

def _lgbm(mcs):
    return lightgbm.LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        subsample=1.0, colsample_bytree=1.0, n_jobs=4,
        random_state=9301, deterministic=True, force_row_wise=True,
        verbose=-1, min_child_samples=mcs)


FOLD_LOG = []   # (fold index, training-row cells, feature-builder cells) per fold fit -- read by the tests


def fold_pools(ctx, y, folds):
    """Review 052 C3: each fold's mu_h (a pool without the fold), computed once per run_h3 and shared by the three arms."""
    out = []
    for fold_cells in folds:
        pool = ctx.fit_mask & ~np.isin(ctx.rows['cell'], list(fold_cells))
        cf.assert_no_dev(ctx, pool, 'fold pool')
        out.append(cf.condition_means(ctx, y, pool)[0])
    return out


def run_loocv_arm(ctx, y, make_B, arm, fit_cells, fit_rows, sampled_r, sampled_g, folds, _leaky=False, mus=None):
    """Choose min_child_samples for one arm by grouped 4-fold over the covered fitting cells (93.5 C5 / PI amendment).
    PI fix (W30's N1 leak): each fold's features are built from THAT fold's training cells (make_B(train_cells)), so N1's
    gene-generic means never include the held-out fold's chromatin; its target uses mu_h from a pool without the fold.
    Each fold's design is built once and reused for the three min-child values (review 052 C3)."""
    G = ctx.y.shape[1]
    mus = fold_pools(ctx, y, folds) if mus is None else mus
    prepared = []
    for fi, fold_cells in enumerate(folds):
        fold_cells = list(fold_cells)
        train_cells = [c for c in fit_cells if c not in fold_cells]
        mu_h = mus[fi]
        train_mask = np.ones(len(sampled_r), bool) if _leaky else ~np.isin(ctx.rows['cell'][sampled_r], fold_cells)
        train_r, train_g = sampled_r[train_mask], sampled_g[train_mask]
        if not _leaky:
            assert not np.isin(ctx.rows['cell'][train_r], fold_cells).any(), 'a fold cell entered its own fit'
        cf.assert_no_dev(ctx, train_r, 'train rows')
        F = make_B(train_cells)[arm](None)
        val_r = fit_rows[np.isin(ctx.rows['cell'][fit_rows], fold_cells)]
        X_val = make_dataset(ctx, np.repeat(val_r, G), np.tile(np.arange(G), len(val_r)), mu_h, F) if len(val_r) else None
        prepared.append((fi, train_r, train_g, mu_h, F, val_r, X_val, sorted(train_cells)))
    best_mcs, best_score = None, -np.inf
    for mcs in (50, 200, 1000):
        fold_scores = []
        for fi, train_r, train_g, mu_h, F, val_r, X_val, train_cells in prepared:
            FOLD_LOG.append((fi, sorted(set(ctx.rows['cell'][train_r].tolist())), train_cells))
            model = _lgbm(mcs)
            t0 = time.time()
            model.fit(make_dataset(ctx, train_r, train_g, mu_h, F), (y - mu_h)[train_r, train_g])
            print('fit %s mcs %d fold %d: %.1f s' % (arm, mcs, fi, time.time() - t0), flush=True)
            if X_val is None:
                continue
            e_hat = model.predict(X_val)
            fold_scores.append(float(np.nanmean(cf.row_pearson(mu_h[val_r] + e_hat.reshape(len(val_r), G), y[val_r]))))
        m = float(np.mean(fold_scores))
        if m > best_score:
            best_score, best_mcs = m, mcs
    return best_mcs, best_score


def fit_predict_arm(ctx, y, make_B, arm, fit_cells, best_mcs, sampled_r, sampled_g):
    mu, _ = cf.condition_means(ctx, y, ctx.fit_mask)
    cf.assert_no_dev(ctx, sampled_r, 'final fit train rows')
    F = make_B(fit_cells)[arm](None)
    model = _lgbm(best_mcs)
    t0 = time.time()
    model.fit(make_dataset(ctx, sampled_r, sampled_g, mu, F), (y - mu)[sampled_r, sampled_g])
    print('final fit %s: %.1f s' % (arm, time.time() - t0), flush=True)
    dev = np.flatnonzero(ctx.dev_mask)
    G = ctx.y.shape[1]
    e_hat = model.predict(make_dataset(ctx, np.repeat(dev, G), np.tile(np.arange(G), len(dev)), mu, F))
    return mu[dev] + e_hat.reshape(len(dev), G)


ARMS = ('FB', 'FBC', 'N1')


def run_h3(ctx, y, make_B):
    """make_B(fit_cells) -> {'FB', 'FBC', 'N1': feature builder}; the real run passes the primary-encoding builders, a
    calibration draw passes cp.builders for that draw (93.2 item 2; 93.5 C2/C3)."""
    fit_cells = list(ctx.enc['rank_normal']['cov_dt'])
    fit_rows, sampled_r, sampled_g = get_fitting_pairs(ctx, y, seed=9301)
    folds = get_folds(ctx)
    chosen, cv = {}, {}
    mus = fold_pools(ctx, y, folds)
    for arm in ARMS:
        chosen[arm], cv[arm] = run_loocv_arm(ctx, y, make_B, arm, fit_cells, fit_rows, sampled_r, sampled_g, folds,
                                             mus=mus)
    yh = {arm: fit_predict_arm(ctx, y, make_B, arm, fit_cells, chosen[arm], sampled_r, sampled_g) for arm in ARMS}
    _, level = cf.condition_means(ctx, y, ctx.fit_mask)
    known = level[ctx.dev_mask] <= 2
    out = {'chosen_mcs': chosen, 'cv_score': cv, 'n_pairs_sampled': int(len(sampled_r)),
           'folds': [sorted(list(f)) for f in folds]}
    for name, sub in (('all', None), ('known', known)):
        s = {k: cf.score(ctx, y, yh[k], sub) for k in ARMS}
        out[name] = h3_reading(s['FBC'], s['FB'], s['N1'], cf.BARS[name])
    return out


def real_builders(ctx):
    return lambda fit: {k: cf.feature_builder(ctx, 'rank_normal', k, fit) for k in ARMS}


def calibrate(ctx, draws=3):
    """93.5 C3, draw by draw: each draw installs its own calibration tracks (cp.builders) before its cases run, and every case
    re-runs the whole procedure (fold choice included) on its planted y. PI fix: W30 built all draws' tasks first."""
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    records = []
    for d in range(draws):
        fstar, rho = cp.synthetic_feature(ctx, d)
        make_B = lambda fit, fstar=fstar, d=d: cp.builders(ctx, fstar, fit, d)   # noqa: E731
        fbar = np.mean(np.stack(list(fstar.values())), 0)
        fg = {c: fbar for c in fstar}                       # gene-generic: the same vector in every covered cell (dev too)
        cases = [('null', None, None, None)] + [(f, pi, f, fstar) for pi in PIS for f in ('P1', 'P2')] + \
                [('P3', 0.05, 'P3', fstar)] + [('G', pi, 'P1', fg) for pi in (0.02, 0.05)]
        rec = {'draw': d, 'rho': rho, 'cases': {}}
        for cid, pi, form, fmap in cases:
            t0 = time.time()
            if cid == 'null':
                yc, alpha = ctx.y, 0.0
            else:
                yc, alpha = cp.plant(ctx, ctx.y, mu_e, fmap, form, pi, d)
            r = run_h3(ctx, yc, make_B)
            r['alpha'] = float(alpha)
            rec['cases'][cid if pi is None else '%s_%g' % (cid, pi)] = r
            print('draw %d case %s %s done in %.0f s' % (d, cid, pi, time.time() - t0), flush=True)
        records.append(rec)
    return records


PIS = (0.005, 0.02, 0.05)


def faults_and_mde(records, rowset='known'):
    """Mechanical, as committed before the run (read_chromatin93.py reads the same way)."""
    p = lambda r, k: bool(r['cases'][k][rowset]['pass'])   # noqa: E731
    out = {'VOID_null': any(p(r, 'null') for r in records), 'VOID_P3': any(p(r, 'P3_0.05') for r in records),
           'VOID_G': any(p(r, 'G_%g' % pi) for r in records for pi in (0.02, 0.05)),
           'G_raw_delta': [r['cases']['G_%g' % pi][rowset]['delta']['all'] for r in records for pi in (0.02, 0.05)],
           'MDE': {}}
    for form in ('P1', 'P2'):
        out['MDE'][form] = next((pi for pi in PIS if all(p(r, '%s_%g' % (form, pi)) for r in records)), None)
    return out


def bench():
    n_samples = int(20000 * 978 * 0.05)
    X = np.random.randn(n_samples, 7).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    t0 = time.time()
    _lgbm(50).fit(X, y)
    print('One fit takes %.2f seconds' % (time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--dti', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=3)
    a = ap.parse_args()
    t = time.time()
    ctx = cf.prepare(a.data_dir, a.provenance, a.dti)
    assert len(ctx.enc['rank_normal']['cov_dt']) == 11
    records = calibrate(ctx, a.draws)
    res = {'real': run_h3(ctx, ctx.y, real_builders(ctx)), 'calibration': records,
           'faults_and_mde': {rs: faults_and_mde(records, rs) for rs in ('known', 'all')},
           'draws': a.draws, 'pis': PIS, 'inputs': ctx.inputs, 'seconds': time.time() - t}
    json.dump(cf.jsonable(res), open(a.out, 'w'), indent=1)
    marker = {'complete': True, 'outputs': {os.path.basename(a.out): cf.sha1_file(a.out)}, 'inputs': ctx.inputs}
    json.dump(marker, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'CHROMATIN93_H3_COMPLETE.json'), 'w'), indent=1)
    print('wrote', a.out, 'seconds %.0f' % res['seconds'], flush=True)   # no reading is echoed


if __name__ == '__main__':
    main()
