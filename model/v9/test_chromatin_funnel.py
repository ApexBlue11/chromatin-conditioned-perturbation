# -*- coding: utf-8 -*-
"""Tests for chromatin_funnel.py (RESULTS 91). Synthetic worlds only; every test calls the module's own functions."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402

G = cf.G
DEV = list(cf.DEV_CELLS)
TRAIN_COV = ['T%02d' % i for i in range(12)]
TRAIN_UNC = ['U00', 'U01']


def world(seed=0, n_perts=40, plant=None, alpha=0.0, chrom_mark=1, noise=0.3):
    """Every cell x drug at two doses, one time. y = drug profile (+ dose scaling) + cell offset + noise (+ a planted term on
    the cell's own chromatin mark `chrom_mark`). Returns ctx (both encodings identical) and the planted feature per cell."""
    rng = np.random.default_rng(seed)
    cells = DEV + TRAIN_COV + TRAIN_UNC
    perts = ['p%03d' % i for i in range(n_perts)]
    prof = rng.normal(size=(n_perts, G)).astype(np.float32)
    basal = {c: rng.normal(size=G).astype(np.float32) * 2 + 8 for c in cells}
    Ez = {c: rng.normal(size=(G, 3)).astype(np.float32) for c in cells}
    has = {c: np.array([True, True, True]) if c not in TRAIN_UNC else np.zeros(3, bool) for c in cells}
    has['U937'] = np.array([True, False, False])
    rows = {k: [] for k in ('X', 'C', 'pert', 'dose', 'time', 'cell', 'row_index')}
    w_d = rng.normal(size=n_perts)
    i = 0
    for c in cells:
        off = rng.normal(size=G).astype(np.float32) * 0.2
        for p in range(n_perts):
            for dose in (1.0, 10.0):
                mu = prof[p] * (0.5 if dose == 1.0 else 1.0)
                y = mu + off + rng.normal(size=G).astype(np.float32) * noise
                if plant and has[c][chrom_mark]:
                    f = Ez[c][:, chrom_mark]
                    y = y + alpha * {'gain': mu * f, 'shift': f, 'drug': w_d[p] * f}[plant]
                ctl = basal[c] + rng.normal(size=G).astype(np.float32) * 0.1
                rows['X'].append(ctl + y)
                rows['C'].append(ctl)
                rows['pert'].append(perts[p])
                rows['dose'].append(dose)
                rows['time'].append(24.0)
                rows['cell'].append(c)
                rows['row_index'].append(i)
                i += 1
    rows = {k: np.asarray(v) for k, v in rows.items()}
    rows['X'] = rows['X'].astype(np.float32)
    rows['C'] = rows['C'].astype(np.float32)
    rows['pert'] = rows['pert'].astype(str)
    rows['cell'] = rows['cell'].astype(str)
    ctx = cf.ctx_from_arrays(rows, np.isin(rows['cell'], DEV))
    order = ctx.cells
    b = np.stack([(basal[c] - basal[c].mean()) / basal[c].std() for c in order]).astype(np.float32)
    E = np.stack([np.where(has[c][None, :], Ez[c], 0.0) for c in order]).astype(np.float32)
    H = np.stack([has[c] for c in order])
    cf.set_encodings(ctx, {'rank_normal': (b, E, H), 'v9': (b, E, H)}, np.zeros((len(order), 2), np.float32))
    return ctx


def slow_condition_means(ctx, y, pool):
    """Oracle on toy inputs: per row, the finest level with >= 1 pool cell other than the row's own; mean of cell means."""
    out = np.zeros_like(y)
    for i in range(len(y)):
        for codes in ctx.levels:
            cm = {}
            for j in np.flatnonzero(pool & (codes == codes[i]) & (ctx.cell_id != ctx.cell_id[i])):
                cm.setdefault(ctx.cell_id[j], []).append(y[j])
            if cm:
                out[i] = np.mean([np.mean(v, 0) for v in cm.values()], 0)
                break
    return out


def test_condition_means_match_the_slow_reference_with_fold_exclusion():
    ctx = world(n_perts=4)
    sub = np.zeros(len(ctx.y), bool)
    sub[::7] = True                                 # sparse rows so back-off happens
    pool = ctx.fit_mask & sub & (ctx.rows['cell'] != 'T03')
    mu, level = cf.condition_means(ctx, ctx.y, pool)
    ref = slow_condition_means(ctx, ctx.y, pool)
    np.testing.assert_allclose(mu, ref, rtol=1e-5, atol=1e-5)
    assert set(np.unique(level)) - {0, 1, 2, 3} == set()


def test_condition_mean_pool_refuses_a_dev_row():
    ctx = world(n_perts=4)
    with pytest.raises(AssertionError, match='GUARD'):
        cf.condition_means(ctx, ctx.y, np.ones(len(ctx.y), bool))


def test_suff_stats_refuse_a_dev_row():
    ctx = world(n_perts=4)
    mu, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    with pytest.raises(AssertionError, match='GUARD'):
        cf.suff_stats(ctx, ctx.y, mu, np.flatnonzero(ctx.dev_mask)[:5])


def _specs(ctx, kinds, fit):
    return {k: (cf.feature_builder(ctx, 'rank_normal', k, fit), ['full']) for k in kinds}


def test_t1_detects_a_planted_gain_rule_and_not_a_permuted_one():
    ctx = world(plant='gain', alpha=0.4, chrom_mark=1)
    fit = ctx.enc['rank_normal']['cov_dt']
    out, mu, _ = cf.run_t1(ctx, ctx.y, _specs(ctx, ['FB', 'FBC'], fit), fit)
    s_fb = cf.score(ctx, ctx.y, out[('FB', 'full')]['y_hat_dev'])
    s_fbc = cf.score(ctx, ctx.y, out[('FBC', 'full')]['y_hat_dev'])
    assert s_fbc['all'] - s_fb['all'] > 0.01
    th = np.asarray(out[('FBC', 'full')]['theta'])
    assert th[4 + 2] > 0.2                           # the gain coefficient on mark 1 (regressor block [f, mu*f], f=[b, A, K, M])
    # the same world with the chromatin permuted across genes: the rule has nothing to find
    perm = np.random.default_rng(1).permutation(G)
    ctx.enc['rank_normal']['Ez'] = ctx.enc['rank_normal']['Ez'][:, perm, :]
    out2, _, _ = cf.run_t1(ctx, ctx.y, _specs(ctx, ['FB', 'FBC'], fit), fit)
    d2 = cf.score(ctx, ctx.y, out2[('FBC', 'full')]['y_hat_dev'])['all'] - cf.score(ctx, ctx.y, out2[('FB', 'full')]['y_hat_dev'])['all']
    assert d2 < 0.002


def test_a_drug_independent_shift_moves_the_raw_score_not_the_centred_one():
    ctx = world(plant='shift', alpha=0.5, chrom_mark=1)
    fit = ctx.enc['rank_normal']['cov_dt']
    out, _, _ = cf.run_t1(ctx, ctx.y, _specs(ctx, ['FB', 'FBC'], fit), fit)
    d = cf.paired(cf.score(ctx, ctx.y, out[('FBC', 'full')]['y_hat_dev']), cf.score(ctx, ctx.y, out[('FB', 'full')]['y_hat_dev']))
    assert d['all'] > 0.005
    assert abs(d['centred_all']) < d['all'] / 5


def test_an_unseen_drug_gets_mu_plus_the_global_rule():
    ctx = world(n_perts=4)
    F = cf.feature_builder(ctx, 'rank_normal', 'FBC', ctx.enc['rank_normal']['cov_dt'])(None)
    theta = np.arange(8, dtype=float) / 10
    delta = np.zeros((len(ctx.perts), 8))
    rows = np.flatnonzero(ctx.dev_mask)[:3]
    mu = np.random.default_rng(0).normal(size=(3, G)).astype(np.float32)
    got = cf.t1_predict(ctx, mu, rows, F, theta, delta)
    for q, r in enumerate(rows):
        f = F[ctx.rows['cell'][r]].astype(float)
        np.testing.assert_allclose(got[q], mu[q] + f @ theta[:4] + mu[q] * (f @ theta[4:]), rtol=1e-5, atol=1e-5)
    assert np.all(np.isfinite(got))


def test_hierarchical_ridge_matches_the_stacked_least_squares():
    rng = np.random.default_rng(0)
    D, K = 5, 4
    X = [rng.normal(size=(30, K)) for _ in range(D)]
    t = [rng.normal(size=30) for _ in range(D)]
    H = np.stack([x.T @ x for x in X])
    B = np.stack([x.T @ y for x, y in zip(X, t)])
    nc = np.array([3, 3, 3, 1, 3])                   # drug 3 is below the floor: no deviation
    lg, ld = 2.0, 5.0
    th, de = cf.solve_hier(H, B, nc, lg, ld)
    # stacked design over [theta, delta_0..delta_4 (active only)]
    act = [d for d in range(D) if nc[d] >= cf.MIN_DRUG_CELLS]
    cols = K * (1 + len(act))
    Z = np.zeros((30 * D, cols))
    for d in range(D):
        Z[30 * d:30 * (d + 1), :K] = X[d]
        if d in act:
            j = act.index(d)
            Z[30 * d:30 * (d + 1), K * (1 + j):K * (2 + j)] = X[d]
    P = np.diag([lg] * K + [ld] * (cols - K))
    sol = np.linalg.solve(Z.T @ Z + P, Z.T @ np.concatenate(t))
    np.testing.assert_allclose(th, sol[:K], rtol=1e-6, atol=1e-8)
    for j, d in enumerate(act):
        np.testing.assert_allclose(de[d], sol[K * (1 + j):K * (2 + j)], rtol=1e-6, atol=1e-8)
    assert np.all(de[3] == 0)


def test_n1_uses_each_marks_own_cells_and_excludes_the_held_out_cell():
    ctx = world(n_perts=2)
    fit = ctx.enc['rank_normal']['cov_dt']
    Ez, has = ctx.enc['rank_normal']['Ez'], ctx.enc['rank_normal']['has']
    F = cf.feature_builder(ctx, 'rank_normal', 'N1', fit)('T00')
    others = [ctx.cells.index(c) for c in fit if c != 'T00']
    np.testing.assert_allclose(F['T05'][:, 2], Ez[others, :, 1].mean(0), rtol=1e-5)
    assert np.all(F['U937'][:, 2] == 0)              # U937 has ATAC only: its H3K27ac slot stays empty
    np.testing.assert_allclose(F['U937'][:, 1], Ez[others, :, 0].mean(0), rtol=1e-5)
    F2 = cf.n2_features(ctx, 'rank_normal')
    np.testing.assert_allclose(F2['HEK293T'][:, 1:], Ez[ctx.cells.index('HL60')], rtol=1e-6)
    np.testing.assert_allclose(F2['U937'][:, 1:], Ez[ctx.cells.index('VCAP')], rtol=1e-6)   # U937 takes VCAP's three marks


def test_t2_uniform_is_the_neighbour_mean_and_excludes_the_own_cell():
    ctx = world(n_perts=3)
    cov = ctx.enc['rank_normal']['cov_dt']
    q = np.flatnonzero(ctx.rows['cell'] == 'T02')[:4]
    nb_cell, nb_mean = cf.neighbour_table(ctx, ctx.y, cov, q)
    assert not np.any(nb_cell == ctx.cells.index('T02'))
    got = cf.retrieve(ctx, q, nb_cell, nb_mean, np.zeros((len(ctx.cells),) * 2), np.inf)
    for k, r in enumerate(q):
        same = np.flatnonzero(np.isin(ctx.rows['cell'], [c for c in cov if c != 'T02']) & (ctx.levels[0] == ctx.levels[0][r]))
        np.testing.assert_allclose(got[k], ctx.y[same].mean(0), rtol=1e-4, atol=1e-5)


def test_t3_held_out_cell_never_enters_its_own_fit():
    ctx = world(n_perts=30)
    v, _ = cf.spreads(ctx, ctx.y, TRAIN_COV)
    F = cf.feature_builder(ctx, 'rank_normal', 'FBC', TRAIN_COV)(None)
    a = cf.t3_fit_predict(v, F, TRAIN_COV[1:], ['T00'], 1e-2)['T00']
    v2 = dict(v)
    v2['T00'] = -v['T00']                           # the held-out cell's own target flipped: its prediction must not move
    b = cf.t3_fit_predict(v2, F, TRAIN_COV[1:], ['T00'], 1e-2)['T00']
    np.testing.assert_allclose(a, b)


def test_load_refuses_a_bundle_that_is_not_the_split(tmp_path):
    n = 100
    np.savez(tmp_path / 'xpert_mdmt_splits.npz', X=np.zeros((n, 3)), X_ctl=np.zeros((n, 3)), meta_pert_id=np.array(['a'] * n),
             meta_dose=np.zeros(n), meta_time=np.zeros(n), meta_cell=np.array(['c'] * n), row_index=np.arange(n),
             split_split_cold_cell_1=np.array(['train'] * 99 + ['test']))
    with pytest.raises(AssertionError, match='47,509'):
        cf.load_train_rows(str(tmp_path))


def test_scores_use_the_scorer_of_record():
    assert cf.row_pearson is cf.score_dev.row_pearson and cf.centred_r is cf.score_dev.centred_r
