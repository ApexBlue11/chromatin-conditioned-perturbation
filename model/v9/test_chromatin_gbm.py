# -*- coding: utf-8 -*-
"""Tests for chromatin_gbm.py (RESULTS 93 H3). Synthetic worlds only (test_chromatin_funnel.world). Written by W30; PI
rewrite after the fixes: the leak test reads the normal path's fold log, the gene-generic case is planted on dev cells too,
and the fault/MDE logic is tested."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm  # noqa: E402

_Orig = lightgbm.LGBMRegressor


class _OneThread(_Orig):                                     # tests run single-threaded (thermal rule)
    def __init__(self, **kw):
        kw['n_jobs'] = 1
        super().__init__(**kw)


lightgbm.LGBMRegressor = _OneThread
import chromatin_funnel as cf  # noqa: E402
import chromatin_power as cp  # noqa: E402
import chromatin_gbm as gbm  # noqa: E402
from test_chromatin_funnel import world  # noqa: E402


def test_a_planted_cell_specific_gain_is_found_and_beats_N1():
    ctx = world(seed=1, n_perts=6, plant='gain', alpha=0.9, noise=0.05)
    res = gbm.run_h3(ctx, ctx.y, gbm.real_builders(ctx))
    assert res['known']['delta']['all'] > 0.01
    assert res['known']['vs_N1']['all'] > 0.002


def test_random_y_finds_nothing():
    ctx = world(seed=2, n_perts=6, noise=1.0)
    res = gbm.run_h3(ctx, ctx.y, gbm.real_builders(ctx))
    assert abs(res['known']['delta']['all']) < 0.01 and not res['known']['pass']


def test_no_fold_cell_enters_its_own_fit_or_its_N1_features_and_the_check_can_see_a_leak():
    ctx = world(seed=3, n_perts=4)
    folds = gbm.get_folds(ctx)
    gbm.FOLD_LOG.clear()
    gbm.run_h3(ctx, ctx.y, gbm.real_builders(ctx))
    assert len(gbm.FOLD_LOG) == 3 * 3 * 4                   # arms x min-child values x folds
    for fi, train_cells, feat_cells in gbm.FOLD_LOG:
        fold = set(folds[fi])
        assert not fold & set(train_cells), 'a fold cell entered its own training rows'
        assert not fold & set(feat_cells), 'a fold cell entered its own fold features (N1 means)'
        assert not set(cf.DEV_CELLS) & set(train_cells)
    # the same log shows a leak when one is forced
    gbm.FOLD_LOG.clear()
    fit_cells = list(ctx.enc['rank_normal']['cov_dt'])
    fit_rows, sr, sg = gbm.get_fitting_pairs(ctx, ctx.y)
    gbm.run_loocv_arm(ctx, ctx.y, gbm.real_builders(ctx), 'FB', fit_cells, fit_rows, sr, sg, folds, _leaky=True)
    assert any(set(folds[fi]) & set(tc) for fi, tc, _ in gbm.FOLD_LOG)


def test_determinism_and_one_pair_sample_for_every_arm():
    ctx = world(seed=4, n_perts=4)
    r1 = gbm.run_h3(ctx, ctx.y, gbm.real_builders(ctx))
    r2 = gbm.run_h3(ctx, ctx.y, gbm.real_builders(ctx))
    assert r1['known']['delta']['all'] == r2['known']['delta']['all'] and r1['chosen_mcs'] == r2['chosen_mcs']
    a = gbm.get_fitting_pairs(ctx, ctx.y)
    b = gbm.get_fitting_pairs(ctx, ctx.y * 2.0)                  # the sample depends on the rows, not on y's values
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_h3_reading_needs_an_N1_magnitude():
    mk = lambda v: {'all': v, 'top': v, 'per_cell': {c: v for c in cf.DEV_CELLS}, 'centred_all': v,   # noqa: E731
                    'centred_per_cell': {c: v for c in cf.DEV_CELLS}}
    s_B, s_C = mk(0.10), mk(0.11)
    bars = {'t1': 0.004, 't1_top': 0.008, 't1_centred': 0.002}
    for n1_all, expect in ((0.109, False), (0.107, True)):
        s_N1 = mk(n1_all)
        s_N1['per_cell'][cf.DEV_CELLS[0]] = s_N1['per_cell'][cf.DEV_CELLS[1]] = 0.111   # C > N1 in 4 of 6 cells
        assert gbm.h3_reading(s_C, s_B, s_N1, bars)['pass'] is expect


def test_a_gene_generic_planted_effect_is_absorbed_by_mu_and_does_not_pass():
    # Review 052 C1(a), as answered by the PI and corrected by the critic: a plant common to every COVERED cell is absorbed
    # into mu (the mean over all dev-train cells) except for about (uncovered share)^2 of it. In THIS world 12 of 14 train
    # cells are covered, so ~2 % survives and the planted G cannot clear the raw bar here. In the real data ~11 of 26 are
    # covered, so ~(15/26)^2 ~ 33 % survives and the real G check may well be informative (then VOID_G binds); its outcome is
    # not pre-stated. The reading's own protection is the S(C) - S(N1) magnitude conjunct (test_h3_reading_needs_an_N1_magnitude).
    ctx = world(seed=5, n_perts=12, noise=0.05)
    e = ctx.enc['rank_normal']
    cov = list(e['cov_dt']) + list(e['cov_dev'])
    rng = np.random.default_rng(0)
    fbar = rng.normal(size=ctx.y.shape[1]).astype(np.float32)
    for c in cov:
        i = ctx.cells.index(c)
        for k in range(3):
            if e['has'][i, k]:
                e['Ez'][i, :, k] = fbar + 0.05 * rng.normal(size=ctx.y.shape[1]).astype(np.float32)
    fg = {c: fbar for c in cov}
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    y_g, _ = cp.plant(ctx, ctx.y, mu_e, fg, 'P1', 0.5, 0)
    res = gbm.run_h3(ctx, y_g, gbm.real_builders(ctx))
    assert abs(res['known']['delta']['all']) < cf.BARS['known']['t1']     # absorbed by mu in this world (12 of 14 covered)
    assert res['known']['vs_N1']['all'] < cf.BARS['known']['t1'] / 2
    assert not res['known']['pass']


def test_faults_and_mde_read_three_of_three():
    def rec(passes):
        return {'cases': {k: {'known': {'pass': v, 'delta': {'all': 0.01}}} for k, v in passes.items()}}
    base = {'null': False, 'P3_0.05': False, 'G_0.02': False, 'G_0.05': False,
            'P1_0.005': False, 'P1_0.02': True, 'P1_0.05': True, 'P2_0.005': False, 'P2_0.02': False, 'P2_0.05': True}
    r = [rec(dict(base)) for _ in range(3)]
    r[1]['cases']['P2_0.05']['known']['pass'] = False            # P2 at 5 % passes in 2 of 3 draws only
    f = gbm.faults_and_mde(r)
    assert f['MDE'] == {'P1': 0.02, 'P2': None} and not (f['VOID_null'] or f['VOID_P3'] or f['VOID_G'])
    r[2]['cases']['null']['known']['pass'] = True
    assert gbm.faults_and_mde(r)['VOID_null']
