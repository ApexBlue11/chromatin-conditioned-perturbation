# -*- coding: utf-8 -*-
"""Tests for chromatin_power.py (RESULTS 91.9 M2). Synthetic worlds only; every test calls the module's own functions."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402
import chromatin_power as cp  # noqa: E402
from test_chromatin_funnel import world  # noqa: E402


def test_fstar_keeps_redundancy_with_basal_and_loses_the_gene_link():
    ctx = world(n_perts=4)
    e = ctx.enc['rank_normal']
    # give the base tracks a known correlation with b so rho is meaningful
    for i in range(len(ctx.cells)):
        if e['has'][i, 1]:
            e['Ez'][i, :, 1] = (0.6 * e['b'][i] + 0.8 * e['Ez'][i, :, 1]).astype(np.float32)
    f, rho = cp.synthetic_feature(ctx, draw=0)
    assert 0.4 < rho < 0.8
    tr = cp.base_track(ctx)
    for c in e['cov_dt'][:5]:
        b = e['b'][ctx.cells.index(c)]
        assert abs(np.corrcoef(f[c], b)[0, 1] - rho) < 0.15
        resid_track = tr[c] - (tr[c] @ b) / (b @ b) * b       # the part of the real track not explained by b
        resid_f = f[c] - (f[c] @ b) / (b @ b) * b
        assert abs(np.corrcoef(resid_track, resid_f)[0, 1]) < 0.15


def test_plant_scales_the_component_to_the_requested_variance_fraction():
    ctx = world(n_perts=6)
    f, _ = cp.synthetic_feature(ctx, draw=1)
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    R = np.flatnonzero(np.isin(ctx.rows['cell'], list(f)) & ctx.fit_mask)
    var_e = np.var(ctx.y[R] - mu_e[R])
    for form in ('P1', 'P2', 'P3', 'P4'):
        yp, alpha = cp.plant(ctx, ctx.y, mu_e, f, form, 0.05, draw=1)
        frac = np.var((yp - ctx.y)[R]) / var_e
        assert abs(frac - 0.05) < 0.0005, (form, frac)
        uncovered = ~np.isin(ctx.rows['cell'], list(f))
        assert np.array_equal(yp[uncovered], ctx.y[uncovered])


def test_a_large_planted_gain_passes_t1_and_nothing_planted_does_not():
    ctx = world(n_perts=40, noise=0.3)
    fit = ctx.enc['rank_normal']['cov_dt']
    f, _ = cp.synthetic_feature(ctx, draw=0)
    B = cp.builders(ctx, f, fit)
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    yp, _ = cp.plant(ctx, ctx.y, mu_e, f, 'P1', 0.10, draw=0)
    hit, null = cp.t1_case(ctx, yp, B, fit), cp.t1_case(ctx, ctx.y, B, fit)
    assert hit['all']['pass'] and hit['known']['pass']
    assert not null['all']['pass'] and not null['known']['pass']


def test_a_large_drug_independent_shift_is_stopped_by_the_centred_conjunct():
    ctx = world(n_perts=40, noise=0.3)
    fit = ctx.enc['rank_normal']['cov_dt']
    f, _ = cp.synthetic_feature(ctx, draw=0)
    B = cp.builders(ctx, f, fit)
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    yp, _ = cp.plant(ctx, ctx.y, mu_e, f, 'P3', 0.10, draw=0)
    r = cp.t1_case(ctx, yp, B, fit)
    assert r['all']['delta_all'] > 0               # the raw score rewards the cell-constant offset ...
    assert not r['all']['conjuncts']['cells_centred_pos_ge_4'] and not r['all']['pass']   # ... the centred conjunct does not


def test_mde_and_summary_logic():
    def rec(passes):
        nul = {'pass': False, 'resid': {'all': 0.0}}
        return {'null_T1': {'all': nul, 'known': nul},
                'T1': {'%s_%g' % (fm, pi): {rs: {'pass': passes.get((fm, pi), False)} for rs in cp.ROW_SETS}
                       for fm in cp.FORMS_T1 for pi in cp.PIS},
                'T3': {'P4_%g' % pi: {'pass': passes.get(('P4', pi), False)} for pi in cp.PIS}}
    recs = [rec({('P1', 0.02): True, ('P1', 0.05): True, ('P1', 0.1): True}) for _ in range(4)] + [rec({})]
    assert cp.mde(recs, 'T1', 'P1') == 0.02
    assert cp.mde(recs, 'T1', 'P2') is None
    s = cp.summarise(recs)
    for rs in cp.ROW_SETS:
        assert s[rs]['reading']['T1_P1'] == 'informative null (MDE <= 2 %)'
        assert s[rs]['reading']['T1_P2'] == 'not measurable with this design'
        assert not s[rs]['instrument_faults']['T1_void']
