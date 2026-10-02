# -*- coding: utf-8 -*-
"""Tests for chromatin_power.py (RESULTS 91.9 M2). Synthetic worlds only; every test calls the module's own functions."""
import os
import sys

import numpy as np
import pytest

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


def test_a_large_drug_independent_shift_is_stopped_by_the_centred_magnitude():
    ctx = world(n_perts=40, noise=0.3)
    fit = ctx.enc['rank_normal']['cov_dt']
    f, _ = cp.synthetic_feature(ctx, draw=0)
    B = cp.builders(ctx, f, fit)
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    yp, _ = cp.plant(ctx, ctx.y, mu_e, f, 'P3', 0.10, draw=0)
    r = cp.t1_case(ctx, yp, B, fit)
    for rs in ('all', 'known'):
        assert r[rs]['delta_all'] > 0              # the raw score rewards the cell-constant offset ...
        # ... the centred MAGNITUDE does not (review 041 C1: the sign count alone can pass on numerical dust)
        assert not r[rs]['conjuncts']['centred_all_ge_bar'] and not r[rs]['pass']


def test_mde_and_summary_logic():
    def rec(passes):
        nul = {'pass': False, 'resid': {'all': 0.0}}
        return {'null_T1': {'all': nul, 'known': nul}, 'null_T3': {'pass': False},
                'T1': {'%s_%g' % (fm, pi): {rs: {'pass': passes.get((fm, pi), False)} for rs in cp.ROW_SETS}
                       for fm in cp.FORMS_T1 for pi in cp.PIS},
                'T3': {'P4_%g' % pi: {'pass': passes.get(('P4', pi), False)} for pi in cp.PIS}}
    recs = [rec({('P1', 0.02): True, ('P1', 0.05): True, ('P1', 0.1): True}) for _ in range(4)] + [rec({})]
    assert cp.mde(recs, 'T1', 'P1') == 0.02
    assert cp.mde(recs, 'T1', 'P2') is None
    s = cp.summarise(recs)
    for rs in cp.ROW_SETS:
        assert s[rs]['reading']['T1_P1'] == 'informative null (MDE <= 2 %; linear in the encoded tracks)'
        assert s[rs]['reading']['T1_P2'] == 'not measurable with this design'
        assert not s[rs]['instrument_faults']['T1_void']


def test_calibration_tracks_have_the_real_tests_shape():
    """Review 041 C2: FBC* = real 4-column set, primary mark -> f*, other marks gene-permuted copies, availability unchanged."""
    ctx = world(n_perts=4)
    e = ctx.enc['rank_normal']
    f, _ = cp.synthetic_feature(ctx, draw=2)
    cal = cp.install_calibration_tracks(ctx, f, draw=2)
    assert np.array_equal(cal['has'], e['has'])
    for i, c in enumerate(ctx.cells):
        if c not in f:
            assert not cal['Ez'][i].any()
            continue
        k0 = 1 if e['has'][i, 1] else (0 if e['has'][i, 0] else 2)
        np.testing.assert_allclose(cal['Ez'][i, :, k0], f[c], rtol=1e-5)
        for k in range(3):
            if k != k0 and e['has'][i, k]:
                assert np.allclose(np.sort(cal['Ez'][i, :, k]), np.sort(e['Ez'][i, :, k]))       # same values ...
                assert abs(np.corrcoef(cal['Ez'][i, :, k], e['Ez'][i, :, k])[0, 1]) < 0.15     # ... gene link destroyed
            if not e['has'][i, k]:
                assert not cal['Ez'][i, :, k].any()
    B = cp.builders(ctx, f, e['cov_dt'], draw=2)
    assert B['FBC'](None)[e['cov_dt'][0]].shape == (cf.G, 4)


def test_p3_fails_t1_across_ten_world_and_pi_cases():
    """Review 041 C1: a drug-independent shift must not pass T1's full rule on either row set (10 world x pi cases)."""
    n_pass = 0
    for seed in range(5):
        ctx = world(seed=seed, n_perts=30, noise=0.3)
        fit = ctx.enc['rank_normal']['cov_dt']
        f, _ = cp.synthetic_feature(ctx, draw=seed)
        B = cp.builders(ctx, f, fit, draw=seed)
        mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
        for pi in (0.05, 0.10):
            yp, _ = cp.plant(ctx, ctx.y, mu_e, f, 'P3', pi, draw=seed)
            r = cp.t1_case(ctx, yp, B, fit)
            n_pass += int(r['all']['pass']) + int(r['known']['pass'])
    assert n_pass == 0


def test_the_reader_refuses_without_the_marker_or_on_a_hash_mismatch(tmp_path):
    import json
    import read_chromatin91 as rd
    with pytest.raises(SystemExit, match='COMPLETE'):
        rd.read(str(tmp_path))
    for f in ('chromatin_funnel_91.json', 'chromatin_power_91.json'):
        (tmp_path / f).write_text('{}')
    json.dump({'outputs': {'chromatin_funnel_91.json': 'deadbeef', 'chromatin_power_91.json': 'deadbeef'}},
              open(tmp_path / 'CHROMATIN91_COMPLETE.json', 'w'))
    with pytest.raises(SystemExit, match='not the file'):
        rd.read(str(tmp_path))


def test_unplanted_world_gives_no_t3_pass_and_the_reader_voids_t3_on_one():
    """Review 042 C3: T3's pi = 0 fault check runs in the calibration, and a pass there voids T3's reading."""
    ctx = world(n_perts=30, noise=0.3)
    fit = ctx.enc['rank_normal']['cov_dt']
    f, _ = cp.synthetic_feature(ctx, draw=0)
    B = cp.builders(ctx, f, fit, draw=0)
    assert not cp.t3_case(ctx, ctx.y, B, fit)['pass']
    def rec(t3_null):
        nul = {'pass': False, 'resid': {'all': 0.0}}
        return {'null_T1': {'all': nul, 'known': nul}, 'null_T3': {'pass': t3_null},
                'T1': {'%s_%g' % (fm, pi): {rs: {'pass': False} for rs in cp.ROW_SETS} for fm in cp.FORMS_T1 for pi in cp.PIS},
                'T3': {'P4_%g' % pi: {'pass': False} for pi in cp.PIS}}
    s = cp.summarise([rec(False)] * 4 + [rec(True)])
    assert s['known']['instrument_faults']['T3_void'] and not s['known']['instrument_faults']['T1_void']
