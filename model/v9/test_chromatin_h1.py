# -*- coding: utf-8 -*-
"""Tests for chromatin_h1.py (RESULTS 93 H1). Synthetic worlds only (test_chromatin_funnel.world)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402
import chromatin_power as cp  # noqa: E402
import chromatin_h1 as h1  # noqa: E402
from test_chromatin_funnel import world  # noqa: E402


def make_fake_arrays(ctx, seed=42):
    rng = np.random.default_rng(seed)
    n = len(ctx.cells)
    # Give 5 dev cells has=True, and VCAP has=False
    # Dev cells: HEK293T, HL60, LNCAP, SKBR3, U937 covered; VCAP not covered
    has = np.array([c != 'VCAP' and c not in ('U00', 'U01') for c in ctx.cells], dtype=bool)
    return {
        'cells': np.array(ctx.cells),
        'genes': np.array(['g%03d' % i for i in range(cf.G)]),
        'F_prom': rng.normal(size=(n, cf.G)).astype(np.float32),
        'F_enh': rng.normal(size=(n, cf.G)).astype(np.float32),
        'F_reg': rng.normal(size=(n, cf.G)).astype(np.float32),
        'has': has,
    }


def test_install_c93():
    # 1. install_c93: rank-normal columns; cells without has are zeros; cov_dt holds dev-train cells only, cov_dev dev cells only.
    ctx = world(seed=0, n_perts=4)
    arrays = make_fake_arrays(ctx)
    h1.install_c93(ctx, arrays)

    c93 = ctx.enc['c93']
    assert 'b' in c93 and 'Ez' in c93 and 'has' in c93 and 'cov_dt' in c93 and 'cov_dev' in c93

    # Check rank-normal columns
    for i, c in enumerate(ctx.cells):
        if c != 'VCAP' and c not in ('U00', 'U01'):
            assert c93['has'][i].all()
            for k, key in enumerate(('F_prom', 'F_enh', 'F_reg')):
                expected_rn = cf.rank_normal(np.asarray(arrays[key][i], np.float64))
                np.testing.assert_allclose(c93['Ez'][i, :, k], expected_rn, atol=1e-5)
        else:
            # cells without has are zeros
            assert not c93['has'][i].any()
            np.testing.assert_array_equal(c93['Ez'][i], 0.0)

    # cov_dt holds dev-train cells only, cov_dev dev cells only
    assert not (set(c93['cov_dt']) & set(cf.DEV_CELLS))
    assert set(c93['cov_dev']).issubset(set(cf.DEV_CELLS))
    assert 'VCAP' not in c93['cov_dev']
    assert len(c93['cov_dev']) == 5
    assert len(c93['cov_dt']) == 12


def test_h1_builder():
    # 2. h1_builder: N1's means exclude h and non-fit cells; S at λ = 0 equals C and at λ = 1 equals N1; Cp has no P column.
    ctx = world(seed=0, n_perts=4)
    arrays = make_fake_arrays(ctx)
    h1.install_c93(ctx, arrays)

    c93 = ctx.enc['c93']
    fit_cells = list(c93['cov_dt'])

    # N1's means exclude h and non-fit cells
    build_n1 = h1.h1_builder(ctx, 'c93', 'N1', fit_cells)
    h0, h1_cell = fit_cells[0], fit_cells[1]
    f_h0 = build_n1(h0)
    f_h1 = build_n1(h1_cell)

    target_cell = fit_cells[2]
    # Check that h0 is excluded when h=h0
    cells_in_mean_h0 = [c for c in fit_cells if c != h0]
    expected_mean_E_h0 = np.mean([c93['Ez'][cf.cell_index(ctx, c), :, 1] for c in cells_in_mean_h0], axis=0)
    expected_mean_R_h0 = np.mean([c93['Ez'][cf.cell_index(ctx, c), :, 2] for c in cells_in_mean_h0], axis=0)
    np.testing.assert_allclose(f_h0[target_cell][:, 2], expected_mean_E_h0, atol=1e-5)
    np.testing.assert_allclose(f_h0[target_cell][:, 3], expected_mean_R_h0, atol=1e-5)

    # Mean changes when h changes (so h is truly excluded)
    assert not np.allclose(f_h0[target_cell][:, 2], f_h1[target_cell][:, 2])

    # Check non-fit cells (e.g. dev cells) do not enter N1's mean
    all_cells_mean_E = np.mean([c93['Ez'][cf.cell_index(ctx, c), :, 1] for c in ctx.cells if c93['has'][cf.cell_index(ctx, c), 0]], axis=0)
    assert not np.allclose(f_h0[target_cell][:, 2], all_cells_mean_E)

    # S at λ = 0 equals C and at λ = 1 equals N1
    build_c = h1.h1_builder(ctx, 'c93', 'C', fit_cells)
    build_s0 = h1.h1_builder(ctx, 'c93', 'S', fit_cells, lam=0.0)
    build_s1 = h1.h1_builder(ctx, 'c93', 'S', fit_cells, lam=1.0)

    f_c = build_c(None)
    f_s0 = build_s0(None)
    f_s1 = build_s1(None)
    f_n1_none = build_n1(None)

    for c in ctx.cells:
        np.testing.assert_allclose(f_s0[c], f_c[c], atol=1e-6)
        np.testing.assert_allclose(f_s1[c], f_n1_none[c], atol=1e-6)

    # Cp has no P column (shape [G, 3], columns b, E, R)
    build_cp = h1.h1_builder(ctx, 'c93', 'Cp', fit_cells)
    f_cp = build_cp(None)
    for c in ctx.cells:
        assert f_cp[c].shape == (cf.G, 3)
        i = cf.cell_index(ctx, c)
        np.testing.assert_array_equal(f_cp[c][:, 0], c93['b'][i])
        if c93['has'][i, 0]:
            np.testing.assert_array_equal(f_cp[c][:, 1], c93['Ez'][i, :, 1])
            np.testing.assert_array_equal(f_cp[c][:, 2], c93['Ez'][i, :, 2])
        else:
            np.testing.assert_array_equal(f_cp[c][:, 1:], 0.0)


def test_cell_rule_and_h1_reading():
    # 3. cell_rule (5 → 4, 4 → 3, 3 → None) and h1_reading counting only the given cells.
    # A VCAP-like unmeasured cell with a huge positive Δ must not count.
    assert h1.cell_rule(5) == 4
    assert h1.cell_rule(4) == 3
    assert h1.cell_rule(3) is None
    assert h1.cell_rule(2) is None
    with pytest.raises(AssertionError):
        h1.cell_rule(6)

    measured = ['HEK293T', 'HL60', 'LNCAP', 'SKBR3', 'U937']
    bars = cf.BARS['known']

    # Unmeasured VCAP has huge positive delta, but only 3 of 5 measured cells are positive
    sB = {
        'all': 0.10, 'top': 0.10, 'centred_all': 0.10,
        'per_cell': {c: 0.10 for c in cf.DEV_CELLS},
        'centred_per_cell': {c: 0.10 for c in cf.DEV_CELLS}
    }
    sC = {
        'all': 0.11, 'top': 0.12, 'centred_all': 0.11,
        'per_cell': {
            'HEK293T': 0.12, 'HL60': 0.12, 'LNCAP': 0.12,  # 3 positive
            'SKBR3': 0.08, 'U937': 0.08,                  # 2 negative
            'VCAP': 100.0                                 # unmeasured, huge positive
        },
        'centred_per_cell': {
            'HEK293T': 0.12, 'HL60': 0.12, 'LNCAP': 0.12,
            'SKBR3': 0.08, 'U937': 0.08,
            'VCAP': 100.0
        }
    }
    sN1 = {
        'all': 0.10, 'top': 0.10, 'centred_all': 0.10,
        'per_cell': {c: 0.10 for c in cf.DEV_CELLS},
        'centred_per_cell': {c: 0.10 for c in cf.DEV_CELLS}
    }

    rd = h1.h1_reading(sC, sB, sN1, bars, measured, k=4)
    # VCAP must NOT be in d['per_cell']
    assert 'VCAP' not in rd['delta']['per_cell']
    assert rd['delta']['cells_pos'] == 3
    # Needs >= 4 of measured cells, so pass must be False
    assert not rd['pass']
    assert not rd['conjuncts']['cells_delta_pos_ge_k']

    # Now make 4 of 5 positive:
    sC['per_cell']['SKBR3'] = 0.12
    sC['centred_per_cell']['SKBR3'] = 0.12
    rd4 = h1.h1_reading(sC, sB, sN1, bars, measured, k=4)
    assert rd4['delta']['cells_pos'] == 4
    assert rd4['conjuncts']['cells_delta_pos_ge_k']
    assert rd4['pass']


def test_rows_of_record_exclude_unmeasured_dev_cells():
    # 4. The rows of record exclude unmeasured dev cells: change y on an unmeasured dev cell's rows and the reading is unchanged.
    ctx = world(seed=1, n_perts=4)
    arrays = make_fake_arrays(ctx)
    h1.install_c93(ctx, arrays)

    res1 = h1.run_h1(ctx, ctx.y, 'c93', report_extras=False)

    # Change y on VCAP (unmeasured dev cell) rows
    y_perturbed = ctx.y.copy()
    vcap_rows = np.flatnonzero(ctx.rows['cell'] == 'VCAP')
    assert len(vcap_rows) > 0
    y_perturbed[vcap_rows] += 500.0

    res2 = h1.run_h1(ctx, y_perturbed, 'c93', report_extras=False)

    assert res1['reading']['pass'] == res2['reading']['pass']
    assert res1['reading']['delta']['all'] == res2['reading']['delta']['all']
    assert res1['reading']['delta']['top'] == res2['reading']['delta']['top']
    assert res1['reading']['delta']['centred_all'] == res2['reading']['delta']['centred_all']
    for c in res1['measured_dev_cells']:
        assert res1['reading']['delta']['per_cell'][c] == res2['reading']['delta']['per_cell'][c]


def test_calibration_permutations_and_planted_p1():
    # 5. Calibration: the c93_calib columns other than the slot are exact gene permutations of the real ones,
    # so no real column enters. A planted P1 at 5 % passes in the world; null does not.
    ctx = world(seed=2, n_perts=6, noise=1.0)
    arrays = make_fake_arrays(ctx)
    h1.install_c93(ctx, arrays)

    slot = 1
    # Run calibrate for 1 draw to inspect c93_calib and verify planted P1 passes
    records = h1.calibrate(ctx, slot=slot, draws=1)
    rec = records[0]

    # Check c93_calib columns
    calib = ctx.enc['c93_calib']
    c93 = ctx.enc['c93']
    for c in calib['cov_dt'] + calib['cov_dev']:
        i = cf.cell_index(ctx, c)
        # Column 0 (P) and Column 2 (R) must be exact permutations of the real ones
        for k in (0, 2):
            real_col = c93['Ez'][i, :, k]
            calib_col = calib['Ez'][i, :, k]
            assert np.array_equal(np.sort(real_col), np.sort(calib_col)), f"col {k} is not a permutation"
            assert not np.array_equal(real_col, calib_col), f"col {k} was not permuted"

    # In calibration records: null does not pass, P1 at 5% passes
    assert not rec['cases']['null']['pass']
    assert rec['cases']['P1_0.05']['pass']


def test_faults_and_mde_reads_three_of_three():
    # 6. faults_and_mde reads 3 of 3.
    def mk_case(passed, delta_all=0.01):
        return {'reading': {'pass': passed, 'delta': {'all': delta_all}}}

    base = {
        'null': mk_case(False),
        'P3_0.05': mk_case(False),
        'G_0.02': mk_case(False, delta_all=0.005),
        'G_0.05': mk_case(False, delta_all=0.005),
        'P1_0.005': mk_case(False),
        'P1_0.02': mk_case(True),
        'P1_0.05': mk_case(True),
        'P2_0.005': mk_case(False),
        'P2_0.02': mk_case(False),
        'P2_0.05': mk_case(True),
    }

    # 3 draws
    records = [{'cases': dict(base)} for _ in range(3)]
    # In draw 1, P2_0.05 fails -> passes in 2 of 3 draws only
    records[1]['cases']['P2_0.05'] = mk_case(False)

    f = h1.faults_and_mde(records)
    assert f['MDE']['P1'] == 0.02
    assert f['MDE']['P2'] is None  # 2 of 3 is not enough
    assert not f['VOID_null']
    assert not f['VOID_P3']
    assert not f['VOID_G']
    assert f['G_informative']  # 0.005 >= 0.004 in every draw for G_0.02 and G_0.05

    # If null passes in 1 draw -> VOID_null
    records[0]['cases']['null'] = mk_case(True)
    f2 = h1.faults_and_mde(records)
    assert f2['VOID_null']


# ---- PI additions (mutation check after W32): the N1 magnitude, the drug-known rows, the G-informative rule ----

def _score(per_cell, centred=None, all_=None, top=0.0):
    centred = per_cell if centred is None else centred
    return {'per_cell': dict(per_cell), 'centred_per_cell': dict(centred),
            'all': float(np.mean(list(per_cell.values()))) if all_ is None else all_, 'top': top,
            'centred_all': float(np.mean(list(centred.values())))}


def test_h1_reading_needs_the_N1_magnitude_not_only_the_cell_count():
    cells = ['HEK293T', 'HL60', 'LNCAP', 'U937']
    bars = cf.BARS['known']
    sB = _score({c: 0.40 for c in cells})
    sC = _score({c: 0.41 for c in cells})                           # +0.01 everywhere: the C - B conjuncts pass
    sN1_close = _score({c: 0.409 for c in cells})                   # C > N1 in 4 of 4, but S(C) - S(N1) = 0.001 < 0.002
    sN1_far = _score({c: 0.407 for c in cells})                     # 0.003 >= 0.002
    r = h1.h1_reading(sC, sB, sN1_close, bars, cells, 3)
    assert r['conjuncts']['cells_C_gt_N1_ge_k'] and not r['conjuncts']['vs_N1_all_ge_bar'] and not r['pass']
    assert h1.h1_reading(sC, sB, sN1_far, bars, cells, 3)['pass']


def test_rows_of_record_are_the_drug_known_rows_only(monkeypatch):
    """A perfect C prediction on drug-unknown (level 3) rows of a measured cell must not change the reading."""
    ctx = world(seed=0, n_perts=4)
    h1.install_c93(ctx, make_fake_arrays(ctx))
    dev = np.flatnonzero(ctx.dev_mask)
    rng = np.random.default_rng(0)
    base = {k: rng.normal(size=(len(dev), ctx.y.shape[1])).astype(np.float32) for k in ('B', 'C', 'N1')}
    level = np.full(len(ctx.y), 2)
    measured = np.isin(ctx.rows['cell'][dev], ctx.enc['c93']['cov_dev'])
    unknown = measured & (np.arange(len(dev)) % 3 == 0)
    level[dev[unknown]] = 3

    def fake_run_t1(ctx_, y, specs, fit):
        return {(k, 'full'): {'y_hat_dev': base[k], 'loco': 0.0} for k in specs}, None, level
    monkeypatch.setattr(cf, 'run_t1', fake_run_t1)
    r1 = h1.run_h1(ctx, ctx.y, 'c93', report_extras=False)
    base['C'] = base['C'].copy()
    base['C'][unknown] = ctx.y[dev[unknown]]                       # C is perfect on the level-3 rows only
    r2 = h1.run_h1(ctx, ctx.y, 'c93', report_extras=False)
    assert r1['delta']['all'] == r2['delta']['all'] and r1['delta']['per_cell'] == r2['delta']['per_cell']


def test_G_is_informative_only_if_its_raw_delta_clears_the_bar_in_every_draw():
    def mk(passed, d=0.01):
        return {'reading': {'pass': passed, 'delta': {'all': d}}}
    recs = []
    for i in range(3):
        g = 0.005 if i < 2 else 0.001                                # the third draw falls below 0.004
        recs.append({'cases': {'null': mk(False), 'P3_0.05': mk(False), 'G_0.02': mk(False, g), 'G_0.05': mk(False, g),
                               **{'%s_%g' % (f, pi): mk(True) for f in ('P1', 'P2') for pi in h1.PIS}}})
    assert not h1.faults_and_mde(recs)['G_informative']
    recs[2]['cases']['G_0.05'] = mk(False, 0.006)
    assert h1.faults_and_mde(recs)['G_informative']
