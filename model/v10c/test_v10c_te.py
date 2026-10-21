"""Tests for T-E (RESULTS 101 / 101.1)."""
import os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import te_tests as te  # noqa: E402


def test_signed_contrast_excludes_flagged_mates_by_hand():
    Y = np.array([[1.], [3.], [100.], [5.], [7.]])
    plate = np.array([0, 0, 0, 0, 1])
    pert = np.array(['a', 'b', 'x', 'c', 'a'])
    ok = np.array([True, True, False, True, True])            # 'x' is an excluded-class compound
    C, n = te.signed_contrast(Y, plate, pert, ok)
    assert n.tolist() == [2, 2, 3, 2, 0]
    assert C[0, 0] == pytest.approx(1 - 4) and C[1, 0] == pytest.approx(3 - 3) and C[3, 0] == pytest.approx(5 - 2)
    assert C[2, 0] == pytest.approx(100 - 3)                  # an excluded row is still contrasted, against allowed mates
    assert np.isnan(C[4, 0])                                  # alone on its plate


def test_partial_r_removes_a_confounder():
    r = np.random.default_rng(0)
    x = r.normal(size=2000)
    m, u = x + 0.3 * r.normal(size=2000), x + 0.3 * r.normal(size=2000)
    assert np.corrcoef(m, u)[0, 1] > 0.8 and abs(te.partial_r(m, u, x[:, None])) < 0.08


def test_lco_mean_by_hand():
    V = np.array([[1., 2.], [3., 4.], [5., 6.]])
    valid = np.array([[True, True], [True, False], [True, True]])
    L = te.lco_mean(V, valid)
    assert L[0].tolist() == [4.0, 6.0] and L[1].tolist() == [3.0, 4.0] and L[2].tolist() == [2.0, 2.0]


def test_decile_dummies_and_derangement():
    D = te.decile_dummies(np.arange(100.))
    assert D.shape == (100, 9) and (D.sum(1) <= 1).all() and D.sum() == 90
    p = te.derangement(7, np.random.default_rng(0))
    assert sorted(p) == list(range(7)) and not (p == np.arange(7)).any()


def _world(seed, beta, generic=0.0, C=14, G=300):
    r = np.random.default_rng(seed)
    gene_m = r.random(G)
    F = np.clip(gene_m[None, :] * 0.5 + 0.5 * r.random((C, G)), 0, 1)          # cell-specific + gene-generic mark
    mask = np.ones((C, G), bool)
    mask[:, :20] = False
    basal = r.random((C, G))
    gene_u = r.normal(size=G)
    U = gene_u[None, :] + 0.5 * basal + beta * (F - F.mean(1, keepdims=True)) * 3 + generic * gene_m[None, :] * 3 \
        + r.normal(scale=1.0, size=(C, G))
    U_ref = gene_u[None, :] + 0.5 * basal + r.normal(scale=1.0, size=(C, G))
    return F, mask, U, U_ref, basal, list(range(C))


def test_planted_cell_specific_effect_passes_and_noise_does_not():
    F, mask, U, U_ref, basal, cells = _world(0, beta=0.5)
    old = te.N_NULL
    te.N_NULL = 30
    try:
        rec = te.evaluate('planted', F, mask, U, U_ref, basal, cells, +1, 0.95)
        assert rec['passes'] and rec['mean_r'] > 0.1 and rec['ci'][0] > 0 and rec['delta_spec'] > 0
        F0, mask0, U0, U_ref0, basal0, cells0 = _world(1, beta=0.0)
        rec0 = te.evaluate('null', F0, mask0, U0, U_ref0, basal0, cells0, +1, 0.95)
        assert not rec0['passes']
    finally:
        te.N_NULL = old


def test_a_gene_generic_effect_does_not_pass():
    """101.1 C4(a): a response that follows the gene-level part of the mark, identical in every cell, must not pass."""
    F, mask, U, U_ref, basal, cells = _world(2, beta=0.0, generic=1.0)
    old = te.N_NULL
    te.N_NULL = 30
    try:
        rec = te.evaluate('generic', F, mask, U, U_ref, basal, cells, +1, 0.95)
        assert not rec['passes']
    finally:
        te.N_NULL = old


def test_masked_genes_never_enter_the_correlation():
    F, mask, U, U_ref, basal, cells = _world(3, beta=0.5)
    U2 = U.copy()
    U2[:, :20] = 1e6                                           # masked genes carry absurd values
    r1 = te.per_cell_r(F, mask, U, basal, cells)
    r2 = te.per_cell_r(F, mask, U2, basal, cells)
    assert all(abs(r1[k] - r2[k]) < 1e-9 for k in r1)


def test_negative_sign_hypotheses_use_the_lower_tail():
    F, mask, U, U_ref, basal, cells = _world(4, beta=-0.5)
    old = te.N_NULL
    te.N_NULL = 30
    try:
        rec = te.evaluate('neg', F, mask, U, U_ref, basal, cells, -1, 0.95)
        assert rec['passes'] and rec['mean_r'] < -0.1 and rec['ci'][1] < 0
        assert not te.evaluate('neg-as-pos', F, mask, U, U_ref, basal, cells, +1, 0.95)['passes']
    finally:
        te.N_NULL = old


def test_class_profiles_need_min_rows():
    C = np.array([[1.], [3.], [5.], [np.nan], [2.]])
    P, N = te.class_profiles(C, np.array([0, 0, 1, 1, 1]), 2, np.ones(5, bool), min_rows=2)
    assert N.tolist() == [2, 2] and P[0, 0] == pytest.approx(2.0) and P[1, 0] == pytest.approx(3.5)
    P3, _ = te.class_profiles(C, np.array([0, 0, 1, 1, 1]), 2, np.ones(5, bool), min_rows=3)
    assert np.isnan(P3).all()


def test_decide_needs_all_three_conditions():
    assert te.decide(+1, (0.01, 0.2), (0.01, 0.1), 0.1, 0.05)
    assert not te.decide(+1, (-0.01, 0.2), (0.01, 0.1), 0.1, 0.05)       # CI touches 0
    assert not te.decide(+1, (0.01, 0.2), (-0.01, 0.1), 0.1, 0.05)       # not specific
    assert not te.decide(+1, (0.01, 0.2), (0.01, 0.1), 0.1, 0.12)        # within the cell-swap null
    assert te.decide(-1, (-0.2, -0.01), (-0.1, -0.01), -0.1, -0.05)
    assert not te.decide(-1, (-0.2, -0.01), (-0.1, -0.01), -0.1, -0.12)
    assert not te.decide(-1, (0.01, 0.2), (0.01, 0.1), 0.1, 0.05)        # the wrong sign


def test_per_cell_r_by_hand_with_cell_specific_masks():
    F, mask, U, U_ref, basal, cells = _world(5, beta=0.5)
    mask = np.ones_like(mask)
    mask[0, 50:80] = False                                     # masked in cell 0 only: other cells still measure them
    U2 = U.copy()
    U2[0, 50:80] = 1e3
    r = te.per_cell_r(F, mask, U, basal, cells)
    assert te.per_cell_r(F, mask, U2, basal, cells)[0] == pytest.approx(r[0])
    ubar = te.lco_mean(U, np.isfinite(U))
    mbar = te.lco_mean(F, mask)
    g = mask[0]
    want = te.partial_r(F[0, g], U[0, g], np.column_stack([basal[0, g], ubar[0, g], mbar[0, g]]))
    assert r[0] == pytest.approx(want)


def test_decide_negative_direction_needs_specificity():
    assert not te.decide(-1, (-0.2, -0.01), (-0.1, 0.01), -0.1, -0.05)


def test_cell_swap_threshold_is_the_95th_percentile_in_the_predicted_direction():
    F, mask, U, U_ref, basal, cells = _world(6, beta=0.5)
    old = te.N_NULL
    te.N_NULL = 25
    try:
        for sign in (+1, -1):
            rec = te.evaluate('q', F, mask, U, U_ref, basal, cells, sign, 0.95)
            r = te.per_cell_r(F, mask, U, basal, cells)
            _, cs = te.nulls(F, mask, U, basal, list(r))
            assert rec['cell_swap']['p95_predicted_dir'] == pytest.approx(np.quantile(cs, 0.95 if sign > 0 else 0.05))
    finally:
        te.N_NULL = old
