"""Tests for stage1.py (RESULTS 103 / 103.1 / 103.2), each on synthetic data where the answer is known."""
import os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stage1 as s1  # noqa: E402


def synthetic(n_genes=80, cells=s1.CORE + ('HA1E', 'HCC515'), J=50, cell_signal=0.0, seed=0):
    """y(g, c) = gene effect + cell_signal * (gene x cell effect) + noise; one signature per (gene, cell)."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n_genes, J))
    rows, gl, cl = [], [], []
    for gi in range(n_genes):
        for c in cells:
            rows.append(G[gi] + cell_signal * rng.normal(size=J) + rng.normal(size=J))
            gl.append(f'g{gi}')
            cl.append(c)
    Y = np.array(rows)
    idx = s1.index_signatures(np.array(cl), np.array(gl), np.ones(len(rows), bool))
    return Y, idx


def test_row_r_skips_the_own_coordinate_and_is_zero_for_constant_predictions():
    t = np.array([1.0, 2.0, 3.0, -50.0])
    assert s1.row_r([1, 2, 3, 0], t, skip=3) == pytest.approx(1.0)
    assert s1.row_r(np.zeros(4), t) == 0.0


def test_index_signatures_refuses_two_signatures_for_one_pair():
    with pytest.raises(ValueError):
        s1.index_signatures(np.array(['A', 'A']), np.array(['g', 'g']), np.array([True, True]))


def test_the_leak_test_detects_the_103_leak_and_passes_the_fix():
    Y, idx = synthetic()
    skip = lambda g: None
    for c0 in s1.CORE:
        tr, te, _ = s1.fold_targets(Y, idx, c0)
        trl, tel, _ = s1.fold_targets(Y, idx, c0, leaky=True)
        assert abs(s1.per_gene_model_scores(tr, te, skip)) < 0.02
        assert s1.per_gene_model_scores(trl, tel, skip) < -0.99


def test_no_training_target_contains_the_held_out_signature():
    Y, idx = synthetic(n_genes=10)
    c0 = 'PC3'
    tr, te, _ = s1.fold_targets(Y, idx, c0)
    Y2 = Y.copy()
    for (g, c), i in idx.items():
        if c == c0:
            Y2[i] += 100.0                       # change only the held-out signatures
    tr2, _, _ = s1.fold_targets(Y2, idx, c0)
    assert all(np.allclose(a[2], b[2]) for a, b in zip(tr, tr2))


def test_own_percentile_direction():
    y = np.array([0.0, -9.0, 1.0, 2.0])
    assert s1.own_percentile(y, 1) == 0.0
    assert s1.own_percentile(y, 3, ascending=False) == 0.0
    assert s1.own_percentile(y, 3) == 1.0


def test_two_way_shares_recover_known_components():
    rng = np.random.default_rng(1)
    gene_only = rng.normal(size=(200, 1)) + np.zeros((200, 6))
    assert s1.two_way_shares(gene_only)['gene'] == pytest.approx(1.0)
    inter = rng.normal(size=(200, 6))
    inter = inter - inter.mean(1, keepdims=True) - inter.mean(0, keepdims=True) + inter.mean()
    assert s1.two_way_shares(inter)['gene_x_cell'] == pytest.approx(1.0)


def test_half_references_are_disjoint_and_alternate():
    A, B = s1.half_references(['NPC', 'A549', 'HA1E', 'A375', 'PC3'])
    assert A == ['A375', 'HA1E', 'PC3'] and B == ['A549', 'NPC'] and not set(A) & set(B)


def test_partial_spearman_removes_a_shared_confound():
    rng = np.random.default_rng(2)
    z = rng.normal(size=2000)
    x, y = z + 0.3 * rng.normal(size=2000), z + 0.3 * rng.normal(size=2000)
    assert s1.partial_spearman(x, y) > 0.8 and abs(s1.partial_spearman(x, y, z)) < 0.1


def test_mean_within_cell_ignores_a_cell_level_shift():
    rng = np.random.default_rng(3)
    R, F = rng.normal(size=(6, 300)), rng.normal(size=(6, 300))
    shift = np.arange(6)[:, None] * 5.0
    assert s1.mean_within_cell(R, F)[0] == pytest.approx(s1.mean_within_cell(R + shift, F + shift)[0])


def test_dose_parsing_and_the_dose_time_rule():
    assert s1.dose_um('10 uM') == 10 and s1.dose_um('10.0 um') == 10 and s1.dose_um('500 nM') == 0.5
    assert s1.dose_um('10 µM') == 10 and s1.dose_um('10 �M') == 10 and s1.dose_um('-666') is None
    rows = [(0, '24 h', 1.0), (1, '24 h', 10.0), (2, '24 h', 10.0), (3, '6 h', 20.0)]
    assert s1.choose_rows(rows) == [1, 2]
    assert s1.choose_rows([(5, '6 h', 1.0), (6, '6 h', 3.0)]) == [6]


def test_percentile_zero_is_the_most_correlated_and_skip_applies():
    rng = np.random.default_rng(4)
    v = rng.normal(size=60)
    M = rng.normal(size=(30, 60))
    M[7] = v + 0.01 * rng.normal(size=60)
    cr = s1.corr_rows(v, M)
    assert s1.percentile_of(cr, 7) == 0.0
    M2 = M.copy()
    M2[7, 3] = 1e6                               # a huge own-coordinate dip / spike is skipped
    assert s1.percentile_of(s1.corr_rows(v, M2, skip=3), 7) == 0.0


def test_cluster_bootstrap_is_reproducible():
    v, cl = np.linspace(0, 1, 40), np.repeat(np.arange(10), 4)
    assert s1.cluster_bootstrap_median(v, cl, 500) == s1.cluster_bootstrap_median(v, cl, 500)


def test_corr_rows_own_matches_a_direct_computation():
    rng = np.random.default_rng(5)
    v, M = rng.normal(size=40), rng.normal(size=(12, 40))
    own = np.array([-1, 3, 0, 39, -1, 7, 7, -1, 12, 1, 2, 5])
    got = s1.corr_rows_own(v, M, own)
    for k in range(12):
        keep = np.ones(40, bool)
        if own[k] >= 0:
            keep[own[k]] = False
        assert got[k] == pytest.approx(np.corrcoef(v[keep], M[k, keep])[0, 1])


def test_own_exclusion_removes_the_dip_bias_the_one_set_ranking_has():
    """Review 108 C2: with a shared generic component, competitors that keep a deep own-gene dip rank too low."""
    rng = np.random.default_rng(6)
    J, n = 200, 400
    gen = rng.normal(size=J)
    own = np.full(n, -1)
    own[:120] = rng.integers(0, J, 120)                       # 30 % landmark knockdowns with their own dip
    M = 0.5 * gen + rng.normal(size=(n, J))
    M[np.arange(120), own[:120]] -= 15.0
    fixed, old = [], []
    for trial in range(60):
        v = 0.5 * gen + rng.normal(size=J)
        t = int(rng.integers(120, n))                         # a target with no dip and no specific relation
        fixed.append(s1.percentile_of(s1.corr_rows_own(v, M, own), t))
        old.append(s1.percentile_of(s1.corr_rows(v, M), t))
    assert abs(np.mean(fixed) - 0.5) < 0.06 and np.mean(old) < np.mean(fixed) - 0.05


def test_d1_gate_follows_the_primary_version():
    rng = np.random.default_rng(7)
    J = 300
    E = rng.normal(size=(6, J))
    feats = {'ACT:prom_1kb': (rng.normal(size=(6, J)), np.ones(6, bool))}
    R_cons = E + 0.3 * rng.normal(size=(6, J))               # its positive control passes
    R_prim = rng.normal(size=(6, J))                          # its positive control fails
    p = s1.d1_analysis(R_prim, R_prim, E, feats, n_boot=200)
    c = s1.d1_analysis(R_cons, R_cons, E, feats, n_boot=200)
    assert p['positive_control']['passes'] is False and 'ACT:prom_1kb' not in p and p['verdict'].startswith('not interpreted')
    assert c['positive_control']['passes'] is True and 'passes' in c['ACT:prom_1kb'] and 'reported' in c


def test_partial_spearman_with_two_covariates_equals_the_precision_matrix_formula():
    """The rank partial with several covariates is the partial correlation of the ranked variables."""
    rng = np.random.default_rng(8)
    z1, z2 = rng.normal(size=3000), rng.normal(size=3000)
    x, y = z1 + z2 + rng.normal(size=3000), z1 - z2 + rng.normal(size=3000)
    R = np.column_stack([s1.rank(v) for v in (x, y, z1, z2)])
    P = np.linalg.inv(np.corrcoef(R, rowvar=False))
    want = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    assert s1.partial_spearman(x, y, np.stack([z1, z2], 1)) == pytest.approx(want, abs=1e-9)
