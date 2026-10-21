"""Tests for T-E2 (RESULTS 101.4)."""
import os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import te2_tests as t2  # noqa: E402


def S(mean, lo, hi):
    return {'mean_r': mean, 'ci': (lo, hi)}


def test_a1_rule_all_three_readings():
    act = S(-0.07, -0.09, -0.05)
    assert t2.a1_rule(S(-0.05, -0.07, -0.03), act, S(-0.04, -0.06, -0.02)) == 'beyond two expression measurements'
    assert t2.a1_rule(S(-0.02, -0.04, -0.01), act, S(-0.04, -0.06, -0.02)) == 'inconclusive'      # < half of ACT-only
    assert t2.a1_rule(S(-0.01, -0.03, +0.01), act, S(-0.04, -0.06, -0.02)) == 'explained by expression'
    assert t2.a1_rule(S(-0.01, -0.03, +0.01), act, S(-0.01, -0.03, +0.01)) == 'inconclusive'      # neither explains
    assert t2.a1_rule(S(-0.035, -0.05, -0.02), act, S(0.0, -0.01, 0.01)) == 'beyond two expression measurements'


def test_rank_rows_nan_keeps_nan_and_ranks_finite():
    A = np.array([[3., np.nan, 1., 2.], [np.nan, np.nan, np.nan, 5.]])
    R = t2.rank_rows_nan(A)
    assert np.allclose(R[0, [0, 2, 3]], [1.0, 0.0, 0.5]) and np.isnan(R[0, 1]) and np.isnan(R[1]).all()


def test_summary_and_paired():
    s = t2.summary({0: -0.1, 1: -0.2, 2: 0.05})
    assert s['n_cells'] == 3 and s['n_negative'] == 2 and s['mean_r'] == pytest.approx(-0.25 / 3)
    assert 'note' in t2.summary({0: -0.1})
    p = t2.paired({0: -0.1, 1: -0.2, 2: -0.3, 5: 1.0}, {0: 0.0, 1: 0.0, 2: 0.0})
    assert p['n_cells'] == 3 and p['mean_diff'] == pytest.approx(-0.2) and p['ci'][1] < 0


def test_a2_split_half_survives_nan_rows():
    """Review 096 C1: a cell without two distinct controls (NaN b2) must not poison the leave-cell-out mean."""
    r = np.random.default_rng(0)
    C, G = 8, 120
    truth = r.random((C, G))
    b1, b2 = truth + 0.3 * r.normal(size=(C, G)), truth + 0.3 * r.normal(size=(C, G))
    b2[7] = np.nan
    b1[7] = np.nan
    U = r.normal(size=(C, G)) + 0.5 * b2
    out, rho, ratio = t2.a2_split_half(b1, b2, U, list(range(7)))
    assert len(out) == 7 and all(np.isfinite(v) for v in out.values())
    assert set(rho) == set(out) and set(ratio) == set(out)
    assert all(abs(ratio[k] - 1 / (1 + rho[k])) < 1e-12 and 0.5 < ratio[k] < 1 for k in ratio)


def test_a4_reading_is_mechanical():
    bet = {'mean_r': -0.07, 'ci': (-0.09, -0.05)}
    assert t2.a4_reading(bet, {'mean_r': -0.06}) == 'not BET-specific'
    assert t2.a4_reading(bet, {'mean_r': 0.0}) == 'differs from BET'
    assert t2.a4_reading({'mean_r': -0.07}, {'mean_r': 0.0}).startswith('inconclusive')
