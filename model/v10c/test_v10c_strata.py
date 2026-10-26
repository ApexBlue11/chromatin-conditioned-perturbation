"""Tests for strata.py (RESULTS 100.13 / 101.1 C3)."""
import os, sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata  # noqa: E402


def test_group_stats_by_hand():
    Ra = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    Rb = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    g = strata.group_stats(Ra, Rb, np.array([True, True]))
    assert g['slot_means_a'] == [2.0, 3.0, 4.0] and g['slot_means_b'] == [0.0, 1.0, 2.0]
    assert np.isclose(g['delta'], 2.0) and np.isclose(g['se'], np.sqrt(1 / 3 + 1 / 3)) and g['n_rows'] == 2
    assert np.isclose(g['welch_df'], 4.0) and not g['all_a_above_all_b'] and not g['all_b_above_all_a']   # 2.0 ties
    g2 = strata.group_stats(Ra + 0.5, Rb, np.array([True, True]))
    assert g2['all_a_above_all_b'] and not g2['all_b_above_all_a']
    from scipy import stats
    assert np.isclose(g['p_t_two_sided'], 2 * stats.t.sf(2.0 / np.sqrt(2 / 3), 4.0))


def test_finite_rows_refuses_a_mask_that_differs_between_slots():
    Ra = np.ones((3, 4))
    Rb = np.ones((3, 4))
    Ra[:, 1] = np.nan
    assert strata.finite_rows(Ra, Rb).tolist() == [True, False, True, True]
    Rb[2, 3] = np.nan
    with pytest.raises(ValueError):
        strata.finite_rows(Ra, Rb)


def test_cluster_ci_collapses_for_a_constant_and_resamples_compounds():
    d = np.full(30, 0.2)
    comp = np.repeat(np.arange(10), 3)
    assert np.allclose(strata.cluster_ci(d, comp, 500), [0.2, 0.2])
    d2 = np.r_[np.full(15, 1.0), np.zeros(15)]
    lo, hi = strata.cluster_ci(d2, np.repeat(np.arange(2), 15), 2000)
    assert lo == 0.0 and hi == 1.0                                  # two clusters: the CI spans both


def test_fit_variance_recovers_a_and_b():
    rng = np.random.default_rng(0)
    n = rng.integers(10, 200, 4000).astype(float)
    true = 0.001 + 0.05 / n
    v = true * rng.chisquare(4, len(n)) / 4                          # sample variances on 4 df
    a, b = strata.fit_variance(v, n)
    assert abs(a - 0.001) < 0.0003 and abs(b - 0.05) < 0.01
    a0, b0 = strata.fit_variance(np.full(50, 0.01), np.linspace(10, 100, 50))
    assert np.isclose(a0, 0.01) and abs(b0) < 1e-9 and a0 >= 0 and b0 >= 0


def test_bh_by_hand():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.059, 0.074, 0.205, 0.212, 0.216])
    assert strata.bh(p, 0.05).tolist() == [True, True] + [False] * 8
    assert strata.bh(p, 0.10).tolist() == [True] * 6 + [False] * 4    # 0.059 <= 0.06; 0.074 > 0.07
    assert not strata.bh(np.array([0.5, 0.9]), 0.1).any()


def test_by_compound_finds_a_planted_compound_and_calibrates_by_relabelling():
    rng = np.random.default_rng(1)
    comp = np.repeat(np.arange(60), 20)
    Ra = rng.normal(0.1, 0.2, (3, len(comp)))
    Rb = rng.normal(0.1, 0.2, (3, len(comp)))
    Ra[:, comp == 7] += 0.4
    out = strata.by_compound(Ra, Rb, comp.astype(str), labels={'7': {'class': 'kinase'}})
    assert out['n_compounds_tested'] == 60 and len(out['relabelling_calibration']) == 9
    assert [d['compound'] for d in out['discoveries']] == ['7'] and out['discoveries'][0]['class'] == 'kinase'
    splits = {tuple(c['a_slots']) for c in out['relabelling_calibration']}
    assert (0, 1, 2) not in splits and len(splits) == 9 and all(0 in s for s in splits)
    small = strata.by_compound(Ra[:, :50], Rb[:, :50], comp[:50].astype(str))
    assert small['n_compounds_tested'] == 3                          # compounds 0, 1 (20 rows) and 2 (10 rows)


def test_by_class_reports_the_cluster_ci_and_counts():
    rng = np.random.default_rng(2)
    comp = np.repeat(np.arange(12), 10).astype(str)
    cls = np.where(np.repeat(np.arange(12), 10) < 6, 'kinase', 'other')
    Ra, Rb = rng.normal(0, 0.1, (3, 120)), rng.normal(0, 0.1, (3, 120))
    out = strata.by_class(Ra, Rb, cls, comp, 1000)
    assert set(out) == {'kinase', 'other'} and out['kinase']['n_compounds'] == 6 and out['kinase']['n_rows'] == 60
    lo, hi = out['kinase']['ci95_compound_cluster']
    assert lo <= out['kinase']['delta'] <= hi
