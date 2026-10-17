"""Tests for T-A (RESULTS 100.6)."""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ta_gate as ta  # noqa: E402


def test_responsiveness_by_hand():
    Y = np.array([[1., 0.], [3., 0.], [5., 0.], [2., 2.]])
    plate = np.array([0, 0, 0, 1])
    pert = np.array(['a', 'b', 'c', 'a'])
    cell = np.array(['X', 'X', 'Y', 'Y'])
    cells, R, n = ta.responsiveness(Y, plate, pert, cell)
    # row 0: |1 - 4| = 3; row 1: |3 - 3| = 0; row 2: |5 - 2| = 3; row 3 alone on its plate: excluded
    assert list(cells) == ['X', 'Y'] and list(n) == [2, 1]
    assert np.allclose(R, [[1.5, 0.0], [3.0, 0.0]])


def _world(seed, beta, C=40, G=120):
    r = np.random.default_rng(seed)
    gene = r.normal(size=G)
    basal = r.random((C, G))
    chrom = r.random((C, G, 4))
    R = gene[None, :] + 0.3 * basal + beta * (chrom[..., 0] - 0.5) * 3 + r.normal(scale=1.0, size=(C, G))
    return R, basal, chrom, np.ones(C, bool)


def test_planted_effect_detected_and_noise_not():
    R, basal, chrom, has = _world(0, beta=0.5)
    rec = ta.run(R, basal, chrom, has, n_null=30, betas=(0.1,), n_null_cal=10)
    assert rec['passes'] and rec['gain'] > rec['cell_swap_null']['p95']
    R0, basal0, chrom0, has0 = _world(1, beta=0.0)
    rec0 = ta.run(R0, basal0, chrom0, has0, n_null=30, betas=(0.1,), n_null_cal=10)
    assert not rec0['passes']


def test_calibration_detects_a_large_plant():
    R, basal, chrom, has = _world(2, beta=0.0)
    rec = ta.run(R, basal, chrom, has, n_null=20, betas=(0.0, 1.0), n_null_cal=20)
    assert rec['calibration']['1.0']['detected'] and not rec['calibration']['0.0']['detected']


def test_derangement_moves_every_cell():
    p = ta.derangement(10, np.random.default_rng(0))
    assert sorted(p) == list(range(10)) and not (p == np.arange(10)).any()
