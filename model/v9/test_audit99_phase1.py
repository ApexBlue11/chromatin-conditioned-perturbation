# -*- coding: utf-8 -*-
"""Tests for audit99_phase1.py (RESULTS 99, packet 084). Synthetic worlds only: no real prediction or test row is read."""
import json, os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit99_phase1 as A
import score_p9


def _world(n_plates=40, per_plate=6, G=50, seed=0, plate_sd=2.0, drug_sd=0.5, noise_sd=0.3):
    """Plates with several compounds each. truth = plate effect + drug effect + noise."""
    rng = np.random.default_rng(seed)
    n_drugs = 60
    drug_eff = rng.normal(0, drug_sd, (n_drugs, G))
    plate, pert, Y, mol = [], [], [], []
    for p in range(n_plates):
        pe = rng.normal(0, plate_sd, G)
        drugs = rng.choice(n_drugs, per_plate, replace=False)
        for d in drugs:
            plate.append(p); pert.append('D%02d' % d); mol.append('D%02d' % d)
            Y.append(pe + drug_eff[d] + rng.normal(0, noise_sd, G))
    plate = np.array(plate); pert = np.array(pert); Y = np.array(Y)
    plate_eff = np.stack([Y[plate == p].mean(0) for p in plate])
    return {'plate': plate, 'pert': pert, 'mol': np.array(mol), 'y': Y, 'drug_eff': drug_eff, 'plate_mean': plate_eff, 'rng': rng}


def test_plate_contrast_hand_example():
    plate = np.array([0, 0, 0, 0, 1, 1])
    pert = np.array(['a', 'a', 'b', 'c', 'a', 'b'])
    V = np.array([[1.], [2.], [4.], [7.], [10.], [20.]])
    C, cnt = A.plate_contrast(V, plate, pert)
    # row 0 (a): mates b, c -> mean 5.5 ; row 2 (b): mates a, a, c -> mean 10/3 ; plate 1 rows have 1 mate -> NaN
    assert cnt.tolist() == [2, 2, 3, 3, 1, 1]
    assert C[0, 0] == pytest.approx(1 - 5.5) and C[1, 0] == pytest.approx(2 - 5.5)
    assert C[2, 0] == pytest.approx(4 - 10 / 3) and C[3, 0] == pytest.approx(7 - 7 / 3)
    assert np.isnan(C[4, 0]) and np.isnan(C[5, 0])


def test_other_mean_respects_selection():
    plate = np.array([0, 0, 0, 0, 0])
    pert = np.array(['a', 'b', 'c', 'd', 'e'])
    V = np.arange(5, dtype=float)[:, None]
    sel = np.array([False, True, True, False, True])
    M, cnt = A.other_mean(V, plate, pert, sel, np.array([0, 3]))
    assert cnt.tolist() == [3, 3]
    assert M[0, 0] == pytest.approx((1 + 2 + 4) / 3) and M[1, 0] == pytest.approx((1 + 2 + 4) / 3)
    M2, cnt2 = A.other_mean(V, plate, pert, sel, np.array([1]))
    assert cnt2.tolist() == [2] and M2[0, 0] == pytest.approx(3.0)


def test_plate_only_model_scores_zero_even_with_rounding():
    w = _world()
    cy, cnt = A.plate_contrast(w['y'], w['plate'], w['pert'])
    have = cnt >= 2
    pred = w['plate_mean'] + 1e-9 * w['rng'].normal(size=w['y'].shape)     # same response for every compound, + rounding
    cp, _ = A.plate_contrast(pred, w['plate'], w['pert'])
    r, z = A.rowr_pred(cp[have], cy[have])
    assert z == have.sum() and np.all(r == 0)


def test_truth_scores_one_and_plate_noise_is_removed():
    w = _world()
    cy, cnt = A.plate_contrast(w['y'], w['plate'], w['pert'])
    have = cnt >= 2
    shifted = w['y'] + np.stack([np.full(w['y'].shape[1], 3.0 * p) for p in w['plate']])   # a plate-level error
    cp, _ = A.plate_contrast(shifted, w['plate'], w['pert'])
    r, z = A.rowr_pred(cp[have], cy[have])
    assert z == 0 and np.allclose(r, 1.0)


def test_rowr_pred_zero_and_nan_rules():
    t = np.array([[1., 2., 3.], [1., 1., 1.]])
    p = np.array([[5., 5., 5.], [1., 2., 3.]])
    r, z = A.rowr_pred(p, t)
    assert r[0] == 0.0 and np.isnan(r[1]) and z == 1
    r2, z2 = A.rowr_pred(np.array([[1., 2., 3.]]), np.array([[2., 4., 7.]]))
    assert z2 == 0 and 0.9 < r2[0] <= 1


def test_plates_for_refuses_a_different_control():
    b = {'X_ctl': np.array([[1., 2.], [1., 2.], [3., 4.]], np.float32)}
    ids = A._plates_for(b, np.array([0, 1, 2]), b['X_ctl'])
    assert ids[0] == ids[1] != ids[2]
    with pytest.raises(SystemExit, match='differs'):
        A._plates_for(b, np.array([0, 1, 2]), b['X_ctl'] + np.array([[0, 0], [0, 0], [0, 1e-3]]))


def test_every_verdict_string_has_a_statement():
    base = {'ci_width': 0.02, 'mean_d': 0.05, 'ci_lo': 0.01, 'ci_hi': 0.09, 'frac_favour_ours': 0.8,
            'frac_favour_ref': 0.2, 'sign_p': 1e-6}
    worlds = [base, dict(base, mean_d=-0.05, ci_lo=-0.09, ci_hi=-0.01, frac_favour_ours=0.2, frac_favour_ref=0.8),
              dict(base, ci_lo=-0.01), dict(base, ci_width=0.2)]
    seen = {score_p9.verdict(r, 'M+dt') for r in worlds}
    assert seen == set(A.C1_STATEMENTS)


def _patch_p9(monkeypatch, w, v9_list, mdt_list, m_list, clean=None):
    n = len(w['y'])
    p9 = {'y': w['y'], 'pert': w['pert'], 'molecule': w['mol'], 'clean_mask': np.ones(n, bool) if clean is None else clean,
          'seeds_data': [{'y_pred': v} for v in v9_list], 'ctl_true': np.zeros_like(w['y'])}
    monkeypatch.setattr(A, '_load_p9_side', lambda *a, **k: (p9, {'pred_deltas': mdt_list}, {'pred_deltas': m_list},
                                                             w['plate'], [v - p9['ctl_true'] for v in v9_list]))


def test_c1_detects_a_model_that_uses_the_compound(monkeypatch):
    w = _world()
    rng = w['rng']
    good = [w['plate_mean'] + np.stack([w['drug_eff'][int(p[1:])] for p in w['pert']]) + rng.normal(0, 0.3, w['y'].shape)
            for _ in range(3)]
    plate_only = [w['plate_mean'] + rng.normal(0, 1e-9, w['y'].shape) for _ in range(3)]
    _patch_p9(monkeypatch, w, good, plate_only, [g.copy() for g in good])   # M = v9 here, so comparing against M would read no claim
    res = A.test_c1(n_boot=2000)
    assert res['primary_v9_vs_mdt']['verdict'] == 'v9 predicts unseen compounds better than M+dt'
    assert res['statement'].startswith('v9 uses the compound')
    assert res['reported']['zero_variance_prediction_rows']['mdt_all_seeds'] == 3 * res['rows']['with_ge2_mates']


def test_c1_does_not_credit_a_plate_only_v9(monkeypatch):
    w = _world()
    rng = w['rng']
    plate_only = [w['plate_mean'] + rng.normal(0, 1e-9, w['y'].shape) for _ in range(3)]
    noisy = [w['plate_mean'] + rng.normal(0, 0.5, w['y'].shape) for _ in range(3)]       # varies, but not by compound
    _patch_p9(monkeypatch, w, noisy, plate_only, plate_only)
    res = A.test_c1(n_boot=2000)
    assert res['primary_v9_vs_mdt']['verdict'] != 'v9 predicts unseen compounds better than M+dt'


def test_c1_uses_only_clean_rows(monkeypatch):
    w = _world()
    rng = w['rng']
    good = [w['y'] + rng.normal(0, 0.1, w['y'].shape) for _ in range(3)]
    flat = [w['plate_mean'] for _ in range(3)]
    clean = np.zeros(len(w['y']), bool)
    clean[: len(w['y']) // 2] = True
    _patch_p9(monkeypatch, w, good, flat, flat, clean=clean)
    res = A.test_c1(n_boot=500)
    have = A.plate_contrast(w['y'], w['plate'], w['pert'])[1] >= 2
    assert res['rows']['scored_clean'] == int((have & clean).sum())


def test_main_runs_once(tmp_path, monkeypatch):
    monkeypatch.setitem(A.RUNNERS, 'C4', lambda: {'test': 'C4'})
    A.main(['--test', 'C4', '--out_dir', str(tmp_path)])
    assert json.load(open(tmp_path / 'phase1_C4.json'))['test'] == 'C4'
    with pytest.raises(SystemExit, match='run once'):
        A.main(['--test', 'C4', '--out_dir', str(tmp_path)])
