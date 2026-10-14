# -*- coding: utf-8 -*-
"""Tests for audit99_phase1.py (RESULTS 99, packet 084). Synthetic worlds only: no real prediction or test row is read."""
import json, os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit99_phase1 as A
import score_p9
import score_p10


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


def _bundle_for(w, rng, with_dose_effect=False):
    """A synthetic bundle: the scored rows first (split 'test', the world's plates), then a training copy of each on a
    new plate (other compounds), same cell / time / dose. X_ctl is byte-identical within a plate and distinct across."""
    n, G = w['y'].shape
    dose = rng.choice([1.0, 3.0, 10.0], n).astype(np.float32)
    dose_eff = {d: rng.normal(0, 1.0, G) for d in (1.0, 3.0, 10.0)}
    ctl_by_plate = {}
    def ctl(p):
        if p not in ctl_by_plate:
            ctl_by_plate[p] = rng.normal(size=G).astype(np.float32)
        return ctl_by_plate[p]
    y_test = w['y'] + (np.stack([dose_eff[float(d)] for d in dose]) if with_dose_effect else 0)
    y_train = (np.stack([dose_eff[float(d)] for d in dose]) if with_dose_effect else 0) + rng.normal(0, 0.3, (n, G))
    C = np.concatenate([np.stack([ctl(int(p)) for p in w['plate']]), np.stack([ctl(1000 + int(p)) for p in w['plate']])])
    Y = np.concatenate([y_test, y_train])
    b = {'X': (C + Y).astype(np.float32), 'X_ctl': C, 'meta_cell': np.array(['C'] * (2 * n)),
         'meta_time': np.full(2 * n, 24.0, np.float32), 'meta_dose': np.concatenate([dose, dose]),
         'meta_pert_id': np.concatenate([w['pert'], np.array(['T%04d' % i for i in range(n)])]),
         score_p10.SPLIT_KEY: np.array(['test'] * n + ['train'] * n)}
    return b, y_test


def _patch_p9(monkeypatch, w, v9_list, mdt_list, m_list, clean=None, with_dose_effect=False):
    n = len(w['y'])
    rng = np.random.default_rng(7)
    b, y_test = _bundle_for(w, rng, with_dose_effect)
    p9 = {'y': y_test, 'pert': w['pert'], 'molecule': w['mol'], 'clean_mask': np.ones(n, bool) if clean is None else clean,
          'seeds_data': [{'y_pred': v} for v in v9_list], 'ctl_true': np.zeros_like(w['y']),
          'bundle_data': b, 'bundle_idx': np.arange(n)}
    fps = {p: rng.random(64) < 0.3 for p in np.unique(w['pert'])}
    monkeypatch.setattr(A, '_fp_lookup', lambda: (lambda p: fps[p]))
    monkeypatch.setattr(A, '_load_p9_side', lambda *a, **k: (p9, {'pred_deltas': mdt_list}, {'pred_deltas': m_list},
                                                             w['plate'], [v - p9['ctl_true'] for v in v9_list]))
    return p9


def test_c1_detects_a_model_that_beats_the_drug_free_one(monkeypatch):
    w = _world()
    rng = w['rng']
    good = [w['plate_mean'] + np.stack([w['drug_eff'][int(p[1:])] for p in w['pert']]) + rng.normal(0, 0.3, w['y'].shape)
            for _ in range(3)]
    plate_only = [w['plate_mean'] + rng.normal(0, 1e-9, w['y'].shape) for _ in range(3)]
    _patch_p9(monkeypatch, w, good, plate_only, [g.copy() for g in good])   # M = v9 here, so comparing against M would read no claim
    res = A.test_c1(n_boot=2000)
    assert res['primary_v9_vs_mdt']['verdict'] == 'v9 predicts unseen compounds better than M+dt'
    assert res['statement'].startswith('v9 beats a drug-free model') and 'C3' in res['statement']
    assert res['reported']['zero_variance_prediction_rows']['mdt_all_seeds'] == 3 * res['rows']['with_ge2_mates']
    assert set(res['reported']['by_mates_tanimoto']) == {'median', 'mates_tanimoto_below_median',
                                                        'mates_tanimoto_at_or_above_median'}


def test_c1_does_not_credit_a_plate_only_v9(monkeypatch):
    w = _world()
    rng = w['rng']
    plate_only = [w['plate_mean'] + rng.normal(0, 1e-9, w['y'].shape) for _ in range(3)]
    noisy = [w['plate_mean'] + rng.normal(0, 0.5, w['y'].shape) for _ in range(3)]       # varies, but not by compound
    _patch_p9(monkeypatch, w, noisy, plate_only, plate_only)
    res = A.test_c1(n_boot=2000)
    assert res['primary_v9_vs_mdt']['verdict'] != 'v9 predicts unseen compounds better than M+dt'


def test_c1_no_claim_statement_carries_its_bound(monkeypatch):
    w = _world()
    rng = w['rng']
    same = [w['plate_mean'] + rng.normal(0, 0.5, w['y'].shape) for _ in range(3)]
    _patch_p9(monkeypatch, w, same, [s.copy() for s in same], same)
    res = A.test_c1(n_boot=1000)
    assert res['primary_v9_vs_mdt']['verdict'] == 'NO COMPOUND-LEVEL CLAIM'
    assert ('at most %+.3f per molecule' % res['primary_v9_vs_mdt']['ci_hi']) in res['statement']


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


def test_dose_only_floor_sees_dose_structure_and_only_from_training_rows(monkeypatch):
    w = _world(drug_sd=0.05)
    rng = w['rng']
    flat = [w['plate_mean'] for _ in range(3)]
    p9 = _patch_p9(monkeypatch, w, flat, flat, flat, with_dose_effect=True)
    res = A.test_c1(n_boot=500)
    fl = res['reported']['dose_only_floor']
    assert fl['row_mean_r']['mean'] > 0.3 and fl['rows'] > 0
    # with no training row at any (cell, time, dose), the floor has no rows
    p9['bundle_data'][score_p10.SPLIT_KEY][:] = 'test'
    res2 = A.test_c1(n_boot=200)
    assert res2['reported']['dose_only_floor']['rows'] == 0


def test_loo_mean_hand_example():
    V = np.arange(6, dtype=float)[:, None]
    outer = np.array([0, 0, 0, 0, 1, 1])
    inner = np.array([0, 0, 1, 2, 3, 4])
    sel = np.array([True, True, True, False, True, True])
    M, cnt = A.loo_mean(V, outer, inner, sel, np.array([0, 2, 3, 4]))
    assert cnt.tolist() == [1, 2, 3, 1]
    assert M[0, 0] == pytest.approx(2.0) and M[1, 0] == pytest.approx(0.5) and M[2, 0] == pytest.approx(1.0)
    assert M[3, 0] == pytest.approx(5.0)


def test_mates_tanimoto_hand_example():
    fps = {'a': np.array([1, 1, 0, 0], bool), 'b': np.array([1, 0, 0, 0], bool), 'c': np.array([0, 0, 1, 1], bool)}
    t = A.mates_tanimoto(np.array([0, 0, 0, 1]), np.array(['a', 'b', 'c', 'a']), lambda p: fps[p])
    assert t[0] == pytest.approx((0.5 + 0.0) / 2) and t[1] == pytest.approx((0.5 + 0.0) / 2) and t[2] == pytest.approx(0.0)
    assert np.isnan(t[3])


def _write_infer(tmp, mode, k, arrays, record_over=None):
    tag = A.TAG[mode]
    d = tmp / ('%s_s%d' % (tag, k))
    d.mkdir(parents=True, exist_ok=True)
    f = d / ('%s_seed%d.npz' % (mode, k))
    np.savez_compressed(f, **arrays)
    rec = {'mode': mode, 'seed': k, 'limit': 0}
    rec.update(record_over or {})
    json.dump(rec, open(str(f).replace('.npz', '.json'), 'w'))
    json.dump({'complete': True, 'seed': k, 'files': {f.name: A.sha1(str(f))}},
              open(d / ('AUDIT99_COMPLETE_%s_s%d.json' % (tag, k)), 'w'))
    return f


def _seen_bound(tmp, rng, n=50, G=20, drop=0.2):
    """P9-side outputs for C2b's seen-cell bound: intact better than the x_cell replacements by about `drop`."""
    y = rng.normal(size=(n, G)).astype(np.float32)
    for k in (0, 1, 2):
        d = {'row_index': np.arange(n), 'ctl': np.zeros((n, G), np.float32), 'y': y,
             'deg_intact': (y + rng.normal(size=(n, G))).astype(np.float32),
             'deg_ctl': (y * (1 - drop) + rng.normal(size=(n, G))).astype(np.float32),
             'deg_global': (y * (1 - 2 * drop) + rng.normal(size=(n, G))).astype(np.float32),
             'deg_tani': y, 'deg_free': y, 'donor_row_index_tani': np.arange(n), 'donor_row_index_free': np.arange(n)}
        _write_infer(tmp, 'swap', k, d)


def _c2b_world(tmp, lead_v9=0.4, replaced_level=0.0, n_per_cell=60, cells=8, seed=0):
    rng = np.random.default_rng(seed)
    G = 40
    n = n_per_cell * cells
    ri = np.arange(n) + 100
    ctl = rng.normal(size=(n, G)).astype(np.float32)
    y = rng.normal(size=(n, G)).astype(np.float32)
    xp = (0.3 * y + rng.normal(size=(n, G))).astype(np.float32)
    def pred(s):
        return (s * y + rng.normal(size=(n, G))).astype(np.float32)
    for k in (0, 1, 2):
        _write_infer(tmp, 'xcell', k, {'row_index': ri, 'ctl': ctl, 'y': y, 'deg_intact': pred(0.3 + lead_v9),
                                        'deg_ctl': pred(0.3 + replaced_level), 'deg_global': pred(0.3)})
    _seen_bound(tmp, rng)
    o2 = tmp / 'o2.npz'
    np.savez(o2, row_index=ri, y_pred=xp + ctl, y_true=y + ctl, ctl_true=ctl)
    bun = tmp / 'bundle.npz'
    np.savez(bun, row_index=ri, meta_cell=np.repeat(np.array(['c%d' % i for i in range(cells)]), n_per_cell))
    return str(o2), str(bun)


def test_read_c2b_carried_not_carried_and_not_assessable(tmp_path):
    o2, bun = _c2b_world(tmp_path / 'a', lead_v9=0.5, replaced_level=0.0)
    r = A.read_c2b(str(tmp_path / 'a'), o2, bun, n_boot=500)
    assert r['assessable'] and 'substantially carried' in r['statement'] and 'not substantially' not in r['statement']
    assert 'available only transductively' in r['statement']
    sb = r['seen_cell_bound_p9']
    assert sb['ctl']['drop_row_mean'] > 0 and sb['global']['drop_row_mean'] > sb['ctl']['drop_row_mean']
    o2, bun = _c2b_world(tmp_path / 'b', lead_v9=0.5, replaced_level=0.5)
    r = A.read_c2b(str(tmp_path / 'b'), o2, bun, n_boot=500)
    assert r['assessable'] and 'not substantially carried' in r['statement']
    o2, bun = _c2b_world(tmp_path / 'c', lead_v9=0.0, replaced_level=0.0)
    r = A.read_c2b(str(tmp_path / 'c'), o2, bun, n_boot=500)
    assert not r['assessable'] and r['statement'].startswith('not assessable')


def test_read_c2b_refuses_a_tampered_or_smoke_output(tmp_path):
    o2, bun = _c2b_world(tmp_path / 'a')
    f = tmp_path / 'a' / 'p7_s1' / 'xcell_seed1.npz'
    z = dict(np.load(f))
    z['deg_ctl'] = z['deg_ctl'] + 1
    np.savez_compressed(f, **z)
    with pytest.raises(SystemExit, match='manifest'):
        A.read_c2b(str(tmp_path / 'a'), o2, bun, n_boot=100)
    o2, bun = _c2b_world(tmp_path / 'b')
    _write_infer(tmp_path / 'b', 'xcell', 2, dict(np.load(tmp_path / 'b' / 'p7_s2' / 'xcell_seed2.npz')), {'limit': 8})
    with pytest.raises(SystemExit, match='not a full'):
        A.read_c2b(str(tmp_path / 'b'), o2, bun, n_boot=100)


def _c3_world(tmp, monkeypatch, right_matters=True):
    w = _world()
    rng = w['rng']
    n, G = w['y'].shape
    eff = np.stack([w['drug_eff'][int(p[1:])] for p in w['pert']])
    donor = rng.permutation(n)
    plate_mean = w['plate_mean']
    intact = plate_mean + eff
    swapped = plate_mean + (eff[donor] if right_matters else eff)
    C = np.stack([np.full(G, float(p)) for p in w['plate']]).astype(np.float32)
    for k in (0, 1, 2):
        _write_infer(tmp, 'swap', k, {'row_index': np.arange(n), 'ctl': C, 'y': w['y'].astype(np.float32),
                                       'deg_intact': (intact + rng.normal(0, 0.2, (n, G))).astype(np.float32),
                                       'deg_tani': (swapped + rng.normal(0, 0.2, (n, G))).astype(np.float32),
                                       'deg_free': (intact + rng.normal(0, 0.2, (n, G))).astype(np.float32),   # differs from tani on purpose
                                       'deg_ctl': (intact + rng.normal(0, 0.2, (n, G))).astype(np.float32),
                                       'deg_global': (intact + rng.normal(0, 0.2, (n, G))).astype(np.float32),
                                       'donor_row_index_tani': donor, 'donor_row_index_free': donor})
    p9 = {'row_index': np.arange(n), 'y': w['y'], 'bundle_data': {'X_ctl': C}, 'bundle_idx': np.arange(n),
          'ctl_true': C.astype(np.float64), 'pert': w['pert'], 'clean_mask': np.ones(n, bool), 'molecule': w['mol']}
    monkeypatch.setattr(A.score_p10, 'load_p9_and_metadata', lambda *a, **k: p9)


def test_read_c3_of_record_variant_and_bound(tmp_path, monkeypatch):
    _c3_world(tmp_path / 'a', monkeypatch, right_matters=True)
    r = A.read_c3(str(tmp_path / 'a'), 'unused', n_boot=1000)
    assert r['statement'].startswith("v9's predictions depend on the right compound") and set(r['variants']) == {'tani', 'free'}
    assert 'on_tani_rows' in r['variants']['free']
    _c3_world(tmp_path / 'b', monkeypatch, right_matters=False)
    r = A.read_c3(str(tmp_path / 'b'), 'unused', n_boot=1000)
    hi = r['variants']['tani']['contrast_intact_vs_swapped']['ci_hi']
    assert r['statement'].startswith("v9's dependence on the right compound is not shown") and ('at most %+.3f' % hi) in r['statement']


def test_main_runs_once(tmp_path, monkeypatch):
    monkeypatch.setitem(A.RUNNERS, 'C4', lambda: {'test': 'C4'})
    A.main(['--test', 'C4', '--out_dir', str(tmp_path)])
    assert json.load(open(tmp_path / 'phase1_C4.json'))['test'] == 'C4'
    with pytest.raises(SystemExit, match='run once'):
        A.main(['--test', 'C4', '--out_dir', str(tmp_path)])


def test_dose_only_floor_excludes_the_rows_own_plate(monkeypatch):
    w = _world(n_plates=1, per_plate=8)
    flat = [w['plate_mean'] for _ in range(3)]
    p9 = _patch_p9(monkeypatch, w, flat, flat, flat, with_dose_effect=True)
    b = p9['bundle_data']
    n = len(w['y'])
    b['X_ctl'][n:] = b['X_ctl'][:n]                       # every training row now shares the (single) plate's control
    b['X'][n:] = b['X_ctl'][n:] + (b['X'][n:] - b['X'][n:].mean())
    res = A.test_c1(n_boot=200)
    assert res['reported']['dose_only_floor']['rows'] == 0


def test_floor_contrast_ignores_plate_mates_without_a_floor():
    plate = np.array([0, 0, 0, 0, 1, 1, 1])
    pert = np.array(['a', 'b', 'c', 'd', 'a', 'b', 'c'])
    D = np.array([[100.], [2.], [4.], [9.], [1.], [3.], [5.]])
    has = np.array([False, True, True, True, True, True, False])
    C, ok = A.floor_contrast(D, has, plate, pert)
    assert ok.tolist() == [False, True, True, True, False, False, False]
    assert C[1, 0] == pytest.approx(2 - (4 + 9) / 2)      # row 0 has no floor: it is not a zero, it is absent
    assert C[2, 0] == pytest.approx(4 - (2 + 9) / 2) and C[3, 0] == pytest.approx(9 - 3.0)
    assert np.isnan(C[0, 0]) and np.isnan(C[4, 0])        # no floor / only one floor-bearing mate


def test_c3_refuses_donors_that_differ_across_seeds(tmp_path, monkeypatch):
    _c3_world(tmp_path / 'a', monkeypatch, right_matters=True)
    f = tmp_path / 'a' / 'p9_s2' / 'swap_seed2.npz'
    z = dict(np.load(f))
    z['donor_row_index_tani'] = z['donor_row_index_tani'][::-1].copy()
    _write_infer(tmp_path / 'a', 'swap', 2, z)
    with pytest.raises(SystemExit, match='disagree'):
        A.read_c3(str(tmp_path / 'a'), 'unused', n_boot=100)
