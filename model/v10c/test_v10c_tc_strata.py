"""Tests for tc_strata.py (RESULTS 100.13 / 101.1 C2, C3) with tc_gap.completed_preds stubbed."""
import json, os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tc_strata as ts  # noqa: E402
import tc_read  # noqa: E402

CELLS = tc_read.CLEAN + tc_read.APART


def world(tmp, monkeypatch, missing_off=()):
    rng = np.random.default_rng(0)
    n_per = 30
    cell = np.repeat(np.array(CELLS), n_per)
    pert = np.array([f'p{k % 15}' for k in range(len(cell))])
    rows = np.arange(len(cell)) + 5
    mc = np.array(['X'] * (len(cell) + 5), dtype=object)
    mp = np.array(['q'] * (len(cell) + 5), dtype=object)
    mc[rows], mp[rows] = cell, pert
    np.savez(tmp / 'sub.npz', meta_cell_iname=mc.astype(str), meta_pert_id=mp.astype(str))
    cl = {'class': {f'p{k}': ('epigenetic' if k < 3 else 'kinase') for k in range(15)},
          'sensitivity_sets': {'hdac_core': ['p0', 'p1'], 'bet_with_apabetalone': ['p2', 'p99']}}
    json.dump(cl, open(tmp / 'cls.json', 'w'))
    base = rng.normal(0.15, 0.1, len(cell))
    base[::11] = np.nan
    paths = {}
    for arm in ('on', 'off'):
        paths[arm] = []
        for s in range(3):
            r = base + rng.normal(0, 0.002, len(cell)) + (0.05 * np.isin(pert, ['p0', 'p1', 'p2']) if arm == 'on' else 0)
            p = tmp / f'{arm}{s}.npz'
            np.savez(p, sub_row=rows, r_contrast=r)
            paths[arm].append(str(p))
    monkeypatch.setattr(ts.tc_gap, 'completed_preds', lambda rd, prefix, arm, ledger=None:
                        (paths[arm], [{}] * 3, list(missing_off) if arm == 'off' else []))
    return ['--runs_dir', 'unused', '--substrate', str(tmp / 'sub.npz'), '--classes', str(tmp / 'cls.json'), '--n_boot', '300']


def test_strata_cover_cells_classes_sets_compounds_and_epigenetic_cells(tmp_path, monkeypatch):
    rec = ts.main(world(tmp_path, monkeypatch) + ['--out', str(tmp_path / 'o.json')])
    assert set(rec['per_cell']) == set(CELLS) and rec['per_cell']['H1975']['apart'] and not rec['per_cell']['MCF7']['apart']
    assert set(rec['per_class']) == {'epigenetic', 'kinase'}
    assert set(rec['per_class_all8_labelled']) == {'label', 'epigenetic', 'kinase'}
    assert rec['per_class_all8_labelled']['kinase']['n_rows'] > rec['per_class']['kinase']['n_rows']
    e = rec['per_class']['epigenetic']
    assert abs(e['delta'] - 0.05) < 0.005 and e['n_compounds'] == 3 and 36 <= e['n_rows'] <= 42     # 3 compounds x 2 rows x 7 cells, less NaN rows
    assert abs(rec['per_class']['kinase']['delta']) < 0.005
    assert set(rec['sensitivity_sets']) == {'hdac_core', 'bet_with_apabetalone'}
    assert rec['sensitivity_sets']['bet_with_apabetalone']['n_compounds'] == 1
    assert set(rec['epigenetic_x_cell']) == set(tc_read.CLEAN)
    found = {d['compound'] for d in rec['per_compound']['discoveries']}
    assert found == {'p0', 'p1', 'p2'} and all(d['class'] == 'epigenetic' for d in rec['per_compound']['discoveries'])
    assert len(rec['per_compound']['relabelling_calibration']) == 9
    assert rec['n_clean_rows'] == 30 * 7 and rec['label'].startswith('reported, exploratory')
    with pytest.raises(SystemExit, match='read once'):
        ts.main(world(tmp_path, monkeypatch) + ['--out', str(tmp_path / 'o.json')])


def test_h1975_rows_stay_out_of_class_and_compound_strata(tmp_path, monkeypatch):
    rec = ts.main(world(tmp_path, monkeypatch) + ['--out', str(tmp_path / 'o.json')])
    n_class = sum(v['n_rows'] for v in rec['per_class'].values())
    n_clean_finite = sum(v['n_rows'] for c, v in rec['per_cell'].items() if c != 'H1975')
    assert n_class == n_clean_finite


def test_a_missing_slot_gives_no_strata(tmp_path, monkeypatch):
    rec = ts.main(world(tmp_path, monkeypatch, missing_off=(2,)) + ['--out', str(tmp_path / 'o.json')])
    assert rec['verdict'].startswith('no strata') and 'per_class' not in rec
