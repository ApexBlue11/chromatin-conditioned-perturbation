"""Tests for the v10-C L1 full-release substrate (RESULTS 100.2)."""
import json, os, sys

import h5py
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_full_substrate as bfs  # noqa: E402


def test_assign_splits_rules():
    cell = np.array(['A', 'A', 'B', 'B', 'T', 'T', 'C'])
    pert = np.array(['p1', 'p2', 'p1', 'p3', 'p1', 'p2', 'p9'])
    has = np.array([True, True, True, True, True, True, False])
    smiles = {'p1': 'CC', 'p2': 'CO', 'p3': 'CO', 'p9': 'N'}           # p3 is p2's structure under another ID
    # benchmark rows: full rows 0 (train), 4 (test, cell T, compound p1), 1 (test for cold-drug: compound p2)
    bench_pos = np.array([0, 4, 1])
    split = {'cold_cell_1': np.array(['train', 'test', 'train']), 'cold_drug_1': np.array(['train', 'test', 'test'])}
    lab = bfs.assign_splits(cell, pert, bench_pos, split, has, smiles)
    cc, cd = lab['cold_cell_1'], lab['cold_drug_1']
    assert list(cc) == ['train', 'train', 'train', 'train', 'test', 'excluded_test_cell', 'excluded_no_features']
    # cold drug: test compounds p1 (row 4) and p2 (row 1); p3 shares p2's structure; benchmark train row 0 (p1) stays train
    assert list(cd) == ['train', 'test', 'excluded_test_compound', 'excluded_test_compound', 'test',
                        'excluded_test_compound', 'excluded_no_features']


def _write_fixture(tmp, perturb_x=False):
    rng = np.random.default_rng(0)
    cells = ['A', 'A', 'B', 'B', 'T', 'T', 'A']
    perts = ['p1', 'p2', 'p1', 'p3', 'p1', 'p2', 'p9']
    times, didx = [24, 24, 24, 6, 24, 24, 24], [1, 1, 2, 1, 1, 1, 1]
    keys = [f'{c}:{p}:{t} h:{d}' for c, p, t, d in zip(cells, perts, times, didx)]
    X = rng.normal(size=(7, 978))
    ctl_u = rng.normal(size=(3, 978))
    C = ctl_u[[0, 0, 1, 1, 2, 2, 0]]
    h = tmp / 'full.h5ad'
    with h5py.File(h, 'w') as f:
        f['X'] = X
        f.create_group('obsm')['X_ctl'] = C
        o = f.create_group('obs')
        for k, v in [('cell_drug_dose_time', keys), ('cell_iname', cells), ('pert_id', perts),
                     ('pert_dose', ['1.0'] * 7), ('pert_dose_idx', [str(d) for d in didx]),
                     ('pert_time', [str(float(t)) for t in times]), ('det_plate', ['P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P1'])]:
            o[k] = np.array(v, dtype='S')
    bi = [0, 4, 1]
    bx = X[bi].astype(np.float32)
    if perturb_x:
        bx[1, 5] += 1.0
    np.savez(tmp / 'bundle.npz', X=bx, X_ctl=C[bi].astype(np.float32), meta_cell=np.array([cells[i] for i in bi]),
             meta_pert_id=np.array([perts[i] for i in bi]), meta_time=np.array([float(times[i]) for i in bi]),
             meta_dose_idx=np.array([didx[i] for i in bi]),
             split_split_cold_cell_1=np.array(['train', 'test', 'train']),
             split_split_cold_drug_1=np.array(['train', 'test', 'test']))
    json.dump({'p1': 0, 'p2': 1, 'p3': 2}, open(tmp / 'idx.json', 'w'))
    (tmp / 'list.tsv').write_text('pert_id\tcanonical_smiles\np1\tCC\np2\tCO\np3\tCO\np9\tN\n', encoding='utf-8')
    return h, X, C


def test_build_end_to_end(tmp_path):
    h, X, C = _write_fixture(tmp_path)
    out = tmp_path / 'sub.npz'
    rec = bfs.build(str(out), h5ad=str(h), bundle=str(tmp_path / 'bundle.npz'), drug_index=str(tmp_path / 'idx.json'),
                    drug_list=str(tmp_path / 'list.tsv'))
    z = np.load(out)
    # kept: rows that are train or test in either split -> not row 5 (excluded in both) nor row 6 (no features)
    assert list(z['full_row']) == [0, 1, 2, 3, 4]
    assert np.array_equal(z['X'], X[:5].astype(np.float32))
    assert np.array_equal(z['ctl'][z['ctl_id']], C[:5].astype(np.float32)) and rec['n_unique_controls'] == 3
    assert list(z['bench_row']) == [0, 2, -1, -1, 1]
    assert list(z['split_cold_cell_1']) == ['train', 'train', 'train', 'train', 'test']
    assert list(z['split_cold_drug_1']) == ['train', 'test', 'excluded_test_compound', 'excluded_test_compound', 'test']
    assert list(z['plate_id']) == [0, 0, 1, 1, 2]
    assert list(z['plate_key']) == [bfs.hashlib.sha1(x.encode()).hexdigest()[:16] for x in ('P1', 'P2', 'P3')]
    assert rec['compounds_without_features'] == 1 and rec['out_sha1'] == bfs.sha1(str(out))


def test_build_refuses_value_mismatch(tmp_path):
    h, _, _ = _write_fixture(tmp_path, perturb_x=True)
    with pytest.raises(SystemExit, match='differ from the full release'):
        bfs.build(str(tmp_path / 'sub.npz'), h5ad=str(h), bundle=str(tmp_path / 'bundle.npz'),
                  drug_index=str(tmp_path / 'idx.json'), drug_list=str(tmp_path / 'list.tsv'))


def test_build_refuses_unmatched_benchmark_row(tmp_path):
    h, _, _ = _write_fixture(tmp_path)
    b = dict(np.load(tmp_path / 'bundle.npz'))
    b['meta_dose_idx'] = np.array([1, 1, 7])
    np.savez(tmp_path / 'bundle.npz', **b)
    with pytest.raises(SystemExit, match='not in the full release'):
        bfs.build(str(tmp_path / 'sub.npz'), h5ad=str(h), bundle=str(tmp_path / 'bundle.npz'),
                  drug_index=str(tmp_path / 'idx.json'), drug_list=str(tmp_path / 'list.tsv'))
