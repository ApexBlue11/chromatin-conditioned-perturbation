"""Tests for hdd_test.py (RESULTS 100.15)."""
import json, os, shutil, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_v10c as dv  # noqa: E402
import hdd_test as hd  # noqa: E402
import train_v10c as tv  # noqa: E402
from test_v10c_model import tiny_cfg, write_fixture  # noqa: E402


def test_relabel_keeps_train_moves_the_cells_excluded_rows_and_drops_the_rest():
    lab = np.array(['train', 'test', 'excluded_test_cell', 'excluded_test_cell', 'train'])
    cell = np.array(['A', 'MCF7', 'MCF7', 'H1975', 'MCF7'])
    assert hd.relabel(lab, cell).tolist() == ['train', 'dropped', 'test', 'dropped', 'train']


def test_rule_by_hand():
    comp = np.repeat(np.arange(10), 3)
    on = np.full((3, 30), 0.3) + np.array([[0.01], [0.02], [0.03]])
    off = np.full((3, 30), 0.2)
    r = hd.rule(on, off, comp, 500)
    assert r['cond_i_all_on_above_all_off'] and r['cond_ii_lower_bound_above_0'] and r['verdict'].startswith('in MCF7')
    assert np.isclose(r['delta'], 0.12) and np.isclose(r['cluster_one_sided_lower_95'], 0.12)
    mixed = off.copy()
    mixed[0] += 0.5                                                  # one off slot above every on slot
    assert hd.rule(on, mixed, comp, 500)['verdict'] == 'no claim'
    assert hd.rule(off, on, comp, 500)['verdict'].startswith('mirror')


def make_world(tmp, monkeypatch):
    write_fixture(tmp)
    z = dict(np.load(tmp / 'v10c_full_substrate.npz'))
    lab = z['split_cold_cell_1'].astype('<U20')
    c6 = z['meta_cell_iname'] == 'C6'
    lab[c6 & (np.arange(len(lab)) % 2 == 0)] = 'excluded_test_cell'      # half of C6's rows are untouched
    z['split_cold_cell_1'] = lab
    np.savez(tmp / 'v10c_full_substrate.npz', **z)
    cls = {'class': {f'p{k}': ('dna_damage' if k < 5 else 'kinase') for k in range(12)}}
    json.dump(cls, open(tmp / 'cls.json', 'w'))
    monkeypatch.setattr(hd, 'CELLS', ('C6',))
    monkeypatch.setattr(hd, 'PRIMARY', 'C6')
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    runs = tmp / 'runs'
    for arm, slots in hd.CKPTS.items():
        for slot, sub, seed in slots:
            out = tmp / 'train' / f'{arm}{seed}'
            tv.main(['--roots', str(tmp), '--out', str(out), '--cpu_ok', '--epochs', '1', '--batch', '16', '--d_model', '32',
                     '--seed', str(seed)] + (['--chrom_off'] if arm == 'off' else []))
            d = runs / slot / sub
            d.mkdir(parents=True)
            for f in out.glob('v10c_*.pt'):
                shutil.copy(f, d / f.name)
    counts = hd.count_rows(np.load(tmp / 'v10c_full_substrate.npz'), json.load(open(tmp / 'drug_feature_index.json')),
                           cls['class'], ('C6',))
    return runs, counts


def test_end_to_end_scores_only_untouched_rows_and_checks_the_count(tmp_path, monkeypatch):
    runs, counts = make_world(tmp_path, monkeypatch)
    base = dv.V10CData([str(tmp_path)])
    D = hd.load_data([str(tmp_path)], chrom_off=False)
    assert np.array_equal(D.sub_row[D.tr], base.sub_row[base.tr])                      # training rows are T-C's
    z = np.load(tmp_path / 'v10c_full_substrate.npz')
    assert set(z['split_cold_cell_1'][D.sub_row[D.ev]]) == {'excluded_test_cell'}       # only untouched rows held out
    assert np.isclose(D.dose_n[D.tr].mean(), base.dose_n[base.tr].mean())
    assert dv.np.load is np.load                                                       # the patch is undone
    common = ['--tc_runs', str(runs), '--roots', str(tmp_path), '--classes', str(tmp_path / 'cls.json'), '--n_boot', '300']
    rec = hd.main(common + ['--out', str(tmp_path / 'o.json'), '--expected', json.dumps({'C6': counts['C6']})])
    assert tuple(rec['counts']['C6']) == tuple(counts['C6']) and counts['C6'][0] > 0
    assert rec['C6']['n_rows'] == counts['C6'][0] and len(rec['C6']['slot_means_on']) == 3
    assert rec['verdict'] == rec['C6']['verdict']
    with pytest.raises(SystemExit, match='read once'):
        hd.main(common + ['--out', str(tmp_path / 'o.json')])
    with pytest.raises(SystemExit, match='registration counted'):
        hd.main(common + ['--out', str(tmp_path / 'o2.json'), '--expected', json.dumps({'C6': [counts['C6'][0] + 1, 1]})])
