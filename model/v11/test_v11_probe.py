"""Tests for v11_probe.py (RESULTS 102.1 C6)."""
import csv, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_v10c as dv  # noqa: E402
import train_v10c as tv  # noqa: E402
import train_v11 as t11  # noqa: E402
import v11_probe as vp  # noqa: E402
from test_v10c_model import tiny_cfg  # noqa: E402
from test_v11_model import write_v11_fixture, built_pair  # noqa: E402


def test_recall_tanimoto_and_bins_by_hand():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert vp.recall_at_k(s, {1, 2}, (1, 2, 3)) == {1: 0.5, 2: 0.5, 3: 1.0}
    fp = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0]])
    assert np.allclose(vp.tanimoto_max(fp, [0], [1, 2]), [0.5]) and np.allclose(vp.tanimoto_max(fp, [0], [3]), [2 / 3])
    assert vp.bin_of(0.29) == '<0.3' and vp.bin_of(0.3) == '0.3-0.5' and vp.bin_of(0.5) == '>=0.5' and vp.bin_of(1.0) == '>=0.5'


def test_atom_mass_recomputes_the_attention_the_module_used(tmp_path):
    """The hook's weights rebuild the module's own output (attention @ v, then o), and padded atoms get none."""
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    v11.eval()
    blk = v11.perturb[0].cross
    got = {}

    def grab(m, inp, out):
        x, mem, mk = inp
        B, L, _ = x.shape
        M = mem.shape[1]
        k, v = m.kv(mem).chunk(2, dim=-1)
        q = m.q_norm(m.q(x).view(B, L, m.h, m.dh).transpose(1, 2))
        k = m.k_norm(k.view(B, M, m.h, m.dh).transpose(1, 2))
        a = torch.softmax(((q @ k.transpose(-1, -2)) / np.sqrt(m.dh)).masked_fill(mk[:, None, None, :], float('-inf')), -1)
        got.update(a=a, v=v.view(B, M, m.h, m.dh).transpose(1, 2), out=out, m=m)
    h = blk.register_forward_hook(grab)
    am = vp.AtomMass(v11)
    b = D.batch(D.ev[:5], 'cpu')
    b['atom_mask'][0, -2:] = False                                   # padded atoms for row 0
    with torch.no_grad():
        v11(b)
    h.remove()
    maps = am.maps
    am.close()
    assert len(maps) == len(v11.perturb) and maps[0].shape == (5, 978)
    assert float(maps[0].min()) >= 0 and float(maps[0].max()) <= 1
    assert torch.allclose(maps[0], got['a'][..., 1:].sum(-1).mean(1), atol=1e-6)
    rebuilt = got['m'].o((got['a'] @ got['v']).transpose(1, 2).reshape(5, 978, -1))
    assert torch.allclose(rebuilt, got['out'], atol=1e-5)
    assert float(got['a'][0, :, :, -2:].abs().max()) == 0.0


def test_probe_end_to_end_on_trained_tiny_models(tmp_path, monkeypatch):
    write_v11_fixture(tmp_path)
    monkeypatch.setattr(dv, 'dev_carve', lambda tc, rpc, ax, **k: (['C0', 'C2'], sorted(set(tc))))
    monkeypatch.setattr(t11, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    with open(tmp_path / 'dti_reference.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['pert_id', 'gene_idx', 'evidence', 'chembl_direct', 'chembl_actions'])
        w.writerow(['p6', '11', 'chembl+stitch', '1', 'INHIBITOR'])
        w.writerow(['p6', '40', 'stitch', '0', ''])
        w.writerow(['p0', '3', 'chembl', '1', 'INHIBITOR'])
    common = ['--roots', str(tmp_path), '--cpu_ok', '--epochs', '1', '--batch', '16', '--d_model', '32', '--dev']
    for arm in ('full', 'no_targets'):
        t11.main(common + ['--out', str(tmp_path / 'ck'), '--arm', arm])
    tv.main(common + ['--out', str(tmp_path / 'tb'), '--chrom_off'])
    rec = vp.main(['--ckpts', str(tmp_path / 'ck'), '--tb_ckpts', str(tmp_path / 'tb'), '--roots', str(tmp_path),
                   '--out', str(tmp_path / 'probe.json')])
    assert rec['n_probe_compounds'] == 1 and rec['n_scored'] == {'heldout': 1, 'train': 1}
    assert rec['n_by_subset']['heldout'] == {'all': 1, 'chembl': 1, 'both': 1}
    assert set(rec['models']) == {'cold_cell_1_dev_v11full_d32_s0', 'cold_cell_1_dev_v11no_targets_d32_s0',
                                  'reference_cold_cell_1_dev_off_d32_s0'}
    for v in rec['models'].values():
        assert set(v) == {'heldout', 'train'}
        for s in v.values():
            a = s['all']['all_bins']
            assert a['n_compounds'] == 1 and 0 <= a['recall@5'] <= 1 and np.isclose(a['chance@50'], 50 / 978)
            assert s['chembl']['all_bins']['recall@50'] in (0.0, 1.0)
    assert 0 <= rec['max_tanimoto']['p6'] <= 1
