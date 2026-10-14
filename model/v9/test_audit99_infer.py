# -*- coding: utf-8 -*-
"""Tests for audit99_infer.py (RESULTS 99, packet 084): the swap rules and the identity guard. Synthetic only."""
import json, os, sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit99_infer as I


def _cells(rng, n_cells=3, comps=6, rows_per=4, bits=64):
    cell, pert, fp = [], [], []
    base = {}
    for c in range(n_cells):
        for k in range(comps):
            name = 'K%02d' % (k if c == 0 else (k + c) % (comps + 2))
            if name not in base:
                base[name] = rng.random(bits) < 0.3
            for _ in range(rows_per):
                cell.append('C%d' % c); pert.append(name); fp.append(base[name])
    return np.array(cell), np.array(pert), np.array(fp, dtype=np.float32), base


def test_tanimoto_hand_example():
    A = np.array([[1, 1, 0, 0], [0, 0, 0, 0]])
    B = np.array([[1, 0, 0, 0], [1, 1, 0, 0]])
    T = I.tanimoto(A, B)
    assert T[0, 0] == pytest.approx(0.5) and T[0, 1] == pytest.approx(1.0) and T[1, 0] == 0.0


def test_substitutes_are_within_cell_other_compound_and_compound_level():
    rng = np.random.default_rng(0)
    cell, pert, fp, _ = _cells(rng)
    donor = I.substitutes(cell, pert, fp, np.random.default_rng(0))
    assert (donor >= 0).all()
    assert (cell[donor] == cell).all() and (pert[donor] != pert).all()
    for c in np.unique(cell):
        for k in np.unique(pert[cell == c]):
            m = (cell == c) & (pert == k)
            assert len(set(pert[donor[m]])) == 1                    # one substitute per (compound, cell)


def test_tanimoto_constraint_is_respected_and_unswappable_marked():
    rng = np.random.default_rng(1)
    cell, pert, fp, base = _cells(rng, n_cells=1, comps=4)
    fp[pert == 'K01'] = fp[pert == 'K00'][0]                      # K00 and K01 identical: never each other's substitute
    donor = I.substitutes(cell, pert, fp, np.random.default_rng(0), tani_max=0.4)
    ok = donor >= 0
    T = np.array([I.tanimoto(fp[i:i + 1], fp[donor[i]:donor[i] + 1])[0, 0] for i in np.flatnonzero(ok)])
    assert (T < 0.4).all()
    # a cell whose compounds are all near-identical has no allowed substitute: every row is -1
    fp2 = np.repeat(fp[:1], len(fp), 0)
    d2 = I.substitutes(cell, pert, fp2, np.random.default_rng(0), tani_max=0.4)
    assert (d2 == -1).all()


def test_substitutes_are_deterministic_for_the_seed():
    rng = np.random.default_rng(2)
    cell, pert, fp, _ = _cells(rng)
    a = I.substitutes(cell, pert, fp, np.random.default_rng(0), tani_max=0.4)
    b = I.substitutes(cell, pert, fp, np.random.default_rng(0), tani_max=0.4)
    assert np.array_equal(a, b)


class _D:
    def __init__(self, n=10, G=8):
        self.row_index = np.arange(n) + 50
        self.Y = np.random.default_rng(0).normal(size=(n, G)).astype(np.float32)

    def batch(self, idx, dev):
        return {'idx': torch.as_tensor(idx)}


class _M:
    def __init__(self, D, noise=0.0):
        self.D, self.noise = D, noise

    def __call__(self, b):
        y = self.D.Y[b['idx'].numpy()]
        return {'delta': torch.as_tensor(y + self.noise * np.random.default_rng(1).normal(size=y.shape).astype(np.float32))}


def test_identity_passes_on_the_same_predictions_and_refuses_otherwise(tmp_path):
    D = _D()
    f = tmp_path / 'ident.npz'
    np.savez(f, row_index=D.row_index[[1, 3, 5]], deg_pred=D.Y[[1, 3, 5]])
    res = I.identity(_M(D), D, str(f), 'cpu')
    assert res['n'] == 3 and res['min_r'] > 0.9999
    with pytest.raises(SystemExit, match='identity check failed'):
        I.identity(_M(D, noise=1.0), D, str(f), 'cpu')
    np.savez(f, row_index=np.array([999]), deg_pred=D.Y[:1])
    with pytest.raises(SystemExit, match='not in the data'):
        I.identity(_M(D), D, str(f), 'cpu')


FAKE_INFER = '''import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser()
for k in ('--mode', '--ckpt', '--identity', '--out'):
    ap.add_argument(k)
ap.add_argument('--roots', nargs='*')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
np.savez(os.path.join(a.out, '%s_seed0.npz' % a.mode), x=np.zeros(2))
json.dump({'mode': a.mode}, open(os.path.join(a.out, '%s_seed0.json' % a.mode), 'w'))
'''


def _kernel_world(tmp, tamper=None, duplicate=None):
    import hashlib, subprocess
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'orchestration'))
    import make_audit99_kernels as K
    ds = tmp / 'input' / 'lincs-audit99'
    ds.mkdir(parents=True)
    names = list(K.SRC_FILES) + ['v9p9_seed0.pt', 'v9p9_seed0_ident.npz', 'v9p7_seed0.pt', 'v9p7_seed0_ident.npz']
    for n in names:
        (ds / n).write_text(FAKE_INFER if n == 'audit99_infer.py' else 'x = %r\n' % n)
    pins = {n: hashlib.sha1((ds / n).read_bytes()).hexdigest() for n in names}
    if tamper:
        (ds / tamper).write_text('tampered\n')
    if duplicate:
        other = tmp / 'input' / 'another'
        other.mkdir()
        (other / duplicate).write_text('dup\n')
    code = (K.TEMPLATE.replace('__SEED__', '0').replace('__PINS__', repr(pins)).replace('__RUNS__', repr(K.RUNS))
            .replace('/kaggle/input', (tmp / 'input').as_posix()).replace('/kaggle/working', (tmp / 'working').as_posix()))
    kp = tmp / 'kernel.py'
    kp.write_text(code)
    return subprocess.run([sys.executable, str(kp)], capture_output=True, text=True), tmp / 'working' / 'out'


def test_kernel_runs_end_to_end_and_writes_its_manifest(tmp_path):
    r, out = _kernel_world(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.load(open(out / 'AUDIT99_COMPLETE_s0.json'))
    assert m['complete'] and set(m['files']) == {'swap_seed0.npz', 'swap_seed0.json', 'xcell_seed0.npz', 'xcell_seed0.json'}
    assert not (tmp_path / 'working' / 'src').exists()


def test_kernel_refuses_a_tampered_or_duplicated_input(tmp_path):
    r, _ = _kernel_world(tmp_path / 'a', tamper='model_v9.py')
    assert r.returncode != 0 and 'sha1' in (r.stdout + r.stderr)
    r, _ = _kernel_world(tmp_path / 'b', duplicate='v9p9_seed0.pt')
    assert r.returncode != 0 and 'expected one mounted copy' in (r.stdout + r.stderr)
