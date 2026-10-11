# -*- coding: utf-8 -*-
"""Tests for B1's --drug_blind flag in model/v9/xpert_arm.py (RESULTS §98.2 B1, §98.6 C4)."""
import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xpert_arm as xa  # noqa: E402


def _old_batch(self, idx, device, max_atoms=96):
    """Verbatim copy of XPertData.batch before TASK W38 (--drug_blind flag)."""
    d = self.drug_row[idx]
    n = min(max_atoms, max(1, int(max(self.atom_off[k + 1] - self.atom_off[k] for k in d))))
    atoms = np.zeros((len(idx), n, 512), np.float32)
    amask = np.zeros((len(idx), n), bool)
    for b, k in enumerate(d):
        a0, a1 = int(self.atom_off[k]), int(self.atom_off[k + 1])
        m = min(n, a1 - a0)
        atoms[b, :m] = np.asarray(self.atom_reprs[a0:a0 + m], np.float32)
        amask[b, :m] = True
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(device)
    return {'x_ctl': t(self.C[idx]), 'x_cell': t(self.x_cell[idx]),
            'y_abs': t(self.X[idx]), 'y_delta': t(self.X[idx] - self.C[idx]),
            'y_l5': t(np.zeros((len(idx), self.X.shape[1]), np.float32)),
            'm_l3': torch.ones(len(idx), dtype=torch.bool, device=device),
            'm_l5': torch.zeros(len(idx), dtype=torch.bool, device=device),
            'E': t(self.E[idx]), 'r': t(self.r[idx]), 'E_mask': t(self.Em[idx].astype(np.float32)), 'cell_ctx': t(self.ctx[idx]),
            'atoms': t(atoms), 'atom_mask': torch.as_tensor(amask).to(device),
            'u_feats': t(self.u[idx]), 'dose': t(self.dose_n[idx]), 'time': t(self.time_n[idx])}


def _make_dummy_xpert_data(seed=42, n_train=6, n_test=4, d_u=64, drug_blind=False):
    """Synthetic XPertData-like object for self-contained and fast testing."""
    rng = np.random.default_rng(seed)
    N = n_train + n_test
    G = 20
    D = object.__new__(xa.XPertData)
    D.tr = np.arange(n_train)
    D.te = np.arange(n_train, N)
    D.X = rng.standard_normal((N, G)).astype(np.float32)
    D.C = rng.standard_normal((N, G)).astype(np.float32)
    D.x_cell = rng.standard_normal((N, G)).astype(np.float32)
    D.E = rng.standard_normal((N, G, 3)).astype(np.float32)
    D.r = rng.integers(0, 2, (N, G)).astype(np.float32)
    D.Em = rng.integers(0, 2, (N, G, 3)).astype(bool)
    D.ctx = rng.standard_normal((N, 16)).astype(np.float32)
    D.dose_n = rng.standard_normal(N).astype(np.float32)
    D.time_n = rng.standard_normal(N).astype(np.float32)

    # 3 compounds:
    # compound 0: 10 atoms
    # compound 1: 120 atoms (> max_atoms 96)
    # compound 2: 15 atoms
    atom_counts = [10, 120, 15]
    offsets = [0]
    for c in atom_counts:
        offsets.append(offsets[-1] + c)
    D.atom_off = np.array(offsets, dtype=np.int64)
    total_atoms = offsets[-1]
    D.atom_reprs = rng.standard_normal((total_atoms, 512)).astype(np.float32)

    drug_rows = []
    for i in range(n_train):
        drug_rows.append(0 if i < 5 else 1)
    for i in range(n_test):
        drug_rows.append(i % 3)
    D.drug_row = np.array(drug_rows, dtype=np.int64)

    D.u = rng.standard_normal((N, d_u)).astype(np.float32)

    if drug_blind:
        D._apply_drug_blind()
    else:
        D.drug_blind = False
        D.blind_atom = None
        D.drug_blind_info = None

    return D


def test_1_flag_off_unchanged():
    """Test 1: batch() without the flag equals verbatim copy of the old batch() for every key, bitwise."""
    D = _make_dummy_xpert_data(drug_blind=False)
    idx = np.array([0, 2, 5, 6, 8])
    b_new = D.batch(idx, 'cpu')
    b_old = _old_batch(D, idx, 'cpu')

    assert set(b_new.keys()) == set(b_old.keys())
    for k in b_new:
        assert torch.equal(b_new[k], b_old[k]), f"Key {k} differs between new and old batch implementations"


def test_2_flag_on_u():
    """Test 2: Every row equals the training-row mean. Changing a test row does not change it; changing a training row does."""
    D_base = _make_dummy_xpert_data(drug_blind=False)
    raw_u_copy = D_base.u.copy()
    expected_tr_mean = raw_u_copy[D_base.tr].mean(0, keepdims=True)

    D_blind = copy.deepcopy(D_base)
    D_blind._apply_drug_blind()

    # Every row equals the training-row mean computed independently
    for i in range(len(D_blind.u)):
        assert np.array_equal(D_blind.u[i], expected_tr_mean[0])

    # Changing a test row's raw features does NOT change the resulting u
    D_test_mod = copy.deepcopy(D_base)
    D_test_mod.u[D_test_mod.te[0]] += 500.0
    D_test_mod._apply_drug_blind()
    assert np.array_equal(D_test_mod.u, D_blind.u)

    # Changing a training row's raw features DOES change the resulting u
    D_train_mod = copy.deepcopy(D_base)
    D_train_mod.u[D_train_mod.tr[0]] += 500.0
    D_train_mod._apply_drug_blind()
    assert not np.array_equal(D_train_mod.u, D_blind.u)


def test_3_flag_on_the_atom():
    """Test 3: blind_atom equals hand computation over unique training compounds, atom-weighted != row-weighted, full atoms."""
    D = object.__new__(xa.XPertData)
    D.tr = np.array([0, 1, 2, 3, 4, 5])  # 6 training rows
    D.te = np.array([6, 7])               # 2 test rows
    D.drug_row = np.array([0, 0, 0, 0, 0, 1, 1, 2], dtype=np.int64)  # c0 has 5 train rows, c1 has 1 train row; c2 in test

    # c0: 10 atoms filled with 1.0
    # c1: 120 atoms filled with 3.0 (tests that > 96 is not truncated)
    # c2: 15 atoms filled with 5.0 (test-only compound)
    offsets = [0, 10, 130, 145]
    D.atom_off = np.array(offsets, dtype=np.int64)
    total_atoms = offsets[-1]
    reprs = np.zeros((total_atoms, 512), dtype=np.float32)
    reprs[0:10] = 1.0
    reprs[10:130] = 3.0
    reprs[130:145] = 5.0
    D.atom_reprs = reprs
    D.u = np.zeros((8, 16), dtype=np.float32)

    D._apply_drug_blind()

    # Hand calculation:
    # unique training compounds: {0, 1}
    # c0 contributes 10 atoms of 1.0; c1 contributes 120 atoms of 3.0
    # Total pooled atoms = 130
    # Atom-weighted mean = (10 * 1.0 + 120 * 3.0) / 130 = 370 / 130 = 37 / 13
    expected_atom_mean = 37.0 / 13.0
    assert D.drug_blind_info == {'n_train_compounds': 2, 'n_atoms_pooled': 130}
    assert np.allclose(D.blind_atom, expected_atom_mean, atol=1e-6)

    # Row-weighted compound mean: (5 * 1.0 + 1 * 3.0) / 6 = 8 / 6 = 4 / 3 ≈ 1.3333 != 37/13 (≈ 2.8462)
    row_weighted_compounds = (5.0 * 1.0 + 1.0 * 3.0) / 6.0
    assert not np.allclose(D.blind_atom, row_weighted_compounds, atol=1e-2)

    # Row-pooled atom mean: (5 * 10 * 1.0 + 1 * 120 * 3.0) / (50 + 120) = 410 / 170 = 41 / 17 ≈ 2.4118 != 37/13
    row_pooled_atoms = (50.0 * 1.0 + 120.0 * 3.0) / 170.0
    assert not np.allclose(D.blind_atom, row_pooled_atoms, atol=1e-2)

    # Truncation check: if compound 1 were truncated at max_atoms=96:
    # (10 * 1.0 + 96 * 3.0) / 106 = 298 / 106 ≈ 2.8113 != 37/13
    truncated_mean = (10.0 * 1.0 + 96.0 * 3.0) / 106.0
    assert not np.allclose(D.blind_atom, truncated_mean, atol=1e-3)


def test_4_flag_on_batch():
    """Test 4: atoms.shape == (B, 1, 512), atom_mask all True shape (B, 1), other keys bitwise equal to flag-off."""
    D_off = _make_dummy_xpert_data(drug_blind=False)
    D_on = copy.deepcopy(D_off)
    D_on._apply_drug_blind()

    idx = np.array([1, 2, 6, 7])
    B = len(idx)
    b_off = D_off.batch(idx, 'cpu')
    b_on = D_on.batch(idx, 'cpu')

    assert b_on['atoms'].shape == (B, 1, 512)
    assert b_on['atom_mask'].shape == (B, 1)
    assert b_on['atom_mask'].dtype == torch.bool
    assert torch.all(b_on['atom_mask']).item() is True

    # Every row in atoms matches blind_atom
    expected_atoms = torch.as_tensor(D_on.blind_atom).view(1, 1, 512).expand(B, 1, 512)
    assert torch.equal(b_on['atoms'], expected_atoms)

    # Every other key is bitwise equal to the flag-off batch
    keys_to_compare = ['x_ctl', 'x_cell', 'y_abs', 'y_delta', 'y_l5', 'm_l3', 'm_l5',
                       'E', 'r', 'E_mask', 'cell_ctx', 'dose', 'time']
    for k in keys_to_compare:
        assert torch.equal(b_on[k], b_off[k]), f"Key {k} not bitwise equal between flag-on and flag-off batches"


def test_5_assert_drug_constant():
    """Test 5: assert_drug_constant passes on a blind batch; raises on isolated mutations to u_feats, atoms, or atom_mask."""
    B = 4
    u_row = torch.randn(1, 32)
    atom_row = torch.randn(1, 1, 512)
    mask_row = torch.ones(1, 1, dtype=torch.bool)
    blind_batch = {
        'u_feats': u_row.expand(B, -1).clone(),
        'atoms': atom_row.expand(B, 1, 512).clone(),
        'atom_mask': mask_row.expand(B, 1).clone(),
    }

    # Passes on clean blind batch
    xa.assert_drug_constant(blind_batch)

    # Isolated case 1: ONLY u_feats row 3 differs
    b1 = {k: v.clone() for k, v in blind_batch.items()}
    b1['u_feats'][3] = b1['u_feats'][3] + 1.0
    with pytest.raises(RuntimeError, match='u_feats row 3 does not match row 0'):
        xa.assert_drug_constant(b1)

    # Isolated case 2: ONLY atoms row 3 differs
    b2 = {k: v.clone() for k, v in blind_batch.items()}
    b2['atoms'][3] = b2['atoms'][3] + 1.0
    with pytest.raises(RuntimeError, match='atoms row 3 does not match row 0'):
        xa.assert_drug_constant(b2)

    # Isolated case 3: ONLY atom_mask row 3 differs
    b3 = {k: v.clone() for k, v in blind_batch.items()}
    b3['atom_mask'][3] = False
    with pytest.raises(RuntimeError, match='atom_mask row 3 does not match row 0'):
        xa.assert_drug_constant(b3)


class StandInModel(torch.nn.Module):
    def __init__(self, d_u=64, d_x=20):
        super().__init__()
        self.fc_u = torch.nn.Linear(d_u, 5, bias=False)
        self.fc_x = torch.nn.Linear(d_x, 5, bias=False)
        with torch.no_grad():
            self.fc_u.weight.fill_(1.0)
            self.fc_x.weight.fill_(1.0)
        self.eval_states_during_forward = []

    def forward(self, batch):
        self.eval_states_during_forward.append(self.training)
        return {'delta': self.fc_u(batch['u_feats']) + self.fc_x(batch['x_ctl'])}


def test_6_drug_permutation_check():
    """Test 6: drug_permutation_check raises with flag off, passes with 0 with flag on, restores state in try/finally, mode check."""
    d_u = 64
    d_x = 20
    model = StandInModel(d_u=d_u, d_x=d_x)

    # --- Part A: Flag OFF ---
    D_off = _make_dummy_xpert_data(drug_blind=False, d_u=d_u)
    idx = D_off.te.copy()
    orig_drug_row = D_off.drug_row[idx].copy()
    orig_u = D_off.u[idx].copy()

    model.train()
    with pytest.raises(RuntimeError, match='Drug permutation check failed'):
        xa.drug_permutation_check(model, D_off, idx, 'cpu')

    # State restored even after RuntimeError
    assert np.array_equal(D_off.drug_row[idx], orig_drug_row)
    assert np.array_equal(D_off.u[idx], orig_u)
    # Model mode restored to train
    assert model.training is True

    # --- Part B: Flag ON ---
    D_on = _make_dummy_xpert_data(drug_blind=True, d_u=d_u)
    orig_drug_row_on = D_on.drug_row[idx].copy()
    orig_u_on = D_on.u[idx].copy()

    model.train()
    diff = xa.drug_permutation_check(model, D_on, idx, 'cpu')
    assert diff == 0.0

    # State restored after pass
    assert np.array_equal(D_on.drug_row[idx], orig_drug_row_on)
    assert np.array_equal(D_on.u[idx], orig_u_on)
    # Model mode restored to train
    assert model.training is True

    # Model was in eval mode during execution
    assert len(model.eval_states_during_forward) > 0
    assert all(st is False for st in model.eval_states_during_forward)

    # When called in eval mode, original eval mode is preserved
    model.eval()
    diff2 = xa.drug_permutation_check(model, D_on, idx, 'cpu')
    assert diff2 == 0.0
    assert model.training is False


def test_7_wiring():
    """Test 7: candidate_flags['drug_blind'], suffix _drugblind, and without flag unchanged except drug_blind: False."""
    parser = xa.build_parser()

    # Case 1: Without --drug_blind
    args_off = parser.parse_args([])
    assert args_off.drug_blind is False
    flags_off, suffix_off = xa.get_candidate_flags_and_suffix(args_off)

    assert flags_off['drug_blind'] is False
    assert 'drug_blind_info' not in flags_off
    assert '_drugblind' not in suffix_off

    # Keys are as today, except drug_blind: False
    today_keys = {'snapshot_cycles', 'no_atoms', 'drug_blind', 'l_control', 'listnet_w',
                  'deg_adapt_k', 'sign_head_w', 'post_pathway', 'chromatin_edges', 'union_edges'}
    assert set(flags_off.keys()) == today_keys

    # Case 2: With --drug_blind
    args_on = parser.parse_args(['--drug_blind'])
    assert args_on.drug_blind is True
    dummy_info = {'n_train_compounds': 1497, 'n_atoms_pooled': 48123}
    flags_on, suffix_on = xa.get_candidate_flags_and_suffix(args_on, drug_blind_info=dummy_info)

    assert flags_on['drug_blind'] is True
    assert flags_on['drug_blind_info'] == dummy_info
    assert '_drugblind' in suffix_on

    # Case 3: Help text verification
    blind_action = [act for act in parser._actions if act.dest == 'drug_blind'][0]
    expected_help = ("RESULTS 98.2 B1: no input varies with the compound except dose and time; "
                     "u = training-row mean; one atom token = the atom-weighted mean over unique training compounds")
    assert blind_action.help == expected_help


# =========================================================================
# PI additions after the mutation check (4 of 22 survived): the 1e-5 threshold, and main()'s wiring (statically; main() is
# exercised for real by the B1 kernel's local smoke run before any GPU spend).
# =========================================================================
def test_pi_permutation_threshold_is_1e_5():
    """A world whose compound dependence is SMALL (a diff of ~1e-3): it must still raise (1e-5 is the bar)."""
    model = StandInModel(d_u=64, d_x=20)
    with torch.no_grad():
        model.fc_u.weight.fill_(1e-4)                                        # tiny dependence on u_feats
    D = _make_dummy_xpert_data(drug_blind=False, d_u=64)
    with pytest.raises(RuntimeError, match='max abs diff'):
        xa.drug_permutation_check(model, D, D.te.copy(), 'cpu')
    # ...and a world whose diff is exactly zero passes
    with torch.no_grad():
        model.fc_u.weight.zero_()
    assert xa.drug_permutation_check(model, D, D.te.copy(), 'cpu') == 0.0


def test_pi_main_wiring_static():
    """main() passes the flag to the data, checks EVERY batch it builds (eval and training paths), and runs the permutation
    check after each seed's training."""
    import inspect
    src = inspect.getsource(xa.main)
    assert 'drug_blind=a.drug_blind)' in src
    assert src.count('if a.drug_blind:\n') >= 3
    assert src.count('assert_drug_constant(b)') >= 2
    i_train = src.index('opt.zero_grad(set_to_none=True)')
    j = src.rfind('assert_drug_constant(b)', 0, i_train)
    assert j != -1 and i_train - j < 200                                    # the training batch is checked just before use
    assert 'drug_permutation_check(model, D, D.te[:512], dev)' in src
    assert "print(f'DRUGBLIND PERMUTATION CHECK seed {seed} max_abs_diff {chk_diff:.2e} OK', flush=True)" in src
