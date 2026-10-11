# -*- coding: utf-8 -*-
"""Tests for model/v9/score_p10.py (RESULTS §98.3 as amended by §98.6)."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_p10
from score_p9 import per_row_pearson


def file_sha1(path):
    return hashlib.sha1(open(path, 'rb').read()).hexdigest()


def make_p10_fixture(tmp_path, n_train=30, n_test=60, n_clean=50, n_genes=10, seed=42):
    """Build a complete synthetic fixture in tmp_path with valid inputs satisfying all guards."""
    rng = np.random.default_rng(seed)

    p9_dir = tmp_path / 'p9_dir'
    p9_dir.mkdir(parents=True, exist_ok=True)

    ref_dir_m = tmp_path / 'ref_dir_m'
    ref_dir_m.mkdir(parents=True, exist_ok=True)

    test_row_index = np.arange(100, 100 + n_test, dtype=np.int64)
    train_row_index = np.arange(200, 200 + n_train, dtype=np.int64)
    all_row_index = np.concatenate([test_row_index, train_row_index])

    cell_cycle = ['A375', 'MCF7', 'PC3']
    test_cells = np.array([cell_cycle[i % len(cell_cycle)] for i in range(n_test)])
    train_cells = np.array([cell_cycle[i % len(cell_cycle)] for i in range(n_train)])
    all_cells = np.concatenate([test_cells, train_cells])

    test_perts = []
    for i in range(n_test):
        if i < n_clean:
            pert_num = i // (n_clean // 5)
            test_perts.append(f'PERT_{pert_num}')
        else:
            test_perts.append('PERT_DIRTY')
    test_perts = np.array(test_perts)
    train_perts = np.array([f'TR_PERT_{i}' for i in range(n_train)])
    all_perts = np.concatenate([test_perts, train_perts])

    splits = np.array(['test'] * n_test + ['train'] * n_train)

    # Truth and controls
    ctl_test = rng.normal(0, 1, size=(n_test, n_genes))
    y_test_delta = rng.normal(0, 1, size=(n_test, n_genes))
    y_test = ctl_test + y_test_delta

    ctl_train = rng.normal(0, 1, size=(n_train, n_genes))
    y_train_delta = rng.normal(0, 1, size=(n_train, n_genes))
    x_train = ctl_train + y_train_delta

    all_x = np.concatenate([y_test, x_train], axis=0)
    all_ctl = np.concatenate([ctl_test, ctl_train], axis=0)

    # Bundle
    bundle_path = tmp_path / 'bundle.npz'
    np.savez(
        bundle_path,
        row_index=all_row_index,
        meta_pert_id=all_perts,
        meta_cell=all_cells,
        split_split_cold_drug_1=splits,
        X=all_x,
        X_ctl=all_ctl,
    )

    # P9 seeds
    seed_files = []
    manifest_files = {}
    for s in range(3):
        ypred = ctl_test + y_test_delta + rng.normal(0, 1.0, size=(n_test, n_genes))
        s_name = f'v9p9_seed{s}.npz'
        s_path = p9_dir / s_name
        np.savez(
            s_path,
            row_index=test_row_index,
            y_pred=ypred,
            y_true=y_test,
            ctl_true=ctl_test,
        )
        manifest_files[s_name] = file_sha1(s_path)
        seed_files.append(s_name)

    manifest_path = p9_dir / 'P9_COMPLETE.json'
    manifest_data = {
        'complete': True,
        'n_seeds': 3,
        'stack': 'baseline',
        'files': manifest_files,
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)

    # Units JSON
    pert_to_mol = {f'PERT_{i}': f'MOL_{i}' for i in range(5)}
    pert_to_mol['PERT_DIRTY'] = 'MOL_DIRTY'
    units_data = {'pert_to_molecule': pert_to_mol, 'molecule_stratum': {}}
    units_path = tmp_path / 'units.json'
    with open(units_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, indent=2)

    # Clean JSON
    clean_data = {'dirty_compounds': ['PERT_DIRTY']}
    clean_path = tmp_path / 'clean.json'
    with open(clean_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2)

    # Carves JSON
    distinct_rows = test_row_index[:15]
    distinct_sha1 = score_p10.row_set_sha1(distinct_rows)
    carves_data = {
        'rule': 'test',
        'split': 'split_split_cold_drug_1',
        'validation_carve': {},
        'distinct_control_stratum': {
            'clean_distinct_row_index': [int(r) for r in distinct_rows],
            'clean_distinct_rows_sha1': distinct_sha1,
            'n_clean_distinct': len(distinct_rows),
        },
    }
    carves_path = tmp_path / 'carves.json'
    with open(carves_path, 'w', encoding='utf-8') as f:
        json.dump(carves_data, f, indent=2)

    # MLP_M seed files
    mlp_seeds = [131419, 131420, 131421]
    for idx, s in enumerate(mlp_seeds):
        delta_pred = y_test_delta + rng.normal(0, 1.5, size=(n_test, n_genes))
        np.savez(
            ref_dir_m / f'p10_m_seed{idx}.npz',
            row_index=test_row_index,
            delta_pred=delta_pred.astype(np.float32),
            seed=np.int64(s),
            arm=np.str_('m'),
            best_epoch=np.int64(10),
            n_epochs_run=np.int64(20),
        )

    clean_rows = test_row_index[:n_clean]

    return {
        'p9_dir': p9_dir,
        'ref_dir_m': ref_dir_m,
        'bundle_path': bundle_path,
        'units_path': units_path,
        'units_sha1': file_sha1(units_path),
        'clean_path': clean_path,
        'clean_sha1': file_sha1(clean_path),
        'carves_path': carves_path,
        'carves_sha1': file_sha1(carves_path),
        'test_row_index': test_row_index,
        'clean_rows': clean_rows,
        'distinct_rows': distinct_rows,
        'distinct_sha1': distinct_sha1,
        'y_test_delta': y_test_delta,
        'y_test': y_test,
        'ctl_test': ctl_test,
        'test_cells': test_cells,
    }


def patch_constants(monkeypatch, fix, tmp_path):
    monkeypatch.setattr(score_p10, 'REPO', str(tmp_path))
    monkeypatch.setattr(score_p10, 'BUNDLE', str(fix['bundle_path']))
    monkeypatch.setattr(score_p10, 'UNITS_JSON', str(fix['units_path']))
    monkeypatch.setattr(score_p10, 'UNITS_SHA1', fix['units_sha1'])
    monkeypatch.setattr(score_p10, 'CLEAN_JSON', str(fix['clean_path']))
    monkeypatch.setattr(score_p10, 'CLEAN_SHA1', fix['clean_sha1'])
    monkeypatch.setattr(score_p10, 'CARVES_JSON', str(fix['carves_path']))
    monkeypatch.setattr(score_p10, 'CARVES_SHA1', fix['carves_sha1'])
    monkeypatch.setattr(score_p10, 'EXPECT_N_FULL', len(fix['test_row_index']))
    monkeypatch.setattr(score_p10, 'EXPECT_SHA1_FULL', score_p10.row_set_sha1(fix['test_row_index']))
    monkeypatch.setattr(score_p10, 'EXPECT_N_CLEAN', len(fix['clean_rows']))
    monkeypatch.setattr(score_p10, 'EXPECT_SHA1_CLEAN', score_p10.row_set_sha1(fix['clean_rows']))
    monkeypatch.setattr(score_p10, 'EXPECT_N_DISTINCT', len(fix['distinct_rows']))
    monkeypatch.setattr(score_p10, 'EXPECT_SHA1_DISTINCT', fix['distinct_sha1'])
    monkeypatch.setattr(score_p10, 'N_BOOT', 100)


# =========================================================================
# 1. The M seed averaging
# =========================================================================

def test_m_seed_averaging(tmp_path, monkeypatch):
    """Assert r_ref equals the mean of the three per-seed Pearsons, not Pearson of mean prediction."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])
    y_true_delta = p9_data['y']

    # Construct 3 seeds whose per-seed Pearsons differ from Pearson of the average prediction
    # Seed 0: perfect positive
    pred0 = y_true_delta.copy()
    # Seed 1: perfect negative
    pred1 = -y_true_delta.copy()
    # Seed 2: perfect positive
    pred2 = y_true_delta.copy()

    # Mean of per-seed Pearsons is (1 - 1 + 1) / 3 = 1/3 ~ 0.3333
    # Mean prediction is (pred0 + pred1 + pred2) / 3 = y_true_delta / 3, whose Pearson is 1.0!
    seeds = [131419, 131420, 131421]
    for idx, (s, pred) in enumerate(zip(seeds, [pred0, pred1, pred2])):
        np.savez(
            fix['ref_dir_m'] / f'p10_m_seed{idx}.npz',
            row_index=fix['test_row_index'],
            delta_pred=pred.astype(np.float32),
            seed=np.int64(s),
            arm=np.str_('m'),
        )

    ref_data = score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)
    r_ref = ref_data['r_ref']

    mean_per_seed = np.mean(ref_data['r_ref_seeds'], axis=0)
    mean_prediction = np.mean(ref_data['pred_deltas'], axis=0)
    pearson_of_mean_pred = per_row_pearson(mean_prediction, y_true_delta)

    # Assert r_ref equals the mean of the three per-seed Pearsons
    assert np.allclose(r_ref, mean_per_seed)
    assert np.allclose(r_ref, 1.0 / 3.0)

    # Assert r_ref does NOT equal the Pearson of the mean prediction
    assert np.allclose(pearson_of_mean_pred, 1.0)
    assert not np.allclose(r_ref, pearson_of_mean_pred)


# =========================================================================
# 2. The M guards, each isolated
# =========================================================================

def test_m_guard_wrong_arm_fails(tmp_path, monkeypatch):
    """Guard: a wrong arm fails."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])

    # Overwrite seed0 with wrong arm 'mdt'
    np.savez(
        fix['ref_dir_m'] / 'p10_m_seed0.npz',
        row_index=fix['test_row_index'],
        delta_pred=np.zeros((len(fix['test_row_index']), 10), dtype=np.float32),
        seed=np.int64(131419),
        arm=np.str_('mdt'),
    )
    with pytest.raises(SystemExit, match='arm'):
        score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)


def test_m_guard_duplicate_seed_fails(tmp_path, monkeypatch):
    """Guard: duplicate seed fails."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])

    # Overwrite seed1 with duplicate seed 131419
    np.savez(
        fix['ref_dir_m'] / 'p10_m_seed1.npz',
        row_index=fix['test_row_index'],
        delta_pred=np.zeros((len(fix['test_row_index']), 10), dtype=np.float32),
        seed=np.int64(131419),
        arm=np.str_('m'),
    )
    with pytest.raises(SystemExit, match='seeds'):
        score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)


def test_m_guard_reordered_row_index_passes_and_aligns(tmp_path, monkeypatch):
    """Guard: one seed file with different row_index order passes and aligns by row_index."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])

    # Reverse row_index and delta_pred in seed0
    rev_ri = fix['test_row_index'][::-1]
    test_delta = p9_data['y'][::-1]
    np.savez(
        fix['ref_dir_m'] / 'p10_m_seed0.npz',
        row_index=rev_ri,
        delta_pred=test_delta.astype(np.float32),
        seed=np.int64(131419),
        arm=np.str_('m'),
    )

    ref_data = score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)
    # Alignment should ensure pred_deltas[0] matches p9_data['y'] order
    assert np.allclose(ref_data['pred_deltas'][0], p9_data['y'])
    assert np.allclose(ref_data['r_ref_seeds'][0], 1.0)


def test_m_guard_missing_p9_row_fails(tmp_path, monkeypatch):
    """Guard: missing P9 row fails."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])

    # Drop one row in all 3 seeds so they agree on row_index set, but miss a P9 row
    truncated_ri = fix['test_row_index'][:-1]
    for idx, s in enumerate([131419, 131420, 131421]):
        np.savez(
            fix['ref_dir_m'] / f'p10_m_seed{idx}.npz',
            row_index=truncated_ri,
            delta_pred=np.zeros((len(truncated_ri), 10), dtype=np.float32),
            seed=np.int64(s),
            arm=np.str_('m'),
        )
    with pytest.raises(SystemExit, match='missing'):
        score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)


# =========================================================================
# 3. No double subtraction
# =========================================================================

def test_no_double_subtraction(tmp_path, monkeypatch):
    """For mlp_m, delta_pred == true Δ gives r_ref == 1 on every row."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])

    # Set each seed's delta_pred directly to true Δ
    mlp_seeds = [131419, 131420, 131421]
    for idx, s in enumerate(mlp_seeds):
        np.savez(
            fix['ref_dir_m'] / f'p10_m_seed{idx}.npz',
            row_index=fix['test_row_index'],
            delta_pred=p9_data['y'].astype(np.float32),
            seed=np.int64(s),
            arm=np.str_('m'),
        )

    ref_data = score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)
    assert np.allclose(ref_data['r_ref'], 1.0)


# =========================================================================
# 4. f0
# =========================================================================

def test_f0_hand_compute():
    """Hand-compute f0's pred for 3 rows of a synthetic bundle and assert equality."""
    # Synthetic bundle data with 2 cells in train
    bundle_data = {
        'split_split_cold_drug_1': np.array(['train', 'train', 'train']),
        'meta_cell': np.array(['CellA', 'CellA', 'CellB']),
        'X_ctl': np.array([
            [1.0, 2.0],  # CellA
            [3.0, 4.0],  # CellA -> mean ctl(A) = [2.0, 3.0]
            [10.0, 20.0],  # CellB -> mean ctl(B) = [10.0, 20.0]
        ]),
        'X': np.array([
            [3.0, 5.0],  # delta = [2.0, 3.0]
            [7.0, 6.0],  # delta = [4.0, 2.0] -> mean delta(A) = [3.0, 2.5]
            [15.0, 25.0],  # delta = [5.0, 5.0] -> mean delta(B) = [5.0, 5.0]
        ]),
    }

    test_cells = np.array(['CellA', 'CellA', 'CellB'])
    test_ctl = np.array([
        [4.0, 1.0],   # pred = -([4, 1] - [2, 3]) + [3.0, 2.5] = -[2, -2] + [3, 2.5] = [1.0, 4.5]
        [2.0, 3.0],   # pred = -([2, 3] - [2, 3]) + [3.0, 2.5] = [3.0, 2.5]
        [12.0, 18.0], # pred = -([12, 18] - [10, 20]) + [5, 5] = -[2, -2] + [5, 5] = [3.0, 7.0]
    ])

    pred_delta, valid_mask, mean_ctl, mean_delta, n_no_tr = score_p10.compute_f0(
        bundle_data, test_cells, test_ctl, split_key='split_split_cold_drug_1'
    )

    expected = np.array([
        [1.0, 4.5],
        [3.0, 2.5],
        [3.0, 7.0],
    ])
    assert np.allclose(pred_delta, expected)
    assert valid_mask.all()
    assert n_no_tr == 0


def test_f0_unseen_cell_excluded_and_counted():
    """A test row in a cell with no training row is excluded and counted."""
    bundle_data = {
        'split_split_cold_drug_1': np.array(['train']),
        'meta_cell': np.array(['CellA']),
        'X_ctl': np.array([[1.0, 2.0]]),
        'X': np.array([[3.0, 5.0]]),
    }
    test_cells = np.array(['CellA', 'CellNew'])
    test_ctl = np.array([[1.0, 2.0], [5.0, 5.0]])

    pred_delta, valid_mask, mean_ctl, mean_delta, n_no_tr = score_p10.compute_f0(
        bundle_data, test_cells, test_ctl, split_key='split_split_cold_drug_1'
    )
    assert valid_mask[0] is True or valid_mask[0] == 1
    assert valid_mask[1] is False or valid_mask[1] == 0
    assert np.isnan(pred_delta[1]).all()
    assert n_no_tr == 1


def test_f0_changing_test_ctl_affects_only_that_row():
    """Changing a test row's X_ctl changes only that row's f0 prediction."""
    bundle_data = {
        'split_split_cold_drug_1': np.array(['train', 'train']),
        'meta_cell': np.array(['CellA', 'CellA']),
        'X_ctl': np.array([[1.0, 2.0], [3.0, 4.0]]),
        'X': np.array([[2.0, 4.0], [4.0, 6.0]]),
    }
    test_cells = np.array(['CellA', 'CellA'])
    test_ctl_orig = np.array([[2.0, 3.0], [5.0, 6.0]])
    pred_orig, _, _, _, _ = score_p10.compute_f0(bundle_data, test_cells, test_ctl_orig)

    # Change only row 0's control
    test_ctl_mod = np.array([[100.0, 200.0], [5.0, 6.0]])
    pred_mod, _, _, _, _ = score_p10.compute_f0(bundle_data, test_cells, test_ctl_mod)

    # Row 0 changes, row 1 is completely unaffected
    assert not np.allclose(pred_orig[0], pred_mod[0])
    assert np.allclose(pred_orig[1], pred_mod[1])


# =========================================================================
# 5. The distinct stratum
# =========================================================================

def test_distinct_stratum_guards_and_reads(tmp_path, monkeypatch):
    """The distinct stratum uses exactly those rows. Count and sha1 guarded. Row outside clean fails."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    # Scorer runs and records distinct stratum
    out = score_p10.score_p10(p9_dir=str(fix['p9_dir']), ref='mlp_m', ref_dir=str(fix['ref_dir_m']))
    ds = out['distinct_control_stratum']
    assert ds['n_rows'] == len(fix['distinct_rows'])
    assert ds['LABEL'] == 'reported, never read (RESULTS 98.6)'

    # Guard: wrong count fails
    monkeypatch.setattr(score_p10, 'EXPECT_N_DISTINCT', 999)
    with pytest.raises(SystemExit, match='count'):
        score_p10.load_carves(carves_path=str(fix['carves_path']), clean_rows=fix['clean_rows'])

    # Guard: wrong sha1 fails
    monkeypatch.setattr(score_p10, 'EXPECT_N_DISTINCT', len(fix['distinct_rows']))
    monkeypatch.setattr(score_p10, 'EXPECT_SHA1_DISTINCT', 'badsha1')
    with pytest.raises(SystemExit, match='sha1'):
        score_p10.load_carves(carves_path=str(fix['carves_path']), clean_rows=fix['clean_rows'])

    # Guard: stratum row outside clean scored rows fails
    monkeypatch.setattr(score_p10, 'EXPECT_SHA1_DISTINCT', fix['distinct_sha1'])
    # Pass clean_rows that does not contain all distinct rows
    subset_clean = fix['clean_rows'][5:]
    with pytest.raises(SystemExit, match='clean scored rows'):
        score_p10.load_carves(carves_path=str(fix['carves_path']), clean_rows=subset_clean)


# =========================================================================
# 6. The noise-free target
# =========================================================================

def test_noise_free_target(tmp_path, monkeypatch):
    """Noise-free target: pred_Δ = true Δ gives r′ == 1; shared-control gives near-zero r′; NaN dropped."""
    fix = make_p10_fixture(tmp_path, n_genes=100)
    patch_constants(monkeypatch, fix, tmp_path)
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])
    _, _, mean_ctl, _, _ = score_p10.compute_f0(p9_data['bundle_data'], p9_data['cell'], p9_data['ctl_true'])

    # (a) pred_Δ = true Δ -> r' == 1
    mlp_seeds = [131419, 131420, 131421]
    for idx, s in enumerate(mlp_seeds):
        np.savez(
            fix['ref_dir_m'] / f'p10_m_seed{idx}.npz',
            row_index=fix['test_row_index'],
            delta_pred=p9_data['y'].astype(np.float32),
            seed=np.int64(s),
            arm=np.str_('m'),
        )
    ref_data = score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)
    reading, r_v9_prime, r_ref_prime = score_p10.compute_control_noise_free_target(p9_data, ref_data, mean_ctl)
    assert np.allclose(r_ref_prime, 1.0)

    # (b) pred_Δ = -(X_ctl - x̄_ctl) + ε gives high Δ-Pearson but near-zero r'
    rng = np.random.default_rng(99)
    x_bar_ctl = np.array([mean_ctl[c] for c in p9_data['cell']])
    # High variance control so -X_ctl dominates true delta
    eps = rng.normal(0, 0.01, size=p9_data['y'].shape)
    shared_pred_delta = -(p9_data['ctl_true'] - x_bar_ctl) + eps

    # Check that Δ-Pearson is high when X is independent of X_ctl
    # Here delta = y_true - ctl_true
    r_delta = per_row_pearson(shared_pred_delta, p9_data['y'])
    # pred' = (ctl + pred_Δ) - x̄_ctl = eps, which is independent of y' = X - x̄_ctl
    y_prime = p9_data['y_true'] - x_bar_ctl
    r_prime = per_row_pearson(eps, y_prime)

    assert float(np.mean(np.abs(r_prime))) < 0.20

    # (c) A NaN r' row (constant pred') is dropped and counted, never silently scored
    # In row 0, set ctl_true equal to x_bar_ctl and pred_Δ to 0.0 -> pred' == 0 (constant across all genes)
    p9_data_nan = dict(p9_data)
    p9_data_nan['ctl_true'] = p9_data['ctl_true'].copy()
    p9_data_nan['ctl_true'][0] = x_bar_ctl[0]
    p9_data_nan['seeds_data'] = [dict(s) for s in p9_data['seeds_data']]
    for s_dict in p9_data_nan['seeds_data']:
        s_dict['y_pred'] = s_dict['y_pred'].copy()
        s_dict['y_pred'][0] = x_bar_ctl[0]

    zero_seed = [shared_pred_delta.copy() for _ in range(3)]
    for s_idx in range(3):
        zero_seed[s_idx][0] = 0.0

    for idx, s in enumerate(mlp_seeds):
        np.savez(
            fix['ref_dir_m'] / f'p10_m_seed{idx}.npz',
            row_index=fix['test_row_index'],
            delta_pred=zero_seed[idx].astype(np.float32),
            seed=np.int64(s),
            arm=np.str_('m'),
        )
    ref_data = score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data_nan)
    reading_nan, r_v9_prime_nan, r_ref_prime_nan = score_p10.compute_control_noise_free_target(p9_data_nan, ref_data, mean_ctl)
    assert np.isnan(r_ref_prime_nan[0])
    assert reading_nan['n_dropped'] >= 1


# =========================================================================
# 7. Centred
# =========================================================================

def test_centred_per_cell_constant_gives_r_1(tmp_path, monkeypatch):
    """Centred: a world where pred = truth + per-cell constant gives centred r of 1."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])

    # Add different per-cell constant to truth
    cell_offsets = {'A375': 5.0, 'MCF7': -10.0, 'PC3': 20.0}
    pred_shifted = p9_data['y'].copy()
    for c, offset in cell_offsets.items():
        m = (p9_data['cell'] == c)
        pred_shifted[m] += offset

    mlp_seeds = [131419, 131420, 131421]
    for idx, s in enumerate(mlp_seeds):
        np.savez(
            fix['ref_dir_m'] / f'p10_m_seed{idx}.npz',
            row_index=fix['test_row_index'],
            delta_pred=pred_shifted.astype(np.float32),
            seed=np.int64(s),
            arm=np.str_('m'),
        )
    ref_data = score_p10.load_ref('mlp_m', fix['ref_dir_m'], p9_data)
    reading, r_v9_c, r_ref_c = score_p10.compute_cell_centred_reading(p9_data, ref_data)
    assert np.allclose(r_ref_c, 1.0)


# =========================================================================
# 8. The reader
# =========================================================================

from score_p9 import verdict as _p9_verdict  # noqa: E402


def _reading(mean_d, ci, frac_ours, frac_ref, sign_p):
    return {'mean_d': mean_d, 'ci_lo': ci[0], 'ci_hi': ci[1], 'ci_width': ci[1] - ci[0], 'ci95': list(ci),
            'frac_favour_ours': frac_ours, 'frac_favour_ref': frac_ref, 'sign_p': sign_p}


# Each world isolates ONE decision of score_p9.verdict / RESULTS 98.6 (PI, after W37).
WORLDS = {
    'pass': _reading(0.05, (0.03, 0.07), 0.80, 0.20, 1e-6),
    'pass_but_frac_59': _reading(0.05, (0.03, 0.07), 0.59, 0.41, 1e-6),          # only the 60 % conjunct fails
    'pass_but_sign_p': _reading(0.05, (0.03, 0.07), 0.80, 0.20, 0.02),          # only sign p fails
    'pass_but_ci_lo': _reading(0.05, (-0.001, 0.07), 0.80, 0.20, 1e-6),         # only the CI conjunct fails
    'ref_better': _reading(-0.05, (-0.07, -0.03), 0.20, 0.80, 1e-6),            # a ref win: mean <= 0
    'mean_neg_no_claim': _reading(-0.01, (-0.03, 0.01), 0.45, 0.55, 0.2),       # mean <= 0 without a ref win
    'mean_zero': _reading(0.0, (-0.02, 0.02), 0.5, 0.5, 1.0),                   # mean exactly 0 is <= 0
    'otherwise': _reading(0.01, (-0.01, 0.03), 0.58, 0.42, 0.05),
    'uninformative_pos': _reading(0.05, (0.0, 0.11), 0.80, 0.20, 1e-6),         # width 0.11 only
    'uninformative_neg': _reading(-0.05, (-0.11, 0.0), 0.20, 0.80, 1e-6),       # wide CI beats mean <= 0
}
EXPECT = {'pass': 'pass', 'pass_but_frac_59': 'otherwise', 'pass_but_sign_p': 'otherwise', 'pass_but_ci_lo': 'otherwise',
          'ref_better': 'mean_le_0', 'mean_neg_no_claim': 'mean_le_0', 'mean_zero': 'mean_le_0', 'otherwise': 'otherwise',
          'uninformative_pos': 'uninformative', 'uninformative_neg': 'uninformative'}


def _write_scored(path, ref, rec, extra=None, stored_verdict=None):
    rec = dict(rec)
    rec['verdict'] = stored_verdict if stored_verdict is not None else _p9_verdict(rec, ref)
    data = {'ref': ref, 'ref_label': ref, 'of_record': rec}
    data.update(extra or {})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    with open(str(path) + '.marker', 'w', encoding='utf-8') as f:
        json.dump({'complete': True, 'sha1': file_sha1(path)}, f)
    return path


@pytest.mark.parametrize('world', sorted(WORLDS))
def test_reader_r1_each_world(tmp_path, world):
    r1 = _write_scored(tmp_path / 'r1.json', 'mlp_m', WORLDS[world])
    res = score_p10.read_p10(r1)
    assert res['r1_case'] == EXPECT[world]
    assert res['r1'].endswith('(benchmark statement only)')
    assert res['r2'] == 'pending (B1)' and res['restate_p9_headline'] is None


@pytest.mark.parametrize('world', sorted(WORLDS))
def test_reader_r2_each_world_and_restate_only_on_mean_le_0(tmp_path, world):
    r1 = _write_scored(tmp_path / 'r1.json', 'mlp_m', WORLDS['pass'])
    r2 = _write_scored(tmp_path / 'r2.json', 'v9_drugblind', WORLDS[world])
    res = score_p10.read_p10(r1, r2)
    assert res['r2_case'] == EXPECT[world]
    assert res['restate_p9_headline'] is (EXPECT[world] == 'mean_le_0')
    want = {'pass': "v9 needs the compound on unseen compounds; Bai et al.'s finding does not hold for v9",
            'mean_le_0': "v9's unseen-compound accuracy does not depend on the compound",
            'otherwise': "not separable from v9's drug-blind twin at this power", 'uninformative': 'UNINFORMATIVE'}
    assert res['r2'] == want[EXPECT[world]]


def test_reader_r1_sentences_exact(tmp_path):
    for world, sentence in (('pass', 'v9 beats a drug-free model on unseen compounds'),
                            ('ref_better', 'v9 does not beat a drug-free model on unseen compounds'),
                            ('otherwise', 'v9 is not separable from a drug-free model on unseen compounds at this power')):
        r1 = _write_scored(tmp_path / ('r1_%s.json' % world), 'mlp_m', WORLDS[world])
        assert score_p10.read_p10(r1)['r1'] == sentence + ' (benchmark statement only)'


@pytest.mark.parametrize('slot, ref', [('r1', 'f0'), ('r1', 'mlp_mdt'), ('r1', 'v9_drugblind'), ('r2', 'mlp_m'), ('r2', 'f0')])
def test_reader_refuses_the_wrong_reference(tmp_path, slot, ref):
    good1 = _write_scored(tmp_path / 'g1.json', 'mlp_m', WORLDS['pass'])
    bad = _write_scored(tmp_path / 'bad.json', ref, WORLDS['pass'])
    with pytest.raises(SystemExit, match='not'):
        if slot == 'r1':
            score_p10.read_p10(bad)
        else:
            score_p10.read_p10(good1, bad)


def test_reader_refuses_a_stored_verdict_that_disagrees_with_its_numbers(tmp_path):
    r1 = _write_scored(tmp_path / 'r1.json', 'mlp_m', WORLDS['otherwise'],
                       stored_verdict='v9 predicts unseen compounds better than mlp_m')
    with pytest.raises(SystemExit, match='disagrees'):
        score_p10.read_p10(r1)


def test_reader_ignores_reported_only_garbage(tmp_path):
    """Only of_record is opened: garbage in every reported-only block changes nothing."""
    garbage = {'full': 'GARBAGE', 'per_seed_v9': None, 'row_pooled': 42, 'distinct_control_stratum': [None],
               'control_noise_free_target': {'verdict': 'v9 predicts unseen compounds better than mlp_m'},
               'cell_centred': False, 'subsets': {'clean': {'of_record': {'verdict': 'bogus'}}}}
    a = score_p10.read_p10(_write_scored(tmp_path / 'a.json', 'mlp_m', WORLDS['otherwise']))
    b = score_p10.read_p10(_write_scored(tmp_path / 'b.json', 'mlp_m', WORLDS['otherwise'], extra=garbage))
    assert a == b


def test_reader_wrong_marker_sha1_fails(tmp_path):
    r1 = _write_scored(tmp_path / 'r1.json', 'mlp_m', WORLDS['pass'])
    with open(str(r1) + '.marker', 'w', encoding='utf-8') as f:
        json.dump({'complete': True, 'sha1': '0' * 40}, f)
    with pytest.raises(SystemExit, match='marker'):
        score_p10.read_p10(r1)


def test_reader_missing_of_record_numbers_fails(tmp_path):
    rec = dict(WORLDS['pass'])
    del rec['sign_p']
    path = tmp_path / 'r1.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'ref': 'mlp_m', 'ref_label': 'mlp_m', 'of_record': dict(rec, verdict='x')}, f)
    with open(str(path) + '.marker', 'w', encoding='utf-8') as f:
        json.dump({'complete': True, 'sha1': file_sha1(path)}, f)
    with pytest.raises(SystemExit, match='incomplete'):
        score_p10.read_p10(path)


# =========================================================================
# 9. Touch-once
# =========================================================================

def test_touch_once_refuses_existing_file(tmp_path, monkeypatch):
    """An existing output JSON makes the scorer refuse."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    out_file = tmp_path / 'model' / 'results' / 'p10_accuracy_mlp_m.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text('{}', encoding='utf-8')

    with pytest.raises(SystemExit, match='touched once'):
        score_p10.score_p10(p9_dir=str(fix['p9_dir']), ref='mlp_m', ref_dir=str(fix['ref_dir_m']))


# =========================================================================
# Additional coverage: mlp_mdt, v9_drugblind, f0, and CLI
# =========================================================================

def test_mlp_mdt_scoring_and_guards(tmp_path, monkeypatch):
    """mlp_mdt: arm == 'mdt' scored as reported-only, wrong arm fails."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    ref_dir_mdt = tmp_path / 'ref_dir_mdt'
    ref_dir_mdt.mkdir(parents=True, exist_ok=True)

    mlp_seeds = [131419, 131420, 131421]
    for idx, s in enumerate(mlp_seeds):
        np.savez(
            ref_dir_mdt / f'p10_mdt_seed{idx}.npz',
            row_index=fix['test_row_index'],
            delta_pred=fix['y_test_delta'].astype(np.float32),
            seed=np.int64(s),
            arm=np.str_('mdt'),
        )

    out = score_p10.score_p10(p9_dir=str(fix['p9_dir']), ref='mlp_mdt', ref_dir=str(ref_dir_mdt))
    assert out['LABEL'] == 'reported, never read (RESULTS 98.6)'
    assert out['of_record']['LABEL'] == 'reported, never read (RESULTS 98.6)'

    # Wrong arm for mdt fails
    np.savez(
        ref_dir_mdt / 'p10_mdt_seed0.npz',
        row_index=fix['test_row_index'],
        delta_pred=fix['y_test_delta'].astype(np.float32),
        seed=np.int64(131419),
        arm=np.str_('m'),
    )
    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])
    with pytest.raises(SystemExit, match='arm'):
        score_p10.load_ref('mlp_mdt', str(ref_dir_mdt), p9_data)


def test_v9_drugblind_scoring_and_guards(tmp_path, monkeypatch):
    """v9_drugblind: manifest checked, 1e-4 tolerance on y_true/ctl_true, R2 label."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    ref_dir_b1 = tmp_path / 'ref_dir_b1'
    ref_dir_b1.mkdir(parents=True, exist_ok=True)

    manifest_files = {}
    for s in range(3):
        fname = f'v9p10b1_seed{s}.npz'
        fpath = ref_dir_b1 / fname
        np.savez(
            fpath,
            row_index=fix['test_row_index'],
            y_pred=fix['y_test'],
            y_true=fix['y_test'],
            ctl_true=fix['ctl_test'],
        )
        manifest_files[fname] = file_sha1(fpath)

    mf_path = ref_dir_b1 / 'P10B1_COMPLETE.json'
    with open(mf_path, 'w', encoding='utf-8') as f:
        json.dump({'complete': True, 'n_seeds': 3, 'files': manifest_files}, f, indent=2)

    out = score_p10.score_p10(p9_dir=str(fix['p9_dir']), ref='v9_drugblind', ref_dir=str(ref_dir_b1))
    assert out['of_record']['LABEL'] == 'R2: carries the compound-use inference (98.6 C1)'

    # y_true mismatch > 1e-4 in all seeds fails
    for s in range(3):
        fname = f'v9p10b1_seed{s}.npz'
        fpath = ref_dir_b1 / fname
        np.savez(
            fpath,
            row_index=fix['test_row_index'],
            y_pred=fix['y_test'],
            y_true=fix['y_test'] + 0.01,
            ctl_true=fix['ctl_test'],
        )
        manifest_files[fname] = file_sha1(fpath)
    with open(mf_path, 'w', encoding='utf-8') as f:
        json.dump({'complete': True, 'n_seeds': 3, 'files': manifest_files}, f, indent=2)

    p9_data = score_p10.load_p9_and_metadata(fix['p9_dir'])
    with pytest.raises(SystemExit, match='differs from P9'):
        score_p10.load_ref('v9_drugblind', str(ref_dir_b1), p9_data)


def test_f0_end_to_end_scoring(tmp_path, monkeypatch):
    """f0: scores directly against bundle, reported-only label."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    out = score_p10.score_p10(p9_dir=str(fix['p9_dir']), ref='f0')
    assert out['LABEL'] == 'reported, never read (RESULTS 98.6)'
    assert out['of_record']['LABEL'] == 'reported, never read (RESULTS 98.6)'


def test_cli_dry_check_and_read(tmp_path, monkeypatch, capsys):
    """CLI dry_check passes and --read prints parsed 2x2 output."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    # CLI dry_check
    score_p10.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'mlp_m', '--ref_dir', str(fix['ref_dir_m']), '--dry_check'])
    captured = capsys.readouterr()
    assert 'P10 inputs verified' in captured.out

    # Run score to produce an output file
    score_p10.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'mlp_m', '--ref_dir', str(fix['ref_dir_m'])])
    out_file = tmp_path / 'model' / 'results' / 'p10_accuracy_mlp_m.json'
    assert out_file.exists()
    assert (tmp_path / 'model' / 'results' / 'p10_accuracy_mlp_m.json.marker').exists()

    # CLI read
    score_p10.main(['--read', str(out_file)])
    read_out = capsys.readouterr()
    parsed = json.loads(read_out.out)
    assert 'r1' in parsed
    assert parsed['r2'] == 'pending (B1)'



# =========================================================================
# PI additions after the mutation check (6 of 23 mutants survived; 2 are equivalent: f0's no-training-cell rows are
# already NaN, and a verdict string cannot be both UNINFORMATIVE and a pass). Each test isolates ONE decision.
# =========================================================================
def _b1_dir(tmp_path, fix, ctl_shift=0.0):
    d = tmp_path / 'b1_iso'
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in range(3):
        fp = d / f'v9p10b1_seed{s}.npz'
        np.savez(fp, row_index=fix['test_row_index'], y_pred=fix['y_test'], y_true=fix['y_test'],
                 ctl_true=fix['ctl_test'] + ctl_shift)
        files[fp.name] = file_sha1(fp)
    with open(d / 'P10B1_COMPLETE.json', 'w', encoding='utf-8') as f:
        json.dump({'complete': True, 'n_seeds': 3, 'files': files}, f)
    return d


def test_pi_b1_ctl_check_isolated(tmp_path, monkeypatch):
    """B1's ctl_true must equal P9's: y_true intact, ctl_true off by 0.01 -> FATAL naming ctl_true."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    p9 = score_p10.load_p9_and_metadata(fix['p9_dir'])
    score_p10.load_ref('v9_drugblind', str(_b1_dir(tmp_path, fix, 0.0)), p9)          # the clean world passes
    with pytest.raises(SystemExit, match='ctl_true differs'):
        score_p10.load_ref('v9_drugblind', str(_b1_dir(tmp_path, fix, 0.01)), p9)


def test_pi_f0_cell_means_use_training_rows_only():
    """x_bar_ctl(c) and delta_bar(c) are means over the cell's TRAINING rows: huge test-row values must not move them."""
    rng = np.random.default_rng(3)
    ctl = rng.normal(size=(6, 5))
    x = ctl + rng.normal(size=(6, 5))
    ctl[4:] = 1e6
    x[4:] = -1e6
    bundle = {'X': x, 'X_ctl': ctl, 'meta_cell': np.array(['A'] * 6),
              score_p10.SPLIT_KEY: np.array(['train'] * 4 + ['test'] * 2)}
    _, _, mc, md, _ = score_p10.compute_f0(bundle, np.array(['A', 'A']), ctl[4:])
    assert np.allclose(mc['A'], ctl[:4].mean(0))
    assert np.allclose(md['A'], (x - ctl)[:4].mean(0))


def test_pi_dropped_fraction_cap(tmp_path, monkeypatch):
    """More than MAX_DROPPED_FRAC (1 %) non-finite rows is fatal: one constant M row of 60 is 1.7 %."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    for k in range(3):
        fp = fix['ref_dir_m'] / f'p10_m_seed{k}.npz'
        z = dict(np.load(fp, allow_pickle=True))
        z['delta_pred'] = np.array(z['delta_pred'], copy=True)
        z['delta_pred'][0] = 0.0                                             # a constant row: Pearson undefined
        np.savez(fp, **z)
    with pytest.raises(SystemExit, match='MAX_DROPPED_FRAC'):
        score_p10.score_p10(p9_dir=str(fix['p9_dir']), ref='mlp_m', ref_dir=str(fix['ref_dir_m']), n_boot=50)


def test_pi_of_record_is_the_clean_subset(tmp_path, monkeypatch):
    """The reading of record is on the clean rows (5 molecules); 'full' adds the dirty one (6)."""
    fix = make_p10_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    out = score_p10.score_p10(p9_dir=str(fix['p9_dir']), ref='mlp_m', ref_dir=str(fix['ref_dir_m']), n_boot=50)
    assert out['of_record']['n_molecules'] == 5
    assert out['full']['n_molecules'] == 6
    assert out['row_pooled']['n_rows'] == len(fix['clean_rows'])
