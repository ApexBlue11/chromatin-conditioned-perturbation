# -*- coding: utf-8 -*-
"""Tests for model/v9/score_p9.py — synthetic fixtures only (RESULTS §96.3 / §96.7)."""
import hashlib
import json
import os
import re
import sys
import numpy as np
import pytest
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_p9


def file_sha1(path):
    return hashlib.sha1(open(path, 'rb').read()).hexdigest()


def make_synthetic_fixture(tmp_path, n_rows=60, n_clean_rows=50, n_genes=10, seed=42):
    """Builds a complete synthetic fixture in tmp_path with valid inputs satisfying all guards."""
    rng = np.random.default_rng(seed)
    p9_dir = tmp_path / 'p9_dir'
    p9_dir.mkdir(parents=True, exist_ok=True)

    row_index = np.arange(100, 100 + n_rows, dtype=np.int64)

    # 5 clean perts and 1 dirty pert
    # Clean rows: 0..49 (10 rows each for PERT_0..PERT_4)
    # Dirty rows: 50..59 (10 rows for PERT_DIRTY)
    perts = []
    cells = []
    cell_cycle = ['A375', 'MCF7', 'PC3']
    for i in range(n_rows):
        cells.append(cell_cycle[i % len(cell_cycle)])
        if i < n_clean_rows:
            pert_num = i // (n_clean_rows // 5)
            perts.append(f'PERT_{pert_num}')
        else:
            perts.append('PERT_DIRTY')
    perts = np.array(perts)
    cells = np.array(cells)

    # Bundle
    bundle_path = tmp_path / 'bundle.npz'
    np.savez(
        bundle_path,
        row_index=row_index,
        meta_pert_id=perts,
        meta_cell=cells,
        split_split_cold_drug_1=np.array(['test'] * n_rows),
    )

    # Truth & Controls
    ctl_true = rng.normal(0, 1, size=(n_rows, n_genes))
    y = rng.normal(0, 1, size=(n_rows, n_genes))
    y_true = ctl_true + y

    # Seeds and snapshots for V2 stack
    seed_files = []
    snap_files = []
    last_files = []
    manifest_files = {}

    for s in range(3):
        # 3 snapshots per seed
        snaps = []
        for sn in range(3):
            snap_ypred = ctl_true + y + rng.normal(0, 1.5, size=(n_rows, n_genes))
            snap_degpred = snap_ypred - ctl_true
            s_name = f'v9p9_seed{s}_snap{sn}.npz'
            s_path = p9_dir / s_name
            np.savez(
                s_path,
                row_index=row_index,
                y_pred=snap_ypred,
                deg_pred=snap_degpred,
                y_true=y_true,
                ctl_true=ctl_true,
            )
            manifest_files[s_name] = file_sha1(s_path)
            snaps.append(snap_ypred)
            snap_files.append(s_name)

        # main seed is mean of snaps
        main_ypred = np.mean(snaps, axis=0)
        main_degpred = main_ypred - ctl_true
        m_name = f'v9p9_seed{s}.npz'
        m_path = p9_dir / m_name
        np.savez(
            m_path,
            row_index=row_index,
            y_pred=main_ypred,
            deg_pred=main_degpred,
            y_true=y_true,
            ctl_true=ctl_true,
        )
        manifest_files[m_name] = file_sha1(m_path)
        seed_files.append(m_name)

        # _last is identical to _snap2
        l_name = f'v9p9_seed{s}_last.npz'
        l_path = p9_dir / l_name
        np.savez(
            l_path,
            row_index=row_index,
            y_pred=snaps[-1].copy(),
            deg_pred=(snaps[-1] - ctl_true).copy(),
            y_true=y_true,
            ctl_true=ctl_true,
        )
        manifest_files[l_name] = file_sha1(l_path)
        last_files.append(l_name)

    # Manifest
    manifest_path = p9_dir / 'P9_COMPLETE.json'
    manifest_data = {
        'complete': True,
        'n_seeds': 3,
        'stack': 'V2',
        'files': manifest_files,
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)

    # Reference
    ref_path = tmp_path / 'ridge.npz'
    ref_pred = ctl_true + y + rng.normal(0, 2.0, size=(n_rows, n_genes))
    np.savez(
        ref_path,
        row_index=row_index,
        y_true=y_true,
        ctl_true=ctl_true,
        ridge_pred=ref_pred,
    )

    # Units JSON
    # Map PERT_0..PERT_4 to MOL_0..MOL_4, and PERT_DIRTY to MOL_DIRTY
    pert_to_molecule = {f'PERT_{i}': f'MOL_{i}' for i in range(5)}
    pert_to_molecule['PERT_DIRTY'] = 'MOL_DIRTY'
    molecule_stratum = {
        'MOL_0': 'lt_0.6',
        'MOL_1': '0.6_0.8',
        'MOL_2': '0.8_0.999',
        'MOL_3': 'lt_0.6',
        'MOL_4': '0.6_0.8',
        'MOL_DIRTY': 'ge_0.999',
    }
    units_data = {
        'pert_to_molecule': pert_to_molecule,
        'molecule_stratum': molecule_stratum,
    }
    units_path = tmp_path / 'units.json'
    with open(units_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, indent=2)

    # Clean JSON
    clean_data = {
        'dirty_compounds': ['PERT_DIRTY'],
        'clean_scored_rows': n_clean_rows,
        'clean_compounds': 5,
    }
    clean_path = tmp_path / 'clean.json'
    with open(clean_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2)

    return {
        'p9_dir': p9_dir,
        'bundle_path': bundle_path,
        'ref_path': ref_path,
        'units_path': units_path,
        'clean_path': clean_path,
        'row_index': row_index,
        'clean_rows': row_index[:n_clean_rows],
        'manifest_path': manifest_path,
        'y_true': y_true,
        'ctl_true': ctl_true,
    }


def patch_constants(monkeypatch, fix, tmp_path):
    """Monkeypatches module-level constants in score_p9 to point to the fixture."""
    monkeypatch.setattr(score_p9, 'REPO', str(tmp_path))
    monkeypatch.setattr(score_p9, 'BUNDLE', str(fix['bundle_path']))
    monkeypatch.setattr(score_p9, 'UNITS_JSON', str(fix['units_path']))
    monkeypatch.setattr(score_p9, 'UNITS_SHA1', file_sha1(fix['units_path']))
    monkeypatch.setattr(score_p9, 'CLEAN_JSON', str(fix['clean_path']))
    monkeypatch.setattr(score_p9, 'CLEAN_SHA1', file_sha1(fix['clean_path']))
    monkeypatch.setattr(score_p9, 'EXPECT_N_FULL', len(fix['row_index']))
    monkeypatch.setattr(score_p9, 'EXPECT_SHA1_FULL', score_p9.row_set_sha1(fix['row_index']))
    monkeypatch.setattr(score_p9, 'EXPECT_N_CLEAN', len(fix['clean_rows']))
    monkeypatch.setattr(score_p9, 'EXPECT_SHA1_CLEAN', score_p9.row_set_sha1(fix['clean_rows']))
    monkeypatch.setattr(score_p9, 'REFS', {
        'ridge': {
            'path': str(fix['ref_path']),
            'key': 'ridge_pred',
            'minus_ctl': True,
            'label': 'ridge (bundle: control + ECFP4 + descriptors)',
        }
    })


# =========================================================================
# 1. The verdicts, on molecule_reading + verdict directly
# =========================================================================

def test_verdict_a_v9_better():
    """(a) d_m mostly +0.05 with small noise -> v9 better."""
    rng = np.random.default_rng(42)
    k = 300
    molecules = np.array([f'M_{i}' for i in range(k)])
    diff = 0.05 + rng.normal(0, 0.005, size=k)
    reading = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    v = score_p9.verdict(reading)
    assert v == 'v9 predicts unseen compounds better than R'


def test_verdict_b_r_better():
    """(b) mirror of (a) -> R better."""
    rng = np.random.default_rng(42)
    k = 300
    molecules = np.array([f'M_{i}' for i in range(k)])
    diff = -0.05 + rng.normal(0, 0.005, size=k)
    reading = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    v = score_p9.verdict(reading)
    assert v == 'R predicts unseen compounds better than v9'


def test_verdict_c_symmetric_noise():
    """(c) symmetric noise around 0 -> NO CLAIM."""
    rng = np.random.default_rng(42)
    k = 300
    molecules = np.array([f'M_{i}' for i in range(k)])
    diff = rng.normal(0, 0.02, size=k)
    reading = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    v = score_p9.verdict(reading)
    assert v == 'NO COMPOUND-LEVEL CLAIM'


def test_verdict_d_sixty_pct_conjunct():
    """(d) 55% at +0.08 and 45% at -0.02: mean > 0 and CI excludes 0, but frac < 0.60 -> NO CLAIM."""
    k = 300
    molecules = np.array([f'M_{i}' for i in range(k)])
    # 165 positive (55%), 135 negative (45%)
    diff = np.array([0.08] * 165 + [-0.02] * 135)
    reading = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    assert reading['mean_d'] > 0
    assert reading['ci_lo'] > 0
    assert reading['ci_width'] <= 0.10
    assert reading['frac_favour_ours'] == 0.55
    v = score_p9.verdict(reading)
    assert v == 'NO COMPOUND-LEVEL CLAIM'


def test_verdict_e_sign_conjunct():
    """(e) 25 molecules, 16 favour v9 with frac 0.64, but two-sided sign p > 0.01 -> NO CLAIM."""
    k = 25
    molecules = np.array([f'M_{i}' for i in range(k)])
    # 16 positive, 9 negative; small variance to keep CI width <= 0.10
    diff = np.array([0.05] * 16 + [-0.01] * 9)
    reading = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    assert reading['frac_favour_ours'] == 0.64
    assert reading['sign_p'] > 0.01
    assert reading['ci_lo'] > 0
    assert reading['ci_width'] <= 0.10
    v = score_p9.verdict(reading)
    assert v == 'NO COMPOUND-LEVEL CLAIM'


def test_verdict_f_ci_width_rule():
    """(f) ~8 molecules with huge variance -> UNINFORMATIVE, even if means positive."""
    molecules = np.array([f'M_{i}' for i in range(8)])
    diff = np.array([0.5, -0.4, 0.6, -0.5, 0.7, -0.3, 0.8, -0.2])
    reading = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    assert reading['ci_width'] > 0.10
    v = score_p9.verdict(reading)
    assert v == 'UNINFORMATIVE: cluster CI wider than 0.10; no compound-level claim'


# =========================================================================
# 2. The molecule collapse
# =========================================================================

def test_molecule_collapse():
    """Two pert_ids map to one molecule: median of union != mean of per-pert medians.

    Changing map (splitting them) changes n_molecules and d.
    """
    # Pert 1 has 3 rows: [1.0, 1.0, 1.0] -> median 1.0
    # Pert 2 has 5 rows: [10.0, 10.0, 10.0, 10.0, 10.0] -> median 10.0
    # Union of 8 rows has median 10.0
    # Mean of two per-pert medians would be (1.0 + 10.0)/2 = 5.5
    diff = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    r_ref = np.zeros_like(diff)

    # 1 molecule
    mol_single = np.array(['MOL_COMBINED'] * 8)
    res_single = score_p9.molecule_reading(diff, r_ref, mol_single, n_boot=1000, seed=0)
    assert res_single['n_molecules'] == 1
    assert res_single['d_m']['MOL_COMBINED'] == 10.0
    assert res_single['mean_d'] == 10.0

    # 2 molecules (split)
    mol_split = np.array(['MOL_A'] * 3 + ['MOL_B'] * 5)
    res_split = score_p9.molecule_reading(diff, r_ref, mol_split, n_boot=1000, seed=0)
    assert res_split['n_molecules'] == 2
    assert res_split['d_m']['MOL_A'] == 1.0
    assert res_split['d_m']['MOL_B'] == 10.0
    assert res_split['mean_d'] == 5.5


# =========================================================================
# 3. Score averaging
# =========================================================================

def test_score_averaging(tmp_path, monkeypatch):
    """With independent noise per seed, r_v9 == mean of per-seed Pearsons (to 1e-12).

    Assert it differs from Pearson of mean prediction by > 1e-3. Read through main().
    """
    fix = make_synthetic_fixture(tmp_path, seed=123)
    patch_constants(monkeypatch, fix, tmp_path)

    score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])

    out_file = tmp_path / 'model' / 'results' / 'p9_accuracy_ridge.json'
    assert out_file.exists()
    with open(out_file, encoding='utf-8') as f:
        out = json.load(f)

    clean_rp = out['subsets']['clean']['row_pooled']
    mean_r_v9 = clean_rp['mean_r_v9']
    per_seed_means = clean_rp['per_seed_means']
    mean_of_per_seed_means = np.mean(per_seed_means)

    # Score averaging identity (to 1e-12)
    assert abs(mean_r_v9 - mean_of_per_seed_means) < 1e-12

    # Differs from Pearson of mean prediction by > 1e-3
    mean_r_ens = clean_rp['mean_r_ens']
    assert abs(mean_r_ens - mean_r_v9) > 1e-3


# =========================================================================
# 4. The bootstrap
# =========================================================================

def test_bootstrap():
    """Bootstrap is deterministic, contains true planted mean, changes with seed."""
    rng = np.random.default_rng(999)
    k = 300
    molecules = np.array([f'M_{i}' for i in range(k)])
    diff = 0.03 + rng.normal(0, 0.02, size=k)

    r1 = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    r2 = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=0)
    assert r1['ci95'] == r2['ci95']

    # Planted true mean is inside CI
    assert r1['ci_lo'] <= 0.03 <= r1['ci_hi']

    # Changing seed changes endpoints
    r3 = score_p9.molecule_reading(diff, np.zeros_like(diff), molecules, n_boot=20000, seed=1)
    assert r1['ci95'] != r3['ci95']


# =========================================================================
# 5. The refusals through main()
# =========================================================================

def test_refusal_no_manifest(tmp_path, monkeypatch):
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    os.remove(fix['manifest_path'])

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'FATAL: no P9_COMPLETE.json' in str(exc.value)


def test_refusal_sha1_mismatch(tmp_path, monkeypatch):
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    # Corrupt seed0 file without updating manifest
    seed0_path = fix['p9_dir'] / 'v9p9_seed0.npz'
    with open(seed0_path, 'ab') as f:
        f.write(b'extra_bytes')

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'missing or sha1 differs from the manifest' in str(exc.value)


def test_refusal_output_already_exists(tmp_path, monkeypatch):
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)
    out_dir = tmp_path / 'model' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'p9_accuracy_ridge.json'
    out_file.write_text('{}')

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'exists -- touched once' in str(exc.value)


def test_refusal_p9_row_count_mismatch(tmp_path, monkeypatch):
    """Drop one row from all seeds and re-hash manifest -> row count != EXPECT_N_FULL."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    manifest = json.load(open(fix['manifest_path']))
    for s_name in manifest['files']:
        p = fix['p9_dir'] / s_name
        z = np.load(p)
        new_data = {k: z[k][:-1] for k in z.files}
        np.savez(p, **new_data)
        manifest['files'][s_name] = file_sha1(p)
    with open(fix['manifest_path'], 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'P9 row count' in str(exc.value)


def test_refusal_row_set_sha1_mismatch(tmp_path, monkeypatch):
    """Change one row index value in all seeds -> row-set sha1 != EXPECT_SHA1_FULL."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    manifest = json.load(open(fix['manifest_path']))
    for s_name in manifest['files']:
        p = fix['p9_dir'] / s_name
        z = np.load(p)
        new_row_index = z['row_index'].copy()
        new_row_index[0] = 999999
        new_data = {k: z[k] for k in z.files}
        new_data['row_index'] = new_row_index
        np.savez(p, **new_data)
        manifest['files'][s_name] = file_sha1(p)
    with open(fix['manifest_path'], 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'P9 row-set sha1' in str(exc.value)


def test_refusal_seed_files_disagree(tmp_path, monkeypatch):
    """A seed with a different row_index order or y_true."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    manifest = json.load(open(fix['manifest_path']))
    p = fix['p9_dir'] / 'v9p9_seed1.npz'
    z = np.load(p)
    new_data = {k: z[k] for k in z.files}
    new_data['y_true'] = new_data['y_true'] + 1.0
    np.savez(p, **new_data)
    manifest['files']['v9p9_seed1.npz'] = file_sha1(p)
    with open(fix['manifest_path'], 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'seed files disagree on' in str(exc.value)


def test_refusal_reference_y_true_perturbed(tmp_path, monkeypatch):
    """Reference y_true perturbed by 1e-3."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    r = np.load(fix['ref_path'])
    new_data = {k: r[k] for k in r.files}
    new_data['y_true'] = new_data['y_true'].copy()
    new_data['y_true'][0, 0] += 1e-3
    np.savez(fix['ref_path'], **new_data)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'reference y_true differs from P9' in str(exc.value)


def test_refusal_p9_row_missing_from_reference(tmp_path, monkeypatch):
    """A P9 row missing from reference."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    r = np.load(fix['ref_path'])
    new_data = {k: r[k][:-1] for k in r.files}
    np.savez(fix['ref_path'], **new_data)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'reference is missing' in str(exc.value)


def test_refusal_units_sha1_mismatch(tmp_path, monkeypatch):
    """Units JSON sha1 mismatch."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    with open(fix['units_path'], 'a', encoding='utf-8') as f:
        f.write(' ')

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'units JSON sha1 differs from UNITS_SHA1' in str(exc.value)


def test_refusal_pert_missing_from_units(tmp_path, monkeypatch):
    """A pert missing from pert_to_molecule."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    with open(fix['units_path'], encoding='utf-8') as f:
        u = json.load(f)
    del u['pert_to_molecule']['PERT_0']
    with open(fix['units_path'], 'w', encoding='utf-8') as f:
        json.dump(u, f, indent=2)
    monkeypatch.setattr(score_p9, 'UNITS_SHA1', file_sha1(fix['units_path']))

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'pert_ids missing from pert_to_molecule' in str(exc.value)


def test_refusal_clean_count_mismatch(tmp_path, monkeypatch):
    """Clean count != expected."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    with open(fix['clean_path'], encoding='utf-8') as f:
        c = json.load(f)
    # Empty dirty compounds -> clean count becomes 60 != EXPECT_N_CLEAN (50)
    c['dirty_compounds'] = []
    with open(fix['clean_path'], 'w', encoding='utf-8') as f:
        json.dump(c, f, indent=2)
    monkeypatch.setattr(score_p9, 'CLEAN_SHA1', file_sha1(fix['clean_path']))

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'clean count' in str(exc.value)


def test_refusal_train_row_in_bundle(tmp_path, monkeypatch):
    """A P9 row labelled 'train' in the bundle."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    b = np.load(fix['bundle_path'])
    new_data = {k: b[k] for k in b.files}
    new_split = new_data['split_split_cold_drug_1'].copy()
    new_split[0] = 'train'
    new_data['split_split_cold_drug_1'] = new_split
    np.savez(fix['bundle_path'], **new_data)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'not all P9 rows are "test"' in str(exc.value)


def test_refusal_snapshot_identity_violation(tmp_path, monkeypatch):
    """_last != _snap2 (the snapshot identity check)."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    manifest = json.load(open(fix['manifest_path']))
    p = fix['p9_dir'] / 'v9p9_seed0_last.npz'
    z = np.load(p)
    new_data = {k: z[k] for k in z.files}
    new_data['y_pred'] = new_data['y_pred'] + 0.1
    np.savez(p, **new_data)
    manifest['files']['v9p9_seed0_last.npz'] = file_sha1(p)
    with open(fix['manifest_path'], 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert '_last' in str(exc.value) and 'differs from _snap2' in str(exc.value)


def test_refusal_non_finite_rows_exceed_cap(tmp_path, monkeypatch):
    """More than 1% non-finite rows."""
    fix = make_synthetic_fixture(tmp_path)
    patch_constants(monkeypatch, fix, tmp_path)

    manifest = json.load(open(fix['manifest_path']))
    # For seed 0, put NaN into 2 rows (2/60 = 3.3% > 1%)
    for f_name in ('v9p9_seed0.npz', 'v9p9_seed0_snap0.npz', 'v9p9_seed0_snap1.npz', 'v9p9_seed0_snap2.npz', 'v9p9_seed0_last.npz'):
        p = fix['p9_dir'] / f_name
        z = np.load(p)
        new_data = {k: z[k] for k in z.files}
        new_ypred = new_data['y_pred'].copy()
        # To avoid snapshot check non-finite error, put NaN into reference instead!
    
    # Put NaN into reference ridge_pred on 2 rows
    r = np.load(fix['ref_path'])
    new_data = {k: r[k] for k in r.files}
    new_pred = new_data['ridge_pred'].copy()
    new_pred[0:2] = np.nan
    new_data['ridge_pred'] = new_pred
    np.savez(fix['ref_path'], **new_data)

    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'non-finite scores' in str(exc.value)


# =========================================================================
# 6. End to end
# =========================================================================

def test_end_to_end(tmp_path, monkeypatch):
    """main() writes JSON and matching marker; clean molecule count equals fixture;

    full includes dirty pert and clean does not; per_stratum sums to n_molecules;
    second run refuses touch-once.
    """
    fix = make_synthetic_fixture(tmp_path, seed=42)
    patch_constants(monkeypatch, fix, tmp_path)

    score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])

    out_file = tmp_path / 'model' / 'results' / 'p9_accuracy_ridge.json'
    marker_file = tmp_path / 'model' / 'results' / 'p9_accuracy_ridge.json.marker'
    assert out_file.exists()
    assert marker_file.exists()

    # Marker sha1 matches
    marker = json.load(open(marker_file))
    assert marker['complete'] is True
    assert marker['file'] == 'p9_accuracy_ridge.json'
    assert marker['sha1'] == file_sha1(out_file)

    with open(out_file, encoding='utf-8') as f:
        res = json.load(f)

    # subsets.clean.of_record.n_molecules equals clean molecule count (5)
    assert res['subsets']['clean']['of_record']['n_molecules'] == 5
    assert res['subsets']['clean']['n_molecules'] == 5

    # Full subset includes dirty pert's rows (60 rows) and clean subset does not (50 rows)
    assert res['subsets']['full']['n_rows'] == 60
    assert res['subsets']['clean']['n_rows'] == 50
    assert res['subsets']['full']['n_molecules'] == 6

    # per_stratum n_molecules sum to n_molecules
    for sub_name in ('clean', 'full'):
        sub = res['subsets'][sub_name]
        total_mols = sub['n_molecules']
        stratum_sum = sum(st['n_molecules'] for st in sub['per_stratum'].values())
        assert stratum_sum == total_mols

    # Touch-once rule on second run
    with pytest.raises(SystemExit) as exc:
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    assert 'exists -- touched once' in str(exc.value)


# =========================================================================
# 7. --dry_check writes nothing
# =========================================================================

def test_dry_check_writes_nothing(tmp_path, monkeypatch, capsys):
    """--dry_check runs guards and stops, writing nothing."""
    fix = make_synthetic_fixture(tmp_path, seed=42)
    patch_constants(monkeypatch, fix, tmp_path)

    score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge', '--dry_check'])

    out_file = tmp_path / 'model' / 'results' / 'p9_accuracy_ridge.json'
    marker_file = tmp_path / 'model' / 'results' / 'p9_accuracy_ridge.json.marker'
    assert not out_file.exists()
    assert not marker_file.exists()

    captured = capsys.readouterr()
    assert 'P9 inputs verified' in captured.out


# =========================================================================
# PI additions (mutation check of W34's suite: 7 of 15 mutants survived; each test below kills one or more)
# =========================================================================

def _reading_from_d(d, seed=0):
    """One row per molecule, so d_m == d exactly."""
    d = np.asarray(d, dtype=np.float64)
    return score_p9.molecule_reading(d, np.zeros_like(d), np.array(['M%05d' % i for i in range(len(d))]), 2000, seed)


def test_the_sixty_percent_conjunct_decides_alone_on_both_sides():
    """57 % favour with a tiny sign p, a positive mean and a CI excluding 0: only FRAC_MIN can refuse the claim."""
    d = np.r_[np.full(1140, 0.02), np.full(860, -0.01)]
    r = _reading_from_d(d)
    assert r['sign_p'] < 1e-6 and r['ci_lo'] > 0 and r['ci_width'] < 0.1 and abs(r['frac_favour_ours'] - 0.57) < 1e-12
    assert score_p9.verdict(r) == 'NO COMPOUND-LEVEL CLAIM'
    rm = _reading_from_d(-d)
    assert rm['ci_hi'] < 0 and rm['sign_p'] < 1e-6
    assert score_p9.verdict(rm) == 'NO COMPOUND-LEVEL CLAIM'


def test_the_ci_conjunct_decides_alone_on_both_sides():
    """65 % favour and a tiny sign p, but the mean is ~0 and its CI straddles 0: only ci_lo / ci_hi can refuse."""
    d = np.r_[np.full(260, 0.010), np.full(140, -0.0185)]
    r = _reading_from_d(d)
    assert r['mean_d'] > 0 and r['ci_lo'] < 0 < r['ci_hi'] and r['frac_favour_ours'] >= 0.6 and r['sign_p'] < 1e-6
    assert score_p9.verdict(r) == 'NO COMPOUND-LEVEL CLAIM'
    rm = _reading_from_d(-d)
    assert rm['mean_d'] < 0 and rm['ci_lo'] < 0 < rm['ci_hi'] and rm['frac_favour_ref'] >= 0.6
    assert score_p9.verdict(rm) == 'NO COMPOUND-LEVEL CLAIM'


def test_ties_count_in_the_sixty_percent_denominator():
    """500 exact ties, 400 favour v9, 100 favour R: frac_favour_ours is 0.4 of ALL molecules, not 0.8 of the non-ties."""
    d = np.r_[np.zeros(500), np.full(400, 0.02), np.full(100, -0.01)]
    r = _reading_from_d(d)
    assert r['n_tie'] == 500 and r['frac_favour_ours'] == 0.4 and r['ci_lo'] > 0 and r['sign_p'] < 1e-6
    assert score_p9.verdict(r) == 'NO COMPOUND-LEVEL CLAIM'


def test_the_cluster_bootstrap_resamples_molecules():
    """The CI is exactly the percentile CI of means of molecule resamples; 10 molecules x 100 rows with spread-out d_m is
    UNINFORMATIVE (a row bootstrap would call it narrow)."""
    rng = np.random.default_rng(5)
    dm = rng.normal(0.05, 0.1, size=10)
    mol = np.repeat(['M%d' % i for i in range(10)], 100)
    diff = np.repeat(dm, 100) + rng.normal(0, 1e-4, size=1000)
    r = score_p9.molecule_reading(diff, np.zeros(1000), mol, 4000, seed=3)
    d_sorted = np.array([np.median(diff[mol == 'M%d' % i]) for i in range(10)])
    idx = np.random.default_rng(3).integers(0, 10, size=(4000, 10))
    want = np.percentile(d_sorted[idx].mean(1), [2.5, 97.5])
    assert abs(r['ci_lo'] - want[0]) < 1e-12 and abs(r['ci_hi'] - want[1]) < 1e-12
    assert r['ci_width'] > 0.1 and score_p9.verdict(r).startswith('UNINFORMATIVE')


def test_the_reference_score_uses_its_prediction_minus_control(tmp_path, monkeypatch):
    """mean_r_ref must equal the mean per-row Pearson of (ridge_pred - ctl_true, y_true - ctl_true) on the clean rows."""
    fix = make_synthetic_fixture(tmp_path, seed=7)
    patch_constants(monkeypatch, fix, tmp_path)
    score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    out = json.load(open(tmp_path / 'model' / 'results' / 'p9_accuracy_ridge.json', encoding='utf-8'))
    z = np.load(fix['ref_path'])
    y = fix['y_true'] - fix['ctl_true']
    r_raw = score_p9.per_row_pearson(z['ridge_pred'] - z['ctl_true'], y)[:50]
    r_wrong = score_p9.per_row_pearson(z['ridge_pred'], y)[:50]
    assert abs(out['subsets']['clean']['row_pooled']['mean_r_ref'] - float(np.mean(r_raw))) < 1e-9
    assert abs(float(np.mean(r_raw)) - float(np.mean(r_wrong))) > 1e-3       # the fixture can tell the two apart


def test_duplicate_lift_is_paired_and_combinable():
    """97.6 item 5: a model scoring +0.2 on duplicate rows only has lift 0.2; the DiD against a flat model is 0.2 with a CI
    excluding 0; a molecule with both kinds of rows refuses; per-molecule sums reproduce the lift."""
    rng = np.random.default_rng(0)
    mol = np.repeat(['D%d' % i for i in range(20)] + ['C%d' % i for i in range(80)], 5)
    is_dup = np.array([m.startswith('D') for m in mol])
    base = rng.normal(0.5, 0.05, size=len(mol))
    out = score_p9.duplicate_lift({'xpert': base + 0.2 * is_dup, 'v9': base}, mol, is_dup, 2000, seed=0)
    assert abs(out['models']['xpert']['lift'] - out['models']['v9']['lift'] - 0.2) < 1e-12
    did = out['did']['xpert_minus_v9']
    assert abs(did['did'] - 0.2) < 1e-12 and did['ci95'][0] > 0.19        # paired draws: the shared base cancels
    pm = out['per_molecule']
    s, n, dup = np.array(pm['row_sum']['v9']), np.array(pm['n_rows']), np.array(pm['is_dup'])
    assert abs(s[dup].sum() / n[dup].sum() - s[~dup].sum() / n[~dup].sum() - out['models']['v9']['lift']) < 1e-12
    bad = is_dup.copy()
    bad[0] = False
    with pytest.raises(SystemExit, match='both duplicate and clean rows'):
        score_p9.duplicate_lift({'v9': base}, mol, bad, 100)


def test_main_writes_the_duplicate_lift_block(tmp_path, monkeypatch):
    fix = make_synthetic_fixture(tmp_path, seed=11)
    patch_constants(monkeypatch, fix, tmp_path)
    score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'ridge'])
    out = json.load(open(tmp_path / 'model' / 'results' / 'p9_accuracy_ridge.json', encoding='utf-8'))
    dl = out['duplicate_lift']
    assert dl['n_dup_rows'] == 10 and dl['n_clean_rows'] == 50 and dl['n_dup_molecules'] == 1
    assert set(dl['models']) == {'v9', 'ridge'} and 'v9_minus_ridge' in dl['did']


def test_comparator_verdict_rules():
    """97.3 / 97.6 item 3: below-band or inadmissible removes a v9 win; above-band leaves it; an XPert win is
    uninterpretable; ridge (no comparator keys) passes through untouched."""
    o9, ridge = score_p9.REFS['xpert_o9'], score_p9.REFS['ridge']
    win, lose = 'v9 predicts unseen compounds better than X', 'X predicts unseen compounds better than v9'
    below = {'below_band': True, 'above_band': False}
    above = {'below_band': False, 'above_band': True}
    inside = {'below_band': False, 'above_band': False}
    assert score_p9.comparator_verdict(win, o9, inside, True) == win
    assert score_p9.comparator_verdict(win, o9, above, True) == win
    assert score_p9.comparator_verdict(win, o9, below, True).startswith('NO v9-WIN CLAIM: comparator reproduction below')
    assert score_p9.comparator_verdict(win, o9, inside, False).startswith('NO v9-WIN CLAIM: the comparator run is not admissible')
    assert score_p9.comparator_verdict(lose, o9, inside, True).startswith('UNINTERPRETABLE')
    assert score_p9.comparator_verdict(lose, ridge, None, True) == lose
    assert score_p9.comparator_verdict('NO COMPOUND-LEVEL CLAIM', o9, below, False) == 'NO COMPOUND-LEVEL CLAIM'


def _o9_fixture(tmp_path, monkeypatch, admissible=True, shift=0.0, with_record=True):
    o9_conf = dict(score_p9.REFS['xpert_o9'])               # before patch_constants replaces REFS
    fix = make_synthetic_fixture(tmp_path, seed=21)
    patch_constants(monkeypatch, fix, tmp_path)
    z = np.load(fix['ref_path'])
    prof = {k: np.asarray(z[k]) for k in ('row_index', 'y_true', 'ctl_true')}
    rng = np.random.default_rng(1)
    y = prof['y_true'] - prof['ctl_true']
    prof['y_pred'] = prof['ctl_true'] + y + rng.normal(0, 1.0 + shift, size=y.shape)
    extra = np.arange(10_000, 10_005)                       # comparator rows P9 does not score (its 13,445 vs 13,364)
    for k in ('y_true', 'ctl_true', 'y_pred'):
        prof[k] = np.concatenate([prof[k], rng.normal(size=(5, prof[k].shape[1]))])
    prof['row_index'] = np.concatenate([prof['row_index'], extra])
    p = tmp_path / 'o9_profile.npy'
    np.save(p, prof, allow_pickle=True)
    rr = tmp_path / 'run_record.json'
    if with_record:
        json.dump({'admissible_for_v9_win': admissible}, open(rr, 'w'))
    monkeypatch.setattr(score_p9, 'REFS', {'xpert_o9': dict(o9_conf, path=str(p), run_record=str(rr))})
    return fix, prof


def test_o9_profile_is_scored_with_reproduction_on_all_its_rows(tmp_path, monkeypatch):
    fix, prof = _o9_fixture(tmp_path, monkeypatch)
    score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'xpert_o9'])
    out = json.load(open(tmp_path / 'model' / 'results' / 'p9_accuracy_xpert_o9.json', encoding='utf-8'))
    rep_ = out['comparator']['reproduction']
    want = float(np.nanmean(score_p9.per_row_pearson(prof['y_pred'] - prof['ctl_true'], prof['y_true'] - prof['ctl_true'])))
    assert rep_['n_rows'] == 65 and abs(rep_['mean_all_rows'] - want) < 1e-9       # all comparator rows, not P9's 60
    assert out['comparator']['admissible_for_v9_win'] is True and out['comparator']['run_record_sha1']
    for sub in ('clean', 'full'):
        o = out['subsets'][sub]['of_record']
        assert o['verdict'] == score_p9.comparator_verdict(o['verdict_unadjusted'], score_p9.REFS['xpert_o9'],  # patched
                                                            rep_, True)


def test_o9_needs_its_run_record(tmp_path, monkeypatch):
    fix, _ = _o9_fixture(tmp_path, monkeypatch, with_record=False)
    with pytest.raises(SystemExit, match='run_record'):
        score_p9.main(['--p9_dir', str(fix['p9_dir']), '--ref', 'xpert_o9'])
