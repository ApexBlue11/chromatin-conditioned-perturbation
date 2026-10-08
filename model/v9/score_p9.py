# -*- coding: utf-8 -*-
"""RESULTS §96.3, as amended by §96.7: P9 accuracy reading (unseen-compound split against reference R).

The one scoring of P9's blinded test predictions (v9 trained on XPert's cold-drug split)
against a reference model R. The unit is the molecule, not the row.

    python model/v9/score_p9.py --p9_dir external/kaggle_out/v9p9 --ref ridge [--dry_check]
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_p7 import check_snapshot_identities
from coldcell_h2h import load_profile, per_row_pearson

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPECT_N_FULL = 13364
EXPECT_SHA1_FULL = '5f85ef0b5bec3b82bd1b0823c2f4d507f2e094a9'
EXPECT_N_CLEAN = 11983
EXPECT_SHA1_CLEAN = '6024dbf8a8d5301e68174d678e69f28b31f3aea7'
UNITS_JSON = 'model/results/mechanism96/cold_drug_units.json'
UNITS_SHA1 = '953abb636e828934995de3446e9b5a38e45cebf8'
CLEAN_JSON = 'model/results/mechanism96/cold_drug_clean_subset.json'
CLEAN_SHA1 = '3ba03db61aec5906766c28c74ec3f200ef727905'
BUNDLE = 'external/xpert_split_bundle/xpert_mdmt_splits.npz'
SPLIT_KEY = 'split_split_cold_drug_1'
N_BOOT = 20000
CI_WIDTH_MAX = 0.10
FRAC_MIN = 0.60
SIGN_P_MAX = 0.01
MAX_DROPPED_FRAC = 0.01

REFS = {
    'ridge': {
        'path': 'external/xpert_split_bundle/baselines_split_cold_drug_1.npz',
        'key': 'ridge_pred',
        'minus_ctl': True,
        'label': 'ridge (bundle: control + ECFP4 + descriptors)',
    },
    # RESULTS 97.4: O9's test profile (a pickled dict, as O2's); 97.3 / 97.6 comparator rules apply
    'xpert_o9': {
        'path': 'external/kaggle_out/cd1_o9/xpert_trained_split_cold_drug_1_final_test_profile.npy',
        'key': 'y_pred',
        'minus_ctl': True,
        'label': 'XPert as published (O9; checkpoint selected on these test rows)',
        'run_record': 'external/kaggle_out/cd1_o9/run_record.json',
        'repro_band': (0.621, 0.669),
        'win_uninterpretable': True,
    },
}


def sha1(p):
    """Compute sha1 hex digest of file at path p."""
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def row_set_sha1(row_index):
    """Compute sha1 of sorted int64 row index array (GUARD 6 convention)."""
    return hashlib.sha1(np.sort(np.asarray(row_index, dtype=np.int64)).tobytes()).hexdigest()


def resolve_path(p):
    """Resolve path relative to REPO if relative, else return p."""
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def molecule_reading(r_ours, r_ref, molecule, n_boot=N_BOOT, seed=0):
    """RESULTS §96.3 / §96.7: pure function computing the cluster estimand over molecules.

    diff = r_ours - r_ref
    For each molecule m: d_m = median(diff over ALL rows of molecule m).
    mean_d = mean(d_m), unweighted over molecules.
    Cluster bootstrap over molecules: n_boot resamples of molecules with replacement (seed).
    """
    diff = np.asarray(r_ours, dtype=np.float64) - np.asarray(r_ref, dtype=np.float64)
    molecule = np.asarray(molecule)

    unique_mols = np.unique(molecule)
    k = len(unique_mols)
    if k == 0:
        return {
            'mean_d': 0.0,
            'ci95': [0.0, 0.0],
            'ci_lo': 0.0,
            'ci_hi': 0.0,
            'ci_width': 0.0,
            'n_molecules': 0,
            'n_favour_ours': 0,
            'n_favour_ref': 0,
            'n_tie': 0,
            'frac_favour_ours': 0.0,
            'frac_favour_ref': 0.0,
            'sign_p': 1.0,
            'd_m': {},
        }

    order = np.argsort(molecule)
    mol_sorted = molecule[order]
    diff_sorted = diff[order]
    u_mols, idx_start, counts = np.unique(mol_sorted, return_index=True, return_counts=True)

    d_m = {}
    d_m_list = []
    for m, s, c in zip(u_mols, idx_start, counts):
        val = float(np.median(diff_sorted[s:s + c]))
        d_m[str(m)] = val
        d_m_list.append(val)
    d_m_arr = np.array(d_m_list, dtype=np.float64)

    mean_d = float(np.mean(d_m_arr))

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, k, size=(n_boot, k))
    boot_means = d_m_arr[boot_idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    ci_width = float(ci_hi - ci_lo)

    n_favour_ours = int((d_m_arr > 0).sum())
    n_favour_ref = int((d_m_arr < 0).sum())
    n_tie = int((d_m_arr == 0).sum())
    frac_favour_ours = float(n_favour_ours / k)
    frac_favour_ref = float(n_favour_ref / k)

    denom = n_favour_ours + n_favour_ref
    if denom == 0:
        sign_p = 1.0
    else:
        sign_p = float(scipy.stats.binomtest(n_favour_ours, denom, 0.5).pvalue)

    return {
        'mean_d': mean_d,
        'ci95': [ci_lo, ci_hi],
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'ci_width': ci_width,
        'n_molecules': k,
        'n_favour_ours': n_favour_ours,
        'n_favour_ref': n_favour_ref,
        'n_tie': n_tie,
        'frac_favour_ours': frac_favour_ours,
        'frac_favour_ref': frac_favour_ref,
        'sign_p': sign_p,
        'd_m': d_m,
    }


def verdict(reading, ref_label='R'):
    """RESULTS §96.3 / §96.7: pure function determining compound-level claim verdict."""
    if reading['ci_width'] > CI_WIDTH_MAX:
        return 'UNINFORMATIVE: cluster CI wider than 0.10; no compound-level claim'
    if (reading['mean_d'] > 0 and reading['ci_lo'] > 0 and
            reading['frac_favour_ours'] >= FRAC_MIN and reading['sign_p'] < SIGN_P_MAX):
        return f'v9 predicts unseen compounds better than {ref_label}'
    if (reading['mean_d'] < 0 and reading['ci_hi'] < 0 and
            reading['frac_favour_ref'] >= FRAC_MIN and reading['sign_p'] < SIGN_P_MAX):
        return f'{ref_label} predicts unseen compounds better than v9'
    return 'NO COMPOUND-LEVEL CLAIM'


def duplicate_lift(r_by_model, molecule, is_dup, n_boot=N_BOOT, seed=0):
    """RESULTS 97.6 item 5 (reported, never a claim): per model, the mean row score on the duplicate compounds' rows minus
    the mean on the clean rows, and every pairwise difference-in-differences. Molecule-level bootstrap: the duplicate and the
    clean molecules are resampled separately, with replacement, and the SAME draws serve every model (paired). Per-molecule
    row sums and counts are returned so another reference's run (same molecules, same seed) can be combined draw for draw."""
    molecule = np.asarray(molecule)
    is_dup = np.asarray(is_dup, dtype=bool)
    mols = np.unique(molecule)
    pos = np.searchsorted(mols, molecule)
    dup_of_mol = np.zeros(len(mols), dtype=bool)
    np.logical_or.at(dup_of_mol, pos, is_dup)
    clean_of_mol = np.zeros(len(mols), dtype=bool)
    np.logical_or.at(clean_of_mol, pos, ~is_dup)
    if (dup_of_mol & clean_of_mol).any():
        raise SystemExit('FATAL: a molecule has both duplicate and clean rows')
    n_m = np.bincount(pos, minlength=len(mols)).astype(np.float64)
    sums = {k: np.bincount(pos, weights=np.asarray(r, dtype=np.float64), minlength=len(mols)) for k, r in r_by_model.items()}
    d_idx, c_idx = np.flatnonzero(dup_of_mol), np.flatnonzero(clean_of_mol)
    rng = np.random.default_rng(seed)
    bd = d_idx[rng.integers(0, len(d_idx), size=(n_boot, len(d_idx)))]
    bc = c_idx[rng.integers(0, len(c_idx), size=(n_boot, len(c_idx)))]

    def lift(sm, idx_d, idx_c):
        return sm[idx_d].sum(-1) / n_m[idx_d].sum(-1) - sm[idx_c].sum(-1) / n_m[idx_c].sum(-1)
    out = {'LABEL': 'reported, never a claim (RESULTS 97.6 item 5); the contrast also reflects which compounds are duplicates',
           'n_dup_molecules': int(len(d_idx)), 'n_clean_molecules': int(len(c_idx)),
           'n_dup_rows': int(is_dup.sum()), 'n_clean_rows': int((~is_dup).sum()), 'models': {}, 'did': {},
           'per_molecule': {'molecule': [str(m) for m in mols], 'n_rows': n_m.astype(int).tolist(),
                            'is_dup': dup_of_mol.tolist(), 'row_sum': {k: v.tolist() for k, v in sums.items()}}}
    boot = {}
    for k, sm in sums.items():
        boot[k] = lift(sm, bd, bc)
        out['models'][k] = {'dup_mean': float(sm[d_idx].sum() / n_m[d_idx].sum()),
                            'clean_mean': float(sm[c_idx].sum() / n_m[c_idx].sum()),
                            'lift': float(lift(sm, d_idx, c_idx)),
                            'lift_ci95': [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]}
    keys = list(sums)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            dd = boot[a] - boot[b]
            out['did']['%s_minus_%s' % (a, b)] = {
                'did': out['models'][a]['lift'] - out['models'][b]['lift'],
                'ci95': [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    return out


def comparator_verdict(v, ref_conf, repro=None, admissible=True):
    """RESULTS 97.3 / 97.4 / 97.6 item 3, applied to a verdict string for a comparator whose checkpoint saw the test rows:
    a v9 win needs the run admissible (run_record) and the reproduction not BELOW the band; above the band is flagged only;
    the comparator's own win is uninterpretable as a model comparison. References without these keys pass through."""
    if v.startswith('v9 predicts'):
        if not admissible:
            return 'NO v9-WIN CLAIM: the comparator run is not admissible for a v9 win (run_record; RESULTS 97.3 item 2)'
        if repro is not None and repro['below_band']:
            return 'NO v9-WIN CLAIM: comparator reproduction below the band (possible failure; RESULTS 97.6 item 3)'
        return v
    if ref_conf.get('win_uninterpretable') and v.endswith('better than v9'):
        return ("UNINTERPRETABLE as a model comparison: the comparator's checkpoint was selected on these test rows "
                '(RESULTS 97.3 item 1)')
    return v


def boot_ci_rows(diff, n_boot=N_BOOT, seed=0, chunk_size=2000):
    """Row bootstrap percentile CI (95%) over diff, drawn in memory-safe chunks."""
    n = len(diff)
    if n == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = []
    drawn = 0
    while drawn < n_boot:
        k_chunk = min(chunk_size, n_boot - drawn)
        idx = rng.integers(0, n, size=(k_chunk, n))
        means.append(diff[idx].mean(axis=1))
        drawn += k_chunk
    all_means = np.concatenate(means)
    return [float(np.percentile(all_means, 2.5)), float(np.percentile(all_means, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser(description='RESULTS 96.3 / 96.7: P9 accuracy reading against reference R.')
    ap.add_argument('--p9_dir', required=True, help='Directory holding P9 kernel outputs and P9_COMPLETE.json')
    ap.add_argument('--ref', required=True, help='Reference model to score against (entry in REFS)')
    ap.add_argument('--dry_check', action='store_true', help='Verify all input guards (steps 1-5) and stop')
    args = ap.parse_args(argv)

    if args.ref not in REFS:
        raise SystemExit(f'FATAL: unknown reference {args.ref!r}; choices are {sorted(REFS.keys())}')
    ref_conf = REFS[args.ref]

    # Step 1: The touch-once rule
    out_json = os.path.join(REPO, 'model', 'results', f'p9_accuracy_{args.ref}.json')
    if os.path.exists(out_json):
        raise SystemExit(f'FATAL: {out_json} exists -- touched once')

    # Step 2: The manifest
    mf = os.path.join(args.p9_dir, 'P9_COMPLETE.json')
    if not os.path.exists(mf):
        raise SystemExit('FATAL: no P9_COMPLETE.json -- the kernel did not finish; nothing here may be scored')
    with open(mf, encoding='utf-8') as f:
        manifest = json.load(f)
    if not manifest.get('complete') or manifest.get('n_seeds') != 3:
        raise SystemExit(
            f'FATAL: manifest incomplete: {{"complete": {manifest.get("complete")}, "n_seeds": {manifest.get("n_seeds")}}}'
        )
    for name, h in manifest.get('files', {}).items():
        p = os.path.join(args.p9_dir, name)
        if not os.path.exists(p) or sha1(p) != h:
            raise SystemExit(f'FATAL: {name} missing or sha1 differs from the manifest')

    seed_files = sorted(n for n in manifest.get('files', {}) if re.fullmatch(r'v9p9_seed\d\.npz', n))
    if len(seed_files) != 3:
        raise SystemExit(f'FATAL: expected 3 seed files, manifest has {seed_files!r}')

    v2 = 'V2' in manifest.get('stack', '')
    if v2:
        last_files = sorted(n for n in manifest.get('files', {}) if re.fullmatch(r'v9p9_seed\d_last\.npz', n))
        if len(last_files) != 3:
            raise SystemExit(f'FATAL: the stack contains V2 but the manifest has {len(last_files)} V2-last files (need 3)')
        snap_files = sorted(n for n in manifest.get('files', {}) if re.fullmatch(r'v9p9_seed\d_snap\d\.npz', n))
        if len(snap_files) != 9:
            raise SystemExit(f'FATAL: the stack contains V2 but the manifest has {len(snap_files)} snapshot files (need 9)')
        check_snapshot_identities(args.p9_dir, seed_files)

    # Step 3: The seed files
    seeds_data = []
    for s_name in seed_files:
        z = np.load(os.path.join(args.p9_dir, s_name))
        for k in ('y_pred', 'y_true', 'ctl_true', 'row_index'):
            if k not in z:
                raise SystemExit(f'FATAL: {s_name} missing required key {k!r}')
        seeds_data.append(z)

    seed0 = seeds_data[0]
    for idx, s in enumerate(seeds_data[1:], start=1):
        for k in ('row_index', 'y_true', 'ctl_true'):
            if not np.array_equal(s[k], seed0[k]):
                raise SystemExit(f'FATAL: seed files disagree on {k!r}')

    row_index = seed0['row_index']
    if len(row_index) != EXPECT_N_FULL:
        raise SystemExit(f'FATAL: P9 row count {len(row_index)} != EXPECT_N_FULL {EXPECT_N_FULL}')
    row_sha1 = row_set_sha1(row_index)
    if row_sha1 != EXPECT_SHA1_FULL:
        raise SystemExit(f'FATAL: P9 row-set sha1 {row_sha1} != EXPECT_SHA1_FULL {EXPECT_SHA1_FULL}')

    # Step 4: The reference
    ref_path = resolve_path(ref_conf['path'])
    if not os.path.exists(ref_path):
        raise SystemExit(f'FATAL: reference file {ref_path} not found')
    ref_data = load_profile(ref_path)
    for k in ('row_index', 'y_true', 'ctl_true', ref_conf['key']):
        if k not in ref_data:
            raise SystemExit(f'FATAL: reference missing required key {k!r}')

    ref_row_pos = {idx: i for i, idx in enumerate(ref_data['row_index'])}
    missing_in_ref = [idx for idx in row_index if idx not in ref_row_pos]
    if missing_in_ref:
        raise SystemExit(f'FATAL: reference is missing {len(missing_in_ref)} P9 rows')

    ref_align_idx = np.array([ref_row_pos[idx] for idx in row_index], dtype=np.int64)
    ref_y_true = ref_data['y_true'][ref_align_idx]
    ref_ctl_true = ref_data['ctl_true'][ref_align_idx]
    ref_pred = ref_data[ref_conf['key']][ref_align_idx]

    for k, ref_arr, p9_arr in [('y_true', ref_y_true, seed0['y_true']), ('ctl_true', ref_ctl_true, seed0['ctl_true'])]:
        dmax = float(np.max(np.abs(ref_arr - p9_arr)))
        if dmax > 1e-4:
            raise SystemExit(f'FATAL: reference {k} differs from P9 by {dmax:.3g} (max abs > 1e-4)')

    if ref_conf.get('minus_ctl', False):
        ref_delta = ref_pred - ref_ctl_true
    else:
        ref_delta = ref_pred

    # RESULTS 97.3 / 97.6: the comparator's admissibility and its reproduction on ALL its rows (the full split, their convention)
    admissible, repro, run_record = True, None, None
    if ref_conf.get('run_record'):
        rr_path = resolve_path(ref_conf['run_record'])
        if not os.path.exists(rr_path):
            raise SystemExit('FATAL: the comparator run_record %s is missing (RESULTS 97.3)' % rr_path)
        run_record = json.load(open(rr_path, encoding='utf-8'))
        if 'admissible_for_v9_win' not in run_record:
            raise SystemExit('FATAL: run_record has no admissible_for_v9_win')
        admissible = bool(run_record['admissible_for_v9_win'])
    if ref_conf.get('repro_band'):
        lo, hi = ref_conf['repro_band']
        r_all = per_row_pearson(np.asarray(ref_data[ref_conf['key']], np.float64) - ref_data['ctl_true'],
                                np.asarray(ref_data['y_true'], np.float64) - ref_data['ctl_true'])
        m_all = float(np.nanmean(r_all))
        repro = {'mean_all_rows': m_all, 'n_rows': int(np.isfinite(r_all).sum()), 'band': [lo, hi],
                 'below_band': m_all < lo, 'above_band': m_all > hi,
                 'LABEL': 'row-pooled on all comparator rows (the full split); below: no v9-win claim; above: flagged'}

    # Step 5: The row metadata
    bundle_path = resolve_path(BUNDLE)
    if not os.path.exists(bundle_path):
        raise SystemExit(f'FATAL: bundle file {bundle_path} not found')
    bundle_data = np.load(bundle_path)
    for k in ('row_index', 'meta_pert_id', 'meta_cell', SPLIT_KEY):
        if k not in bundle_data:
            raise SystemExit(f'FATAL: bundle missing required key {k!r}')

    bundle_row_pos = {idx: i for i, idx in enumerate(bundle_data['row_index'])}
    missing_in_bundle = [idx for idx in row_index if idx not in bundle_row_pos]
    if missing_in_bundle:
        raise SystemExit(f'FATAL: bundle is missing {len(missing_in_bundle)} P9 rows')
    bundle_idx = np.array([bundle_row_pos[idx] for idx in row_index], dtype=np.int64)

    pert = np.array([p.decode() if isinstance(p, bytes) else str(p) for p in bundle_data['meta_pert_id'][bundle_idx]])
    cell = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in bundle_data['meta_cell'][bundle_idx]])

    split_vals = bundle_data[SPLIT_KEY][bundle_idx]
    split_str = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in split_vals])
    if not (split_str == 'test').all():
        raise SystemExit(f'FATAL: not all P9 rows are "test" under {SPLIT_KEY}')

    # Units JSON
    units_path = resolve_path(UNITS_JSON)
    if not os.path.exists(units_path) or sha1(units_path) != UNITS_SHA1:
        raise SystemExit(f'FATAL: units JSON sha1 differs from UNITS_SHA1 ({UNITS_SHA1})')
    with open(units_path, encoding='utf-8') as f:
        units_data = json.load(f)
    p2m = units_data.get('pert_to_molecule', {})
    molecule_stratum = units_data.get('molecule_stratum', {})

    missing_perts = [p for p in np.unique(pert) if p not in p2m]
    if missing_perts:
        raise SystemExit(f'FATAL: {len(missing_perts)} pert_ids missing from pert_to_molecule: {missing_perts[:5]}')
    molecule = np.array([p2m[p] for p in pert])

    # Clean JSON
    clean_path = resolve_path(CLEAN_JSON)
    if not os.path.exists(clean_path) or sha1(clean_path) != CLEAN_SHA1:
        raise SystemExit(f'FATAL: clean JSON sha1 differs from CLEAN_SHA1 ({CLEAN_SHA1})')
    with open(clean_path, encoding='utf-8') as f:
        clean_data = json.load(f)
    dirty_compounds = set(clean_data.get('dirty_compounds', []))

    clean_mask = np.array([p not in dirty_compounds for p in pert], dtype=bool)
    clean_rows = row_index[clean_mask]

    if len(clean_rows) != EXPECT_N_CLEAN:
        raise SystemExit(f'FATAL: clean count {len(clean_rows)} != EXPECT_N_CLEAN {EXPECT_N_CLEAN}')
    c_sha1 = row_set_sha1(clean_rows)
    if c_sha1 != EXPECT_SHA1_CLEAN:
        raise SystemExit(f'FATAL: clean row-set sha1 {c_sha1} != EXPECT_SHA1_CLEAN {EXPECT_SHA1_CLEAN}')

    if args.dry_check:
        print('P9 inputs verified')
        return

    # Step 6: The row scores
    y = seed0['y_true'] - seed0['ctl_true']
    ctl_true = seed0['ctl_true']
    r_seed = [per_row_pearson(s['y_pred'] - ctl_true, y) for s in seeds_data]
    r_v9 = np.mean(r_seed, axis=0)
    r_ref = per_row_pearson(ref_delta, y)
    mean_pred = np.mean([s['y_pred'] for s in seeds_data], axis=0)
    r_ens = per_row_pearson(mean_pred - ctl_true, y)

    finite_mask = np.isfinite(r_v9) & np.isfinite(r_ref) & np.isfinite(r_ens)
    for rs in r_seed:
        finite_mask &= np.isfinite(rs)

    n_total = len(row_index)
    n_dropped_full = int((~finite_mask).sum())
    if n_dropped_full / n_total > MAX_DROPPED_FRAC:
        raise SystemExit(
            f'FATAL: dropped {n_dropped_full} rows ({n_dropped_full / n_total:.3%} > MAX_DROPPED_FRAC {MAX_DROPPED_FRAC:.3%}) due to non-finite scores'
        )

    # Step 7 & 10: Subsets
    subsets_out = {}
    for subset_name, mask, label in [
        ('clean', clean_mask & finite_mask, 'reading of record for unseen-compound claims (RESULTS 96.6/96.7)'),
        ('full', finite_mask, 'the benchmark as defined (includes same-molecule duplicates)'),
    ]:
        r_v9_sub = r_v9[mask]
        r_ref_sub = r_ref[mask]
        r_ens_sub = r_ens[mask]
        r_seed_subs = [rs[mask] for rs in r_seed]
        mol_sub = molecule[mask]
        cell_sub = cell[mask]
        diff_sub = r_v9_sub - r_ref_sub

        if subset_name == 'clean':
            n_dropped_sub = int((clean_mask & ~finite_mask).sum())
        else:
            n_dropped_sub = n_dropped_full

        # of_record
        of_rec = molecule_reading(r_v9_sub, r_ref_sub, mol_sub, N_BOOT, seed=0)
        of_rec['verdict_unadjusted'] = verdict(of_rec, ref_conf['label'])
        of_rec['verdict'] = comparator_verdict(of_rec['verdict_unadjusted'], ref_conf, repro, admissible)

        # per_seed
        per_seed_list = []
        for s_idx, rs_sub in enumerate(r_seed_subs):
            s_reading = molecule_reading(rs_sub, r_ref_sub, mol_sub, N_BOOT, seed=0)
            s_reading['verdict'] = verdict(s_reading, ref_conf['label'])
            s_reading['seed'] = s_idx
            s_reading['LABEL'] = 'reported, not the claim'
            per_seed_list.append(s_reading)

        # seed_ensemble
        seed_ens = molecule_reading(r_ens_sub, r_ref_sub, mol_sub, N_BOOT, seed=0)
        seed_ens['verdict'] = verdict(seed_ens, ref_conf['label'])
        seed_ens['LABEL'] = 'v9 seed ensemble (not the claim)'

        # row_pooled
        row_pooled = {
            'LABEL': "the field's convention, not the claim",
            'mean_r_v9': float(np.mean(r_v9_sub)),
            'mean_r_ref': float(np.mean(r_ref_sub)),
            'mean_r_ens': float(np.mean(r_ens_sub)),
            'per_seed_means': [float(np.mean(rs_sub)) for rs_sub in r_seed_subs],
            'paired_mean': float(np.mean(diff_sub)),
            'paired_mean_ci95_row_bootstrap': boot_ci_rows(diff_sub, N_BOOT, seed=0),
            'n_rows': int(len(diff_sub)),
        }

        # per_cell
        per_cell = []
        u_cells = np.unique(cell_sub)
        for c in sorted(u_cells, key=lambda c: -int((cell_sub == c).sum())):
            m = (cell_sub == c)
            per_cell.append({
                'cell': str(c),
                'n_rows': int(m.sum()),
                'n_molecules': int(len(np.unique(mol_sub[m]))),
                'median_diff': float(np.median(diff_sub[m])),
                'mean_r_v9': float(np.mean(r_v9_sub[m])),
                'mean_r_ref': float(np.mean(r_ref_sub[m])),
            })

        # per_stratum
        per_stratum = {}
        for st in ('lt_0.6', '0.6_0.8', '0.8_0.999', 'ge_0.999'):
            st_mask = np.array([molecule_stratum.get(m) == st for m in mol_sub], dtype=bool)
            if st_mask.sum() > 0:
                st_reading = molecule_reading(r_v9_sub[st_mask], r_ref_sub[st_mask], mol_sub[st_mask], N_BOOT, seed=0)
                st_reading['LABEL'] = 'reported, not a reading'
                per_stratum[st] = st_reading

        subsets_out[subset_name] = {
            'LABEL': label,
            'of_record': of_rec,
            'per_seed': per_seed_list,
            'seed_ensemble': seed_ens,
            'row_pooled': row_pooled,
            'per_cell': per_cell,
            'per_stratum': per_stratum,
            'n_rows': int(len(diff_sub)),
            'n_molecules': int(len(np.unique(mol_sub))),
            'n_dropped': int(n_dropped_sub),
        }

    # RESULTS 97.6 item 5: what the same-molecule duplicates do to each model's score (full rows; reported only)
    dup_block = duplicate_lift({'v9': r_v9[finite_mask], args.ref: r_ref[finite_mask]}, molecule[finite_mask],
                               ~clean_mask[finite_mask], N_BOOT, seed=0)

    # Step 11: Top level output
    inputs = {
        'manifest': {
            'path': mf,
            'sha1': sha1(mf),
            'stack': manifest.get('stack', ''),
            'files': {f: sha1(os.path.join(args.p9_dir, f)) for f in manifest.get('files', {})},
        },
        'bundle': {
            'path': BUNDLE,
            'sha1': sha1(bundle_path),
        },
        'units': {
            'path': UNITS_JSON,
            'sha1': sha1(units_path),
        },
        'clean': {
            'path': CLEAN_JSON,
            'sha1': sha1(clean_path),
        },
        'ref': {
            'name': args.ref,
            'path': ref_conf['path'],
            'sha1': sha1(ref_path),
        },
    }

    out_data = {
        'inputs': inputs,
        'ref': args.ref,
        'ref_label': ref_conf['label'],
        'estimand': 'Per-molecule median difference of per-row Pearson correlation against reference, with cluster bootstrap over molecules (RESULTS §96.3 / §96.7).',
        'subsets': subsets_out,
        'duplicate_lift': dup_block,
        'comparator': {'admissible_for_v9_win': admissible, 'reproduction': repro,
                       'run_record_sha1': sha1(resolve_path(ref_conf['run_record'])) if run_record is not None else None},
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)

    marker = {
        'complete': True,
        'file': os.path.basename(out_json),
        'sha1': sha1(out_json),
    }
    with open(out_json + '.marker', 'w', encoding='utf-8') as f:
        json.dump(marker, f, indent=2)

    # Print summary table
    print('\n' + '=' * 80)
    print(f'P9 ACCURACY READING vs {ref_conf["label"]}')
    print('=' * 80)
    print(f'{"Subset":<10} | {"Verdict":<40} | {"mean_d [95% CI]":<25} | {"Row-pooled (v9 vs Ref)":<20}')
    print('-' * 105)
    for sub_name in ('clean', 'full'):
        sub = subsets_out[sub_name]
        o = sub['of_record']
        ci_str = f"{o['mean_d']:.4f} [{o['ci95'][0]:.4f}, {o['ci95'][1]:.4f}]"
        rp = sub['row_pooled']
        rp_str = f"{rp['mean_r_v9']:.4f} vs {rp['mean_r_ref']:.4f}"
        print(f'{sub_name:<10} | {o["verdict"]:<40} | {ci_str:<25} | {rp_str:<20}')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
