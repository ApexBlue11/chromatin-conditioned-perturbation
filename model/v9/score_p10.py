# -*- coding: utf-8 -*-
"""RESULTS §98.3, as amended by §98.6: P10 accuracy reading against reference models.

Evaluates v9 on XPert's cold-drug split against reference models:
- mlp_m: Bai et al.'s drug-free MLP (M)
- mlp_mdt: Bai et al.'s drug-free MLP with dose and time (M+dt, reported only)
- f0: zero-parameter drug-free floor (f0, reported only)
- v9_drugblind: v9 retrained drug-blind (B1)
- pertdit / pertdit_s118: PertDiT through Bai et al.'s harness, seed 117 (the D reading of record) / 118 (reported only)
  (RESULTS 98.3, 98.15 ruling 6, 98.16, 98.20)

P10 uses P9's estimand (per-molecule median difference of per-row Pearson correlation,
cluster bootstrap over molecules) and verdict unchanged.

CLI:
    python score_p10.py --p9_dir external/kaggle_out/v9p9 --ref {mlp_m,mlp_mdt,f0,v9_drugblind} [--ref_dir D] [--dry_check]
    python score_p10.py --read R1_JSON [R2_JSON]
    python score_p10.py --p9_dir external/kaggle_out/v9p9 --ref pertdit [--ref_dir D] [--d_truncated_reported]
    python score_p10.py --read_d p10_accuracy_pertdit.json
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_p7 import check_snapshot_identities
from coldcell_h2h import per_row_pearson
from score_p9 import (
    molecule_reading,
    verdict,
    row_set_sha1,
    sha1,
    resolve_path,
    boot_ci_rows,
    EXPECT_N_FULL,
    EXPECT_SHA1_FULL,
    EXPECT_N_CLEAN,
    EXPECT_SHA1_CLEAN,
    UNITS_JSON,
    UNITS_SHA1,
    CLEAN_JSON,
    CLEAN_SHA1,
    BUNDLE,
    SPLIT_KEY,
    N_BOOT,
    MAX_DROPPED_FRAC,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CARVES_JSON = 'model/results/p10/p10_carves.json'
CARVES_SHA1 = 'd6fc78447228b87641cbd5af8ff0143a714edeb8'
EXPECT_N_DISTINCT = 2091
EXPECT_SHA1_DISTINCT = 'def96fc0ab190fcb6e8c1c0cfb28493229167680'

D_REFS = ('pertdit', 'pertdit_s118')
REFS_P10 = ('mlp_m', 'mlp_mdt', 'f0', 'v9_drugblind') + D_REFS
MLP_SEEDS = {131419, 131420, 131421}   # Bai et al.'s experiments/mlp_baseline/train_default.py:38 default (1d750cd), +1, +2

# Review 075 C2: the bytes every M / M+dt kernel verified in-run (their records' 'sha1s'), pinned here.
PIN_SHA1 = {'bundle': '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464', 'train_mlp': 'ddc0725ee0a7d73bae0ac478d92e5f8a879b13bd',
            'p10_mlp': '97f84874df4609c463289fbc9bf387de3ff8775a', 'carves': 'd6fc78447228b87641cbd5af8ff0143a714edeb8'}
PIN_DT_SHA1 = '09bcf0b2d0c1afebdd87324043f21dc65f460644'
EXPECT_N_TEST_M = 13445

REF_LABELS = {
    'mlp_m': 'mlp_m',
    'mlp_mdt': 'mlp_mdt',
    'v9_drugblind': 'v9_drugblind',
    'f0': 'f0',
    'pertdit': 'PertDiT (our run of their harness)',
    'pertdit_s118': 'PertDiT (our run of their harness)',
}

# ---- P10 arm D (RESULTS 98.3, 98.15 ruling 6, 98.16 C2 / Ask 6, 98.20 C5) ----
D_SEEDS = {'pertdit': 117, 'pertdit_s118': 118}
D_FILES = {'p10_pertdit_seed117.npz', 'p10_pertdit_seed118.npz', 'p10_pertdit_seed117_record.json',
           'p10_pertdit_seed118_record.json', 'p10d_timing.json'}
# the in-run sha1s every D record must carry (the run kernel's GUARD 2 pins, built from the merge's emb_record)
PIN_D = {'bundle': '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464', 'carves': 'd6fc78447228b87641cbd5af8ff0143a714edeb8',
         'p10_mlp': '97f84874df4609c463289fbc9bf387de3ff8775a', 'p10_pertdit': '929ce1fe6d9e9c87c89eccb153f5c7d8c8926b12',
         'train_pertdit': 'a431f53ba18d43348161ee19b97b01148229f1ac', 'pertdit_src_zip': '06bf66bdb2b21f53ddf72d6f1459636e96bca7ad',
         'compound_map': '918e08f61352e8f122c3451d71a800e616de4f7c', 'dose_list': '7941bcd3155ae73fbdcca2f11ce6af647300d2ce',
         'bench_drug_emb': 'eba75fdc2ef761fbd844dc7a2a967d4e8a3b0a0c', 'bench_dose_emb': 'c42bce8eeca3cf47c4ff25e16608c6e00474f1a0',
         'cap_trainer': 'e910a855bb4405a7bb1fb56c84572470be9c6316'}
D_CAP_S = 21600.0
PINNED_GPU_VERSIONS = "scanpy=1.12.4 anndata=0.13.3.post0 diffusers=0.30.2 einops=0.8.0 huggingface_hub=0.24.6 rdkit=2023.09.1"   # the GPU kernel's asserted version line (review 081 C4 / 082 C2)
BAI_EVALUATE_ALL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external',
                                'bai_benchmark_1d750cd', 'experiments', '_shared', 'evaluation', 'evaluate_all.py')
COLLISION_JSON = 'model/results/p10/p10d_caption_collision.json'
COLLISION_SHA1 = 'bb3d6fb3b42bf0765cb3e2b3d38f216e453faab5'
ORACLE_INVERSE_R = 0.99979   # 98.15: the input-only inverse applied to a perfect transformed-space prediction
D_LABEL = 'D: head-to-head with the newest published L1000 model with code; our run of their harness (98.3)'
D_TRUNCATED_LABEL = 'reported only: truncated at the 6 GPU-h cap, never read (98.15 ruling 5)'

REPORTED_LABEL = 'reported, never read (RESULTS 98.6)'
R1_LABEL = 'R1: benchmark statement only (98.6 C1)'
R2_LABEL = 'R2: carries the compound-use inference (98.6 C1)'


def load_p9_and_metadata(p9_dir, bundle_path=None, units_path=None, clean_path=None):
    """Mirror score_p9.py steps 2-5: manifest, seed files, bundle row guards, units/clean JSON."""
    # Step 2: The manifest
    mf = os.path.join(p9_dir, 'P9_COMPLETE.json')
    if not os.path.exists(mf):
        raise SystemExit('FATAL: no P9_COMPLETE.json -- the kernel did not finish; nothing here may be scored')
    with open(mf, encoding='utf-8') as f:
        manifest = json.load(f)
    if not manifest.get('complete') or manifest.get('n_seeds') != 3:
        raise SystemExit(
            f'FATAL: manifest incomplete: {{"complete": {manifest.get("complete")}, "n_seeds": {manifest.get("n_seeds")}}}'
        )
    for name, h in manifest.get('files', {}).items():
        p = os.path.join(p9_dir, name)
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
        check_snapshot_identities(p9_dir, seed_files)

    # Step 3: The seed files
    seeds_data = []
    for s_name in seed_files:
        z = np.load(os.path.join(p9_dir, s_name))
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

    # Step 5: Bundle and row metadata
    bundle_path = resolve_path(bundle_path or BUNDLE)
    if not os.path.exists(bundle_path):
        raise SystemExit(f'FATAL: bundle file {bundle_path} not found')
    bundle_data = np.load(bundle_path, allow_pickle=True)
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
    units_path = resolve_path(units_path or UNITS_JSON)
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
    clean_path = resolve_path(clean_path or CLEAN_JSON)
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

    # Compute v9 row scores
    y = seed0['y_true'] - seed0['ctl_true']
    ctl_true = seed0['ctl_true']
    r_seed = [per_row_pearson(s['y_pred'] - ctl_true, y) for s in seeds_data]
    r_v9 = np.mean(r_seed, axis=0)

    return {
        'manifest': manifest,
        'manifest_path': mf,
        'seed_files': seed_files,
        'seeds_data': seeds_data,
        'row_index': row_index,
        'y_true': seed0['y_true'],
        'ctl_true': ctl_true,
        'y': y,
        'r_seed': r_seed,
        'r_v9': r_v9,
        'bundle_data': bundle_data,
        'bundle_path': bundle_path,
        'bundle_idx': bundle_idx,
        'cell': cell,
        'pert': pert,
        'molecule': molecule,
        'molecule_stratum': molecule_stratum,
        'clean_mask': clean_mask,
        'clean_rows': clean_rows,
        'units_path': units_path,
        'clean_path': clean_path,
    }


def check_m_provenance(ref_dir, arm):
    """Review 075 C2: the download manifest P10_<ARM>_COMPLETE.json (sha1 of each npz and record, written at download), and
    each kernel record's in-run verified sha1s against the pinned bytes, its arm, seed and n_test. Returns the npz sha1s."""
    mf = os.path.join(ref_dir, 'P10_%s_COMPLETE.json' % arm.upper())
    if not os.path.exists(mf):
        raise SystemExit(f'FATAL: no download manifest {mf} (review 075 C2)')
    with open(mf, encoding='utf-8') as f:
        man = json.load(f)
    want_files = {f'p10_{arm}_seed{k}.npz' for k in range(3)} | {f'p10_{arm}_seed{k}_record.json' for k in range(3)}
    if not man.get('complete') or man.get('arm') != arm or set(man.get('files', {})) != want_files:
        raise SystemExit(f'FATAL: manifest {mf} incomplete or not arm {arm!r}: files {sorted(man.get("files", {}))}')
    for name, h in man['files'].items():
        fp = os.path.join(ref_dir, name)
        if not os.path.exists(fp) or sha1(fp) != h:
            raise SystemExit(f'FATAL: {name} missing or sha1 differs from the download manifest')
    pins = dict(PIN_SHA1, **({'dt_script': PIN_DT_SHA1} if arm == 'mdt' else {}))
    npz_sha1 = {}
    for k in range(3):
        with open(os.path.join(ref_dir, f'p10_{arm}_seed{k}_record.json'), encoding='utf-8') as f:
            rec = json.load(f)
        if rec.get('sha1s') != pins:
            raise SystemExit(f'FATAL: seed {k} record verified {rec.get("sha1s")}, not the pinned {pins} (review 075 C2)')
        if rec.get('arm') != arm or int(rec.get('n_test', -1)) != EXPECT_N_TEST_M:
            raise SystemExit(f'FATAL: seed {k} record arm {rec.get("arm")!r} / n_test {rec.get("n_test")} != {arm!r} / {EXPECT_N_TEST_M}')
        z = np.load(os.path.join(ref_dir, f'p10_{arm}_seed{k}.npz'), allow_pickle=True)
        if int(z['seed']) != int(rec.get('seed', -1)) or int(rec['seed']) != 131419 + k:
            raise SystemExit(f'FATAL: seed {k}: npz seed {int(z["seed"])}, record seed {rec.get("seed")}, expected {131419 + k}')
        npz_sha1[f'p10_{arm}_seed{k}.npz'] = man['files'][f'p10_{arm}_seed{k}.npz']
    return {'manifest': mf, 'manifest_sha1': sha1(mf), 'npz_sha1': npz_sha1}


def check_d_provenance(ref_dir, ref):
    """RESULTS 98.16 Ask 6 / review 081 C2: the download manifest, the records' in-run pins, the seed and n_test, of_record,
    the binding to the timing JSON (its decision re-derived from its timings), and the npz's arm / stop_reason."""
    mf = os.path.join(ref_dir, 'P10D_COMPLETE.json')
    if not os.path.exists(mf):
        raise SystemExit(f'FATAL: (d1) no download manifest {mf}')
    with open(mf, encoding='utf-8') as f:
        man = json.load(f)
    if not man.get('complete') or set(man.get('files', {})) != D_FILES:
        raise SystemExit(f'FATAL: (d1) manifest {mf} incomplete or not the D file set: {sorted(man.get("files", {}))}')
    for name, h in man['files'].items():
        fp = os.path.join(ref_dir, name)
        if not os.path.exists(fp) or sha1(fp) != h:
            raise SystemExit(f'FATAL: (d1) {name} missing or sha1 differs from the download manifest')
    seed = D_SEEDS[ref]
    recs = {}
    for s_ in (117, 118):
        with open(os.path.join(ref_dir, f'p10_pertdit_seed{s_}_record.json'), encoding='utf-8') as f:
            recs[s_] = json.load(f)
    rec = recs[seed]
    # review 082 C2: both runs were made with the pinned versions on two non-P100 GPUs
    for s_, r_ in recs.items():
        g = r_.get('gpu_names') or []
        if r_.get('versions') != PINNED_GPU_VERSIONS or len(g) != 2 or any('P100' in n for n in g):
            raise SystemExit(f'FATAL: (d2) seed {s_} record versions {r_.get("versions")!r} / GPUs {g!r} are not the pinned '
                             f'versions on two non-P100 GPUs')
    if rec.get('sha1s') != PIN_D:
        raise SystemExit(f'FATAL: (d2) seed {seed} record verified {rec.get("sha1s")}, not PIN_D')
    z = np.load(os.path.join(ref_dir, f'p10_pertdit_seed{seed}.npz'), allow_pickle=False)
    if int(rec.get('n_test', -1)) != EXPECT_N_TEST_M or int(rec.get('seed', -1)) != seed or int(z['seed']) != seed:
        raise SystemExit(f'FATAL: (d3) record n_test {rec.get("n_test")} / seed {rec.get("seed")} / npz seed {int(z["seed"])} '
                         f'!= {EXPECT_N_TEST_M} / {seed}')
    if recs[117].get('of_record') is not True or recs[118].get('of_record') is not False:
        raise SystemExit(f'FATAL: (d4) of_record must be True for 117 and False for 118: '
                         f'{recs[117].get("of_record")!r} / {recs[118].get("of_record")!r}')
    tpath = os.path.join(ref_dir, 'p10d_timing.json')
    with open(tpath, encoding='utf-8') as f:
        timing = json.load(f)
    # review 082 C1 / C2: the timing was measured with the run's own inputs, versions and GPUs
    tg = timing.get('gpu_names') or []
    if (timing.get('verified_sha1s') != PIN_D or timing.get('versions') != PINNED_GPU_VERSIONS or len(tg) != 2
            or any('P100' in n for n in tg)):
        raise SystemExit('FATAL: (d5) p10d_timing.json was not timed with the run\'s pins / versions / two non-P100 GPUs')
    import p10_pertdit  # the run kernel's own decision function (98.15 ruling 5)
    redo = p10_pertdit.timing_decision(timing['timings'], cap_s=D_CAP_S)['decision']
    if (rec.get('timing_sha1') != sha1(tpath) or rec.get('timing_decision') not in ('run', 'run_capped')
            or rec.get('timing_decision') != redo):
        raise SystemExit(f'FATAL: (d5) record timing {rec.get("timing_sha1")} / {rec.get("timing_decision")!r} is not bound to '
                         f'p10d_timing.json ({sha1(tpath)}, which gives {redo!r})')
    if str(z['arm']) != 'pertdit' or str(z['stop_reason']) != rec.get('stop_reason'):
        raise SystemExit(f'FATAL: (d6) npz arm {str(z["arm"])!r} / stop_reason {str(z["stop_reason"])!r} disagree with the '
                         f'record ({rec.get("stop_reason")!r})')
    return {'manifest': mf, 'manifest_sha1': sha1(mf), 'npz_sha1': man['files'][f'p10_pertdit_seed{seed}.npz'],
            'stop_reason': rec['stop_reason'], 'timing_decision': redo, 'timing_sha1': sha1(tpath)}


def d_own_target_truth(bundle_data, bundle_idx):
    """98.15 ruling 1 (reported): PertDiT's own target, their transform (train_pertdit.py:144-159) recomputed from the bundle:
    clip(0, 1e3), normalize_total to the median positive row total over ALL bundle rows (zero rows left at zero, as scanpy
    does), log1p, separately for X and X_ctl; returned for the given rows."""
    def tr(a):
        a = np.clip(np.asarray(a, dtype=np.float64), 0, 1e3)
        tot = a.sum(axis=1)
        t = np.median(tot[tot > 0])
        return np.log1p(a * (t / (tot + (tot == 0)))[:, None])
    return (tr(bundle_data['X']) - tr(bundle_data['X_ctl']))[bundle_idx]


def _per_molecule_mean_of_medians(r, molecule):
    mols = np.unique(molecule)
    return float(np.mean([np.median(r[molecule == m]) for m in mols])), int(len(mols))


def _relabel(obj, label):
    if isinstance(obj, dict):
        if 'LABEL' in obj:
            obj['LABEL'] = label
        for v in obj.values():
            _relabel(v, label)
    elif isinstance(obj, list):
        for v in obj:
            _relabel(v, label)


def load_carves(carves_path=None, clean_rows=None):
    """Load and guard p10_carves.json distinct_control_stratum."""
    carves_path = resolve_path(carves_path or CARVES_JSON)
    if not os.path.exists(carves_path):
        raise SystemExit(f'FATAL: carves file {carves_path} not found')
    if sha1(carves_path) != CARVES_SHA1:
        raise SystemExit(f'FATAL: carves file sha1 differs from CARVES_SHA1 ({CARVES_SHA1})')

    with open(carves_path, encoding='utf-8') as f:
        carves_data = json.load(f)

    distinct_info = carves_data.get('distinct_control_stratum', {})
    distinct_rows = np.asarray(distinct_info.get('clean_distinct_row_index', []), dtype=np.int64)

    if len(distinct_rows) != EXPECT_N_DISTINCT:
        raise SystemExit(f'FATAL: distinct stratum row count {len(distinct_rows)} != EXPECT_N_DISTINCT ({EXPECT_N_DISTINCT})')
    d_sha1 = row_set_sha1(distinct_rows)
    if d_sha1 != EXPECT_SHA1_DISTINCT:
        raise SystemExit(f'FATAL: distinct stratum row-set sha1 {d_sha1} != EXPECT_SHA1_DISTINCT ({EXPECT_SHA1_DISTINCT})')

    if clean_rows is not None:
        clean_set = set(clean_rows)
        missing = [r for r in distinct_rows if r not in clean_set]
        if missing:
            raise SystemExit(f'FATAL: {len(missing)} distinct stratum rows not among clean scored rows')

    return distinct_rows, carves_path


def compute_f0(bundle_data, test_cells, test_ctl_true, split_key=SPLIT_KEY):
    """Compute f0 predictions from bundle training rows.

    For each cell c:
    x̄_ctl(c) = mean of X_ctl over cell c's 'train' rows
    Δ̄(c) = mean of (X - X_ctl) over cell c's 'train' rows
    For each test row of cell c: pred_Δ = -(X_ctl - x̄_ctl(c)) + Δ̄(c)

    A test row whose cell has no training row is excluded (valid_mask is False).
    """
    split_col = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in bundle_data[split_key]])
    train_mask = (split_col == 'train')
    train_cells = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in bundle_data['meta_cell'][train_mask]])
    train_ctl = np.asarray(bundle_data['X_ctl'][train_mask], dtype=np.float64)
    train_x = np.asarray(bundle_data['X'][train_mask], dtype=np.float64)
    train_delta = train_x - train_ctl

    mean_ctl_by_cell = {}
    mean_delta_by_cell = {}
    for c in np.unique(train_cells):
        c_mask = (train_cells == c)
        mean_ctl_by_cell[c] = np.mean(train_ctl[c_mask], axis=0)
        mean_delta_by_cell[c] = np.mean(train_delta[c_mask], axis=0)

    n_test = len(test_cells)
    n_genes = train_ctl.shape[1]
    pred_delta = np.full((n_test, n_genes), np.nan, dtype=np.float64)
    valid_mask = np.zeros(n_test, dtype=bool)

    test_ctl = np.asarray(test_ctl_true, dtype=np.float64)
    for c in np.unique(test_cells):
        if c in mean_ctl_by_cell:
            idx = np.flatnonzero(test_cells == c)
            pred_delta[idx] = -(test_ctl[idx] - mean_ctl_by_cell[c]) + mean_delta_by_cell[c]
            valid_mask[idx] = True

    n_rows_no_training_cell = int((~valid_mask).sum())
    return pred_delta, valid_mask, mean_ctl_by_cell, mean_delta_by_cell, n_rows_no_training_cell


def load_ref(ref_name, ref_dir, p9_data, bundle_data=None):
    """Load and guard reference predictions, aligning rows to P9's order."""
    if ref_name not in REFS_P10:
        raise SystemExit(f'FATAL: unknown reference {ref_name!r}; choices are {REFS_P10}')

    p9_ri = p9_data['row_index']
    y_target = p9_data['y']

    if ref_name in ('mlp_m', 'mlp_mdt'):
        arm = 'm' if ref_name == 'mlp_m' else 'mdt'
        if not ref_dir:
            ref_dir = os.path.join(REPO, 'external', 'kaggle_out', f'p10_{arm}')
        ref_dir = resolve_path(ref_dir)
        if not os.path.exists(ref_dir):
            raise SystemExit(f'FATAL: reference directory {ref_dir} not found')

        seed_files = [f'p10_{arm}_seed{s}.npz' for s in (0, 1, 2)]
        seeds_data = []
        found_seeds = []
        for sf in seed_files:
            p = os.path.join(ref_dir, sf)
            if not os.path.exists(p):
                raise SystemExit(f'FATAL: reference seed file {p} not found')
            z = np.load(p, allow_pickle=True)
            for k in ('delta_pred', 'row_index', 'seed', 'arm'):
                if k not in z:
                    raise SystemExit(f'FATAL: {sf} missing required key {k!r}')
            if str(z['arm']) != arm:
                raise SystemExit(f'FATAL: {sf} arm is {z["arm"]!r}, expected {arm!r}')
            found_seeds.append(int(z['seed']))
            seeds_data.append(z)

        if len(found_seeds) != 3 or set(found_seeds) != MLP_SEEDS:
            raise SystemExit(f'FATAL: seeds {found_seeds} != {MLP_SEEDS}')

        s0_ri = set(seeds_data[0]['row_index'])
        for idx, s in enumerate(seeds_data[1:], start=1):
            if set(s['row_index']) != s0_ri:
                raise SystemExit(f'FATAL: seed file {seed_files[idx]} has different row_index set')

        missing_p9 = [r for r in p9_ri if r not in s0_ri]
        if missing_p9:
            raise SystemExit(f'FATAL: reference is missing {len(missing_p9)} P9 rows')

        # Align each seed by P9 row_index
        aligned_deltas = []
        for s in seeds_data:
            pos_map = {idx: i for i, idx in enumerate(s['row_index'])}
            align_idx = np.array([pos_map[idx] for idx in p9_ri], dtype=np.int64)
            aligned_deltas.append(np.asarray(s['delta_pred'][align_idx], dtype=np.float64))

        r_ref_seeds = [per_row_pearson(d, y_target) for d in aligned_deltas]
        r_ref = np.mean(r_ref_seeds, axis=0)

        return {
            'r_ref': r_ref,
            'r_ref_seeds': r_ref_seeds,
            'pred_deltas': aligned_deltas,
            'ypred_seeds': [d + p9_data['ctl_true'] for d in aligned_deltas],
            'has_seeds': True,
            'ref_dir': ref_dir,
            'files': seed_files,
        }

    elif ref_name == 'v9_drugblind':
        if not ref_dir:
            ref_dir = os.path.join(REPO, 'external', 'kaggle_out', 'v9p10b1')
        ref_dir = resolve_path(ref_dir)
        if not os.path.exists(ref_dir):
            raise SystemExit(f'FATAL: reference directory {ref_dir} not found')

        mf = os.path.join(ref_dir, 'P10B1_COMPLETE.json')
        if not os.path.exists(mf):
            raise SystemExit(f'FATAL: no P10B1_COMPLETE.json in {ref_dir}')
        with open(mf, encoding='utf-8') as f:
            manifest = json.load(f)
        if not manifest.get('complete') or manifest.get('n_seeds') != 3:
            raise SystemExit(f'FATAL: manifest incomplete: {manifest}')
        for name, h in manifest.get('files', {}).items():
            p = os.path.join(ref_dir, name)
            if not os.path.exists(p) or sha1(p) != h:
                raise SystemExit(f'FATAL: {name} missing or sha1 mismatch with manifest')

        seed_files = sorted(n for n in manifest.get('files', {}) if re.fullmatch(r'v9p10b1_seed\d\.npz', n))
        if len(seed_files) != 3:
            raise SystemExit(f'FATAL: expected 3 seed files, manifest has {seed_files!r}')
        # review 076 C1: the manifest must record a DRUG-BLIND run of P9's V2 stack, and the V2 snapshot identities must hold
        stack = manifest.get('stack', '')
        if '--drug_blind' not in manifest.get('argv', []) or 'DRUG-BLIND' not in stack or 'V2' not in stack:
            raise SystemExit(f'FATAL: P10B1_COMPLETE.json does not record a drug-blind V2 run (argv / stack {stack!r}; review 076 C1)')
        check_snapshot_identities(ref_dir, seed_files)

        seeds_data = []
        for s_name in seed_files:
            z = np.load(os.path.join(ref_dir, s_name), allow_pickle=True)
            for k in ('y_pred', 'y_true', 'ctl_true', 'row_index'):
                if k not in z:
                    raise SystemExit(f'FATAL: {s_name} missing required key {k!r}')
            seeds_data.append(z)

        s0 = seeds_data[0]
        for s in seeds_data[1:]:
            for k in ('row_index', 'y_true', 'ctl_true'):
                if not np.array_equal(s[k], s0[k]):
                    raise SystemExit(f'FATAL: v9_drugblind seeds disagree on {k!r}')

        s0_ri = set(s0['row_index'])
        missing_p9 = [r for r in p9_ri if r not in s0_ri]
        if missing_p9:
            raise SystemExit(f'FATAL: v9_drugblind is missing {len(missing_p9)} P9 rows')

        pos_map = {idx: i for i, idx in enumerate(s0['row_index'])}
        align_idx = np.array([pos_map[idx] for idx in p9_ri], dtype=np.int64)

        aligned_ytrue = np.asarray(s0['y_true'][align_idx], dtype=np.float64)
        aligned_ctltrue = np.asarray(s0['ctl_true'][align_idx], dtype=np.float64)

        dmax_y = float(np.max(np.abs(aligned_ytrue - p9_data['y_true'])))
        dmax_ctl = float(np.max(np.abs(aligned_ctltrue - p9_data['ctl_true'])))
        if dmax_y > 1e-4:
            raise SystemExit(f'FATAL: v9_drugblind y_true differs from P9 by {dmax_y:.3g} > 1e-4')
        if dmax_ctl > 1e-4:
            raise SystemExit(f'FATAL: v9_drugblind ctl_true differs from P9 by {dmax_ctl:.3g} > 1e-4')

        aligned_ypred = [np.asarray(s['y_pred'][align_idx], dtype=np.float64) for s in seeds_data]
        pred_deltas = [yp - aligned_ctltrue for yp in aligned_ypred]

        r_ref_seeds = [per_row_pearson(d, y_target) for d in pred_deltas]
        r_ref = np.mean(r_ref_seeds, axis=0)

        return {
            'r_ref': r_ref,
            'r_ref_seeds': r_ref_seeds,
            'pred_deltas': pred_deltas,
            'ypred_seeds': aligned_ypred,
            'has_seeds': True,
            'ref_dir': ref_dir,
            'files': seed_files,
            'manifest': manifest,
        }

    elif ref_name in D_REFS:
        if not ref_dir:
            ref_dir = os.path.join(REPO, 'external', 'kaggle_out', 'p10_d')
        ref_dir = resolve_path(ref_dir)
        if not os.path.exists(ref_dir):
            raise SystemExit(f'FATAL: reference directory {ref_dir} not found')
        fn = f'p10_pertdit_seed{D_SEEDS[ref_name]}.npz'
        p = os.path.join(ref_dir, fn)
        if not os.path.exists(p):
            raise SystemExit(f'FATAL: reference file {p} not found')
        z = np.load(p, allow_pickle=False)
        for k in ('row_index', 'delta_pred', 'delta_pred_t', 'ctrl_t', 'seed', 'arm', 'stop_reason'):
            if k not in z:
                raise SystemExit(f'FATAL: {fn} missing required key {k!r}')
        pos_map = {int(idx): i for i, idx in enumerate(z['row_index'])}
        missing_p9 = [r for r in p9_ri if int(r) not in pos_map]
        if missing_p9:
            raise SystemExit(f'FATAL: {fn} is missing {len(missing_p9)} P9 rows')
        align_idx = np.array([pos_map[int(idx)] for idx in p9_ri], dtype=np.int64)
        delta = np.asarray(z['delta_pred'][align_idx], dtype=np.float64)
        return {
            'r_ref': per_row_pearson(delta, y_target),
            'r_ref_seeds': None,
            'pred_deltas': [delta],
            'ypred_seeds': [delta + p9_data['ctl_true']],
            'has_seeds': False,
            'ref_dir': ref_dir,
            'files': [fn],
            'delta_pred_t': np.asarray(z['delta_pred_t'][align_idx], dtype=np.float64),
        }

    elif ref_name == 'f0':
        pred_delta, valid_mask, mean_ctl_by_cell, mean_delta_by_cell, n_no_train = compute_f0(
            bundle_data or p9_data['bundle_data'], p9_data['cell'], p9_data['ctl_true'], SPLIT_KEY
        )
        r_ref = per_row_pearson(pred_delta, y_target)
        r_ref[~valid_mask] = np.nan

        return {
            'r_ref': r_ref,
            'r_ref_seeds': None,
            'pred_deltas': [pred_delta],
            'ypred_seeds': [pred_delta + p9_data['ctl_true']],
            'valid_mask': valid_mask,
            'mean_ctl_by_cell': mean_ctl_by_cell,
            'mean_delta_by_cell': mean_delta_by_cell,
            'n_rows_no_training_cell': n_no_train,
            'has_seeds': False,
        }


def compute_control_noise_free_target(p9_data, ref_data, mean_ctl_by_cell, n_boot=N_BOOT):
    """Compute control-noise-free target readings (RESULTS 98.6 C2c).

    y' = X - x̄_ctl(c)
    pred' = (ctl + pred_Δ) - x̄_ctl(c)
    Per-row Pearson of pred' against y' for v9 (mean over 3 seeds) and for ref.
    Molecule reading on clean rows; rows of cells without training rows are excluded and counted.
    """
    cell = p9_data['cell']
    clean_mask = p9_data['clean_mask']
    molecule = p9_data['molecule']

    has_train_cell = np.array([c in mean_ctl_by_cell for c in cell], dtype=bool)
    n_rows_no_training_cell = int((clean_mask & ~has_train_cell).sum())

    x_bar_ctl = np.full_like(p9_data['y_true'], np.nan, dtype=np.float64)
    for c in np.unique(cell):
        if c in mean_ctl_by_cell:
            idx = np.flatnonzero(cell == c)
            x_bar_ctl[idx] = mean_ctl_by_cell[c]

    y_prime = np.asarray(p9_data['y_true'], dtype=np.float64) - x_bar_ctl

    # v9 predictions
    v9_seeds_pred_prime = [np.asarray(s['y_pred'], dtype=np.float64) - x_bar_ctl for s in p9_data['seeds_data']]
    r_v9_seeds_prime = [per_row_pearson(p, y_prime) for p in v9_seeds_pred_prime]
    r_v9_prime = np.mean(r_v9_seeds_prime, axis=0)

    # Reference predictions
    if ref_data.get('has_seeds', False):
        ref_seeds_pred_prime = [np.asarray(p, dtype=np.float64) - x_bar_ctl for p in ref_data['ypred_seeds']]
        r_ref_seeds_prime = [per_row_pearson(p, y_prime) for p in ref_seeds_pred_prime]
        r_ref_prime = np.mean(r_ref_seeds_prime, axis=0)
    else:
        # f0: pred_prime = (ctl + pred_Δ) - x̄_ctl(c) = mean_delta(c)
        ref_pred_prime = np.asarray(ref_data['ypred_seeds'][0], dtype=np.float64) - x_bar_ctl
        r_ref_prime = per_row_pearson(ref_pred_prime, y_prime)
        r_ref_seeds_prime = [r_ref_prime]

    r_v9_prime[~has_train_cell] = np.nan
    r_ref_prime[~has_train_cell] = np.nan

    valid_clean = clean_mask & has_train_cell & np.isfinite(r_v9_prime) & np.isfinite(r_ref_prime)
    for rs in r_v9_seeds_prime:
        valid_clean &= np.isfinite(rs)
    if ref_data.get('has_seeds', False):
        for rs in r_ref_seeds_prime:
            valid_clean &= np.isfinite(rs)

    n_dropped = int((clean_mask & ~valid_clean).sum())

    reading = molecule_reading(r_v9_prime[valid_clean], r_ref_prime[valid_clean], molecule[valid_clean], n_boot=n_boot, seed=0)
    reading['LABEL'] = REPORTED_LABEL
    reading['n_rows_no_training_cell'] = n_rows_no_training_cell
    reading['n_dropped'] = n_dropped
    return reading, r_v9_prime, r_ref_prime


def compute_cell_centred_reading(p9_data, ref_data, n_boot=N_BOOT):
    """Compute cell-centred version (review 071 diagnostic, reported).

    Subtract each test cell's mean of both y and pred across the scored rows
    of that cell from each row before the Pearson.
    Then take molecule reading on clean rows.
    """
    cell = p9_data['cell']
    clean_mask = p9_data['clean_mask']
    molecule = p9_data['molecule']

    y = np.asarray(p9_data['y'], dtype=np.float64)
    y_centred = np.empty_like(y)

    v9_pred_deltas = [np.asarray(s['y_pred'] - p9_data['ctl_true'], dtype=np.float64) for s in p9_data['seeds_data']]
    v9_centred_deltas = [np.empty_like(d) for d in v9_pred_deltas]

    ref_pred_deltas = [np.asarray(d, dtype=np.float64) for d in ref_data['pred_deltas']]
    ref_centred_deltas = [np.empty_like(d) for d in ref_pred_deltas]

    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        y_centred[idx] = y[idx] - np.mean(y[idx], axis=0)
        for s_idx, d in enumerate(v9_pred_deltas):
            v9_centred_deltas[s_idx][idx] = d[idx] - np.mean(d[idx], axis=0)
        for r_idx, d in enumerate(ref_pred_deltas):
            ref_centred_deltas[r_idx][idx] = d[idx] - np.mean(d[idx], axis=0)

    r_v9_seeds_c = [per_row_pearson(d, y_centred) for d in v9_centred_deltas]
    r_v9_c = np.mean(r_v9_seeds_c, axis=0)

    if ref_data.get('has_seeds', False):
        r_ref_seeds_c = [per_row_pearson(d, y_centred) for d in ref_centred_deltas]
        r_ref_c = np.mean(r_ref_seeds_c, axis=0)
    else:
        r_ref_c = per_row_pearson(ref_centred_deltas[0], y_centred)
        r_ref_seeds_c = [r_ref_c]

    valid_clean = clean_mask & np.isfinite(r_v9_c) & np.isfinite(r_ref_c)
    for rs in r_v9_seeds_c:
        valid_clean &= np.isfinite(rs)
    if ref_data.get('has_seeds', False):
        for rs in r_ref_seeds_c:
            valid_clean &= np.isfinite(rs)

    reading = molecule_reading(r_v9_c[valid_clean], r_ref_c[valid_clean], molecule[valid_clean], n_boot=n_boot, seed=0)
    reading['LABEL'] = REPORTED_LABEL
    return reading, r_v9_c, r_ref_c


def score_p10(p9_dir, ref, ref_dir=None, dry_check=False, n_boot=N_BOOT, carves_path=None, bundle_path=None, units_path=None,
              clean_path=None, d_truncated_reported=False):
    """Run full P10 scoring against reference model."""
    if ref not in REFS_P10:
        raise SystemExit(f'FATAL: unknown reference {ref!r}; choices are {REFS_P10}')

    out_json = os.path.join(REPO, 'model', 'results', f'p10_accuracy_{ref}.json')
    if os.path.exists(out_json):
        raise SystemExit(f'FATAL: {out_json} exists -- touched once')

    # Step 1-5: load P9 and inputs
    p9_data = load_p9_and_metadata(p9_dir, bundle_path=bundle_path, units_path=units_path, clean_path=clean_path)

    # Step 5b: load carves
    distinct_rows, c_path = load_carves(carves_path=carves_path, clean_rows=p9_data['clean_rows'])

    # Step 4: load reference (and, for M / M+dt, its download provenance; review 075 C2)
    ref_data = load_ref(ref, ref_dir, p9_data, bundle_data=p9_data['bundle_data'])
    provenance = (check_m_provenance(ref_data['ref_dir'], 'm' if ref == 'mlp_m' else 'mdt')
                  if ref in ('mlp_m', 'mlp_mdt') else None)
    d_truncated = False
    if ref in D_REFS:
        provenance = check_d_provenance(ref_data['ref_dir'], ref)
        if provenance['stop_reason'] == 'cap':
            if ref == 'pertdit' and not d_truncated_reported:
                raise SystemExit('FATAL: D stopped at the 6 GPU-h cap: its reading is "not run (cost)" (98.15 ruling 5); '
                                 'pass --d_truncated_reported to score it as reported only')
            d_truncated = True

    if dry_check:
        print('P10 inputs verified')
        return None

    r_v9 = p9_data['r_v9']
    r_ref = ref_data['r_ref']
    r_v9_seeds = p9_data['r_seed']
    molecule = p9_data['molecule']
    clean_mask = p9_data['clean_mask']
    row_index = p9_data['row_index']

    finite_mask = np.isfinite(r_v9) & np.isfinite(r_ref)
    for rs in r_v9_seeds:
        finite_mask &= np.isfinite(rs)
    if ref_data.get('has_seeds', False):
        for rs in ref_data['r_ref_seeds']:
            finite_mask &= np.isfinite(rs)

    n_total = len(row_index)
    n_dropped_full = int((~finite_mask).sum())
    if n_dropped_full / n_total > MAX_DROPPED_FRAC:
        raise SystemExit(
            f'FATAL: dropped {n_dropped_full} rows ({n_dropped_full / n_total:.3%} > MAX_DROPPED_FRAC {MAX_DROPPED_FRAC:.3%}) due to non-finite scores'
        )

    clean_finite = clean_mask & finite_mask
    diff_clean = r_v9[clean_finite] - r_ref[clean_finite]

    ref_label = REF_LABELS.get(ref, ref)

    # Reading 1: of record on clean subset
    of_rec = molecule_reading(r_v9[clean_finite], r_ref[clean_finite], molecule[clean_finite], n_boot=n_boot, seed=0)
    of_rec['verdict'] = verdict(of_rec, ref_label)
    if ref == 'mlp_m':
        of_rec['LABEL'] = R1_LABEL
    elif ref == 'v9_drugblind':
        of_rec['LABEL'] = R2_LABEL
    elif ref == 'pertdit':
        of_rec['LABEL'] = D_LABEL
    else:
        of_rec['LABEL'] = REPORTED_LABEL

    # Reading 2: full rows ("as defined")
    full_reading = molecule_reading(r_v9[finite_mask], r_ref[finite_mask], molecule[finite_mask], n_boot=n_boot, seed=0)
    full_reading['verdict'] = verdict(full_reading, ref_label)
    full_reading['LABEL'] = REPORTED_LABEL

    # Reading 3: per seed of v9
    per_seed_list = []
    for s_idx, rs in enumerate(r_v9_seeds):
        s_reading = molecule_reading(rs[clean_finite], r_ref[clean_finite], molecule[clean_finite], n_boot=n_boot, seed=0)
        s_reading['verdict'] = verdict(s_reading, ref_label)
        s_reading['seed'] = s_idx
        s_reading['LABEL'] = REPORTED_LABEL
        per_seed_list.append(s_reading)

    # Reading 4: row-pooled means
    row_pooled = {
        'LABEL': REPORTED_LABEL,
        'mean_r_v9': float(np.mean(r_v9[clean_finite])),
        'mean_r_ref': float(np.mean(r_ref[clean_finite])),
        'paired_mean': float(np.mean(diff_clean)),
        'paired_mean_ci95_row_bootstrap': boot_ci_rows(diff_clean, n_boot, seed=0),
        'n_rows': int(clean_finite.sum()),
    }

    # Reading 5: distinct-control stratum
    distinct_mask = np.isin(row_index, distinct_rows) & clean_finite
    distinct_reading = molecule_reading(r_v9[distinct_mask], r_ref[distinct_mask], molecule[distinct_mask], n_boot=n_boot, seed=0)
    distinct_reading['verdict'] = verdict(distinct_reading, ref_label)
    distinct_reading['LABEL'] = REPORTED_LABEL
    distinct_reading['n_rows'] = int(distinct_mask.sum())

    # Reading 6: control-noise-free target
    if ref == 'f0':
        mean_ctl_by_cell = ref_data['mean_ctl_by_cell']
    else:
        _, _, mean_ctl_by_cell, _, _ = compute_f0(
            p9_data['bundle_data'], p9_data['cell'], p9_data['ctl_true'], SPLIT_KEY
        )
    noise_free_reading, _, _ = compute_control_noise_free_target(p9_data, ref_data, mean_ctl_by_cell, n_boot=n_boot)
    noise_free_reading['verdict'] = verdict(noise_free_reading, ref_label)

    # Reading 7: cell-centred version
    cell_centred_reading, _, _ = compute_cell_centred_reading(p9_data, ref_data, n_boot=n_boot)
    cell_centred_reading['verdict'] = verdict(cell_centred_reading, ref_label)

    inputs = {
        'manifest': {
            'path': p9_data['manifest_path'],
            'sha1': sha1(p9_data['manifest_path']),
            'stack': p9_data['manifest'].get('stack', ''),
            'files': {f: sha1(os.path.join(p9_dir, f)) for f in p9_data['manifest'].get('files', {})},
        },
        'bundle': {
            'path': p9_data['bundle_path'],
            'sha1': sha1(p9_data['bundle_path']),
        },
        'units': {
            'path': p9_data['units_path'],
            'sha1': sha1(p9_data['units_path']),
        },
        'clean': {
            'path': p9_data['clean_path'],
            'sha1': sha1(p9_data['clean_path']),
        },
        'carves': {
            'path': c_path,
            'sha1': sha1(c_path),
        },
        'ref': {
            'name': ref,
            'label': ref_label,
            'dir': ref_data.get('ref_dir'),
            'provenance': provenance,
            'files_sha1': ({f: sha1(os.path.join(ref_data['ref_dir'], f)) for f in ref_data.get('files', [])}
                           if ref_data.get('ref_dir') else None),
        },
    }

    out_data = {
        'inputs': inputs,
        'ref': ref,
        'ref_label': ref_label,
        'estimand': 'Per-molecule median difference of per-row Pearson correlation against reference, with cluster bootstrap over molecules (RESULTS §96.3 / §98.3 / §98.6).',
        'of_record': of_rec,
        'full': full_reading,
        'per_seed_v9': per_seed_list,
        'row_pooled': row_pooled,
        'distinct_control_stratum': distinct_reading,
        'control_noise_free_target': noise_free_reading,
        'cell_centred': cell_centred_reading,
        'subsets': {
            'clean': {
                'of_record': of_rec,
                'per_seed': per_seed_list,
                'row_pooled': row_pooled,
                'distinct_control_stratum': distinct_reading,
                'control_noise_free_target': noise_free_reading,
                'cell_centred': cell_centred_reading,
            },
            'full': {
                'of_record': full_reading,
            },
        },
    }

    if ref in ('mlp_mdt', 'f0'):
        out_data['LABEL'] = REPORTED_LABEL

    if ref in D_REFS:
        # (a) PertDiT on its own target, as Bai et al. score it (98.15 ruling 1)
        truth_t = d_own_target_truth(p9_data['bundle_data'], p9_data['bundle_idx'])
        r_own = per_row_pearson(ref_data['delta_pred_t'], truth_t)
        own_ok = clean_mask & np.isfinite(r_own)
        own_mol, own_nmol = _per_molecule_mean_of_medians(r_own[own_ok], molecule[own_ok])
        # review 082 C3: their own metric too (evaluate_all.pcc_top_k_deg, top 100 genes by |truth| per row), vendored
        import importlib.util
        spec = importlib.util.spec_from_file_location('bai_evaluate_all', BAI_EVALUATE_ALL)
        bai_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bai_eval)
        pcc_deg = bai_eval.pcc_top_k_deg(truth_t[clean_mask], ref_data['delta_pred_t'][clean_mask], k=100)
        # (b) the caption-collision stratum (98.20, review 081 C5)
        c_path = resolve_path(COLLISION_JSON)
        if not os.path.exists(c_path) or sha1(c_path) != COLLISION_SHA1:
            raise SystemExit(f'FATAL: caption-collision JSON missing or sha1 differs from COLLISION_SHA1 ({COLLISION_SHA1})')
        with open(c_path, encoding='utf-8') as f:
            coll = json.load(f)
        coll_rows = np.asarray(coll['row_index'], dtype=np.int64)
        if row_set_sha1(coll_rows) != coll.get('rows_sha1'):
            raise SystemExit('FATAL: caption-collision rows do not match their rows_sha1')
        if not set(coll_rows.tolist()) <= set(p9_data['clean_rows'].tolist()):
            raise SystemExit('FATAL: caption-collision rows are not all clean rows')
        in_coll = np.isin(row_index, coll_rows)
        parts = {}
        for name, m in (('collision', clean_finite & in_coll), ('rest', clean_finite & ~in_coll)):
            rd = molecule_reading(r_v9[m], r_ref[m], molecule[m], n_boot=n_boot, seed=0)
            rd['verdict'] = verdict(rd, ref_label)
            rd.update(LABEL=REPORTED_LABEL, n_rows=int(m.sum()), n_molecules=int(len(np.unique(molecule[m]))),
                      row_pooled_r_ref=float(np.mean(r_ref[m])), row_pooled_r_v9=float(np.mean(r_v9[m])))
            parts[name] = rd
        out_data['d_reported'] = {
            'LABEL': REPORTED_LABEL,
            'own_target': {'LABEL': REPORTED_LABEL,
                           'what': "per-row Pearson (P9's, 978 genes) on PertDiT's own (transformed) target; and Bai et al.'s "
                                   "PCC_DEG (evaluate_all.pcc_top_k_deg, top 100 genes by |truth|) on the same arrays",
                           'row_pooled_mean_r': float(np.mean(r_own[own_ok])),
                           'per_molecule_mean_of_medians': own_mol, 'n_rows': int(own_ok.sum()), 'n_molecules': own_nmol,
                           'pcc_deg_top100_row_pooled': float(np.mean(pcc_deg)), 'n_rows_pcc_deg': int(clean_mask.sum())},
            'caption_collision': dict(parts, LABEL=REPORTED_LABEL, collision_json_sha1=COLLISION_SHA1),
            'oracle_inverse_r': ORACLE_INVERSE_R,
            'stop_reason': provenance['stop_reason'],
            'timing_decision': provenance['timing_decision'],
        }
        if d_truncated:
            _relabel(out_data, D_TRUNCATED_LABEL)
            out_data['LABEL'] = D_TRUNCATED_LABEL
        elif ref == 'pertdit_s118':
            _relabel(out_data, REPORTED_LABEL)
            out_data['LABEL'] = REPORTED_LABEL

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

    return out_data


READ_KEYS = ('mean_d', 'ci_lo', 'ci_hi', 'ci_width', 'frac_favour_ours', 'frac_favour_ref', 'sign_p', 'verdict')


def _load_marked(path, want_ref):
    """Provenance guard: the marker's sha1 must match, and the JSON must be the scorer's output for want_ref."""
    m_path = str(path) + '.marker'
    if not os.path.exists(m_path):
        raise SystemExit(f'FATAL: marker file {m_path} not found')
    with open(m_path, encoding='utf-8') as f:
        m = json.load(f)
    if not m.get('complete') or m.get('sha1') != sha1(path):
        raise SystemExit(f'FATAL: marker sha1 mismatch for {path}')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('ref') != want_ref:
        raise SystemExit(f'FATAL: {path} is the reading for ref {data.get("ref")!r}, not {want_ref!r} (RESULTS 98.6)')
    return data


def _decide(data):
    """One of 'uninformative', 'pass', 'mean_le_0', 'otherwise', from the of-record block ONLY (98.6 C1).

    The verdict is re-derived from the stored numbers with score_p9.verdict and must equal the stored one."""
    rec = data.get('of_record')
    if not isinstance(rec, dict) or any(k not in rec for k in READ_KEYS):
        raise SystemExit('FATAL: of_record block missing or incomplete; needs %s' % (READ_KEYS,))
    v = verdict(rec, data.get('ref_label', data['ref']))
    if v != rec['verdict']:
        raise SystemExit(f'FATAL: stored verdict {rec["verdict"]!r} disagrees with its numbers ({v!r})')
    if v.startswith('UNINFORMATIVE'):
        return 'uninformative'
    if v.startswith('v9 predicts unseen compounds better than'):
        return 'pass'
    if rec['mean_d'] <= 0:
        return 'mean_le_0'
    return 'otherwise'


R1_SENTENCE = {
    'pass': 'v9 beats a drug-free model on unseen compounds',
    'mean_le_0': 'v9 does not beat a drug-free model on unseen compounds',
    'otherwise': 'v9 is not separable from a drug-free model on unseen compounds at this power',
    'uninformative': 'UNINFORMATIVE: the v9-vs-drug-free cluster CI is wider than 0.10',
}
R2_SENTENCE = {
    'pass': "v9 needs the compound on unseen compounds; Bai et al.'s finding does not hold for v9",
    'mean_le_0': ("no detectable dependence of v9's unseen-compound accuracy on the compound: its contribution is at most "
                  "{ci_hi:+.3f} per molecule (95 % cluster CI)"),   # review 075 C4: the bound, not 'does not depend'
    'otherwise': "not separable from v9's drug-blind twin at this power",
    'uninformative': 'UNINFORMATIVE',
}


def read_p10(r1_json, r2_json=None):
    """RESULTS 98.6's 2 x 2, mechanical.

    R1 (p10_accuracy_mlp_m.json) is a benchmark statement only. R2 (p10_accuracy_v9_drugblind.json) carries the
    compound-use inference; restate_p9_headline is True ONLY when R2's mean <= 0 (and the CI is informative). Only the
    of_record block is opened; every reported-only block is ignored."""
    d1 = _decide(_load_marked(r1_json, 'mlp_m'))
    out = {'r1_case': d1, 'r1': R1_SENTENCE[d1] + ' (benchmark statement only)'}
    if r2_json is None:
        out.update(r2_case=None, r2='pending (B1)', restate_p9_headline=None)
    else:
        data2 = _load_marked(r2_json, 'v9_drugblind')
        d2 = _decide(data2)
        out.update(r2_case=d2, r2=R2_SENTENCE[d2].format(ci_hi=data2['of_record']['ci_hi']),
                   restate_p9_headline=(d2 == 'mean_le_0'))
    return out


D_STATEMENT_NO_CLAIM = 'NO COMPOUND-LEVEL CLAIM: v9 and PertDiT (our run of their harness) are not separable at this power'


def read_p10_d(d_json):
    """The D reading (98.3), mechanical: of-record block only, verdict re-derived from its numbers (as _decide), and a
    truncated or s118 JSON refused. Every d_reported block is ignored."""
    data = _load_marked(d_json, 'pertdit')
    rec = data.get('of_record')
    if not isinstance(rec, dict) or rec.get('LABEL') != D_LABEL:
        raise SystemExit(f'FATAL: {d_json} is not a D reading of record (LABEL {None if not isinstance(rec, dict) else rec.get("LABEL")!r})')
    case = _decide(data)
    label = REF_LABELS['pertdit']
    v = verdict(rec, label)
    if case == 'uninformative':
        statement = 'UNINFORMATIVE: cluster CI wider than 0.10; no compound-level claim'
    elif v == f'v9 predicts unseen compounds better than {label}':
        statement = v
    elif v == f'{label} predicts unseen compounds better than v9':
        statement = v
    elif v == 'NO COMPOUND-LEVEL CLAIM':
        statement = D_STATEMENT_NO_CLAIM
    else:
        raise SystemExit(f'FATAL: unrecognised verdict {v!r}')
    return {'d_case': case, 'd': statement}


def main(argv=None):
    ap = argparse.ArgumentParser(description='RESULTS §98.3 / §98.6: P10 accuracy reading against reference models.')
    ap.add_argument('--p9_dir', help='Directory holding P9 kernel outputs and P9_COMPLETE.json')
    ap.add_argument('--ref', choices=REFS_P10, help='Reference model to score against')
    ap.add_argument('--ref_dir', default=None, help='Directory holding reference predictions (mlp_m, mlp_mdt, v9_drugblind)')
    ap.add_argument('--dry_check', action='store_true', help='Verify all input guards and stop')
    ap.add_argument('--read', nargs='+', default=None, metavar=('R1_JSON', 'R2_JSON'), help='Read and interpret P10 results: R1_JSON [R2_JSON]')
    ap.add_argument('--read_d', default=None, metavar='D_JSON', help='Read the D head-to-head (p10_accuracy_pertdit.json)')
    ap.add_argument('--d_truncated_reported', action='store_true',
                    help='Score a cap-truncated D run as reported only (its reading is "not run (cost)")')
    args = ap.parse_args(argv)

    if args.read is not None:
        if len(args.read) not in (1, 2):
            raise SystemExit('FATAL: --read expects 1 or 2 JSON paths: R1_JSON [R2_JSON]')
        r1_path = args.read[0]
        r2_path = args.read[1] if len(args.read) > 1 else None
        res = read_p10(r1_path, r2_path)
        print(json.dumps(res, indent=2))
        return

    if args.read_d is not None:
        print(json.dumps(read_p10_d(args.read_d), indent=2))
        return

    if not args.p9_dir or not args.ref:
        ap.error('Both --p9_dir and --ref are required when not using --read')

    score_p10(p9_dir=args.p9_dir, ref=args.ref, ref_dir=args.ref_dir, dry_check=args.dry_check,
              d_truncated_reported=args.d_truncated_reported)


if __name__ == '__main__':
    main()
