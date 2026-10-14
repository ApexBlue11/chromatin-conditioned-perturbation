# -*- coding: utf-8 -*-
"""RESULTS 99 / model/V10_AUDIT_AND_PLAN.md §5: Phase 1's rescoring tests, run ONCE each after review (packet 084).

  C1   the within-plate contrast on P9's unseen-compound rows: v9 (3 seeds) against M+dt, per molecule, clean subset,
       score_p9.verdict -- the reading of record, a BENCHMARK statement (review 083 C2: dose alone earns part of the
       contrast, so "uses the compound" is C3's). Reported: per seed, v9 against 0, M (0 by construction), the
       contrast-magnitude ratio, the zero-parameter DOSE-ONLY floor (review 083 C2), and strata by the mean ECFP4
       Tanimoto of each row's plate-mates (review 083 ask 2).
  C1b  the oracle-plate version: (truth - P_meas) against (prediction - P_meas), P_meas = measured mean delta of the
       row's other-compound plate-mates in ANY split. Reported.
  C1c  the zero-parameter training-plate predictor (mean delta of the other-compound TRAINING wells on the plate), on
       the ORIGINAL target, beside v9 and M on the same rows. Reported.
  C2a  unseen cells (P7 rows): F_tr = -(X_ctl - the cell's mean control over its test rows) and F_ind = -(X_ctl - the
       mean training control), per cell, beside v9 (P7) and XPert (O2). Reported.
  C2c  the within-plate contrast on P7's rows, v9 against XPert, by cell (coldcell_h2h's estimand). Reported.
  C4   dose coupling: per (compound, cell, time) with >= 3 doses, Spearman(dose, |y - P_meas|) and
       Spearman(dose, |yhat - P_meas|) for v9 and M+dt. Reported.

THE WITHIN-PLATE CONTRAST (binding, §5): for a quantity v on scored row i,
    c_i = v_i - mean(v_j) over the SCORED rows j on i's plate whose compound differs from i's, required >= 2 such j;
the plate proxy is the control group (byte-identical X_ctl, the bundle has no plate key). Per-row Pearson of
c(prediction) against c(truth); a prediction contrast with zero variance scores 0 and is counted.

Every test reads saved predictions only; nothing is trained or inferred here. C2b and C3 (inference) are
audit99_infer.py; their readers are read_c2b / read_c3 below. Every no-claim statement carries its CI bound
(review 083 C4).

Run:  python model/v9/audit99_phase1.py --test C1 [--out_dir model/results/audit99]
      (each test refuses to run if its output already exists: run once.)
"""
import argparse, hashlib, json, os, sys

import numpy as np
import scipy.stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))

import score_p9
import score_p10
from audit99_target_sources import control_ids, group_ids, group_sum

OUT_DIR = os.path.join(REPO, 'model', 'results', 'audit99')
P9_DIR = os.path.join(REPO, 'external', 'kaggle_out', 'v9p9')
P7_DIR = os.path.join(REPO, 'external', 'kaggle_out', 'v9p7')
O2 = os.path.join(REPO, 'external', 'kaggle_out', 'cc1_v9', 'xpert_trained_split_cold_cell_1_final_test_profile.npy')
BUNDLE = os.path.join(REPO, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz')
MIN_MATES = 2
N_BOOT = 20000
C1_STATEMENTS = {
    'v9 predicts unseen compounds better than M+dt':
        'v9 beats a drug-free model with dose and time on the within-plate contrast (a benchmark statement; '
        'whether v9 uses the compound is C3)',
    'NO COMPOUND-LEVEL CLAIM':
        'v9 is not shown to beat a drug-free model with dose and time on the within-plate contrast: its margin is at '
        'most {ci_hi:+.3f} per molecule (95 % cluster CI)',
    'M+dt predicts unseen compounds better than v9':
        'a drug-free model with dose and time beats v9 on the within-plate contrast',
    'UNINFORMATIVE: cluster CI wider than 0.10; no compound-level claim':
        'uninformative: the cluster CI is wider than 0.10',
}
DRUG_INDEX = os.path.join(REPO, 'drug', 'outputs', 'drug_feature_index.json')
DRUG_FP = os.path.join(REPO, 'drug', 'outputs', 'drug_fingerprints.npy')
TESTS = ('C1', 'C1b', 'C1c', 'C2a', 'C2b', 'C2c', 'C3', 'C4')


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


ZERO_REL = 1e-4   # a prediction contrast whose norm is below 1e-4 of the truth contrast's is "zero variance" (rounding)


def rowr_pred(pred, truth):
    """Per-row Pearson. A prediction row with (effectively) zero variance scores 0 (registered, §5); "effectively" means
    its centred norm is below ZERO_REL of the truth row's, so a model that predicts the same response for every compound
    on a plate is not scored on float rounding. A truth row with zero variance is NaN."""
    a = pred - pred.mean(1, keepdims=True)
    t = truth - truth.mean(1, keepdims=True)
    aa, tt = (a * a).sum(1), (t * t).sum(1)
    r = np.full(len(a), np.nan)
    zero = (tt > 0) & (aa <= (ZERO_REL ** 2) * tt)
    ok = (tt > 0) & ~zero
    r[ok] = (a[ok] * t[ok]).sum(1) / np.sqrt(aa[ok] * tt[ok])
    r[zero] = 0.0
    return r, int(zero.sum())


def plate_contrast(V, plate, pert, min_mates=MIN_MATES):
    """c_i = V_i - mean(V_j) over rows j (of THESE rows) on i's plate with a different compound. NaN where < min_mates."""
    g_p = group_ids(plate)
    g_pp = group_ids(plate, pert)
    Sp, n_p = group_sum(g_p, V)
    Spp, n_pp = group_sum(g_pp, V)
    cnt = n_p[g_p] - n_pp[g_pp]
    with np.errstate(invalid='ignore', divide='ignore'):
        mates = (Sp[g_p] - Spp[g_pp]) / cnt[:, None]
    C = V - mates
    C[cnt < min_mates] = np.nan
    return C, cnt


def other_mean(V_all, plate_all, pert_all, sel_all, rows, min_mates=MIN_MATES):
    """For each row in `rows` (indices into the *_all arrays): the mean of V_all over rows j with sel_all[j], on the same
    plate, with a different compound. NaN where < min_mates."""
    w = sel_all.astype(np.float64)
    g_p = group_ids(plate_all)
    g_pp = group_ids(plate_all, pert_all)
    Vw = V_all * w[:, None]
    Sp, _ = group_sum(g_p, Vw)
    Spp, _ = group_sum(g_pp, Vw)
    n_p = np.bincount(g_p, weights=w)
    n_pp = np.bincount(g_pp, weights=w)
    cnt = n_p[g_p[rows]] - n_pp[g_pp[rows]]
    with np.errstate(invalid='ignore', divide='ignore'):
        M = (Sp[g_p[rows]] - Spp[g_pp[rows]]) / cnt[:, None]
    M[cnt < min_mates] = np.nan
    return M, cnt


def loo_mean(V_all, outer, inner, sel_all, rows, min_n=1):
    """For each row in `rows`: the mean of V_all over rows j with sel_all[j], in the row's `outer` group but NOT in its
    `inner` group (inner nests inside outer). NaN where fewer than min_n such rows."""
    w = sel_all.astype(np.float64)
    So, _ = group_sum(outer, V_all * w[:, None])
    Si, _ = group_sum(inner, V_all * w[:, None])
    cnt = np.bincount(outer, weights=w)[outer[rows]] - np.bincount(inner, weights=w)[inner[rows]]
    with np.errstate(invalid='ignore', divide='ignore'):
        M = (So[outer[rows]] - Si[inner[rows]]) / cnt[:, None]
    M[cnt < min_n] = np.nan
    return M, cnt


def mates_tanimoto(plate, pert, fp_of):
    """Per row: the mean ECFP4 Tanimoto between its compound and each of its other-compound plate-mates (rows given)."""
    out = np.full(len(plate), np.nan)
    for g in np.unique(plate):
        ix = np.flatnonzero(plate == g)
        if len(ix) < 2:
            continue
        F = np.stack([fp_of(p) for p in pert[ix]]).astype(np.float64)
        inter = F @ F.T
        n = F.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            T = np.where(n[:, None] + n[None, :] - inter > 0, inter / (n[:, None] + n[None, :] - inter), 0.0)
        other = pert[ix][:, None] != pert[ix][None, :]
        with np.errstate(invalid='ignore'):
            out[ix] = np.where(other.sum(1) > 0, (T * other).sum(1) / np.maximum(other.sum(1), 1), np.nan)
    return out


def _fp_lookup():
    idx = json.load(open(DRUG_INDEX, encoding='utf-8'))
    fp = np.load(DRUG_FP)
    return lambda p: fp[idx[str(p)]] > 0.5


def _plates_for(bundle, bundle_idx, ctl_true):
    """The plate proxy for scored rows, with the identity guard that the bundle's X_ctl IS the scored file's control."""
    C_b = bundle['X_ctl'][bundle_idx].astype(np.float64)
    dmax = float(np.abs(C_b - np.asarray(ctl_true, np.float64)).max())
    if dmax > 1e-4:
        raise SystemExit('FATAL: the bundle X_ctl differs from the scored file ctl_true by %.3g' % dmax)
    return control_ids(bundle['X_ctl'])[bundle_idx]


def _summ(r):
    r = np.asarray(r, np.float64)
    return {'mean': float(np.nanmean(r)), 'median': float(np.nanmedian(r)), 'n': int(np.isfinite(r).sum())}


def _mol_block(r_ours, r_ref, molecule, ref_label, n_boot):
    rd = score_p9.molecule_reading(r_ours, r_ref, molecule, n_boot=n_boot, seed=0)
    rd.pop('d_m', None)
    rd['verdict'] = score_p9.verdict(rd, ref_label)
    return rd


def _load_p9_side(p9_dir):
    p9 = score_p10.load_p9_and_metadata(p9_dir)
    mdt = score_p10.load_ref('mlp_mdt', None, p9)
    m = score_p10.load_ref('mlp_m', None, p9)
    plate = _plates_for(p9['bundle_data'], p9['bundle_idx'], p9['ctl_true'])
    v9_deltas = [np.asarray(s['y_pred'], np.float64) - p9['ctl_true'] for s in p9['seeds_data']]
    return p9, mdt, m, plate, v9_deltas


def test_c1(p9_dir=P9_DIR, n_boot=N_BOOT):
    p9, mdt, m, plate, v9_deltas = _load_p9_side(p9_dir)
    pert = p9['pert']
    cy, cnt = plate_contrast(p9['y'], plate, pert)
    have = cnt >= MIN_MATES
    rows = have & p9['clean_mask']

    def score(deltas):
        out, zeros = [], 0
        for d in deltas:
            cd, _ = plate_contrast(d, plate, pert)
            r, z = rowr_pred(np.where(have[:, None], cd, 0.0), np.where(have[:, None], cy, 0.0))
            r[~have] = np.nan
            out.append(r)
            zeros += int(z)
        return out, zeros

    r_v9_s, z_v9 = score(v9_deltas)
    r_mdt_s, z_mdt = score(mdt['pred_deltas'])
    r_m_s, z_m = score(m['pred_deltas'])
    r_v9, r_mdt = np.mean(r_v9_s, 0), np.mean(r_mdt_s, 0)
    if not (np.isfinite(r_v9[rows]).all() and np.isfinite(r_mdt[rows]).all()):
        raise SystemExit('FATAL: non-finite contrast scores on the scored rows')
    mol = p9['molecule'][rows]
    primary = _mol_block(r_v9[rows], r_mdt[rows], mol, 'M+dt', n_boot)
    statement = C1_STATEMENTS[primary['verdict']].format(ci_hi=primary['ci_hi'])

    # review 083 C2: the zero-parameter dose-only floor -- the mean y of TRAINING rows at the same (cell, time, dose)
    # on a different control (compound-blind), contrasted over the scored plate-mates exactly as the predictions are
    b = p9['bundle_data']
    y_all = b['X'].astype(np.float64) - b['X_ctl'].astype(np.float64)
    plate_all = control_ids(b['X_ctl'])
    ctd = group_ids(b['meta_cell'], b['meta_time'], b['meta_dose'])
    ctdp = group_ids(b['meta_cell'], b['meta_time'], b['meta_dose'], plate_all)
    D_dose, n_dose = loo_mean(y_all, ctd, ctdp, b[score_p10.SPLIT_KEY] == 'train', p9['bundle_idx'])
    has_dose = n_dose >= 1
    cd_dose, _ = plate_contrast(np.where(has_dose[:, None], D_dose, 0.0), plate, pert)
    r_dose, _ = rowr_pred(np.where(have[:, None], cd_dose, 0.0), np.where(have[:, None], cy, 0.0))
    r_dose[~(have & has_dose)] = np.nan
    rows_d = rows & has_dose

    # review 083 ask 2 (reported, never read): strata by the plate-mates' mean ECFP4 Tanimoto
    tan = mates_tanimoto(plate, pert, _fp_lookup())
    med = float(np.nanmedian(tan[rows]))
    strata = {}
    for name, m in (('mates_tanimoto_below_median', rows & (tan < med)), ('mates_tanimoto_at_or_above_median', rows & (tan >= med))):
        strata[name] = _mol_block(r_v9[m], r_mdt[m], p9['molecule'][m], 'M+dt', n_boot)

    def ratio(deltas):
        cd, _ = plate_contrast(np.mean(deltas, 0), plate, pert)
        return float(np.median(np.linalg.norm(cd[rows], axis=1) / np.linalg.norm(cy[rows], axis=1)))

    return {
        'test': 'C1', 'registered': 'RESULTS 99.3 / V10_AUDIT_AND_PLAN.md §5', 'reading_of_record': True,
        'rows': {'p9_rows': int(len(cy)), 'with_ge2_mates': int(have.sum()), 'scored_clean': int(rows.sum()),
                 'molecules': int(len(np.unique(mol)))},
        'primary_v9_vs_mdt': primary, 'statement': statement,
        'reported': {
            'per_seed_vs_mdt': [_mol_block(r[rows], r_mdt[rows], mol, 'M+dt', n_boot) for r in r_v9_s],
            'v9_vs_zero': _mol_block(r_v9[rows], np.zeros(rows.sum()), mol, 'zero', n_boot),
            'row_mean_r': {'v9': _summ(r_v9[rows]), 'mdt': _summ(r_mdt[rows]), 'm': _summ(np.mean(r_m_s, 0)[rows])},
            'zero_variance_prediction_rows': {'v9_all_seeds': z_v9, 'mdt_all_seeds': z_mdt, 'm_all_seeds': z_m,
                                              'note': 'M sees only the control, so its contrast is 0 by construction'},
            'contrast_magnitude_ratio_median': {'v9': ratio(v9_deltas), 'mdt': ratio(mdt['pred_deltas'])},
            'dose_only_floor': {'definition': 'mean y of training rows at the same (cell, time, dose) on another control, '
                                              'contrasted over the scored plate-mates',
                                'rows': int(rows_d.sum()), 'row_mean_r': _summ(r_dose[rows_d]),
                                'v9_vs_floor': _mol_block(r_v9[rows_d], r_dose[rows_d], p9['molecule'][rows_d],
                                                          'the dose-only floor', n_boot),
                                'mdt_vs_floor': _mol_block(r_mdt[rows_d], r_dose[rows_d], p9['molecule'][rows_d],
                                                           'the dose-only floor', n_boot)},
            'by_mates_tanimoto': {'median': med, **strata},
        }}


def test_c1b(p9_dir=P9_DIR, bundle_path=BUNDLE, n_boot=N_BOOT):
    p9, mdt, m, plate, v9_deltas = _load_p9_side(p9_dir)
    b = p9['bundle_data']
    plate_all = control_ids(b['X_ctl'])
    y_all = b['X'].astype(np.float64) - b['X_ctl'].astype(np.float64)
    P_meas, cnt = other_mean(y_all, plate_all, b['meta_pert_id'], np.ones(len(y_all), bool), p9['bundle_idx'])
    have = cnt >= MIN_MATES
    rows = have & p9['clean_mask']
    tr = p9['y'] - np.nan_to_num(P_meas)

    def score(deltas):
        return np.mean([rowr_pred(d - np.nan_to_num(P_meas), tr)[0] for d in deltas], 0)

    r_v9, r_mdt, r_m = score(v9_deltas), score(mdt['pred_deltas']), score(m['pred_deltas'])
    mol = p9['molecule'][rows]
    return {'test': 'C1b', 'reported': True, 'rows': {'with_ge2_mates_any_split': int(have.sum()), 'scored_clean': int(rows.sum())},
            'v9_vs_mdt': _mol_block(r_v9[rows], r_mdt[rows], mol, 'M+dt', n_boot),
            'v9_vs_m': _mol_block(r_v9[rows], r_m[rows], mol, 'M', n_boot),
            'row_mean_r': {'v9': _summ(r_v9[rows]), 'mdt': _summ(r_mdt[rows]), 'm': _summ(r_m[rows])}}


def test_c1c(p9_dir=P9_DIR, n_boot=N_BOOT):
    p9, mdt, m, plate, v9_deltas = _load_p9_side(p9_dir)
    b = p9['bundle_data']
    plate_all = control_ids(b['X_ctl'])
    y_all = b['X'].astype(np.float64) - b['X_ctl'].astype(np.float64)
    is_tr = b[score_p10.SPLIT_KEY] == 'train'
    P_tr, cnt = other_mean(y_all, plate_all, b['meta_pert_id'], is_tr, p9['bundle_idx'])
    have = cnt >= MIN_MATES
    rows = have & p9['clean_mask']
    y = p9['y']
    r_p = rowr_pred(np.nan_to_num(P_tr), y)[0]
    r_v9 = np.mean([rowr_pred(d, y)[0] for d in v9_deltas], 0)
    r_m = np.mean([rowr_pred(d, y)[0] for d in m['pred_deltas']], 0)
    mol = p9['molecule'][rows]
    return {'test': 'C1c', 'reported': True,
            'rows': {'with_ge2_training_mates': int(have.sum()), 'scored_clean': int(rows.sum())},
            'row_mean_r_original_target': {'training_plate_predictor': _summ(r_p[rows]), 'v9': _summ(r_v9[rows]),
                                           'm': _summ(r_m[rows])},
            'v9_vs_training_plate_predictor': _mol_block(r_v9[rows], r_p[rows], mol, 'the training-plate predictor', n_boot),
            'm_vs_training_plate_predictor': _mol_block(r_m[rows], r_p[rows], mol, 'the training-plate predictor', n_boot)}


def _load_p7_side(p7_dir=P7_DIR, o2_path=O2, bundle_path=BUNDLE):
    import coldcell_h2h
    seeds = [coldcell_h2h.load_profile(os.path.join(p7_dir, 'v9p7_seed%d.npz' % k)) for k in (0, 1, 2)]
    for s in seeds[1:]:
        for k in ('row_index', 'y_true', 'ctl_true'):
            if not np.array_equal(s[k], seeds[0][k]):
                raise SystemExit('FATAL: the P7 seed files disagree on %r' % k)
    T = coldcell_h2h.load_profile(o2_path)
    common, it, io_ = np.intersect1d(T['row_index'], seeds[0]['row_index'], return_indices=True)
    if len(common) / min(len(T['row_index']), len(seeds[0]['row_index'])) < 0.95:
        raise SystemExit('FATAL: P7 and O2 overlap below 95 %')
    for k in ('y_true', 'ctl_true'):
        if float(np.abs(T[k][it] - seeds[0][k][io_]).max()) > 1e-4:
            raise SystemExit('FATAL: P7 and O2 disagree on %s' % k)
    b = np.load(bundle_path, allow_pickle=True)
    pos = {int(r): i for i, r in enumerate(b['row_index'])}
    bidx = np.array([pos[int(r)] for r in common], np.int64)
    ctl = np.asarray(T['ctl_true'][it], np.float64)
    y = np.asarray(T['y_true'][it], np.float64) - ctl
    plate = _plates_for(b, bidx, ctl)
    v9 = [np.asarray(s['y_pred'][io_], np.float64) - ctl for s in seeds]
    xp = np.asarray(T['y_pred'][it], np.float64) - ctl
    return {'b': b, 'bidx': bidx, 'y': y, 'ctl': ctl, 'plate': plate, 'v9': v9, 'xp': xp,
            'cell': b['meta_cell'][bidx].astype(str), 'pert': b['meta_pert_id'][bidx], 'row_index': common}


def _by_cell(r, cell):
    return {c: _summ(r[cell == c]) for c in np.unique(cell)}


def test_c2a(p7_dir=P7_DIR, o2_path=O2, bundle_path=BUNDLE):
    S = _load_p7_side(p7_dir, o2_path, bundle_path)
    b, ctl, y, cell = S['b'], S['ctl'], S['y'], S['cell']
    split = b['split_split_cold_cell_1']
    C_all = b['X_ctl'].astype(np.float64)
    cell_all = b['meta_cell'].astype(str)
    cm = {c: C_all[(cell_all == c) & (split == 'test')].mean(0) for c in np.unique(cell)}
    F_tr = -(ctl - np.stack([cm[c] for c in cell]))
    F_ind = -(ctl - C_all[split == 'train'].mean(0)[None, :])
    r = {'F_tr': rowr_pred(F_tr, y)[0], 'F_ind': rowr_pred(F_ind, y)[0],
         'v9': np.mean([rowr_pred(d, y)[0] for d in S['v9']], 0), 'xpert': rowr_pred(S['xp'], y)[0]}
    return {'test': 'C2a', 'reported': True, 'rows': int(len(y)),
            'row_mean_r': {k: _summ(v) for k, v in r.items()},
            'per_cell_mean_r': {k: _by_cell(v, cell) for k, v in r.items()},
            'cluster_mean_of_cell_means': {k: float(np.mean([np.nanmean(v[cell == c]) for c in np.unique(cell)]))
                                           for k, v in r.items()}}


def test_c2c(p7_dir=P7_DIR, o2_path=O2, bundle_path=BUNDLE, n_boot=N_BOOT):
    import coldcell_h2h
    S = _load_p7_side(p7_dir, o2_path, bundle_path)
    plate, pert, cell = S['plate'], S['pert'], S['cell']
    cy, cnt = plate_contrast(S['y'], plate, pert)
    have = cnt >= MIN_MATES

    def r_of(d):
        cd, _ = plate_contrast(d, plate, pert)
        return rowr_pred(np.where(have[:, None], cd, 0.0), np.where(have[:, None], cy, 0.0))[0][have]

    r_v9 = np.mean([r_of(d) for d in S['v9']], 0)
    r_xp = r_of(S['xp'])
    cells = cell[have]
    u, n_by = np.unique(cells, return_counts=True)
    rng = np.random.default_rng(0)
    cluster, per_cell, pooled = coldcell_h2h.compute_cluster_and_pooled(r_v9 - r_xp, r_v9, r_xp, cells, u, n_by,
                                                                          u[np.argmax(n_by)], n_boot, rng)
    return {'test': 'C2c', 'reported': True, 'rows_with_ge2_mates': int(have.sum()),
            'row_mean_r': {'v9': _summ(r_v9), 'xpert': _summ(r_xp)},
            'cluster': cluster, 'per_cell': per_cell, 'pooled': pooled}


def test_c4(p9_dir=P9_DIR):
    p9, mdt, m, plate, v9_deltas = _load_p9_side(p9_dir)
    b = p9['bundle_data']
    plate_all = control_ids(b['X_ctl'])
    y_all = b['X'].astype(np.float64) - b['X_ctl'].astype(np.float64)
    P_meas, cnt = other_mean(y_all, plate_all, b['meta_pert_id'], np.ones(len(y_all), bool), p9['bundle_idx'])
    have = cnt >= MIN_MATES
    bi = p9['bundle_idx']
    dose = b['meta_dose'][bi].astype(np.float64)
    time = b['meta_time'][bi].astype(np.float64)
    P0 = np.nan_to_num(P_meas)
    mag = {'truth': np.linalg.norm(p9['y'] - P0, axis=1),
           'v9': np.linalg.norm(np.mean(v9_deltas, 0) - P0, axis=1),
           'mdt': np.linalg.norm(np.mean(mdt['pred_deltas'], 0) - P0, axis=1)}
    g = group_ids(p9['pert'], p9['cell'], time)
    rho = {k: [] for k in mag}
    n_groups = 0
    for k_g in np.unique(g[have]):
        ix = np.flatnonzero((g == k_g) & have)
        if len(np.unique(dose[ix])) < 3:
            continue
        n_groups += 1
        for k in mag:
            rho[k].append(scipy.stats.spearmanr(dose[ix], mag[k][ix]).statistic)
    rho = {k: np.asarray(v, np.float64) for k, v in rho.items()}
    out = {'test': 'C4', 'reported': True, 'n_groups': n_groups}
    for k, v in rho.items():
        out[k] = {'mean_rho': float(np.nanmean(v)), 'frac_rho_pos': float(np.nanmean(v > 0))}
    for k in ('v9', 'mdt'):
        ok = np.isfinite(rho['truth']) & np.isfinite(rho[k])
        out['corr_rho_truth_' + k] = float(np.corrcoef(rho['truth'][ok], rho[k][ok])[0, 1])
    return out


C2B_RULE = 0.5        # §5: replaced cluster mean (v9 - XPert) below half the intact one -> "substantially carried"
C2B_OF_RECORD = 'ctl'  # review 083 C5: x_cell := the row's own X_ctl; 'global' (all cell information removed) is reported
AUDIT_OUT = os.path.join(REPO, 'external', 'kaggle_out', 'audit99')


def _load_infer(out_dir, mode):
    """The three seeds' inference outputs, each verified against its kernel's AUDIT99_COMPLETE manifest."""
    runs = []
    for k in (0, 1, 2):
        man = os.path.join(out_dir, 's%d' % k, 'AUDIT99_COMPLETE_s%d.json' % k)
        if not os.path.exists(man):
            raise SystemExit('FATAL: no manifest %s' % man)
        m = json.load(open(man, encoding='utf-8'))
        if not m.get('complete') or int(m.get('seed', -1)) != k:
            raise SystemExit('FATAL: manifest %s incomplete or wrong seed' % man)
        f = '%s_seed%d.npz' % (mode, k)
        p = os.path.join(out_dir, 's%d' % k, f)
        if f not in m['files'] or not os.path.exists(p) or sha1(p) != m['files'][f]:
            raise SystemExit('FATAL: %s missing or not the manifest file' % p)
        rec = json.load(open(p.replace('.npz', '.json'), encoding='utf-8'))
        if rec.get('mode') != mode or int(rec.get('seed', -1)) != k or rec.get('limit'):
            raise SystemExit('FATAL: %s record is not a full %s run of seed %d' % (p, mode, k))
        z = np.load(p, allow_pickle=False)
        runs.append({key: z[key] for key in z.files})
    for r in runs[1:]:
        for key in ('row_index', 'y', 'ctl'):
            if not np.array_equal(r[key], runs[0][key]):
                raise SystemExit('FATAL: the seeds disagree on %r' % key)
    return runs


def read_c2b(out_dir=AUDIT_OUT, o2_path=O2, bundle_path=BUNDLE, n_boot=N_BOOT):
    import coldcell_h2h
    runs = _load_infer(out_dir, 'xcell')
    T = coldcell_h2h.load_profile(o2_path)
    common, it, io_ = np.intersect1d(T['row_index'], runs[0]['row_index'], return_indices=True)
    ctl = np.asarray(T['ctl_true'][it], np.float64)
    y = np.asarray(T['y_true'][it], np.float64) - ctl
    if float(np.abs(y - runs[0]['y'][io_]).max()) > 1e-4 or float(np.abs(ctl - runs[0]['ctl'][io_]).max()) > 1e-4:
        raise SystemExit('FATAL: the inference outputs and O2 disagree on the target')
    r_int = np.mean([rowr_pred(r['deg_intact'][io_].astype(np.float64), y)[0] for r in runs], 0)
    r_var = {v: np.mean([rowr_pred(r['deg_' + v][io_].astype(np.float64), y)[0] for r in runs], 0) for v in ('ctl', 'global')}
    r_mod = r_var[C2B_OF_RECORD]
    r_xp = rowr_pred(np.asarray(T['y_pred'][it], np.float64) - ctl, y)[0]
    ok = np.isfinite(r_int) & np.isfinite(r_var['ctl']) & np.isfinite(r_var['global']) & np.isfinite(r_xp)
    b = np.load(bundle_path, allow_pickle=True)
    pos = {int(r): i for i, r in enumerate(b['row_index'])}
    cells = b['meta_cell'][np.array([pos[int(r)] for r in common[ok]])].astype(str)
    u, n_by = np.unique(cells, return_counts=True)
    top = u[np.argmax(n_by)]
    blocks = {}
    for name, (a_, b_) in {'intact_vs_xpert': (r_int, r_xp), 'replaced_vs_xpert': (r_mod, r_xp),
                           'intact_vs_replaced': (r_int, r_mod), 'global_vs_xpert': (r_var['global'], r_xp),
                           'intact_vs_global': (r_int, r_var['global'])}.items():
        rng = np.random.default_rng(0)
        cl, pc, po = coldcell_h2h.compute_cluster_and_pooled(a_[ok] - b_[ok], a_[ok], b_[ok], cells, u, n_by, top, n_boot, rng)
        blocks[name] = {'cluster': cl, 'per_cell': pc, 'pooled': po}
    lead_int = blocks['intact_vs_xpert']['cluster']['mean_of_d_c']
    lead_mod = blocks['replaced_vs_xpert']['cluster']['mean_of_d_c']
    ci_int = blocks['intact_vs_xpert']['cluster']['cluster_ci95']
    assessable = bool(lead_int > 0 and ci_int[0] > 0)          # review 083 C5: the precondition
    carried = bool(assessable and lead_mod < C2B_RULE * lead_int)
    if not assessable:
        statement = ('not assessable: the intact final-snapshot lead over XPert is %+.4f with cluster CI [%+.4f, %+.4f], '
                     'which does not exclude 0' % (lead_int, ci_int[0], ci_int[1]))
    elif carried:
        statement = "v9's unseen-cell lead over XPert is substantially carried by the transductive cell-average input"
    else:
        statement = "v9's unseen-cell lead over XPert is not substantially carried by the transductive cell-average input"
    return {'test': 'C2b', 'reading_of_record': True, 'basis': 'final snapshot against final snapshot (P7 of record is '
            'the three-snapshot mean)', 'variant_of_record': "x_cell := the row's own X_ctl",
            'rows': int(ok.sum()), 'blocks': blocks, 'lead_intact': lead_int, 'lead_replaced': lead_mod,
            'lead_global_reported': blocks['global_vs_xpert']['cluster']['mean_of_d_c'],
            'rule': 'replaced < %.1f x intact, if the intact lead is > 0 with its cluster CI excluding 0' % C2B_RULE,
            'assessable': assessable, 'statement': statement}


def read_c3(out_dir=AUDIT_OUT, p9_dir=P9_DIR, n_boot=N_BOOT):
    runs = _load_infer(out_dir, 'swap')
    p9 = score_p10.load_p9_and_metadata(p9_dir)
    pos = {int(r): i for i, r in enumerate(runs[0]['row_index'])}
    missing = [int(r) for r in p9['row_index'] if int(r) not in pos]
    if missing:
        raise SystemExit('FATAL: %d P9 rows missing from the swap outputs' % len(missing))
    ix = np.array([pos[int(r)] for r in p9['row_index']], np.int64)
    y = p9['y']
    if float(np.abs(runs[0]['y'][ix] - y).max()) > 1e-4:
        raise SystemExit('FATAL: the swap outputs and P9 disagree on the target')
    C3_OF_RECORD = 'tani'                                       # review 083 C6; 'free' is reported
    plate = _plates_for(p9['bundle_data'], p9['bundle_idx'], p9['ctl_true'])
    pert = p9['pert']
    cy, cnt = plate_contrast(y, plate, pert)
    have = cnt >= MIN_MATES

    def contrast_r(d):
        cd, _ = plate_contrast(d, plate, pert)
        r = rowr_pred(np.where(have[:, None], cd, 0.0), np.where(have[:, None], cy, 0.0))[0]
        r[~have] = np.nan
        return r

    deg_i = [r['deg_intact'][ix].astype(np.float64) for r in runs]
    r_orig_i = np.mean([rowr_pred(d, y)[0] for d in deg_i], 0)
    r_c_i = np.mean([contrast_r(d) for d in deg_i], 0)
    out = {'test': 'C3', 'reading_of_record': True, 'variant_of_record': 'compound-level swap within cell, ECFP4 '
           'Tanimoto < 0.4', 'variants': {}}
    for v in ('tani', 'free'):
        swapped = np.all([r['donor_row_index_' + v][ix] >= 0 for r in runs], 0)
        deg_m = [r['deg_' + v][ix].astype(np.float64) for r in runs]
        r_orig_m = np.mean([rowr_pred(d, y)[0] for d in deg_m], 0)
        r_c_m = np.mean([contrast_r(d) for d in deg_m], 0)
        rows_o = swapped & p9['clean_mask']
        rows_c = rows_o & have
        reading = _mol_block(r_c_i[rows_c], r_c_m[rows_c], p9['molecule'][rows_c], 'the swapped compound', n_boot)
        sens = np.median(np.mean([np.linalg.norm(a - b, axis=1) for a, b in zip(deg_i, deg_m)], 0)[rows_c]
                         / np.linalg.norm(cy[rows_c], axis=1))
        out['variants'][v] = {
            'rows': {'p9': int(len(y)), 'swapped_clean': int(rows_o.sum()), 'contrast_scored': int(rows_c.sum())},
            'contrast_intact_vs_swapped': reading,
            'original_target_intact_vs_swapped': _mol_block(r_orig_i[rows_o], r_orig_m[rows_o], p9['molecule'][rows_o],
                                                            'the swapped compound', n_boot),
            'sensitivity_median': float(sens)}
        if v == C3_OF_RECORD:
            depends = bool(reading['mean_d'] > 0 and reading['ci_lo'] > 0)
            out['statement'] = ("v9's predictions depend on the right compound: swapping in a dissimilar compound "
                                "lowers the contrast score by %+.3f per molecule (95 %% cluster CI [%+.3f, %+.3f])"
                                % (reading['mean_d'], reading['ci_lo'], reading['ci_hi']) if depends else
                                "v9's dependence on the right compound is not shown: the contrast gain of the right "
                                "compound over a dissimilar one is at most %+.3f per molecule (95 %% cluster CI)"
                                % reading['ci_hi'])
    return out


RUNNERS = {'C1': test_c1, 'C1b': test_c1b, 'C1c': test_c1c, 'C2a': test_c2a, 'C2c': test_c2c, 'C4': test_c4,
           'C2b': read_c2b, 'C3': read_c3}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True, choices=TESTS)
    ap.add_argument('--out_dir', default=OUT_DIR)
    a = ap.parse_args(argv)
    out = os.path.join(a.out_dir, 'phase1_%s.json' % a.test)
    if os.path.exists(out):
        raise SystemExit('FATAL: %s exists; each Phase 1 test is run once' % out)
    res = RUNNERS[a.test]()
    res['code_sha1'] = sha1(os.path.abspath(__file__))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(res, f, indent=1, default=float)
    print('wrote', out)
    if a.test == 'C1':
        print('C1 verdict:', res['primary_v9_vs_mdt']['verdict'], '->', res['statement'])


if __name__ == '__main__':
    sys.exit(main())
