"""The gap and the head-to-head (RESULTS 100.5, reported; registered in detail in 100.11). Written before any T-C run.

Every number is recomputed here with v9's own scoring functions (`audit99_phase1`: plate_contrast, rowr_pred, the P7 / P9
loaders), on v9's own rows and mate sets, the same for truth and prediction and for every model:
  unseen cells      the P7 rows (= P7 n O2, 21,151); mates among all of them (>= 2 other-compound mates); scored on the
                    7 clean cells' rows (primary) and on all 8 (labelled, H1975 included). Contrast r (C2c) and delta r (C2a).
  unseen compounds  the P9 rows (13,364); mates among all of them; contrast r scored on P9's clean rows with >= 2 mates (C1);
                    delta r on P9's clean rows (97.10's row-pooled delta r).
Per-row r is averaged over a model's seeds, then over rows. v10-C's rows must equal the reference rows through bench_row
(asserted), and its truth must equal the reference truth (asserted against the substrate).
  the gap           unseen-compound value - unseen-cell value, per target: full-release v10-C (the gap arm against T-C's on
                    arm) beside v9 (P9 against P7)
  the head-to-head  per clean cell: full-release v10-C (T-C on) against XPert trained on the benchmark (O2), and v9 (P7);
                    reported, no claim (086 C1)
A slot with no completed attempt: T-C's side follows tc_read (no claim); the gap arm reports over its completed slots, n
stated (100.11 C3a).

    python model/v10c/tc_gap.py --runs_dir external/kaggle_out/v10c_tc --out model/results/v10c/tc_gap.json
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'model', 'v9'))
import tc_read  # noqa: E402

CLEAN, ALL8 = tc_read.CLEAN, tc_read.CLEAN + tc_read.APART
SUBSTRATE = os.path.join(REPO, 'external', 'v10c_full', 'v10c_full_substrate.npz')


def align(pred_path, ref_rows):
    """v10-C's saved delta predictions reordered to the reference rows (by bench_row); refuses any other row set."""
    z = np.load(pred_path)
    br = z['bench_row'].astype(np.int64)
    ref = np.asarray(ref_rows, np.int64)
    if len(br) != len(ref) or not np.array_equal(np.sort(br), np.sort(ref)):
        raise SystemExit(f'FATAL: {pred_path} rows differ from the reference rows (100.11 C2)')
    pos = {int(r): i for i, r in enumerate(br)}
    order = np.array([pos[int(r)] for r in ref])
    return np.asarray(z['delta'], np.float64)[order], np.asarray(z['sub_row'])[order]


_SUB = {}


def check_truth(sub_rows, y_ref, substrate=SUBSTRATE, tol=1e-4):
    if substrate not in _SUB:
        z = np.load(substrate, allow_pickle=True)
        _SUB[substrate] = (z['X'], z['ctl'], z['ctl_id'])
    X, ctl, ctl_id = _SUB[substrate]
    y = X[sub_rows].astype(np.float64) - ctl[ctl_id[sub_rows]].astype(np.float64)
    d = float(np.abs(y - y_ref).max())
    if d > tol:
        raise SystemExit(f'FATAL: v10-C truth differs from the reference truth by {d:.3g}')
    return d


def contrast_rows(ap, deltas, y, plate, pert):
    """C1 / C2c: per-row contrast r averaged over the given seeds' deltas; NaN where < 2 mates (mates = all given rows)."""
    cy, cnt = ap.plate_contrast(y, plate, pert)
    have = cnt >= ap.MIN_MATES
    out = []
    for d in deltas:
        cd, _ = ap.plate_contrast(d, plate, pert)
        r, _ = ap.rowr_pred(np.where(have[:, None], cd, 0.0), np.where(have[:, None], cy, 0.0))
        r[~have] = np.nan
        out.append(r)
    return np.mean(out, 0), have


def delta_rows(ap, deltas, y):
    return np.mean([ap.rowr_pred(d, y)[0] for d in deltas], 0)


def summarise(r, cell, cells):
    m = np.isin(cell, cells) & np.isfinite(r)
    per = {c: float(np.mean(r[(cell == c) & np.isfinite(r)])) for c in cells}
    return {'row_mean': float(np.mean(r[m])), 'n_rows': int(m.sum()), 'cluster_mean': float(np.mean(list(per.values()))),
            'per_cell': per}


def cell_side(ap, S, models):
    """models: name -> list of deltas in S's row order. Contrast (C2c) and delta r (C2a), 7 clean and 8 cells."""
    out = {}
    for name, deltas in models.items():
        rc, _ = contrast_rows(ap, deltas, S['y'], S['plate'], S['pert'])
        rd = delta_rows(ap, deltas, S['y'])
        out[name] = {'n_seeds': len(deltas),
                     'contrast': {'clean7': summarise(rc, S['cell'], CLEAN), 'all8_labelled': summarise(rc, S['cell'], ALL8)},
                     'delta_r': {'clean7': summarise(rd, S['cell'], CLEAN), 'all8_labelled': summarise(rd, S['cell'], ALL8)}}
    return out


def compound_side(ap, p9, plate, models):
    """models: name -> list of deltas in P9's row order. Contrast on clean rows with >= 2 mates (C1); delta r on clean rows."""
    clean = p9['clean_mask']
    out = {}
    for name, deltas in models.items():
        rc, have = contrast_rows(ap, deltas, p9['y'], plate, p9['pert'])
        rd = delta_rows(ap, deltas, p9['y'])
        sel = have & clean
        out[name] = {'n_seeds': len(deltas), 'contrast_clean': float(np.mean(rc[sel])), 'n_contrast_rows': int(sel.sum()),
                     'delta_r_clean': float(np.mean(rd[clean])), 'n_delta_rows': int(clean.sum())}
    return out


def completed_preds(runs_dir, prefix, arm):
    """The first completed attempt's prediction file per seed slot (tc_read's rule), and the slots without one."""
    preds, missing = [], []
    for s in tc_read.SEEDS:
        k, rec = tc_read.first_completed(tc_read.slot_attempts(runs_dir, arm, s, prefix=prefix))
        if rec is None:
            missing.append(s)
            continue
        r = json.load(open(rec, encoding='utf-8'))
        p = os.path.join(os.path.dirname(rec), f"v10c_pred_{r['tag']}.npz")
        preds.append(p)
    return preds, missing


def gap(cell, comp, model_cell, model_comp):
    g = {}
    for scope in ('clean7', 'all8_labelled'):
        g[scope] = {'contrast': comp[model_comp]['contrast_clean'] - cell[model_cell]['contrast'][scope]['row_mean'],
                    'delta_r': comp[model_comp]['delta_r_clean'] - cell[model_cell]['delta_r'][scope]['row_mean']}
    return g


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--runs_dir', required=True)
    ap_.add_argument('--out', required=True)
    a = ap_.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- read once')
    import audit99_phase1 as ap
    S = ap._load_p7_side()
    p9, _, _, plate9, v9_comp = ap._load_p9_side(ap.P9_DIR)
    tc_preds, tc_missing = completed_preds(a.runs_dir, 'tc', 'on')
    gap_preds, gap_missing = completed_preds(a.runs_dir, 'gap', 'on')
    v10_cell, v10_comp = [], []
    for p in tc_preds:
        d, sub = align(p, S['row_index'])
        check_truth(sub, S['y'])
        v10_cell.append(d)
    for p in gap_preds:
        d, sub = align(p, p9['row_index'])
        check_truth(sub, p9['y'])
        v10_comp.append(d)
    cell_models = {'v9': S['v9'], 'xpert_o2': [S['xp']]}
    if v10_cell and not tc_missing:
        cell_models['v10c_full_release'] = v10_cell
    comp_models = {'v9': v9_comp}
    if v10_comp:
        comp_models['v10c_full_release'] = v10_comp
    cell = cell_side(ap, S, cell_models)
    comp = compound_side(ap, p9, plate9, comp_models)
    rec = {'rule': 'RESULTS 100.5 / 100.11 (reported, no claim)', 'tc_missing_on_slots': tc_missing,
           'gap_arm_slots_used': len(gap_preds), 'gap_arm_missing_slots': gap_missing,
           'unseen_cell': cell, 'unseen_compound': comp,
           'gap': {'v9': gap(cell, comp, 'v9', 'v9')},
           'wording': 'full-release v10-C trained on the substrate against XPert trained on the benchmark; nothing is '
                      'attributed to architecture (086 C1); L6 leave-own-plate-out, input-transductive across a cell\'s '
                      'plates (086 C3); among runs whose drug path trained (100.8)'}
    if 'v10c_full_release' in cell and 'v10c_full_release' in comp:
        rec['gap']['v10c_full_release'] = gap(cell, comp, 'v10c_full_release', 'v10c_full_release')
    if 'v10c_full_release' in cell:
        rec['head_to_head_per_clean_cell'] = {
            t: {c: {'v10c_full_release': cell['v10c_full_release'][t]['clean7']['per_cell'][c],
                    'xpert_o2': cell['xpert_o2'][t]['clean7']['per_cell'][c],
                    'v9': cell['v9'][t]['clean7']['per_cell'][c]} for c in CLEAN}
            for t in ('contrast', 'delta_r')}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({'gap': rec['gap'], 'tc_missing': tc_missing, 'gap_slots': len(gap_preds)}, indent=1))


if __name__ == '__main__':
    main()
