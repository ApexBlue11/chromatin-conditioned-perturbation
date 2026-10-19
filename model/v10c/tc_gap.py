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
A slot with no completed attempt: both reported sides (T-C's on arm and the gap arm) report over their completed slots,
n stated per side (review 092 C2); T-C's RULE still makes no claim (tc_read, 100.8).
Provenance (092 C1): each record's sha1 must equal the T-C ledger's, and each prediction file's sha1 its record's pred_sha1.
Self-check (092 C3): v9's and XPert's values must reproduce the registered references to 1e-4, and the sha1s of the reused
modules and of the P7 / P9 / O2 inputs are recorded. C4: v10-C's clean-7 per-cell contrast is reported from both scorers
(the trainer's records, used by T-C's rule, and this file's), with their maximum difference.

    python model/v10c/tc_gap.py --runs_dir external/kaggle_out/v10c_tc --out model/results/v10c/tc_gap.json
"""
import argparse, glob, hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'model', 'v9'))
import tc_read  # noqa: E402
import tc_ledger  # noqa: E402

CLEAN, ALL8 = tc_read.CLEAN, tc_read.CLEAN + tc_read.APART
SUBSTRATE = os.path.join(REPO, 'external', 'v10c_full', 'v10c_full_substrate.npz')
# 100.11 / review 092 C3: the references this reader must reproduce (tc_gap_repro.json, my and the critic's recomputations)
REFS = {('cell', 'v9', 'contrast', 'all8_labelled'): 0.16023819, ('cell', 'xpert_o2', 'contrast', 'all8_labelled'): 0.14602053,
        ('cell', 'v9', 'delta_r', 'all8_labelled'): 0.48121207, ('cell', 'xpert_o2', 'delta_r', 'all8_labelled'): 0.38678191,
        ('cell', 'v9', 'contrast', 'clean7'): 0.15972099, ('cell', 'xpert_o2', 'contrast', 'clean7'): 0.14564001,
        ('cell', 'v9', 'delta_r', 'clean7'): 0.48104096, ('cell', 'xpert_o2', 'delta_r', 'clean7'): 0.38628060,
        ('compound', 'v9', 'contrast_clean'): 0.10939654, ('compound', 'v9', 'delta_r_clean'): 0.64171444}
REF_TOL = 1e-4


def sha1_file(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def self_check(cell, comp):
    """Review 092 C3: refuse to report if v9 / XPert no longer reproduce the registered references."""
    got = {}
    for k, want in REFS.items():
        v = cell[k[1]][k[2]][k[3]]['row_mean'] if k[0] == 'cell' else comp[k[1]][k[2]]
        got['/'.join(k)] = v
        if abs(v - want) > REF_TOL:
            raise SystemExit(f'FATAL: {k} = {v:.6f}, registered reference {want:.6f} (092 C3): a reused scorer changed')
    if comp['v9']['n_contrast_rows'] != 7980:
        raise SystemExit('FATAL: C1 row count is not 7,980 (092 C3)')
    return got


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


def completed_preds(runs_dir, prefix, arm, ledger=None):
    """The first completed attempt's prediction file per seed slot (tc_read's rule), its record, and the slots without one.
    Review 092 C1: the record must match the ledger's sha1, and the prediction file its record's pred_sha1."""
    ledger = ledger or tc_ledger.LEDGER
    rec_sha, preds, records, missing = tc_ledger.ledger_sha1s(ledger), [], [], []
    for s in tc_read.SEEDS:
        slot = f'{prefix}_{arm}_s{s}'
        k, rec = tc_read.first_completed(tc_read.slot_attempts(runs_dir, arm, s, prefix=prefix, ledger=ledger))
        if rec is None:
            missing.append(s)
            continue
        if rec_sha.get((slot, k)) != sha1_file(rec):
            raise SystemExit(f'FATAL: {slot} attempt {k}: the record is not the one in the ledger (092 C1)')
        r = json.load(open(rec, encoding='utf-8'))
        p = os.path.join(os.path.dirname(rec), f"v10c_pred_{r['tag']}.npz")
        if sha1_file(p) != r.get('pred_sha1'):
            raise SystemExit(f'FATAL: {p} is not the prediction file its record names (092 C1)')
        preds.append(p)
        records.append(r)
    return preds, records, missing


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
    ap_.add_argument('--ledger', default=tc_ledger.LEDGER)
    a = ap_.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- read once')
    import audit99_phase1 as ap
    S = ap._load_p7_side()
    p9, _, _, plate9, v9_comp = ap._load_p9_side(ap.P9_DIR)
    tc_preds, tc_records, tc_missing = completed_preds(a.runs_dir, 'tc', 'on', a.ledger)
    gap_preds, _, gap_missing = completed_preds(a.runs_dir, 'gap', 'on', a.ledger)
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
    if v10_cell:                                            # 092 C2: over completed slots, n stated
        cell_models['v10c_full_release'] = v10_cell
    comp_models = {'v9': v9_comp}
    if v10_comp:
        comp_models['v10c_full_release'] = v10_comp
    cell = cell_side(ap, S, cell_models)
    comp = compound_side(ap, p9, plate9, comp_models)
    refs = self_check(cell, comp)
    import audit99_phase1 as _a, score_p10 as _s, coldcell_h2h as _c
    provenance = {'modules': {os.path.basename(m.__file__): sha1_file(m.__file__) for m in (_a, _s, _c)},
                  'inputs': {os.path.basename(p): sha1_file(p) for p in
                             sorted(glob.glob(os.path.join(ap.P7_DIR, 'v9p7_seed[012].npz')) +
                                    glob.glob(os.path.join(ap.P9_DIR, 'v9p9_seed[012].npz'))) + [ap.O2]}}
    scorers = None
    if tc_records:                                          # 092 C4, reported only
        trainer = {c: float(np.mean([r['scores']['per_cell'][c]['contrast_r'] for r in tc_records])) for c in CLEAN}
        here = cell['v10c_full_release']['contrast']['clean7']['per_cell']
        scorers = {'trainer_records': trainer, 'tc_gap': here,
                   'max_abs_difference': float(max(abs(trainer[c] - here[c]) for c in CLEAN))}
    rec = {'rule': 'RESULTS 100.5 / 100.11 / review 092 (reported, no claim)',
           'tc_on_slots_used': len(tc_preds), 'tc_on_missing_slots': tc_missing,
           'gap_arm_slots_used': len(gap_preds), 'gap_arm_missing_slots': gap_missing,
           'reference_self_check': refs, 'provenance': provenance, 'clean7_contrast_two_scorers': scorers,
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
    print(json.dumps({'gap': rec['gap'], 'tc_on_slots': len(tc_preds), 'gap_slots': len(gap_preds),
                      'two_scorers_max_diff': None if scorers is None else scorers['max_abs_difference']}, indent=1))


if __name__ == '__main__':
    main()
