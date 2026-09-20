# -*- coding: utf-8 -*-
"""Re-score the cold-cell chromatin ablation from SAVED PREDICTIONS. Zero GPU.

Why this script exists
----------------------
RESULTS 51.3: `v9_chromatin_ablation_cold_cell_1.json` had NO generating script, so the row-alignment
guards in `head_to_head_mdmt.py` never ran on the comparison the chromatin claim rested on. Adversarial
review 001 (C2) called that a BLOCKING provenance failure and it was right.

It also claimed no saved predictions existed, so the comparison could only be redone by retraining
(~21 GPU-h). That part was WRONG, and the fault is the packet's: both arms' `--save_pred` outputs have
existed since 2026-08-31 in `external/v9_mdmt_preds/`, which is gitignored, so a reviewer reading the
repository could not see them. Packets must list artefact paths.

So everything C1/C5 asked for is a CPU job:
  * C5  the PLACEBO STRATUM -- 3 of the 8 test cell lines have no chromatin track, so BOTH arms see
        identical zeros there. The paired delta on those rows is a direct, within-comparison estimate of
        the training-noise floor, needing no extra seeds and no transferred variance estimate.
  * C1  the correct uncertainty -- cluster-bootstrap over the 8 test CELL LINES, because rows within a
        cell line share the chromatin vector exactly and are not independent draws.

Run:  python model/v9/chromatin_ablation_analysis.py
"""
import io, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PRED = os.path.join(ROOT, 'external', 'v9_mdmt_preds')
BUNDLE = os.path.join(ROOT, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz')
OUT = os.path.join(ROOT, 'model', 'results', 'v9_chromatin_ablation_cold_cell_1_RESCORED.json')
SPLIT = 'split_split_cold_cell_1'


def per_row_pearson(a, b):
    """Pearson per row. Degenerate rows -> NaN, and we COUNT them rather than absorbing them (C12)."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    na = np.sqrt((a * a).sum(1))
    nb = np.sqrt((b * b).sum(1))
    den = na * nb
    out = np.full(len(a), np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = (a[ok] * b[ok]).sum(1) / den[ok]
    return out


def main():
    # ---------------- load both arms ----------------
    arms = {}
    for name, fn in (('epi_on', 'v9_cc1_epi_seed0.npz'), ('epi_ablated', 'v9_cc1_noepi_seed0.npz')):
        p = os.path.join(PRED, fn)
        if not os.path.exists(p):
            sys.exit('MISSING: %s' % p)
        arms[name] = np.load(p, allow_pickle=True)
        print('loaded %-12s %s  rows=%d' % (name, fn, arms[name]['row_index'].shape[0]))

    A, B = arms['epi_on'], arms['epi_ablated']

    # ---------------- the guards that never ran ----------------
    print('\n--- alignment guards (RESULTS 51.3: these never executed on the original number) ---')
    assert np.array_equal(A['row_index'], B['row_index']), 'row_index differs between arms'
    print('[PASS] row_index identical between arms  (n=%d)' % len(A['row_index']))
    for k in ('y_true', 'ctl_true'):
        m = np.abs(A[k] - B[k]).max()
        assert m == 0.0, '%s differs between arms by %.3e' % (k, m)
        print('[PASS] %-9s byte-identical between arms  (max|diff| = %.1e)' % (k, m))

    ri = A['row_index']
    y_true, ctl = A['y_true'], A['ctl_true']
    d_true = y_true - ctl

    # ---------------- cell identity, from the bundle ----------------
    bd = np.load(BUNDLE, allow_pickle=True)
    cells_all = bd['meta_cell']
    split_all = bd['split_%s' % SPLIT.replace('split_split_', 'split_cold_cell_')] \
        if ('split_%s' % SPLIT.replace('split_split_', 'split_cold_cell_')) in bd.files else bd[SPLIT]
    cell = cells_all[ri]
    assert (split_all[ri] == 'test').all(), 'some scored rows are not test rows in this split'
    print('[PASS] every scored row is a TEST row of %s' % SPLIT)
    tr_cells = set(np.unique(cells_all[split_all == 'train']))
    te_cells = sorted(set(np.unique(cell)))
    assert not (set(te_cells) & tr_cells), 'cold-cell split is not cold'
    print('[PASS] split is genuinely cold: %d test cells, %d train cells, intersection empty'
          % (len(te_cells), len(tr_cells)))

    # ---------------- chromatin coverage per cell ----------------
    idx = json.load(io.open(os.path.join(ROOT, 'epigenetics', 'outputs', 'epigenetics_cell_index.json'),
                            encoding='utf-8-sig'))
    cell_pos = {k: int(v) for k, v in idx['cell_id_to_row'].items()}
    mask = np.load(os.path.join(ROOT, 'phase2_assembly', 'outputs', 'E_final_mask.npy'))  # [83, 978, 3]
    assert mask.ndim == 3 and mask.shape[0] == idx['n_cells'], 'mask/index shape mismatch'
    # a TRACK counts as present for a cell if it is available for any gene
    ntrk = {c: (int(mask[cell_pos[c]].any(0).sum()) if c in cell_pos else 0) for c in te_cells}

    # ---------------- score ----------------
    r_on = per_row_pearson(A['deg_pred'], d_true)
    r_ab = per_row_pearson(B['deg_pred'], d_true)
    nan_rows = int(np.isnan(r_on).sum() + np.isnan(r_ab).sum())
    good = ~(np.isnan(r_on) | np.isnan(r_ab))
    print('\n[INFO] degenerate rows counted rather than absorbed (C12): %d' % nan_rows)

    diff = r_on - r_ab
    res = {
        'split': SPLIT, 'n_rows': int(good.sum()), 'n_degenerate_rows': nan_rows,
        'metric': 'per-row Pearson on delta; mean is XPert metrics.py convention',
        'source': 'RE-SCORED from saved predictions in external/v9_mdmt_preds/ -- ZERO GPU',
        'seeds_per_arm': 1,
        'epi_on_mean': round(float(np.nanmean(r_on)), 4),
        'epi_ablated_mean': round(float(np.nanmean(r_ab)), 4),
        'paired_diff_mean': round(float(np.nanmean(diff[good])), 6),
        'frac_rows_epi_better': round(float((diff[good] > 0).mean()), 4),
        'test_cells': {c: {'n_rows': int((cell == c).sum()), 'n_chromatin_tracks': ntrk[c]} for c in te_cells},
    }

    # ---------------- C5: the placebo stratum ----------------
    has = np.array([ntrk[c] > 0 for c in cell])
    print('\n--- C5 PLACEBO STRATUM: both arms see identical zeros where there is no chromatin ---')
    strata = {}
    for label, sel in (('chromatin_present', has & good), ('NO_chromatin_PLACEBO', (~has) & good), ('pooled', good)):
        if sel.sum() == 0:
            continue
        strata[label] = {
            'n_rows': int(sel.sum()),
            'cells': sorted(set(cell[sel].tolist())),
            'epi_on': round(float(np.nanmean(r_on[sel])), 4),
            'epi_ablated': round(float(np.nanmean(r_ab[sel])), 4),
            'paired_diff': round(float(np.nanmean(diff[sel])), 6),
        }
        print('  %-22s n=%6d  on=%.4f  ablated=%.4f  paired_diff=%+.6f  cells=%s'
              % (label, sel.sum(), strata[label]['epi_on'], strata[label]['epi_ablated'],
                 strata[label]['paired_diff'], ','.join(strata[label]['cells'])))
    res['strata'] = strata

    # ---------------- C1: per cell line, and a CLUSTER bootstrap ----------------
    print('\n--- C1 per-cell-line paired delta (rows within a cell share the chromatin vector exactly) ---')
    per_cell = {}
    for c in te_cells:
        sel = (cell == c) & good
        if sel.sum() == 0:
            continue
        per_cell[c] = {'n_rows': int(sel.sum()), 'n_tracks': ntrk[c],
                       'paired_diff': round(float(np.nanmean(diff[sel])), 6)}
        print('  %-10s n=%6d tracks=%d  paired_diff=%+.6f'
              % (c, sel.sum(), ntrk[c], per_cell[c]['paired_diff']))
    res['per_cell_line'] = per_cell

    cl = np.array([per_cell[c]['paired_diff'] for c in per_cell])
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(cl, size=len(cl), replace=True).mean() for _ in range(20000)])
    res['cluster_bootstrap_over_cell_lines'] = {
        'n_clusters': int(len(cl)),
        'mean_of_cell_line_means': round(float(cl.mean()), 6),
        'ci95': [round(float(np.percentile(boot, 2.5)), 6), round(float(np.percentile(boot, 97.5)), 6)],
        'frac_cell_lines_positive': round(float((cl > 0).mean()), 4),
    }
    cb = res['cluster_bootstrap_over_cell_lines']
    print('\n--- C1 CLUSTER BOOTSTRAP over %d cell lines (the correct denominator) ---' % cb['n_clusters'])
    print('  mean of per-cell-line means : %+.6f' % cb['mean_of_cell_line_means'])
    print('  95%% CI                      : [%+.6f, %+.6f]' % tuple(cb['ci95']))
    print('  cell lines with a positive effect: %.0f%%' % (100 * cb['frac_cell_lines_positive']))
    print('  ROW-bootstrap CI for comparison (the WRONG one, quoted in RESULTS 45): [+0.0036, +0.0049]')

    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(res, indent=2) + '\n')
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
