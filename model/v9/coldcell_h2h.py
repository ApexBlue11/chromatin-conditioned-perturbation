# -*- coding: utf-8 -*-
"""
The v9-versus-XPert head-to-head, read EXACTLY as RESULTS 71 / 71.7 pre-committed. [packets 007-008]

WHY A NEW SCRIPT RATHER THAN head_to_head_mdmt.py ALONE
head_to_head_mdmt.py pairs rows and reports a row-pooled paired mean. On split_cold_cell_1 that number is
mostly a statement about MCF7, which is 51.4 % of the test rows [RESULTS 71.1] -- the substitution that
produced the retracted chromatin +0.0042 against a cluster estimate of +0.00036. For a claim about UNSEEN
CELL LINES the unit is the cell line [review 006 C3], so the estimand of record here is per cell:

    d_c = median over cell c's rows of (r_ours - r_theirs),   r = per-row Pearson of (pred - ctl, y - ctl)

summarised as the unweighted mean of the d_c with a CLUSTER bootstrap over cells, plus a sign count. The
row-pooled number is still reported, labelled, because it is the convention the field quotes.

WHAT IT REFUSES
  * pairing unless row_index intersects on >= --min_overlap of the smaller set;
  * any pairing where the two files disagree about the TARGETS on shared rows (y_true, ctl_true) -- two
    models scored against different targets would produce a plausible, meaningless difference;
  * reading a v9 win from a run whose run_record says admissible_for_v9_win is False [RESULTS 71.7].

VALIDATION
Run on warm split_2 against XPert's RELEASED checkpoint, its row-pooled output must reproduce RESULTS 43,
+0.0120 [0.0113, 0.0127] on 13,615 rows. That is the check that the pairing and the row score are right.

P7 (RESULTS 85.2 rule 10, W22): --ours takes the 3 seed files; the row score is the MEAN of the per-file per-row
Pearsons (scores averaged, never predictions). Secondary blocks, never in the verdict: per_seed, centred (85.10 / 90.2,
both models), ensemble_own (v9 only, no paired difference, 90.6), alt (e.g. V2-last). The headline row set never depends
on --ours_alt.
"""
import argparse
import io
import json
import os

import h5py
import numpy as np
from scipy.stats import binomtest


def load_profile(path):
    z = np.load(path, allow_pickle=True)
    if isinstance(z, np.lib.npyio.NpzFile):
        return {k: z[k] for k in z.files}
    z = z.item() if z.dtype == object else z
    return {k: np.asarray(v) for k, v in z.items()}


def per_row_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def obs_col(h5ad, name):
    with h5py.File(h5ad, 'r') as f:
        d = f['obs'][name]
        if isinstance(d, h5py.Group):
            cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in d['categories'][:]])
            return cats[d['codes'][:]]
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in d[:]])


def boot_ci(x, stat, n_boot, rng):
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    b = stat(x[idx], axis=1)
    return [round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)]


def compute_cluster_and_pooled(diff, r_o_arr, r_t_arr, cells, u, n_by, top, n_boot, rng):
    # --- row-pooled
    pooled = {'n_rows': int(len(diff)),
              'ours_mean': round(float(r_o_arr.mean()), 5), 'theirs_mean': round(float(r_t_arr.mean()), 5),
              'paired_mean': round(float(diff.mean()), 5),
              'paired_mean_ci95_row_bootstrap': boot_ci(diff, np.mean, n_boot, rng),
              'paired_median': round(float(np.median(diff)), 5),
              'frac_rows_ours_better': round(float((diff > 0).mean()), 4),
              'largest_cell': str(top), 'largest_cell_share': round(float(n_by.max() / n_by.sum()), 4),
              'LABEL': 'row-pooled; dominated by %s at %.1f%% of rows; NOT a cell-level generalisation claim'
                       % (top, 100 * n_by.max() / n_by.sum())}
    # --- per cell
    per_cell = []
    for c in sorted(u, key=lambda c: -int((cells == c).sum())):
        m = cells == c
        dc = diff[m]
        per_cell.append({'cell': str(c), 'n_scored': int(m.sum()),
                         'd_c_median': round(float(np.median(dc)), 5),
                         'd_c_median_ci95': boot_ci(dc, np.median, n_boot, rng),
                         'mean': round(float(dc.mean()), 5),
                         'ours_mean': round(float(r_o_arr[m].mean()), 5),
                         'theirs_mean': round(float(r_t_arr[m].mean()), 5),
                         'favours': 'ours' if np.median(dc) > 0 else ('theirs' if np.median(dc) < 0 else 'tie')})
    d = np.array([pc['d_c_median'] for pc in per_cell])
    k = len(d)
    cb = d[rng.integers(0, k, size=(n_boot, k))].mean(1)
    cl_ci = [round(float(np.percentile(cb, 2.5)), 5), round(float(np.percentile(cb, 97.5)), 5)]
    n_ours = int((d > 0).sum())
    n_theirs = int((d < 0).sum())
    nz = n_ours + n_theirs
    cluster = {'n_cells': k, 'mean_of_d_c': round(float(d.mean()), 5), 'cluster_ci95': cl_ci,
               'cluster_ci_width': round(cl_ci[1] - cl_ci[0], 5),
               'cells_favouring_ours': n_ours, 'cells_favouring_theirs': n_theirs,
               'sign_test_p': float(binomtest(n_ours, nz, 0.5).pvalue) if nz else 1.0}
    return cluster, per_cell, pooled


def compute_centred_pearson(pred_deltas, true_deltas, cells, u):
    pred_c = np.copy(pred_deltas)
    true_c = np.copy(true_deltas)
    for c in u:
        m = cells == c
        if m.sum() > 0:
            pred_c[m] = pred_deltas[m] - pred_deltas[m].mean(axis=0, keepdims=True)
            true_c[m] = true_deltas[m] - true_deltas[m].mean(axis=0, keepdims=True)
    return per_row_pearson(pred_c, true_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theirs', required=True, help='XPert profile .npy with row_index')
    ap.add_argument('--ours', required=True, nargs='+', help='v9 predictions .npz with row_index')
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--run_record', default=None, help="the kernel's run_record.json, for 71.7 admissibility")
    ap.add_argument('--theirs_label', required=True)
    ap.add_argument('--n_boot', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min_overlap', type=float, default=0.95)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ours_alt', nargs='+', help='alternative v9 predictions')
    ap.add_argument('--ours_alt_label', default='alt')
    a = ap.parse_args()

    T = load_profile(a.theirs)
    O_list = [load_profile(p) for p in a.ours]
    O0 = O_list[0]

    for name, P in (('theirs', T), ('ours', O0)):
        for k in ('row_index', 'y_pred', 'y_true', 'ctl_true'):
            if k not in P:
                raise SystemExit('FATAL: %s profile has no %r' % (name, k))
                
    for O in O_list[1:]:
        for k in ('row_index', 'y_true', 'ctl_true'):
            if k not in O or not np.array_equal(O[k], O0[k]):
                raise SystemExit('FATAL: --ours files disagree on %r' % k)
                
    if a.ours_alt:
        O_alt_list = [load_profile(p) for p in a.ours_alt]
        for O in O_alt_list:
            for k in ('row_index', 'y_true', 'ctl_true'):
                if k not in O or not np.array_equal(O[k], O0[k]):
                    raise SystemExit('FATAL: --ours_alt files disagree with --ours on %r' % k)

    # --- pairing on row_index -------------------------------------------------------------------------
    common, it, io_ = np.intersect1d(T['row_index'], O0['row_index'], return_indices=True)
    overlap = len(common) / min(len(T['row_index']), len(O0['row_index']))
    if overlap < a.min_overlap:
        raise SystemExit('FATAL: only %.1f%% row overlap; refusing to pair.' % (100 * overlap))

    # --- the two files must agree about the TARGETS on every shared row --------------------------------
    for k in ('y_true', 'ctl_true'):
        dmax = float(np.abs(T[k][it] - O0[k][io_]).max())
        if dmax > 1e-4:
            raise SystemExit('FATAL: %s differs between the two files by %.3g on shared rows. They are not '
                             'scored against the same target.' % (k, dmax))
    y, ctl = T['y_true'][it], T['ctl_true'][it]
    
    r_t_raw = per_row_pearson(T['y_pred'][it] - ctl, y - ctl)
    r_o_list_raw = [per_row_pearson(O['y_pred'][io_] - ctl, y - ctl) for O in O_list]
    
    ok = np.isfinite(r_t_raw)
    for r in r_o_list_raw:
        ok &= np.isfinite(r)
        
    if a.ours_alt:                      # computed here, but its finiteness never narrows the headline rows (PI fix)
        r_o_alt_list_raw = [per_row_pearson(O['y_pred'][io_] - ctl, y - ctl) for O in O_alt_list]

    r_t, rows = r_t_raw[ok], common[ok]
    r_o_list = [r[ok] for r in r_o_list_raw]
    r_o = np.mean(r_o_list, axis=0)
    diff = r_o - r_t
    rng = np.random.default_rng(a.seed)

    cells_all = obs_col(a.h5ad, 'cell_iname')
    cells = cells_all[rows]
    u, n_by = np.unique(cells, return_counts=True)
    top = u[np.argmax(n_by)]

    cluster, per_cell, pooled = compute_cluster_and_pooled(diff, r_o, r_t, cells, u, n_by, top, a.n_boot, rng)

    # --- reproduction check, on ALL of their rows, their convention ------------------------------------
    r_all = per_row_pearson(T['y_pred'] - T['ctl_true'], T['y_true'] - T['ctl_true'])
    theirs_all = float(np.nanmean(r_all))
    repro = {'theirs_all_rows_mean': round(theirs_all, 5), 'n': int(np.isfinite(r_all).sum()),
             'band': [0.302, 0.464], 'inside_band': bool(0.302 <= theirs_all <= 0.464)}

    # --- admissibility [71.7] --------------------------------------------------------------------------
    adm, rec = True, None
    if a.run_record:
        rec = json.load(open(a.run_record))
        adm = bool(rec.get('admissible_for_v9_win', False))

    # --- the pre-committed reading [71.3 + 71.7], applied mechanically ---------------------------------
    wide = cluster['cluster_ci_width'] > 0.10
    if not repro['inside_band'] and a.run_record:
        verdict = 'REPRODUCTION FLAG: our XPert scores outside [0.302, 0.464]; nothing concluded until explained'
    elif wide:
        verdict = 'UNINFORMATIVE: cluster CI wider than 0.10 [71.3]; no cell-level claim'
    elif cluster['mean_of_d_c'] > 0 and cluster['cluster_ci95'][0] > 0 and cluster['cells_favouring_ours'] >= 7:
        verdict = ('V9 WINS, conservative [71.3]' if adm else
                   'NO V9-WIN CLAIM: run not admissible, XPert was cut off still improving [71.7]')
    elif cluster['mean_of_d_c'] < 0 and cluster['cluster_ci95'][1] < 0 and cluster['cells_favouring_theirs'] >= 7:
        verdict = 'XPERT WINS: UNINTERPRETABLE as a model comparison -- test-guided checkpoint selection [69.1]'
    else:
        verdict = 'NO CELL-LEVEL CLAIM [71.3]'

    ours_label = os.path.basename(a.ours[0]) if len(a.ours) == 1 else [os.path.basename(p) for p in a.ours]
    out = {'split': a.split, 'theirs_label': a.theirs_label, 'ours': ours_label,
           'theirs': os.path.basename(a.theirs), 'row_overlap': round(overlap, 4),
           'rows_theirs': int(len(T['row_index'])), 'rows_ours': int(len(O0['row_index'])),
           'row_score': 'per-row Pearson of (pred - ctl) against (y - ctl)',
           'estimand_of_record': 'mean over cells of d_c = median_rows(r_ours - r_theirs), cluster bootstrap',
           'cluster': cluster, 'per_cell': per_cell, 'row_pooled': pooled, 'reproduction': repro,
           'admissible_for_v9_win': adm, 'run_record': rec, 'verdict': verdict,
           'n_boot': a.n_boot, 'seed': a.seed}

    # 2. per_seed block (only when > 1 file)
    if len(O_list) > 1:
        per_seed = []
        for i, r_o_s in enumerate(r_o_list):
            rng_s = np.random.default_rng(a.seed)
            cl_s, pc_s, po_s = compute_cluster_and_pooled(r_o_s - r_t, r_o_s, r_t, cells, u, n_by, top, a.n_boot, rng_s)
            per_seed.append({
                'file': os.path.basename(a.ours[i]),
                'mean_of_d_c': cl_s['mean_of_d_c'],
                'cluster_ci95': cl_s['cluster_ci95'],
                'cells_favouring_ours': cl_s['cells_favouring_ours'],
                'cells_favouring_theirs': cl_s['cells_favouring_theirs'],
                'row_pooled': po_s
            })
        out['per_seed'] = per_seed
        out['per_seed_label'] = "per seed, secondary; the estimand of record is the seed-averaged score"

    # 3. centred block (always)
    rng_c = np.random.default_rng(a.seed + 1)
    y_ok = y[ok]
    ctl_ok = ctl[ok]
    true_delta = y_ok - ctl_ok
    pred_t_delta = T['y_pred'][it][ok] - ctl_ok
    
    r_t_c = compute_centred_pearson(pred_t_delta, true_delta, cells, u)
    
    r_o_c_list = []
    for O in O_list:
        pred_o_delta = O['y_pred'][io_][ok] - ctl_ok
        r_o_c_list.append(compute_centred_pearson(pred_o_delta, true_delta, cells, u))
    
    r_o_c = np.mean(r_o_c_list, axis=0)
    cl_c, pc_c, po_c = compute_cluster_and_pooled(r_o_c - r_t_c, r_o_c, r_t_c, cells, u, n_by, top, a.n_boot, rng_c)
    
    out['centred'] = {
        'ours_mean_over_rows': round(float(r_o_c.mean()), 5),
        'theirs_mean_over_rows': round(float(r_t_c.mean()), 5),
        'per_cell': pc_c,
        'cluster': cl_c,
        'LABEL': "cell-centred (RESULTS 85.10 / 90.2), labelled secondary for both models"
    }

    # 4. ensemble_own block (only when > 1 file)
    if len(O_list) > 1:
        pred_o_avg = np.mean([O['y_pred'][io_][ok] for O in O_list], axis=0)
        r_ens = per_row_pearson(pred_o_avg - ctl_ok, true_delta)
        
        per_cell_ens = []
        for c in sorted(u, key=lambda c: -int((cells == c).sum())):
            m = cells == c
            per_cell_ens.append({
                'cell': str(c),
                'n_scored': int(m.sum()),
                'ours_mean': round(float(r_ens[m].mean()), 5)
            })
            
        out['ensemble_own'] = {
            'ours_mean_over_rows': round(float(r_ens.mean()), 5),
            'per_cell': per_cell_ens,
            'LABEL': "v9 seed ensemble — v9's own labelled secondary; NOT a comparison with XPert's single run (RESULTS 90.6)"
        }

    # 5. ours_alt block
    if a.ours_alt:
        r_o_alt = np.mean([r[ok] for r in r_o_alt_list_raw], axis=0)
        fa = np.isfinite(r_o_alt)                      # alt-only exclusions, reported, headline rows untouched
        ca = cells[fa]
        ua, na = np.unique(ca, return_counts=True)
        rng_alt = np.random.default_rng(a.seed)
        cl_alt, pc_alt, po_alt = compute_cluster_and_pooled(r_o_alt[fa] - r_t[fa], r_o_alt[fa], r_t[fa], ca, ua, na,
                                                            ua[np.argmax(na)], a.n_boot, rng_alt)
        out['alt'] = {
            'label': a.ours_alt_label,
            'n_rows_dropped_alt_only': int((~fa).sum()),
            'cluster': cl_alt,
            'row_pooled': po_alt,
            'files': [os.path.basename(p) for p in a.ours_alt]
        }

    io.open(a.out, 'w', encoding='utf-8').write(json.dumps(out, indent=2, default=str) + '\n')

    # 6. Print new blocks after existing output without changing existing print lines
    print('=' * 96)
    print('%s vs ours on %s | %d paired rows (%.1f%% overlap)' % (a.theirs_label, a.split, len(diff), 100 * overlap))
    print('=' * 96)
    print('%-10s %6s %10s %24s %9s %9s' % ('cell', 'n', 'd_c', 'CI95', 'ours', 'theirs'))
    for pc in per_cell:
        print('%-10s %6d %+10.5f   [%+.5f, %+.5f] %9.4f %9.4f' % (pc['cell'], pc['n_scored'], pc['d_c_median'],
              pc['d_c_median_ci95'][0], pc['d_c_median_ci95'][1], pc['ours_mean'], pc['theirs_mean']))
    print('\nCLUSTER (estimand of record): mean d_c %+.5f  CI95 %s  width %.5f  | cells ours %d / theirs %d  '
          'sign p %.3g' % (cluster['mean_of_d_c'], cluster['cluster_ci95'], cluster['cluster_ci_width'], cluster['cells_favouring_ours'], cluster['cells_favouring_theirs'],
                           cluster['sign_test_p']))
    print('ROW-POOLED (labelled): paired mean %+.5f  CI95 %s  median %+.5f  -- %s'
          % (pooled['paired_mean'], pooled['paired_mean_ci95_row_bootstrap'], pooled['paired_median'],
             pooled['LABEL']))
    print('REPRODUCTION: theirs on all rows %.5f  inside [0.302, 0.464]: %s' % (theirs_all, repro['inside_band']))
    print('ADMISSIBLE for a v9-win claim [71.7]:', adm)
    print('\nVERDICT:', verdict)
    print('wrote', a.out)

    if len(O_list) > 1:
        print('\n--- PER SEED (secondary) ---')
        for sinfo in out['per_seed']:
            print('%s: cluster mean %+.5f CI95 %s (ours %d/theirs %d), row-pooled %+.5f' % (
                sinfo['file'], sinfo['mean_of_d_c'], sinfo['cluster_ci95'],
                sinfo['cells_favouring_ours'], sinfo['cells_favouring_theirs'],
                sinfo['row_pooled']['paired_mean']
            ))

    print('\n--- CENTRED (secondary) ---')
    print('ours: %.5f | theirs: %.5f' % (out['centred']['ours_mean_over_rows'], out['centred']['theirs_mean_over_rows']))
    print('cluster: mean d_c %+.5f CI95 %s (ours %d/theirs %d)' % (
        cl_c['mean_of_d_c'], cl_c['cluster_ci95'], cl_c['cells_favouring_ours'], cl_c['cells_favouring_theirs']
    ))

    if len(O_list) > 1:
        print('\n--- ENSEMBLE OWN (secondary) ---')
        print('ours: %.5f' % (out['ensemble_own']['ours_mean_over_rows']))

    if a.ours_alt:
        print('\n--- ALT: %s ---' % a.ours_alt_label)
        print('cluster: mean d_c %+.5f CI95 %s' % (cl_alt['mean_of_d_c'], cl_alt['cluster_ci95']))
        print('row-pooled: paired mean %+.5f' % po_alt['paired_mean'])


if __name__ == '__main__':
    main()
