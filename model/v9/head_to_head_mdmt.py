# -*- coding: utf-8 -*-
"""
The paired head-to-head on XPert's main benchmark: their checkpoint vs v9, SAME rows, THEIR metric.

RESULTS 39 compared our number on our rows against their number on their rows and read the difference as a
model difference. RESULTS 40 fixed that for one 3,439-row figure subset. This does it on the benchmark the
field actually quotes -- 13,766 held-out conditions of the 68,830-row mdmt corpus -- and does it PAIRED:
both models predict the same row, so the per-row difference is a real quantity and the comparison gets a
signed-rank test and a bootstrap CI rather than two summary numbers side by side.

Inputs are produced by, and only by:
    model/v9/xpert_native_eval.py  --nfold split_1 --out xpert_split_1.npy      (their weights, their code)
    model/v9/xpert_arm.py --bundle xpert_mdmt_splits.npz --split split_1 --save_pred v9_split_1.npz

Both carry `row_index` into their h5ad. This script REFUSES to compare unless those indices match exactly,
because two 13,766-row arrays built in different orders would otherwise pair row i with row j and produce
a plausible, wrong number.

Stratifications reported, because each is a caveat a reader would otherwise have to take on trust:
  * pooled-dose rows      18.9 % of their corpus pools 2-8 doses into one condition
  * chromatin-known cells v9 gets no epigenetic input for cells outside our panel
  * (cell, compound) seen  whether v9 saw that pair in its own training rows

    python model/v9/head_to_head_mdmt.py --theirs xpert_split_1.npy --ours v9_split_1_seed0.npz
"""
import os, sys, json, argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = r'C:\Projects\LINCS\external\xpert_split_bundle\xpert_mdmt_splits.npz'


def per_row_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


def summarize(r, boot=2000, seed=0):
    """XPert's metric is the MEAN of per-row Pearson. The median is reported alongside because this
    project has always used it, and the two differ once a left tail of hard rows exists."""
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(seed)
    bs = np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(boot)])
    return {'mean': round(float(r.mean()), 4),
            'ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            'median': round(float(np.median(r)), 4), 'n': int(len(r))}


def paired(a, b, boot=2000, seed=0):
    """a - b on the SAME rows: bootstrap CI of the mean difference, plus a Wilcoxon signed-rank test."""
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(boot)])
    out = {'delta_mean': round(float(d.mean()), 4),
           'ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
           'frac_rows_a_better': round(float((d > 0).mean()), 4), 'n': int(m.sum())}
    try:
        from scipy.stats import wilcoxon
        out['wilcoxon_p'] = float(wilcoxon(a[m], b[m]).pvalue)
    except Exception as e:
        out['wilcoxon_p'] = 'unavailable (%s)' % type(e).__name__
    return out


def load_theirs(p):
    d = np.load(p, allow_pickle=True)
    d = d.item() if isinstance(d, np.ndarray) and d.dtype == object else d
    return {k: np.asarray(d[k]) for k in ['y_true', 'y_pred', 'ctl_true', 'row_index']}


def load_ours(p):
    z = np.load(p)
    return {k: z[k] for k in ['y_pred', 'y_true', 'ctl_true', 'row_index']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theirs', required=True)
    ap.add_argument('--ours', nargs='+', required=True, help='one .npz per v9 seed')
    ap.add_argument('--bundle', default=BUNDLE)
    ap.add_argument('--split', default='split_1')
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default='v9 (ours)',
                    help='what --ours actually is; passing a baseline through here and leaving the label '
                         'as v9 would put a wrong row in a results table')
    ap.add_argument('--min_overlap', type=float, default=0.95,
                    help='refuse if the two models score fewer than this fraction of the same rows')
    a = ap.parse_args()

    T = load_theirs(a.theirs)
    seeds = [load_ours(p) for p in a.ours]

    # Their checkpoint scores every row of their split; ours cannot score rows whose compound we have no
    # features for (164 of 13,766 on split_1). The comparison therefore runs on the INTERSECTION, which is
    # reported rather than absorbed -- and it must be nearly all of their split, or the two models are
    # being compared on different tasks.
    common = set(int(r) for r in T['row_index'])
    for O in seeds:
        common &= set(int(r) for r in O['row_index'])
    common = np.array(sorted(common), np.int64)
    frac = len(common) / max(1, len(T['row_index']))
    if frac < a.min_overlap:
        raise SystemExit('FATAL: only %d of %d rows are scored by both models (%.1f%%), below the %.0f%% '
                         'floor -- these are different rows, not a comparison'
                         % (len(common), len(T['row_index']), 100 * frac, 100 * a.min_overlap))
    if len(common) < len(T['row_index']):
        print('comparing on the %d of %d rows BOTH models can score (%d dropped: no drug features on our '
              'side)' % (len(common), len(T['row_index']), len(T['row_index']) - len(common)), flush=True)

    def take(d, ridx):
        pos = {int(r): i for i, r in enumerate(d['row_index'])}
        sel = np.array([pos[int(r)] for r in ridx])
        return {k: (v[sel] if hasattr(v, '__len__') and len(v) == len(d['row_index']) else v)
                for k, v in d.items()}

    T = take(T, common)
    seeds = [take(O, common) for O in seeds]
    for O in seeds:
        if not np.allclose(O['y_true'], T['y_true'], atol=1e-3):
            raise SystemExit('FATAL: the two runs disagree about the TARGET on aligned rows; '
                             'one of them is not scoring what it thinks it is')
        if not np.allclose(O['ctl_true'], T['ctl_true'], atol=1e-3):
            raise SystemExit('FATAL: the two runs disagree about the CONTROL on aligned rows')

    y, ctl = T['y_true'], T['ctl_true']
    deg_true = y - ctl
    r_them_abs = per_row_pearson(T['y_pred'], y)
    r_them_deg = per_row_pearson(T['y_pred'] - ctl, deg_true)
    r_ours_abs = np.mean([per_row_pearson(O['y_pred'], y) for O in seeds], 0)
    r_ours_deg = np.mean([per_row_pearson(O['y_pred'] - ctl, deg_true) for O in seeds], 0)
    r_copy_abs = per_row_pearson(ctl, y)

    res = {'split': a.split, 'n_rows': int(len(y)), 'n_seeds': len(seeds),
           'metric': 'per-row Pearson; mean is XPert metrics.py convention',
           'XPert_released_ckpt': {'abs': summarize(r_them_abs), 'delta': summarize(r_them_deg)},
           'ours_label': a.label,
           'v9_ours': {'abs': summarize(r_ours_abs), 'delta': summarize(r_ours_deg)},
           'copy_the_control': {'abs': summarize(r_copy_abs), 'delta': 0.0},
           'paired_delta_ours_minus_theirs': paired(r_ours_deg, r_them_deg),
           'paired_abs_ours_minus_theirs': paired(r_ours_abs, r_them_abs),
           'per_seed_delta': [round(float(np.nanmean(per_row_pearson(O['y_pred'] - ctl, deg_true))), 4)
                              for O in seeds]}

    print('\n' + '=' * 100)
    print('XPERT vs v9 ON XPERT\'S OWN mdmt BENCHMARK (%s), %d identical held-out rows' % (a.split, len(y)))
    print('=' * 100)
    print('  %-26s %-26s %-26s' % ('', 'absolute Pearson', 'delta Pearson (Pearson_deg)'))
    for name, ab, dl in [('XPert released ckpt', res['XPert_released_ckpt']['abs'],
                          res['XPert_released_ckpt']['delta']),
                         (a.label, res['v9_ours']['abs'], res['v9_ours']['delta']),
                         ('copy-the-control', res['copy_the_control']['abs'], None)]:
        d = '%.4f %s' % (dl['mean'], dl['ci95']) if dl else '0.0000  (by construction)'
        print('  %-26s %.4f %-19s %s' % (name, ab['mean'], str(ab['ci95']), d))
    p = res['paired_delta_ours_minus_theirs']
    print('')
    print('  paired delta (%s - XPert) : %+.4f %s   better on %.1f%% of rows   wilcoxon p=%s'
          % (a.label, p['delta_mean'], p['ci95'], 100 * p['frac_rows_a_better'],
             ('%.3g' % p['wilcoxon_p']) if isinstance(p['wilcoxon_p'], float) else p['wilcoxon_p']))

    # ---- signal strength: this project has been caught twice by a number that only held on one
    # stratum [RESULTS 25, 35], so the comparison is reported by quartile of the TRUE effect size ----
    strength = np.abs(deg_true).mean(1)
    q = np.quantile(strength, [0.25, 0.5, 0.75])
    res['strata_strength'] = {}
    print('')
    print('  delta Pearson by quartile of TRUE effect size (mean |delta| per row):')
    print('    %-26s %6s %9s %9s' % ('quartile', 'n', 'XPert', a.label[:9]))
    for qi, (lo, hi) in enumerate(zip([-np.inf] + list(q), list(q) + [np.inf])):
        m = (strength >= lo) & (strength < hi)
        if m.sum() < 50:
            continue
        nm = 'Q%d  mean|d| %.2f-%.2f' % (qi + 1, max(lo, strength.min()), min(hi, strength.max()))
        res['strata_strength'][nm] = {'n': int(m.sum()),
                                      'XPert': round(float(np.nanmean(r_them_deg[m])), 4),
                                      'ours': round(float(np.nanmean(r_ours_deg[m])), 4)}
        print('    %-26s %6d %9.4f %9.4f' % (nm, m.sum(), np.nanmean(r_them_deg[m]),
                                             np.nanmean(r_ours_deg[m])))

    # ---- stratifications ----
    if os.path.exists(a.bundle):
        z = np.load(a.bundle, allow_pickle=True)
        ridx = T['row_index'].astype(np.int64)
        strat = {}
        pooled = z['meta_dose_pooled'][ridx].astype(bool) if 'meta_dose_pooled' in z.files else None
        if pooled is not None:
            for nm, m in [('single-dose rows', ~pooled), ('pooled-dose rows', pooled)]:
                if m.sum() > 50:
                    strat[nm] = {'n': int(m.sum()),
                                 'XPert': round(float(np.nanmean(r_them_deg[m])), 4),
                                 'ours': round(float(np.nanmean(r_ours_deg[m])), 4)}
        lab = z['split_' + a.split].astype(str)
        tr = np.flatnonzero(lab == 'train')
        pair_tr = set(zip(z['meta_cell'][tr].astype(str).tolist(),
                          z['meta_pert_id'][tr].astype(str).tolist()))
        seen = np.array([(c, d) in pair_tr for c, d in
                         zip(z['meta_cell'][ridx].astype(str), z['meta_pert_id'][ridx].astype(str))])
        for nm, m in [('(cell,compound) pair in train', seen), ('pair NOT in train', ~seen)]:
            if m.sum() > 50:
                strat[nm] = {'n': int(m.sum()),
                             'XPert': round(float(np.nanmean(r_them_deg[m])), 4),
                             'ours': round(float(np.nanmean(r_ours_deg[m])), 4)}
        res['strata_delta'] = strat
        print('\n  delta Pearson by stratum:')
        print('    %-34s %6s %9s %9s' % ('stratum', 'n', 'XPert', a.label[:9]))
        for nm, v in strat.items():
            print('    %-34s %6d %9.4f %9.4f' % (nm, v['n'], v['XPert'], v['ours']))

    dst = a.out or os.path.join(os.path.dirname(HERE), 'results', 'v9_vs_xpert_mdmt_%s.json' % a.split)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, 'w'), indent=2)
    print('\nwrote %s' % dst)


if __name__ == '__main__':
    main()
