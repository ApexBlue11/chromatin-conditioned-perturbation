# -*- coding: utf-8 -*-
"""
How much of the residual error on XPert's benchmark is measurement noise in the TARGET rather than model
error? Their aggregated conditions carry `n_replicates` -- the number of wells averaged into the row --
so the question is answerable without any replicate-splitting of our own.

This project measured a Level-3 delta self-agreement of 0.5283 on its own data [RESULTS 27.3] and has
argued since that a delta Pearson has to be read against a noise ceiling. Their benchmark ships no ceiling,
and their metric reports none. `n_replicates` is a proxy for one: averaging k wells cuts the target's noise
by ~sqrt(k), so if the score is noise-limited it must rise with k.

THE CONFOUND, and why the raw number understates the effect. Conditions with more replicates have WEAKER
measured effects here (mean |delta| falls 0.560 -> 0.291 from 1 to 6+ wells), and weak effects are harder
to predict, so the two pull in opposite directions. Controlling for effect size roughly doubles the
apparent replicate effect. Reporting only the raw correlation would have understated it by half.

    python model/v9/replicate_noise_mdmt.py --pred xpert_split_2_test.npy --label XPert
"""
import os, sys, json, argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H5AD = r'C:\Projects\LINCS\external\xpert\code\XPert\processed_data\l1000_mdmt_68830_subset.h5ad'


def per_row_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


def load_pred(p, key='y_pred'):
    d = np.load(p, allow_pickle=True)
    d = d.item() if isinstance(d, np.ndarray) and d.dtype == object else d
    if key not in d:
        avail = [k for k in (d.files if hasattr(d, 'files') else d) if k.endswith('_pred')]
        raise SystemExit('FATAL: %r not in %s; prediction columns present: %s' % (key, p, avail))
    out = {k: np.asarray(d[k]) for k in ['y_true', 'ctl_true', 'row_index']}
    out['y_pred'] = np.asarray(d[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='a *_predict_profile-style .npy or .npz with row_index')
    ap.add_argument('--label', default='model')
    ap.add_argument('--pred_key', default='y_pred',
                    help='which prediction column to score (the baselines file holds several)')
    ap.add_argument('--h5ad', default=H5AD)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    import anndata as ad

    P = load_pred(a.pred, a.pred_key)
    obs = ad.read_h5ad(a.h5ad, backed='r').obs
    nrep = np.asarray(obs['n_replicates']).astype(str)
    # a pooled row carries a ';'-joined list; the first entry is that condition's own well count
    nrep = np.array([int(float(x.split(';')[0])) for x in nrep])[P['row_index'].astype(int)]

    ctl = P['ctl_true']
    deg_true = P['y_true'] - ctl
    r = per_row_pearson(P['y_pred'] - ctl, deg_true)
    strength = np.abs(deg_true).mean(1)
    q = np.quantile(strength, [0.25, 0.5, 0.75])
    qi = np.digitize(strength, q)
    bins = [(1, 2, 'n<=2'), (3, 3, 'n=3'), (4, 5, 'n=4-5'), (6, 10 ** 6, 'n>=6')]

    res = {'label': a.label, 'n_rows': int(len(r)), 'raw': {}, 'by_quartile': {}}
    print('\n%s: delta Pearson vs replicate count (n=%d rows)' % (a.label, len(r)))
    print('  %-10s %7s %9s %12s' % ('replicates', 'rows', 'Pearson', 'mean|delta|'))
    for lo, hi, nm in bins:
        m = (nrep >= lo) & (nrep <= hi)
        if m.sum() < 40:
            continue
        res['raw'][nm] = {'n': int(m.sum()), 'pearson': round(float(np.nanmean(r[m])), 4),
                          'mean_abs_delta': round(float(np.abs(deg_true[m]).mean()), 4)}
        print('  %-10s %7d %9.4f %12.3f' % (nm, m.sum(), np.nanmean(r[m]), np.abs(deg_true[m]).mean()))

    print('\n  the confound: more replicates go with WEAKER measured effects, and weak effects are harder,')
    print('  so the raw column understates the noise effect. Within effect-size quartiles:')
    print('  %-8s %9s %9s %9s %9s' % ('quartile', 'n<=2', 'n=3', 'n=4-5', 'n>=6'))
    for k in range(4):
        row, cells = [], {}
        for lo, hi, nm in bins:
            m = (qi == k) & (nrep >= lo) & (nrep <= hi)
            if m.sum() >= 40:
                v = float(np.nanmean(r[m]))
                cells[nm] = {'n': int(m.sum()), 'pearson': round(v, 4)}
                row.append('%9.4f' % v)
            else:
                row.append('      n/a')
        res['by_quartile']['Q%d' % (k + 1)] = cells
        print('  Q%-7d %s' % (k + 1, ' '.join(row)))

    try:
        from scipy.stats import spearmanr
        raw = float(spearmanr(nrep, r).statistic)
        within = [float(spearmanr(nrep[qi == k], r[qi == k]).statistic) for k in range(4)]
        res['spearman_raw'] = round(raw, 4)
        res['spearman_within_quartile'] = [round(v, 4) for v in within]
        res['spearman_within_mean'] = round(float(np.mean(within)), 4)
        print('\n  Spearman(replicates, Pearson): raw %.4f  ->  within-quartile mean %.4f'
              % (raw, float(np.mean(within))))
    except ImportError:
        pass

    best = res['by_quartile'].get('Q4', {}).get('n>=6')
    if best:
        print('\n  On the best-measured stratum -- strongest effect quartile, >=6 replicate wells, n=%d --'
              % best['n'])
        print('  %s reaches %.4f. That is what the model does where the LABEL is most trustworthy;'
              % (a.label, best['pearson']))
        print('  the headline number is depressed by target noise, not only by model error.')

    dst = a.out or os.path.join(os.path.dirname(HERE), 'results',
                                'replicate_noise_%s.json' % a.label.replace(' ', '_'))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, 'w'), indent=2)
    print('\nwrote %s' % dst)


if __name__ == '__main__':
    main()
