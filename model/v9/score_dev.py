# -*- coding: utf-8 -*-
"""Score RESULTS 85 dev runs from their saved predictions (PI-written instrument).

    python model/v9/score_dev.py --preds A_seed0.npz [A_seed1.npz ...] --label NAME [--baseline B_seed*.npz]

Per row: Pearson of (pred delta, true delta) over the 978 genes, XPert's convention (`y_true - ctl_true` vs `deg_pred`).
Reports per seed: the per-row mean (primary, 85.2 rule 3), the mean of per-cell means (secondary), per-cell medians.
With --baseline: the paired per-cell comparison of the seed-mean per-row scores, and the 85.2 advance/accept inputs.
All files must share one row_index set (asserted), and it must hash to the 85.4 dev rows.
"""
import argparse
import hashlib
import json
import os

import numpy as np

DEV_SHA1 = '51e7e4ab8b9c3c3709d43da7fa4a8c80b77d5980'
HERE = os.path.dirname(os.path.abspath(__file__))


def row_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def cell_of_rows(row_index):
    bundle = None
    for root in (os.path.join(HERE, '..', '..', 'external'), '/kaggle/input'):
        for dp, _, fs in os.walk(root):
            if 'xpert_mdmt_splits.npz' in fs:
                bundle = os.path.join(dp, 'xpert_mdmt_splits.npz')
                break
        if bundle:
            break
    z = np.load(bundle, allow_pickle=True)
    ri = z['meta_row_index'] if 'meta_row_index' in z.files else np.arange(len(z['meta_cell']))
    pos = {int(r): i for i, r in enumerate(ri)}
    return np.array([z['meta_cell'][pos[int(r)]] for r in row_index])


def load(paths):
    runs = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        runs.append({'path': os.path.basename(p), 'row_index': np.asarray(z['row_index']).astype(np.int64),
                     'r': row_pearson(z['deg_pred'], z['y_true'] - z['ctl_true'])})
    ri = runs[0]['row_index']
    for r in runs[1:]:
        assert np.array_equal(r['row_index'], ri), 'row sets differ between runs'
    sha = hashlib.sha1(np.sort(ri).tobytes()).hexdigest()
    assert sha == DEV_SHA1, 'not the RESULTS 85.4 dev rows: %s' % sha
    return runs, ri


def summarise(runs, cells):
    out = []
    for r in runs:
        per_cell = {c: float(np.nanmedian(r['r'][cells == c])) for c in np.unique(cells)}
        out.append({'file': r['path'], 'per_row_mean': float(np.nanmean(r['r'])),
                    'mean_of_cell_means': float(np.mean([np.nanmean(r['r'][cells == c]) for c in np.unique(cells)])),
                    'per_cell_median': per_cell})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', nargs='+', required=True)
    ap.add_argument('--baseline', nargs='*', default=None)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    runs, ri = load(a.preds)
    cells = cell_of_rows(ri)
    res = {'label': a.label, 'n_rows': int(len(ri)), 'cells': {c: int((cells == c).sum()) for c in np.unique(cells)},
           'seeds': summarise(runs, cells)}
    m = [s['per_row_mean'] for s in res['seeds']]
    res['mean'] = float(np.mean(m))
    res['sd'] = float(np.std(m, ddof=1)) if len(m) > 1 else None
    if a.baseline:
        bruns, bri = load(a.baseline)
        assert np.array_equal(bri, ri)
        b = [s['per_row_mean'] for s in summarise(bruns, cells)]
        rv = np.nanmean(np.stack([r['r'] for r in runs]), 0)
        rb = np.nanmean(np.stack([r['r'] for r in bruns]), 0)
        per_cell = {c: float(np.nanmedian(rv[cells == c] - rb[cells == c])) for c in np.unique(cells)}
        res['vs_baseline'] = {
            'delta_per_row_mean': float(np.mean(m) - np.mean(b)),
            'delta_mean_of_cell_means': float(np.mean([np.nanmean(rv[cells == c]) - np.nanmean(rb[cells == c])
                                                       for c in np.unique(cells)])),
            'per_cell_median_delta': per_cell, 'cells_favouring': int(sum(v > 0 for v in per_cell.values())),
            'baseline_mean': float(np.mean(b)), 'baseline_sd': float(np.std(b, ddof=1)) if len(b) > 1 else None,
            'n_seeds_variant': len(m), 'n_seeds_baseline': len(b)}
    out = a.out or os.path.join(HERE, '..', 'results', 'v9_dev_score_%s.json' % a.label)
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'seeds'}, indent=1))
    for s in res['seeds']:
        print(s['file'], 'per-row mean %.4f | mean of cell means %.4f' % (s['per_row_mean'], s['mean_of_cell_means']))


if __name__ == '__main__':
    main()
