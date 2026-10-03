# -*- coding: utf-8 -*-
"""RESULTS 91.11 T4 / T4b, per dev cell, for Figure 9 (PI glue; reads saved predictions only, no model is run).

    python model/v9/t4_per_cell.py

Per arm (T4: all chromatin mean-ablated at inference; T4b: only the failed H3K27me3 tracks, HEK293T and VCAP), the per-row
Pearson (score_dev.row_pearson) of each seed's ablated predictions minus the same seed's intact P2 predictions, averaged over
the three seeds per row; then the mean over each cell's rows, and over all rows. Also the cell-centred difference (90.2) and
the per-seed overall differences. Asserts the values RESULTS 91.11 reports.
"""
import json
import os

import numpy as np

import score_dev as sd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'external', 'kaggle_out')
INTACT = [os.path.join(OUT, 'v9dev_base2', 'v9dev_base_dev6s0_seed%d.npz' % s) for s in range(3)]
ARMS = {'T4': [os.path.join(OUT, 't4_chromatin', 't4_ablate_seed%d.npz' % s) for s in range(3)],
        'T4b': [os.path.join(OUT, 't4_chromatin', 't4b_failed_tracks_seed%d.npz' % s) for s in range(3)]}
EXPECT = {'T4': {'all': 0.00596, 'HEK293T': 0.0225, 'LNCAP': 0.0272, 'VCAP': 0.0143, 'U937': 0.0003, 'SKBR3': -0.0038,
                 'HL60': -0.0133},
          'T4b': {'all': 0.00317, 'VCAP': 0.0133, 'HEK293T': -0.0002}}


def main():
    base, ri = sd.load(INTACT)
    cells = sd.cell_of_rows(ri)
    names = [str(c) for c in np.unique(cells)]
    res = {'source': 'RESULTS 91.11 (T4, T4b)', 'n_rows': int(len(ri)), 'cells': {c: int((cells == c).sum()) for c in names},
           'intact': [os.path.basename(p) for p in INTACT], 'arms': {}}
    for arm, paths in ARMS.items():
        runs, ri2 = sd.load(paths)
        assert np.array_equal(ri2, ri), arm
        d = np.stack([r['r'] - b['r'] for r, b in zip(runs, base)])          # seeds x rows
        dc = np.stack([sd.centred_r(r, cells) - sd.centred_r(b, cells) for r, b in zip(runs, base)])
        dm = np.nanmean(d, 0)
        res['arms'][arm] = {
            'files': [os.path.basename(p) for p in paths],
            'all': float(np.nanmean(dm)), 'centred_all': float(np.nanmean(np.nanmean(dc, 0))),
            'per_seed': [float(np.nanmean(x)) for x in d],
            'per_cell_mean': {c: float(np.nanmean(dm[cells == c])) for c in names},
            'per_cell_per_seed': {c: [float(np.nanmean(x[cells == c])) for x in d] for c in names}}
        for k, v in EXPECT[arm].items():
            got = res['arms'][arm]['all'] if k == 'all' else res['arms'][arm]['per_cell_mean'][k]
            assert abs(got - v) < 6e-5, (arm, k, got, v)
    p = os.path.join(HERE, '..', 'results', 'v9_dev_T4_T4b_per_cell.json')
    json.dump(res, open(p, 'w'), indent=1)
    for arm, a in res['arms'].items():
        print(arm, 'all %+.5f centred %+.5f' % (a['all'], a['centred_all']),
              ' '.join('%s %+.4f' % (c, v) for c, v in a['per_cell_mean'].items()))
    print('wrote', os.path.normpath(p))


if __name__ == '__main__':
    main()
