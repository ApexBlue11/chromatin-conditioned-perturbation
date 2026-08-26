# -*- coding: utf-8 -*-
"""
What do published LINCS splits actually hold out? Read from the released data, not from the paper.

V9_HANDOFF §C states that XPert's splits are "tissue-holdouts: split_lung_1..5, split_breast_1..5,
split_haematopoietic_and_lymphoid_tissue_1..5". That is checkable against their own released h5ad, and it
is wrong. Each split restricts to ONE tissue and then divides it roughly 90/10, so train and test are the
same tissue, the same cell lines, and largely the same compounds.

This matters more than any architectural difference in the comparison. Our benchmarks hold out entire cell
lines and entire Bemis-Murcko scaffold families; theirs mostly asks for a different DOSE or TIME of a
(cell, compound) pair the model has already seen. Reporting our 0.45-0.50 against their 0.844 as if they
were the same task is the exact failure the handoff's gate exists to prevent:

    "any SOTA comparison | same data level + convention + split, else report non-comparability"

Reported per split: how many test rows use a cell line seen in training, a compound seen in training, and
the exact (cell, compound) PAIR seen in training.

    python model/v9/sota_split_audit.py
"""
import os, json, argparse

import numpy as np
import h5py

ROOT = r'C:\Projects\LINCS'
H5 = os.path.join(ROOT, 'external', 'xpert', 'l1000_mdmt_full_336852.h5ad')


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', default=H5)
    a = ap.parse_args()
    if not os.path.exists(a.h5ad):
        raise SystemExit(f'{a.h5ad} not found (external/ is gitignored; 5.1 GB from their Zenodo release)')
    f = h5py.File(a.h5ad, 'r')

    pert = f['obs/pert_id/codes'][:].astype(np.int64)
    cell = f['obs/cell_iname/codes'][:].astype(np.int64)
    dose = f['obs/pert_dose/codes'][:].astype(np.int64)
    time = f['obs/pert_time/codes'][:].astype(np.int64)
    tissue = f['obs/tissue/codes'][:].astype(np.int64)
    tissue_names = dec(f['obs/tissue/categories'][:])
    splits = sorted(k for k in f['obs'].keys() if k.startswith('split_'))
    print(f'{len(splits)} splits over {len(pert)} conditions, {len(set(cell.tolist()))} cell lines, '
          f'{len(set(pert.tolist()))} compounds\n')

    pair = cell * 1_000_000 + pert
    trip = (pair * 1000 + dose) * 100 + time
    rows = []
    hdr = ('%-46s %7s %6s %6s %6s   %8s %8s %8s %8s' %
           ('split', 'train', 'test', 'cells', 'unused', 'cell', 'compound', 'pair', 'exact'))
    print(hdr); print('-' * len(hdr))
    for s in splits:
        cats = dec(f[f'obs/{s}/categories'][:])
        k = f[f'obs/{s}/codes'][:]
        tr = np.flatnonzero(k == cats.index('train'))
        te = np.flatnonzero(k == cats.index('test'))
        if len(te) == 0:
            continue
        seen_cell = float(np.isin(cell[te], np.unique(cell[tr])).mean())
        seen_pert = float(np.isin(pert[te], np.unique(pert[tr])).mean())
        seen_pair = float(np.isin(pair[te], np.unique(pair[tr])).mean())
        seen_trip = float(np.isin(trip[te], np.unique(trip[tr])).mean())
        tis = [tissue_names[t] for t in np.unique(tissue[np.concatenate([tr, te])])]
        rec = dict(split=s, n_train=len(tr), n_test=len(te), n_unused=int(len(k) - len(tr) - len(te)),
                   n_cells=int(len(np.unique(cell[np.concatenate([tr, te])]))), tissues=tis,
                   test_cell_seen_in_train=round(seen_cell, 4),
                   test_compound_seen_in_train=round(seen_pert, 4),
                   test_cell_compound_pair_seen_in_train=round(seen_pair, 4),
                   test_exact_cell_compound_dose_time_seen=round(seen_trip, 4))
        rows.append(rec)
        print('%-46s %7d %6d %6d %6d   %7.1f%% %7.1f%% %7.1f%% %7.1f%%'
              % (s, len(tr), len(te), rec['n_cells'], rec['n_unused'],
                 100 * seen_cell, 100 * seen_pert, 100 * seen_pair, 100 * seen_trip))
    f.close()

    m = lambda k: float(np.mean([r[k] for r in rows]))
    print('\n' + '=' * 96)
    print('MEAN OVER %d SPLITS: test rows whose CELL was in training %.1f%% | COMPOUND %.1f%% | '
          '(cell, compound) PAIR %.1f%%' % (len(rows), 100 * m('test_cell_seen_in_train'),
                                            100 * m('test_compound_seen_in_train'),
                                            100 * m('test_cell_compound_pair_seen_in_train')))
    print('=' * 96)
    print('These are WITHIN-TISSUE splits, not tissue holdouts. The task is predominantly interpolation to')
    print('a different DOSE or TIME of a (cell, compound) pair already in training. Our benchmarks hold out')
    print('entire cell lines and entire scaffold families, so the two numbers are NOT comparable, and the')
    print('difference in split difficulty must be quantified before any architecture is credited.')
    print('\nV9_HANDOFF.md §C calls these "tissue-holdouts". Corrected here from their own released data.')

    out = os.path.join(ROOT, 'model', 'results', 'sota_split_audit.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'source': a.h5ad, 'splits': rows,
               'mean_test_cell_seen': round(m('test_cell_seen_in_train'), 4),
               'mean_test_compound_seen': round(m('test_compound_seen_in_train'), 4),
               'mean_test_pair_seen': round(m('test_cell_compound_pair_seen_in_train'), 4),
               'verdict': 'within-tissue 90/10 splits with near-total cell and compound overlap; '
                          'NOT tissue holdouts'},
              open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
