# -*- coding: utf-8 -*-
"""Build the `all_drugs_unimol_arr.npy` that XPert's loader expects, from the compact npz we hold.

WHY THIS EXISTS [RESULTS 68.6]
`configs/config_l1000.yaml` points `drug_unimol_path` at `processed_data/all_drugs_unimol_arr.npy`, which
is NOT in the released assets. What we do hold is `processed_data/unimol_mdmt_1970.npz` with two arrays,
`idx` (1970,) and `feat` (1970, 122, 514). Their loader does `self.drug_feat[pert_idx]`
(`datasets/MyDataset.py:146`), i.e. it indexes by the h5ad's `pert_idx` directly, so it needs a dense
array whose row `k` is the features for `pert_idx == k`.

WHY IT IS BUILT AT RUNTIME RATHER THAN UPLOADED
The dense array is (8950, 122, 514) float32 = 2.24 GB, against 494 MB for the compact npz it is built
from. Uploading the expansion would cost 2.24 GB of transfer to carry no information. It is derived, so it
is rebuilt where it is used.

WHY THE ZERO ROWS ARE SAFE
Verified against `l1000_mdmt_68830_subset.h5ad`: the subset's `pert_idx` column has exactly 1970 unique
values in [1, 8949], and ALL 1970 are present in the npz's `idx` -- zero missing. So every row the loader
can reach is populated, and the unpopulated rows are unreachable. The guard below re-checks that at build
time against the actual h5ad rather than trusting this note, because a stale or different subset file
would silently hand the model all-zero drug features -- which is retraction-class 7 [RESULTS 32]: a dead
input that trains to convergence and reports plausible metrics.
"""
import argparse
import os
import sys

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='unimol_mdmt_1970.npz')
    ap.add_argument('--h5ad', required=True, help='l1000_mdmt_68830_subset.h5ad, for the reachability guard')
    ap.add_argument('--out', default=None, help='where to write all_drugs_unimol_arr.npy')
    ap.add_argument('--verify_only', action='store_true',
                    help='run every guard and report, but do not write the 2.24 GB array')
    a = ap.parse_args()

    z = np.load(a.npz)
    for k in ('idx', 'feat'):
        if k not in z.files:
            raise SystemExit('FATAL: %s has no %r array; got %r' % (a.npz, k, z.files))
    idx = z['idx'].astype(np.int64)
    feat = z['feat']
    print('npz: idx %s in [%d, %d], feat %s %s' % (idx.shape, idx.min(), idx.max(), feat.shape, feat.dtype),
          flush=True)

    if feat.ndim != 3 or feat.shape[0] != idx.shape[0]:
        raise SystemExit('FATAL: feat %s does not line up with idx %s' % (feat.shape, idx.shape))
    if len(np.unique(idx)) != len(idx):
        raise SystemExit('FATAL: idx contains duplicates; the mapping would be ambiguous.')

    # GUARD: every pert_idx the loader can reach must be populated.
    with h5py.File(a.h5ad, 'r') as f:
        d = f['obs']['pert_idx']
        if isinstance(d, h5py.Group):           # categorical encoding
            codes = d['codes'][:]
            cats = d['categories'][:]
            reachable = np.unique(cats[codes].astype(str).astype(np.int64))
        else:
            reachable = np.unique(d[:].astype(np.int64))
    missing = reachable[~np.isin(reachable, idx)]
    print('h5ad: %d unique pert_idx in [%d, %d]; %d not covered by the npz'
          % (len(reachable), reachable.min(), reachable.max(), len(missing)), flush=True)
    if len(missing):
        raise SystemExit('FATAL: %d reachable pert_idx have no unimol features: %s. Training would feed '
                         'all-zero drug features for those rows.' % (len(missing), missing[:20]))

    # GUARD: the features must discriminate. A file of zeros would pass every shape check.
    valid_mask = feat[:, :, 0]
    atom_feat = feat[:, :, 2:]
    n_valid = valid_mask.sum(axis=1)
    if not np.isfinite(atom_feat).all():
        raise SystemExit('FATAL: non-finite values in the atom features.')
    if float(np.abs(atom_feat).max()) == 0.0:
        raise SystemExit('FATAL: every atom feature is zero. The input is dead.')
    pad = valid_mask == 0
    print('valid atoms/drug: min %d max %d mean %.1f of %d slots | padded slots %.1f%% | '
          'padded features all zero: %s'
          % (n_valid.min(), n_valid.max(), n_valid.mean(), feat.shape[1], 100 * pad.mean(),
             bool(np.all(atom_feat[pad] == 0))), flush=True)

    n_rows = int(idx.max()) + 1
    nbytes = n_rows * feat.shape[1] * feat.shape[2] * 4
    print('dense array would be (%d, %d, %d) float32 = %.2f GB'
          % (n_rows, feat.shape[1], feat.shape[2], nbytes / 1e9), flush=True)

    if a.verify_only:
        print('VERIFY_ONLY: all guards passed, nothing written.', flush=True)
        return

    if not a.out:
        raise SystemExit('FATAL: --out is required unless --verify_only.')
    arr = np.zeros((n_rows, feat.shape[1], feat.shape[2]), dtype=np.float32)
    arr[idx] = feat

    # Round-trip: a sample of source rows must be byte-identical at their destination index.
    rng = np.random.default_rng(0)
    probe = rng.choice(len(idx), size=min(64, len(idx)), replace=False)
    for i in probe:
        if not np.array_equal(arr[idx[i]], feat[i]):
            raise SystemExit('FATAL: round-trip failed for npz row %d -> pert_idx %d.' % (i, idx[i]))
    print('round-trip verified on %d sampled drugs' % len(probe), flush=True)

    np.save(a.out, arr)
    print('wrote %s (%.2f GB)' % (a.out, os.path.getsize(a.out) / 1e9), flush=True)


if __name__ == '__main__':
    main()
