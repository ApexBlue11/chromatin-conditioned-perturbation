"""Outcome-free check (before T-C): the substrate's real plates against v9's control-identity plate proxy on the benchmark test
rows. Result 2026-10-19: identical partitions on cold_cell_1 (5,075 plates) and cold_drug_1 (4,403), identical controls.

    python model/v10c/plate_defs_check.py > model/results/v10c/plate_defs_check.txt
"""
import sys
import numpy as np
sys.path.insert(0, 'model/v9')
from audit99_target_sources import control_ids

z = np.load('external/v10c_full/v10c_full_substrate.npz', allow_pickle=True)
b = np.load('external/xpert_split_bundle/xpert_mdmt_splits.npz', allow_pickle=True)
print('substrate keys', [k for k in z.files if not k.startswith('meta_')][:20])
print('bundle keys', b.files[:30])
proxy_all = control_ids(b['X_ctl'])
for split in ('cold_cell_1', 'cold_drug_1'):
    te = np.flatnonzero(z['split_' + split] == 'test')
    br = z['bench_row'][te]
    ok = br >= 0
    te, br = te[ok], br[ok].astype(int)
    real = z['plate_id'][te]
    # the bundle row of each benchmark row
    pos = {int(r): i for i, r in enumerate(b['row_index'])}
    bidx = np.array([pos.get(int(r), -1) for r in br])
    print(split, 'test rows', len(te), 'mapped to bundle', int((bidx >= 0).sum()))
    m = bidx >= 0
    proxy = proxy_all[bidx[m]]
    real = real[m]
    # partition agreement
    pairs = np.unique(np.stack([real, proxy], 1), axis=0)
    n_real, n_proxy = len(np.unique(real)), len(np.unique(proxy))
    print(f'  real plates {n_real}, proxy plates {n_proxy}, distinct (real, proxy) pairs {len(pairs)}')
    # how many proxy groups span >1 real plate, and vice versa
    from collections import defaultdict
    pr, rp = defaultdict(set), defaultdict(set)
    for r_, p_ in pairs:
        pr[p_].add(r_); rp[r_].add(p_)
    print('  proxy groups spanning >1 real plate:', sum(len(v) > 1 for v in pr.values()),
          '| real plates split into >1 proxy group:', sum(len(v) > 1 for v in rp.values()))
    # control vectors: does the substrate ctl equal the bundle X_ctl on these rows?
    d = np.abs(z['ctl'][z['ctl_id'][te[m]]].astype(np.float64) - b['X_ctl'][bidx[m]].astype(np.float64)).max()
    print('  max |substrate ctl - bundle X_ctl|', float(d))
