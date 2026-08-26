# -*- coding: utf-8 -*-
"""
Build TWO independent estimates of each plate's DMSO control, from disjoint halves of that plate's DMSO
wells, so the delta convention can be evaluated without the input and the target sharing noise.

The problem this solves. v9 trains on delta = X_trt - X_ctl and is HANDED X_ctl as an input. The control's
own measurement noise therefore enters the target with a minus sign and the input with a plus sign, so a
model can improve its delta score by cancelling noise rather than by predicting biology. Every published
model that reports a Pearson_deg on a plate-matched control has this property; nobody appears to measure it.

The first attempt at a control for this (`l3delta_ind` in ab_matched_control.py: target = X_trt - cellmean)
FAILS, and is kept only as a recorded negative. It replaces the coupling with a worse one -- the target then
contains (X_ctl - cellmean), the plate offset, which the matched arm can read straight off its own input.
It scored matched-minus-cellmean = +0.3771 on unseen compounds, which is the size of a leak, not an effect.

The correct construction is here: split a plate's DMSO wells in half.
    ctlA = median of half A        -> given to the model as input
    ctlB = median of half B        -> used to build the target
Both estimate the same plate state, so the biology and the batch offset are shared; only the measurement
noise is independent. Then

    matched_same  - matched_indep  =  the noise-cancellation component (an artefact of the convention)
    matched_indep - cellmean       =  what the plate-matched control genuinely contributes

P1 only (GSE92742). That is 65 % of the signatures and its GCTX is on disk; extending to P2 costs one
decompression and adds nothing to the argument.

    python model/level3/split_dmso_control.py
"""
import os, csv, json, time, argparse
from collections import defaultdict

import numpy as np
import h5py

csv.field_size_limit(10 ** 9)
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_level3_distil import PHASE, OUT, N_GENES, canonical_gene_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_dmso', type=int, default=6, help='a plate needs enough wells to halve')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    P = PHASE['P1']
    t0 = time.time()

    rows = {}
    for r in csv.DictReader(open(os.path.join(OUT, 'l3_meta.tsv'), encoding='utf-8'), delimiter='\t'):
        if r['phase'] == 'P1' and r['covered'] == '1':
            rows[r['sig_id']] = int(r['sig_row'])
    n_all = sum(1 for _ in open(os.path.join(OUT, 'l3_meta.tsv'), encoding='utf-8')) - 1
    print('P1 covered signatures: %d of %d rows' % (len(rows), n_all))

    distil = {}
    with open(P['sig'], encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['sig_id'] in rows:
                distil[r['sig_id']] = r['distil_id'].split('|')
    need = set().union(*distil.values())
    plate_of, dmso = {}, defaultdict(list)
    with open(P['inst'], encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['pert_type'] == 'ctl_vehicle':
                dmso[r[P['plate_col']]].append(r['inst_id'])
                plate_of[r['inst_id']] = r[P['plate_col']]
            elif r['inst_id'] in need:
                plate_of[r['inst_id']] = r[P['plate_col']]
    dmso = {p: v for p, v in dmso.items() if len(v) >= a.min_dmso}
    print('plates with >=%d DMSO wells: %d' % (a.min_dmso, len(dmso)))

    f = h5py.File(P['gctx'], 'r')
    mat = f['0/DATA/0/matrix']
    gi, _ = canonical_gene_rows(f)
    col_pos = {c.decode() if isinstance(c, bytes) else str(c): i
               for i, c in enumerate(f['0/META/COL/id'][:])}

    # targeted read: only the DMSO wells, not the whole 65 GB file
    plate_idx = {p: i for i, p in enumerate(sorted(dmso))}
    want, meta = [], []
    for p, wells in dmso.items():
        ws = [w for w in wells if w in col_pos]
        rng.shuffle(ws)
        half = len(ws) // 2
        for k, w in enumerate(ws):
            want.append(col_pos[w]); meta.append((plate_idx[p], 0 if k < half else 1))
    order = np.argsort(want)
    want = np.array(want)[order]
    meta = np.array(meta)[order]
    print('reading %d DMSO wells directly (%.1f GB of row reads)'
          % (len(want), len(want) * mat.shape[1] * 4 / 2 ** 30))
    buf = np.zeros((len(want), N_GENES), np.float32)
    for lo in range(0, len(want), 2000):
        hi = min(lo + 2000, len(want))
        buf[lo:hi] = mat[want[lo:hi], :][:, gi]
        if lo % 10000 == 0:
            print('  %5.1f%%  %4.0fs' % (100.0 * lo / len(want), time.time() - t0), flush=True)
    f.close()

    medA = np.full((len(plate_idx), N_GENES), np.nan, np.float32)
    medB = np.full((len(plate_idx), N_GENES), np.nan, np.float32)
    for p in range(len(plate_idx)):
        for h, M in [(0, medA), (1, medB)]:
            s = (meta[:, 0] == p) & (meta[:, 1] == h)
            if s.sum() >= 1:
                M[p] = np.median(buf[s], axis=0)
    del buf
    ok_plate = np.isfinite(medA).all(1) & np.isfinite(medB).all(1)
    print('plates with both halves: %d/%d' % (int(ok_plate.sum()), len(plate_idx)))

    A = np.full((n_all, N_GENES), np.nan, np.float32)
    B = np.full((n_all, N_GENES), np.nan, np.float32)
    nsig = 0
    for s, wells in distil.items():
        pl = [plate_of[w] for w in wells
              if w in col_pos and w in plate_of and plate_of[w] in plate_idx
              and ok_plate[plate_idx[plate_of[w]]]]
        if not pl:
            continue
        pi = [plate_idx[p] for p in pl]
        A[rows[s]] = medA[pi].mean(0)
        B[rows[s]] = medB[pi].mean(0)
        nsig += 1
    np.save(os.path.join(OUT, 'X_ctl_halfA_P1.npy'), A)
    np.save(os.path.join(OUT, 'X_ctl_halfB_P1.npy'), B)
    d = np.abs(A[np.isfinite(A).all(1)] - B[np.isfinite(B).all(1)])
    json.dump(dict(phase='P1', min_dmso=a.min_dmso, plates=int(ok_plate.sum()), signatures=nsig,
                   half_disagreement_mean=float(d.mean()), seconds=round(time.time() - t0, 1)),
              open(os.path.join(OUT, 'split_dmso_provenance.json'), 'w'), indent=2)
    print('\nsignatures with both control halves: %d' % nsig)
    print('mean |ctlA - ctlB| = %.4f  (the control\'s own measurement noise, per gene)' % float(d.mean()))
    print('-> %s' % OUT)


if __name__ == '__main__':
    main()
