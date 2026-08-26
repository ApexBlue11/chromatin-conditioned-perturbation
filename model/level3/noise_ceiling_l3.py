# -*- coding: utf-8 -*-
"""
GATE 2 for v9: what can a model trained on the Level-3 target possibly reach?

Method rule 4 says compute the noise ceiling before chasing a gap, and v9 changes the target, so v7's
0.4549 / 0.4985 / 0.4825 stop being the reference point. A model cannot predict a target more accurately
than that target agrees with itself.

RESULTS 25 measured this once, on 400 conditions grouped by (cell, pert, dose, time), and concluded Level 5
is the better-denoised target (Level-3 3+3 split-half 0.1441, top quartile 0.2429, versus Level-5 MODZ
0.127 all / 0.509-0.619 reproducible). That conclusion is load-bearing -- it is why v9's move to Level 3 is
a COMPARABILITY decision and not an accuracy one -- so it is re-measured here on 56,811 signatures with the
exact well sets MODZ itself used, rather than on 400 key-grouped conditions.

Construction, kept identical to RESULTS 25 so the numbers are comparable:
  * a well's delta = its expression MINUS the median of the DMSO wells ON THE SAME PLATE;
  * the two halves are drawn from DISJOINT PLATES, because same-plate wells share a control vector and
    would inflate agreement;
  * halves are replicate-AVERAGED before correlating, since a MODZ signature is itself a replicate average.

Reported for BOTH conventions, because the difference between them is the whole reason the field's
absolute numbers look high: the absolute profile is dominated by the cell's baseline, which is trivially
reproducible, while the delta is the part that carries the perturbation.

    python model/level3/noise_ceiling_l3.py
"""
import os, csv, json, argparse, time
from collections import defaultdict

import numpy as np
import h5py

csv.field_size_limit(10 ** 9)
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_level3_distil import PHASE, ROOT, OUT, SIGS, canonical_gene_rows, N_GENES

RES = os.path.join(ROOT, 'model', 'results')


def rowwise_pearson(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def spearman_brown(r, k):
    """Reliability of a k-well average given the reliability of a (k/2)-well average."""
    return np.clip(2 * r / (1 + r), -1, 1) if k is None else np.clip(k * r / (1 + (k - 1) * r), -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_wells', type=int, default=4)
    ap.add_argument('--block', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t00 = time.time()

    meta = {}
    for r in csv.DictReader(open(os.path.join(OUT, 'l3_meta.tsv'), encoding='utf-8'), delimiter='\t'):
        if r['covered'] == '1' and int(r['n_wells_used']) >= a.min_wells and int(r['n_plates']) >= 2:
            meta[r['sig_id']] = (int(r['sig_row']), r['phase'], int(r['y_row']))
    print('signatures with >=%d wells on >=2 plates: %d' % (a.min_wells, len(meta)))

    sel = sorted(meta.values())
    pos = {sr: i for i, (sr, _, _) in enumerate(sel)}
    n = len(sel)
    sumA = np.zeros((n, N_GENES), np.float32); sumB = np.zeros((n, N_GENES), np.float32)
    ctlA = np.zeros((n, N_GENES), np.float32); ctlB = np.zeros((n, N_GENES), np.float32)
    cntA = np.zeros(n, np.float32); cntB = np.zeros(n, np.float32)
    nwell = np.zeros(n, np.int32)

    for ph in ['P1', 'P2']:
        P = PHASE[ph]
        want = {s: v for s, v in meta.items() if v[1] == ph}
        distil = {}
        with open(P['sig'], encoding='utf-8') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                if r['sig_id'] in want:
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
        dmso = {p: v for p, v in dmso.items() if len(v) >= 3}

        f = h5py.File(P['gctx'], 'r')
        mat = f['0/DATA/0/matrix']
        gi, _ = canonical_gene_rows(f)
        col_pos = {c.decode() if isinstance(c, bytes) else str(c): i
                   for i, c in enumerate(f['0/META/COL/id'][:])}

        # ---- assign whole PLATES to half A / half B so the halves never share a control vector ----
        w_row, w_sig, w_half, w_plate = [], [], [], []
        plates_used, kept = {}, 0
        for s, wells in distil.items():
            ok = [w for w in wells if w in col_pos and w in plate_of and plate_of[w] in dmso]
            by_plate = defaultdict(list)
            for w in ok:
                by_plate[plate_of[w]].append(w)
            if len(by_plate) < 2:
                continue
            pl = sorted(by_plate)
            rng.shuffle(pl)
            half = {p: (0 if i % 2 == 0 else 1) for i, p in enumerate(pl)}
            i = pos[want[s][0]]
            nwell[i] = len(ok)
            kept += 1
            for p, ws in by_plate.items():
                if p not in plates_used:
                    plates_used[p] = len(plates_used)
                for w in ws:
                    w_row.append(col_pos[w]); w_sig.append(i)
                    w_half.append(half[p]); w_plate.append(plates_used[p])
        print('  [%s] usable signatures %d, wells %d, plates %d' % (ph, kept, len(w_row), len(plates_used)))

        w_row = np.array(w_row, np.int64); w_sig = np.array(w_sig, np.int64)
        w_half = np.array(w_half, np.int64); w_plate = np.array(w_plate, np.int64)
        d_row, d_slot, d_plate = [], {}, []
        for p, pi in plates_used.items():
            for w in dmso[p]:
                if w in col_pos:
                    d_slot[col_pos[w]] = len(d_row)
                    d_row.append(col_pos[w]); d_plate.append(pi)
        d_row = np.array(d_row, np.int64); d_plate = np.array(d_plate, np.int64)
        dbuf = np.zeros((len(d_row), N_GENES), np.float32)

        mask = np.zeros(mat.shape[0], np.bool_)
        mask[w_row] = True; mask[d_row] = True
        t0 = time.time()
        for start in range(0, mat.shape[0], a.block):
            stop = min(start + a.block, mat.shape[0])
            loc = np.nonzero(mask[start:stop])[0]
            if len(loc) == 0:
                continue
            blk = mat[start:stop, :][loc][:, gi]
            dm = (d_row >= start) & (d_row < stop)
            if dm.any():
                dbuf[np.array([d_slot[int(v)] for v in d_row[dm]])] = blk[np.searchsorted(loc, d_row[dm] - start)]
            wm = (w_row >= start) & (w_row < stop)
            if wm.any():
                take = blk[np.searchsorted(loc, w_row[wm] - start)]
                hA = w_half[wm] == 0
                np.add.at(sumA, w_sig[wm][hA], take[hA])
                np.add.at(sumB, w_sig[wm][~hA], take[~hA])
                np.add.at(cntA, w_sig[wm][hA], 1.0)
                np.add.at(cntB, w_sig[wm][~hA], 1.0)
            if (start // a.block) % 40 == 0:
                print('    scan %5.1f%%  %5.0fs' % (100.0 * stop / mat.shape[0], time.time() - t0),
                      flush=True)
        f.close()

        med = np.zeros((len(plates_used), N_GENES), np.float32)
        for p in range(len(plates_used)):
            s_ = d_plate == p
            med[p] = np.median(dbuf[s_], axis=0) if s_.any() else np.nan
        del dbuf
        for lo in range(0, len(w_sig), 50000):
            hi = min(lo + 50000, len(w_sig))
            sl = slice(lo, hi)
            hA = w_half[sl] == 0
            np.add.at(ctlA, w_sig[sl][hA], med[w_plate[sl][hA]])
            np.add.at(ctlB, w_sig[sl][~hA], med[w_plate[sl][~hA]])

    ok = (cntA > 0) & (cntB > 0)
    print('\nsignatures with both halves populated: %d/%d' % (int(ok.sum()), n))
    absA = sumA[ok] / cntA[ok][:, None]; absB = sumB[ok] / cntB[ok][:, None]
    dA = absA - ctlA[ok] / cntA[ok][:, None]
    dB = absB - ctlB[ok] / cntB[ok][:, None]
    r_delta = rowwise_pearson(dA.astype(np.float64), dB.astype(np.float64))
    r_abs = rowwise_pearson(absA.astype(np.float64), absB.astype(np.float64))
    strength = np.abs((dA + dB) / 2).mean(1)
    kw = nwell[ok]

    # the project's own reproducible stratum, defined on the Level-5 target, for a like-for-like read
    l5s = np.load(os.path.join(ROOT, 'phase2_assembly', 'outputs', 'sig_strength.npy'))
    yrow = np.array([sel[i][2] for i in np.flatnonzero(ok)])
    l5_strength = l5s[yrow]

    out = {'n_signatures': int(ok.sum()), 'min_wells': a.min_wells,
           'median_wells': int(np.median(kw)), 'seconds': round(time.time() - t00, 1)}
    print('\n%-46s %8s %8s' % ('stratum', 'delta', 'absolute'))
    print('-' * 64)

    def line(tag, m, store):
        if m.sum() < 50:
            return
        rd, ra = np.nanmedian(r_delta[m]), np.nanmedian(r_abs[m])
        sb = np.nanmedian(spearman_brown(r_delta[m], int(np.median(kw[m]))))
        print('%-46s %8.4f %8.4f   (n=%d, SB->%d wells %.4f)'
              % (tag, rd, ra, m.sum(), int(np.median(kw[m])), sb))
        store[tag] = dict(n=int(m.sum()), delta=round(float(rd), 4), absolute=round(float(ra), 4),
                          spearman_brown_full=round(float(sb), 4))

    st = {}
    line('all signatures (half vs half)', np.ones_like(strength, bool), st)
    line('3+3 averaged (>=6 wells) [matches RESULTS 25]', kw >= 6, st)
    q = np.percentile(strength, [75, 85])
    line('top strength quartile (Level-3 delta)', strength >= q[0], st)
    line('top 15%% strength (Level-3 delta)', strength >= q[1], st)
    line('Level-5 reproducible stratum (strength >= 1.0)', l5_strength >= 1.0, st)
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        b = np.percentile(strength, [lo, hi])
        line('  delta-strength pct %3d-%3d' % (lo, hi), (strength >= b[0]) & (strength < b[1] + 1e-12), st)
    out['strata'] = st
    out['reference_RESULTS_25'] = {'level3_3+3_splithalf': 0.1441, 'level3_top_quartile': 0.2429,
                                   'level5_modz_all': 0.127, 'level5_modz_reproducible': [0.509, 0.619]}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'v9_noise_ceiling_l3.json'), 'w'), indent=2)
    print('\nwrote model/results/v9_noise_ceiling_l3.json')


if __name__ == '__main__':
    main()
