# -*- coding: utf-8 -*-
"""
GATE 3 for v9, and the answer to "should CCLE go?".

The original A/B (model/level3/test_matched_control.py) measured +0.0314 / +0.0435 / +0.0372 for the
plate-matched control over the CCLE proxy, and that "both ~ matched alone" -- from which the handoff
concluded CCLE is SUBSUMED and should be dropped. Two reasons to redo it here:

  * that run used the key join, in which 4,419 P2 signatures were matched to controls from GSE92742 --
    a different experiment entirely -- so 3.2 % of its rows had a control that was not plate-matched at all;
  * "subsumed" is not the same claim as "harmful". Redundant inputs cost parameters and can leak cell
    identity on a cold-cell split; they are not automatically bad. The decision needs the number.

Four baseline arms, so the question decomposes instead of being answered by assertion:

    ccle        CCLE proxy               different assay, different platform, per-CELL constant
    cellmean    mean L1000 DMSO of that   same platform as the target, per-CELL constant
                cell across its plates    -> ccle vs cellmean isolates the PLATFORM effect
    matched     this signature's own      per-SIGNATURE, carries plate/batch state
                plate DMSO median         -> cellmean vs matched isolates the PLATE-SPECIFIC effect
    both        ccle + matched            -> does CCLE add anything on top of matched?

Three targets, because v9 changes the target and each convention has to be reported:

    l5       Level-5 MODZ z-score  (what every previous number in this project used)
    l3delta  X_trt - X_ctl         (what v9 trains on, and what XPert reports as Pearson_deg)
    l3abs    X_trt                 (the absolute convention, reported for comparability only)

A WARNING THE l3delta COLUMN NEEDS. The target is trt - ctl and one arm feeds ctl as an input, so a model
can cancel the control's own measurement noise rather than predict biology. `cellmean` is the control for
exactly that: it carries the same cell-level information with independent plate noise. If matched beats
cellmean far more on l3delta than on l5, the extra is noise cancellation, not signal.

Closed-form ridge, so there is no seed variance to report.

    python model/level3/ab_matched_control.py
"""
import os, csv, json, argparse, sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MODEL, 'v6'))
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

from config_v6 import V6DataConfig
from data import LincsDataset, build_splits
from eval_v6 import metrics
from extract_level3_distil import ROOT, OUT as L3


def gram(X, chunk=8192):
    """X^T X and the float64 accumulation, without ever materialising X in float64."""
    D = X.shape[1]
    G = np.zeros((D, D), np.float64)
    for lo in range(0, len(X), chunk):
        b = np.asarray(X[lo:lo + chunk], np.float64)
        G += b.T @ b
    return G


def solve(X, Y, lam, chunk=8192):
    D = X.shape[1]
    G = gram(X, chunk)
    G[np.diag_indices(D)] += lam
    G[-1, -1] -= lam                                     # never penalise the intercept
    B = np.zeros((D, Y.shape[1]), np.float64)
    for lo in range(0, len(X), chunk):
        B += np.asarray(X[lo:lo + chunk], np.float64).T @ np.asarray(Y[lo:lo + chunk], np.float64)
    return np.linalg.solve(G, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambdas', default='1e3,1e4,1e5')
    ap.add_argument('--max_train', type=int, default=40000)
    ap.add_argument('--max_eval', type=int, default=6000)
    ap.add_argument('--arms', default='ccle,cellmean,matched,both')
    ap.add_argument('--targets', default='l5,l3delta,l3delta_ind,l3abs')
    ap.add_argument('--split_dmso', action='store_true',
                    help='THE noise-cancellation test. Restricts to P1 rows that have two independent '
                         'half-plate DMSO medians and adds arms ctlA/ctlB plus target l3delta_B: the '
                         'target is built from half B, ctlB is the coupled input, ctlA the uncoupled one.')
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    dc = V6DataConfig()
    ds = LincsDataset(dc)
    sp = build_splits(ds, dc)

    # --- bridge: the dataset drops signatures with no resolvable cell/drug, so its positions are not
    # the Level-3 arrays' positions. Both carry y_row, which is unique per signature. ---
    yrow_l3 = np.load(os.path.join(L3, 'l3_yrow.npy'))
    cov = np.load(os.path.join(L3, 'l3_covered.npy'))
    y2pos = {int(y): i for i, y in enumerate(yrow_l3)}
    ds_to_l3 = np.array([y2pos.get(int(y), -1) for y in ds.y_row])
    have = np.flatnonzero((ds_to_l3 >= 0) & cov[np.clip(ds_to_l3, 0, None)])
    print('dataset rows: %d | with a Level-3 substrate row: %d (%.1f%%)'
          % (len(ds.y_row), len(have), 100.0 * len(have) / len(ds.y_row)))

    Xtrt = np.load(os.path.join(L3, 'X_trt_l3.npy'), mmap_mode='r')
    Xctl = np.load(os.path.join(L3, 'X_ctl_l3.npy'), mmap_mode='r')
    ctlA = ctlB = None
    if a.split_dmso:
        ctlA = np.load(os.path.join(L3, 'X_ctl_halfA_P1.npy'), mmap_mode='r')
        ctlB = np.load(os.path.join(L3, 'X_ctl_halfB_P1.npy'), mmap_mode='r')
        fin = np.isfinite(np.asarray(ctlA[ds_to_l3[have]][:, 0])) &               np.isfinite(np.asarray(ctlB[ds_to_l3[have]][:, 0]))
        have = have[fin]
        print('restricted to P1 rows with two independent control halves: %d' % len(have))

    # per-cell mean of the matched controls: same platform as the target, but no plate-specific noise
    cellmean = np.zeros((ds.Xb.shape[0], 978), np.float32)
    cnt = np.zeros(ds.Xb.shape[0], np.float32)
    for lo in range(0, len(have), 20000):
        idx = have[lo:lo + 20000]
        rows = ds_to_l3[idx]
        blk = np.asarray(Xctl[np.sort(rows)], np.float32)
        order = np.argsort(rows)
        np.add.at(cellmean, ds.cell_row[idx][order], blk)
        np.add.at(cnt, ds.cell_row[idx][order], 1.0)
    cellmean /= np.maximum(cnt, 1)[:, None]
    print('cells with a matched-control mean: %d/%d' % (int((cnt > 0).sum()), len(cnt)))

    fp = ds.u_feats[:, -2048:]
    desc = ds.u_feats[:, -2068:-2048]

    def features(idx, arm):
        d, c = ds.drug_row[idx], ds.cell_row[idx]
        blocks = [fp[d], desc[d], ds.cell_ctx[c], ds.dose_n[idx][:, None], ds.time_n[idx][:, None]]
        if arm in ('ccle', 'both'):
            blocks.append(ds.Xb[c])
        if arm == 'cellmean':
            blocks.append(cellmean[c])
        if arm in ('matched', 'both'):
            blocks.append(np.asarray(Xctl[ds_to_l3[idx]], np.float32))
        if arm == 'ctlA':
            blocks.append(np.asarray(ctlA[ds_to_l3[idx]], np.float32))
        if arm == 'ctlB':
            blocks.append(np.asarray(ctlB[ds_to_l3[idx]], np.float32))
        X = np.concatenate(blocks, 1).astype(np.float32)
        return np.concatenate([X, np.ones((len(idx), 1), np.float32)], 1)

    def target(idx, tgt):
        rows = ds_to_l3[idx]
        if tgt == 'l5':
            return np.asarray(ds.Y[ds.y_row[idx]], np.float32)
        if tgt == 'l3abs':
            return np.asarray(Xtrt[rows], np.float32)
        if tgt == 'l3delta_B':
            return np.asarray(Xtrt[rows], np.float32) - np.asarray(ctlB[rows], np.float32)
        if tgt == 'l3delta_ind':
            # delta against a control the `matched` arm is NOT handed. trt - ctl shares ctl's measurement
            # noise with the input, so a model can cancel noise instead of predicting biology; subtracting
            # the per-cell control mean instead breaks that coupling while keeping the same quantity's
            # biology. The matched-vs-cellmean gap on THIS target is the part that is not noise cancellation.
            return np.asarray(Xtrt[rows], np.float32) - cellmean[ds.cell_row[idx]]
        return np.asarray(Xtrt[rows], np.float32) - np.asarray(Xctl[rows], np.float32)

    tr_all = np.intersect1d(sp['train'], have)
    tr = np.sort(rng.choice(tr_all, min(a.max_train, len(tr_all)), replace=False))
    va = np.intersect1d(sp['val'], have)
    va = np.sort(rng.choice(va, min(8000, len(va)), replace=False))
    print('train %d | val %d' % (len(tr), len(va)))

    # The evaluation rows are chosen ONCE and shared by every arm and target. Re-drawing them per arm
    # would put a sampling difference inside a comparison whose whole purpose is that only the baseline
    # input changes.
    eval_idx = {}
    for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                      ('unseen_both', 'test_coldboth')]:
        idx = np.intersect1d(sp[key], have)
        idx = idx[ds.strength[idx] >= dc.eval_min_strength]
        if len(idx) < 300:
            eval_idx[name] = None
            continue
        if len(idx) > a.max_eval:
            idx = np.sort(rng.choice(idx, a.max_eval, replace=False))
        eval_idx[name] = idx
    print('eval rows: ' + ', '.join('%s %s' % (k, 0 if v is None else len(v)) for k, v in eval_idx.items()))

    res = defaultdict(dict)
    tgts = a.targets.split(',')
    Ytr = {t: target(tr, t) for t in tgts}
    Yva = {t: target(va, t) for t in tgts}
    for arm in a.arms.split(','):
        # the Gram depends only on the feature arm, so it is built ONCE and reused across every target
        # and every lambda -- 4 Grams instead of 36
        Xtr = features(tr, arm)
        mu, sd = Xtr[:, :-1].mean(0), Xtr[:, :-1].std(0) + 1e-6
        Xtr[:, :-1] = (Xtr[:, :-1] - mu) / sd
        Xva = features(va, arm); Xva[:, :-1] = (Xva[:, :-1] - mu) / sd
        G0 = gram(Xtr)
        D = G0.shape[0]
        for tgt in tgts:
            B = np.zeros((D, Ytr[tgt].shape[1]), np.float64)
            for lo in range(0, len(Xtr), 8192):
                B += np.asarray(Xtr[lo:lo + 8192], np.float64).T @ np.asarray(Ytr[tgt][lo:lo + 8192],
                                                                              np.float64)
            best = None
            for lam in [float(x) for x in a.lambdas.split(',')]:
                G = G0.copy()
                G[np.diag_indices(D)] += lam
                G[-1, -1] -= lam                          # never penalise the intercept
                W = np.linalg.solve(G, B)
                mse = float(((Xva @ W - Yva[tgt]) ** 2).mean())
                if best is None or mse < best[0]:
                    best = (mse, lam, W)
            _, lam, W = best
            rec = {'lambda': lam, 'n_features': int(Xtr.shape[1] - 1), 'n_train': int(len(tr))}
            for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                              ('unseen_both', 'test_coldboth')]:
                idx = eval_idx[name]
                if idx is None:
                    continue
                Xe = features(idx, arm); Xe[:, :-1] = (Xe[:, :-1] - mu) / sd
                m = metrics(torch.from_numpy((Xe @ W).astype(np.float32)),
                            torch.from_numpy(target(idx, tgt)))
                rec[name] = {'n': int(len(idx)), 'pearson': round(float(m['pearson_median']), 4),
                             'r2': round(float(m['r2_overall']), 4)}
            res[tgt][arm] = rec
            print('  %-8s %-9s lam=%.0e  ' % (tgt, arm, lam) + '  '.join(
                '%s %.4f' % (k[7:], rec[k]['pearson']) for k in
                ['unseen_cell', 'unseen_compound', 'unseen_both'] if k in rec), flush=True)

    print('\n' + '=' * 108)
    print('BASELINE-INPUT A/B  (pearson on the reproducible stratum, closed-form so no seed variance)')
    print('=' * 108)
    arms = a.arms.split(',')
    PAIRS = [('matched', 'ccle'), ('matched', 'cellmean'), ('both', 'matched'),
             ('ctlB', 'ctlA'), ('ctlA', 'cellmean'), ('cellmean', 'ccle')]
    for tgt in a.targets.split(','):
        pairs = [(x, y) for x, y in PAIRS if x in arms and y in arms]
        print('\n--- target: %s ---' % tgt)
        print('  %-16s' % 'split' + ''.join('%12s' % x for x in arms) +
              ''.join('%16s' % ('%s-%s' % (x, y)) for x, y in pairs))
        for name in ['unseen_cell', 'unseen_compound', 'unseen_both']:
            if not all(name in res[tgt].get(x, {}) for x in arms):
                continue
            v = {x: res[tgt][x][name]['pearson'] for x in arms}
            print('  %-16s' % name + ''.join('%12.4f' % v[x] for x in arms) +
                  ''.join('%+16.4f' % (v[x] - v[y]) for x, y in pairs))

    # merge rather than overwrite, so arms and targets from separate invocations accumulate in one record
    path = os.path.join(ROOT, 'model', 'results', 'v9_baseline_ab.json')
    prev = json.load(open(path)) if os.path.exists(path) else {}
    for tgt, d in res.items():
        prev.setdefault(tgt, {}).update(d)
    json.dump(prev, open(path, 'w'), indent=2)
    print('\nwrote model/results/v9_baseline_ab.json')


if __name__ == '__main__':
    main()
