# -*- coding: utf-8 -*-
"""
THE TEST THE WHOLE v8 MIGRATION RESTS ON.

The gate on a Level-3 TARGET failed (gate_modz.py): MODZ-weighted aggregation of Level-3 deltas scores
0.157 split-half against a plain mean's 0.164, and both are far below Level-5 MODZ's 0.509-0.619. So we
keep the Level-5 z-score as the target -- it is the better-denoised one and always was [RESULTS 25].

What Level 3 is actually FOR is the INPUT. Today the model's baseline is CCLE `X_base`: a different assay,
on a different platform, from a different experiment, years apart from the L1000 plate. Level 3 supplies the
DMSO wells from the SAME PLATE in the SAME BATCH -- which is what every SOTA LINCS model is handed.

This is the controlled A/B that decides it. Identical signatures, identical drug/dose/time features,
identical protocol; the ONLY thing that changes is which baseline vector the model sees:

    (a) X_base     CCLE proxy                      <- what we use today
    (b) X_ctl_l3   plate-matched DMSO median       <- what Level 3 buys
    (c) both

If (b) does not beat (a), the migration's premise is wrong and we should say so plainly rather than build
on it.
"""
import os, csv, json, argparse

import numpy as np
import torch

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v6 import V6DataConfig
from data import LincsDataset, build_splits
from eval_v6 import metrics

ROOT = r'C:\Projects\LINCS'
L3 = os.path.join(ROOT, 'phase2_assembly', 'outputs', 'level3')


def align_genes(dc):
    """X_ctl_l3 is in gene_info landmark order; our arrays are in the canonical project order."""
    l3_syms = [l.strip() for l in open(os.path.join(L3, 'genes_l3.txt'), encoding='utf-8') if l.strip()]
    our = [l.strip() for l in open(os.path.join(ROOT, 'Data Info', 'pathway_landmark_genes.txt'),
                                   encoding='utf-8') if l.strip()]
    pos = {g: i for i, g in enumerate(l3_syms)}
    idx = np.array([pos[g] for g in our if g in pos])
    keep = np.array([i for i, g in enumerate(our) if g in pos])
    print("gene alignment: %d/%d of our genes found in the Level-3 landmark set" % (len(idx), len(our)))
    return idx, keep


def ridge_fit(Xb, Y, lam):
    D = Xb.shape[1]
    G = Xb.T @ Xb + lam * np.eye(D)
    G[-1, -1] -= lam                      # never penalise the intercept
    return np.linalg.solve(G, Xb.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambdas', default='1e3,1e4,1e5')
    ap.add_argument('--max_train', type=int, default=60000)
    a = ap.parse_args()

    dc = V6DataConfig()
    ds = LincsDataset(dc)
    sp = build_splits(ds, dc)

    join = np.load(os.path.join(L3, 'join_l5row_to_l3row.npy'))
    l5_to_l3 = {int(r): int(j) for r, j in join}
    Xctl_all = np.load(os.path.join(L3, 'X_ctl_l3.npy'), mmap_mode='r')
    gidx, keep = align_genes(dc)

    # our dataset indexes signatures by position; y_row is the row into Y
    have = np.array([i for i in range(len(ds.y_row)) if int(ds.y_row[i]) in l5_to_l3])
    print("signatures with a plate-matched control: %d / %d" % (len(have), len(ds.y_row)))

    fp = ds.u_feats[:, -2048:]; desc = ds.u_feats[:, -2068:-2048]

    def build(idx, mode):
        d, c = ds.drug_row[idx], ds.cell_row[idx]
        blocks = [fp[d], desc[d], ds.cell_ctx[c],
                  ds.dose_n[idx][:, None], ds.time_n[idx][:, None]]
        if mode in ('ccle', 'both'):
            blocks.append(ds.Xb[c][:, keep])
        if mode in ('matched', 'both'):
            rows = np.array([l5_to_l3[int(ds.y_row[i])] for i in idx])
            blocks.append(np.asarray(Xctl_all[rows], np.float32)[:, gidx])
        X = np.concatenate(blocks, 1).astype(np.float64)
        return np.concatenate([X, np.ones((len(idx), 1))], 1)

    rng = np.random.default_rng(0)
    res = {}
    for mode in ['ccle', 'matched', 'both']:
        tr = np.intersect1d(sp['train'], have)
        if len(tr) > a.max_train:
            tr = rng.choice(tr, a.max_train, replace=False)
        va = np.intersect1d(sp['val'], have)
        Xtr, Ytr = build(tr, mode), np.asarray(ds.Y[ds.y_row[tr]], np.float64)
        mu, sd = Xtr[:, :-1].mean(0), Xtr[:, :-1].std(0) + 1e-6
        Xtr[:, :-1] = (Xtr[:, :-1] - mu) / sd
        best = None
        for lam in [float(x) for x in a.lambdas.split(',')]:
            W = ridge_fit(Xtr, Ytr, lam)
            Xv = build(va, mode); Xv[:, :-1] = (Xv[:, :-1] - mu) / sd
            mse = float(((Xv @ W - np.asarray(ds.Y[ds.y_row[va]], np.float64)) ** 2).mean())
            if best is None or mse < best[0]:
                best = (mse, lam, W)
        _, lam, W = best
        rec = {'lambda': lam, 'n_features': int(Xtr.shape[1] - 1), 'splits': {}}
        print("\n=== baseline input: %s ===  (lambda %.0e, %d features, train %d)"
              % (mode.upper(), lam, Xtr.shape[1] - 1, len(tr)))
        for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                          ('unseen_both', 'test_coldboth')]:
            idx = np.intersect1d(sp[key], have)
            idx = idx[ds.strength[idx] >= dc.eval_min_strength]
            if len(idx) < 300:
                continue
            if len(idx) > 6000:
                idx = np.sort(rng.choice(idx, 6000, replace=False))
            Xe = build(idx, mode); Xe[:, :-1] = (Xe[:, :-1] - mu) / sd
            yh = torch.from_numpy((Xe @ W).astype(np.float32))
            yt = torch.from_numpy(np.asarray(ds.Y[ds.y_row[idx]], np.float32))
            m = metrics(yh, yt)
            rec['splits'][name] = m
            print("  %-16s n=%-5d pearson=%.4f  R2=%+.4f" % (name, len(idx), m['pearson_median'],
                                                             m['r2_overall']))
        res[mode] = rec

    print("\n" + "=" * 74)
    print("DOES THE PLATE-MATCHED CONTROL BEAT THE CCLE PROXY?")
    print("=" * 74)
    print("  %-16s %10s %10s %10s" % ("split", "CCLE", "matched", "both"))
    for name in ['unseen_cell', 'unseen_compound', 'unseen_both']:
        if all(name in res[m]['splits'] for m in res):
            v = [res[m]['splits'][name]['pearson_median'] for m in ['ccle', 'matched', 'both']]
            print("  %-16s %10.4f %10.4f %10.4f   matched-CCLE = %+.4f" % (name, v[0], v[1], v[2],
                                                                           v[1] - v[0]))
    json.dump(res, open(os.path.join(ROOT, 'model', 'results', 'level3_matched_control_ab.json'), 'w'),
              indent=2)
    print("\nwrote model/results/level3_matched_control_ab.json")


if __name__ == '__main__':
    main()
