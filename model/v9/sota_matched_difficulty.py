# -*- coding: utf-8 -*-
"""
How much of the gap to published LINCS numbers is the SPLIT rather than the model?

`sota_split_audit.py` establishes that XPert's 15 released splits are within-tissue 90/10 divisions in
which 100 % of test rows use a cell line seen in training, 98.6 % use a compound seen in training, and
89.4 % use the exact (cell, compound) PAIR seen in training -- a different dose or time of something
already in the training set. Our benchmarks hold out whole cell lines and whole Bemis-Murcko scaffold
families.

So the honest question is not "their 0.844 versus our 0.4985". It is: with ONE model, ONE feature set and
ONE target, what does changing only the SPLIT do? That is a controlled experiment we can run, and it
converts "not comparable" from a caveat into a number.

Four regimes, identical ridge and identical features throughout:

    xpert_style   random 90/10 over all conditions      -- seen cell, seen compound, new dose/time
    xpert_tissue  random 90/10 WITHIN one lineage        -- their exact construction
    cold_cell     unseen cell line                       -- our benchmark
    cold_compound unseen scaffold family                 -- our benchmark
    cold_both     both                                   -- our hardest

Every regime ships with its nulls, because an absolute-convention number without them is uninterpretable:
`copy_ctl` predicts the control unchanged, and `mean_drug` predicts that drug's average response.

    python model/v9/sota_matched_difficulty.py
"""
import os, sys, json, argparse
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9DataConfig
from data_v9 import LincsV9Dataset, build_splits


def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


def gram(X, chunk=8192):
    G = np.zeros((X.shape[1], X.shape[1]), np.float64)
    for lo in range(0, len(X), chunk):
        b = np.asarray(X[lo:lo + chunk], np.float64)
        G += b.T @ b
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_train', type=int, default=40000)
    ap.add_argument('--max_eval', type=int, default=6000)
    ap.add_argument('--lam', type=float, default=1e4)
    ap.add_argument('--lineage', type=int, default=None,
                    help='lineage column for xpert_tissue; default = the most populated one')
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    dc = V9DataConfig()
    dc.cache_in_ram = False
    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)
    have = np.flatnonzero(ds.has_l3 & (ds.strength >= dc.eval_min_strength))
    all_l3 = np.flatnonzero(ds.has_l3)
    print(f'rows with a Level-3 substrate: {len(all_l3)} | of those on the reproducible stratum: {len(have)}')

    lin = ds.cell_ctx.argmax(1)[ds.cell_row]
    if a.lineage is None:
        counts = np.bincount(lin[have], minlength=ds.cell_ctx.shape[1])
        a.lineage = int(np.argmax(counts))
    print(f'xpert_tissue uses lineage column {a.lineage} ({int((lin[have] == a.lineage).sum())} rows)')

    fp = ds.u_feats[:, -2048:]
    desc = ds.u_feats[:, -2068:-2048]

    def feats(idx):
        d, c = ds.drug_row[idx], ds.cell_row[idx]
        rows = ds.ds_to_l3[idx]
        X = np.concatenate([fp[d], desc[d], ds.cell_ctx[c], ds.dose_n[idx][:, None],
                            ds.time_n[idx][:, None], np.asarray(ds.Xctl[rows], np.float32)], 1)
        return np.concatenate([X.astype(np.float32), np.ones((len(idx), 1), np.float32)], 1)

    def targets(idx):
        rows = ds.ds_to_l3[idx]
        trt = np.asarray(ds.Xtrt[rows], np.float32)
        ctl = np.asarray(ds.Xctl[rows], np.float32)
        return {'delta': trt - ctl, 'abs': trt, 'ctl': ctl}

    # ---- the five regimes -------------------------------------------------------------------------
    regimes = {}
    pool = np.array(sorted(set(all_l3.tolist())))
    perm = rng.permutation(len(pool))
    cut = int(0.9 * len(pool))
    regimes['xpert_style'] = (pool[perm[:cut]], pool[perm[cut:]])
    tpool = pool[lin[pool] == a.lineage]
    tperm = rng.permutation(len(tpool))
    tcut = int(0.9 * len(tpool))
    regimes['xpert_tissue'] = (tpool[tperm[:tcut]], tpool[tperm[tcut:]])
    # XPert's splits sit at 89.4 % (cell, compound) pair overlap -- higher than a plain random 90/10
    # reaches on our data. This regime restricts evaluation to test rows whose pair IS in training, which
    # matches their construction and isolates the last piece of the difficulty difference.
    tr0, te0 = regimes['xpert_style']
    pair_tr = np.unique(ds.cell_row[tr0].astype(np.int64) * 1_000_000 + ds.drug_row[tr0])
    te_pair = te0[np.isin(ds.cell_row[te0].astype(np.int64) * 1_000_000 + ds.drug_row[te0], pair_tr)]
    regimes['xpert_pair'] = (tr0, te_pair)
    for name, key in [('cold_cell', 'test_coldcell'), ('cold_compound', 'test_colddrug'),
                      ('cold_both', 'test_coldboth')]:
        regimes[name] = (np.intersect1d(sp['train'], all_l3), np.intersect1d(sp[key], all_l3))

    # drug-mean baseline, computed on each regime's OWN training rows
    res = {}
    for name, (tr_all, te_all) in regimes.items():
        te = te_all[ds.strength[te_all] >= dc.eval_min_strength]
        if len(te) < 300:
            print(f'{name}: only {len(te)} evaluable test rows, skipping')
            continue
        if len(te) > a.max_eval:
            te = np.sort(rng.choice(te, a.max_eval, replace=False))
        tr = tr_all if len(tr_all) <= a.max_train else np.sort(
            rng.choice(tr_all, a.max_train, replace=False))

        Xtr = feats(tr)
        mu, sd = Xtr[:, :-1].mean(0), Xtr[:, :-1].std(0) + 1e-6
        Xtr[:, :-1] = (Xtr[:, :-1] - mu) / sd
        G = gram(Xtr)
        G[np.diag_indices(G.shape[0])] += a.lam
        G[-1, -1] -= a.lam
        Ttr = targets(tr)
        Xe = feats(te); Xe[:, :-1] = (Xe[:, :-1] - mu) / sd
        Tte = targets(te)

        rec = {'n_train': int(len(tr)), 'n_test': int(len(te))}
        # overlap structure of this regime, in the same terms as the audit of theirs
        rec['test_cell_seen'] = round(float(np.isin(ds.cell_row[te], np.unique(ds.cell_row[tr])).mean()), 4)
        rec['test_compound_seen'] = round(
            float(np.isin(ds.drug_row[te], np.unique(ds.drug_row[tr])).mean()), 4)
        pair = lambda i: ds.cell_row[i].astype(np.int64) * 1_000_000 + ds.drug_row[i]
        rec['test_pair_seen'] = round(float(np.isin(pair(te), np.unique(pair(tr))).mean()), 4)

        for tgt in ['delta', 'abs']:
            B = np.zeros((Xtr.shape[1], 978), np.float64)
            for lo in range(0, len(Xtr), 8192):
                B += np.asarray(Xtr[lo:lo + 8192], np.float64).T @ \
                     np.asarray(Ttr[tgt][lo:lo + 8192], np.float64)
            W = np.linalg.solve(G, B)
            pred = (Xe @ W).astype(np.float32)
            rec[f'ridge_{tgt}'] = round(float(np.nanmedian(pearson_rows(pred, Tte[tgt]))), 4)

        # nulls
        rec['copy_ctl_abs'] = round(float(np.nanmedian(pearson_rows(Tte['ctl'], Tte['abs']))), 4)
        rec['zero_delta'] = 0.0
        dm = defaultdict(list)
        for i, d in zip(tr, ds.drug_row[tr]):
            dm[int(d)].append(i)
        gm = Ttr['delta'].mean(0)
        pred = np.stack([Ttr['delta'][np.searchsorted(tr, dm[int(d)])].mean(0)
                         if int(d) in dm and len(dm[int(d)]) else gm for d in ds.drug_row[te]])
        rec['mean_drug_delta'] = round(float(np.nanmedian(pearson_rows(pred, Tte['delta']))), 4)
        res[name] = rec
        print(f'  {name:14s} train {rec["n_train"]:6d} test {rec["n_test"]:5d} | '
              f'cell seen {100 * rec["test_cell_seen"]:5.1f}% compound {100 * rec["test_compound_seen"]:5.1f}% '
              f'pair {100 * rec["test_pair_seen"]:5.1f}% | delta {rec["ridge_delta"]:.4f} '
              f'abs {rec["ridge_abs"]:.4f} | copy_ctl {rec["copy_ctl_abs"]:.4f} '
              f'mean_drug {rec["mean_drug_delta"]:.4f}', flush=True)

    print('\n' + '=' * 108)
    print('ONE MODEL, ONE FEATURE SET, ONE TARGET -- ONLY THE SPLIT CHANGES')
    print('=' * 108)
    print('  %-14s %7s %7s %7s   %8s %8s   %9s %10s'
          % ('regime', 'cell%', 'cmpd%', 'pair%', 'delta', 'abs', 'copy_ctl', 'mean_drug'))
    for k, r in res.items():
        print('  %-14s %6.1f%% %6.1f%% %6.1f%%   %8.4f %8.4f   %9.4f %10.4f'
              % (k, 100 * r['test_cell_seen'], 100 * r['test_compound_seen'], 100 * r['test_pair_seen'],
                 r['ridge_delta'], r['ridge_abs'], r['copy_ctl_abs'], r['mean_drug_delta']))
    print('\nXPert\'s released splits sit at cell 100.0%, compound 98.6%, pair 89.4% [sota_split_audit],')
    print('i.e. alongside the xpert_style / xpert_tissue rows, not the cold_* rows.')
    out = os.path.join(dc.root, 'model', 'results', 'sota_matched_difficulty.json')
    json.dump(res, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
