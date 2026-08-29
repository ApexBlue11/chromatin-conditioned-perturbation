# -*- coding: utf-8 -*-
"""
Is GSE70138 (Phase 2) a CLEANER substrate than GSE92742 (Phase 1), enough that training on P2 alone beats
training on both despite a third of the data?

The hypothesis is reasonable: P2 is the later, more standardised production run, and this project has been
burned repeatedly by treating "more rows" as "more signal" when ~75 % of LINCS is inert [rule 1].

But "P2-only is better" and "less data is worse" push in opposite directions, so a naive P2-only vs
both comparison cannot separate them. Four arms, and the third is the one that makes the test valid:

    p2_only     train on P2 rows only                       (N2 rows)
    p1_only     train on P1 rows only, SUBSAMPLED to N2     -- isolates the phase, holding size fixed
    both_eq     train on P1+P2,        SUBSAMPLED to N2     -- isolates the phase MIXTURE, size fixed
    both_full   train on everything available               -- what v9 actually does today

Every arm is evaluated on the SAME held-out rows, reported separately for P2 test rows and P1 test rows,
because "cleaner" should show up as P2 predicting P2 well, not as P2 predicting everything well.

Closed-form ridge, so there is no seed variance to confound the comparison. This is a GATE, not a result:
if it shows nothing, it costs one CPU run instead of three GPU sessions.

    python model/v9/phase_ab.py
"""
import os, sys, json, csv, argparse
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
    ap.add_argument('--lam', type=float, default=1e4)
    ap.add_argument('--max_eval', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    dc = V9DataConfig()
    dc.cache_in_ram = False
    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)

    phase = np.array([p for p in ds.phase])
    ok = ds.has_l3
    print('rows with a Level-3 substrate: P1 %d, P2 %d'
          % (int((ok & (phase == 'P1')).sum()), int((ok & (phase == 'P2')).sum())))

    tr_all = sp['train'][ok[sp['train']]]
    tr_p1 = tr_all[phase[tr_all] == 'P1']
    tr_p2 = tr_all[phase[tr_all] == 'P2']
    N2 = len(tr_p2)
    print('training rows: P1 %d, P2 %d -> every size-matched arm uses N2 = %d' % (len(tr_p1), N2, N2))
    if N2 < 5000:
        raise SystemExit('too few P2 training rows for a meaningful comparison')

    arms = {
        'p2_only': tr_p2,
        'p1_only': np.sort(rng.choice(tr_p1, N2, replace=False)),
        'both_eq': np.sort(rng.choice(tr_all, N2, replace=False)),
        'both_full': tr_all if len(tr_all) <= 60000 else np.sort(rng.choice(tr_all, 60000, replace=False)),
    }

    fp = ds.u_feats[:, -2048:]
    desc = ds.u_feats[:, -2068:-2048]

    def feats(idx):
        d, c = ds.drug_row[idx], ds.cell_row[idx]
        rows = ds.ds_to_l3[idx]
        X = np.concatenate([fp[d], desc[d], ds.cell_ctx[c], ds.dose_n[idx][:, None],
                            ds.time_n[idx][:, None], np.asarray(ds.Xctl[rows], np.float32)], 1)
        return np.concatenate([X.astype(np.float32), np.ones((len(idx), 1), np.float32)], 1)

    def delta(idx):
        rows = ds.ds_to_l3[idx]
        return np.asarray(ds.Xtrt[rows], np.float32) - np.asarray(ds.Xctl[rows], np.float32)

    # evaluation rows chosen ONCE and shared by every arm, split by phase
    ev = {}
    for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                      ('unseen_both', 'test_coldboth')]:
        base = sp[key][ok[sp[key]] & (ds.strength[sp[key]] >= dc.eval_min_strength)]
        for ph in ['P1', 'P2']:
            idx = base[phase[base] == ph]
            if len(idx) < 200:
                continue
            if len(idx) > a.max_eval:
                idx = np.sort(rng.choice(idx, a.max_eval, replace=False))
            ev[(name, ph)] = idx
    print('eval rows: ' + ', '.join(f'{k[0]}/{k[1]}={len(v)}' for k, v in ev.items()))

    res = defaultdict(dict)
    for arm, tr in arms.items():
        Xtr = feats(tr)
        mu, sd = Xtr[:, :-1].mean(0), Xtr[:, :-1].std(0) + 1e-6
        Xtr[:, :-1] = (Xtr[:, :-1] - mu) / sd
        G = gram(Xtr)
        G[np.diag_indices(G.shape[0])] += a.lam
        G[-1, -1] -= a.lam
        Ytr = delta(tr)
        B = np.zeros((Xtr.shape[1], Ytr.shape[1]), np.float64)
        for lo in range(0, len(Xtr), 8192):
            B += np.asarray(Xtr[lo:lo + 8192], np.float64).T @ np.asarray(Ytr[lo:lo + 8192], np.float64)
        W = np.linalg.solve(G, B)
        for k, idx in ev.items():
            Xe = feats(idx); Xe[:, :-1] = (Xe[:, :-1] - mu) / sd
            r = float(np.nanmedian(pearson_rows((Xe @ W).astype(np.float32), delta(idx))))
            res[arm][f'{k[0]}/{k[1]}'] = round(r, 4)
        res[arm]['n_train'] = int(len(tr))
        print('  %-10s n_train %6d  ' % (arm, len(tr))
              + '  '.join(f'{k}={v:.4f}' for k, v in res[arm].items() if k != 'n_train'), flush=True)

    print('\n' + '=' * 104)
    print('IS PHASE 2 A CLEANER SUBSTRATE?  (ridge on the Level-3 delta; only the TRAINING rows change)')
    print('=' * 104)
    cols = [k for k in res['p2_only'] if k != 'n_train']
    print('  %-10s %7s ' % ('arm', 'n_train') + ''.join('%18s' % c for c in cols))
    for arm in ['p2_only', 'p1_only', 'both_eq', 'both_full']:
        print('  %-10s %7d ' % (arm, res[arm]['n_train']) + ''.join('%18.4f' % res[arm][c] for c in cols))
    print('\n  p2_only - both_eq  (the PHASE effect, training size held fixed):')
    for c in cols:
        d = res['p2_only'][c] - res['both_eq'][c]
        print(f'    {c:22s} {d:+.4f}')
    print('\n  both_full - both_eq  (what the extra data is worth, phase mixture held fixed):')
    for c in cols:
        d = res['both_full'][c] - res['both_eq'][c]
        print(f'    {c:22s} {d:+.4f}')
    out = os.path.join(dc.root, 'model', 'results', 'v9_phase_ab.json')
    json.dump(res, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
