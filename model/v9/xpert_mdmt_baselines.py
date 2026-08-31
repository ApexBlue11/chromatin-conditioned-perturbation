# -*- coding: utf-8 -*-
"""
The baselines XPert's benchmark is missing, computed on its own held-out rows.

Their `get_metrics_new` reports Pearson and Pearson_deg and no null, so an absolute Pearson of 0.98 reads
as near-perfect prediction when copy-the-control alone scores 0.92-0.94 on the same rows [RESULTS 33, 40].
This script puts three reference predictors on exactly the rows their split holds out, under exactly their
metric (the MEAN of per-row Pearson), and writes predictions in the same format their harness does, so the
paired comparison in head_to_head_mdmt.py can consume them.

    copy_control     y_hat = x_ctl                    the null their metric never reports
    mean_drug_delta  y_hat = x_ctl + mean delta of that compound over TRAINING rows (global mean if unseen)
    ridge            y_hat = x_ctl + W [x_ctl, ECFP4, descriptors, log dose, time]

The ridge is closed-form, so it has no seed variance, and it is fitted ONLY on their training rows. It is
not a strawman for the published baselines (DeepCE, PRnet, TranSiGen) -- those are not in their release and
are not reproduced here; it is the linear reference this project has used since v3, on their rows.

    python model/v9/xpert_mdmt_baselines.py --split split_1
"""
import os, sys, json, glob, argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r'C:\Projects\LINCS'
BUNDLE = os.path.join(ROOT, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz')


def find(name):
    for r in [ROOT]:
        hit = glob.glob(os.path.join(r, '**', name), recursive=True)
        hit = [h for h in hit if 'external' not in h]
        if hit:
            return hit[0]
    raise SystemExit('FATAL: %s not found' % name)


def per_row_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


def summarize(r, boot=2000, seed=0):
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(seed)
    bs = np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(boot)])
    return {'mean': round(float(r.mean()), 4),
            'ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            'median': round(float(np.median(r)), 4), 'n': int(len(r))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=BUNDLE)
    ap.add_argument('--split', default='split_1')
    ap.add_argument('--lam', type=float, default=1e4)
    ap.add_argument('--save_pred', default=None)
    ap.add_argument('--with_chromatin', action='store_true',
                    help='append the cell chromatin tracks to the ridge features. The sharpest '
                         'form of the epigenetics question: the control profile already '
                         'characterises the cell line, so does chromatin add anything BEYOND '
                         'it? Closed-form, so no seed noise and no training-budget confound.')
    a = ap.parse_args()

    z = np.load(a.bundle, allow_pickle=True)
    lab = z['split_' + a.split].astype(str)
    tr, te = np.flatnonzero(lab == 'train'), np.flatnonzero(lab == 'test')
    if not len(te):
        raise SystemExit('FATAL: %s has no test rows' % a.split)
    X, C = z['X'], z['X_ctl']
    pert = z['meta_pert_id'].astype(str)
    print('%s: %d train / %d test rows' % (a.split, len(tr), len(te)), flush=True)

    # ---- our drug features, keyed by THEIR pert_id; rows we cannot featurise are reported, not hidden ----
    di = json.load(open(find('drug_feature_index.json')))
    have = np.array([p in di for p in pert])
    if not have[te].all():
        print('  %d of %d test rows use a compound we cannot featurise; the ridge falls back to the '
              'global mean delta for those' % (int((~have[te]).sum()), len(te)), flush=True)
    rows = np.array([di.get(p, -1) for p in pert])
    desc = np.load(find('drug_descriptors.npy')).astype(np.float32)
    desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)
    fp = np.load(find('drug_fingerprints.npy')).astype(np.float32)

    ld = np.log10(np.clip(z['meta_dose'], 1e-4, None)).astype(np.float32)
    tm = z['meta_time'].astype(np.float32)

    Echrom = None
    if a.with_chromatin:
        cidx = json.load(open(find('lincs_cell_index.json')))
        cidx = cidx.get('cell_id_to_row', cidx)
        E = np.load(find('E_final.npy')).astype(np.float32)
        Em = np.load(find('E_final_mask.npy'))
        for k in range(E.shape[2]):                       # standardise each track where it exists
            hask = Em[:, :, k].any(1)
            for cc in np.where(hask)[0]:
                v = E[cc, :, k]
                E[cc, :, k] = (v - v.mean()) / (v.std() + 1e-6)
        cellnames = z['meta_cell'].astype(str)
        uniq = sorted(set(cellnames.tolist()))
        G = E.shape[1]
        # A per-cell feature block over K cell lines has rank <= K, so the 3,912 raw chromatin dimensions
        # carry at most K-1 degrees of freedom beyond a constant. Reducing to K components by SVD is
        # EXACT -- it discards no information the ridge could have used -- and turns an intractable
        # 6,961-column design matrix into a fast one.
        per_cell = np.zeros((len(uniq), G * E.shape[2] + G), np.float32)
        n_known = 0
        for i, cc in enumerate(uniq):
            j = cidx.get(str(cc))
            if j is None:
                continue
            n_known += 1
            per_cell[i] = np.concatenate([E[j].ravel(), Em[j].any(-1).astype(np.float32)])
        mu_c = per_cell.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(per_cell - mu_c, full_matrices=False)
        keep = int((S > S.max() * 1e-8).sum())
        emb = (U[:, :keep] * S[:keep]).astype(np.float32)
        rank_ok = np.allclose(emb @ np.linalg.pinv(emb) @ (per_cell - mu_c), per_cell - mu_c, atol=1e-3)
        pos = {c: i for i, c in enumerate(uniq)}
        Echrom = emb[np.array([pos[c] for c in cellnames])]
        print('  chromatin: %d cell lines, %d with a track; %d raw dims -> %d exact SVD components '
              '(reconstruction exact: %s); %.1f%% of rows have a known cell'
              % (len(uniq), n_known, per_cell.shape[1], keep, rank_ok,
                 100 * np.mean([c in cidx for c in cellnames])), flush=True)

    def feats(idx):
        r = np.clip(rows[idx], 0, None)
        blocks = [C[idx], fp[r], desc[r], ld[idx][:, None], tm[idx][:, None],
                  have[idx][:, None].astype(np.float32)]
        if Echrom is not None:
            blocks.append(Echrom[idx])
        F = np.concatenate(blocks, 1).astype(np.float32)
        return np.concatenate([F, np.ones((len(idx), 1), np.float32)], 1)

    D = (X - C).astype(np.float32)
    res, preds = {}, {}

    # ---- 1. copy the control ----
    preds['copy_control'] = C[te].copy()

    # ---- 2. per-compound mean delta from the training rows ----
    gm = D[tr].mean(0)
    per = {}
    for p in np.unique(pert[tr]):
        per[p] = D[tr][pert[tr] == p].mean(0)
    seen = np.array([p in per for p in pert[te]])
    preds['mean_drug_delta'] = C[te] + np.stack([per.get(p, gm) for p in pert[te]])
    print('  mean-drug baseline: %d of %d test rows use a compound seen in training'
          % (int(seen.sum()), len(te)), flush=True)

    # ---- 3. ridge on the delta ----
    Ftr = feats(tr)
    mu, sd = Ftr[:, :-1].mean(0), Ftr[:, :-1].std(0) + 1e-6
    Ftr[:, :-1] = (Ftr[:, :-1] - mu) / sd
    G = np.zeros((Ftr.shape[1], Ftr.shape[1]), np.float64)
    B = np.zeros((Ftr.shape[1], X.shape[1]), np.float64)
    for lo in range(0, len(Ftr), 8192):
        b = np.asarray(Ftr[lo:lo + 8192], np.float64)
        G += b.T @ b
        B += b.T @ np.asarray(D[tr][lo:lo + 8192], np.float64)
    G[np.diag_indices(G.shape[0])] += a.lam
    G[-1, -1] -= a.lam
    W = np.linalg.solve(G, B)
    Fte = feats(te)
    Fte[:, :-1] = (Fte[:, :-1] - mu) / sd
    preds['ridge'] = C[te] + (Fte @ W).astype(np.float32)

    # ---- score every one of them under THEIR metric ----
    y, ctl = X[te], C[te]
    print('\n' + '=' * 92)
    print('BASELINES ON XPERT\'S OWN HELD-OUT ROWS (%s, n=%d), mean of per-row Pearson' % (a.split, len(te)))
    print('=' * 92)
    print('  %-18s %-26s %-26s' % ('predictor', 'absolute Pearson', 'delta Pearson'))
    for name in ['copy_control', 'mean_drug_delta', 'ridge']:
        f = preds[name]
        ab = summarize(per_row_pearson(f, y))
        dl = summarize(per_row_pearson(f - ctl, y - ctl)) if name != 'copy_control' else None
        res[name] = {'abs': ab, 'delta': dl}
        print('  %-18s %.4f %-19s %s' % (name, ab['mean'], str(ab['ci95']),
                                         ('%.4f %s' % (dl['mean'], dl['ci95'])) if dl
                                         else '0.0000  (zero delta by construction)'))
    res['n_test'] = int(len(te))
    res['n_train'] = int(len(tr))
    res['unfeaturisable_test_rows'] = int((~have[te]).sum())
    res['with_chromatin'] = bool(a.with_chromatin)

    if a.save_pred:
        np.savez_compressed(a.save_pred, row_index=z['row_index'][te], y_true=y, ctl_true=ctl,
                            **{k + '_pred': v for k, v in preds.items()})
        print('\nwrote %s' % a.save_pred)
    dst = os.path.join(os.path.dirname(HERE), 'results', 'xpert_mdmt_baselines_%s%s.json'
                       % (a.split, '_chrom' if a.with_chromatin else ''))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, 'w'), indent=2)
    print('wrote %s' % dst)


if __name__ == '__main__':
    main()
