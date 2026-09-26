# -*- coding: utf-8 -*-
"""RESULTS 85.2 rule 8 instrument (PI-written, review 031 C1): the named pathway layer's cell-level alignment on the dev rows.

    python model/v9/align_dev.py --ckpts A_seed0.pt [A_seed1.pt ...] --label NAME [--n_perm 200]

Per checkpoint: `aux['pathway_activations']` (the pre-drug named layer) averaged over its channels -> [rows, P]; target per
row = measured |y_true - ctl_true| averaged over each pathway's member genes (M row-normalised) -> [rows, P]; alignment =
mean over rows of the per-row Spearman between the two (interp_v9.pathway_alignment, as RESULTS 37). Null: the same with the
pathway columns of the activations permuted (one permutation for all rows), n_perm times, rng seed 0 (as probe_v9).
Rows: the 4,043 dev rows (sha1 51e7e4ab..., asserted), rebuilt through xpert_arm's XPertData as mc_infer_dev.py does.
Rule 8 (operationalised in RESULTS 85.10): a variant passes iff mean_s(obs) >= baseline mean_s(obs) - 0.02 AND
mean_s(obs) >= mean_s(null_mean) + 5 * mean_s(null_sd).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config
from model_v9 import LincsV9
from xpert_arm import XPertData, find
from interp_v9 import pathway_alignment

DEV_SHA1 = '51e7e4ab8b9c3c3709d43da7fa4a8c80b77d5980'


def load_dev(split, dev_cells=6, dev_seed=0, ablate_epi=False):
    roots = ['/kaggle/input', r'C:\Projects\LINCS', os.path.join(r'C:\Projects\LINCS', 'external')]
    npz = find('xpert_mdmt_splits.npz', roots)
    dev_args = argparse.Namespace(dev_cells=dev_cells, dev_seed=dev_seed, dev_min_rows=200, dev_max_rows=2000)
    D = XPertData(npz, roots, split, ablate_epi=ablate_epi, dev_args=dev_args)
    sha = hashlib.sha1(np.sort(D.row_index[D.te].astype(np.int64)).tobytes()).hexdigest()
    assert sha == DEV_SHA1, 'not the RESULTS 85.4 dev rows: %s' % sha
    M = np.load(find('M_pathway_v9.npy', roots))
    ppi = np.load(find('STRING_adj_978_v9.npy', roots))
    gv = np.load(find('gene_vectors_978.npy', roots))
    return D, M, ppi, gv


def activations(model, D, idx, dev, readout='mean'):
    """readout 'mean': channel mean of the named layer (RESULTS 37); 'aux': the aux-supervised per-node readout
    aux['pathway_pred'] (the quantity the aux loss trains)."""
    A = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = D.batch(idx[s:s + 64], dev)
            _, aux = model(b, return_aux=True)
            A.append((aux['pathway_activations'].float().mean(-1) if readout == 'mean'
                      else aux['pathway_pred'].float()).cpu().numpy())
    return np.concatenate(A)


def align_with_null(acts, delta, M, n_perm, seed=0):
    obs = pathway_alignment(acts, delta, M)
    rng = np.random.default_rng(seed)
    null = np.array([pathway_alignment(acts[:, rng.permutation(acts.shape[1])], delta, M) for _ in range(n_perm)])
    return float(obs), float(null.mean()), float(null.std())


def pathway_target(delta, M):
    Mn = np.asarray(M, np.float64)
    Mn = Mn / np.maximum(Mn.sum(1, keepdims=True), 1)
    return np.abs(np.asarray(delta, np.float64)) @ Mn.T                     # [rows, P]


def training_prior(D, M):
    """Review 032 C1(a): the cell-agnostic ranking a model could learn from training rows alone -- the mean pathway target
    over the training rows, the same vector for every dev row."""
    return pathway_target(D.X[D.tr] - D.C[D.tr], M).mean(0)


def loco_prior(D, M, cells):
    """Review 032's leave-one-cell-out proxy: per dev cell, the mean pathway target over the OTHER dev cells' rows."""
    T = pathway_target(D.X[D.te] - D.C[D.te], M)
    out = np.empty_like(T)
    for c in np.unique(cells):
        out[cells == c] = T[cells != c].mean(0)
    return out


def rowwise_spearman(A, T):
    """Per-row Spearman between A [rows, P] and T [rows, P] (vectorised form of interp_v9.pathway_alignment's inner loop)."""
    ra = np.argsort(np.argsort(A, 1), 1).astype(np.float64)
    rt = np.argsort(np.argsort(T, 1), 1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True)
    rt -= rt.mean(1, keepdims=True)
    d = np.sqrt((ra ** 2).sum(1) * (rt ** 2).sum(1))
    return np.where(d > 0, (ra * rt).sum(1) / np.where(d > 0, d, 1), 0.0)


def in_cell_increment(acts, delta, M, cells):
    """Review 032 addendum: per row, rho(own readout) - rho(the same model's readout averaged over the OTHER dev cells' rows);
    returns the per-cell mean increment {cell: value}."""
    T = pathway_target(delta, M)
    own = rowwise_spearman(acts, T)
    ref = np.empty_like(acts, dtype=np.float64)
    for c in np.unique(cells):
        ref[cells == c] = acts[cells != c].mean(0)
    inc = own - rowwise_spearman(ref, T)
    return {str(c): float(inc[cells == c].mean()) for c in np.unique(cells)}


def cell_shuffle_null(acts, delta, M, cells, n_perm=200, seed=0):
    """Review 032 C1(b): each dev cell's rows get readouts drawn from rows of ANOTHER dev cell (a random derangement of the
    cells per permutation; pathway columns intact). Returns (mean, sd) of the alignment."""
    rng = np.random.default_rng(seed)
    uc = np.unique(cells)
    idx_by = {c: np.flatnonzero(cells == c) for c in uc}
    vals = []
    for _ in range(n_perm):
        while True:
            perm = rng.permutation(len(uc))
            if not np.any(perm == np.arange(len(uc))):
                break
        A = np.empty_like(acts)
        for c, c2 in zip(uc, uc[perm]):
            A[idx_by[c]] = acts[rng.choice(idx_by[c2], size=len(idx_by[c]), replace=True)]
        vals.append(pathway_alignment(A, delta, M))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--split', default='split_cold_cell_1')
    ap.add_argument('--n_perm', type=int, default=200)
    ap.add_argument('--out', default=None)
    ap.add_argument('--readout', choices=['mean', 'aux'], default='mean')
    ap.add_argument('--no_nulls', action='store_true', help='skip the permutation and cell-shuffle nulls (slow)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    D, M, ppi, gv = None, None, None, None
    per = []
    for p in a.ckpts:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert ck['split'] == a.split
        if D is None:
            D, M, ppi, gv = load_dev(a.split, ablate_epi=ck.get('ablate_epi', False))
            delta = D.X[D.te] - D.C[D.te]
            cells = np.asarray(D.cell[D.te]) if hasattr(D, 'cell') else None
            prior_tr = training_prior(D, M)
            ref = {'training_prior': pathway_alignment(np.tile(prior_tr, (len(D.te), 1)), delta, M)}
            if cells is not None:
                ref['loco_prior'] = pathway_alignment(loco_prior(D, M, cells), delta, M)
            print('references (data only):', ref, flush=True)
        cfg = V9Config()
        for k, v in ck['cfg'].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        model = LincsV9(cfg, M, ppi, gv)
        model.load_state_dict(ck['model'], strict=True)
        model = model.to(dev).eval()
        acts = activations(model, D, D.te, dev, a.readout)
        if a.no_nulls:
            obs, nm, ns, csm, css = pathway_alignment(acts, delta, M), None, None, None, None
        else:
            obs, nm, ns = align_with_null(acts, delta, M, a.n_perm)
            csm, css = cell_shuffle_null(acts, delta, M, cells, a.n_perm) if cells is not None else (None, None)
        inc = in_cell_increment(acts, delta, M, cells) if cells is not None else None
        per.append({'ckpt': os.path.basename(p), 'ckpt_sha1': hashlib.sha1(open(p, 'rb').read()).hexdigest(),
                    'seed': ck.get('seed'), 'alignment': obs, 'null_mean': nm, 'null_sd': ns,
                    'z': (obs - nm) / ns if ns else None,
                    'cell_shuffle_mean': csm, 'cell_shuffle_sd': css, 'in_cell_increment': inc,
                    'in_cell_increment_mean': float(np.mean(list(inc.values()))) if inc else None})
        print('%s alignment %.4f | in-cell increment by cell %s mean %+.4f' % (
            per[-1]['ckpt'], obs, {k: round(v, 4) for k, v in (inc or {}).items()}, per[-1]['in_cell_increment_mean'] or 0),
            flush=True)
        del model
    res = {'label': a.label, 'readout': a.readout, 'n_rows': int(len(D.te)), 'n_perm': a.n_perm, 'references': ref,
           'per_checkpoint': per,
           'mean_alignment': float(np.mean([r['alignment'] for r in per])),
           'mean_null_mean': None if a.no_nulls else float(np.mean([r['null_mean'] for r in per])),
           'mean_null_sd': None if a.no_nulls else float(np.mean([r['null_sd'] for r in per])),
           'sd_alignment': float(np.std([r['alignment'] for r in per], ddof=1)) if len(per) > 1 else None}
    if per and per[0]['in_cell_increment']:
        # RESULTS 85.10 (review 032 addendum): "in this cell" licensed iff the 3-seed mean per-cell increment is > 0 in >= 5 of
        # 6 dev cells AND the mean of the six per-cell increments is > 0 on every seed.
        cs = sorted(per[0]['in_cell_increment'])
        m3 = {c: float(np.mean([r['in_cell_increment'][c] for r in per])) for c in cs}
        res['in_cell'] = {'per_cell_3seed_mean': m3, 'cells_positive': int(sum(v > 0 for v in m3.values())),
                          'seed_means': [r['in_cell_increment_mean'] for r in per],
                          'licensed': bool(sum(v > 0 for v in m3.values()) >= 5 and
                                           all(r['in_cell_increment_mean'] > 0 for r in per))}
    out = a.out or os.path.join(HERE, '..', 'results', 'v9_dev_align_%s_%s.json' % (a.label, a.readout))
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'per_checkpoint'}, indent=1))


if __name__ == '__main__':
    main()
