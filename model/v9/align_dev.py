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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--split', default='split_cold_cell_1')
    ap.add_argument('--n_perm', type=int, default=200)
    ap.add_argument('--out', default=None)
    ap.add_argument('--readout', choices=['mean', 'aux'], default='mean')
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
        cfg = V9Config()
        for k, v in ck['cfg'].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        model = LincsV9(cfg, M, ppi, gv)
        model.load_state_dict(ck['model'], strict=True)
        model = model.to(dev).eval()
        acts = activations(model, D, D.te, dev, a.readout)
        obs, nm, ns = align_with_null(acts, delta, M, a.n_perm)
        per.append({'ckpt': os.path.basename(p), 'ckpt_sha1': hashlib.sha1(open(p, 'rb').read()).hexdigest(),
                    'seed': ck.get('seed'), 'alignment': obs, 'null_mean': nm, 'null_sd': ns,
                    'z': (obs - nm) / ns if ns > 0 else None})
        print('%s alignment %.4f | null %.4f +/- %.4f | z %.1f' % (per[-1]['ckpt'], obs, nm, ns, per[-1]['z'] or 0), flush=True)
        del model
    res = {'label': a.label, 'readout': a.readout, 'n_rows': int(len(D.te)), 'n_perm': a.n_perm, 'per_checkpoint': per,
           'mean_alignment': float(np.mean([r['alignment'] for r in per])),
           'mean_null_mean': float(np.mean([r['null_mean'] for r in per])),
           'mean_null_sd': float(np.mean([r['null_sd'] for r in per])),
           'sd_alignment': float(np.std([r['alignment'] for r in per], ddof=1)) if len(per) > 1 else None}
    out = a.out or os.path.join(HERE, '..', 'results', 'v9_dev_align_%s_%s.json' % (a.label, a.readout))
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'per_checkpoint'}, indent=1))


if __name__ == '__main__':
    main()
