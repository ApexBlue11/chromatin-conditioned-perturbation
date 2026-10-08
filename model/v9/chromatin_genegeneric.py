# -*- coding: utf-8 -*-
"""RESULTS 93.15: is §91's gene-generic chromatin gain (T1's N1 - FB, +0.0012) chromatin-specific? PI glue on chromatin_funnel.

    python model/v9/chromatin_genegeneric.py --data_dir <dir> --provenance <json> --out <json>      # Kaggle: scores only
    python model/v9/chromatin_genegeneric.py --read DIR                                             # local: the one reading

N1 is §91's (the fitting-cell mean of each mark's Ez, LOCO). N1perm_d gene-permutes each mark's mean vector (capacity null,
20 draws); N1expr replaces it with the fitting-cell mean of basal expression b, quantile-matched to the mark mean (a per-gene
covariate that is not chromatin). Same marginal, same availability, same T1. The kernel writes scores and no reading; the
reading is applied locally, once, after the marker's sha1 is verified, with the harness check against 91.12.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
F91 = os.path.join(REPO, 'model', 'results', 'chromatin_funnel_91.json')
N_PERM = 20
OUT = 'chromatin_genegeneric_93.json'
MARKER = 'CHROMATIN93_GG_COMPLETE.json'


def n1_variant(ctx, enc, fit_cells, kind, draw=None):
    """Builder like cf.feature_builder(..., 'N1', ...): [b, mean mark where the cell has it]. kind 'N1' (identical to §91's),
    'perm' (each mark's mean vector gene-permuted, rng 9500 + 10 draw + k), 'expr' (each mark's mean vector replaced by the
    fitting-cell mean of b, quantile-matched to it). The held-out cell h is excluded from every mean, as in §91."""
    e = ctx.enc[enc]
    b, Ez, has = e['b'], e['Ez'], e['has']
    G = cf.G

    def build(h):
        src = [cf.cell_index(ctx, c) for c in fit_cells if c != h]
        mean_mark = np.zeros((G, 3), np.float32)
        for k in range(3):
            idx = [i for i in src if has[i, k]]
            if idx:
                mean_mark[:, k] = Ez[idx, :, k].mean(0)
        if kind == 'perm':
            for k in range(3):
                mean_mark[:, k] = mean_mark[np.random.default_rng(9500 + 10 * draw + k).permutation(G), k]
        elif kind == 'expr':
            order = np.argsort(b[src].mean(0), kind='stable')            # genes from lowest to highest mean basal expression
            for k in range(3):
                v = np.empty(G, np.float32)
                v[order] = np.sort(mean_mark[:, k])
                mean_mark[:, k] = v
        elif kind != 'N1':
            raise ValueError(kind)
        return {c: np.concatenate([b[i][:, None], np.where(has[i][None, :], mean_mark, 0.0)], 1).astype(np.float32)
                for i, c in enumerate(ctx.cells)}
    return build


def run(ctx, y, n_perm=N_PERM):
    enc = 'rank_normal'
    fit = list(ctx.enc[enc]['cov_dt'])
    specs = {'FB': (cf.feature_builder(ctx, enc, 'FB', fit), ['full']), 'N1': (cf.feature_builder(ctx, enc, 'N1', fit), ['full']),
             'N1expr': (n1_variant(ctx, enc, fit, 'expr'), ['full'])}
    for d in range(n_perm):
        specs['N1perm_%d' % d] = (n1_variant(ctx, enc, fit, 'perm', d), ['full'])
    out, mu, level = cf.run_t1(ctx, y, specs, fit)
    known = level[ctx.dev_mask] <= 2
    sc = {k: cf.score(ctx, y, out[(k, 'full')]['y_hat_dev'], known) for k in specs}
    return {'scores': {k: {'all': v['all'], 'top': v['top'], 'per_cell': v['per_cell'], 'centred_all': v['centred_all']}
                       for k, v in sc.items()},
            'kappa': {k: [out[(k, 'full')]['kappa'], out[(k, 'full')]['kappa_d']] for k in specs},
            'fitting_cells': fit, 'n_perm': n_perm, 'dev_cells': list(cf.DEV_CELLS)}


def reading(res, n1_minus_fb_91):
    """93.15's mechanical reading. Refuses (HARNESS_FAULT) if N1 - FB does not reproduce 91.12."""
    s = res['scores']
    g = lambda k: s[k]['all'] - s['FB']['all']                         # noqa: E731
    g_chr, g_expr = g('N1'), g('N1expr')
    g_perm = [g('N1perm_%d' % d) for d in range(res['n_perm'])]
    cells = res['dev_cells']
    diff_cells = {c: (s['N1']['per_cell'][c] - s['FB']['per_cell'][c]) - (s['N1expr']['per_cell'][c] - s['FB']['per_cell'][c])
                  for c in cells}
    out = {'G_chr': g_chr, 'G_expr': g_expr, 'G_perm_max': max(g_perm), 'G_perm': g_perm, 'G_chr_minus_G_expr': g_chr - g_expr,
           'G_chr_minus_G_expr_per_cell': diff_cells, 'cells_chr_gt_expr': int(sum(v > 0 for v in diff_cells.values())),
           'harness': {'N1_minus_FB': g_chr, '91_12': n1_minus_fb_91, 'reproduces': bool(abs(g_chr - n1_minus_fb_91) < 1e-6)}}
    if not out['harness']['reproduces']:
        out['reading'] = 'HARNESS_FAULT'
    elif g_chr <= max(g_perm):
        out['reading'] = 'NOT DISTINGUISHABLE FROM CAPACITY'
    elif g_chr - g_expr > 0 and out['cells_chr_gt_expr'] >= 4:
        out['reading'] = 'CHROMATIN-SPECIFIC'
    else:
        out['reading'] = 'GENE-LEVEL, NOT CHROMATIN-SPECIFIC'
    return out


def read(d):
    m = json.load(open(os.path.join(d, MARKER)))
    p = os.path.join(d, OUT)
    got = hashlib.sha1(open(p, 'rb').read()).hexdigest()
    if not m.get('complete') or m['outputs'].get(OUT) != got:
        raise SystemExit('REFUSED: %s sha1 %s does not match the marker' % (OUT, got))
    sc91 = json.load(open(F91))['row_sets']['known']['scores']
    return reading(json.load(open(p)), sc91['N1']['all'] - sc91['FB']['all'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir')
    ap.add_argument('--provenance')
    ap.add_argument('--out')
    ap.add_argument('--read')
    a = ap.parse_args()
    if a.read:
        print(json.dumps(cf.jsonable(read(a.read)), indent=1))
        return
    t = time.time()
    ctx = cf.prepare(a.data_dir, a.provenance)
    assert len(ctx.enc['rank_normal']['cov_dt']) == 11
    res = run(ctx, ctx.y)
    res.update({'inputs': ctx.inputs, 'seconds': time.time() - t})
    json.dump(cf.jsonable(res), open(a.out, 'w'), indent=1)
    marker = {'complete': True, 'outputs': {os.path.basename(a.out): cf.sha1_file(a.out)}, 'inputs': ctx.inputs}
    json.dump(marker, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), MARKER), 'w'), indent=1)
    print('wrote', a.out, 'seconds %.0f' % res['seconds'], flush=True)          # no reading is echoed


if __name__ == '__main__':
    main()
