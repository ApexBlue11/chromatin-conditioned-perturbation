# -*- coding: utf-8 -*-
"""RESULTS 93.15 as amended by 93.15a (review 057) and 93.15b (review 057a): is §91's gene-generic chromatin gain (T1's N1 - FB,
+0.0012) chromatin-specific? PI glue on chromatin_funnel.

    python model/v9/chromatin_genegeneric.py --data_dir <dir> --provenance <json> --out <json>      # Kaggle: scores only
    python model/v9/chromatin_genegeneric.py --read DIR                                             # local: the one reading

Arms (one run_t1 call, §91's T1, drug-known dev rows):
  FB, N1 (§91); N1perm_d (mark means gene-permuted, ONE permutation per draw shared across marks, rng 9500 + d), d = 0..19;
  E = [b, x1, x2, x3]: over the fitting cells (held-out excluded) x1 = mean b, x2 = sd b, x3 = (mean b)^2, each quantile-matched
  to mark k's mean vector and present where the cell has mark k; E+N1 and E+perm_d add the (permuted) mark means to E.
  93.15b: the arms OF RECORD use the tie-corrected marks ('rank_normal_tie': every step10/step12 tie-break block tied before
  rank_normal, all three marks), with their own permutation null and E matched to the corrected means. The same arms on §91's
  features are reported, and §91's N1 - FB is the harness check.
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
# 93.16: step10 (ATAC, H3K27ac) and step12 (H3K27me3) ranked each mark jointly over all covered entries; the Z tied (no-peak /
# zero-coverage) entries hold ranks 0..Z-1 of N-1. ATAC and H3K27ac: Z exact from E_peaks_log. H3K27me3: step12's RAW tensor and
# log are not in the repo, so Z is the E_final estimate (the end of the per-cell value bands, review 057a; 93.15b).
TIE = {0: (6465, 31264), 1: (6726, 34195), 2: (22778, 25402)}


def _quantile_match(x, target):
    """x's ranks carry target's sorted values: same marginal as target, same ordering as x."""
    v = np.empty(len(x), np.float32)
    v[np.argsort(x, kind='stable')] = np.sort(target)
    return v


def variant(ctx, enc, fit_cells, kind, draw=None, ref_enc='rank_normal'):
    """Builder like cf.feature_builder N1. kind in {'N1', 'perm', 'E', 'E+N1', 'E+perm'}. Mark means are over the fitting
    cells with the mark, the held-out cell h excluded (as §91). E's columns are quantile-matched to ref_enc's mark means."""
    e, r = ctx.enc[enc], ctx.enc[ref_enc]
    b, has = e['b'], e['has']
    G = cf.G

    def means(ez, src):
        m = np.zeros((G, 3), np.float32)
        for k in range(3):
            idx = [i for i in src if has[i, k]]
            if idx:
                m[:, k] = ez[idx, :, k].mean(0)
        return m

    def build(h):
        src = [cf.cell_index(ctx, c) for c in fit_cells if c != h]
        M = means(e['Ez'], src)
        if kind in ('perm', 'E+perm'):
            M = M[np.random.default_rng(9500 + draw).permutation(G)]          # one permutation, all three marks
        cols = []
        if kind.startswith('E'):
            Mr = means(r['Ez'], src)
            bs = b[src].astype(np.float64)
            stats = (bs.mean(0), bs.std(0), bs.mean(0) ** 2)
            cols.append(np.stack([_quantile_match(stats[k], Mr[:, k]) for k in range(3)], 1))
        if kind in ('N1', 'perm', 'E+N1', 'E+perm'):
            cols.append(M)
        X = np.concatenate(cols, 1)
        mask = np.tile(has, (1, X.shape[1] // 3))
        return {c: np.concatenate([b[i][:, None], np.where(mask[i][None, :], X, 0.0)], 1).astype(np.float32)
                for i, c in enumerate(ctx.cells)}
    return build


def install_tie(ctx, E, cidx):
    """93.16, reported only: 'rank_normal_tie' = rank_normal of each present (cell, mark) after setting the step10 no-peak block
    (value <= (Z - 0.5) / (N - 1), ATAC and H3K27ac) to one tied value 0. b, has and the cell lists are rank_normal's."""
    r = ctx.enc['rank_normal']
    Ez = np.zeros_like(r['Ez'])
    for i, c in enumerate(ctx.cells):
        j = cidx.get(c)
        for k in range(3):
            if j is None or not r['has'][i, k]:
                continue
            v = E[j, :, k].astype(np.float64)
            if k in TIE:
                Z, N = TIE[k]
                v = np.where(v <= (Z - 0.5) / (N - 1), 0.0, v)
            Ez[i, :, k] = cf.rank_normal(v)
    ctx.enc['rank_normal_tie'] = dict(r, Ez=Ez.astype(np.float32))
    return ctx.enc['rank_normal_tie']


FEATURE_SETS = (('tie', 'rank_normal_tie'), ('v9', 'rank_normal'))   # 93.15b: 'tie' is of record; 'v9' (§91's) is reported


def run(ctx, y, n_perm=N_PERM, sets=FEATURE_SETS):
    """One run_t1 call. For each feature set f: N1_f, N1perm_f_d, E_f (matched to f's mark means), E_f+N1, E_f+perm_d. FB is
    shared. 'N1_v9' is §91's N1 (cf.feature_builder), the harness check."""
    fit = list(ctx.enc['rank_normal']['cov_dt'])
    specs = {'FB': cf.feature_builder(ctx, 'rank_normal', 'FB', fit)}
    for f, enc in sets:
        specs['N1_' + f] = (cf.feature_builder(ctx, enc, 'N1', fit) if enc == 'rank_normal' else variant(ctx, enc, fit, 'N1'))
        specs['E_' + f] = variant(ctx, enc, fit, 'E', ref_enc=enc)
        specs['E_%s+N1' % f] = variant(ctx, enc, fit, 'E+N1', ref_enc=enc)
        for d in range(n_perm):
            specs['N1perm_%s_%d' % (f, d)] = variant(ctx, enc, fit, 'perm', d)
            specs['E_%s+perm_%d' % (f, d)] = variant(ctx, enc, fit, 'E+perm', d, ref_enc=enc)
    out, mu, level = cf.run_t1(ctx, y, {k: (v, ['full']) for k, v in specs.items()}, fit)
    known = level[ctx.dev_mask] <= 2
    sc = {k: cf.score(ctx, y, out[(k, 'full')]['y_hat_dev'], known) for k in specs}
    return {'scores': {k: {'all': v['all'], 'top': v['top'], 'per_cell': v['per_cell'], 'centred_all': v['centred_all']}
                       for k, v in sc.items()},
            'kappa': {k: [out[(k, 'full')]['kappa'], out[(k, 'full')]['kappa_d']] for k in specs},
            'fitting_cells': fit, 'n_perm': n_perm, 'dev_cells': list(cf.DEV_CELLS), 'sets': [f for f, _ in sets]}


def reading_on(res, f):
    """93.15a's mechanical reading on feature set f (no harness check here)."""
    s, cells, n = res['scores'], res['dev_cells'], res['n_perm']
    d = lambda a, b: s[a]['all'] - s[b]['all']                                          # noqa: E731
    dc = lambda a, b: {c: s[a]['per_cell'][c] - s[b]['per_cell'][c] for c in cells}     # noqa: E731
    g_chr, g_perm = d('N1_' + f, 'FB'), [d('N1perm_%s_%d' % (f, k), 'FB') for k in range(n)]
    x_chr, x_perm = d('E_%s+N1' % f, 'E_' + f), [d('E_%s+perm_%d' % (f, k), 'E_' + f) for k in range(n)]
    x_cells = dc('E_%s+N1' % f, 'E_' + f)
    out = {'G_chr': g_chr, 'G_perm_max': max(g_perm), 'G_perm': g_perm, 'D_chr_given_E': x_chr, 'D_perm_given_E_max': max(x_perm),
           'D_perm_given_E': x_perm, 'D_chr_given_E_per_cell': x_cells,
           'cells_D_chr_given_E_pos': int(sum(v > 0 for v in x_cells.values())), 'G_chr_per_cell': dc('N1_' + f, 'FB'),
           'G_E': d('E_' + f, 'FB')}
    if g_chr <= max(g_perm):
        out['reading'] = 'NOT DISTINGUISHABLE FROM CAPACITY'
    elif x_chr <= max(x_perm):
        out['reading'] = 'GENE-LEVEL, MATCHED BY EXPRESSION (2a)'
    elif out['cells_D_chr_given_E_pos'] < 4:
        out['reading'] = 'GENE-LEVEL, EXCESS NOT ESTABLISHED ACROSS CELLS (2b)'
    else:
        out['reading'] = 'CHROMATIN-SPECIFIC'
    return out


def reading(res, n1_minus_fb_91):
    """93.15b: HARNESS_FAULT unless §91's N1 - FB reproduces 91.12; the reading of record is on the tie-corrected features;
    §91's features are read the same way and reported, with the artefact named."""
    g = res['scores']['N1_v9']['all'] - res['scores']['FB']['all']
    harness = {'N1_v9_minus_FB': g, '91_12': n1_minus_fb_91, 'reproduces': bool(abs(g - n1_minus_fb_91) < 1e-6)}
    out = {'harness': harness, 'caveat': '91.12: a per-gene parameter (e.g. v9 gene embedding) could represent it; not tested'}
    if not harness['reproduces']:
        out['reading'] = 'HARNESS_FAULT'
        return out
    rec = reading_on(res, 'tie')
    out.update({'reading': rec['reading'], 'of_record_tie_corrected': rec,
                'reported_v9_features': dict(reading_on(res, 'v9'), artefact='§91 features: ~21 % / 20 % / 90 % of ATAC / '
                                             'H3K27ac / H3K27me3 entries are step10/step12 tie-break codes (93.16)')})
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
    E = np.load(os.path.join(a.data_dir, 'E_final.npy'))
    Em = np.load(os.path.join(a.data_dir, 'E_final_mask.npy'))
    for k, (Z, N) in TIE.items():                    # the pinned block sizes describe THIS E_final
        assert int(Em[:, :, k].sum()) == N and int((E[:, :, k][Em[:, :, k]] <= (Z - 0.5) / (N - 1)).sum()) == Z, k
    cidx = json.load(open(os.path.join(a.data_dir, 'lincs_cell_index.json')))
    install_tie(ctx, E, cidx.get('cell_id_to_row', cidx))
    res = run(ctx, ctx.y)
    res.update({'inputs': ctx.inputs, 'seconds': time.time() - t})
    json.dump(cf.jsonable(res), open(a.out, 'w'), indent=1)
    marker = {'complete': True, 'outputs': {os.path.basename(a.out): cf.sha1_file(a.out)}, 'inputs': ctx.inputs}
    json.dump(marker, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), MARKER), 'w'), indent=1)
    print('wrote', a.out, 'seconds %.0f' % res['seconds'], flush=True)          # no reading is echoed


if __name__ == '__main__':
    main()
