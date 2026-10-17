"""v10-C L3: per-(cell, gene) chromatin features from the data kernels' outputs (RESULTS 100.1, normalisation).

For each track and window: rank the 977 genes' window means (average ties; log1p first is rank-invariant), scale to
[0, 1]; average a slot's K tracks. Features: 3 axes x 5 windows + CCLE promoter methylation = 16 per gene, with a
4-channel availability mask (ACC, ACT, PCG, methylation). A missing value is 0 with mask 0.
"""
import csv, glob, json, os

import numpy as np

AXES = ('ACC', 'ACT', 'PCG')


def rank01(v):
    """Average-tie ranks of the finite entries of v scaled to [0, 1]; NaN stays NaN."""
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    x = v[ok]
    n = len(x)
    if n < 2:
        return out
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n)
    xs = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    out[ok] = ranks / (n - 1)
    return out


def track_features(values):
    """values [n_genes, n_windows] raw window means -> per-window ranks in [0, 1]."""
    return np.stack([rank01(values[:, w]) for w in range(values.shape[1])], 1)


def load_signal(paths):
    tracks, windows, genes = {}, None, None
    for p in paths:
        z = np.load(p)
        w, g = list(z['windows']), list(z['genes'])
        if windows is None:
            windows, genes = w, g
        elif w != windows or g != genes:
            raise SystemExit(f'FATAL: {p} has other windows or genes')
        for s, v in zip(z['srx'], z['values']):
            if str(s) in tracks:
                raise SystemExit(f'FATAL: track {s} in two shards')
            tracks[str(s)] = v.astype(np.float64)
    return tracks, windows, genes


def gene_map(tss_path, model_entrez):
    """Index of each TSS-table gene in the model's gene order, by Entrez (symbols differ: TOMM70A vs TOMM70)."""
    pos = {e: i for i, e in enumerate(model_entrez)}
    rows = list(csv.DictReader(open(tss_path, encoding='utf-8', newline=''), delimiter='\t'))
    return np.array([pos.get(r['entrez'], -1) for r in rows]), [r['symbol'] for r in rows]


def build(signal_paths, selection, cells, tss_path, model_entrez, methyl=None):
    tracks, windows, tss_genes = load_signal(signal_paths)
    gmap, syms = gene_map(tss_path, model_entrez)
    if syms != tss_genes:
        raise SystemExit('FATAL: the signal genes are not the TSS table genes')
    if (gmap < 0).any():
        raise SystemExit(f'FATAL: TSS genes without a model gene: {[s for s, k in zip(syms, gmap) if k < 0]}')
    nW, nG, nC = len(windows), len(model_entrez), len(cells)
    F = np.zeros((nC, nG, len(AXES) * nW + 1), np.float32)
    M = np.zeros((nC, nG, len(AXES) + 1), bool)
    used, missing_tracks = {}, []
    for c, cell in enumerate(cells):
        for a, ax in enumerate(AXES):
            srxs = [s for s in selection.get(cell, {}).get(ax, [])]
            have = [s for s in srxs if s in tracks]
            missing_tracks += [s for s in srxs if s not in tracks]
            if not have:
                continue
            r = np.nanmean(np.stack([track_features(tracks[s]) for s in have]), 0) if len(have) > 1 \
                else track_features(tracks[have[0]])
            ok = np.isfinite(r).all(1)
            F[c, gmap[ok], a * nW:(a + 1) * nW] = r[ok]
            M[c, gmap[ok], a] = True
            used[(cell, ax)] = have
    if methyl is not None:
        mc = {str(x): i for i, x in enumerate(methyl['cells'])}
        mg = {str(x): i for i, x in enumerate(methyl['genes'])}
        cols = np.array([mg.get(s, -1) for s in syms])
        for c, cell in enumerate(cells):
            if cell not in mc:
                continue
            v = np.full(len(syms), np.nan)
            v[cols >= 0] = methyl['values'][mc[cell], cols[cols >= 0]]
            ok = np.isfinite(v)
            F[c, gmap[ok], -1] = v[ok]
            M[c, gmap[ok], -1] = True
    names = [f'{ax}:{w}' for ax in AXES for w in windows] + ['METH:tss_1kb']
    if missing_tracks:
        raise SystemExit(f'FATAL: selected tracks with no signal: {sorted(set(missing_tracks))[:10]}')
    return F, M, names, used
