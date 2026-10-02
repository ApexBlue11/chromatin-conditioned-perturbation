# -*- coding: utf-8 -*-
"""RESULTS 91 (pre-registered; amended 91.8, 91.9): the chromatin funnel -- closed-form CPU tests on the dev carve.

    python model/v9/chromatin_funnel.py --data_dir <dir> --provenance <E_final_provenance.json> --dti <chembl_dti_edges.tsv> --out <json>

T1  drug-conditioned gene-local chromatin rule   y_hat = mu*(1 + v_d.f) + w_d.f, w_d = w + delta_d, v_d = v + eps_d (hierarchical ridge)
T2  retrieval of other cells' responses, weighted by chromatin / basal-expression similarity
T3  which genes can move in a cell (per-gene response spread) from gene-local features
T0, M1, M3, M4 as 91.6 / 91.9. M2 (planted-effect power) is chromatin_power.py, which calls run_t1 / run_t3 below.

Rows are split_cold_cell_1 TRAIN rows only; the 4,043 dev rows (sha1 51e7e4ab...) never enter a fit, a mean pool or a
hyper-parameter choice. Every quantity derived from the targets (condition means, sufficient statistics, neighbour means, spreads)
is recomputed from the y passed in, so a planted copy of y is analysed exactly like the real one. PI-written (W27/W27b were
rejected: wrong fit set, LOCO leaks, missing outputs); reuses W27b's normal equations and hierarchical ridge, which were correct.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_dev  # noqa: E402
from score_dev import DEV_SHA1, centred_r, row_pearson  # noqa: E402

assert row_pearson is score_dev.row_pearson and centred_r is score_dev.centred_r   # the scorer of record, never a copy

DEV_CELLS = ('HEK293T', 'HL60', 'LNCAP', 'SKBR3', 'U937', 'VCAP')
N2_DONOR = {'HEK293T': 'HL60', 'HL60': 'LNCAP', 'LNCAP': 'SKBR3', 'SKBR3': 'U937', 'U937': 'VCAP', 'VCAP': 'HEK293T'}
EPI_TARGETS = re.compile(r'^(HDAC\d+|BRD[234]|EZH2|EED|SUZ12|DNMT\w*|KDM\w+|EP300|CREBBP|DOT1L|SIRT\d)$')
KAPPAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KAPPA_DS = (1e-2, 1e-1, 1.0, 10.0, np.inf)
TAUS = (0.01, 0.03, 0.1, 0.3, 1.0, np.inf)
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MIN_DRUG_CELLS = 3
P2_DEV_MEAN = 0.43693
G = 978


# ----------------------------------------------------------------------------------------------------------------- data
class Ctx:
    """Everything that does not depend on y."""


def sha1_file(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def load_train_rows(data_dir):
    z = np.load(os.path.join(data_dir, 'xpert_mdmt_splits.npz'), allow_pickle=True)
    lab = z['split_split_cold_cell_1']
    tr = np.flatnonzero(lab == 'train')
    assert len(tr) == 47509, 'expected 47,509 train rows, got %d' % len(tr)
    assert not np.any(lab[tr] == 'test')
    rows = {'X': np.asarray(z['X'][tr], np.float32), 'C': np.asarray(z['X_ctl'][tr], np.float32),
            'pert': np.asarray(z['meta_pert_id'][tr]).astype(str), 'dose': np.asarray(z['meta_dose'][tr], np.float64),
            'time': np.asarray(z['meta_time'][tr], np.float64), 'cell': np.asarray(z['meta_cell'][tr]).astype(str),
            'row_index': np.asarray(z['row_index'][tr]).astype(np.int64)}
    is_dev = np.isin(rows['cell'], DEV_CELLS)
    assert is_dev.sum() == 4043, 'expected 4,043 dev rows, got %d' % is_dev.sum()
    sha = hashlib.sha1(np.sort(rows['row_index'][is_dev]).tobytes()).hexdigest()
    assert sha == DEV_SHA1, 'not the RESULTS 85.4 dev rows: %s' % sha
    return rows, is_dev


def rank_normal(v):
    return norm.ppf((rankdata(v, method='average') - 0.5) / len(v))


def gene_features(cells, C_by_cell, E, Em, cidx, failed, encoding):
    """b [n_cells, G]: basal expression z-scored across genes within the cell. Ez [n_cells, G, 3], has_mark [n_cells, 3].
    A mark is present iff the mask has it and (primary encoding) it is not a failed-ChIP H3K27me3 track (91.8 item 1).
    'rank_normal' (primary): rank-based inverse-normal within (cell, mark). 'v9': xpert_arm.py 190-194, failed tracks kept."""
    n = len(cells)
    b = np.zeros((n, G), np.float32)
    Ez = np.zeros((n, G, 3), np.float32)
    has = np.zeros((n, 3), bool)
    for i, c in enumerate(cells):
        if c in C_by_cell:
            m = C_by_cell[c].mean(0)
            b[i] = (m - m.mean()) / (m.std() + 1e-6)
        j = cidx.get(c)
        if j is None:
            continue
        for k in range(3):
            if not Em[j, :, k].any():
                continue
            if encoding == 'rank_normal' and k == 2 and c in failed:
                continue
            v = E[j, :, k].astype(np.float64)
            if encoding == 'rank_normal':
                Ez[i, :, k] = rank_normal(v)
            elif encoding == 'v9':
                Ez[i, :, k] = (v - v.mean()) / (v.std() + 1e-6)
            else:
                raise ValueError(encoding)
            has[i, k] = True
    return b, Ez, has


def ctx_from_arrays(rows, is_dev):
    """The y-independent row bookkeeping shared by prepare() and the tests: cell / drug codes and the four condition-key levels,
    finest first: (pert, dose, time) -> (pert, time) -> pert -> global."""
    ctx = Ctx()
    ctx.rows, ctx.dev_mask = rows, np.asarray(is_dev, bool)
    ctx.fit_mask = ~ctx.dev_mask
    ctx.y = (rows['X'] - rows['C']).astype(np.float32)
    ctx.cells = sorted(set(rows['cell']))
    ctx.cell_id = np.searchsorted(ctx.cells, rows['cell']).astype(np.int64)
    ctx.perts = sorted(set(rows['pert']))
    ctx.pert_id = np.searchsorted(ctx.perts, rows['pert']).astype(np.int64)
    k0 = pd.factorize(pd.Series(rows['pert']) + '|' + pd.Series(rows['dose']).map(repr) + '|' + pd.Series(rows['time']).map(repr))[0]
    k1 = pd.factorize(pd.Series(rows['pert']) + '|' + pd.Series(rows['time']).map(repr))[0]
    ctx.levels = [k0.astype(np.int64), k1.astype(np.int64), ctx.pert_id.copy(), np.zeros(len(ctx.dev_mask), np.int64)]
    ctx.epi_perts = set()
    return ctx


def set_encodings(ctx, enc_features, lineage):
    """enc_features: {encoding: (b, Ez, has)} over ctx.cells. Derives the covered dev-train / dev cell lists."""
    ctx.enc = {}
    for enc, (b, Ez, has) in enc_features.items():
        cov = [c for i, c in enumerate(ctx.cells) if has[i].any()]
        ctx.enc[enc] = {'b': b, 'Ez': Ez, 'has': has, 'cov_dt': [c for c in cov if c not in DEV_CELLS],
                        'cov_dev': [c for c in cov if c in DEV_CELLS]}
    ctx.b = ctx.enc['rank_normal']['b']
    ctx.lineage = lineage
    return ctx


def prepare(data_dir, provenance, dti=None):
    t0 = time.time()
    rows, is_dev = load_train_rows(data_dir)
    ctx = ctx_from_arrays(rows, is_dev)
    failed = set(json.load(open(provenance))['h3k27me3_failed_chip_downweighted'])
    cidx = json.load(open(os.path.join(data_dir, 'lincs_cell_index.json')))
    cidx = cidx.get('cell_id_to_row', cidx)
    E = np.load(os.path.join(data_dir, 'E_final.npy')).astype(np.float32)
    Em = np.load(os.path.join(data_dir, 'E_final_mask.npy'))
    lin = np.load(os.path.join(data_dir, 'cell_lineage.npy')).astype(np.float32)
    C_by_cell = {c: rows['C'][ctx.cell_id == i] for i, c in enumerate(ctx.cells)}
    set_encodings(ctx, {enc: gene_features(ctx.cells, C_by_cell, E, Em, cidx, failed, enc) for enc in ('rank_normal', 'v9')},
                  np.stack([lin[cidx[c]] if c in cidx else np.full(lin.shape[1], np.nan, np.float32) for c in ctx.cells]))
    ctx.failed = sorted(failed)
    if dti and os.path.exists(dti):
        d = pd.read_csv(dti, sep='\t', usecols=['pert_id', 'gene_symbol']).dropna()
        ctx.epi_perts = set(d.loc[d['gene_symbol'].astype(str).str.match(EPI_TARGETS), 'pert_id'].astype(str))
    ctx.inputs = {os.path.basename(p): sha1_file(p) for p in
                  [os.path.join(data_dir, f) for f in ('xpert_mdmt_splits.npz', 'E_final.npy', 'E_final_mask.npy',
                                                      'lincs_cell_index.json', 'cell_lineage.npy')] + [provenance] + ([dti] if dti else [])
                  if os.path.exists(p)}
    ctx.prepare_seconds = time.time() - t0
    return ctx


def cell_index(ctx, c):
    return ctx.cells.index(c)


def assert_no_dev(ctx, mask_or_idx, where):
    m = np.zeros(len(ctx.dev_mask), bool)
    m[mask_or_idx] = True
    assert not (m & ctx.dev_mask).any(), 'GUARD: a dev row entered %s' % where


# ------------------------------------------------------------------------------------------------------- condition means
def group_sum(codes, values, n_groups):
    A = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(n_groups, len(codes)))
    return np.asarray(A @ values)


def condition_means(ctx, y, pool):
    """mu[i] = mean over the pool's cells c' != cell(i) of (c's mean y over pool rows sharing row i's key), at the finest level with
    >= 1 such cell; each cell counts once. Rows whose cell is not in the pool use every pool cell. Returns (mu, level)."""
    assert_no_dev(ctx, pool, 'a condition-mean pool')
    n = len(y)
    nc = len(ctx.cells)
    mu = np.zeros((n, y.shape[1]), np.float32)
    level = np.full(n, -1, np.int8)
    todo = np.ones(n, bool)
    pidx = np.flatnonzero(pool)
    for L, codes in enumerate(ctx.levels):
        if not todo.any():
            break
        kc = codes[pidx] * nc + ctx.cell_id[pidx]
        ukc, ginv = np.unique(kc, return_inverse=True)
        cnt = np.bincount(ginv, minlength=len(ukc)).astype(np.float32)
        M = group_sum(ginv, y[pidx], len(ukc)) / cnt[:, None]
        ukey, kinv = np.unique(ukc // nc, return_inverse=True)
        Ksum = group_sum(kinv, M, len(ukey))
        Kcnt = np.bincount(kinv, minlength=len(ukey))
        r = np.flatnonzero(todo)
        rk = codes[r]
        pos = np.minimum(np.searchsorted(ukey, rk), len(ukey) - 1)
        haskey = ukey[pos] == rk
        rkc = rk * nc + ctx.cell_id[r]
        gpos = np.minimum(np.searchsorted(ukc, rkc), len(ukc) - 1)
        own = (ukc[gpos] == rkc) & haskey
        ncell = np.where(haskey, Kcnt[pos], 0) - own
        ok = ncell > 0
        S = Ksum[pos[ok]].copy()
        o = own[ok]
        S[o] -= M[gpos[ok][o]]
        mu[r[ok]] = S / ncell[ok][:, None]
        level[r[ok]] = L
        todo[r[ok]] = False
    assert not todo.any(), 'condition mean unresolved for %d rows' % todo.sum()
    return mu, level


# ------------------------------------------------------------------------------------------------------------------- T1
def suff_stats(ctx, y, mu, rows):
    """Per (cell, drug) group over the given fit rows: n, sum mu, sum mu^2, sum t, sum mu*t with t = y - mu (per gene)."""
    assert_no_dev(ctx, rows, 'a T1 fit')
    npt = len(ctx.perts)
    cp = ctx.cell_id[rows].astype(np.int64) * npt + ctx.pert_id[rows]
    ucp, inv = np.unique(cp, return_inverse=True)
    m = mu[rows]
    t = y[rows] - m
    return {'cell': (ucp // npt).astype(int), 'pert': (ucp % npt).astype(int),
            'n': np.bincount(inv, minlength=len(ucp)).astype(np.float64),
            'S1': group_sum(inv, m, len(ucp)), 'S2': group_sum(inv, m * m, len(ucp)),
            'T0': group_sum(inv, t, len(ucp)), 'T1': group_sum(inv, m * t, len(ucp))}


def build_HB(ctx, ss, F):
    """Normal equations per drug for regressors [f, mu*f] (f = F[cell], [G, J]). Returns H [D, 2J, 2J], B [D, 2J], cells per drug."""
    D = len(ctx.perts)
    J = next(iter(F.values())).shape[1]
    H = np.zeros((D, 2 * J, 2 * J))
    B = np.zeros((D, 2 * J))
    ncell = np.zeros(D, int)
    for ci in np.unique(ss['cell']):
        f = np.asarray(F[ctx.cells[ci]], np.float64)
        sel = ss['cell'] == ci
        p = ss['pert'][sel]
        n = ss['n'][sel]
        S1, S2, T0, T1 = (ss[k][sel].astype(np.float64) for k in ('S1', 'S2', 'T0', 'T1'))
        Hg = np.zeros((sel.sum(), 2 * J, 2 * J))
        Hg[:, :J, :J] = n[:, None, None] * (f.T @ f)[None]
        H01 = np.einsum('gj,ng,gk->njk', f, S1, f, optimize=True)
        Hg[:, :J, J:] = H01
        Hg[:, J:, :J] = H01
        Hg[:, J:, J:] = np.einsum('gj,ng,gk->njk', f, S2, f, optimize=True)
        Bg = np.concatenate([T0 @ f, T1 @ f], 1)
        np.add.at(H, p, Hg)
        np.add.at(B, p, Bg)
        np.add.at(ncell, p, 1)
    return H, B, ncell


def solve_hier(H, B, ncell, lam_g, lam_d):
    """min sum_d ||t_d - X_d(theta + delta_d)||^2 + lam_g|theta|^2 + lam_d sum|delta_d|^2; delta_d = 0 for drugs below the
    cell floor or when lam_d is infinite (profiled closed form)."""
    D, K, _ = H.shape
    I = np.eye(K)
    act = (ncell >= MIN_DRUG_CELLS) & np.isfinite(lam_d)
    Ht, Bt = H.sum(0), B.sum(0)
    inv = None
    if act.any():
        Ha, Ba = H[act], B[act]
        inv = np.linalg.inv(Ha + lam_d * I)
        Fa = Ha @ inv
        Ht = Ht - np.einsum('dij,djk->ik', Fa, Ha)
        Bt = Bt - np.einsum('dij,dj->i', Fa, Ba)
    theta = np.linalg.solve(Ht + lam_g * I, Bt)
    delta = np.zeros((D, K))
    if act.any():
        delta[act] = np.einsum('dij,dj->di', inv, Ba - Ha @ theta)
    return theta, delta


def t1_solve(H, B, ncell, N, variant, kappa, kappa_d):
    J = H.shape[1] // 2
    if variant == 'additive':                                   # regressors f only, no per-drug deviation
        th = np.linalg.solve(H[:, :J, :J].sum(0) + kappa * N * np.eye(J), B[:, :J].sum(0))
        return np.concatenate([th, np.zeros(J)]), np.zeros((H.shape[0], 2 * J))
    kd = np.inf if variant == 'global' else kappa_d
    return solve_hier(H, B, ncell, kappa * N, kd * N)


def t1_predict(ctx, mu_rows, rows, F, theta, delta):
    """y_hat = mu + f.(theta_w + delta_w[d]) + mu * f.(theta_v + delta_v[d]); unseen or below-floor drugs have delta = 0."""
    J = len(theta) // 2
    out = mu_rows.astype(np.float64).copy()
    cid = ctx.cell_id[rows]
    for ci in np.unique(cid):
        sel = cid == ci
        f = np.asarray(F[ctx.cells[ci]], np.float64)
        d = ctx.pert_id[rows[sel]]
        W = theta[:J][None] + delta[d, :J]
        V = theta[J:][None] + delta[d, J:]
        out[sel] += W @ f.T + mu_rows[sel] * (V @ f.T)
    return out


def grid(variant):
    if variant == 'full':
        return [(k, kd) for k in KAPPAS for kd in KAPPA_DS]
    return [(k, np.inf) for k in KAPPAS]


def run_t1(ctx, y, specs, fit_cells):
    """specs: {name: (builder, [variants])}; builder(held_out_cell_or_None) -> {cell: [G, J]} for every fit and dev cell.
    LOCO over fit_cells chooses (kappa, kappa_d) per (spec, variant); then refit on all fit cells and predict the dev rows.
    Returns {(name, variant): {'y_hat_dev', 'kappa', 'kappa_d', 'loco'}} and the final mu of the dev rows."""
    assert not set(fit_cells) & set(DEV_CELLS)
    fold_scores = {}
    for h in fit_cells:
        pool = ctx.fit_mask & (ctx.cell_id != cell_index(ctx, h))
        mu_h, _ = condition_means(ctx, y, pool)
        fit_rows = np.flatnonzero(ctx.fit_mask & np.isin(ctx.rows['cell'], [c for c in fit_cells if c != h]))
        ho_rows = np.flatnonzero(ctx.fit_mask & (ctx.rows['cell'] == h))
        ss = suff_stats(ctx, y, mu_h, fit_rows)
        N = ss['n'].sum() * G
        for name, (builder, variants) in specs.items():
            F = builder(h)
            H, B, nc = build_HB(ctx, ss, F)
            for var in variants:
                for kk in grid(var):
                    th, de = t1_solve(H, B, nc, N, var, *kk)
                    yh = t1_predict(ctx, mu_h[ho_rows], ho_rows, F, th, de)
                    fold_scores.setdefault((name, var, kk), []).append(float(np.nanmean(row_pearson(yh, y[ho_rows]))))
        del mu_h, ss
    mu, level = condition_means(ctx, y, ctx.fit_mask)
    fit_rows = np.flatnonzero(ctx.fit_mask & np.isin(ctx.rows['cell'], fit_cells))
    dev_rows = np.flatnonzero(ctx.dev_mask)
    ss = suff_stats(ctx, y, mu, fit_rows)
    N = ss['n'].sum() * G
    out = {}
    for name, (builder, variants) in specs.items():
        F = builder(None)
        H, B, nc = build_HB(ctx, ss, F)
        for var in variants:
            cand = {kk: float(np.mean(v)) for (nm, vv, kk), v in fold_scores.items() if nm == name and vv == var}
            kk = max(cand, key=cand.get)
            th, de = t1_solve(H, B, nc, N, var, *kk)
            out[(name, var)] = {'y_hat_dev': t1_predict(ctx, mu[dev_rows], dev_rows, F, th, de), 'kappa': kk[0],
                                'kappa_d': kk[1], 'loco': cand[kk], 'theta': th.tolist(), 'H': H, 'B': B, 'nc': nc,
                                'N': N, 'delta': de}
    return out, mu, level


def feature_builder(ctx, enc, kind, fit_cells):
    """Gene-local features per cell [G, J]. FB=[b]; FBC=[b, Ez]; FBC_perp: Ez residualised on b within the cell; FC=[Ez];
    N1=[b, mean Ez over fit cells having the mark (LOCO: excluding the held-out cell), only for marks the cell has]."""
    e = ctx.enc[enc]
    b, Ez, has = e['b'], e['Ez'], e['has']

    def build(h):
        out = {}
        if kind == 'N1':
            src = [cell_index(ctx, c) for c in fit_cells if c != h]
            mean_mark = np.zeros((G, 3), np.float32)
            for k in range(3):
                idx = [i for i in src if has[i, k]]
                if idx:
                    mean_mark[:, k] = Ez[idx, :, k].mean(0)
        for i, c in enumerate(ctx.cells):
            if kind == 'FB':
                f = b[i][:, None]
            elif kind == 'FC':
                f = Ez[i]
            elif kind == 'FBC':
                f = np.concatenate([b[i][:, None], Ez[i]], 1)
            elif kind == 'FBC_perp':
                z = Ez[i].copy()
                for k in range(3):
                    if has[i, k]:
                        z[:, k] -= (b[i] @ z[:, k]) / (b[i] @ b[i] + 1e-8) * b[i]
                f = np.concatenate([b[i][:, None], z], 1)
            elif kind == 'N1':
                f = np.concatenate([b[i][:, None], np.where(has[i][None, :], mean_mark, 0.0)], 1)
            else:
                raise ValueError(kind)
            out[c] = f.astype(np.float32)
        return out
    return build


def n2_features(ctx, enc):
    """FBC features with each dev cell given its donor's chromatin (marks the donor lacks are 0); training cells unchanged."""
    e = ctx.enc[enc]
    F = feature_builder(ctx, enc, 'FBC', [])(None)
    for c, donor in N2_DONOR.items():
        i, j = cell_index(ctx, c), cell_index(ctx, donor)
        F[c] = np.concatenate([e['b'][i][:, None], np.where(e['has'][j][None, :], e['Ez'][j], 0.0)], 1).astype(np.float32)
    return F


# ------------------------------------------------------------------------------------------------------------------- T2
def similarities(ctx, enc, cov_dt):
    """s_B (basal), s_C (chromatin, with 91.8's no-shared-mark rule applied per target over its neighbours), lineage match."""
    e = ctx.enc[enc]
    n = len(ctx.cells)
    sB = np.corrcoef(e['b'])
    sC = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            sh = e['has'][i] & e['has'][j]
            if i != j and sh.any():
                sC[i, j] = np.mean([np.corrcoef(e['Ez'][i, :, k], e['Ez'][j, :, k])[0, 1] for k in np.flatnonzero(sh)])
    nb = [cell_index(ctx, c) for c in cov_dt]
    n_noshare = 0
    for i in range(n):
        others = [j for j in nb if j != i]
        vals = [sC[i, j] for j in others if np.isfinite(sC[i, j])]
        med = float(np.median(vals)) if vals else 0.0
        for j in others:
            if not np.isfinite(sC[i, j]):
                sC[i, j] = med
                n_noshare += ctx.cells[i] in DEV_CELLS
    sC = np.nan_to_num(sC)
    lin = ctx.lineage
    sL = np.array([[float(np.all(np.isfinite(lin[i])) and np.array_equal(lin[i], lin[j])) for j in range(n)] for i in range(n)])
    return {'s_B': sB, 's_C': sC, 'lineage': sL}, n_noshare


def neighbour_table(ctx, y, cov_dt, query_rows):
    """For each query row: the finest level at which >= 1 covered dev-train cell other than the row's own has the key, those
    cells, and their within-cell means of y at that level (pool = covered dev-train rows only). Independent of the weights."""
    pool = ctx.fit_mask & np.isin(ctx.rows['cell'], cov_dt)
    assert_no_dev(ctx, pool, 'a T2 neighbour pool')
    nc = len(ctx.cells)
    pidx = np.flatnonzero(pool)
    nbr_cell = np.full((len(query_rows), len(cov_dt)), -1, int)
    nbr_mean = np.zeros((len(query_rows), len(cov_dt), y.shape[1]), np.float32)
    todo = np.ones(len(query_rows), bool)
    for L, codes in enumerate(ctx.levels):
        if not todo.any():
            break
        kc = codes[pidx] * nc + ctx.cell_id[pidx]
        ukc, ginv = np.unique(kc, return_inverse=True)
        M = group_sum(ginv, y[pidx], len(ukc)) / np.bincount(ginv, minlength=len(ukc)).astype(np.float32)[:, None]
        by_key = {}
        for g, v in enumerate(ukc):
            by_key.setdefault(int(v // nc), []).append((int(v % nc), g))
        for q in np.flatnonzero(todo):
            r = query_rows[q]
            lst = [(c, g) for c, g in by_key.get(int(codes[r]), []) if c != ctx.cell_id[r]]
            if lst:
                for s, (c, g) in enumerate(lst):
                    nbr_cell[q, s] = c
                    nbr_mean[q, s] = M[g]
                todo[q] = False
    assert not todo.any()
    return nbr_cell, nbr_mean


def retrieve(ctx, query_rows, nbr_cell, nbr_mean, S, tau):
    qc = ctx.cell_id[query_rows]
    valid = nbr_cell >= 0
    s = np.where(valid, S[qc[:, None], np.maximum(nbr_cell, 0)], -np.inf)
    if np.isinf(tau):
        w = valid.astype(np.float64)
    else:
        w = np.where(valid, np.exp((s - np.max(s, 1, keepdims=True)) / tau), 0.0)
    w /= w.sum(1, keepdims=True)
    out = np.zeros((len(query_rows), nbr_mean.shape[2]))
    for k in range(nbr_mean.shape[1]):
        out += w[:, k:k + 1] * nbr_mean[:, k]
    return out


def run_t2(ctx, y, enc='rank_normal'):
    """Each similarity variant chooses its own tau (and beta for s_BC) by LOCO over the covered dev-train cells."""
    cov_dt = ctx.enc[enc]['cov_dt']
    sims, n_noshare = similarities(ctx, enc, cov_dt)
    dev_rows = np.flatnonzero(ctx.dev_mask)
    dev_nb = neighbour_table(ctx, y, cov_dt, dev_rows)
    loco_rows = {c: np.flatnonzero(ctx.fit_mask & (ctx.rows['cell'] == c)) for c in cov_dt}
    loco_nb = {c: neighbour_table(ctx, y, cov_dt, loco_rows[c]) for c in cov_dt}
    variants = {'s_B': [(sims['s_B'], t, 0.0) for t in TAUS], 's_C': [(sims['s_C'], t, 0.0) for t in TAUS],
                's_BC': [(sims['s_B'] + bt * sims['s_C'], t, bt) for t in TAUS for bt in BETAS],
                'lineage': [(sims['lineage'], t, 0.0) for t in TAUS], 'uniform': [(sims['s_B'], np.inf, 0.0)]}
    out = {}
    for name, cands in variants.items():
        best = None
        for S, tau, bt in cands:
            sc = np.mean([np.nanmean(row_pearson(retrieve(ctx, loco_rows[c], *loco_nb[c], S, tau), y[loco_rows[c]]))
                          for c in cov_dt])
            if best is None or sc > best[0]:
                best = (sc, S, tau, bt)
        out[name] = {'y_hat_dev': retrieve(ctx, dev_rows, *dev_nb, best[1], best[2]), 'tau': best[2], 'beta': best[3],
                     'loco': float(best[0])}
    mu_all, _ = condition_means(ctx, y, ctx.fit_mask)
    out['B0_26'] = {'y_hat_dev': mu_all[dev_rows], 'tau': None, 'beta': None, 'loco': None}
    return out, n_noshare


# ------------------------------------------------------------------------------------------------------------------- T3
def spreads(ctx, y, cells, min_rows=20):
    v, skipped = {}, []
    for c in cells:
        m = ctx.rows['cell'] == c
        if m.sum() < min_rows:
            skipped.append(c)
            continue
        s = y[m].std(0)
        v[c] = (s - s.mean()) / (s.std() + 1e-6)
    return v, skipped


def t3_fit_predict(v, F, train_cells, target_cells, kappa):
    """Fit v[c] ~ a*vbar + beta.F[c] + intercept on train_cells only (vbar = their mean spread) and predict target_cells.
    Nothing about a target cell enters the fit unless it is also a train cell."""
    vbar = np.mean([v[c] for c in train_cells], 0)

    def design(c):
        return np.column_stack([vbar, F[c], np.ones(G)]) if F is not None else np.column_stack([vbar, np.ones(G)])
    X = np.vstack([design(c) for c in train_cells])
    yv = np.concatenate([v[c] for c in train_cells])
    th = np.linalg.solve(X.T @ X + kappa * len(yv) * np.eye(X.shape[1]), X.T @ yv)
    return {c: design(c) @ th for c in target_cells}


def run_t3(ctx, y, builders, fit_cells, eval_cells):
    """v_hat[c,g] = a*vbar_g + beta.f[c,g] + intercept, shared over genes and cells; vbar_g and N1's means exclude the held-out
    cell inside LOCO. builders: {name: builder or None (gene prior only)}. Returns {name: {'rho': {cell: r}, 'kappa'}}."""
    assert not set(fit_cells) & set(DEV_CELLS)
    v, skipped = spreads(ctx, y, list(fit_cells) + list(eval_cells))
    fit = [c for c in fit_cells if c in v]
    out = {}
    for name, builder in builders.items():
        best = None
        for k in KAPPAS:
            sc = [np.corrcoef(t3_fit_predict(v, builder(h) if builder else None, [c for c in fit if c != h], [h], k)[h], v[h])[0, 1]
                  for h in fit]
            if best is None or np.mean(sc) > best[0]:
                best = (float(np.mean(sc)), k)
        pred = t3_fit_predict(v, builder(None) if builder else None, fit, [c for c in eval_cells if c in v], best[1])
        out[name] = {'rho': {c: float(np.corrcoef(p, v[c])[0, 1]) for c, p in pred.items()}, 'kappa': best[1], 'loco': best[0]}
    return out, skipped


# ------------------------------------------------------------------------------------------------------------- T0 / M1
def t0(ctx):
    out = {}
    for enc, e in ctx.enc.items():
        within = {c: {m: float(np.corrcoef(e['Ez'][i, :, k], e['b'][i])[0, 1])
                      for k, m in enumerate(('ATAC', 'H3K27ac', 'H3K27me3')) if e['has'][i, k]}
                  for i, c in enumerate(ctx.cells) if e['has'][i].any()}
        dup = []
        cov = [i for i, c in enumerate(ctx.cells) if e['has'][i].any()]
        for k, m in enumerate(('ATAC', 'H3K27ac', 'H3K27me3')):
            for a in cov:
                for bb in cov:
                    if a < bb and e['has'][a, k] and e['has'][bb, k]:
                        r = float(np.corrcoef(e['Ez'][a, :, k], e['Ez'][bb, :, k])[0, 1])
                        if r > 0.99:
                            dup.append([ctx.cells[a], ctx.cells[bb], m, r])
        out[enc] = {'within_cell_r_with_basal': within, 'pairs_r_gt_0.99': dup, 'cov_dt': e['cov_dt'], 'cov_dev': e['cov_dev'],
                    'marks_dev': {c: [m for k, m in enumerate(('ATAC', 'H3K27ac', 'H3K27me3')) if e['has'][cell_index(ctx, c), k]]
                                  for c in DEV_CELLS}}
    return out


def m1(ctx, y, mu, K=10, chunk=512):
    """(a) dose-neighbour agreement and (b) within-cell drug-neighbour predictability of e = y - mu^(-c), dev-train rows only."""
    dt = np.flatnonzero(ctx.fit_mask)
    e = (y[dt] - mu[dt]).astype(np.float64)
    df = pd.DataFrame({'c': ctx.cell_id[dt], 'p': ctx.pert_id[dt], 't': ctx.rows['time'][dt], 'd': ctx.rows['dose'][dt],
                       'i': np.arange(len(dt))}).sort_values(['c', 'p', 't', 'd'])
    a_ = df['i'].values
    same = (df['c'].values[1:] == df['c'].values[:-1]) & (df['p'].values[1:] == df['p'].values[:-1]) & \
           (df['t'].values[1:] == df['t'].values[:-1])
    i1, i2 = a_[:-1][same], a_[1:][same]
    ra = row_pearson(e[i1], e[i2])
    ca = ctx.cell_id[dt][i1]
    out_a = {'mean': float(np.nanmean(ra)), 'median': float(np.nanmedian(ra)), 'n_pairs': int(len(ra)),
             'per_cell': {ctx.cells[c]: float(np.nanmean(ra[ca == c])) for c in np.unique(ca)}}
    rb, cb = [], []
    for c in np.unique(ctx.cell_id[dt]):
        idx = np.flatnonzero(ctx.cell_id[dt] == c)
        if len(idx) <= K:
            continue
        m = mu[dt][idx].astype(np.float64)
        m = m - m.mean(1, keepdims=True)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
        p = ctx.pert_id[dt][idx]
        for s in range(0, len(idx), chunk):
            sim = m[s:s + chunk] @ m.T
            sim[p[s:s + chunk][:, None] == p[None, :]] = -np.inf
            kk = min(K, int(np.isfinite(sim).sum(1).min()))
            if kk == 0:
                continue
            top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            eh = e[idx][top].mean(1)
            rb.append(row_pearson(eh, e[idx[s:s + chunk]]))
            cb.append(np.full(len(eh), c))
    rb, cb = np.concatenate(rb), np.concatenate(cb)
    out_b = {'mean': float(np.nanmean(rb)), 'per_cell': {ctx.cells[c]: float(np.nanmean(rb[cb == c])) for c in np.unique(cb)}}
    return out_a, out_b


# -------------------------------------------------------------------------------------------------------------- scoring
def score(ctx, y, y_hat, sub=None):
    """Scores over the dev rows, or over the subset `sub` (a boolean mask over the dev rows, in dev-row order): per-row Pearson
    mean, top tercile of |y| (terciles within the scored rows), per cell, cell-centred (centring within the scored rows)."""
    dev = np.flatnonzero(ctx.dev_mask)
    y_hat = np.asarray(y_hat, np.float64)
    if sub is not None:
        dev, y_hat = dev[sub], y_hat[sub]
    yt = y[dev].astype(np.float64)
    r = row_pearson(y_hat, yt)
    cells = ctx.rows['cell'][dev]
    norms = np.linalg.norm(yt, axis=1)
    top = norms > np.percentile(norms, 100 * 2 / 3)
    rc = centred_r({'pred': y_hat, 'true': yt}, cells)
    epi = np.isin(ctx.rows['pert'][dev], list(ctx.epi_perts))
    pc = lambda v: {c: float(np.nanmean(v[cells == c])) for c in DEV_CELLS}   # noqa: E731
    return {'r': r, 'rc': rc, 'all': float(np.nanmean(r)), 'top': float(np.nanmean(r[top])), 'per_cell': pc(r),
            'centred_all': float(np.nanmean(rc)), 'centred_per_cell': pc(rc),
            'epi': float(np.nanmean(r[epi])) if epi.any() else None, 'rest': float(np.nanmean(r[~epi])),
            'n_epi_rows': int(epi.sum()), 'n_nan_rows': int(np.isnan(r).sum())}


def paired(a, b):
    """a - b: all-row mean, top tercile, per cell, cells > 0, centred per cell, centred cells > 0."""
    d = {c: a['per_cell'][c] - b['per_cell'][c] for c in DEV_CELLS}
    dc = {c: a['centred_per_cell'][c] - b['centred_per_cell'][c] for c in DEV_CELLS}
    return {'all': a['all'] - b['all'], 'top': a['top'] - b['top'], 'per_cell': d, 'cells_pos': int(sum(v > 0 for v in d.values())),
            'centred_all': a['centred_all'] - b['centred_all'], 'centred_per_cell': dc,
            'centred_cells_pos': int(sum(v > 0 for v in dc.values()))}


def t1_reading(s_fbc, s_fb, s_n1):
    d, dn = paired(s_fbc, s_fb), paired(s_fbc, s_n1)
    conj = {'delta_all_ge_0.003_or_top_ge_0.006': bool(d['all'] >= 0.003 or d['top'] >= 0.006),
            'cells_delta_pos_ge_4': bool(d['cells_pos'] >= 4), 'cells_FBC_gt_N1_ge_4': bool(dn['cells_pos'] >= 4),
            'cells_centred_pos_ge_4': bool(d['centred_cells_pos'] >= 4)}
    return {'delta': d, 'vs_N1': dn, 'conjuncts': conj, 'pass': bool(all(conj.values()))}


def t2_reading(s_bc, s_b, s_c, s_u):
    d, du = paired(s_bc, s_b), paired(s_c, s_u)
    conj = {'delta_T2_all_ge_0.002': bool(d['all'] >= 0.002), 'sC_minus_uniform_all_ge_0.003': bool(du['all'] >= 0.003),
            'cells_delta_T2_pos_ge_4': bool(d['cells_pos'] >= 4), 'cells_sC_minus_uniform_pos_ge_4': bool(du['cells_pos'] >= 4),
            'cells_centred_pos_ge_4': bool(d['centred_cells_pos'] >= 4)}
    return {'delta': d, 'sC_vs_uniform': du, 'conjuncts': conj, 'pass': bool(all(conj.values()))}


def t3_reading(r_fbc, r_fb, r_n1):
    cells = [c for c in DEV_CELLS if c in r_fbc and c in r_fb]
    rho = {c: r_fbc[c] - r_fb[c] for c in cells}
    vs_n1 = {c: r_fbc[c] - r_n1[c] for c in cells if c in r_n1}
    conj = {'cells_rho_ge_0.02_ge_4': bool(sum(v >= 0.02 for v in rho.values()) >= 4),
            'cells_FBC_gt_N1_ge_4': bool(sum(v > 0 for v in vs_n1.values()) >= 4)}
    return {'rho_T3': rho, 'FBC_minus_N1': vs_n1, 'conjuncts': conj, 'pass': bool(all(conj.values()))}


def strip(d):
    return {k: v for k, v in d.items() if k not in ('r', 'rc')}


# ----------------------------------------------------------------------------------------------------------------- main
def readings_on(ctx, y, pred, sub, t3r):
    """T1 / T2 readings, M3, M4 and every score on one row set (sub=None: all dev rows)."""
    sc = {k: score(ctx, y, v, sub) for k, v in pred.items()}
    t1r = t1_reading(sc['FBC'], sc['FB'], sc['N1'])
    t1r['vs_N2'] = paired(sc['FBC'], sc['N2'])
    t1r['reported'] = {k: paired(sc[a], sc[b]) for k, (a, b) in
                       {'FB_minus_B0': ('FB', 'B0'), 'FC_minus_B0': ('FC', 'B0'), 'FBC_perp_minus_FB': ('FBC_perp', 'FB'),
                        'FBC_global_minus_FB': ('FBC_global', 'FB'), 'FBC_additive_minus_FB': ('FBC_additive', 'FB')}.items()}
    t1r['epi_stratum'] = {'FBC_minus_FB_epi': (sc['FBC']['epi'] - sc['FB']['epi']) if sc['FBC']['epi'] is not None else None,
                          'FBC_minus_FB_rest': sc['FBC']['rest'] - sc['FB']['rest'], 'n_epi_rows': sc['FBC']['n_epi_rows']}
    t1r['v9_encoding'] = t1_reading(sc['FBC_v9'], sc['FB_v9'], sc['N1_v9'])
    t2r = t2_reading(sc['T2_s_BC'], sc['T2_s_B'], sc['T2_s_C'], sc['T2_uniform'])
    t2r['reported'] = {'s_B_minus_uniform': paired(sc['T2_s_B'], sc['T2_uniform']),
                       'lineage_minus_uniform': paired(sc['T2_lineage'], sc['T2_uniform']),
                       'uniform11_minus_B0_26': paired(sc['T2_uniform'], sc['T2_B0_26'])}
    fb_b0 = t1r['reported']['FB_minus_B0']['per_cell']
    sb_u = t2r['reported']['s_B_minus_uniform']['per_cell']
    fb_pr = {c: t3r['rho']['FB'][c] - t3r['rho']['prior'][c] for c in t3r['rho']['FB']}
    m3 = {'T1_FB_minus_B0': fb_b0, 'T1_positive_control_failed': bool(sum(v <= 0 for v in fb_b0.values()) >= 3),
          'T2_sB_minus_uniform': sb_u, 'T2_positive_control_failed': bool(sum(v <= 0 for v in sb_u.values()) >= 3),
          'T3_FB_minus_prior': fb_pr, 'T3_positive_control_failed': bool(sum(v <= 0 for v in fb_pr.values()) >= 3)}
    m4 = {'S_FC_minus_S_B0': t1r['reported']['FC_minus_B0']['all'], 'S_FBC_minus_S_FB': t1r['delta']['all']}
    return {'n_rows': int(len(next(iter(sc.values()))['r'])), 'T1': t1r, 'T2': t2r, 'M3': m3, 'M4': m4,
            'advance': {'T1': t1r['pass'], 'T2': t2r['pass'], 'T3': t3r['pass']}, 'scores': {k: strip(v) for k, v in sc.items()}}


def run_funnel(ctx):
    y = ctx.y
    res = {'inputs': ctx.inputs, 'prepare_seconds': ctx.prepare_seconds, 'P2_DEV_MEAN': P2_DEV_MEAN, 'T0': t0(ctx)}
    t = time.time()
    enc = 'rank_normal'
    fit = ctx.enc[enc]['cov_dt']
    assert len(fit) == 11 and 'PHH' not in fit, fit
    specs = {k: (feature_builder(ctx, enc, k, fit), v) for k, v in
             (('FB', ['full']), ('FBC', ['full', 'global', 'additive']), ('FBC_perp', ['full']), ('FC', ['full']), ('N1', ['full']))}
    t1o, mu, level = run_t1(ctx, y, specs, fit)
    dev_rows = np.flatnonzero(ctx.dev_mask)
    pred = {'B0': mu[dev_rows]}
    for (nm, var), o in t1o.items():
        pred[nm if var == 'full' else '%s_%s' % (nm, var)] = o['y_hat_dev']
    fbc = t1o[('FBC', 'full')]
    pred['N2'] = t1_predict(ctx, mu[dev_rows], dev_rows, n2_features(ctx, enc), np.asarray(fbc['theta']), fbc['delta'])
    hyper1 = {('%s_%s' % k): {'kappa': o['kappa'], 'kappa_d': o['kappa_d'], 'loco': o['loco'], 'theta': o['theta']}
              for k, o in t1o.items()}
    res['coverage'] = {'dev_rows_by_backoff_level': {str(L): int((level[dev_rows] == L).sum()) for L in range(4)},
                       'drugs_with_ge3_covered_dt_cells': int((fbc['nc'] >= MIN_DRUG_CELLS).sum()),
                       'dev_rows_with_active_delta': int((fbc['nc'][ctx.pert_id[dev_rows]] >= MIN_DRUG_CELLS).sum())}
    fit9 = ctx.enc['v9']['cov_dt']
    specs9 = {k: (feature_builder(ctx, 'v9', k, fit9), ['full']) for k in ('FB', 'FBC', 'N1')}
    t19, _, _ = run_t1(ctx, y, specs9, fit9)
    for k, o in t19.items():
        pred[k[0] + '_v9'] = o['y_hat_dev']
    hyper1.update({('v9_%s_%s' % k): {'kappa': o['kappa'], 'kappa_d': o['kappa_d'], 'loco': o['loco']} for k, o in t19.items()})
    res['T1_hyper'], res['T1_v9_fit_cells'], res['T1_seconds'] = hyper1, fit9, time.time() - t
    t = time.time()
    t2o, n_noshare = run_t2(ctx, y, enc)
    for k, o in t2o.items():
        pred['T2_' + k] = o['y_hat_dev']
    res['T2_hyper'] = {k: {'tau': o['tau'], 'beta': o['beta'], 'loco': o['loco']} for k, o in t2o.items()}
    res['T2_dev_neighbour_pairs_without_shared_mark'] = n_noshare
    res['T2_seconds'] = time.time() - t
    t = time.time()
    b3 = {'prior': None, 'FB': feature_builder(ctx, enc, 'FB', fit), 'FBC': feature_builder(ctx, enc, 'FBC', fit),
          'N1': feature_builder(ctx, enc, 'N1', fit)}
    t3o, skipped = run_t3(ctx, y, b3, fit, ctx.enc[enc]['cov_dev'])
    t3r = t3_reading(t3o['FBC']['rho'], t3o['FB']['rho'], t3o['N1']['rho'])
    t3r['rho'] = {k: o['rho'] for k, o in t3o.items()}
    t3r['hyper'] = {k: {'kappa': o['kappa'], 'loco': o['loco']} for k, o in t3o.items()}
    t3r['skipped_cells'] = skipped
    res['T3'] = t3r
    res['T3_seconds'] = time.time() - t
    t = time.time()
    res['M1'] = dict(zip(('a_dose_neighbour', 'b_within_cell_drug_neighbour'), m1(ctx, y, mu)))
    res['M1_seconds'] = time.time() - t
    # Row sets: all dev rows (as registered, 91.2) and the drug-known rows (back-off level <= 2: the drug has a response in
    # >= 1 dev-train cell), proposed in packet 041 before any real reading; the critic's ruling fixes which is of record.
    known = level[dev_rows] <= 2
    res['row_sets'] = {'all_dev_rows': readings_on(ctx, y, pred, None, t3r),
                       'drug_known_rows': readings_on(ctx, y, pred, known, t3r)}
    return res


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--dti', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t = time.time()
    ctx = prepare(a.data_dir, a.provenance, a.dti)
    res = run_funnel(ctx)
    res['seconds'] = time.time() - t
    json.dump(jsonable(res), open(a.out, 'w'), indent=1)
    print('wrote', a.out, 'seconds %.0f' % res['seconds'], flush=True)   # readings are not echoed: they are read from the JSON


if __name__ == '__main__':
    main()
