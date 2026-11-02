"""Cell-representation programme, stage 1 (RESULTS 103, amended by 103.1 and 103.2): the harness and the model-free rungs.
Read once each, in this order, with the registered gates:

  P0        knocked-down landmark genes in the core cells: the ascending percentile of the gene's own coordinate in its own
            signature (0 = lowest of 978). Passes if the median is <= 0.10. If P0 fails nothing else is read.
  VD        2,100 genes knocked down in all six core cells x 6 cells, per landmark: two-way shares (gene, cell, gene x cell)
            and the gene x cell mean square against a noise upper bound from PC3 / MCF7 96 h vs late pairs. Reported.
  ceiling   PC3 / MCF7 pairs: d96 and d_late against disjoint half-references; rho-hat and sqrt(rho-hat). Reported.
  D1        r(c, j) = log(s_kd / s_ctl) over the 2,100 genes; double-centred; the mean within-cell (partial) Spearman.
            Positive control first (expression), then ACT:prom_1kb partial on expression. Landmark-cluster bootstrap.
  B1        a drug's Level 5 signature against its targets' knockdowns in the same cell (PC3, A375, A549, HEPG2; cold-cell
            training cells, cold-drug training compounds; highest dose at 24 h, else 6 h). Percentile 0 = most correlated.

The harness (per-fold references from every knockdown cell with a 96 h signature of g, minus the row's and the held-out
cell, >= 3; per-row r excluding a knocked-down landmark's own coordinate; r = 0 for a zero-variance prediction) is also
what stage 2 uses.

    python model/cellrep/stage1.py --out model/results/cellrep/stage1.json
"""
import argparse, csv, json, os, re, sys
from collections import defaultdict

import numpy as np

CORE = ('PC3', 'MCF7', 'A375', 'A549', 'HEPG2', 'HT29')
B1_CELLS = ('PC3', 'A375', 'A549', 'HEPG2')
LATE = ('120 h', '144 h')
MIN_REF = 3
N_BOOT = 10000
KD = 'external/kaggle_out/genetic_extract/l1000_genetic_lm978.npz'
CHROM = 'external/kaggle_alt2_stage/lincs-v11-src/v10c_chromatin.npz'
CCLE = 'external/kaggle_alt2_stage/lincs-v11-src/ccle_expr_v11.npz'
GRAPH = 'external/kaggle_alt2_stage/lincs-v11-src/string_graph_v11_expdb.npz'
SUBSTRATE = 'external/kaggle_alt2_stage/lincs-v10c-full/v10c_full_substrate.npz'
L5_Y = 'phase2_assembly/outputs/Y_target_level5_978.npy'
L5_META = 'phase2_assembly/outputs/signatures.tsv'
DTI = 'external/kaggle_alt2_stage/lincs-v11-src/chembl_dti_edges.tsv'
SIGN_MAP = 'external/kaggle_alt2_stage/lincs-v11-src/action_sign_map.json'
SIG_METRICS = 'external/dl_geo_sigmetrics/GSE92742_Broad_LINCS_sig_metrics.txt.gz'
KD_HAIRPIN = 'external/kaggle_out/genetic_extract_sh/l1000_trt_sh_hairpin_lm978.npz'


# --- the harness ------------------------------------------------------------------------------------------------------
def row_r(pred, target, skip=None):
    """Pearson r over the landmarks, without the knocked-down landmark's own coordinate; 0 for a zero-variance vector."""
    p, t = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if skip is not None:
        keep = np.ones(len(p), bool)
        keep[skip] = False
        p, t = p[keep], t[keep]
    p, t = p - p.mean(), t - t.mean()
    den = np.sqrt((p * p).sum() * (t * t).sum())
    return float((p * t).sum() / den) if den > 0 else 0.0


def index_signatures(cells, genes, mask):
    """(gene, cell) -> row, over the rows in mask; raises if a (gene, cell) has two."""
    idx = {}
    for i in np.flatnonzero(mask):
        k = (genes[i], cells[i])
        if k in idx:
            raise ValueError(f'two signatures for {k}')
        idx[k] = i
    return idx


def fold_targets(Y, idx, c0, min_ref=MIN_REF, leaky=False, core=CORE):
    """The fold that holds out c0 (103.1 C1). Training rows (g, c') and held-out rows (g, c0) with
    d = y - mean over R(g) minus {row cell, c0}; R(g) = every cell with a signature of g in idx.
    leaky=True rebuilds the 103 definition (reference = the other CORE cells, c0 included), only for the leak test."""
    by_gene = defaultdict(list)
    for (g, c), i in idx.items():
        by_gene[g].append(c)
    train, test, dropped = [], [], 0
    for g, cs in by_gene.items():
        if leaky:
            ref_pool = [c for c in cs if c in core]
        else:
            ref_pool = [c for c in cs if c != c0]
        for c in cs:
            if c != c0 and c not in core:
                continue
            ref = [x for x in ref_pool if x != c]
            if len(ref) < min_ref:
                dropped += 1
                continue
            d = Y[idx[(g, c)]] - Y[[idx[(g, x)] for x in ref]].mean(0)
            (test if c == c0 else train).append((g, c, d))
    return train, test, dropped


def per_gene_model_scores(train, test, skip_of):
    """The leak test's model: predict a held-out row by the mean training d of its gene (r = 0 if constant)."""
    acc = defaultdict(list)
    for g, _, d in train:
        acc[g].append(d)
    rs = [row_r(np.mean(acc[g], 0), d, skip_of(g)) for g, _, d in test if g in acc]
    return float(np.mean(rs)) if rs else float('nan')


# --- P0 -----------------------------------------------------------------------------------------------------------------
def own_percentile(y, j, ascending=True):
    """0 = the lowest (ascending) or highest (descending) of the 978; the coordinate's own rank among the others."""
    v = np.asarray(y, np.float64)
    other = np.delete(v, j)
    return float(((other < v[j]) if ascending else (other > v[j])).sum() / len(other))


# --- the variance decomposition -----------------------------------------------------------------------------------------
def two_way_shares(A):
    """A [genes, cells] for one landmark (complete). Returns SS shares (gene, cell, gene x cell) and MS_gxc."""
    a = np.asarray(A, np.float64)
    m = a.mean()
    gm, cm = a.mean(1, keepdims=True), a.mean(0, keepdims=True)
    ss_t = ((a - m) ** 2).sum()
    ss_g = a.shape[1] * ((gm - m) ** 2).sum()
    ss_c = a.shape[0] * ((cm - m) ** 2).sum()
    ss_gc = ((a - gm - cm + m) ** 2).sum()
    df_gc = (a.shape[0] - 1) * (a.shape[1] - 1)
    return {'gene': ss_g / ss_t, 'cell': ss_c / ss_t, 'gene_x_cell': ss_gc / ss_t, 'ms_gxc': ss_gc / df_gc}


# --- the noise ceiling --------------------------------------------------------------------------------------------------
def half_references(ref_cells):
    """Two disjoint halves of the reference cells: sorted by name, alternating."""
    s = sorted(ref_cells)
    return s[0::2], s[1::2]


# --- D1 -----------------------------------------------------------------------------------------------------------------
def rank(x):
    o = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    # average ties
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def partial_spearman(x, y, z=None):
    """Spearman of x and y, partial on z (Pearson on ranks, z's rank regressed out of both)."""
    rx, ry = rank(np.asarray(x, np.float64)), rank(np.asarray(y, np.float64))
    if z is not None:
        rz = rank(np.asarray(z, np.float64))
        Z = np.stack([np.ones_like(rz), rz], 1)
        rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return row_r(rx, ry)


def centre_by_landmark(X):
    """X [cells, landmarks] -> minus each landmark's mean over cells (NaN-aware)."""
    return X - np.nanmean(X, 0, keepdims=True)


def mean_within_cell(R, F, E=None, cells_ok=None):
    """Mean over cells of the within-cell (partial) Spearman across landmarks; per-cell values too."""
    vals = []
    for c in range(R.shape[0]):
        if cells_ok is not None and not cells_ok[c]:
            vals.append(np.nan)
            continue
        ok = np.isfinite(R[c]) & np.isfinite(F[c]) & (np.isfinite(E[c]) if E is not None else True)
        vals.append(partial_spearman(R[c][ok], F[c][ok], E[c][ok] if E is not None else None))
    vals = np.array(vals)
    return float(np.nanmean(vals)), vals


def landmark_bootstrap(R, F, E=None, cells_ok=None, n_boot=N_BOOT, seed=0):
    """Percentile 95 % CI of mean_within_cell over landmark resamples (the same resample in every cell)."""
    rng = np.random.default_rng(seed)
    J = R.shape[1]
    out = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(0, J, J)
        out[b] = mean_within_cell(R[:, k], F[:, k], None if E is None else E[:, k], cells_ok)[0]
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


# --- B1 -----------------------------------------------------------------------------------------------------------------
_DOSE = re.compile(r'^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Zµμ�]+)\s*$')


def dose_um(s):
    """'10 uM', '10.0 um', '500 nM', '10 µM' / 'μM' (or its mojibake) -> micromolar; None if unparseable."""
    m = _DOSE.match(str(s))
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2).lower().replace('µ', 'u').replace('μ', 'u').replace('�', 'u')
    return {'um': v, 'nm': v / 1000.0, 'mm': v * 1000.0}.get(u)


def choose_rows(rows):
    """rows: list of (row, time, dose_um). The highest dose at 24 h, else at 6 h; all rows at that dose (averaged later)."""
    for t in ('24 h', '6 h'):
        r = [(x[0], x[2]) for x in rows if x[1] == t and x[2] is not None]
        if r:
            top = max(d for _, d in r)
            return [i for i, d in r if d == top]
    return []


def percentile_of(corrs, k):
    """0 = the most correlated; the fraction of the others above k's correlation."""
    c = np.asarray(corrs, np.float64)
    return float((np.delete(c, k) > c[k]).sum() / (len(c) - 1))


def corr_rows(v, M, skip=None):
    """Pearson r of vector v with each row of M, without coordinate skip."""
    if skip is not None:
        keep = np.ones(M.shape[1], bool)
        keep[skip] = False
        v, M = v[keep], M[:, keep]
    v = v - v.mean()
    Mc = M - M.mean(1, keepdims=True)
    den = np.sqrt((v * v).sum() * (Mc * Mc).sum(1))
    return np.where(den > 0, Mc @ v / np.where(den > 0, den, 1), 0.0)


def cluster_bootstrap_median(values, clusters, n_boot=N_BOOT, seed=0):
    v, cl = np.asarray(values, np.float64), np.asarray(clusters)
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        out[b] = np.median(v[np.concatenate([groups[p] for p in pick])])
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def cluster_bootstrap_mean(values, clusters, n_boot=N_BOOT, seed=0):
    v, cl = np.asarray(values, np.float64), np.asarray(clusters)
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        out[b] = v[np.concatenate([groups[p] for p in pick])].mean()
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


# --- the run ------------------------------------------------------------------------------------------------------------
def load_sig_metrics(path):
    """sig_id -> {'tas', 'distil_cc_q75', 'pct_self_rank_q25', 'distil_nsample'} (GSE92742 sig_metrics; -666 -> None)."""
    import gzip
    out = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rec = {}
            for k in ('tas', 'distil_cc_q75', 'pct_self_rank_q25', 'distil_nsample'):
                try:
                    v = float(r[k])
                except (KeyError, ValueError):
                    v = None
                rec[k] = None if v is None or v == -666.0 else v
            out[r['sig_id']] = rec
    return out


def load_kd(path=KD):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def run(a):
    z = load_kd(a.kd)
    Y, pt, cell, gene, itime = z['Y'], z['pert_type'], z['cell_id'], z['pert_iname'], z['pert_itime']
    lm_sym = [str(s) for s in z['landmark_symbol']]
    lm_pos = {s: i for i, s in enumerate(lm_sym)}
    skip_of = lambda g: lm_pos.get(g)
    kd96 = (pt == 'trt_sh.cgs') & (itime == '96 h')
    idx = index_signatures(cell, gene, kd96)
    rec = {'rule': 'RESULTS 103 / 103.1 / 103.2, read once each in order', 'kd_sha1_expected': '5d26ca5d'}

    # the leak test on these exact index structures (synthetic values: gene effect + noise, no cell-specific signal)
    rng = np.random.default_rng(0)
    gl = sorted({g for g, _ in idx})
    G = {g: rng.normal(size=Y.shape[1]) for g in gl}
    Ys = np.zeros_like(Y)
    for (g, c), i in idx.items():
        Ys[i] = G[g] + rng.normal(size=Y.shape[1])
    lt = {}
    for c0 in CORE:
        tr, te, _ = fold_targets(Ys, idx, c0)
        trl, tel, _ = fold_targets(Ys, idx, c0, leaky=True)
        lt[c0] = {'fixed': per_gene_model_scores(tr, te, skip_of), 'leaky': per_gene_model_scores(trl, tel, skip_of)}
    rec['leak_test'] = lt
    if not all(abs(v['fixed']) < 0.02 and v['leaky'] < -0.9 for v in lt.values()):
        raise SystemExit(f'FATAL: leak test {lt}')

    # P0
    p0 = []
    for (g, c), i in idx.items():
        if c in CORE and g in lm_pos:
            p0.append((g, c, own_percentile(Y[i], lm_pos[g])))
    med = float(np.median([p for *_, p in p0]))
    oe = (pt == 'trt_oe') & np.isin(cell, CORE)
    p0_oe = [own_percentile(Y[i], lm_pos[gene[i]], ascending=False) for i in np.flatnonzero(oe) if gene[i] in lm_pos]
    rec['P0'] = {'n_rows': len(p0), 'median_own_percentile': med, 'passes': med <= 0.10,
                 'per_cell_median': {c: float(np.median([p for _, cc, p in p0 if cc == c])) for c in CORE},
                 'oe_median_desc_percentile': float(np.median(p0_oe)) if p0_oe else None, 'oe_n': len(p0_oe),
                 'per_gene': {f'{g}|{c}': p for g, c, p in p0}}
    if not rec['P0']['passes']:
        rec['stopped'] = 'P0 failed: nothing else is read (103)'
        return rec

    # reported strata (sig_metrics, approved 2026-11-02): P0 by the knockdown signature's transcriptional activity score
    sm = load_sig_metrics(a.sig_metrics)
    sid = z['sig_id']
    tas = {f'{gene[i]}|{cell[i]}': sm.get(str(sid[i]), {}).get('tas') for i in idx.values()}
    pv = [(tas.get(f'{g}|{c}'), p) for g, c, p in p0 if tas.get(f'{g}|{c}') is not None]
    if pv:
        cut = np.quantile([t for t, _ in pv], [1 / 3, 2 / 3])
        rec['P0']['reported_by_tas_tertile'] = {
            name: float(np.median([p for t, p in pv if lo <= t < hi]))
            for name, lo, hi in (('low', -np.inf, cut[0]), ('mid', cut[0], cut[1]), ('high', cut[1], np.inf))}
        rec['P0']['reported_tas_coverage'] = len(pv)

    # R(g): every knockdown cell with a 96 h signature of g (103.1 C1); its size per core cell (review 107 C4)
    cells_of = defaultdict(list)
    for (g, c) in idx:
        cells_of[g].append(c)
    rec['R_size_per_core_cell'] = {c: float(np.mean([len(cells_of[g]) - 1 for g in cells_of if c in cells_of[g]]))
                                   for c in CORE}

    # the 2,100 genes knocked down in all six core cells, as one [genes, cells, landmarks] array
    g6 = sorted(g for g in gl if all((g, c) in idx for c in CORE))
    rec['n_all_six_genes'] = len(g6)
    T = Y[np.array([[idx[(g, c)] for c in CORE] for g in g6])].astype(np.float64)     # [g, 6, 978]
    own = np.array([lm_pos.get(g, -1) for g in g6])

    # VD
    late = defaultdict(dict)
    for i in np.flatnonzero((pt == 'trt_sh.cgs') & np.isin(itime, LATE)):
        late[(gene[i], cell[i])][str(itime[i])] = i
    pairs = [(g, c) for (g, c) in idx if c in ('PC3', 'MCF7') and (g, c) in late]
    diffs = []
    for g, c in pairs:
        il = late[(g, c)].get('120 h', late[(g, c)].get('144 h'))
        dv = Y[idx[(g, c)]] - Y[il]
        if g in lm_pos:
            dv = np.delete(dv, lm_pos[g])
        diffs.append(dv)
    noise_bound = float(np.mean(np.concatenate(diffs) ** 2) / 2)
    sh = defaultdict(list)
    for j in range(Y.shape[1]):
        for k, v in two_way_shares(T[own != j, :, j]).items():
            sh[k].append(v)
    rec['VD'] = {k: float(np.mean(v)) for k, v in sh.items()}
    rec['VD'].update(noise_bound=noise_bound, ms_gxc_over_noise_bound=float(np.mean(sh['ms_gxc']) / noise_bound),
                     n_noise_pairs=len(pairs), note='noise bound from PC3 / MCF7 only; late-profiled genes are a selected subset')

    # the ceiling (disjoint half-references)
    rho = []
    for g, c in pairs:
        A, B = half_references([x for x in cells_of[g] if x != c])
        if len(A) < 2 or len(B) < 2:
            continue
        il = late[(g, c)].get('120 h', late[(g, c)].get('144 h'))
        d96 = Y[idx[(g, c)]] - Y[[idx[(g, x)] for x in A]].mean(0)
        dl = Y[il] - Y[[idx[(g, x)] for x in B]].mean(0)
        rho.append((c, row_r(d96, dl, skip_of(g))))
    rr = np.array([r for _, r in rho])
    rec['ceiling'] = {'n_pairs': len(rho), 'rho_hat': float(rr.mean()), 'sqrt_rho_hat': float(np.sqrt(max(rr.mean(), 0))),
                      'rowwise_sqrt_mean': float(np.mean(np.sqrt(np.clip(rr, 0, None)))),
                      'per_cell': {c: float(np.mean([r for cc, r in rho if cc == c])) for c in ('PC3', 'MCF7')},
                      'note': 'lower bounds; PC3 / MCF7 only, assumed elsewhere'}

    # D1
    ch = np.load(a.chrom, allow_pickle=True)
    cpos = {str(c): i for i, c in enumerate(ch['cells'])}
    fn = [str(f) for f in ch['feature_names']]
    ce = np.load(a.ccle, allow_pickle=True)
    epos = {str(c): i for i, c in enumerate(ce['cells'])}
    ctl96 = (pt == 'ctl_vector') & (itime == '96 h')
    missing = [c for c in CORE if c not in cpos or c not in epos]
    if missing:
        raise SystemExit(f'FATAL: no chromatin or CCLE for {missing}')
    S_kd = np.full((6, Y.shape[1]), np.nan)
    S_ctl = np.full((6, Y.shape[1]), np.nan)
    for j in range(Y.shape[1]):
        S_kd[:, j] = T[own != j, :, j].std(0, ddof=1)
    for ci, c in enumerate(CORE):
        S_ctl[ci] = Y[ctl96 & (cell == c)].std(0, ddof=1)
    rec['D1_n_ctl_vector'] = {c: int((ctl96 & (cell == c)).sum()) for c in CORE}
    S_hp = None
    if a.kd_hairpin and os.path.exists(a.kd_hairpin):          # review 107 C3: s_kd from hairpin-level signatures
        h = np.load(a.kd_hairpin)
        hm = (h['pert_type'] == 'trt_sh') & (h['pert_itime'] == '96 h') & np.isin(h['pert_iname'], g6)
        S_hp = np.full((6, Y.shape[1]), np.nan)
        rec['D1_n_hairpin_rows'] = {}
        for ci, c in enumerate(CORE):
            r = np.flatnonzero(hm & (h['cell_id'] == c))
            Hc, tg_c = h['Y'][r].astype(np.float64), h['pert_iname'][r]
            rec['D1_n_hairpin_rows'][c] = int(len(r))
            for j in range(Y.shape[1]):
                S_hp[ci, j] = Hc[tg_c != lm_sym[j], j].std(ddof=1)
    Rr = centre_by_landmark(np.log(S_kd / S_ctl))
    Rctl = centre_by_landmark(np.log(S_ctl))
    E = centre_by_landmark(np.stack([ce['expr_landmarks'][epos[c]] for c in CORE]).astype(np.float64))

    def feat(name):
        k = fn.index(name)
        X = np.stack([ch['F'][cpos[c], :, k] for c in CORE]).astype(np.float64)
        axis = name.split(':')[0]
        ai = {'ACC': 0, 'ACT': 1, 'PCG': 2, 'METH': 3}[axis]
        ok = np.array([bool(ch['M'][cpos[c], :, ai].any()) for c in CORE])
        X[~ok] = np.nan
        return centre_by_landmark(X), ok

    pc_mean, pc_cells = mean_within_cell(Rr, E)
    pc_ci = landmark_bootstrap(Rr, E, n_boot=a.n_boot)
    d1 = {'positive_control': {'mean_within_cell_spearman': pc_mean, 'per_cell': dict(zip(CORE, map(float, pc_cells))),
                               'ci95': pc_ci, 'passes': pc_ci[0] > 0 or pc_ci[1] < 0}}
    if d1['positive_control']['passes']:
        F, ok = feat('ACT:prom_1kb')
        m, per = mean_within_cell(Rr, F, E, ok)
        ci = landmark_bootstrap(Rr, F, E, ok, n_boot=a.n_boot)
        npos = int(np.nansum(per > 0))
        d1['ACT:prom_1kb'] = {'mean_within_cell_partial_spearman': m, 'per_cell': dict(zip(CORE, map(float, per))),
                              'ci95': ci, 'cells_positive': npos, 'passes': ci[0] > 0 and npos >= 5}
        rep = {}
        for name in fn:
            Fx, okx = feat(name)
            rep[name] = {'partial_on_expression': mean_within_cell(Rr, Fx, E, okx)[0],
                         'rung1_no_covariate': mean_within_cell(Rr, Fx, None, okx)[0],
                         'technical_channel_s_ctl': mean_within_cell(Rctl, Fx, E, okx)[0]}
        d1['reported'] = rep
        if S_hp is not None:
            Rh = centre_by_landmark(np.log(S_hp / S_ctl))
            mh, perh = mean_within_cell(Rh, F, E, ok)
            d1['hairpin_variant_ACT:prom_1kb'] = {
                'mean_within_cell_partial_spearman': mh, 'per_cell': dict(zip(CORE, map(float, perh))),
                'ci95': landmark_bootstrap(Rh, F, E, ok, n_boot=a.n_boot), 'cells_positive': int(np.nansum(perh > 0)),
                'positive_control': mean_within_cell(Rh, E)[0],
                'note': 's_kd from hairpin-level trt_sh (same construction as ctl_vector), review 107 C3'}
        Ru = np.log(S_kd / S_ctl)
        Fu = np.stack([ch['F'][cpos[c], :, fn.index('ACT:prom_1kb')] for c in CORE]).astype(np.float64)
        d1['reported_uncentred_ACT_per_cell'] = {c: partial_spearman(Ru[i], Fu[i]) for i, c in enumerate(CORE)}
    else:
        d1['verdict'] = 'not interpreted: the positive control failed (103)'
    rec['D1'] = d1

    # B1
    rec['B1'] = run_b1(a, Y, pt, cell, gene, itime, idx, lm_pos, sm)
    return rec


def run_b1(a, Y, pt, cell, gene, itime, idx, lm_pos, sm=None):
    sub = np.load(a.substrate, allow_pickle=True)
    sp, s2 = sub['meta_pert_id'], sub['split_cold_drug_1']
    bad = set(sp[s2 != 'train'].tolist())
    train_cpd = set(sp.tolist()) - bad
    sc, s1 = sub['meta_cell_iname'], sub['split_cold_cell_1']
    train_cells = {c for c in B1_CELLS if (s1[sc == c] == 'train').all() and (sc == c).any()}
    Y5 = np.load(a.l5_y, mmap_mode='r')
    rows, sig_of, bad_dose = defaultdict(list), {}, 0
    for r in csv.DictReader(open(a.l5_meta, encoding='utf-8'), delimiter='\t'):
        if r['cell_id'] in train_cells and r['pert_id'] in train_cpd:
            du = dose_um(r['dose'])
            bad_dose += du is None
            rows[(r['pert_id'], r['cell_id'])].append((int(r['row']), r['time'], du))
            sig_of[int(r['row'])] = r['sig_id']
    sign = json.load(open(a.sign_map, encoding='utf-8'))['sign']
    tg = defaultdict(lambda: defaultdict(set))
    for r in csv.DictReader(open(a.dti, encoding='utf-8'), delimiter='\t'):
        if r.get('organism') == 'Homo sapiens' and r['gene_symbol']:
            tg[r['pert_id']][int(sign.get(r['action_type'], 0))].add(r['gene_symbol'])
    g = np.load(a.graph, allow_pickle=True)
    deg = np.bincount(g['edge_index'].ravel(), minlength=len(g['nodes']))
    gdeg = {str(n): int(d) for n, d in zip(g['nodes'], deg)}
    kd_by_cell = {c: sorted(gg for (gg, cc) in idx if cc == c) for c in train_cells}
    kd_mat = {c: np.stack([Y[idx[(gg, c)]] for gg in kd_by_cell[c]]) for c in train_cells}
    kd_pos = {c: {gg: k for k, gg in enumerate(kd_by_cell[c])} for c in train_cells}
    oe96 = (pt == 'trt_oe') & (itime == '96 h')
    rng = np.random.default_rng(0)
    out, out_all, out_hub, mm, ag, out_tas = [], [], [], [], [], []
    for (d, c), rr in sorted(rows.items()):
        targets = tg.get(d, {}).get(-1, set())
        chosen = choose_rows(rr)
        if not chosen or not targets:
            continue
        v = np.asarray(Y5[chosen], np.float64).mean(0)
        tv = [sm.get(sig_of[i], {}).get('tas') for i in chosen] if sm else []
        tv = [t for t in tv if t is not None]
        tas_d = float(np.mean(tv)) if tv else None
        v_all = np.asarray(Y5[[i for i, *_ in rr]], np.float64).mean(0)
        for t in sorted(targets):
            if t not in kd_pos[c]:
                continue
            sk = lm_pos.get(t)
            cr = corr_rows(v, kd_mat[c], sk)
            p = percentile_of(cr, kd_pos[c][t])
            out.append((d, c, t, p))
            out_tas.append(tas_d)
            out_all.append(percentile_of(corr_rows(v_all, kd_mat[c], sk), kd_pos[c][t]))
            cand = sorted((abs(gdeg.get(x, 0) - gdeg.get(t, 0)), x) for x in kd_by_cell[c] if x != t)[:10]
            h = cand[rng.integers(0, len(cand))][1]
            out_hub.append(percentile_of(cr, kd_pos[c][h]))
            pm = []
            for c2 in train_cells:
                if c2 != c and t in kd_pos[c2]:
                    pm.append(percentile_of(corr_rows(v, kd_mat[c2], sk), kd_pos[c2][t]))
            mm.append((d, p, float(np.mean(pm)) if pm else None))
        for t in sorted(tg.get(d, {}).get(1, set())):
            oi = [i for i in np.flatnonzero(oe96 & (cell == c)) if gene[i] == t]
            if oi:
                oset = np.flatnonzero(oe96 & (cell == c))
                cr = corr_rows(v, Y[oset], lm_pos.get(t))
                ag.append(percentile_of(cr, int(np.flatnonzero(oset == oi[0])[0])))
    ps = np.array([p for *_, p in out])
    drugs = np.array([d for d, *_ in out])
    res = {'n_pairs': len(out), 'n_drugs': len(set(drugs.tolist())), 'cells': sorted(train_cells),
           'unparseable_doses': int(bad_dose),
           'median_percentile': float(np.median(ps)) if len(ps) else None}
    if len(ps):
        ci = cluster_bootstrap_median(ps, drugs, a.n_boot)
        res.update(ci95=ci, passes=bool(res['median_percentile'] < 0.45 and ci[1] < 0.5),
                   top5_fraction=float((ps <= 0.05).mean()),
                   per_cell={c: float(np.median([p for _, cc, _, p in out if cc == c])) for c in sorted(train_cells)},
                   reported_all_signature_average_median=float(np.median(out_all)),
                   reported_hub_control_median=float(np.median(out_hub)))
        keep = [(d, p, q) for d, p, q in mm if q is not None]
        diff = np.array([q - p for _, p, q in keep])
        res['matched_vs_mismatched'] = {'n_pairs': len(keep), 'dropped_no_other_cell': len(mm) - len(keep),
                                        'mean_p_mis_minus_p_match': float(diff.mean()) if len(diff) else None}
        if len(diff):
            cim = cluster_bootstrap_mean(diff, [d for d, *_ in keep], a.n_boot)
            res['matched_vs_mismatched'].update(ci95=cim, cell_specific=bool(diff.mean() > 0 and cim[0] > 0))
        tp = [(t, p) for t, p in zip(out_tas, ps) if t is not None]
        if len(tp) >= 9:
            cut = np.quantile([t for t, _ in tp], [1 / 3, 2 / 3])
            res['reported_by_drug_tas_tertile'] = {
                name: float(np.median([p for t, p in tp if lo <= t < hi]))
                for name, lo, hi in (('low', -np.inf, cut[0]), ('mid', cut[0], cut[1]), ('high', cut[1], np.inf))}
            res['reported_tas_coverage'] = len(tp)
        res['reported_agonists_vs_oe'] = {'n': len(ag), 'median_percentile': float(np.median(ag)) if ag else None}
        res['meaning'] = ('pass of the main rule = a gene-level drug <-> target-knockdown correspondence; the cell-specific '
                          'meaning needs matched_vs_mismatched.cell_specific (103.1 C4)')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--kd', default=KD)
    ap.add_argument('--chrom', default=CHROM)
    ap.add_argument('--ccle', default=CCLE)
    ap.add_argument('--graph', default=GRAPH)
    ap.add_argument('--substrate', default=SUBSTRATE)
    ap.add_argument('--l5_y', default=L5_Y)
    ap.add_argument('--l5_meta', default=L5_META)
    ap.add_argument('--dti', default=DTI)
    ap.add_argument('--sign_map', default=SIGN_MAP)
    ap.add_argument('--sig_metrics', default=SIG_METRICS)
    ap.add_argument('--kd_hairpin', default=KD_HAIRPIN)
    ap.add_argument('--n_boot', type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- read once')
    rec = run(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: (v if k not in ('P0', 'B1') else {kk: vv for kk, vv in v.items() if kk != 'per_gene'})
                      for k, v in rec.items()}, indent=1, default=str)[:6000])
    return rec


if __name__ == '__main__':
    main()
