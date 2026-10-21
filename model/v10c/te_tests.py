"""T-E (RESULTS 101, amended 101.1): three CPU tests of chromatin where baseline expression cannot stand in. Read once.

Rows: split_cold_cell_1 TRAINING rows only (T-C's test cells are never read).
Signed contrast of a row: its delta minus the mean delta of its other-compound plate-mates that are NOT in the epigenetic,
bet_inactive_control or sirtuin_or_natural_product classes (101.1 C1); >= 2 such mates.
Profile U^K[c, g]: mean contrast of gene g over class K's rows in cell c (>= 5 rows); REF = the non-epigenetic classes.
Per cell: partial Pearson r over the genes with the matched mark measured (mask = 1) between m(c, .) and U(c, .), after
regressing out basal b_cg, the gene's leave-cell-out mean of U and the gene's leave-cell-out mean of m (101.1 C4).
H-E3: silent genes (bottom tercile of b within c), induction I = mean max(contrast, 0) over all of c's rows, repression R the
same for max(-contrast, 0); basal enters as decile indicators within the tercile.
Over cells: the equal-weight mean of r_c, a cell-cluster bootstrap (20,000 draws, default_rng(0)); primaries at 98.33 %.
A primary passes iff r has the predicted sign with its CI excluding 0, Delta_spec likewise, and the real mean lies beyond
the cell-swap null's 95th percentile in the predicted direction.

    python model/v10c/te_tests.py --out model/results/v10c/te_reading.json
"""
import argparse, json, os, time

import numpy as np

SUB = 'external/v10c_full/v10c_full_substrate.npz'
CHROM = 'external/v10c_chromatin.npz'
CLASSES = 'model/results/v10c/compound_classes.json'
EXCLUDED_MATE_CLASSES = ('epigenetic', 'bet_inactive_control', 'sirtuin_or_natural_product')
MIN_MATES, MIN_ROWS, MASK_COVER = 2, 5, 0.5
N_BOOT, N_NULL = 20000, 200
CI_PRIMARY, CI_REPORTED = 0.9833, 0.95
AXIS = {'ACC': 0, 'ACT': 1, 'PCG': 2, 'METH': 3}
FEAT = {'ACC:prom_1kb': 0, 'ACC:tss_50kb': 4, 'ACT:prom_1kb': 5, 'ACT:tss_50kb': 9, 'PCG:prom_1kb': 10, 'PCG:tss_50kb': 14,
        'METH:tss_1kb': 15}
BETAS = (0.02, 0.05, 0.1)


# ---------------------------------------------------------------------------------------------------------------------
def signed_contrast(Y, plate, pert, mate_ok, min_mates=MIN_MATES):
    """Row delta minus the mean delta of its other-compound plate-mates with mate_ok; NaN rows where < min_mates."""
    out = np.full(Y.shape, np.nan, np.float32)
    n_mates = np.zeros(len(Y), np.int64)
    order = np.argsort(plate, kind='stable')
    ps = plate[order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        rows = order[s:e]
        ok = mate_ok[rows]
        if ok.sum() < min_mates:
            continue
        Yp = Y[rows].astype(np.float64)
        tot, ntot = Yp[ok].sum(0), int(ok.sum())
        pr = pert[rows]
        for i, r in enumerate(rows):
            same = ok & (pr == pr[i])
            k = ntot - int(same.sum())
            n_mates[r] = k
            if k >= min_mates:
                out[r] = Yp[i] - (tot - Yp[same].sum(0)) / k
    return out, n_mates


def rank01_rows(A):
    r = np.argsort(np.argsort(A, 1), 1).astype(np.float64)
    return r / max(1, A.shape[1] - 1)


def lco_mean(V, valid):
    """[C, G] leave-cell-out mean over the other cells where valid; NaN where no other cell is valid."""
    Vz = np.where(valid, V, 0.0)
    s, n = Vz.sum(0, keepdims=True), valid.sum(0, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (s - Vz) / (n - valid)


def partial_r(m, u, X):
    """Pearson r between m and u after least-squares removal of X (with an intercept)."""
    X = np.column_stack([np.ones(len(m)), X])
    beta_m = np.linalg.lstsq(X, m, rcond=None)[0]
    beta_u = np.linalg.lstsq(X, u, rcond=None)[0]
    a, b = m - X @ beta_m, u - X @ beta_u
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def decile_dummies(b):
    q = np.quantile(b, np.linspace(0, 1, 11)[1:-1])
    k = np.searchsorted(q, b, side='right')
    D = np.zeros((len(b), 10))
    D[np.arange(len(b)), k] = 1.0
    return D[:, 1:]                                         # drop one level (the intercept carries it)


def covariates(M_feat, mask, U):
    """The gene's leave-cell-out means of U (over cells with a finite profile) and of m (over cells with the mark)."""
    return lco_mean(np.nan_to_num(U), np.isfinite(U)), lco_mean(M_feat, mask)


def per_cell_r(M_feat, mask, U, basal, cells, gene_sel=None, flexible_basal=False, ubar=None, mbar=None, counts=None,
               extra=None):
    """r_c for each eligible cell (rows of M_feat / U / basal index cells). gene_sel: optional [C, G] extra gene filter.
    ubar / mbar: fixed covariates (review 094 C1: the nulls hold the real-data ones); computed here when not given.
    counts: optional dict filled with each candidate cell's eligible gene count.
    extra: optional list of [C, G] covariates added to the design (RESULTS 101.4 A1); genes need them finite."""
    valid_u = np.isfinite(U)
    if ubar is None or mbar is None:
        u0, m0 = covariates(M_feat, mask, U)
        ubar = u0 if ubar is None else ubar
        mbar = m0 if mbar is None else mbar
    out = {}
    for i in cells:
        g = mask[i] & valid_u[i] & np.isfinite(ubar[i]) & np.isfinite(mbar[i])
        if gene_sel is not None:
            g &= gene_sel[i]
        for e in extra or ():
            g &= np.isfinite(e[i])
        if counts is not None:
            counts[int(i)] = int(g.sum())
        if g.sum() < 30:
            continue
        cov = [decile_dummies(basal[i, g])] if flexible_basal else [basal[i, g][:, None]]
        X = np.column_stack(cov + [ubar[i, g], mbar[i, g]] + [e[i, g] for e in extra or ()])
        out[int(i)] = partial_r(M_feat[i, g], U[i, g], X)
    return out


def boot_ci(vals, level, n_boot=N_BOOT, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


def nulls(M_feat, mask, U, basal, cells, gene_sel=None, flexible_basal=False, n=None, seed=1):
    """Gene-permute and cell-swap nulls. Only m (and, for cell-swap, its mask) is permuted; the covariates are the
    real-data ones, held fixed (review 094 C1)."""
    n = n or N_NULL
    rng = np.random.default_rng(seed)
    cells = np.asarray(cells)
    ubar, mbar = covariates(M_feat, mask, U)
    gp, cs = [], []
    for _ in range(n):
        Mp = M_feat.copy()
        for i in cells:
            idx = np.flatnonzero(mask[i])
            Mp[i, idx] = M_feat[i, rng.permutation(idx)]
        r = per_cell_r(Mp, mask, U, basal, cells, gene_sel, flexible_basal, ubar=ubar, mbar=mbar)
        gp.append(np.mean(list(r.values())))
        d = cells[derangement(len(cells), rng)]
        Ms, ms = M_feat.copy(), mask.copy()
        Ms[cells], ms[cells] = M_feat[d], mask[d] & mask[cells]
        r = per_cell_r(Ms, ms, U, basal, cells, gene_sel, flexible_basal, ubar=ubar, mbar=mbar)
        cs.append(np.mean(list(r.values())))
    return np.array(gp), np.array(cs)


def calibrate(M_feat, mask, U, basal, cells, sign, level, gene_sel=None, flexible_basal=False, betas=BETAS, U_ref=None):
    """Planted U' = U + sign * beta * sd(U) * z(m). Reported: the smallest beta at which the CI condition is met (review 094
    C2), and Delta_spec's CI per beta (planted U against the unplanted reference)."""
    rr = per_cell_r(M_feat, mask, U_ref, basal, cells, gene_sel, flexible_basal) if U_ref is not None else None
    out = {}
    for b in betas:
        Up = U.copy()
        for i in cells:
            g = mask[i] & np.isfinite(U[i])
            z = (M_feat[i, g] - M_feat[i, g].mean()) / (M_feat[i, g].std() + 1e-12)
            Up[i, g] = U[i, g] + sign * b * np.nanstd(U[i, g]) * z
        r = per_cell_r(M_feat, mask, Up, basal, cells, gene_sel, flexible_basal)
        lo, hi = boot_ci(list(r.values()), level, n_boot=4000)
        out[str(b)] = {'mean_r': float(np.mean(list(r.values()))), 'ci': [lo, hi],
                       'ci_condition_met': bool(lo > 0) if sign > 0 else bool(hi < 0)}
        if rr is not None:
            common = sorted(set(r) & set(rr))
            out[str(b)]['delta_spec_ci'] = boot_ci([r[c] - rr[c] for c in common], level, n_boot=4000)
    det = [float(b) for b in out if out[b]['ci_condition_met']]
    return out, (min(det) if det else None)


def decide(sign, ci, spec_ci, mean_r, cell_swap_p95):
    """101.1: the predicted sign with the CI excluding 0, Delta_spec likewise, and beyond the cell-swap 95th percentile."""
    lo, hi = ci
    slo, shi = spec_ci
    if sign > 0:
        return bool(lo > 0 and slo > 0 and mean_r > cell_swap_p95)
    return bool(hi < 0 and shi < 0 and mean_r < cell_swap_p95)


def evaluate(name, M_feat, mask, U, U_ref, basal, cells, sign, level, gene_sel=None, flexible_basal=False, do_null=True):
    counts = {}
    r = per_cell_r(M_feat, mask, U, basal, cells, gene_sel, flexible_basal, counts=counts)
    rr = per_cell_r(M_feat, mask, U_ref, basal, cells, gene_sel, flexible_basal)
    common = sorted(set(r) & set(rr))
    res = {'name': name, 'n_cells': len(r), 'mean_r': float(np.mean(list(r.values()))) if r else None,
           'per_cell_r': {str(k): v for k, v in r.items()},
           'eligible_gene_counts': {str(k): v for k, v in counts.items()},
           'skipped_lt30_genes': {str(k): v for k, v in counts.items() if k not in r}}
    if len(r) < 3:
        res['note'] = 'fewer than 3 eligible cells'
        res['passes'] = False
        return res
    res['ci'] = boot_ci(list(r.values()), level)
    spec = [r[c] - rr[c] for c in common]
    res['delta_spec'] = float(np.mean(spec))
    res['delta_spec_ci'] = boot_ci(spec, level)
    res['ref_mean_r'] = float(np.mean([rr[c] for c in common]))
    if do_null:
        gp, cs = nulls(M_feat, mask, U, basal, list(r), gene_sel, flexible_basal)
        res['gene_permute'] = {'mean': float(gp.mean()), 'rank_of_real': float((gp < res['mean_r']).mean())}
        res['cell_swap'] = {'mean': float(cs.mean()), 'p95_predicted_dir': float(np.quantile(cs, 0.95 if sign > 0 else 0.05)),
                            'rank_of_real': float((cs < res['mean_r']).mean())}
        cal, mde = calibrate(M_feat, mask, U, basal, list(r), sign, level, gene_sel, flexible_basal, U_ref=U_ref)
        res['calibration'], res['smallest_beta_ci_condition_met'] = cal, mde
        res['passes'] = decide(sign, res['ci'], res['delta_spec_ci'], res['mean_r'],
                               res['cell_swap']['p95_predicted_dir'])
    return res


# ---------------------------------------------------------------------------------------------------------------------
def class_profiles(C, cell_idx, n_cells, sel, min_rows=MIN_ROWS):
    """[n_cells, G] mean of C over rows in sel per cell (NaN where < min_rows finite rows)."""
    G = C.shape[1]
    S, N = np.zeros((n_cells, G)), np.zeros(n_cells, np.int64)
    rows = np.flatnonzero(sel & np.isfinite(C[:, 0]))
    np.add.at(S, cell_idx[rows], C[rows].astype(np.float64))
    np.add.at(N, cell_idx[rows], 1)
    P = S / np.maximum(N, 1)[:, None]
    P[N < min_rows] = np.nan
    return P, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- T-E is read once')
    t0 = time.time()
    z = np.load(SUB, allow_pickle=True)
    tr = np.flatnonzero(z['split_cold_cell_1'] == 'train')
    pert, cell, plate = z['meta_pert_id'][tr], z['meta_cell_iname'][tr], z['plate_id'][tr]
    ctl, ctl_id = z['ctl'], z['ctl_id'][tr]
    Y = (z['X'][tr].astype(np.float32) - ctl[ctl_id].astype(np.float32))
    cl = json.load(open(CLASSES, encoding='utf-8'))
    cls = np.array([cl['class'].get(p, 'unannotated') for p in pert])
    sub = np.array([cl['sub'].get(p, '') for p in pert])
    sens = {k: set(v) for k, v in cl['sensitivity_sets'].items()}
    mate_ok = ~np.isin(cls, EXCLUDED_MATE_CLASSES)
    C, n_mates = signed_contrast(Y, plate, pert, mate_ok)
    _, n_mates_any = signed_contrast(Y[:, :1], plate, pert, np.ones(len(Y), bool))      # mate counts before exclusion
    del Y
    ch = np.load(CHROM, allow_pickle=True)
    ccells = [str(c) for c in ch['cells']]
    F, Mk = ch['F'].astype(np.float64), ch['M'].astype(bool)
    cidx = {c: i for i, c in enumerate(ccells)}
    keep = np.array([c in cidx for c in cell])
    cell_i = np.array([cidx.get(c, -1) for c in cell])
    nC = len(ccells)
    # basal: rank-normalised mean control expression per cell (training rows' distinct controls)
    basal = np.full((nC, F.shape[1]), np.nan)
    for c in np.unique(cell[keep]):
        basal[cidx[c]] = ctl[np.unique(ctl_id[cell == c])].mean(0)
    ok_b = np.isfinite(basal).all(1)
    basal[ok_b] = rank01_rows(basal[ok_b])
    has_axis = {ax: Mk[:, :, k].mean(1) > MASK_COVER for ax, k in AXIS.items()}

    sets = {'dnmt_inhibitor': sub == 'dnmt_inhibitor', 'bet_inhibitor': sub == 'bet_inhibitor',
            'bet_inactive_enantiomer': sub == 'bet_inactive_enantiomer', 'hdac_inhibitor': sub == 'hdac_inhibitor',
            'hdac_core': np.isin(pert, list(sens['hdac_core'])), 'bet_with_apabetalone': np.isin(pert, list(sens['bet_with_apabetalone'])),
            'REF': ~np.isin(cls, EXCLUDED_MATE_CLASSES), 'ALL': np.ones(len(pert), bool)}
    mate_report = {}
    for k, sel in sets.items():
        if k in ('REF', 'ALL'):
            continue
        s = sel & keep
        mate_report[k] = {'rows': int(s.sum()), 'rows_lost_mates': int((n_mates[s] < n_mates_any[s]).sum()),
                          'rows_dropped_lt2_mates': int((s & (n_mates < MIN_MATES) & (n_mates_any >= MIN_MATES)).sum())}
    prof = {k: class_profiles(C, np.where(keep, cell_i, 0), nC, sel & keep) for k, sel in sets.items()}

    def cells_with(ax, U):
        return [i for i in range(nC) if has_axis[ax][i] and np.isfinite(U[i]).any() and ok_b[i]]

    def why_not(ax, U):
        """Review 094 C3: every chromatin cell that is not a candidate, with the reason."""
        out = {}
        for i in range(nC):
            if not has_axis[ax][i]:
                out[ccells[i]] = 'no mark'
            elif not np.isfinite(U[i]).any():
                out[ccells[i]] = f'profile < {MIN_ROWS} rows'
            elif not ok_b[i]:
                out[ccells[i]] = 'no basal (not a training cell)'
        return out

    res = {'rule': 'RESULTS 101 / 101.1', 'mates': mate_report,
           'rows': {'training_rows': int(len(tr)), 'with_chromatin_cell': int(keep.sum()),
                    'with_contrast': int(np.isfinite(C[:, 0]).sum())}}
    U_ref = prof['REF'][0]
    prim, rep = {}, {}
    # H-E1: DNMT inhibitors x METH, r > 0
    U = prof['dnmt_inhibitor'][0]
    prim['H-E1'] = evaluate('DNMTi x METH:tss_1kb', F[:, :, FEAT['METH:tss_1kb']], Mk[:, :, AXIS['METH']], U, U_ref, basal,
                            cells_with('METH', U), +1, CI_PRIMARY)
    # H-E2: BET inhibitors x ACT broad, r < 0
    U = prof['bet_inhibitor'][0]
    prim['H-E2'] = evaluate('BETi x ACT:tss_50kb', F[:, :, FEAT['ACT:tss_50kb']], Mk[:, :, AXIS['ACT']], U, U_ref, basal,
                            cells_with('ACT', U), -1, CI_PRIMARY)
    # H-E3: induction of silent genes x ACC prom, r > 0; specificity against repression
    Call = C
    pos, neg = np.where(np.isfinite(Call), np.maximum(Call, 0), np.nan), np.where(np.isfinite(Call), np.maximum(-Call, 0), np.nan)
    I = class_profiles(pos, np.where(keep, cell_i, 0), nC, keep)[0]
    R = class_profiles(neg, np.where(keep, cell_i, 0), nC, keep)[0]
    silent = np.zeros_like(Mk[:, :, 0])
    for i in range(nC):
        if ok_b[i]:
            silent[i] = basal[i] <= np.quantile(basal[i], 1 / 3)
    prim['H-E3'] = evaluate('silent-gene induction x ACC:prom_1kb', F[:, :, FEAT['ACC:prom_1kb']], Mk[:, :, AXIS['ACC']], I, R,
                            basal, cells_with('ACC', I), +1, CI_PRIMARY, gene_sel=silent, flexible_basal=True)
    # reported, never read
    rep['JQ1(-) x ACT:tss_50kb (control)'] = evaluate('JQ1(-) x ACT:tss_50kb', F[:, :, FEAT['ACT:tss_50kb']], Mk[:, :, AXIS['ACT']],
                                                     prof['bet_inactive_enantiomer'][0], U_ref, basal,
                                                     cells_with('ACT', prof['bet_inactive_enantiomer'][0]), -1, CI_REPORTED,
                                                     do_null=False)
    for set_name in ('hdac_inhibitor', 'hdac_core'):
        for f, ax in (('ACC:prom_1kb', 'ACC'), ('ACT:prom_1kb', 'ACT'), ('ACC:tss_50kb', 'ACC'), ('ACT:tss_50kb', 'ACT')):
            U = prof[set_name][0]
            rep[f'{set_name} x {f}'] = evaluate(f'{set_name} x {f}', F[:, :, FEAT[f]], Mk[:, :, AXIS[ax]], U, U_ref, basal,
                                                cells_with(ax, U), +1, CI_REPORTED, do_null=False)
    for f, ax in (('PCG:prom_1kb', 'PCG'), ('METH:tss_1kb', 'METH'), ('ACC:tss_50kb', 'ACC'), ('ACT:prom_1kb', 'ACT')):
        rep[f'silent induction x {f}'] = evaluate(f'silent induction x {f}', F[:, :, FEAT[f]], Mk[:, :, AXIS[ax]], I, R, basal,
                                                  cells_with(ax, I), -1 if ax in ('PCG', 'METH') else +1, CI_REPORTED,
                                                  gene_sel=silent, flexible_basal=True, do_null=False)
    for f in ('ACT:prom_1kb',):                              # narrow against broad for H-E2
        rep[f'bet_inhibitor x {f}'] = evaluate(f'bet_inhibitor x {f}', F[:, :, FEAT[f]], Mk[:, :, AXIS['ACT']],
                                               prof['bet_inhibitor'][0], U_ref, basal, cells_with('ACT', prof['bet_inhibitor'][0]),
                                               -1, CI_REPORTED, do_null=False)
    rep['bet_with_apabetalone x ACT:tss_50kb'] = evaluate('bet_with_apabetalone x ACT:tss_50kb', F[:, :, FEAT['ACT:tss_50kb']],
                                                         Mk[:, :, AXIS['ACT']], prof['bet_with_apabetalone'][0], U_ref, basal,
                                                         cells_with('ACT', prof['bet_with_apabetalone'][0]), -1, CI_REPORTED,
                                                         do_null=False)
    for key, ax, U in (('H-E1', 'METH', prof['dnmt_inhibitor'][0]), ('H-E2', 'ACT', prof['bet_inhibitor'][0]),
                       ('H-E3', 'ACC', I)):
        prim[key]['not_candidates'] = why_not(ax, U)
        prim[key]['eligible_cells'] = [ccells[int(k)] for k in prim[key]['per_cell_r']]
        prim[key]['skipped_lt30_genes'] = {ccells[int(k)]: v for k, v in prim[key]['skipped_lt30_genes'].items()}
    res['primary'], res['reported'] = prim, rep
    res['cells'] = ccells
    res['wall_s'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: {kk: v.get(kk) for kk in ('n_cells', 'mean_r', 'ci', 'delta_spec', 'delta_spec_ci', 'passes',
                                                    'smallest_beta_ci_condition_met')} for k, v in prim.items()}, indent=1))


if __name__ == '__main__':
    main()
