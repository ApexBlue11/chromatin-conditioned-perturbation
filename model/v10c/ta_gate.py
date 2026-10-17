"""T-A, the CPU gate (RESULTS 100.6, on review 086's design). Reported; it gates nothing.

Per (cell, gene) responsiveness R_cg = mean over the cell's unseen-cell TRAINING rows of |within-plate contrast| (other-compound
training plate-mates, >= 2), cells with >= 200 such rows, standardised per gene across cells. Leave-cell-out ridge:
  base = gene fixed effect + basal level (the cell's mean control expression, rank-normalised within the cell)
  full = base + the 16 chromatin features + the 4-channel mask
Score: pooled leave-cell-out R^2 gain (full - base) over the cells with chromatin. Nulls: cell-swap and gene-permute.
Calibration: planted R += beta * z(accessibility).

    python model/v10c/ta_gate.py --substrate ... --chromatin ... --out model/results/v10c/ta_gate.json
"""
import argparse, json, os, time

import numpy as np

MIN_MATES, MIN_ROWS = 2, 200
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def responsiveness(Y, plate, pert, cell, min_mates=MIN_MATES):
    """Y [N, G] deltas of training rows. Returns (cells, R [C, G], n_rows [C]) with R the mean |contrast|."""
    order = np.lexsort((pert, plate))
    cells = np.unique(cell)
    cidx = {c: i for i, c in enumerate(cells)}
    acc = np.zeros((len(cells), Y.shape[1]))
    n = np.zeros(len(cells), np.int64)
    ps = plate[order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        rows = order[s:e]
        if len(rows) < min_mates + 1:
            continue
        Yp = Y[rows].astype(np.float64)
        pr = pert[rows]
        tot, ntot = Yp.sum(0), len(rows)
        _, inv, cnt = np.unique(pr, return_inverse=True, return_counts=True)
        same = np.zeros((cnt.size, Y.shape[1]))
        np.add.at(same, inv, Yp)
        k = ntot - cnt[inv]
        ok = k >= min_mates
        if not ok.any():
            continue
        mates = (tot[None, :] - same[inv]) / np.maximum(k, 1)[:, None]
        c = np.abs(Yp - mates)[ok]
        for ci in np.unique(cell[rows][ok]):
            m = cell[rows][ok] == ci
            acc[cidx[ci]] += c[m].sum(0)
            n[cidx[ci]] += int(m.sum())
    R = acc / np.maximum(n, 1)[:, None]
    return cells, R, n


def rank01_rows(A):
    r = np.argsort(np.argsort(A, 1), 1).astype(np.float64)
    return r / max(1, A.shape[1] - 1)


def loco_r2(R, basal, chrom, use_chrom, lam, eval_cells):
    """Leave-cell-out ridge with gene fixed effects (demeaned by the training cells' gene means). R, basal [C, G];
    chrom [C, G, P]. Returns pooled R^2 over the eval cells' (cell, gene) entries."""
    C, G = R.shape
    X = basal[..., None] if not use_chrom else np.concatenate([basal[..., None], chrom], -1)
    ss_res = ss_tot = 0.0
    for c in eval_cells:
        tr = np.ones(C, bool)
        tr[c] = False
        gm_R, gm_X = R[tr].mean(0), X[tr].mean(0)
        Xt = (X[tr] - gm_X).reshape(-1, X.shape[-1])
        yt = (R[tr] - gm_R).ravel()
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ yt)
        pred = gm_R + (X[c] - gm_X) @ w
        ss_res += float(((R[c] - pred) ** 2).sum())
        ss_tot += float(((R[c] - gm_R) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def gain(R, basal, chrom, eval_cells, lam):
    return loco_r2(R, basal, chrom, True, lam, eval_cells) - loco_r2(R, basal, chrom, False, lam, eval_cells)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def centre_cells(A):
    """Remove each cell's mean over genes (for [C, G] or [C, G, P]): what is left is the within-cell gene pattern."""
    return A - A.mean(1, keepdims=True)


def run(R, basal, chrom, has_chrom, n_null=200, seed=0, betas=(0.02, 0.05, 0.1, 0.2), acc_feature=0, n_null_cal=50,
        centre=False):
    rng = np.random.default_rng(seed)
    Rz = (R - R.mean(0)) / (R.std(0) + 1e-9)
    if centre:   # POST HOC diagnostic (100.6a): the within-cell gene pattern only, cell-level levels removed
        Rz, basal, chrom = centre_cells(Rz), centre_cells(basal), centre_cells(chrom)
    eval_cells = np.flatnonzero(has_chrom)
    lam_scores = {lam: loco_r2(Rz, basal, chrom, False, lam, np.arange(len(Rz))) for lam in LAMBDAS}
    lam = max(lam_scores, key=lam_scores.get)
    g = gain(Rz, basal, chrom, eval_cells, lam)

    def null(kind, Rv, n):
        out = []
        for _ in range(n):
            if kind == 'cell_swap':
                ch = chrom[derangement(len(chrom), rng)]
            else:
                ch = np.stack([chrom[c][rng.permutation(chrom.shape[1])] for c in range(len(chrom))])
            out.append(gain(Rv, basal, ch, eval_cells, lam))
        return np.array(out)

    cs, gp = null('cell_swap', Rz, n_null), null('gene_permute', Rz, n_null)
    acc = chrom[..., acc_feature]
    accz = (acc - acc.mean()) / (acc.std() + 1e-9)
    cal = {}
    for b in betas:
        Rp = Rz + b * accz * has_chrom[:, None]
        gb = gain(Rp, basal, chrom, eval_cells, lam)
        nb = null('cell_swap', Rp, n_null_cal)
        cal[str(b)] = {'gain': gb, 'cell_swap_p95': float(np.percentile(nb, 95)), 'detected': bool(gb > np.percentile(nb, 95))}
    return {'lambda': lam, 'lambda_cv_r2_base': {str(k): v for k, v in lam_scores.items()}, 'n_cells': int(len(R)),
            'n_eval_cells': int(len(eval_cells)), 'gain': g,
            'cell_swap_null': {'mean': float(cs.mean()), 'p95': float(np.percentile(cs, 95)), 'n': int(len(cs))},
            'gene_permute_null': {'mean': float(gp.mean()), 'p95': float(np.percentile(gp, 95)), 'n': int(len(gp))},
            'passes': bool(g > np.percentile(cs, 95)),
            'calibration': cal,
            'smallest_detected_beta': min([float(b) for b in cal if cal[b]['detected']], default=None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--substrate', required=True)
    ap.add_argument('--chromatin', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_null', type=int, default=200)
    ap.add_argument('--centre', action='store_true', help='POST HOC diagnostic: centre R, basal and chromatin per cell')
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- touched once')
    t0 = time.time()
    z = np.load(a.substrate)
    tr = z['split_cold_cell_1'] == 'train'
    ctl, ctl_id = z['ctl'], z['ctl_id'][tr]
    X = z['X'][tr]
    Y = X - ctl[ctl_id]
    del X
    cell, pert, plate = z['meta_cell_iname'][tr], z['meta_pert_id'][tr], z['plate_id'][tr]
    cells, R, n = responsiveness(Y, plate, pert, cell)
    del Y
    keep = n >= MIN_ROWS
    cells, R, n = cells[keep], R[keep], n[keep]
    basal = np.stack([ctl[np.unique(ctl_id[cell == c])].mean(0) for c in cells])
    basal = rank01_rows(basal)
    ch = np.load(a.chromatin)
    cidx = {str(c): i for i, c in enumerate(ch['cells'])}
    F, M = ch['F'][[cidx[str(c)] for c in cells]], ch['M'][[cidx[str(c)] for c in cells]].astype(np.float64)
    chrom = np.concatenate([F, M], -1)
    has = M[:, :, :3].any((1, 2))
    rec = run(R, basal, chrom, has, n_null=a.n_null, centre=a.centre)
    rec['centred_post_hoc'] = bool(a.centre)
    rec.update({'rule': 'RESULTS 100.6', 'cells': [str(c) for c in cells], 'rows_per_cell': {str(c): int(k) for c, k in zip(cells, n)},
                'cells_with_chromatin': int(has.sum()), 'wall_s': round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w'), indent=1)
    print(json.dumps({k: rec[k] for k in ('n_cells', 'n_eval_cells', 'lambda', 'gain', 'cell_swap_null', 'gene_permute_null',
                                          'passes', 'smallest_detected_beta')}, indent=1))


if __name__ == '__main__':
    main()
