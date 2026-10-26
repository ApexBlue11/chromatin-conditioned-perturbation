"""Stratified readings of two 3-slot arms on aligned held-out rows (RESULTS 100.13, amended by 101.1 C3; used by T-C's and
v11's readers). Reported, exploratory, never a claim.

Input: R_a, R_b [3, n] per-row contrast r of each arm's slots on the same rows (NaN where a row has < 2 mates; the mask is
the same in every slot because the rows and plates are). Per row, Delta_row = mean_a - mean_b.

  per group (a cell, a class, ...): Delta = the row mean of Delta_row; SE = sqrt(s_a^2 / 3 + s_b^2 / 3) from the per-slot
      group means; z = Delta / SE (seed noise)
  per class, also: a compound-cluster bootstrap 95 % CI (compounds resampled within the class, 20,000 draws,
      default_rng(0); compound heterogeneity)
  per compound with >= 10 rows (101.1 C3):
      the variance of a compound's per-slot mean is fitted across all compounds as v(n) = a + b / n (n = its rows) by
      weighted least squares on the within-arm slot variances (the mean of the two arms' ddof-1 variances; IRLS with
      weights 1 / v_hat^2, the inverse variance of a variance estimate up to a constant; a, b >= 0);
      SE_k = sqrt(2 v(n_k) / 3); t on 4 df; two-sided p; Benjamini-Hochberg at q = 0.10;
      the list-level calibration: the whole procedure rerun under each of the 9 other balanced 3 + 3 relabellings of the
      6 slots, with each one's discovery count reported beside the real list.
"""
import itertools

import numpy as np
from scipy import stats

N_BOOT, MIN_ROWS_COMPOUND, Q_BH, DF = 20000, 10, 0.10, 4


def finite_rows(R_a, R_b):
    ok = np.isfinite(R_a).all(0) & np.isfinite(R_b).all(0)
    if not np.array_equal(ok, np.isfinite(R_a[0])):
        raise ValueError('the contrast mask differs between slots')
    return ok


def group_stats(R_a, R_b, sel):
    """Delta, SE, z and n_rows over rows `sel` (boolean). The 'z' is a Welch t on the 3 + 3 slot means (review 100 C1),
    so its df (about 4) and two-sided t p are reported with it, and the exact 3 + 3 relabelling floor (0.10) applies."""
    ma, mb = R_a[:, sel].mean(1), R_b[:, sel].mean(1)
    d = float(ma.mean() - mb.mean())
    va, vb = ma.var(ddof=1) / len(ma), mb.var(ddof=1) / len(mb)
    se = float(np.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va ** 2 / (len(ma) - 1) + vb ** 2 / (len(mb) - 1))) if va + vb > 0 else None
    t = d / se if se > 0 else None
    return {'delta': d, 'se': se, 'z': t, 'welch_df': df,
            'p_t_two_sided': float(2 * stats.t.sf(abs(t), df)) if t is not None and df else None,
            'all_a_above_all_b': bool(ma.min() > mb.max()), 'all_b_above_all_a': bool(mb.min() > ma.max()),
            'n_rows': int(sel.sum()), 'slot_means_a': ma.tolist(), 'slot_means_b': mb.tolist()}


def by_group(R_a, R_b, groups, min_rows=1):
    ok = finite_rows(R_a, R_b)
    out = {}
    for g in np.unique(groups[ok]):
        sel = ok & (groups == g)
        if sel.sum() >= min_rows:
            out[str(g)] = group_stats(R_a, R_b, sel)
    return out


def cluster_ci(delta_row, comp, n_boot=N_BOOT, seed=0):
    """Compound-cluster bootstrap 95 % CI of the row mean of delta_row (compounds resampled with replacement)."""
    u, inv = np.unique(comp, return_inverse=True)
    s = np.bincount(inv, weights=delta_row, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def by_class(R_a, R_b, cls, comp, n_boot=N_BOOT):
    ok = finite_rows(R_a, R_b)
    dr = R_a.mean(0) - R_b.mean(0)
    out = by_group(R_a, R_b, cls)
    for g, rec in out.items():
        sel = ok & (cls == g)
        rec['ci95_compound_cluster'] = cluster_ci(dr[sel], comp[sel], n_boot)
        rec['n_compounds'] = int(len(np.unique(comp[sel])))
    return out


def fit_variance(v, n, iters=5):
    """v_k ~ a + b / n_k by IRLS (weights 1 / fitted^2), a, b >= 0. Returns (a, b)."""
    X = np.stack([np.ones_like(n, dtype=float), 1.0 / n], 1)
    w = np.ones(len(v))
    a = b = 0.0
    for _ in range(iters):
        W = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * W[:, None], v * W, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if a < 0 or b < 0:                       # refit the single nonnegative term that fits better
            fits = []
            for j in (0, 1):
                c = float(np.sum(w * X[:, j] * v) / np.sum(w * X[:, j] ** 2))
                c = max(c, 0.0)
                fits.append((np.sum(w * (v - c * X[:, j]) ** 2), j, c))
            _, j, c = min(fits)
            a, b = (c, 0.0) if j == 0 else (0.0, c)
        fitted = np.maximum(a + b / n, 1e-12)
        w = 1.0 / fitted ** 2
    return a, b


def bh(p, q=Q_BH):
    """Benjamini-Hochberg: boolean discoveries at level q."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    thr = q * np.arange(1, m + 1) / m
    below = p[order] <= thr
    k = int(np.max(np.flatnonzero(below)) + 1) if below.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def compound_test(S_a, S_b, n, q=Q_BH):
    """S_a, S_b [3, K] per-slot compound means; n [K] rows. Returns the per-compound table and the discoveries."""
    v = (S_a.var(0, ddof=1) + S_b.var(0, ddof=1)) / 2
    a, b = fit_variance(v, n.astype(float))
    se = np.sqrt(2 * np.maximum(a + b / n, 1e-12) / 3)
    d = S_a.mean(0) - S_b.mean(0)
    t = d / se
    p = 2 * stats.t.sf(np.abs(t), DF)
    return {'a': a, 'b': b, 'delta': d, 'se': se, 't': t, 'p': p, 'discovered': bh(p, q)}


def by_compound(R_a, R_b, comp, labels=None, min_rows=MIN_ROWS_COMPOUND, q=Q_BH):
    """101.1 C3 per compound, with the 9 other balanced relabellings as the list-level calibration. `labels`: optional
    {compound: dict} attached to each discovered compound (its class, its cells)."""
    ok = finite_rows(R_a, R_b)
    u, inv = np.unique(comp[ok], return_inverse=True)
    n = np.bincount(inv, minlength=len(u))
    keep = n >= min_rows
    R = np.concatenate([R_a, R_b], 0)[:, ok]                     # [6, rows]
    sums = np.stack([np.bincount(inv, weights=R[s], minlength=len(u)) for s in range(6)])
    S = (sums / np.maximum(n, 1))[:, keep]                       # [6, K]
    nk, names = n[keep], u[keep]
    real = compound_test(S[:3], S[3:], nk, q)
    calib = []
    for A in itertools.combinations(range(6), 3):
        if 0 not in A or A == (0, 1, 2):                         # each unordered split once; the real one apart
            continue
        B = tuple(i for i in range(6) if i not in A)
        r = compound_test(S[list(A)], S[list(B)], nk, q)
        calib.append({'a_slots': list(A), 'b_slots': list(B), 'discoveries': int(r['discovered'].sum())})
    disc = []
    for k in np.flatnonzero(real['discovered']):
        e = {'compound': str(names[k]), 'delta': float(real['delta'][k]), 't': float(real['t'][k]),
             'p': float(real['p'][k]), 'n_rows': int(nk[k])}
        if labels is not None:
            e.update(labels.get(str(names[k]), {}))
        disc.append(e)
    return {'n_compounds_tested': int(keep.sum()), 'min_rows': min_rows, 'variance_fit': {'a': real['a'], 'b': real['b']},
            'discoveries': sorted(disc, key=lambda e: e['p']), 'n_discoveries': len(disc),
            'relabelling_calibration': calib, 'df': DF, 'q': q,
            'note': ('t on 4 df with an SE pooled over all tested compounds is conservative (it loses some power); it is '
                     'what 101.1 C3 registered (review 099 Ask 9)')}
