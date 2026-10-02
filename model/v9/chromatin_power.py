# -*- coding: utf-8 -*-
"""RESULTS 91.9 M2: can the chromatin funnel detect a chromatin effect at all? Planted-effect power calibration (PI-written).

    python model/v9/chromatin_power.py --data_dir <dir> --provenance <json> --out <json> [--draws 5] [--workers 1]

A synthetic feature f* carries real chromatin's redundancy with basal expression and NO link to the response:
    f*_c = rho * b_c + sqrt(1 - rho^2) * xi_c,
xi_c = the cell's own primary-encoded chromatin track (H3K27ac, else ATAC, else H3K27me3) with genes reordered by ONE permutation
shared across cells (between-cell structure kept, gene link destroyed); rho = median within-cell r(track, b) over covered dev-train
cells (a feature-feature statistic). Effects of known size are planted through f* on every covered row (dev and dev-train) and
T1 (P1 gain, P2 drug-specific shift, P3 drug-independent shift) and T3 (P4 responsiveness) are rerun on the test AS IT WILL BE RUN
(review 041 C2): FBC* = the real 4-column feature set with the primary mark replaced by f* and every other mark a gene-permuted copy
(availability unchanged), N1* built from it, the rows of record and their bars (cf.BARS, cf.ROW_SET_OF_RECORD). Real chromatin
enters only through rho and through gene-permuted copies, so this calibration reveals nothing about the real reading.

Planting uses mu^(-c) from the original y (pool = all dev-train cells, own cell excluded; dev rows: all dev-train cells), the
same drug mean T1 fits against, so the planted gain has exactly the form T1 can express. Each effect is scaled so its variance is
pi x Var(e), e = y - mu^(-c), over covered dev-train rows x genes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402

PIS = (0.005, 0.01, 0.02, 0.05, 0.10)
FORMS_T1 = ('P1', 'P2', 'P3')
G = cf.G
_CTX = None                                      # set before forking workers (Linux); never mutated after


def base_track(ctx):
    e = ctx.enc['rank_normal']
    out = {}
    for c in e['cov_dt'] + e['cov_dev']:
        i = ctx.cells.index(c)
        k = 1 if e['has'][i, 1] else (0 if e['has'][i, 0] else 2)
        out[c] = e['Ez'][i, :, k].astype(np.float64)
    return out


def synthetic_feature(ctx, draw):
    """{cell: f* [G]} for covered cells (primary encoding), and rho."""
    e = ctx.enc['rank_normal']
    tr = base_track(ctx)
    rho = float(np.median([np.corrcoef(tr[c], e['b'][ctx.cells.index(c)])[0, 1] for c in e['cov_dt']]))
    perm = np.random.default_rng(1000 + draw).permutation(G)
    f = {}
    for c, v in tr.items():
        b = e['b'][ctx.cells.index(c)].astype(np.float64)
        f[c] = rho * b + np.sqrt(max(1 - rho * rho, 0.0)) * v[perm]
    return f, rho


CALIB = 'calibration'                            # ctx.enc key for the shape-matched synthetic feature set FBC* (per draw)


def install_calibration_tracks(ctx, fstar, draw):
    """Review 041 C2: calibrate the test AS IT WILL BE RUN. FBC* = each covered cell's real 4-column feature set with its primary
    mark (K, else A, else M) replaced by f* and every other mark it has replaced by a gene-permuted copy of itself (one
    permutation per mark, shared across cells); mark availability unchanged. Installed as ctx.enc[CALIB] so the funnel's own
    feature_builder builds FBC* and N1* exactly as it builds FBC and N1."""
    e = ctx.enc['rank_normal']
    Ez = np.zeros_like(e['Ez'])
    perms = [np.random.default_rng(3000 + 10 * draw + k).permutation(G) for k in range(3)]
    for i, c in enumerate(ctx.cells):
        if c not in fstar:
            continue
        k0 = 1 if e['has'][i, 1] else (0 if e['has'][i, 0] else 2)
        for k in range(3):
            if e['has'][i, k]:
                Ez[i, :, k] = fstar[c] if k == k0 else e['Ez'][i, perms[k], k]
    ctx.enc[CALIB] = {'b': e['b'], 'Ez': Ez.astype(np.float32), 'has': e['has'].copy(), 'cov_dt': list(e['cov_dt']),
                      'cov_dev': list(e['cov_dev'])}
    return ctx.enc[CALIB]


def builders(ctx, fstar, fit, draw=0):
    """FB = [b]; FBC* and N1* from the shape-matched calibration tracks (same builders the real test uses)."""
    install_calibration_tracks(ctx, fstar, draw)
    return {'FB': cf.feature_builder(ctx, 'rank_normal', 'FB', fit), 'FBC': cf.feature_builder(ctx, CALIB, 'FBC', fit),
            'N1': cf.feature_builder(ctx, CALIB, 'N1', fit)}


def plant(ctx, y, mu_e, fstar, form, pi, draw):
    """y' = y + alpha * component on covered rows; alpha scales Var(component) to pi * Var(e) over covered dev-train rows."""
    covered = np.isin(ctx.rows['cell'], list(fstar))
    R = np.flatnonzero(covered)
    Fr = np.stack([fstar[c] for c in ctx.rows['cell'][R]])
    if form == 'P1':
        comp = mu_e[R] * Fr
    elif form == 'P2':
        w = np.random.default_rng(2000 + draw).normal(size=len(ctx.perts))
        comp = w[ctx.pert_id[R]][:, None] * Fr
    elif form == 'P3':
        comp = Fr
    elif form == 'P4':
        comp = (y[R] - mu_e[R]) * Fr
    else:
        raise ValueError(form)
    dt = ctx.fit_mask[R]
    var_e = float(np.var((y[R] - mu_e[R])[dt]))
    alpha = float(np.sqrt(pi * var_e / np.var(comp[dt])))
    out = y.copy()
    out[R] = (y[R] + alpha * comp).astype(np.float32)
    return out, alpha


def residual_delta(ctx, y, mu_dev, yh_a, yh_b, sub=None):
    """Reported only: per-row Pearson(y_hat - mu, y - mu) for a minus b on dev rows (or the subset), all-row mean and per cell."""
    dev = np.flatnonzero(ctx.dev_mask)
    a, b, m = np.asarray(yh_a), np.asarray(yh_b), mu_dev
    if sub is not None:
        dev, a, b, m = dev[sub], a[sub], b[sub], m[sub]
    t = y[dev].astype(np.float64) - m
    d = cf.row_pearson(a - m, t) - cf.row_pearson(b - m, t)
    cells = ctx.rows['cell'][dev]
    return {'all': float(np.nanmean(d)), 'per_cell': {c: float(np.nanmean(d[cells == c])) for c in cf.DEV_CELLS}}


ROW_SETS = ('all', 'known')                      # all dev rows (91.2); drug-known rows (back-off level <= 2), packet 041


def t1_case(ctx, y, B, fit):
    specs = {k: (B[k], ['full']) for k in ('FB', 'FBC', 'N1')}
    out, mu, level = cf.run_t1(ctx, y, specs, fit)
    known = level[ctx.dev_mask] <= 2
    rec = {'kappa': {k: [out[(k, 'full')]['kappa'], out[(k, 'full')]['kappa_d']] for k in ('FB', 'FBC', 'N1')}}
    for name, sub in (('all', None), ('known', known)):
        s = {k: cf.score(ctx, y, out[(k, 'full')]['y_hat_dev'], sub) for k in ('FB', 'FBC', 'N1')}
        rd = cf.t1_reading(s['FBC'], s['FB'], s['N1'], cf.BARS[name])
        rec[name] = {'pass': rd['pass'], 'conjuncts': rd['conjuncts'], 'delta_all': rd['delta']['all'],
                     'delta_top': rd['delta']['top'], 'centred_all': rd['delta']['centred_all'], 'cells_pos': rd['delta']['cells_pos'],
                     'centred_cells_pos': rd['delta']['centred_cells_pos'], 'cells_vs_N1_pos': rd['vs_N1']['cells_pos'],
                     'resid': residual_delta(ctx, y, mu[ctx.dev_mask], out[('FBC', 'full')]['y_hat_dev'],
                                             out[('FB', 'full')]['y_hat_dev'], sub)}
    return rec


def t3_case(ctx, y, B, fit):
    o, _ = cf.run_t3(ctx, y, {'FB': B['FB'], 'FBC': B['FBC'], 'N1': B['N1']}, fit, ctx.enc['rank_normal']['cov_dev'])
    rd = cf.t3_reading(o['FBC']['rho'], o['FB']['rho'], o['N1']['rho'])
    return {'pass': rd['pass'], 'conjuncts': rd['conjuncts'], 'rho_T3': rd['rho_T3'], 'FBC_minus_N1': rd['FBC_minus_N1']}


def one_draw(draw):
    ctx = _CTX
    fit = ctx.enc['rank_normal']['cov_dt']
    fstar, rho = synthetic_feature(ctx, draw)
    B = builders(ctx, fstar, fit, draw)
    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    rec = {'draw': draw, 'rho': rho, 'null_T1': t1_case(ctx, ctx.y, B, fit), 'T1': {}, 'T3': {}}
    for form in FORMS_T1:
        for pi in PIS:
            yp, alpha = plant(ctx, ctx.y, mu_e, fstar, form, pi, draw)
            r = t1_case(ctx, yp, B, fit)
            r['alpha'] = alpha
            rec['T1']['%s_%g' % (form, pi)] = r
    for pi in PIS:
        yp, alpha = plant(ctx, ctx.y, mu_e, fstar, 'P4', pi, draw)
        r = t3_case(ctx, yp, B, fit)
        r['alpha'] = alpha
        rec['T3']['P4_%g' % pi] = r
    return rec


def passed(r, test, key, rowset):
    x = r[test][key]
    return x[rowset]['pass'] if test == 'T1' else x['pass']      # T3 is per-cell gene spreads, not row-based


def mde(records, test, form, rowset='all'):
    """Smallest pi with a pass in >= 4 of 5 draws (scaled: >= 80 % of draws); None if no pi qualifies."""
    for pi in PIS:
        passes = [passed(r, test, '%s_%g' % (form, pi), rowset) for r in records]
        if sum(passes) >= int(np.ceil(0.8 * len(passes))):
            return pi
    return None


def label(m):
    return ('informative null (MDE <= 2 %; linear in the encoded tracks)' if m is not None and m <= 0.02 else
            'not measurable with this design' if m is None or m > 0.05 else 'reported with its MDE')


def summarise(records):
    out = {}
    for rs in ROW_SETS:
        o = {'pass_counts': {}, 'MDE': {}, 'reading': {}}
        for test, forms in (('T1', FORMS_T1), ('T3', ('P4',))):
            for form in forms:
                o['pass_counts']['%s_%s' % (test, form)] = {('%g' % pi): int(sum(passed(r, test, '%s_%g' % (form, pi), rs)
                                                                                 for r in records)) for pi in PIS}
                o['MDE']['%s_%s' % (test, form)] = mde(records, test, form, rs)
        false_pass = int(sum(r['null_T1'][rs]['pass'] for r in records))
        p3_any = any(r['T1']['P3_%g' % pi][rs]['pass'] for r in records for pi in PIS)
        o['instrument_faults'] = {'T1_pass_at_pi_0_draws': false_pass, 'T1_passes_P3_anywhere': bool(p3_any),
                                  'T1_void': bool(false_pass > 0 or p3_any)}
        for key in ('T1_P1', 'T1_P2', 'T3_P4'):
            o['reading'][key] = label(o['MDE'][key])
        o['residual_estimand_null_max_abs'] = float(max(abs(r['null_T1'][rs]['resid']['all']) for r in records))
        out[rs] = o
    out['row_set_of_record'] = cf.ROW_SET_OF_RECORD
    return out


def main():
    global _CTX
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    t = time.time()
    _CTX = cf.prepare(a.data_dir, a.provenance)
    assert len(_CTX.enc['rank_normal']['cov_dt']) == 11
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            records = pool.map(one_draw, range(a.draws))
    else:
        records = [one_draw(d) for d in range(a.draws)]
    res = {'records': records, 'summary': summarise(records), 'pis': PIS, 'draws': a.draws, 'inputs': _CTX.inputs,
           'seconds': time.time() - t}
    json.dump(cf.jsonable(res), open(a.out, 'w'), indent=1)
    print('wrote', a.out, json.dumps(cf.jsonable(res['summary'])), flush=True)


if __name__ == '__main__':
    main()
