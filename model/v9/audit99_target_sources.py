# -*- coding: utf-8 -*-
"""RESULTS 99 / model/V10_AUDIT_AND_PLAN.md, findings A0-A3: what the delta target y = X - X_ctl is made of.

TRAINING ROWS ONLY (split_cold_drug_1 and split_cold_cell_1). No model, no prediction, no test row is read.
The bundle has no plate key, so the PLATE PROXY is the control group: rows whose X_ctl vectors are byte-identical.

  A0  row structure: rows, compounds, cells, rows per (drug, cell), control sharing.
  A1  per-row Pearson (978 genes) of y with
        cell-mean delta                        (drug-free)
        F0 = -(X_ctl - cell mean control) + cell-mean delta   (drug-free, RESULTS 98.6)
        P  = mean y of OTHER-drug rows on the same control     (drug-free, plate)
        D  = mean y of the same (drug, cell) on OTHER controls (drug information, other plates)
        P + D - cell mean
  A2  rows with >= 2 other-drug wells on the plate and the same (drug, cell, time) on another plate:
        P, and P + w (D - cell mean) for w in a grid  -> the drug's increment over the plate predictor;
        the PLATE-CONTRAST target yc = X - mean X of the other-drug wells on the plate (CMap's Level-4 logic):
        how well the same (drug, cell, time) on other plates predicts it, by strength tercile, and its
        drug-free predictor (the cell mean of yc).
  A2b (review 083 C1 / C2, added 2026-10-14): the same drug term over a MODEL-AVAILABLE drug-free baseline, F0, on rows
      with >= 2 plate-mates and on rows with none; and the zero-parameter DOSE-ONLY floor on the plate-contrast target
      (mean y of rows at the same (cell, time, dose) on another control, contrasted over the plate-mates). Labels per
      review 083 C3: the bundle has one row per (drug, cell, time, dose), so D is the same drug at OTHER DOSES, and
      0.147 is not a replicate ceiling.
  A3  split_cold_cell_1 training rows: -(X_ctl - cell mean control) alone, the cell-mean delta alone, both.

Run:  python model/v9/audit99_target_sources.py [--bundle B] [--out model/results/audit99/target_sources.json]
"""
import argparse, hashlib, json, os, sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE = os.path.join(REPO, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz')
OUT = os.path.join(REPO, 'model', 'results', 'audit99', 'target_sources.json')
W_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rowr(a, t):
    """Per-row Pearson over genes. Rows with zero variance in either argument give NaN."""
    a = a - a.mean(1, keepdims=True)
    t = t - t.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (t * t).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (a * t).sum(1) / np.where(den > 0, den, 1), np.nan)


def group_ids(*keys):
    """Dense integer id per row for the tuple of keys."""
    return np.unique(np.stack([np.unique(k, return_inverse=True)[1] for k in keys], 1), axis=0,
                     return_inverse=True)[1].ravel()


def group_sum(gid, V):
    n_g = gid.max() + 1
    S = np.zeros((n_g, V.shape[1]))
    np.add.at(S, gid, V)
    return S, np.bincount(gid, minlength=n_g).astype(np.float64)


def loo_other(V, outer, inner):
    """Mean of V over rows sharing `outer` but NOT sharing `inner` (inner nests inside outer).
    Returns (mean, count); mean is NaN where count == 0."""
    So, no = group_sum(outer, V)
    Si, ni = group_sum(inner, V)
    cnt = no[outer] - ni[inner]
    with np.errstate(invalid='ignore', divide='ignore'):
        M = (So[outer] - Si[inner]) / cnt[:, None]
    M[cnt == 0] = np.nan
    return M, cnt


def control_ids(C):
    return np.unique(np.ascontiguousarray(C).view(np.dtype((np.void, C.dtype.itemsize * C.shape[1]))).ravel(),
                     return_inverse=True)[1].ravel()


def nanmean(v):
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float('nan')


def a0_a2(b, split_key):
    tr = b[split_key] == 'train'
    X, C = b['X'][tr].astype(np.float64), b['X_ctl'][tr].astype(np.float64)
    y = X - C
    pert, cell, time = b['meta_pert_id'][tr], b['meta_cell'][tr], b['meta_time'][tr]
    ctl = control_ids(b['X_ctl'][tr])
    n = len(y)
    res = {'split': split_key, 'n_train_rows': int(n)}

    # A0
    pc = group_ids(pert, cell)
    n_pc = np.bincount(pc)
    n_ctl = np.bincount(ctl)
    res['A0'] = {'n_compounds': int(len(np.unique(pert))), 'n_cells': int(len(np.unique(cell))),
                 'rows_per_drug_cell_median': float(np.median(n_pc)),
                 'frac_drug_cell_with_ge2_rows': float((n_pc >= 2).mean()),
                 'n_distinct_controls': int(len(n_ctl)),
                 'frac_rows_sharing_control': float((n_ctl[ctl] > 1).mean())}

    # A1
    cg = group_ids(cell)
    Sc, nc = group_sum(cg, y)
    Mcell = Sc[cg] / nc[cg][:, None]
    Cc, _ = group_sum(cg, C)
    F0 = -(C - Cc[cg] / nc[cg][:, None]) + Mcell
    P1, nP1 = loo_other(y, ctl, group_ids(ctl, pert))          # other drugs, same control
    D1, nD1 = loo_other(y, pc, group_ids(pert, cell, ctl))     # same drug+cell, other controls
    ok = (nP1 > 0) & (nD1 > 0)
    a1 = {'cell_mean_delta': rowr(Mcell, y), 'F0': rowr(F0, y), 'P_other_drugs_same_control': rowr(P1, y),
          'D_same_drug_cell_other_controls': rowr(D1, y), 'P_plus_D_minus_cell_mean': rowr(P1 + D1 - Mcell, y)}
    res['A1'] = {'n_rows_with_P_and_D': int(ok.sum()),
                 'all_available': {k: nanmean(v) for k, v in a1.items()},
                 'common_rows': {k: nanmean(v[ok]) for k, v in a1.items()}}
    s = np.linalg.norm(y, axis=1)
    q = np.quantile(s[ok], [1 / 3, 2 / 3])
    terc = {}
    for lo, hi, name in [(0, q[0], 'weak'), (q[0], q[1], 'mid'), (q[1], np.inf, 'strong')]:
        m = ok & (s >= lo) & (s < hi)
        terc[name] = {'n': int(m.sum()), **{k: nanmean(a1[k][m]) for k in
                      ('P_other_drugs_same_control', 'D_same_drug_cell_other_controls', 'P_plus_D_minus_cell_mean')}}
    res['A1']['by_delta_norm_tercile'] = terc

    # A2
    pct = group_ids(pert, cell, time)
    D2, nD2 = loo_other(y, pct, group_ids(pert, cell, time, ctl))
    Xp, nXp = loo_other(X, ctl, group_ids(ctl, pert))         # mean X of the other-drug wells on the plate
    ok2 = (nP1 >= 2) & (nD2 > 0)
    a2 = {'n_rows': int(ok2.sum()), 'P': nanmean(rowr(P1[ok2], y[ok2])),
          'P_plus_w_D_minus_cell_mean': {str(w): nanmean(rowr(P1[ok2] + w * (D2[ok2] - Mcell[ok2]), y[ok2]))
                                         for w in W_GRID}}
    best_w = max(a2['P_plus_w_D_minus_cell_mean'], key=a2['P_plus_w_D_minus_cell_mean'].get)
    a2['best_w'] = float(best_w)
    a2['drug_increment_over_P'] = a2['P_plus_w_D_minus_cell_mean'][best_w] - a2['P']
    yc = X - Xp
    has_yc = nP1 >= 2
    yc_masked = np.where(has_yc[:, None], yc, 0.0)
    inner = group_ids(pert, cell, time, ctl)
    Sp, _ = group_sum(pct, yc_masked)
    Si, _ = group_sum(inner, yc_masked)
    cnt = np.bincount(pct, weights=has_yc.astype(float))[pct] - np.bincount(inner, weights=has_yc.astype(float))[inner]
    with np.errstate(invalid='ignore', divide='ignore'):
        Dc = (Sp[pct] - Si[inner]) / cnt[:, None]
    okc = has_yc & (cnt > 0)
    rc = rowr(Dc[okc], yc[okc])
    sc = np.linalg.norm(yc[okc], axis=1)
    qc = np.quantile(sc, [1 / 3, 2 / 3])
    cgo = group_ids(cell[okc])
    Sco, nco = group_sum(cgo, yc[okc])
    a2['plate_contrast_target'] = {
        'n_rows': int(okc.sum()),
        'same_drug_cell_time_other_plates_r_mean': nanmean(rc), 'median': float(np.nanmedian(rc)),
        'by_strength_tercile': [nanmean(rc[(sc >= lo) & (sc < hi)]) for lo, hi in
                                [(0, qc[0]), (qc[0], qc[1]), (qc[1], np.inf)]],
        'drug_free_cell_mean_r': nanmean(rowr(Sco[cgo] / nco[cgo][:, None], yc[okc]))}
    res['A2'] = a2

    # A2b -- review 083 C1: the drug term over F0 (no plate information), with and without plate-mates
    a2b = {'label': 'D = the same (drug, cell, time) at OTHER doses on other plates (no replicates exist in the bundle)'}
    for name, m in (('ge2_plate_mates', (nP1 >= 2) & (nD2 > 0)), ('no_plate_mate', (nP1 == 0) & (nD2 > 0))):
        base = nanmean(rowr(F0[m], y[m]))
        grid = {str(w): nanmean(rowr(F0[m] + w * (D2[m] - Mcell[m]), y[m])) for w in W_GRID + (1.5, 2.0)}
        bw = max(grid, key=grid.get)
        a2b['over_F0_' + name] = {'n_rows': int(m.sum()), 'F0': base, 'best_w': float(bw), 'F0_plus_wD': grid[bw],
                                  'increment': grid[bw] - base}
    # review 083 C2: the dose-only floor on the plate-contrast target
    dose = b['meta_dose'][tr]
    ctd = group_ids(cell, time, dose)
    Dd, nDd = loo_other(y, ctd, group_ids(cell, time, dose, ctl))
    okd = has_yc & (nDd > 0)
    w_ = okd.astype(np.float64)
    Dd0 = np.where(okd[:, None], Dd, 0.0)
    gp, gpp = group_ids(ctl), group_ids(ctl, pert)
    Sp_, _ = group_sum(gp, Dd0 * w_[:, None])
    Spp_, _ = group_sum(gpp, Dd0 * w_[:, None])
    cntd = np.bincount(gp, weights=w_)[gp] - np.bincount(gpp, weights=w_)[gpp]
    with np.errstate(invalid='ignore', divide='ignore'):
        cDd = Dd0 - (Sp_[gp] - Spp_[gpp]) / cntd[:, None]
    md = okd & (cntd >= 2)
    a2b['dose_only_floor_on_contrast'] = {'n_rows': int(md.sum()), 'r_mean': nanmean(rowr(cDd[md], yc[md]))}
    gpd = group_ids(ctl, dose)
    same_dose = np.bincount(gpd)[gpd] - np.bincount(group_ids(ctl, dose, pert))[group_ids(ctl, dose, pert)]
    mates = np.bincount(gp)[gp] - np.bincount(gpp)[gpp]
    mm = mates > 0
    a2b['mean_frac_plate_mates_at_another_dose'] = float(np.mean(1 - same_dose[mm] / mates[mm]))
    res['A2b'] = a2b
    return res


def a3(b, split_key):
    tr = b[split_key] == 'train'
    X, C, cell = b['X'][tr].astype(np.float64), b['X_ctl'][tr].astype(np.float64), b['meta_cell'][tr]
    y = X - C
    cg = group_ids(cell)
    Sc, nc = group_sum(cg, C)
    Sy, _ = group_sum(cg, y)
    dev = -(C - Sc[cg] / nc[cg][:, None])
    Mcell = Sy[cg] / nc[cg][:, None]
    return {'split': split_key, 'n_train_rows': int(tr.sum()),
            'minus_ctl_deviation_alone': nanmean(rowr(dev, y)),
            'cell_mean_delta_alone': nanmean(rowr(Mcell, y)),
            'both_F0': nanmean(rowr(dev + Mcell, y))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=BUNDLE)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args(argv)
    b = np.load(a.bundle, allow_pickle=True)
    out = {'bundle': os.path.relpath(a.bundle, REPO).replace('\\', '/'), 'bundle_sha1': sha1(a.bundle),
           'rows_read': 'training rows only; no model, prediction or test row',
           'plate_proxy': 'byte-identical X_ctl (the bundle has no plate key)',
           'cold_drug_1': a0_a2(b, 'split_split_cold_drug_1'),
           'cold_cell_1_A3': a3(b, 'split_split_cold_cell_1')}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1)
    cd = out['cold_drug_1']
    print('A0', json.dumps(cd['A0']))
    print('A1 common rows', json.dumps({k: round(v, 3) for k, v in cd['A1']['common_rows'].items()}))
    print('A2 P %.3f, best P+wD %.3f (w %s), increment %+.3f' % (cd['A2']['P'], cd['A2']['P'] + cd['A2']['drug_increment_over_P'],
          cd['A2']['best_w'], cd['A2']['drug_increment_over_P']))
    pc = cd['A2']['plate_contrast_target']
    print('A2 plate-contrast: same drug other plates %.3f (terciles %s), drug-free %.3f' % (
          pc['same_drug_cell_time_other_plates_r_mean'], [round(x, 3) for x in pc['by_strength_tercile']], pc['drug_free_cell_mean_r']))
    print('A2b', json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                                                      for kk, vv in v.items()}) for k, v in cd['A2b'].items()}))
    print('A3', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out['cold_cell_1_A3'].items()}))
    print('wrote', a.out)


if __name__ == '__main__':
    sys.exit(main())
