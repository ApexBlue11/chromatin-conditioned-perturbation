"""T-E2 A1-A5 (RESULTS 101.4): what H-E2 means, on T-E's own rows. Read once; reported (95 % CIs).
B (replication on untouched rows) is a separate script and runs only after T-C's reading.

  A1  CCLE: (i) ACT with CCLE rank + its LCO gene mean as covariates; (ii) CCLE rank as m; (iii) CCLE as m with ACT + its LCO
      mean as covariates; the ACT-only baseline on the same cells. Rule (101.4): "beyond two expression measurements" iff
      (i)'s CI < 0 and |(i)| >= |ACT-only| / 2; "explained by expression" iff (i)'s CI includes 0 and (ii)'s CI < 0; else
      inconclusive.
  A2  split-half L1000: (i) m = b2 with [b1, ubar, LCO mean of b2]; (ii) H-E2 with b1 alone as basal.
  A3  H-E2 with decile basal.
  A4  comparators x ACT tss_50kb: transcription inhibitors, proteostasis, a magnitude-matched REF set; Delta(BET - comp).
  A5  E / C-only cells (MINO out); leave-lineage-out (the six B-lymphoid cells out); plates with >= 5 allowed mates only.
A guard first: H-E2 must reproduce te_reading.json's mean r (-0.0709) to 1e-9, else nothing is reported.

    python model/v10c/te2_tests.py --out model/results/v10c/te2_reading.json
"""
import argparse, json, os, time

import numpy as np

import te_tests as te

CCLE = 'external/v11_data/derived/ccle_expr_v11.npz'
TE_READING = 'model/results/v10c/te_reading.json'
COMPARATORS = 'model/results/v10c/te2_comparators.json'
SELECTION = 'model/results/v10c/chipatlas_selection.json'
B_LYMPHOID = ('OCILY3', 'OCILY10', 'HBL1', 'TMD8', 'MINO', 'NALM6')
CI = 0.95
F_ACT_BROAD = te.FEAT['ACT:tss_50kb']


def summary(r, level=CI):
    v = list(r.values())
    if len(v) < 3:
        return {'n_cells': len(v), 'mean_r': float(np.mean(v)) if v else None, 'note': 'fewer than 3 cells'}
    return {'n_cells': len(v), 'mean_r': float(np.mean(v)), 'ci': te.boot_ci(v, level),
            'n_negative': int(sum(x < 0 for x in v)), 'per_cell_r': {str(k): x for k, x in r.items()}}


def paired(r1, r2, level=CI):
    c = sorted(set(r1) & set(r2))
    d = [r1[k] - r2[k] for k in c]
    return {'n_cells': len(c), 'mean_diff': float(np.mean(d)) if d else None, 'ci': te.boot_ci(d, level) if len(d) >= 3 else None}


def a1_rule(i_sum, act_sum, ii_sum):
    lo, hi = i_sum['ci']
    if hi < 0 and abs(i_sum['mean_r']) >= 0.5 * abs(act_sum['mean_r']):
        return 'beyond two expression measurements'
    if lo <= 0 <= hi and ii_sum['ci'][1] < 0:
        return 'explained by expression'
    return 'inconclusive'


def a2_split_half(b1, b2, U, cells):
    """A2 (review 096 C1): m = b2 over its FINITE genes (cells without two distinct controls carry NaN and must not enter
    the leave-cell-out mean), basal = b1. Also the per-cell split-half reliability rho = corr(b1, b2) and the Spearman-Brown
    leakage ratio (1 - R_full) / (1 - rho), R_full = 2 rho / (1 + rho), to scale A2(i) toward the full-control basal."""
    k2 = np.isfinite(b2)
    r = te.per_cell_r(np.nan_to_num(b2), k2, U, np.nan_to_num(b1), cells)
    rho = {}
    for i in r:
        f = np.isfinite(b1[i]) & np.isfinite(b2[i])
        rho[i] = float(np.corrcoef(b1[i, f], b2[i, f])[0, 1])
    # (1 - R_full) / (1 - rho) with R_full = 2 rho / (1 + rho) simplifies to 1 / (1 + rho); undefined for rho <= 0
    ratio = {i: 1.0 / (1.0 + q) for i, q in rho.items() if q > 0}
    return r, rho, ratio


def a4_reading(bet_common, comp):
    """Review 096 C4: 'not BET-specific' iff the comparator's mean r lies within BET's 95 % CI over the cells they share."""
    if bet_common.get('ci') is None or comp.get('mean_r') is None:
        return 'inconclusive (fewer than 3 shared cells)'
    lo, hi = bet_common['ci']
    return 'not BET-specific' if lo <= comp['mean_r'] <= hi else 'differs from BET'


def rank_rows_nan(A):
    """Within-row rank in [0, 1] over finite entries; NaN stays NaN."""
    out = np.full(A.shape, np.nan)
    for i in range(A.shape[0]):
        f = np.isfinite(A[i])
        if f.sum() > 1:
            r = np.argsort(np.argsort(A[i, f])).astype(np.float64)
            out[i, f] = r / (f.sum() - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- T-E2 is read once')
    t0 = time.time()
    z = np.load(te.SUB, allow_pickle=True)
    tr = np.flatnonzero(z['split_cold_cell_1'] == 'train')
    pert, cell, plate = z['meta_pert_id'][tr], z['meta_cell_iname'][tr], z['plate_id'][tr]
    ctl, ctl_id = z['ctl'], z['ctl_id'][tr]
    Y = z['X'][tr].astype(np.float32) - ctl[ctl_id].astype(np.float32)
    cl = json.load(open(te.CLASSES, encoding='utf-8'))
    cls = np.array([cl['class'].get(p, 'unannotated') for p in pert])
    sub = np.array([cl['sub'].get(p, '') for p in pert])
    mate_ok = ~np.isin(cls, te.EXCLUDED_MATE_CLASSES)
    C, n_mates = te.signed_contrast(Y, plate, pert, mate_ok)
    ch = np.load(te.CHROM, allow_pickle=True)
    ccells = [str(c) for c in ch['cells']]
    F, Mk = ch['F'].astype(np.float64), ch['M'].astype(bool)
    cidx = {c: i for i, c in enumerate(ccells)}
    nC = len(ccells)
    keep = np.array([c in cidx for c in cell])
    ci_row = np.where(keep, np.array([cidx.get(c, 0) for c in cell]), 0)
    basal = np.full((nC, F.shape[1]), np.nan)
    b1 = np.full_like(basal, np.nan)
    b2 = np.full_like(basal, np.nan)
    rng = np.random.default_rng(0)
    for c in np.unique(cell[keep]):
        ids = np.unique(ctl_id[cell == c])
        basal[cidx[c]] = ctl[ids].mean(0)
        if len(ids) >= 2:
            perm = rng.permutation(ids)
            h = len(perm) // 2
            b1[cidx[c]], b2[cidx[c]] = ctl[perm[:h]].mean(0), ctl[perm[h:]].mean(0)
    for B in (basal, b1, b2):
        ok = np.isfinite(B).all(1)
        B[ok] = te.rank01_rows(B[ok])
    ok_b = np.isfinite(basal).all(1)
    has_act = Mk[:, :, te.AXIS['ACT']].mean(1) > te.MASK_COVER
    mACT, kACT = F[:, :, F_ACT_BROAD], Mk[:, :, te.AXIS['ACT']]

    bet = sub == 'bet_inhibitor'
    U_bet = te.class_profiles(C, ci_row, nC, bet & keep)[0]
    U_ref = te.class_profiles(C, ci_row, nC, mate_ok & keep)[0]
    cells = [i for i in range(nC) if has_act[i] and np.isfinite(U_bet[i]).any() and ok_b[i]]
    base = te.per_cell_r(mACT, kACT, U_bet, basal, cells)
    want = json.load(open(TE_READING, encoding='utf-8'))['primary']['H-E2']['mean_r']
    got = float(np.mean(list(base.values())))
    if abs(got - want) > 1e-9:
        raise SystemExit(f'FATAL: H-E2 reproduces as {got}, te_reading.json has {want}')
    cells = sorted(base)
    res = {'rule': 'RESULTS 101.4 A1-A5 (reported, 95 % CIs)', 'guard_H-E2_mean_r': got, 'H-E2_cells': [ccells[i] for i in cells]}

    # ---- A1: CCLE ----
    cz = np.load(CCLE, allow_pickle=True)
    ccl_cells = [str(c) for c in cz['cells']]
    if [str(e) for e in cz['landmark_entrez']] != [str(e) for e in ch['entrez']]:
        raise SystemExit('FATAL: CCLE landmark order differs from the chromatin genes')
    E = np.full((nC, F.shape[1]), np.nan)
    for j, c in enumerate(ccl_cells):
        if c in cidx:
            E[cidx[c]] = cz['expr_landmarks'][j]
    E = rank_rows_nan(E)
    kE = np.isfinite(E)
    Ebar = te.lco_mean(np.nan_to_num(E), kE)
    Abar = te.lco_mean(mACT, kACT)
    sub_cells = [i for i in cells if kE[i].mean() > 0.5]
    gs = np.isfinite(E) & np.isfinite(Ebar)                    # review 096 C2: ACT-only on (i)'s gene set
    act_only = te.per_cell_r(mACT, kACT, U_bet, basal, sub_cells, gene_sel=gs)
    i_ = te.per_cell_r(mACT, kACT, U_bet, basal, sub_cells, extra=[E, Ebar])
    ii_ = te.per_cell_r(E, kE, U_bet, basal, sub_cells)
    mask_both = kE & kACT
    iii_ = te.per_cell_r(E, mask_both, U_bet, basal, sub_cells, extra=[np.where(kACT, mACT, np.nan), Abar])
    validity = {}                                              # review 096 C3: CCLE against the L1000 basal, per cell
    for i in sub_cells:
        f = np.isfinite(E[i]) & np.isfinite(basal[i])
        validity[ccells[i]] = float(np.corrcoef(E[i, f], basal[i, f])[0, 1])
    A1 = {'cells': [ccells[i] for i in sub_cells], 'ccle_vs_l1000_basal_corr': validity,
          'act_only': summary(act_only), 'i_act_given_ccle': summary(i_),
          'ii_ccle_as_m': summary(ii_), 'iii_ccle_given_act': summary(iii_)}
    if all('ci' in A1[k] for k in ('act_only', 'i_act_given_ccle', 'ii_ccle_as_m')):
        A1['reading'] = a1_rule(A1['i_act_given_ccle'], A1['act_only'], A1['ii_ccle_as_m'])
    else:
        A1['reading'] = 'inconclusive (fewer than 3 cells)'
    res['A1'] = A1

    # ---- A2: split-half L1000 ----
    ok12 = np.isfinite(b1).all(1) & np.isfinite(b2).all(1)
    c12 = [i for i in cells if ok12[i]]
    a2i, rho, ratio = a2_split_half(b1, b2, U_bet, c12)
    a2ii = te.per_cell_r(mACT, kACT, U_bet, np.nan_to_num(b1), c12)
    s_a2i = summary(a2i)
    res['A2'] = {'cells': [ccells[i] for i in c12], 'i_b2_given_b1': s_a2i,
                 'split_half_reliability': {ccells[i]: v for i, v in rho.items()},
                 'spearman_brown_leakage_ratio_mean': float(np.mean(list(ratio.values()))) if ratio else None,
                 'i_scaled_to_full_basal': (s_a2i['mean_r'] * float(np.mean(list(ratio.values())))
                                            if ratio and s_a2i.get('mean_r') is not None else None),
                 'ii_HE2_with_b1_basal': summary(a2ii),
                 'HE2_full_basal_same_cells': summary({k: base[k] for k in c12})}

    # ---- A3: decile basal ----
    res['A3'] = {'HE2_decile_basal': summary(te.per_cell_r(mACT, kACT, U_bet, basal, cells, flexible_basal=True))}

    # ---- A4: comparators ----
    comp = json.load(open(COMPARATORS, encoding='utf-8'))
    tx = np.isin(pert, comp['transcription_inhibitor'])
    prot = cls == 'proteostasis'
    norms = np.linalg.norm(np.nan_to_num(C), axis=1)
    finite = np.isfinite(C[:, 0])
    mm = np.zeros(len(pert), bool)
    rng4 = np.random.default_rng(0)
    for i in cells:
        rows_c = keep & (ci_row == i) & finite
        bet_rows = np.flatnonzero(rows_c & bet)
        if len(bet_rows) == 0:
            continue
        q25, q75 = np.quantile(norms[bet_rows], [0.25, 0.75])
        ref_rows = np.flatnonzero(rows_c & mate_ok)
        if len(ref_rows) == 0:
            continue
        cand = []
        for p in np.unique(pert[ref_rows]):
            rr = ref_rows[pert[ref_rows] == p]
            if q25 <= norms[rr].mean() <= q75:
                cand.extend(rr.tolist())
        if cand:
            pick = rng4.choice(np.array(cand), size=min(len(bet_rows), len(cand)), replace=False)
            mm[pick] = True
    A4 = {}
    for nm, sel in (('transcription_inhibitor', tx), ('proteostasis', prot), ('magnitude_matched_ref', mm)):
        U = te.class_profiles(C, ci_row, nC, sel & keep)[0]
        r = te.per_cell_r(mACT, kACT, U, basal, cells)
        common = sorted(set(base) & set(r))
        bet_common = summary({k: base[k] for k in common})
        sm = summary(r)
        A4[nm] = {'summary': sm, 'bet_on_common_cells': bet_common, 'bet_minus_comparator': paired(base, r),
                  'reading': a4_reading(bet_common, sm), 'rows': int((sel & keep & finite).sum())}
    res['A4'] = A4

    # ---- A5: provenance ----
    sel_json = json.load(open(SELECTION, encoding='utf-8'))
    tier = {r['srx']: r['tier'] for r in sel_json['records']}
    ec = [i for i in cells if ccells[i] != 'MINO' and
          all(tier.get(x) in ('E', 'C') for x in sel_json['selection'].get(ccells[i], {}).get('ACT', ['?']))]
    lin = [i for i in cells if ccells[i] not in B_LYMPHOID]
    C5, _ = te.signed_contrast(Y, plate, pert, mate_ok, min_mates=5)
    U5 = te.class_profiles(C5, ci_row, nC, bet & keep)[0]
    c5 = [i for i in cells if np.isfinite(U5[i]).any()]
    res['A5'] = {'ec_only_no_mino': {'cells': [ccells[i] for i in ec], **summary({k: base[k] for k in ec})},
                 'leave_b_lymphoid_out': {'cells': [ccells[i] for i in lin], **summary({k: base[k] for k in lin})},
                 'plates_ge5_allowed_mates': {'cells': [ccells[i] for i in c5],
                                              **summary(te.per_cell_r(mACT, kACT, U5, basal, c5))}}
    res['cells'] = ccells
    res['wall_s'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    show = {'A1': {k: (v.get('mean_r'), v.get('ci'), v.get('n_cells')) if isinstance(v, dict) else v for k, v in A1.items() if k != 'cells'},
            'A2': {k: (v.get('mean_r'), v.get('ci')) for k, v in res['A2'].items() if k != 'cells'},
            'A3': {k: (v.get('mean_r'), v.get('ci')) for k, v in res['A3'].items()},
            'A4': {k: (v['summary'].get('mean_r'), v['summary'].get('ci'), v['bet_minus_comparator']) for k, v in A4.items()},
            'A5': {k: (v.get('mean_r'), v.get('ci'), v.get('n_cells')) for k, v in res['A5'].items()}}
    print(json.dumps(show, indent=1, default=str))


if __name__ == '__main__':
    main()
