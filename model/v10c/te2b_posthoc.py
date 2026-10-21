"""POST HOC (RESULTS 101.7; review 097 C3 / Ask 2): T-E2's A1 (CCLE) decomposition for HDAC inhibitors and the two A4
comparators, on T-E's rows. Reported, never a claim. A guard first: BET's A1 must reproduce te2_reading.json exactly.

    python model/v10c/te2b_posthoc.py --out model/results/v10c/te2b_posthoc.json
"""
import argparse, json, os, time

import numpy as np

import te_tests as te
import te2_tests as t2


def a1_block(mF, kF, U, basal, cells, E, kE, Ebar):
    sub_cells = [i for i in cells if kE[i].mean() > 0.5]
    gs = np.isfinite(E) & np.isfinite(Ebar)
    act_only = te.per_cell_r(mF, kF, U, basal, sub_cells, gene_sel=gs)
    i_ = te.per_cell_r(mF, kF, U, basal, sub_cells, extra=[E, Ebar])
    ii_ = te.per_cell_r(E, kE, U, basal, sub_cells)
    Fbar = te.lco_mean(mF, kF)
    iii_ = te.per_cell_r(E, kE & kF, U, basal, sub_cells, extra=[np.where(kF, mF, np.nan), Fbar])
    out = {'cells': sub_cells, 'act_only': t2.summary(act_only), 'i_mark_given_ccle': t2.summary(i_),
           'ii_ccle_as_m': t2.summary(ii_), 'iii_ccle_given_mark': t2.summary(iii_)}
    if all('ci' in out[k] for k in ('act_only', 'i_mark_given_ccle', 'ii_ccle_as_m')):
        out['reading_post_hoc'] = t2.a1_rule(out['i_mark_given_ccle'], out['act_only'], out['ii_ccle_as_m'])
    else:
        out['reading_post_hoc'] = 'inconclusive (fewer than 3 cells)'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- read once')
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
    C, _ = te.signed_contrast(Y, plate, pert, mate_ok)
    del Y
    ch = np.load(te.CHROM, allow_pickle=True)
    ccells = [str(c) for c in ch['cells']]
    F, Mk = ch['F'].astype(np.float64), ch['M'].astype(bool)
    cidx = {c: i for i, c in enumerate(ccells)}
    nC = len(ccells)
    keep = np.array([c in cidx for c in cell])
    ci_row = np.where(keep, np.array([cidx.get(c, 0) for c in cell]), 0)
    basal = np.full((nC, F.shape[1]), np.nan)
    for c in np.unique(cell[keep]):
        basal[cidx[c]] = ctl[np.unique(ctl_id[cell == c])].mean(0)
    ok_b = np.isfinite(basal).all(1)
    basal[ok_b] = te.rank01_rows(basal[ok_b])
    cz = np.load(t2.CCLE, allow_pickle=True)
    if [str(e) for e in cz['landmark_entrez']] != [str(e) for e in ch['entrez']]:
        raise SystemExit('FATAL: CCLE landmark order differs from the chromatin genes')
    E = np.full((nC, F.shape[1]), np.nan)
    for j, c in enumerate(cz['cells']):
        if str(c) in cidx:
            E[cidx[str(c)]] = cz['expr_landmarks'][j]
    E = t2.rank_rows_nan(E)
    kE = np.isfinite(E)
    Ebar = te.lco_mean(np.nan_to_num(E), kE)
    comp = json.load(open(t2.COMPARATORS, encoding='utf-8'))

    def eligible(ax, U):
        has = Mk[:, :, te.AXIS[ax]].mean(1) > te.MASK_COVER
        return [i for i in range(nC) if has[i] and np.isfinite(U[i]).any() and ok_b[i]]

    sets = {'bet_inhibitor (guard)': (sub == 'bet_inhibitor', 'ACT:tss_50kb', 'ACT'),
            'hdac_inhibitor x ACT:prom_1kb': (sub == 'hdac_inhibitor', 'ACT:prom_1kb', 'ACT'),
            'hdac_inhibitor x ACC:prom_1kb': (sub == 'hdac_inhibitor', 'ACC:prom_1kb', 'ACC'),
            'transcription_inhibitor x ACT:tss_50kb': (np.isin(pert, comp['transcription_inhibitor']), 'ACT:tss_50kb', 'ACT'),
            'proteostasis x ACT:tss_50kb': (cls == 'proteostasis', 'ACT:tss_50kb', 'ACT')}
    res = {'rule': 'RESULTS 101.7: POST HOC, reported, never a claim; A1 of 101.4 applied as a label only'}
    for name, (sel, feat, ax) in sets.items():
        U = te.class_profiles(C, ci_row, nC, sel & keep)[0]
        blk = a1_block(F[:, :, te.FEAT[feat]], Mk[:, :, te.AXIS[ax]], U, basal, eligible(ax, U), E, kE, Ebar)
        blk['cells'] = [ccells[i] for i in blk['cells']]
        res[name] = blk
    want = json.load(open(t2.TE_READING.replace('te_reading', 'te2_reading'), encoding='utf-8'))['A1']
    g = res['bet_inhibitor (guard)']
    for k_new, k_old in (('act_only', 'act_only'), ('i_mark_given_ccle', 'i_act_given_ccle'), ('ii_ccle_as_m', 'ii_ccle_as_m')):
        if abs(g[k_new]['mean_r'] - want[k_old]['mean_r']) > 1e-9:
            raise SystemExit(f'FATAL: guard {k_new} {g[k_new]["mean_r"]} != te2 {want[k_old]["mean_r"]}')
    res['wall_s'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    for name, blk in res.items():
        if isinstance(blk, dict) and 'act_only' in blk:
            print(name, '| cells', len(blk['cells']), '|', blk['reading_post_hoc'],
                  {k: (round(blk[k]['mean_r'], 4) if blk[k].get('mean_r') is not None else None,
                       [round(x, 4) for x in blk[k].get('ci', [])]) for k in ('act_only', 'i_mark_given_ccle', 'ii_ccle_as_m')})


if __name__ == '__main__':
    main()
