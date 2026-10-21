"""T-C's stratified readings (RESULTS 100.13, amended by 101.1 C2 / C3), written before any T-C run. Reported, exploratory,
never a claim; T-C's rule (100.5) is unchanged. Every stratum is "among runs whose drug path trained (100.8)": each arm's
slots are their first completed attempts, ledger-checked (tc_gap.completed_preds, review 092 C1).

  per test cell     the 7 clean cells, and H1975 apart
  per class         compound_classes.json's classes: delta, seed SE and z, the compound-cluster bootstrap CI (20,000 draws,
                    default_rng(0)), n compounds and rows; the sensitivity sets hdac_core and bet_with_apabetalone beside them
  per compound      >= 10 test rows: pooled-SE t on 4 df, BH at q = 0.10, the 9-relabelling calibration (strata.py)
  epigenetic x cell the epigenetic class in each clean cell
Class, compound and epigenetic x cell strata are over the 7 clean cells' rows (T-C's rule is over them); H1975 is only
reported as its own cell.

    python model/v10c/tc_strata.py --runs_dir external/kaggle_out/v10c_tc --out model/results/v10c/tc_strata.json
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata  # noqa: E402
import tc_gap  # noqa: E402
import tc_ledger  # noqa: E402
import tc_read  # noqa: E402

SUBSTRATE = 'external/v10c_full/v10c_full_substrate.npz'
CLASSES = 'model/results/v10c/compound_classes.json'
WORDING = 'reported, exploratory, never a claim; among runs whose drug path trained (100.8)'


def arm_r(paths):
    zs = [np.load(p) for p in paths]
    rows = zs[0]['sub_row']
    for z in zs[1:]:
        if not np.array_equal(z['sub_row'], rows):
            raise SystemExit('FATAL: the slots are not on the same rows')
    return rows, np.stack([np.asarray(z['r_contrast'], float) for z in zs])


def compute(rows, Ra, Rb, cell, pert, cl, n_boot=strata.N_BOOT):
    cls = np.array([cl['class'].get(p, 'unannotated') for p in pert])
    clean = np.isin(cell, tc_read.CLEAN)
    out = {'label': WORDING, 'n_rows': int(len(rows)), 'n_clean_rows': int(clean.sum())}
    cells = strata.by_group(Ra, Rb, cell)
    out['per_cell'] = {c: dict(v, apart=c in tc_read.APART) for c, v in cells.items()}
    Ca, Cb, cc, cp, ccls = Ra[:, clean], Rb[:, clean], cell[clean], pert[clean], cls[clean]
    out['per_class'] = strata.by_class(Ca, Cb, ccls, cp, n_boot)
    out['per_class_all8_labelled'] = {'label': 'all 8 test cells, H1975 included (clean for v10-C; review 099 Ask 12)',
                                      **strata.by_class(Ra, Rb, cls, pert, n_boot)}
    ok = strata.finite_rows(Ca, Cb)
    dr = Ca.mean(0) - Cb.mean(0)
    sens = {}
    for name, members in cl['sensitivity_sets'].items():
        sel = ok & np.isin(cp, members)
        if sel.any():
            sens[name] = dict(strata.group_stats(Ca, Cb, sel), ci95_compound_cluster=strata.cluster_ci(dr[sel], cp[sel], n_boot),
                              n_compounds=int(len(np.unique(cp[sel]))))
    out['sensitivity_sets'] = sens
    lab = {}
    for p in np.unique(cp[ok]):
        m = ok & (cp == p)
        lab[str(p)] = {'class': str(ccls[m][0]), 'cells': sorted({str(x) for x in cc[m]})}
    out['per_compound'] = strata.by_compound(Ca, Cb, cp, labels=lab)
    epi = {}
    for c in tc_read.CLEAN:
        sel = ok & (ccls == 'epigenetic') & (cc == c)
        if sel.sum() >= 2:
            epi[c] = strata.group_stats(Ca, Cb, sel)
    out['epigenetic_x_cell'] = epi
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ledger', default=tc_ledger.LEDGER)
    ap.add_argument('--substrate', default=SUBSTRATE)
    ap.add_argument('--classes', default=CLASSES)
    ap.add_argument('--n_boot', type=int, default=strata.N_BOOT)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- T-C\'s strata are read once')
    pa, _, ma = tc_gap.completed_preds(a.runs_dir, 'tc', 'on', a.ledger)
    pb, _, mb = tc_gap.completed_preds(a.runs_dir, 'tc', 'off', a.ledger)
    if ma or mb:
        rec = {'label': WORDING, 'verdict': f'no strata: slots without a completed attempt (on {ma}, off {mb}; 100.8)'}
    else:
        rows, Ra = arm_r(pa)
        rows_b, Rb = arm_r(pb)
        if not np.array_equal(rows, rows_b):
            raise SystemExit('FATAL: the two arms are not on the same rows')
        z = np.load(a.substrate, allow_pickle=True)
        cl = json.load(open(a.classes, encoding='utf-8'))
        rec = compute(rows, Ra, Rb, z['meta_cell_iname'][rows], z['meta_pert_id'][rows], cl, a.n_boot)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rec[k] for k in ('label',) + (('n_rows',) if 'n_rows' in rec else ('verdict',))}))
    return rec


if __name__ == '__main__':
    main()
