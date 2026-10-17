"""Review 085 C2 (reported, never a claim): O9 (XPert as published on split_cold_drug_1) added to C1c and to C1's contrast.
C1c: the original target, on the clean rows with >= 2 other-compound training plate-mates, against the training-plate lookup.
C1: the within-plate contrast on the clean rows with >= 2 scored plate-mates, beside v9 and M+dt.

    python model/v9/audit99_o9_report.py
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit99_phase1 as ph  # noqa: E402
import score_p10  # noqa: E402
from coldcell_h2h import load_profile  # noqa: E402

OUT = os.path.join(ph.REPO, 'model', 'results', 'audit99', 'phase1_o9_reported.json')
O9 = os.path.join(ph.REPO, 'external', 'kaggle_out', 'cd1_o9', 'xpert_trained_split_cold_drug_1_final_test_profile.npy')


def o9_deltas(p9):
    ref = load_profile(O9)
    pos = {int(i): k for k, i in enumerate(ref['row_index'])}
    idx = np.array([pos[int(i)] for i in p9['row_index']])
    for k, a in (('y_true', p9['y_true']), ('ctl_true', p9['ctl_true'])):
        d = float(np.max(np.abs(np.asarray(ref[k], np.float64)[idx] - a)))
        if d > 1e-4:
            raise SystemExit(f'FATAL: O9 {k} differs from P9 by {d:.3g}')
    return np.asarray(ref['y_pred'], np.float64)[idx] - p9['ctl_true']


def main(n_boot=ph.N_BOOT):
    if os.path.exists(OUT):
        raise SystemExit(f'FATAL: {OUT} exists -- touched once')
    p9, mdt, m, plate, v9_deltas = ph._load_p9_side(ph.P9_DIR)
    d_o9 = o9_deltas(p9)
    b = p9['bundle_data']
    plate_all = ph.control_ids(b['X_ctl'])
    y_all = b['X'].astype(np.float64) - b['X_ctl'].astype(np.float64)
    is_tr = b[score_p10.SPLIT_KEY] == 'train'
    P_tr, cnt = ph.other_mean(y_all, plate_all, b['meta_pert_id'], is_tr, p9['bundle_idx'])
    have_c = cnt >= ph.MIN_MATES
    rows_c = have_c & p9['clean_mask']
    y = p9['y']
    r_lookup = ph.rowr_pred(np.nan_to_num(P_tr), y)[0]
    r_o9 = ph.rowr_pred(d_o9, y)[0]
    r_v9 = np.mean([ph.rowr_pred(d, y)[0] for d in v9_deltas], 0)
    mol_c = p9['molecule'][rows_c]
    c1c = {'rows': int(rows_c.sum()),
           'row_mean_r_original_target': {'lookup': ph._summ(r_lookup[rows_c]), 'o9': ph._summ(r_o9[rows_c]),
                                          'v9': ph._summ(r_v9[rows_c])},
           'o9_vs_lookup': ph._mol_block(r_o9[rows_c], r_lookup[rows_c], mol_c, 'the training-plate predictor', n_boot)}
    c1c['o9_vs_lookup']['verdict_note'] = 'score_p9.verdict names the left-hand model "v9"; here it is O9 (review 085 C4)'

    pert = p9['pert']
    cy, cnt2 = ph.plate_contrast(y, plate, pert)
    have = cnt2 >= ph.MIN_MATES
    rows = have & p9['clean_mask']

    def con(deltas):
        out = []
        for d in deltas:
            cd, _ = ph.plate_contrast(d, plate, pert)
            r, _ = ph.rowr_pred(np.where(have[:, None], cd, 0.0), np.where(have[:, None], cy, 0.0))
            r[~have] = np.nan
            out.append(r)
        return np.mean(out, 0)

    rc_o9, rc_v9, rc_mdt = con([d_o9]), con(v9_deltas), con(mdt['pred_deltas'])
    mol = p9['molecule'][rows]
    c1 = {'rows': int(rows.sum()),
          'row_mean_contrast_r': {'o9': ph._summ(rc_o9[rows]), 'v9': ph._summ(rc_v9[rows]), 'mdt': ph._summ(rc_mdt[rows])},
          'v9_vs_o9': ph._mol_block(rc_v9[rows], rc_o9[rows], mol, 'XPert as published (O9)', n_boot),
          'o9_vs_mdt': ph._mol_block(rc_o9[rows], rc_mdt[rows], mol, 'M+dt', n_boot)}
    c1['o9_vs_mdt']['verdict_note'] = 'score_p9.verdict names the left-hand model "v9"; here it is O9 (review 085 C4)'
    rec = {'label': 'review 085 C2: reported, never a claim; O9 checkpoint was selected on these test rows (RESULTS 97.3)',
           'C1c_original_target': c1c, 'C1_contrast': c1}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({'C1c': c1c['row_mean_r_original_target'], 'C1c_o9_vs_lookup': {k: c1c['o9_vs_lookup'][k] for k in ('mean_d', 'ci_lo', 'ci_hi', 'frac_favour_ours')},
                      'C1': c1['row_mean_contrast_r'], 'v9_vs_o9': {k: c1['v9_vs_o9'][k] for k in ('mean_d', 'ci_lo', 'ci_hi', 'frac_favour_ours')},
                      'o9_vs_mdt': {k: c1['o9_vs_mdt'][k] for k in ('mean_d', 'ci_lo', 'ci_hi', 'frac_favour_ours')}}, indent=1))


if __name__ == '__main__':
    main()
