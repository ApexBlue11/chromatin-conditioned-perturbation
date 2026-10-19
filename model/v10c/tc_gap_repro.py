"""Outcome-free validation of tc_gap.py before any T-C run (RESULTS 100.11): v9's OWN P7 / P9 predictions are written as if
they were v10-C's T-C and gap-arm runs (kernel layout, bench_row, sub_row), and pushed through the reader end to end.
It must (1) reproduce v9's published references (C2c 0.160, C2a 0.481 on all 8 cells; C1 0.109) and (2) score the stand-in
"v10c" identically to "v9" (so the alignment, truth check and mate sets are exact).

    python model/v10c/tc_gap_repro.py --work <scratch dir> --out model/results/v10c/tc_gap_repro.json
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'model', 'v9'))
import tc_gap  # noqa: E402


def write_kernel(root, slot, tag, sub_rows, bench_rows, deltas_by_seed):
    """One kernel dir per seed slot with a completed attempt 0 holding a v10-C-format prediction file."""
    for s, d in enumerate(deltas_by_seed):
        kd = os.path.join(root, f'{slot}_s{s}')
        sub = os.path.join(kd, f'{slot}_s{s}')
        os.makedirs(sub, exist_ok=True)
        t = f'{tag}_s{s}'
        np.savez(os.path.join(sub, f'v10c_pred_{t}.npz'), sub_row=sub_rows, bench_row=bench_rows,
                 delta=d, drug=np.zeros_like(d, np.float32))
        json.dump({'tag': t}, open(os.path.join(sub, f'V10C_COMPLETE_{t}.json'), 'w'))
        json.dump({'attempts': {f'{slot}_s{s}': [{'attempt': 0, 'returncode': 0, 'out': f'/kaggle/working/{slot}_s{s}'}]}},
                  open(os.path.join(kd, 'V10C_KERNEL_COMPLETE.json'), 'w'))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--work', required=True)
    ap_.add_argument('--out', required=True)
    a = ap_.parse_args()
    import audit99_phase1 as ap
    S = ap._load_p7_side()
    p9, _, _, plate9, v9_comp = ap._load_p9_side(ap.P9_DIR)
    z = np.load(tc_gap.SUBSTRATE, allow_pickle=True)
    # the substrate row of each reference row, via the split's test rows (bench_row is the bundle's row_index)
    def sub_of(rows, split):
        te = np.flatnonzero(z['split_' + split] == 'test')
        m = {int(z['bench_row'][i]): i for i in te}
        return np.array([m[int(r)] for r in rows], np.int64)
    sub7, sub9 = sub_of(S['row_index'], 'cold_cell_1'), sub_of(p9['row_index'], 'cold_drug_1')
    # shuffle the stored row order so the alignment is exercised
    rng = np.random.default_rng(0)
    o7, o9 = rng.permutation(len(sub7)), rng.permutation(len(sub9))
    write_kernel(a.work, 'tc_on', 'cold_cell_1_test_on_d256', sub7[o7], S['row_index'][o7], [d[o7] for d in S['v9']])
    write_kernel(a.work, 'gap_on', 'cold_drug_1_test_on_d256', sub9[o9], p9['row_index'][o9], [d[o9] for d in v9_comp])
    out = os.path.join(a.work, 'tc_gap.json')
    if os.path.exists(out):
        os.remove(out)
    sys.argv = ['tc_gap.py', '--runs_dir', a.work, '--out', out]
    tc_gap.main()
    r = json.load(open(out))
    cell, comp = r['unseen_cell'], r['unseen_compound']
    checks = {
        'C2c_v9_all8_contrast (0.160)': cell['v9']['contrast']['all8_labelled']['row_mean'],
        'C2c_xpert_all8_contrast (0.146)': cell['xpert_o2']['contrast']['all8_labelled']['row_mean'],
        'C2a_v9_all8_delta_r (0.481)': cell['v9']['delta_r']['all8_labelled']['row_mean'],
        'C2a_xpert_all8_delta_r (0.387)': cell['xpert_o2']['delta_r']['all8_labelled']['row_mean'],
        'C2c_rows_with_mates (16,167)': cell['v9']['contrast']['all8_labelled']['n_rows'],
        'C1_v9_contrast_clean (0.109)': comp['v9']['contrast_clean'],
        'C1_rows (7,980)': comp['v9']['n_contrast_rows'],
        'v9_delta_r_clean (97.10: 0.642)': comp['v9']['delta_r_clean'],
        'standin_equals_v9_cell': cell['v10c_full_release'] == cell['v9'],
        'standin_equals_v9_compound': comp['v10c_full_release'] == comp['v9'],
        'v9_gap_clean7': r['gap']['v9']['clean7'], 'v9_gap_all8': r['gap']['v9']['all8_labelled'],
        'v9_clean7': {'contrast': cell['v9']['contrast']['clean7']['row_mean'], 'delta_r': cell['v9']['delta_r']['clean7']['row_mean']},
        'xpert_clean7': {'contrast': cell['xpert_o2']['contrast']['clean7']['row_mean'],
                         'delta_r': cell['xpert_o2']['delta_r']['clean7']['row_mean']}}
    json.dump(checks, open(a.out, 'w'), indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == '__main__':
    main()
