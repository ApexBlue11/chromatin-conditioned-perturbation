"""EXPLORATORY, POST HOC (principal's request, 2026-10-19): does chromatin help specific compounds or cells on T-B's dev
rows? Trained on runs (s1, s2) against off runs (s0, s1, s2), per-row contrast r from each run's saved predictions.
Seed-noise null: the same statistic between off seeds only ((two off seeds) against (the third), all three splits).
Hypothesis-generating only: nothing here is a reading; anything interesting is registered for T-C before T-C runs.

    python model/v10c/tb_strata.py --out model/results/v10c/tb_strata_exploratory.json
"""
import argparse, json, os

import numpy as np

RUNS = 'external/kaggle_out/v10c_tb'
SUB = 'external/v10c_full/v10c_full_substrate.npz'
EPI = 'drug/outputs/dti/epi_drug_pert_ids.json'
MIN_ROWS = 20


def rcon(arm, s):
    t = f'tb_{arm}_s{s}'
    z = np.load(os.path.join(RUNS, t, t, f'v10c_pred_cold_cell_1_dev_{arm}_d256_s{s}.npz'))
    return z['sub_row'], z['r_contrast']


def per_group(delta, groups, min_rows=MIN_ROWS):
    out = {}
    for g in np.unique(groups):
        m = (groups == g) & np.isfinite(delta)
        if m.sum() >= min_rows:
            out[str(g)] = (float(delta[m].mean()), int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sub, on1 = rcon('on', 1)
    runs = {'on1': on1, 'on2': rcon('on', 2)[1], 'off0': rcon('off', 0)[1], 'off1': rcon('off', 1)[1], 'off2': rcon('off', 2)[1]}
    for k in ('on2', 'off0', 'off1', 'off2'):
        assert np.array_equal(rcon(k[:-1], int(k[-1]))[0], sub), k
    z = np.load(SUB, allow_pickle=True)
    pert, cell = z['meta_pert_id'][sub], z['meta_cell_iname'][sub]
    epi = set(json.load(open(EPI, encoding='utf-8')))
    on = (runs['on1'] + runs['on2']) / 2
    off = (runs['off0'] + runs['off1'] + runs['off2']) / 3
    real = on - off
    nulls = [(runs['off0'] + runs['off1']) / 2 - runs['off2'], (runs['off0'] + runs['off2']) / 2 - runs['off1'],
             (runs['off1'] + runs['off2']) / 2 - runs['off0']]
    comp_real = per_group(real, pert)
    comp_null = [per_group(n, pert) for n in nulls]
    vals = np.array([v for v, _ in comp_real.values()])
    nvals = np.concatenate([[v for v, _ in c.values()] for c in comp_null])
    thr = {q: float(np.quantile(np.abs(nvals), q)) for q in (0.95, 0.99)}
    top = sorted(comp_real.items(), key=lambda kv: -kv[1][0])
    is_epi = np.array([p in epi for p in pert])
    rec = {'label': 'EXPLORATORY, POST HOC; trained on (s1, s2) vs off (s0-s2); never a reading',
           'n_rows': int(len(real)), 'min_rows_per_compound': MIN_ROWS,
           'compounds_scored': len(comp_real),
           'per_compound_delta': {'real_sd': float(vals.std()), 'null_sd': float(nvals.std()),
                                  'null_abs_q95': thr[0.95], 'null_abs_q99': thr[0.99],
                                  'real_n_above_null_q95': int((np.abs(vals) > thr[0.95]).sum()),
                                  'real_n_above_null_q99': int((np.abs(vals) > thr[0.99]).sum()),
                                  'expected_above_q95_by_noise': round(0.05 * len(vals), 1),
                                  'expected_above_q99_by_noise': round(0.01 * len(vals), 1),
                                  'top10_on_better': top[:10], 'top10_off_better': top[-10:]},
           'per_cell_delta': {'real': per_group(real, cell, 1), 'null_mean_abs': {
               c: float(np.mean([abs(per_group(n, cell, 1)[c][0]) for n in nulls])) for c in np.unique(cell)}},
           'epigenetic_compounds': {'rows': int(is_epi.sum()), 'compounds': int(len(set(pert[is_epi]))),
                                    'delta_real': float(np.nanmean(real[is_epi])) if is_epi.any() else None,
                                    'delta_null': [float(np.nanmean(n[is_epi])) for n in nulls] if is_epi.any() else None}}
    json.dump(rec, open(a.out, 'w'), indent=1)
    print(json.dumps({k: rec[k] for k in ('n_rows', 'compounds_scored', 'per_compound_delta', 'per_cell_delta',
                                         'epigenetic_compounds')}, indent=1)[:4000])


if __name__ == '__main__':
    main()
