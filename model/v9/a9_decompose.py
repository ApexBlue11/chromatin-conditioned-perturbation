# -*- coding: utf-8 -*-
"""RESULTS 79 (IDEAS A9): where the atom tokens' alpha = 0 residual lives. Reads the seven alpha_sweep runs and applies
the pre-committed checks and readings, in the order 79 fixes: structural checks (79.3), kill switch (79.4), null-key
gate (79.5), readings (79.6). Nothing here chooses a threshold; every number it compares against is in 79.

Every statistic is recomputed from the per-row npz dumps with ONE bootstrap resample matrix per split (seed 0, 20,000
draws), shared by every cell and key, so differences between cells are not contaminated by resampling noise that
differs per cell -- the rule alpha_sweep.py follows along its alpha curve.
"""
import io
import json
import os

import numpy as np
import scipy.stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
STEM = 'v9_alpha_sweep_sa0_ckpt_v9_fold0_seed0'
SPLITS = ['unseen_compound', 'unseen_both', 'unseen_cell']
PRIMARY = 'unseen_compound'
E0_PRIMARY = -0.00322          # RESULTS 74 / 79: atom effect at alpha 0, estimand of record, primary split
KILL = 3 * abs(E0_PRIMARY)     # 79.4: |cost| < 3 x |E0| = 0.00966, two-sided
GATE = 0.25                    # 79.5
ROWS_SHA = {'unseen_cell': '434418d7677d3f9c', 'unseen_compound': '160865d7b95cbc06', 'unseen_both': '02fb5b09d78a8d7e'}
N_BOOT = 20000


def run_name(cut, key):
    k = '' if key == 'atoms' else '_key-%s' % key
    c = '' if cut == 'none' else '_cut-%s' % cut
    return STEM + k + '_op-atom_only' + c + '_a0'


def load(cut, key):
    j = json.load(io.open(os.path.join(R, run_name(cut, key) + '.json'), encoding='utf-8'))
    z = np.load(os.path.join(R, run_name(cut, key) + '_rows.npz'))
    return j, {s: (z[s + '__a0.0__r_full'], z[s + '__a0.0__r_abl']) for s in SPLITS}


def med_ci(x, B):
    b = np.median(x[B], axis=1)
    return float(np.median(x)), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mean_ci(x, B):
    b = x[B].mean(axis=1)
    return float(x.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def sign_p(x):
    nz = x[x != 0]
    return float(scipy.stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else 1.0


def excludes0(ci):
    return ci[0] > 0 or ci[1] < 0


def main():
    runs = {(c, k): load(c, k) for c, k in [('none', 'atoms'), ('none', 'x_cell'), ('xattn', 'x_cell'),
                                            ('global', 'x_cell'), ('xattn', 'atoms'), ('global', 'atoms'),
                                            ('both', 'atoms')]}
    out = {'section': 'RESULTS 79', 'kill_bar_abs': KILL, 'gate_ratio': GATE, 'splits': {}}

    # ---- 79.3 structural checks: stop on failure ----------------------------------------------------------------
    ref = {k: np.load(os.path.join(R, STEM + ('' if k == 'atoms' else '_key-x_cell') + '_op-atom_only_rows.npz'))
           for k in ('atoms', 'x_cell')}
    regress = {}
    for (c, k), (j, rows) in runs.items():
        for s in SPLITS:
            assert j['splits'][s]['rows_sha'] == ROWS_SHA[s], (c, k, s, 'rows_sha')
            if c == 'none':
                a = np.array_equal(rows[s][0], ref[k][s + '__a0.0__r_full'])
                b = np.array_equal(rows[s][1], ref[k][s + '__a0.0__r_abl'])
                regress['%s/%s' % (k, s)] = bool(a and b)
    out['check_1_regression_byte_identical'] = regress
    dymax_both = {s: runs[('both', 'atoms')][0]['splits'][s]['results_by_alpha']['0.0']['dY_max'] for s in SPLITS}
    out['check_2_both_cut_dY_max'] = dymax_both
    out['structural_checks_pass'] = all(regress.values()) and all(v == 0.0 for v in dymax_both.values())
    print('79.3 regression byte-identical:', regress)
    print('79.3 both-cut dY_max:', dymax_both)
    if not out['structural_checks_pass']:
        print('STRUCTURAL CHECK FAILED -- nothing is read (79.3).')
        return write(out)

    # ---- per split ---------------------------------------------------------------------------------------------
    for s in SPLITS:
        arrs = {ck: runs[ck][1][s] for ck in runs}
        ok = np.ones_like(arrs[('none', 'atoms')][0], dtype=bool)
        for f, a in arrs.values():
            ok &= np.isfinite(f) & np.isfinite(a)
        n = int(ok.sum())
        B = np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))
        eff = {ck: (arrs[ck][0] - arrs[ck][1])[ok] for ck in arrs}
        full = {ck: arrs[ck][0][ok] for ck in arrs}
        so = {'n_rows_paired': n, 'cells': {}, 'single_cuts': {}}

        for ck, e in eff.items():
            m, ci = med_ci(e, B)
            so['cells']['%s/%s' % ck] = {'effect_median_per_row': m, 'ci95': ci, 'sign_p': sign_p(e),
                                          'paired_mean': mean_ci(e, B)}

        E0, N0 = eff[('none', 'atoms')], eff[('none', 'x_cell')]
        for cut in ('xattn', 'global'):
            cost = full[('none', 'atoms')] - full[(cut, 'atoms')]           # 79.4, paired, same rows
            cost_m, cost_ci = med_ci(cost, B)
            dE = float(np.median(eff[(cut, 'atoms')]) - np.median(E0))       # 79.5, estimand of record
            dN = float(np.median(eff[(cut, 'x_cell')]) - np.median(N0))
            pd_m, pd_ci = med_ci(eff[(cut, 'atoms')] - E0, B)                # the paired per-row change
            so['single_cuts'][cut] = {
                'kill_cost_median_per_row': cost_m, 'kill_cost_ci95': cost_ci,
                'kill_switch_pass': abs(cost_m) < KILL,
                'dE_estimand_of_record': dE, 'dN_estimand_of_record': dN,
                'gate_ratio_observed': abs(dN) / abs(dE) if dE != 0 else float('inf'),
                'gate_pass': abs(dN) < GATE * abs(dE),
                'paired_change_median': pd_m, 'paired_change_ci95': pd_ci,
                'paired_change_detectable': excludes0(pd_ci)}

        ea, eb = eff[('global', 'atoms')], eff[('xattn', 'atoms')]            # G cut leaves route X; X cut leaves G
        i = E0 - ea - eb
        so['decomposition_paired_means'] = {'e0': mean_ci(E0, B), 'ea_atom_keys': mean_ci(ea, B),
                                            'eb_global_token': mean_ci(eb, B), 'i_remainder': mean_ci(i, B)}
        so['remainder_median_per_row'] = med_ci(i, B)
        out['splits'][s] = so

    # ---- 79.6 readings, primary split only ------------------------------------------------------------------------
    p = out['splits'][PRIMARY]
    readable = {cut: p['single_cuts'][cut]['kill_switch_pass'] for cut in ('xattn', 'global')}
    Ea = p['cells']['global/atoms']; Eb = p['cells']['xattn/atoms']
    sig = lambda c: excludes0(c['ci95'])
    rd = {'Ea_readable (G cut passes kill switch)': readable['global'],
          'Eb_readable (X cut passes kill switch)': readable['xattn'],
          'Ea_significant': sig(Ea), 'Ea_sign': np.sign(Ea['effect_median_per_row']).item(),
          'Eb_significant': sig(Eb), 'Eb_sign': np.sign(Eb['effect_median_per_row']).item()}
    for cut in ('xattn', 'global'):
        sc = p['single_cuts'][cut]
        rd['attribution_%s' % cut] = ('gate applies' if sc['paired_change_detectable'] else
                                      'no detectable change: gate reported, not needed') + \
                                     ('; gate %s' % ('PASS' if sc['gate_pass'] else 'FAIL'))
    out['readings_primary'] = rd
    print(json.dumps(out['readings_primary'], indent=1))
    for s in SPLITS:
        so = out['splits'][s]
        print('\n---', s, 'n', so['n_rows_paired'])
        for k, v in so['cells'].items():
            print('  %-14s E %+.5f [%+.5f, %+.5f] p %.1e | mean %+.5f' % (k, v['effect_median_per_row'], v['ci95'][0],
                                                                        v['ci95'][1], v['sign_p'], v['paired_mean'][0]))
        for cut, v in so['single_cuts'].items():
            print('  cut %-6s cost %+.5f [%+.5f, %+.5f] kill %s | dE %+.5f dN %+.5f ratio %.2f gate %s | paired change '
                  '%+.5f [%+.5f, %+.5f]' % (cut, v['kill_cost_median_per_row'], v['kill_cost_ci95'][0],
                                            v['kill_cost_ci95'][1], 'PASS' if v['kill_switch_pass'] else 'FAIL',
                                            v['dE_estimand_of_record'], v['dN_estimand_of_record'],
                                            v['gate_ratio_observed'], 'PASS' if v['gate_pass'] else 'FAIL',
                                            v['paired_change_median'], v['paired_change_ci95'][0],
                                            v['paired_change_ci95'][1]))
        d = so['decomposition_paired_means']
        print('  means: e0 %+.5f = ea %+.5f + eb %+.5f + i %+.5f %s' % (d['e0'][0], d['ea_atom_keys'][0],
                                                                    d['eb_global_token'][0], d['i_remainder'][0],
                                                                    d['i_remainder'][1]))
    write(out)


def write(out):
    dst = os.path.join(R, 'v9_a9_residual_routes_sa0.json')
    io.open(dst, 'w', encoding='utf-8', newline='\n').write(json.dumps(out, indent=2) + '\n')
    print('\nwrote', dst)


if __name__ == '__main__':
    main()
