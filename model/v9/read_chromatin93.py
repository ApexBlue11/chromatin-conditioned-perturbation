# -*- coding: utf-8 -*-
"""RESULTS 93 (as amended by 93.5, review 050): the ONLY reader of the H2 and H3 outputs, committed before either runs.

    python model/v9/read_chromatin93.py h2 DIR      # DIR holds chromatin_h2_93.json + CHROMATIN93_H2_COMPLETE.json
    python model/v9/read_chromatin93.py h3 DIR      # DIR holds chromatin_gbm_93.json + CHROMATIN93_H3_COMPLETE.json

Refuses unless the kernel's completion marker lists the output with a matching sha1.
H2 (93.2 item 1 / 93.5 C4; reported, never a gate): read on FBC - N1 only (drug-known rows). "Rising" iff the median over
subsets increases at every step k = 4 < 6 < 8 < all, AND Delta(all) - median Delta(k=4) > 2 sd (ddof 1) of the k = 4 subsets.
H3 (93.2 item 2 / 93.5 C2, C3), on the drug-known rows of record:
  VOID if any calibration draw passes at pi = 0, or P3 passes anywhere, or the gene-generic case G passes anywhere;
  else ADVANCES iff the real reading passes (h3_reading: T1's bars + the N1 magnitude);
  else DOES NOT ADVANCE, labelled by the gain form's MDE (smallest pi passing in 3 of 3 draws): informative iff MDE_P1 <= 0.02
  (93.2: "if it misses P1 at 2 %, its null is uninformative and says so").
"""
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F91 = os.path.join(REPO, 'model', 'results', 'chromatin_funnel_91.json')
T1_BAR_KNOWN = 0.004


def _verified(d, out_name, marker_name):
    out, mk = os.path.join(d, out_name), os.path.join(d, marker_name)
    if not os.path.exists(mk):
        raise SystemExit('REFUSED: no completion marker %s' % mk)
    m = json.load(open(mk))
    want = m.get('outputs', {}).get(out_name)
    got = hashlib.sha1(open(out, 'rb').read()).hexdigest()
    if not m.get('complete') or want != got:
        raise SystemExit('REFUSED: %s sha1 %s does not match the marker (%s)' % (out_name, got, want))
    return json.load(open(out)), m


def read_h2(res):
    real = res['real']
    med = {k: float(np.median([r['FBC_minus_N1'] for r in v])) for k, v in real.items()}
    sd4 = float(np.std([r['FBC_minus_N1'] for r in real['4']], ddof=1))
    steps = [med['4'], med['6'], med['8'], med['all']]
    monotone = all(a < b for a, b in zip(steps, steps[1:]))
    rise = med['all'] - med['4']
    rising = bool(monotone and rise > 2 * sd4)
    # review 052 C2: k = all is 91's T1 exactly; it must reproduce 91.12 (a harness check, not a reading)
    sc = json.load(open(F91))['row_sets']['known']['scores']
    a = real['all'][0]
    repro = bool(abs(a['FBC_minus_FB'] - (sc['FBC']['all'] - sc['FB']['all'])) < 1e-6 and
                 abs(a['FBC_minus_N1'] - (sc['FBC']['all'] - sc['N1']['all'])) < 1e-6)
    reading = ('RISING' if rising else 'NOT RISING') if repro else 'HARNESS_FAULT'   # review 052 follow-up: a fault is not a reading
    return {'reading': reading, 'k_all_reproduces_91': repro,
            'HARNESS_FAULT': None if repro else 'k = all does not reproduce 91.12; the curve is not read',
            'median_FBC_minus_N1': med, 'sd_k4': sd4,
            'rise_all_minus_k4': rise, 'monotone': monotone,
            'reported_median_FBC_minus_FB': {k: float(np.median([r['FBC_minus_FB'] for r in v])) for k, v in real.items()},
            'reported_planted_P1_median_FBC_minus_N1': {k: float(np.median([r['FBC_minus_N1'] for r in v]))
                                                        for k, v in res['planted_P1'].items()},
            'licensed': ('"the gene-local chromatin increment grows with the number of fitting cells" (no extrapolation '
                         'beyond k = all; k = all is one fit without spread)') if rising else None}


def read_h3(res):
    f = res['faults_and_mde']['known']
    real = res['real']['known']
    if f['VOID_null'] or f['VOID_P3'] or f['VOID_G']:
        verdict = 'VOID'
    elif real['pass']:
        verdict = 'ADVANCES'
    else:
        verdict = 'DOES NOT ADVANCE'
    mde = f['MDE']
    # review 052 C1(b): the gene-generic check certifies something only if some G case cleared the raw bar in every draw
    recs = res['calibration']
    g_inf = any(all(r['cases']['G_%g' % pi]['known']['delta']['all'] >= T1_BAR_KNOWN for r in recs) for pi in (0.02, 0.05))
    label = None
    if verdict == 'DOES NOT ADVANCE':
        label = ('informative null for the gain form (MDE <= %g; non-linear learner, these features, these dev cells)' % mde['P1']
                 if mde['P1'] is not None and mde['P1'] <= 0.02 else
                 'uninformative null: the calibration does not detect a planted gain at <= 2 %')
    if not g_inf:
        label = (label + '; ' if label else '') + 'the gene-generic check is uninformative (G\'s raw delta below the bar)'
    return {'reading': verdict, 'label': label, 'G_check_informative': g_inf,
            'faults': {k: f[k] for k in ('VOID_null', 'VOID_P3', 'VOID_G')},
            'G_raw_delta': f['G_raw_delta'], 'MDE': mde, 'conjuncts': real['conjuncts'],
            'delta_C_minus_B': real['delta']['all'], 'delta_top': real['delta']['top'],
            'centred': real['delta']['centred_all'], 'C_minus_N1': real['vs_N1']['all'],
            'chosen_mcs': res['real']['chosen_mcs'], 'reported_all_rows_pass': res['real']['all']['pass']}


def main():
    which, d = sys.argv[1], sys.argv[2]
    if which == 'h2':
        res, m = _verified(d, 'chromatin_h2_93.json', 'CHROMATIN93_H2_COMPLETE.json')
        out = read_h2(res)
    elif which == 'h3':
        res, m = _verified(d, 'chromatin_gbm_93.json', 'CHROMATIN93_H3_COMPLETE.json')
        out = read_h3(res)
    else:
        raise SystemExit('usage: read_chromatin93.py {h2,h3} DIR')
    out['marker'] = m
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
