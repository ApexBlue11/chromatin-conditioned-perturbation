# -*- coding: utf-8 -*-
"""RESULTS 91: the ONE mechanical reading of the chromatin funnel + its power calibration (PI-written; review 041 C5).

    python model/v9/read_chromatin91.py --dir external/kaggle_out/chromatin91

Refuses unless CHROMATIN91_COMPLETE.json exists and both outputs match the sha1s it records. Reads the row set of record
(drug-known dev rows, review 041), applies 91.3-91.9 as amended: a test advances only if its rule passes AND its positive control
did not fail AND (T1) the calibration is not void; a null is "informative" only under the 91.9 MDE rule.
"""
import argparse
import hashlib
import json
import os


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def read(d):
    mk = os.path.join(d, 'CHROMATIN91_COMPLETE.json')
    if not os.path.exists(mk):
        raise SystemExit('FATAL: no CHROMATIN91_COMPLETE.json -- the run did not finish; nothing here may be read')
    m = json.load(open(mk))
    for f in ('chromatin_funnel_91.json', 'chromatin_power_91.json'):
        p = os.path.join(d, f)
        if not os.path.exists(p) or sha1(p) != m['outputs'][f]:
            raise SystemExit('FATAL: %s missing or not the file the run wrote' % f)
    fun = json.load(open(os.path.join(d, 'chromatin_funnel_91.json')))
    pw = json.load(open(os.path.join(d, 'chromatin_power_91.json')))
    rs = fun['row_set_of_record']
    assert rs == pw['summary']['row_set_of_record'], 'funnel and calibration disagree on the row set of record'
    R, P = fun['row_sets'][rs], pw['summary'][rs]
    out = {'row_set_of_record': rs, 'n_rows': R['n_rows'], 'tests': {}}
    void_t1 = P['instrument_faults']['T1_void']
    for t, form, pc in (('T1', 'T1_P1', 'T1_positive_control_failed'), ('T2', None, 'T2_positive_control_failed'),
                        ('T3', 'T3_P4', 'T3_positive_control_failed')):
        rule = R['advance'][t]
        pcf = R['M3'][pc]
        mde = P['MDE'].get(form) if form else None
        if t == 'T1' and void_t1:
            verdict = 'VOID (instrument fault in the calibration)'
        elif pcf:
            verdict = 'NOT INTERPRETED (positive control failed)'
        elif rule:
            verdict = 'ADVANCES to its one-seed GPU screen'
        elif form is None:
            verdict = 'does not advance (power not calibrated; its null carries little weight)'
        else:
            verdict = 'does not advance; null is %s' % P['reading'][form]
        out['tests'][t] = {'rule_pass': rule, 'positive_control_failed': pcf, 'MDE': mde, 'verdict': verdict}
    out['T1_P2_MDE'] = P['MDE'].get('T1_P2')
    out['M4'] = R['M4']
    out['M1'] = fun['M1']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    a = ap.parse_args()
    print(json.dumps(read(a.dir), indent=1))


if __name__ == '__main__':
    main()
