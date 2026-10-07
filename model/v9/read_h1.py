# -*- coding: utf-8 -*-
"""RESULTS 93.3 / 93.10: the ONLY reader of the H1 outputs, committed before the run.

    python model/v9/read_h1.py DIR      # DIR holds chromatin_h1_93.json + CHROMATIN93_H1_COMPLETE.json

Refuses unless the completion marker lists the output with a matching sha1.
Evaluates in strict order:
1. NOT RUN if not_run
2. VOID if, in either slot:
   - any null pass;
   - any P3 pass;
   - or any G (gene-generic) pass, as read_chromatin93 reads it (PI fix before any run; G_informative only labels).
3. Else ADVANCES iff real['reading']['pass']
4. Else DOES NOT ADVANCE, labelled "informative null (MDE_P1 <= x in both slots ...)" iff MDE_P1 <= 0.02 in both slots,
   else "uninformative ...".
   Appends the G-uninformative note per slot as read_chromatin93.read_h3 does.
Echoes reported items unread: Cp, LOCO, shrinkage, measured cells and k.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402


def _verified(d, out_name='chromatin_h1_93.json', marker_name='CHROMATIN93_H1_COMPLETE.json'):
    out = os.path.join(d, out_name)
    mk = os.path.join(d, marker_name)
    if not os.path.exists(mk):
        raise SystemExit('REFUSED: no completion marker %s' % mk)
    m = json.load(open(mk))
    want = m.get('outputs', {}).get(out_name)
    if not os.path.exists(out):
        raise SystemExit('REFUSED: no output file %s' % out)
    got = hashlib.sha1(open(out, 'rb').read()).hexdigest()
    if not m.get('complete') or want != got:
        raise SystemExit('REFUSED: %s sha1 %s does not match the marker (%s)' % (out_name, got, want))
    return json.load(open(out)), m


def read_h1(res):
    # 1. NOT RUN if not_run
    if res.get('not_run'):
        return {
            'reading': 'NOT RUN',
            'reason': res['not_run'],
            'reported': {
                'measured_dev_cells': res.get('measured_dev_cells'),
                'fitting_cells': res.get('fitting_cells'),
                'k': res.get('k'),
            }
        }

    fam = res.get('faults_and_mde', {})
    slots = [1, 2] if 1 in fam else (['1', '2'] if '1' in fam else sorted(fam.keys()))

    # 2. VOID if, in either slot:
    # - any null pass;
    # - any P3 pass;
    # - or any G pass (a pass means the instrument accepted a gene-generic effect; G_informative only labels a null)
    void_null = any(fam[s].get('VOID_null') for s in slots)
    void_p3 = any(fam[s].get('VOID_P3') for s in slots)
    void_g = any(fam[s].get('VOID_G') for s in slots)

    real = res.get('real', {})
    real_reading = real.get('reading', {})
    real_pass = bool(real_reading.get('pass')) if isinstance(real_reading, dict) else False

    if void_null or void_p3 or void_g:
        verdict = 'VOID'
    elif real_pass:
        verdict = 'ADVANCES'
    else:
        verdict = 'DOES NOT ADVANCE'

    # 4. DOES NOT ADVANCE label:
    mde_p1_vals = [fam[s].get('MDE', {}).get('P1') for s in slots]
    both_mde_le_02 = (len(mde_p1_vals) > 0 and all(v is not None and v <= 0.02 for v in mde_p1_vals))
    max_mde = max(mde_p1_vals) if both_mde_le_02 else None

    label = None
    if verdict == 'DOES NOT ADVANCE':
        if both_mde_le_02:
            label = f'informative null (MDE_P1 <= {max_mde:g} in both slots; linear T1, richer accessibility features)'
        else:
            label = 'uninformative null: the calibration does not detect a planted gain at <= 2 % in both slots'

    # Append G-uninformative note per slot
    g_inf_map = {s: fam[s].get('G_informative', False) for s in slots}
    for s in slots:
        if not g_inf_map[s]:
            note = f"slot {s} the gene-generic check is uninformative (G's raw delta below the bar)"
            label = (label + '; ' if label else '') + note

    reported = {
        'Cp': real.get('Cp_vs_B'),
        'loco': real.get('loco'),
        'loco_C_minus_B': real.get('loco_C_minus_B'),
        'loco_C_minus_N1': real.get('loco_C_minus_N1'),
        'shrinkage': real.get('shrinkage'),
        'measured_dev_cells': res.get('measured_dev_cells'),
        'fitting_cells': res.get('fitting_cells'),
        'k': res.get('k'),
    }

    out = {
        'reading': verdict,
        'label': label,
        'faults': {
            'VOID_null': void_null,
            'VOID_P3': void_p3,
            'VOID_G': void_g,
        },
        'G_check_informative': g_inf_map,
        'MDE': {s: fam[s].get('MDE', {}) for s in slots},
        'conjuncts': real_reading.get('conjuncts') if isinstance(real_reading, dict) else None,
        'reported': reported,
    }
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: read_h1.py DIR')
    d = sys.argv[1]
    res, m = _verified(d)
    out = read_h1(res)
    out['marker'] = m
    print(json.dumps(cf.jsonable(out), indent=1))


if __name__ == '__main__':
    main()
