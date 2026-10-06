# -*- coding: utf-8 -*-
"""Tests for read_chromatin93.py on fabricated outputs (never real data)."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_chromatin93 as rd  # noqa: E402

_SC = json.load(open(rd.F91))['row_sets']['known']['scores']
D91_FB = _SC['FBC']['all'] - _SC['FB']['all']
D91_N1 = _SC['FBC']['all'] - _SC['N1']['all']


def _write(tmp, name, marker, obj, corrupt=False):
    p = tmp / name
    p.write_text(json.dumps(obj))
    h = hashlib.sha1(p.read_bytes()).hexdigest()
    (tmp / marker).write_text(json.dumps({'complete': True, 'outputs': {name: ('0' * 40 if corrupt else h)}}))


def _h2(meds, sd4=0.0001, repro=True):
    real = {k: [{'FBC_minus_N1': v + (sd4 * (i - 2) if k == '4' else 0.0), 'FBC_minus_FB': v + 0.001}
                for i in range(5 if k != 'all' else 1)] for k, v in meds.items()}
    real['all'] = [{'FBC_minus_N1': meds['all'] if not repro else D91_N1, 'FBC_minus_FB': D91_FB if repro else 0.5}]
    return {'real': real, 'planted_P1': real}


def test_h2_refuses_a_bad_marker_and_reads_rising_only_when_monotone_and_large(tmp_path):
    _write(tmp_path, 'chromatin_h2_93.json', 'CHROMATIN93_H2_COMPLETE.json', _h2({'4': 0.0, '6': 0.0001, '8': 0.0002, 'all': 0.0}), corrupt=True)
    with pytest.raises(SystemExit):
        rd._verified(str(tmp_path), 'chromatin_h2_93.json', 'CHROMATIN93_H2_COMPLETE.json')
    lo, mid = D91_N1 - 0.0008, D91_N1 - 0.0004
    r = rd.read_h2(_h2({'4': lo, '6': mid, '8': D91_N1 - 0.0001, 'all': D91_N1}))
    assert r['reading'] == 'RISING' and r['k_all_reproduces_91'] and r['HARNESS_FAULT'] is None
    assert rd.read_h2(_h2({'4': lo, '6': D91_N1 - 0.0001, '8': mid, 'all': D91_N1}))['reading'] == 'NOT RISING'   # not monotone
    assert rd.read_h2(_h2({'4': lo, '6': mid, '8': D91_N1 - 0.0001, 'all': D91_N1}, sd4=0.01))['reading'] == 'NOT RISING'
    bad = rd.read_h2(_h2({'4': lo, '6': mid, '8': D91_N1 - 0.0001, 'all': D91_N1}, repro=False))
    assert not bad['k_all_reproduces_91'] and bad['HARNESS_FAULT']


def _h3(real_pass, mde_p1, void=False, g_delta=0.01):
    f = {'VOID_null': void, 'VOID_P3': False, 'VOID_G': False, 'G_raw_delta': [g_delta], 'MDE': {'P1': mde_p1, 'P2': None}}
    rk = {'pass': real_pass, 'conjuncts': {}, 'delta': {'all': 0.001, 'top': 0.0, 'centred_all': 0.0}, 'vs_N1': {'all': 0.0}}
    cal = [{'cases': {'G_0.02': {'known': {'delta': {'all': g_delta}}}, 'G_0.05': {'known': {'delta': {'all': g_delta}}}}}
           for _ in range(3)]
    return {'faults_and_mde': {'known': f}, 'real': {'known': rk, 'all': rk, 'chosen_mcs': {}}, 'calibration': cal}


def test_h3_reads_void_then_advances_then_its_null_label_and_flags_an_uninformative_G_check():
    assert rd.read_h3(_h3(True, 0.005, void=True))['reading'] == 'VOID'
    assert rd.read_h3(_h3(True, 0.005))['reading'] == 'ADVANCES'
    r = rd.read_h3(_h3(False, 0.02))
    assert r['reading'] == 'DOES NOT ADVANCE' and r['label'].startswith('informative') and r['G_check_informative']
    assert rd.read_h3(_h3(False, 0.05))['label'].startswith('uninformative')
    assert rd.read_h3(_h3(False, None))['label'].startswith('uninformative')
    g = rd.read_h3(_h3(False, 0.02, g_delta=0.001))
    assert not g['G_check_informative'] and 'gene-generic check is uninformative' in g['label']
    assert rd.read_h3(_h3(True, 0.005, g_delta=0.001))['reading'] == 'ADVANCES'   # not VOID: the magnitude conjunct protects it
