# -*- coding: utf-8 -*-
"""Tests for read_h1.py on fabricated outputs (never real data)."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_h1 as rdh1  # noqa: E402


def _write_file_and_marker(tmp, out_name, marker_name, obj, corrupt=False, complete=True):
    p = tmp / out_name
    p.write_text(json.dumps(obj))
    h = hashlib.sha1(p.read_bytes()).hexdigest()
    (tmp / marker_name).write_text(json.dumps({
        'complete': complete,
        'outputs': {out_name: ('0' * 40 if corrupt else h)}
    }))


def _make_res(real_pass=False, mde_p1_slot1=0.02, mde_p1_slot2=0.02,
              void_null=False, void_p3=False, void_g=False,
              g_inf_slot1=True, g_inf_slot2=True, not_run=None):
    res = {}
    if not_run is not None:
        res['not_run'] = not_run

    res['faults_and_mde'] = {
        1: {
            'VOID_null': void_null,
            'VOID_P3': void_p3,
            'VOID_G': void_g,
            'G_informative': g_inf_slot1,
            'MDE': {'P1': mde_p1_slot1, 'P2': None},
        },
        2: {
            'VOID_null': False,
            'VOID_P3': False,
            'VOID_G': False,
            'G_informative': g_inf_slot2,
            'MDE': {'P1': mde_p1_slot2, 'P2': None},
        }
    }
    res['real'] = {
        'reading': {
            'pass': real_pass,
            'delta': {'all': 0.005},
            'conjuncts': {'dummy': True},
        },
        'Cp_vs_B': {'delta': {'all': 0.003}},
        'loco': {'B': 0.40, 'C': 0.45, 'N1': 0.42},
        'loco_C_minus_B': 0.05,
        'loco_C_minus_N1': 0.03,
        'shrinkage': {'best_lambda': 0.5},
    }
    res['measured_dev_cells'] = ['HEK293T', 'HL60', 'LNCAP', 'SKBR3', 'U937']
    res['fitting_cells'] = ['T00', 'T01']
    res['k'] = 4
    return res


def test_marker_refusal(tmp_path):
    # Missing marker
    with pytest.raises(SystemExit, match='REFUSED: no completion marker'):
        rdh1._verified(str(tmp_path))

    # Corrupt sha1
    res = _make_res(real_pass=True)
    _write_file_and_marker(tmp_path, 'chromatin_h1_93.json', 'CHROMATIN93_H1_COMPLETE.json', res, corrupt=True)
    with pytest.raises(SystemExit, match='REFUSED: .* does not match the marker'):
        rdh1._verified(str(tmp_path))

    # Incomplete marker
    _write_file_and_marker(tmp_path, 'chromatin_h1_93.json', 'CHROMATIN93_H1_COMPLETE.json', res, complete=False)
    with pytest.raises(SystemExit, match='REFUSED'):
        rdh1._verified(str(tmp_path))

    # Valid marker
    _write_file_and_marker(tmp_path, 'chromatin_h1_93.json', 'CHROMATIN93_H1_COMPLETE.json', res, corrupt=False)
    loaded, m = rdh1._verified(str(tmp_path))
    assert loaded['k'] == 4
    assert m['complete'] is True


def test_order_not_run_then_void_then_advances_then_does_not_advance():
    # 1. NOT RUN
    res_nr = _make_res(real_pass=True, void_null=True, not_run='too few measured dev cells')
    assert rdh1.read_h1(res_nr)['reading'] == 'NOT RUN'

    # 2. VOID (null pass, P3 pass, or informative G pass)
    res_void_null = _make_res(real_pass=True, void_null=True)
    assert rdh1.read_h1(res_void_null)['reading'] == 'VOID'

    res_void_p3 = _make_res(real_pass=True, void_p3=True)
    assert rdh1.read_h1(res_void_p3)['reading'] == 'VOID'

    # G pass when G is informative -> VOID
    res_void_g_inf = _make_res(real_pass=True, void_g=True, g_inf_slot1=True)
    assert rdh1.read_h1(res_void_g_inf)['reading'] == 'VOID'

    # G pass when G is NOT informative -> still VOID (PI fix before any run, as read_chromatin93)
    res_g_uninf = _make_res(real_pass=True, void_g=True, g_inf_slot1=False)
    assert rdh1.read_h1(res_g_uninf)['reading'] == 'VOID'

    # 3. ADVANCES (no void, real pass True)
    res_adv = _make_res(real_pass=True)
    assert rdh1.read_h1(res_adv)['reading'] == 'ADVANCES'

    # 4. DOES NOT ADVANCE (no void, real pass False)
    res_no_adv = _make_res(real_pass=False)
    assert rdh1.read_h1(res_no_adv)['reading'] == 'DOES NOT ADVANCE'


def test_informative_and_uninformative_labels():
    # MDE_P1 <= 0.02 in both slots -> informative
    res_inf = _make_res(real_pass=False, mde_p1_slot1=0.005, mde_p1_slot2=0.02)
    out_inf = rdh1.read_h1(res_inf)
    assert out_inf['reading'] == 'DOES NOT ADVANCE'
    assert out_inf['label'].startswith('informative null')

    # MDE_P1 > 0.02 in slot 1 -> uninformative
    res_uninf1 = _make_res(real_pass=False, mde_p1_slot1=0.05, mde_p1_slot2=0.02)
    out_uninf1 = rdh1.read_h1(res_uninf1)
    assert out_uninf1['label'].startswith('uninformative null')

    # MDE_P1 is None in slot 2 -> uninformative
    res_uninf2 = _make_res(real_pass=False, mde_p1_slot1=0.02, mde_p1_slot2=None)
    out_uninf2 = rdh1.read_h1(res_uninf2)
    assert out_uninf2['label'].startswith('uninformative null')


def test_g_uninformative_note():
    # G informative in both slots -> no G-uninformative note
    res_both_g_inf = _make_res(real_pass=False, g_inf_slot1=True, g_inf_slot2=True)
    out = rdh1.read_h1(res_both_g_inf)
    assert 'gene-generic check is uninformative' not in out['label']

    # G uninformative in slot 1 -> note appended
    res_slot1_uninf = _make_res(real_pass=False, g_inf_slot1=False, g_inf_slot2=True)
    out1 = rdh1.read_h1(res_slot1_uninf)
    assert 'slot 1 the gene-generic check is uninformative' in out1['label']
    assert 'slot 2' not in out1['label']

    # G uninformative in both slots -> notes for both
    res_both_uninf = _make_res(real_pass=False, g_inf_slot1=False, g_inf_slot2=False)
    out2 = rdh1.read_h1(res_both_uninf)
    assert 'slot 1 the gene-generic check is uninformative' in out2['label']
    assert 'slot 2 the gene-generic check is uninformative' in out2['label']


def test_reported_items_echoed():
    res = _make_res(real_pass=True)
    out = rdh1.read_h1(res)
    rep = out['reported']
    assert rep['Cp'] == {'delta': {'all': 0.003}}
    assert rep['loco'] == {'B': 0.40, 'C': 0.45, 'N1': 0.42}
    assert rep['loco_C_minus_B'] == 0.05
    assert rep['loco_C_minus_N1'] == 0.03
    assert rep['shrinkage'] == {'best_lambda': 0.5}
    assert rep['measured_dev_cells'] == ['HEK293T', 'HL60', 'LNCAP', 'SKBR3', 'U937']
    assert rep['k'] == 4


def test_any_G_pass_voids_even_when_G_is_not_informative():
    """PI fix (before any run): as read_chromatin93, a gene-generic pass anywhere is VOID; G_informative only labels a null."""
    res = {'real': {'reading': {'pass': True, 'conjuncts': {}}},
           'faults_and_mde': {s: {'VOID_null': False, 'VOID_P3': False, 'VOID_G': s == 1, 'G_informative': False,
                                  'MDE': {'P1': 0.005, 'P2': None}} for s in (1, 2)}}
    assert rdh1.read_h1(res)['reading'] == 'VOID'
