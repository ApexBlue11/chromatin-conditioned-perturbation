"""Unit tests for tc_gap.py (the end-to-end reproduction of v9's references is tc_gap_repro.py, 100.11)."""
import os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tc_gap  # noqa: E402


def test_align_reorders_to_the_reference_and_refuses_other_rows(tmp_path):
    p = tmp_path / 'pred.npz'
    np.savez(p, bench_row=np.array([30, 10, 20]), sub_row=np.array([3, 1, 2]), delta=np.arange(6.).reshape(3, 2))
    d, sub = tc_gap.align(str(p), [10, 20, 30])
    assert np.array_equal(sub, [1, 2, 3]) and np.array_equal(d, [[2., 3.], [4., 5.], [0., 1.]])
    with pytest.raises(SystemExit, match='rows differ'):
        tc_gap.align(str(p), [10, 20, 31])
    with pytest.raises(SystemExit, match='rows differ'):
        tc_gap.align(str(p), [10, 20])


def test_gap_is_compound_minus_cell_per_target():
    cell = {'m': {'contrast': {'clean7': {'row_mean': 0.16}, 'all8_labelled': {'row_mean': 0.17}},
                  'delta_r': {'clean7': {'row_mean': 0.48}, 'all8_labelled': {'row_mean': 0.49}}}}
    comp = {'m': {'contrast_clean': 0.11, 'delta_r_clean': 0.64}}
    g = tc_gap.gap(cell, comp, 'm', 'm')
    assert g['clean7']['contrast'] == pytest.approx(-0.05) and g['all8_labelled']['delta_r'] == pytest.approx(0.15)


def test_summarise_primary_is_the_seven_clean_cells():
    r = np.array([0.1, 0.3, 0.9, np.nan])
    cell = np.array(['MCF7', 'HT29', 'H1975', 'MCF7'])
    s7 = tc_gap.summarise(r, cell, ('MCF7', 'HT29'))
    s8 = tc_gap.summarise(r, cell, ('MCF7', 'HT29', 'H1975'))
    assert s7['n_rows'] == 2 and s7['row_mean'] == pytest.approx(0.2) and s8['row_mean'] == pytest.approx(13 / 30)
    assert s7['per_cell'] == {'MCF7': pytest.approx(0.1), 'HT29': pytest.approx(0.3)}


# --- review 092: C1 provenance, C2 completed slots, C3 the self-check ---------------------------------------------------
import json  # noqa: E402
import tc_ledger  # noqa: E402


def _gap_kernels(root, ledger, rcs=(0, 0, 0)):
    for s, rc in enumerate(rcs):
        slot = f'gap_on_s{s}'
        kd = root / slot
        (kd / slot).mkdir(parents=True)
        if rc == 0:
            pp = kd / slot / f'v10c_pred_t{s}.npz'
            np.savez(pp, bench_row=np.arange(3), sub_row=np.arange(3), delta=np.zeros((3, 2)))
            (kd / slot / 'V10C_COMPLETE_x.json').write_text(json.dumps({'tag': f't{s}', 'pred_sha1': tc_gap.sha1_file(pp)}))
        else:
            (kd / slot / 'V10C_FAILED_x.json').write_text(json.dumps({'step': 1000, 'guard_level': 0.0}))
        att = [{'attempt': 0, 'returncode': rc, 'out': f'/kaggle/working/{slot}', 'args': ['--seed', str(s)]}]
        (kd / 'V10C_KERNEL_COMPLETE.json').write_text(json.dumps({'attempts': {slot: att}}))
        tc_ledger.add(str(kd), ledger=str(ledger))


def test_completed_preds_checks_the_ledger_and_the_prediction_file(tmp_path):
    led = tmp_path / 'ledger.jsonl'
    _gap_kernels(tmp_path / 'runs', led)
    preds, recs, missing = tc_gap.completed_preds(str(tmp_path / 'runs'), 'gap', 'on', str(led))
    assert len(preds) == 3 and missing == []
    np.savez(preds[1], bench_row=np.arange(3), sub_row=np.arange(3), delta=np.ones((3, 2)))      # tampered predictions
    with pytest.raises(SystemExit, match='not the prediction file'):
        tc_gap.completed_preds(str(tmp_path / 'runs'), 'gap', 'on', str(led))


def test_completed_preds_refuses_an_unlogged_record(tmp_path):
    led = tmp_path / 'ledger.jsonl'
    _gap_kernels(tmp_path / 'runs', led)
    rec = tmp_path / 'runs' / 'gap_on_s2' / 'gap_on_s2' / 'V10C_COMPLETE_x.json'
    d = json.load(open(rec))
    d['note'] = 'edited'
    rec.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match='not the one in the ledger'):
        tc_gap.completed_preds(str(tmp_path / 'runs'), 'gap', 'on', str(led))


def test_completed_preds_reports_over_completed_slots(tmp_path):
    led = tmp_path / 'ledger.jsonl'
    _gap_kernels(tmp_path / 'runs', led, rcs=(0, 3, 0))
    preds, recs, missing = tc_gap.completed_preds(str(tmp_path / 'runs'), 'gap', 'on', str(led))
    assert len(preds) == 2 and missing == [1]


def _ref_world(bump=None):
    cell = {m: {t: {sc: {'row_mean': 0.0} for sc in ('clean7', 'all8_labelled')} for t in ('contrast', 'delta_r')}
            for m in ('v9', 'xpert_o2')}
    comp = {'v9': {'contrast_clean': 0.0, 'delta_r_clean': 0.0, 'n_contrast_rows': 7980}}
    for k, v in tc_gap.REFS.items():
        if k[0] == 'cell':
            cell[k[1]][k[2]][k[3]]['row_mean'] = v
        else:
            comp[k[1]][k[2]] = v
    if bump:
        cell['xpert_o2']['delta_r']['clean7']['row_mean'] += bump
    return cell, comp


def test_self_check_passes_on_the_references_and_refuses_drift():
    assert len(tc_gap.self_check(*_ref_world())) == len(tc_gap.REFS)
    tc_gap.self_check(*_ref_world(bump=5e-5))
    with pytest.raises(SystemExit, match='reused scorer changed'):
        tc_gap.self_check(*_ref_world(bump=2e-4))
    cell, comp = _ref_world()
    comp['v9']['n_contrast_rows'] = 7979
    with pytest.raises(SystemExit, match='7,980'):
        tc_gap.self_check(cell, comp)
