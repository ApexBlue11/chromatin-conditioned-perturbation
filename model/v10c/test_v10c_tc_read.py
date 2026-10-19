"""Tests for the T-C reading (RESULTS 100.5 / 100.8)."""
import json, os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tc_read as tc  # noqa: E402


def per_cell(base, bump=None):
    d = {c: base + 0.01 * i for i, c in enumerate(tc.CLEAN + tc.APART)}
    d.update(bump or {})
    return d


def test_rule_claims_only_with_five_cells_and_a_ci_above_zero():
    on = [per_cell(0.20) for _ in range(3)]
    off = [per_cell(0.19) for _ in range(3)]
    r = tc.rule(on, off)
    assert r['n_cells_on_better'] == 7 and r['ci95'][0] > 0 and r['verdict'].startswith('cell-specific chromatin contributes')
    mixed = [per_cell(0.20, {'BJAB': 0.0, 'CD34': 0.0, 'HS578T': 0.0}) for _ in range(3)]
    assert tc.rule(mixed, off)['verdict'] == 'no claim'                     # 4 of 7
    r2 = tc.rule(off, on)
    assert r2['verdict'].startswith('chromatin hurts') and r2['ci95'][1] < 0


def test_h1975_is_not_in_the_rule():
    on = [per_cell(0.20, {'H1975': -1.0}) for _ in range(3)]
    off = [per_cell(0.19) for _ in range(3)]
    r = tc.rule(on, off)
    assert 'H1975' not in r['delta_c'] and r['verdict'].startswith('cell-specific')


def test_cluster_ci_by_hand():
    lo, hi = tc.cluster_ci([0.01] * 7, n_boot=100)
    assert lo == hi == pytest.approx(0.01)
    lo, hi = tc.cluster_ci([-0.01, 0.01] * 3 + [0.0], n_boot=2000)
    assert lo < 0 < hi


def _kernel(root, name, tag, attempts):
    """attempts: list of (attempt, returncode); a completed attempt gets one V10C_COMPLETE record."""
    kd = root / name
    kd.mkdir(parents=True)
    att = []
    for k, rc in attempts:
        sub = tag if k == 0 else f'{tag}_retry{k}'          # the kernel template's out dir
        (kd / sub).mkdir(exist_ok=True)
        if rc == 0:
            (kd / sub / f'V10C_COMPLETE_x_s{k}.json').write_text(json.dumps({'attempt': k}))
        att.append({'attempt': k, 'returncode': rc, 'out': f'/kaggle/working/{sub}'})
    (kd / 'V10C_KERNEL_COMPLETE.json').write_text(json.dumps({'attempts': {tag: att}}))


def test_first_completed_attempt_across_kernels(tmp_path):
    _kernel(tmp_path, 'tc_on_s0', 'tc_on_s0', [(0, 3)])
    _kernel(tmp_path, 'tc_on_s0_r1', 'tc_on_s0', [(1, 3), (2, 0)])
    att = tc.slot_attempts(str(tmp_path), 'on', 0)
    assert [(a, rc) for a, rc, _ in att] == [(0, 3), (1, 3), (2, 0)]
    k, rec = tc.first_completed(att)
    assert k == 2 and json.load(open(rec))['attempt'] == 2


def test_exhausted_slot_and_gaps_in_attempts(tmp_path):
    _kernel(tmp_path, 'tc_off_s1', 'tc_off_s1', [(0, 3), (1, 3), (2, 3)])
    assert tc.first_completed(tc.slot_attempts(str(tmp_path), 'off', 1)) == (None, None)
    _kernel(tmp_path / 'b', 'tc_off_s1', 'tc_off_s1', [(0, 3)])
    _kernel(tmp_path / 'b', 'tc_off_s1_r2', 'tc_off_s1', [(2, 0)])               # attempt 1 is missing
    with pytest.raises(SystemExit, match='not 0..k'):
        tc.slot_attempts(str(tmp_path / 'b'), 'off', 1)


def _deltas(vals):
    off = [{c: 0.2 for c in tc.CLEAN + tc.APART} for _ in range(3)]
    on = [{**{c: 0.2 + v for c, v in zip(tc.CLEAN, vals)}, 'H1975': 0.2} for _ in range(3)]
    return on, off


def test_five_cells_but_ci_spanning_zero_is_no_claim():
    on, off = _deltas([0.001] * 5 + [-0.05, -0.05])
    r = tc.rule(on, off)
    assert r['n_cells_on_better'] == 5 and r['ci95'][0] < 0 and r['verdict'] == 'no claim'


def test_four_cells_with_a_positive_ci_is_no_claim():
    on, off = _deltas([0.02] * 4 + [0.0] * 3)
    r = tc.rule(on, off)
    assert r['n_cells_on_better'] == 4 and r['ci95'][0] > 0 and r['verdict'] == 'no claim'


def test_mirror_needs_the_ci_too():
    on, off = _deltas([-0.001] * 5 + [0.05, 0.05])
    r = tc.rule(on, off)
    assert r['n_cells_off_better'] == 5 and r['ci95'][1] > 0 and r['verdict'] == 'no claim'


def test_the_ci_is_95_percent():
    lo, hi = tc.cluster_ci([0.01] * 6 + [-0.02])          # ~6.5 % of resamples have mean < 0
    assert lo < 0 < hi


def test_the_earliest_completed_attempt_is_used_and_attempts_are_ordered(tmp_path):
    kd = tmp_path / 'tc_on_s2'
    for sub in ('tc_on_s2', 'tc_on_s2_retry1'):
        (kd / sub).mkdir(parents=True)
        (kd / sub / f'V10C_COMPLETE_{sub}.json').write_text(json.dumps({'sub': sub}))
    att = [{'attempt': 1, 'returncode': 0, 'out': '/kaggle/working/tc_on_s2_retry1'},
           {'attempt': 0, 'returncode': 0, 'out': '/kaggle/working/tc_on_s2'}]     # listed out of order
    (kd / 'V10C_KERNEL_COMPLETE.json').write_text(json.dumps({'attempts': {'tc_on_s2': att}}))
    a = tc.slot_attempts(str(tmp_path), 'on', 2)
    assert [x[0] for x in a] == [0, 1]
    k, rec = tc.first_completed(a)
    assert k == 0 and json.load(open(rec))['sub'] == 'tc_on_s2'
