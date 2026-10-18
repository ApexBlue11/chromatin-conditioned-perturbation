"""Tests for the T-B reading (RESULTS 100.4's rule, 100.9's post hoc view)."""
import json, math, os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_read as tb  # noqa: E402

CELLS = ['A', 'B', 'C', 'D', 'E', 'F']


def cells(v):
    return {c: v + 0.001 * i for i, c in enumerate(CELLS)}


def test_rule_by_hand_pass_and_threshold():
    on, off = [0.170, 0.172, 0.174], [0.160, 0.162, 0.164]
    r = tb.rule(on, off, [cells(x) for x in on], [cells(x) for x in off])
    s = np.std(on, ddof=1)
    assert abs(r['delta'] - 0.010) < 1e-12 and abs(r['threshold'] - max(0.003, 2 * math.sqrt(2 * s ** 2 / 3))) < 1e-12
    assert r['passes'] and r['n_cells_favouring_on'] == 6 and r['cell_mean_delta'] > 0


def test_rule_fails_on_any_of_the_three_parts():
    on, off = [0.170, 0.172, 0.174], [0.160, 0.162, 0.164]
    few = [{**cells(x), 'A': 0.0, 'B': 0.0, 'C': 0.0} for x in on]          # 3 of 6 cells favour on
    assert not tb.rule(on, off, few, [cells(x) for x in off])['passes']
    small = [0.1612, 0.1622, 0.1632]                                        # delta 0.0002 < 0.003
    assert not tb.rule(small, off, [cells(x) for x in small], [cells(x) for x in off])['passes']


def test_unequal_n_threshold():
    on, off = [0.17, 0.18], [0.16, 0.161, 0.162]
    r = tb.rule(on, off, [cells(x) for x in on], [cells(x) for x in off])
    want = 2 * math.sqrt(np.var(on, ddof=1) / 2 + np.var(off, ddof=1) / 3)
    assert r['n_on'] == 2 and abs(r['threshold'] - max(0.003, want)) < 1e-12


def test_one_collapsed_on_seed_cannot_pass():
    """Review 090, Ask 2: with one on seed at -0.0001 the rule cannot pass unless the off mean is <= -0.0001."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        on = [-0.0001, *rng.uniform(-0.1, 0.6, 2)]
        off = list(rng.uniform(0.0, 0.3, 3))
        r = tb.rule(on, off, [cells(x) for x in on], [cells(x) for x in off])
        assert not r['delta_clears_threshold']


def test_trained_flags_and_tiers(tmp_path):
    p = tmp_path / 'chk.json'
    json.dump({'runs': [{'checkpoint': 'v10c_cold_cell_1_dev_on_d256_s0.pt', 'drug_path_trained': False},
                        {'checkpoint': 'v10c_cold_cell_1_dev_off_d256_s2.pt', 'drug_path_trained': True}]}, open(p, 'w'))
    f = tb.trained_flags([str(p)])
    assert f == {'cold_cell_1_dev_on_d256_s0': False, 'cold_cell_1_dev_off_d256_s2': True}
    sel = tmp_path / 'sel.json'
    json.dump({'selection': {'X': {'ACC': ['s1', 's2'], 'PCG': ['s3']}},
               'records': [{'srx': 's1', 'tier': 'E'}, {'srx': 's2', 'tier': 'N'}, {'srx': 's3', 'tier': 'C'}]}, open(sel, 'w'))
    assert tb.tiers(['X', 'Y'], str(sel)) == {'X': {'ACC': ['E', 'N'], 'PCG': ['C']}, 'Y': {}}


def test_load_runs_refuses_a_record_that_differs_from_the_ledger(tmp_path):
    import hashlib
    led = tmp_path / 'ledger.jsonl'
    with open(led, 'w') as L:
        for arm in tb.ARMS:
            for s in tb.SEEDS:
                t, tag = f'tb_{arm}_s{s}', tb.tag_of(arm, s)
                d = tmp_path / t / t
                d.mkdir(parents=True)
                raw = json.dumps({'tag': tag}).encode()
                (d / f'V10C_COMPLETE_{tag}.json').write_bytes(raw)
                L.write(json.dumps({'run': tag, 'record_sha1': hashlib.sha1(raw).hexdigest()}) + '\n')
    assert len(tb.load_runs(str(tmp_path), str(led))) == 6
    (tmp_path / 'tb_on_s1' / 'tb_on_s1' / f"V10C_COMPLETE_{tb.tag_of('on', 1)}.json").write_text('{"tag": "edited"}')
    with pytest.raises(SystemExit, match='not the record'):
        tb.load_runs(str(tmp_path), str(led))
