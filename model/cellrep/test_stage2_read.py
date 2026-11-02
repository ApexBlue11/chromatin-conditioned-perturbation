"""Tests for stage2_read.py's rules (RESULTS 103.6 / 103.7)."""
import json, os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stage2 as s2  # noqa: E402
import stage2_read as rd  # noqa: E402


def test_fold_rule_by_hand():
    r = rd.fold_rule([0.01, 0.012, 0.008, 0.011, 0.009, 0.010])
    assert r['passes'] and r['folds_positive'] == 6 and r['threshold'] == pytest.approx(max(0.003, 2 * np.std([0.01, 0.012, 0.008, 0.011, 0.009, 0.010], ddof=1) / np.sqrt(6)))
    assert not rd.fold_rule([0.02, 0.02, -0.01, -0.01, 0.02, 0.0])['passes']          # 3 of 6 positive
    assert not rd.fold_rule([0.002] * 6)['passes']                                    # under the 0.003 floor


def test_gene_bootstrap_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(0)
    assert rd.gene_bootstrap({i: x for i, x in enumerate(0.05 + 0.1 * rng.normal(size=500))})['passes']
    assert not rd.gene_bootstrap({i: x for i, x in enumerate(0.1 * rng.normal(size=500))})['passes']


def test_reader_tags_match_the_runners_tags():
    for job in ({'model': 'M3', 'setting': 'loco:PC3', 'seed': 2, 'K': 4},
                {'model': 'M1', 'setting': 'cold', 'seed': 0},
                {'model': 'M2', 'setting': 'loco:HT29', 'seed': 1, 'inject': {'shape': 'graph', 'share': 0.1}},
                {'model': 'M1', 'setting': 'loco:A549', 'seed': 0, 'inject': {'shape': 'dense', 'share': 0.05, 'residualise': False}}):
        inj = job.get('inject')
        name = None if not inj else f"inj-{inj['shape']}-{inj['share']}" + ('' if inj.get('residualise', True) else '-raw')
        assert rd.tag(job['model'], job['setting'], job['seed'], job.get('K', 3), name) == s2.job_tag(job)


def test_the_inject_phase_never_opens_real_outputs(tmp_path):
    (tmp_path / 'M1_loco-PC3_s0_K3_inj-dense-0.05.json').write_text(json.dumps({'job': {'inject': {}}}))
    (tmp_path / 'M1_loco-PC3_s0_K3.json').write_text('not json: opening this would fail')
    assert list(rd.load(str(tmp_path), inject=True)) == ['M1_loco-PC3_s0_K3_inj-dense-0.05']
