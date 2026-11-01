"""Tests for v11_probe_head.py (RESULTS 102.8, review 104 C1)."""
import os, sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
import v11_probe as vp  # noqa: E402
import v11_probe_head as vh  # noqa: E402
from test_v11_model import write_v11_fixture, built_pair  # noqa: E402


def test_expected_recall_splits_ties_and_equals_the_stable_rank_without_them():
    s = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
    assert vh.expected_recall(s, {1}, (1, 2, 3, 4)) == pytest.approx({1: 0.0, 2: 1 / 3, 3: 2 / 3, 4: 1.0})
    assert vh.expected_recall(s, {0, 4}, (1, 4, 5)) == pytest.approx({1: 0.5, 4: 0.5, 5: 1.0})
    neg = np.array([0.5, -np.inf, -np.inf, 0.1])
    assert vh.expected_recall(neg, {2}, (2, 3, 4)) == pytest.approx({2: 0.0, 3: 0.5, 4: 1.0})
    r = np.random.default_rng(1).normal(size=50)
    t = {3, 7, 21}
    assert vh.expected_recall(r, t, (5, 10, 20)) == pytest.approx(vp.recall_at_k(r, t, (5, 10, 20)))


def test_lexi_score_orders_by_primary_then_secondary_and_ties_only_on_both():
    v = vh.lexi_score([0.9, 0.5, 0.5, -1.0, -1.0], [0.0, 0.2, 0.1, 0.0, 0.0])
    assert v[0] > v[1] > v[2] > v[3] and v[3] == v[4]


def test_nn_ranks_the_nearest_known_compounds_targets_first_then_popularity():
    K = np.zeros((3, 6), bool)
    K[0, [0, 1]] = True                     # the nearest known compound targets genes 0 and 1
    K[1, 2] = True
    K[2, 3] = True
    prior = np.array([0.01, 0.02, 0.05, 0.30, -np.inf, -np.inf])
    s = vh.nn_scores([0.9, 0.5, 0.5], K, prior)
    order = list(np.argsort(-s, kind='stable'))
    assert order[:4] == [1, 0, 3, 2] and s[4] == s[5] and s[4] < s[2]


def test_tanimoto_matrix_maxes_to_the_probes_tanimoto_max():
    fp = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 0]])
    M = vh.tanimoto_matrix(fp, [0, 3], [1, 2])
    assert np.allclose(M.max(1), vp.tanimoto_max(fp, [0, 3], [1, 2])) and np.allclose(M[0], [0.5, 0.0])


def test_same_ranking_needs_the_same_order_and_the_same_ties():
    p = np.array([-3.1, -3.4, -3.4, -6.9])
    assert vh.same_ranking(p * 0.998, p)                    # a uniform shrink keeps order and ties
    assert not vh.same_ranking(np.array([-3.1, -3.4, -3.41, -6.9]), p)      # a broken tie
    assert not vh.same_ranking(np.array([-3.5, -3.4, -3.4, -6.9]), p)       # a swapped order


def test_paired_bootstrap_is_reproducible_and_brackets_the_mean():
    a, b = np.linspace(0, 1, 40), np.linspace(0, 1, 40)[::-1] * 0.5
    r1, r2 = vh.paired_bootstrap(a, b, 2000), vh.paired_bootstrap(a, b, 2000)
    assert r1 == r2 and r1['ci95'][0] <= r1['mean_diff'] <= r1['ci95'][1] and r1['n'] == 40
    z = vh.paired_bootstrap(a, a, 500)
    assert z['mean_diff'] == 0.0 and z['ci95'] == [0.0, 0.0]


def test_verdict_wording_follows_the_registered_rule():
    assert vh.verdict_popularity({'ci95': [0.01, 0.05]}).startswith('the head ranks')
    assert vh.verdict_popularity({'ci95': [-0.01, 0.05]}).startswith('not established')
    assert vh.verdict_nn({'ci95': [0.01, 0.05]}) == 'the head transfers beyond nearest-neighbour lookup'
    assert vh.verdict_nn({'ci95': [-0.05, -0.01]}) == 'a nearest-neighbour lookup does better than the head'
    assert vh.verdict_nn({'ci95': [-0.05, 0.01]}) == 'the head and a nearest-neighbour lookup are not distinguished'


def test_head_logits_equal_the_forward_passs_target_logits(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    with torch.no_grad():
        for w in (v11.w_u[0].weight, v11.target_head.weight):  # move off init so the check is not trivial
            w.add_(0.05 * torch.randn_like(w))
    v11.eval()
    perts = np.unique(D.pert[D.ev])[:4]
    rows = {p: D.ev[D.pert[D.ev] == p][:3] for p in perts}
    got = vh.head_logits(v11, D, rows)
    with torch.no_grad():
        for p, r in rows.items():
            out = v11(D.batch(r, 'cpu'))
            assert np.allclose(got[p], out['target_logits'][0].float().numpy(), atol=1e-6)


def test_popularity_is_the_prior_on_vocabulary_landmarks_and_the_untrained_bias(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path, arm='no_targets')
    pr = vh.landmark_prior(D)
    lm = list(D.landmark_idx)
    inv = {int(n) for n in D.vocab} & {int(n) for n in lm}
    assert np.isfinite(pr).sum() == len(inv)
    q = np.clip(D.vocab_prior.astype(np.float64), 1e-4, 1 - 1e-4)
    assert np.allclose(v11.target_head.bias.detach().double().numpy(), np.log(q / (1 - q)), atol=1e-5)
    for n in inv:
        assert pr[lm.index(n)] == pytest.approx(float(D.vocab_prior[list(D.vocab).index(n)]))
