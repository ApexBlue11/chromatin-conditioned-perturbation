"""Tests for the width screen (RESULTS 100.4 as registered; 100.9 as read)."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import width_screen as ws  # noqa: E402


def test_registered_comparator_is_seed_0_and_can_only_err_toward_256():
    r256 = {0: -0.0001, 1: 0.163, 2: 0.166}
    rec = ws.screen(0.175, r256, True, {0: False, 1: True, 2: True})
    assert rec['registered_100_4']['width'] == 256 and rec['registered_100_4']['threshold'] > 0.18
    assert rec['read_100_9']['comparator_seeds'] == [1, 2] and rec['read_100_9']['width'] == 384


def test_untrained_d384_is_uninformative():
    rec = ws.screen(0.0, {0: -0.0001, 1: 0.163, 2: 0.166}, False, {0: False, 1: True, 2: True})
    assert rec['T_C_width'] == 256 and 'uninformative' in rec['read_100_9']['reason']


def test_small_gain_keeps_256():
    rec = ws.screen(0.1660, {0: 0.164, 1: 0.163, 2: 0.165}, True, {0: True, 1: True, 2: True})
    assert rec['read_100_9']['width'] == 256 and abs(rec['read_100_9']['delta'] - 0.002) < 1e-12
