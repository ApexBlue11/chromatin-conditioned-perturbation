# -*- coding: utf-8 -*-
"""Tests for map_cuda_rng -- the CUDA RNG state mapping across GPU count changes.
[RESULTS 84.4 condition 2, review 018 C2]. Runnable as: python model/v9/test_cuda_rng_map.py
"""
import hashlib
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from xpert_resume_patch import map_cuda_rng


def fake_seed_fn(i, seed):
    """Deterministic fake: 16-byte tensor from the seed, no GPU required."""
    return torch.tensor(list(seed.to_bytes(8, 'little')) + [0] * 8, dtype=torch.uint8)


def _sha1_bytes(t):
    return hashlib.sha1(t.numpy().tobytes()).hexdigest()


def test_a_2saved_1device():
    """(a) 2 saved, 1 device: targets == [saved[0]]; dropped_sha1 has key 1."""
    s0 = torch.tensor([10, 20, 30], dtype=torch.uint8)
    s1 = torch.tensor([40, 50, 60], dtype=torch.uint8)
    targets, rec = map_cuda_rng([s0, s1], 1, fake_seed_fn)
    assert len(targets) == 1
    assert torch.equal(targets[0], s0)
    assert rec['dropped_sha1'] == {'1': _sha1_bytes(s1)} or rec['dropped_sha1'] == {1: _sha1_bytes(s1)}
    # normalise: keys may be int; check the value
    assert list(rec['dropped_sha1'].values()) == [_sha1_bytes(s1)]
    assert int(list(rec['dropped_sha1'].keys())[0]) == 1
    assert rec['seeded'] == {}
    assert rec['restored'] == [0]
    print('  (a) PASS: 2 saved, 1 device')


def test_b_1saved_2devices():
    """(b) 1 saved, 2 devices: targets[0] == saved[0]; seeded == {1: 3024}."""
    s0 = torch.tensor([10, 20, 30], dtype=torch.uint8)
    targets, rec = map_cuda_rng([s0], 2, fake_seed_fn)
    assert len(targets) == 2
    assert torch.equal(targets[0], s0)
    # seeded: device 1 with base_seed + stride * 1 = 2024 + 1000 = 3024
    assert rec['seeded'] == {1: 3024} or rec['seeded'] == {'1': 3024}
    seed_val = list(rec['seeded'].values())[0]
    assert seed_val == 3024
    assert torch.equal(targets[1], fake_seed_fn(1, 3024))
    assert rec['dropped_sha1'] == {}
    print('  (b) PASS: 1 saved, 2 devices')


def test_c_2saved_2devices_distinct():
    """(c) 2 saved, 2 devices, distinct: targets == saved; restored == [0, 1]."""
    s0 = torch.tensor([10, 20, 30], dtype=torch.uint8)
    s1 = torch.tensor([40, 50, 60], dtype=torch.uint8)
    targets, rec = map_cuda_rng([s0, s1], 2, fake_seed_fn)
    assert len(targets) == 2
    assert torch.equal(targets[0], s0)
    assert torch.equal(targets[1], s1)
    assert rec['restored'] == [0, 1]
    assert rec['seeded'] == {}
    assert rec['dropped_sha1'] == {}
    print('  (c) PASS: 2 saved, 2 devices, distinct')


def test_d_2saved_2devices_identical_raises():
    """(d) 2 saved, 2 devices, byte-identical states: raises RuntimeError."""
    s0 = torch.tensor([10, 20, 30], dtype=torch.uint8)
    s1 = s0.clone()
    try:
        map_cuda_rng([s0, s1], 2, fake_seed_fn)
        assert False, 'Expected RuntimeError for byte-identical states'
    except RuntimeError as e:
        assert 'byte-equal' in str(e)
    print('  (d) PASS: byte-identical states raise RuntimeError')


def test_e_0saved_1device():
    """(e) 0 saved, 1 device: seeded == {0: 2024}."""
    targets, rec = map_cuda_rng([], 1, fake_seed_fn)
    assert len(targets) == 1
    assert rec['seeded'] == {0: 2024} or rec['seeded'] == {'0': 2024}
    seed_val = list(rec['seeded'].values())[0]
    assert seed_val == 2024
    assert torch.equal(targets[0], fake_seed_fn(0, 2024))
    assert rec['restored'] == []
    assert rec['dropped_sha1'] == {}
    print('  (e) PASS: 0 saved, 1 device')


def test_f_json_serialisable():
    """(f) the record is json.dumps-able."""
    s0 = torch.tensor([10, 20, 30], dtype=torch.uint8)
    _, rec = map_cuda_rng([s0], 1, fake_seed_fn)
    s = json.dumps(rec)
    assert isinstance(s, str)
    # also test a more complex case
    s1 = torch.tensor([40, 50, 60], dtype=torch.uint8)
    _, rec2 = map_cuda_rng([s0, s1], 1, fake_seed_fn)
    s2 = json.dumps(rec2)
    assert isinstance(s2, str)
    print('  (f) PASS: record is JSON-serialisable')


def test_g_real_device():
    """(g) real-device check on device 0 if CUDA is available."""
    if not torch.cuda.is_available():
        print('  (g) SKIP: no CUDA')
        return
    n_devices = torch.cuda.device_count()
    # Build saved = [s0, s1] where s0 is the state after manual_seed(7) on device 0
    torch.cuda.manual_seed(7)
    s0 = torch.cuda.get_rng_state(0)
    # s1: a different state (seed 42 on device 0, or just a different tensor)
    torch.cuda.manual_seed(42)
    s1 = torch.cuda.get_rng_state(0)
    assert not torch.equal(s0, s1), 's0 and s1 should differ'
    saved = [s0, s1]

    def real_seed_fn(i, seed):
        torch.cuda.default_generators[i].manual_seed(seed)
        return torch.cuda.get_rng_state(i)

    # Scramble device 0's state first
    torch.cuda.manual_seed(999)
    targets, rec = map_cuda_rng(saved, n_devices, real_seed_fn)
    # Apply the targets
    for i, t in enumerate(targets):
        torch.cuda.set_rng_state(t, i)
    # Check device 0's state is s0
    live0 = torch.cuda.get_rng_state(0)
    assert torch.equal(live0, s0), 'device 0 state should be restored to s0'
    print('  (g) PASS: real device 0 state restored (n_devices=%d)' % n_devices)


def main():
    print('test_cuda_rng_map:')
    test_a_2saved_1device()
    test_b_1saved_2devices()
    test_c_2saved_2devices_distinct()
    test_d_2saved_2devices_identical_raises()
    test_e_0saved_1device()
    test_f_json_serialisable()
    test_g_real_device()
    print('ALL CUDA RNG MAP TESTS PASSED')


if __name__ == '__main__':
    main()
