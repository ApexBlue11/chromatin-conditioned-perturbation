# -*- coding: utf-8 -*-
"""Review 038 ask 2: score_p7.py refuses P7 outputs whose snapshot identities fail (PI-written). Calls the code under test:
check_snapshot_identities() directly, and main() through --dry_check on a synthetic P7 directory with a real manifest."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_p7  # noqa: E402


def _save(path, row_index, y, deg):
    np.savez_compressed(path, row_index=row_index, y_pred=y.astype(np.float32), deg_pred=deg.astype(np.float32),
                        y_true=y.astype(np.float32), ctl_true=np.zeros_like(y, dtype=np.float32))


def make_p7(tmp, stack='P2 + C6 + V2', n_rows=7, n_genes=5):
    rng = np.random.default_rng(0)
    ri = np.arange(n_rows, dtype=np.int64)
    for k in range(3):
        snaps = [(rng.normal(size=(n_rows, n_genes)).astype(np.float32), rng.normal(size=(n_rows, n_genes)).astype(np.float32))
                 for _ in range(3)]
        for j, (y, d) in enumerate(snaps):
            _save(os.path.join(tmp, 'v9p7_seed%d_snap%d.npz' % (k, j)), ri, y, d)
        _save(os.path.join(tmp, 'v9p7_seed%d_last.npz' % k), ri, snaps[2][0], snaps[2][1])
        _save(os.path.join(tmp, 'v9p7_seed%d.npz' % k), ri, np.mean([s[0] for s in snaps], 0), np.mean([s[1] for s in snaps], 0))
    write_manifest(tmp, stack)
    return tmp


def write_manifest(tmp, stack, drop=()):
    files = {n: hashlib.sha1(open(os.path.join(tmp, n), 'rb').read()).hexdigest()
             for n in sorted(os.listdir(tmp)) if n.endswith('.npz') and n not in drop}
    json.dump({'complete': True, 'stack': stack, 'n_seeds': 3, 'files': files}, open(os.path.join(tmp, 'P7_COMPLETE.json'), 'w'))


SEEDS = ['v9p7_seed0.npz', 'v9p7_seed1.npz', 'v9p7_seed2.npz']


def test_identities_hold_on_consistent_files(tmp_path):
    d = make_p7(str(tmp_path))
    assert score_p7.check_snapshot_identities(d, SEEDS) <= 1e-5


def test_last_differing_from_snap2_is_refused(tmp_path):
    d = make_p7(str(tmp_path))
    z = dict(np.load(os.path.join(d, 'v9p7_seed1_last.npz')))
    z['y_pred'][0, 0] += 1e-5          # any difference at all: _last must equal _snap2 exactly
    np.savez_compressed(os.path.join(d, 'v9p7_seed1_last.npz'), **z)
    with pytest.raises(SystemExit, match='_last y_pred differs'):
        score_p7.check_snapshot_identities(d, SEEDS)


def test_main_not_the_snapshot_mean_is_refused(tmp_path):
    d = make_p7(str(tmp_path))
    z = dict(np.load(os.path.join(d, 'v9p7_seed2.npz')))
    z['deg_pred'][3, 1] += 1e-3
    np.savez_compressed(os.path.join(d, 'v9p7_seed2.npz'), **z)
    with pytest.raises(SystemExit, match='not the mean of its snapshots'):
        score_p7.check_snapshot_identities(d, SEEDS)


def test_row_index_mismatch_is_refused(tmp_path):
    d = make_p7(str(tmp_path))
    z = dict(np.load(os.path.join(d, 'v9p7_seed0_snap1.npz')))
    z['row_index'] = z['row_index'][::-1].copy()
    np.savez_compressed(os.path.join(d, 'v9p7_seed0_snap1.npz'), **z)
    with pytest.raises(SystemExit, match='different row_index'):
        score_p7.check_snapshot_identities(d, SEEDS)


def _dry(d):
    return subprocess.run([sys.executable, os.path.join(HERE, 'score_p7.py'), '--p7_dir', d, '--dry_check'],
                          capture_output=True, text=True)


@pytest.mark.skipif(not os.path.exists(score_p7.O2), reason='needs the local O2 profile')
def test_dry_check_runs_the_guard_and_passes(tmp_path):
    r = _dry(make_p7(str(tmp_path)))
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'snapshot identities hold' in r.stdout and 'P7 inputs verified' in r.stdout


@pytest.mark.skipif(not os.path.exists(score_p7.O2), reason='needs the local O2 profile')
def test_dry_check_refuses_a_tampered_last_file(tmp_path):
    d = make_p7(str(tmp_path))
    z = dict(np.load(os.path.join(d, 'v9p7_seed0_last.npz')))
    z['deg_pred'][0, 0] += 1.0
    np.savez_compressed(os.path.join(d, 'v9p7_seed0_last.npz'), **z)
    write_manifest(d, 'P2 + C6 + V2')   # the manifest matches the tampered bytes: only the identity check can catch it
    r = _dry(d)
    assert r.returncode != 0 and '_last deg_pred differs' in (r.stdout + r.stderr)


@pytest.mark.skipif(not os.path.exists(score_p7.O2), reason='needs the local O2 profile')
def test_dry_check_refuses_missing_snapshots_in_the_manifest(tmp_path):
    d = make_p7(str(tmp_path))
    write_manifest(d, 'P2 + C6 + V2', drop=('v9p7_seed2_snap0.npz',))
    r = _dry(d)
    assert r.returncode != 0 and '8 snapshot files' in (r.stdout + r.stderr)


@pytest.mark.skipif(not os.path.exists(score_p7.O2), reason='needs the local O2 profile')
def test_dry_check_without_v2_skips_the_snapshot_guard(tmp_path):
    d = make_p7(str(tmp_path), stack='P2 + C6')
    r = _dry(d)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'snapshot identities' not in r.stdout


def test_nan_prediction_is_named_as_the_cause(tmp_path):
    """Review 039: a NaN must be refused with its own message, not blamed on _last or the snapshot mean."""
    d = make_p7(str(tmp_path))
    z = dict(np.load(os.path.join(d, 'v9p7_seed1_snap0.npz')))
    z['y_pred'][2, 3] = np.nan
    np.savez_compressed(os.path.join(d, 'v9p7_seed1_snap0.npz'), **z)
    with pytest.raises(SystemExit, match='_snap0 y_pred has non-finite values'):
        score_p7.check_snapshot_identities(d, SEEDS)
