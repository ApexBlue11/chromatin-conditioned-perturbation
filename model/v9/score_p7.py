# -*- coding: utf-8 -*-
"""RESULTS 85.12 item 5 (review 035 C1): the ONE scoring of P7's test predictions (PI-written).

    python model/v9/score_p7.py --p7_dir external/kaggle_out/v9p7

Refuses unless P7_COMPLETE.json exists and every listed file's sha1 matches; asserts O2's profile is §87's (sha1 69484323…);
refuses if the P7 result file already exists (touched once). Then runs coldcell_h2h.py once on the three seed files (plus the
three final-snapshot `_last` files as the labelled alt row "P7-last" when the stack contains V2, §90.6 amended, review 038 C4).
With V2 stacked it first refuses unless, per seed, `_last` equals `_snap2` exactly and the main file equals the mean of
`_snap0..2` (review 038 ask 2) -- checked here, where a failure costs nothing, not in the kernel, where it would delete P7.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
O2 = os.path.join(REPO, 'external', 'kaggle_out', 'cc1_v9', 'xpert_trained_split_cold_cell_1_final_test_profile.npy')
O2_RECORD = os.path.join(REPO, 'external', 'kaggle_out', 'cc1_v9', 'run_record.json')
H5AD = os.path.join(REPO, 'external', 'xpert', 'code', 'XPert', 'processed_data', 'l1000_mdmt_68830_subset.h5ad')
O2_LABEL = 'XPert trained to its published recipe on split_cold_cell_1 (O2)'
OUT = os.path.join(REPO, 'model', 'results', 'coldcell_h2h_split_cold_cell_1_P7.json')
ALT_LABEL = 'P7-last (final snapshot, no snapshot averaging)'   # review 038 C4: the stack's final snapshot, not V2-only


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def check_snapshot_identities(p7_dir, seeds, n_snap=3, tol=1e-5):
    """Review 038 ask 2: for each main seed file v9p7_seed{k}.npz, its _last file must equal _snap{n-1} exactly and the main
    file must equal the mean of _snap0.._snap{n-1} within tol, on y_pred and deg_pred, over identical row_index. Raises
    SystemExit on any violation; returns the largest |main - mean(snaps)| seen."""
    worst = 0.0
    for name in seeds:
        stem = os.path.join(p7_dir, name[:-len('.npz')])
        with np.load(stem + '.npz') as m, np.load(stem + '_last.npz') as last:
            snaps = [np.load(stem + '_snap%d.npz' % k) for k in range(n_snap)]
            try:
                for z in snaps + [last]:
                    if not np.array_equal(z['row_index'], m['row_index']):
                        raise SystemExit('FATAL: %s: a snapshot file has a different row_index' % name)
                for z, tag in [(m, 'main'), (last, '_last')] + [(z, '_snap%d' % k) for k, z in enumerate(snaps)]:
                    for key in ('y_pred', 'deg_pred'):
                        if not np.isfinite(z[key]).all():   # review 039: a NaN gets its own message, not a false identity blame
                            raise SystemExit('FATAL: %s: %s %s has non-finite values' % (name, tag, key))
                for key in ('y_pred', 'deg_pred'):
                    if not np.array_equal(last[key], snaps[-1][key]):
                        raise SystemExit('FATAL: %s: _last %s differs from _snap%d' % (name, key, n_snap - 1))
                    d = float(np.abs(m[key] - np.mean([z[key] for z in snaps], axis=0)).max())
                    worst = max(worst, d)
                    if not d <= tol:
                        raise SystemExit('FATAL: %s: main %s is not the mean of its snapshots (max |d| %.2e)' % (name, key, d))
            finally:
                for z in snaps:
                    z.close()
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p7_dir', required=True)
    ap.add_argument('--dry_check', action='store_true', help='verify the inputs only; do not score')
    a = ap.parse_args()
    if os.path.exists(OUT):
        raise SystemExit('FATAL: %s exists -- P7 has been scored; the test cells are touched once (85.12)' % OUT)
    mf = os.path.join(a.p7_dir, 'P7_COMPLETE.json')
    if not os.path.exists(mf):
        raise SystemExit('FATAL: no P7_COMPLETE.json -- the kernel did not finish; nothing here may be scored (review 035 C1)')
    m = json.load(open(mf))
    if not m.get('complete') or m.get('n_seeds') != 3:
        raise SystemExit('FATAL: manifest incomplete: %r' % {k: m.get(k) for k in ('complete', 'n_seeds')})
    for name, h in m['files'].items():
        p = os.path.join(a.p7_dir, name)
        if not os.path.exists(p) or sha1(p) != h:
            raise SystemExit('FATAL: %s missing or sha1 differs from the manifest' % name)
    seeds = sorted(n for n in m['files'] if re.fullmatch(r'v9p7_seed\d\.npz', n))
    last = sorted(n for n in m['files'] if re.fullmatch(r'v9p7_seed\d_last\.npz', n))
    if len(seeds) != 3:
        raise SystemExit('FATAL: expected 3 seed files, manifest has %r' % seeds)
    v2 = 'V2' in m.get('stack', '')
    if v2 and len(last) != 3:
        raise SystemExit('FATAL: the stack contains V2 but the manifest has %d V2-last files' % len(last))
    if v2:
        snaps = [n for n in m['files'] if re.fullmatch(r'v9p7_seed\d_snap\d\.npz', n)]
        if len(snaps) != 9:
            raise SystemExit('FATAL: the stack contains V2 but the manifest has %d snapshot files (need 9)' % len(snaps))
        print('snapshot identities hold: _last == _snap2; main = mean(snaps), max |d| %.2e' % check_snapshot_identities(a.p7_dir, seeds))
    if not sha1(O2).startswith('69484323'):
        raise SystemExit('FATAL: the O2 profile is not the one section 87 scored')
    print('P7 inputs verified: %d manifest files, stack %s, O2 sha1 %s' % (len(m['files']), m.get('stack'), sha1(O2)[:12]))
    if a.dry_check:
        return
    cmd = [sys.executable, os.path.join(REPO, 'model', 'v9', 'coldcell_h2h.py'), '--theirs', O2,
           '--ours'] + [os.path.join(a.p7_dir, n) for n in seeds] + [
           '--h5ad', H5AD, '--split', 'split_cold_cell_1', '--run_record', O2_RECORD, '--theirs_label', O2_LABEL,
           '--n_boot', '20000', '--seed', '0', '--out', OUT]
    if v2:
        cmd += ['--ours_alt'] + [os.path.join(a.p7_dir, n) for n in last] + ['--ours_alt_label', ALT_LABEL]
    print('scoring once:', ' '.join(os.path.basename(c) if os.path.sep in c else c for c in cmd[1:]), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit('coldcell_h2h.py exited %d' % r.returncode)


if __name__ == '__main__':
    main()
