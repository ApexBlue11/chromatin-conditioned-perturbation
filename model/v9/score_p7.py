# -*- coding: utf-8 -*-
"""RESULTS 85.12 item 5 (review 035 C1): the ONE scoring of P7's test predictions (PI-written).

    python model/v9/score_p7.py --p7_dir external/kaggle_out/v9p7

Refuses unless P7_COMPLETE.json exists and every listed file's sha1 matches; asserts O2's profile is §87's (sha1 69484323…);
refuses if the P7 result file already exists (touched once). Then runs coldcell_h2h.py once on the three seed files (plus the
three V2-last files as the labelled alt row when the stack contains V2, §90.6 amended).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
O2 = os.path.join(REPO, 'external', 'kaggle_out', 'cc1_v9', 'xpert_trained_split_cold_cell_1_final_test_profile.npy')
O2_RECORD = os.path.join(REPO, 'external', 'kaggle_out', 'cc1_v9', 'run_record.json')
H5AD = os.path.join(REPO, 'external', 'xpert', 'code', 'XPert', 'processed_data', 'l1000_mdmt_68830_subset.h5ad')
O2_LABEL = 'XPert trained to its published recipe on split_cold_cell_1 (O2)'
OUT = os.path.join(REPO, 'model', 'results', 'coldcell_h2h_split_cold_cell_1_P7.json')


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


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
        cmd += ['--ours_alt'] + [os.path.join(a.p7_dir, n) for n in last] + ['--ours_alt_label', 'V2-last']
    print('scoring once:', ' '.join(os.path.basename(c) if os.path.sep in c else c for c in cmd[1:]), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit('coldcell_h2h.py exited %d' % r.returncode)


if __name__ == '__main__':
    main()
