# RESULTS 88 / 88.6 (pre-registered): the C8b post-pathway MoA probe, untrained init 1 (cfg_from the fold-0 seed-0 checkpoint). CPU only.
import glob, hashlib, os, subprocess, sys
hit = glob.glob('/kaggle/input/**/probe_moa_88.py', recursive=True)
if len(hit) != 1:
    raise SystemExit('FATAL: expected one probe_moa_88.py, found %d' % len(hit))
SRC = os.path.dirname(hit[0])
src = open(hit[0], encoding='utf-8').read()
for k in ('def score_within_strata', 'def check_arch_match', 'def post_delta', 'k_raw', 'cfg_from_sha1', 'ckpt_sha1',
          "3e59a7ba7832775596f032dbab06c2c36a7723b4"):
    if k not in src:
        raise SystemExit('FATAL: mounted probe_moa_88.py lacks %r (not the review-028 version)' % k)
CK = 'c8b_ckpt_v9_fold0_seed0.pt'
p = glob.glob('/kaggle/input/**/' + CK, recursive=True)
if len(p) != 1:
    raise SystemExit('FATAL: %s not found once under /kaggle/input: %r' % (CK, p))
h = hashlib.sha1(open(p[0], 'rb').read()).hexdigest()
print('checkpoint', p[0], h, flush=True)
if h != 'efd0e1cf2774ab0aa5ba02d1a75f0c999a45c38a':
    raise SystemExit('FATAL: checkpoint sha1 mismatch')
args = ['--untrained', '--seed', '1', '--cfg_from', p[0], '--n_perm', '1000', '--n_size', '200']
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'probe_moa_88.py')] + args, cwd=SRC)
print('probe exit', r.returncode, flush=True)
if r.returncode != 0:
    raise SystemExit(r.returncode)
print(sorted(glob.glob('/kaggle/working/probe_moa_88_*.json')))
