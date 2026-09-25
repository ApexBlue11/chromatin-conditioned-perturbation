# RESULTS 86 (pre-registered): the drug-specific pathway mechanism probe, one checkpoint per kernel. CPU only.
import glob, hashlib, os, subprocess, sys
hit = glob.glob('/kaggle/input/**/probe_moa_v9.py', recursive=True)
if len(hit) != 1:
    raise SystemExit('FATAL: expected one probe_moa_v9.py, found %d' % len(hit))
SRC = os.path.dirname(hit[0])
src = open(hit[0], encoding='utf-8').read()
for k in ('def score(', 'def project_diff', 'landmark_symbols_v9', "/kaggle/working/probe_moa_v9_"):
    if k not in src:
        raise SystemExit('FATAL: mounted probe lacks %r (not the W16b + fallback version)' % k)
args = ['--seed', '2', '--n_perm', '1000', '--n_size', '200']
CK = 'r2_ckpt_v9_fold0_seed2.pt'
if CK:
    p = glob.glob('/kaggle/input/**/' + CK, recursive=True)[0]
    h = hashlib.sha1(open(p, 'rb').read()).hexdigest()
    print('checkpoint', p, h, flush=True)
    if h != 'c84f3c9ba969b9dd295f8c7eccf7cf485ee0d19c':
        raise SystemExit('FATAL: checkpoint sha1 mismatch')
    args = ['--ckpt', p] + args
else:
    args = ['--untrained'] + args
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'probe_moa_v9.py')] + args, cwd=SRC)
print('probe exit', r.returncode, flush=True)
if r.returncode != 0:
    raise SystemExit(r.returncode)
print(sorted(glob.glob('/kaggle/working/probe_moa_v9_*.json')))
