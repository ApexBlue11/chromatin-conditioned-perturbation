# -*- coding: utf-8 -*-
"""RESULTS 93 H1 (93.3, 93.10): one free Kaggle CPU kernel, run once, after review. PI glue, the pattern of
make_chromatin93_kernels.py.

    python orchestration/make_chromatin93_h1_kernel.py

Every input's sha1 is pinned: the staged code dataset `apexblue/lincs-chromatin-funnel` (a version adding chromatin_h1.py and
the frozen c93_features.npz, RESULTS 93.11) and XPert's split bundle. The kernel writes a completion marker with its output's
sha1, which read_h1.py (local) verifies. CPU only, no internet.
"""
import hashlib
import io
import json
import os

ROOT = r'C:\Projects\LINCS'
SRC = os.path.join(ROOT, 'external', 'kaggle_chromatin_src')
SPLITS_SHA1 = '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464'
C93_SHA1 = '23e12b9dca69d26f628a197392e0890d30f51bc6'            # RESULTS 93.11, frozen
FILES = ('chromatin_funnel.py', 'chromatin_power.py', 'score_dev.py', 'chromatin_h1.py', 'E_final.npy', 'E_final_mask.npy',
         'lincs_cell_index.json', 'cell_lineage.npy', 'E_final_provenance.json', 'c93_features.npz')
REPO = {f: os.path.join('model', 'v9') for f in ('chromatin_funnel.py', 'chromatin_power.py', 'score_dev.py', 'chromatin_h1.py')}
REPO['c93_features.npz'] = os.path.join('model', 'results', 'c93')
SLUG = 'lincs-chromatin93-h1'
OUT = 'chromatin_h1_93.json'
MARKER = 'CHROMATIN93_H1_COMPLETE.json'

CODE = r'''# RESULTS 93 H1 (pre-registered 93.3, 93.10; amended 93.5, 93.8, 93.9): run once. CPU only.
import glob, hashlib, json, os, shutil, subprocess, sys, time
PINS = __PINS__
SPLITS_SHA1 = '__SPLITS__'
C93_SHA1 = '__C93__'
hit = glob.glob('/kaggle/input/**/chromatin_h1.py', recursive=True)
spl = glob.glob('/kaggle/input/**/xpert_mdmt_splits.npz', recursive=True)
if not hit or not spl:
    raise SystemExit('FATAL: inputs not mounted: %r %r' % (hit, spl))
SRC = os.path.dirname(hit[0])
sha = lambda p: hashlib.sha1(open(p, 'rb').read()).hexdigest()
for f, h in PINS.items():
    got = sha(os.path.join(SRC, f))
    if got != h:
        raise SystemExit('FATAL: %s sha1 %s != pinned %s' % (f, got, h))
if sha(spl[0]) != SPLITS_SHA1:
    raise SystemExit('FATAL: xpert_mdmt_splits.npz is not the reviewed bundle')
print('inputs verified: %d pinned files + the split bundle' % len(PINS), flush=True)
DATA = '/tmp/c93_h1_data'
os.makedirs(DATA, exist_ok=True)
for f in ('E_final.npy', 'E_final_mask.npy', 'lincs_cell_index.json', 'cell_lineage.npy'):
    shutil.copyfile(os.path.join(SRC, f), os.path.join(DATA, f))
os.symlink(spl[0], os.path.join(DATA, 'xpert_mdmt_splits.npz'))
OUT = '/kaggle/working/__OUT__'
env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
t = time.time()
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'chromatin_h1.py'), '--data_dir', DATA,
                    '--provenance', os.path.join(SRC, 'E_final_provenance.json'), '--c93', os.path.join(SRC, 'c93_features.npz'),
                    '--c93_sha1', C93_SHA1, '--draws', '3', '--out', OUT], env=env)
print('chromatin_h1.py exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: chromatin_h1.py failed')
json.dump({'pins': PINS, 'splits_sha1': SPLITS_SHA1, 'outputs': {'__OUT__': sha(OUT)}, 'complete': True},
          open('/kaggle/working/__MARKER__', 'w'))
print('done', flush=True)
'''


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def main():
    pins = {}
    for f in FILES:
        p = os.path.join(SRC, f)
        if f in REPO:
            a, b = open(p, 'rb').read(), open(os.path.join(ROOT, REPO[f], f), 'rb').read()
            if not f.endswith('.npz'):
                a, b = a.replace(b'\r\n', b'\n'), b.replace(b'\r\n', b'\n')
            assert a == b, 'staged %s differs from the repo' % f
        pins[f] = sha1(p)
    assert pins['c93_features.npz'] == C93_SHA1, 'staged c93_features.npz is not the frozen one'
    kdir = os.path.join(ROOT, 'external', 'kaggle_kernels', 'kern_chromatin93_h1')
    os.makedirs(kdir, exist_ok=True)
    code = (CODE.replace('__PINS__', repr(pins)).replace('__SPLITS__', SPLITS_SHA1).replace('__C93__', C93_SHA1)
            .replace('__OUT__', OUT).replace('__MARKER__', MARKER))
    io.open(os.path.join(kdir, SLUG + '.py'), 'w', encoding='utf-8', newline='\n').write(code)
    meta = {'id': 'apexblue/' + SLUG, 'title': 'lincs chromatin93 h1', 'code_file': SLUG + '.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
            'keywords': [], 'dataset_sources': ['apexblue/lincs-chromatin-funnel', 'apexblue/xpert-mdmt-benchmark'],
            'kernel_sources': [], 'competition_sources': [], 'model_sources': [],
            'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'}
    io.open(os.path.join(kdir, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n').write(json.dumps(meta, indent=1) + '\n')
    print('wrote', kdir)
    print('pins', pins)


if __name__ == '__main__':
    main()
