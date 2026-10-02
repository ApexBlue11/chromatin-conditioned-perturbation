# -*- coding: utf-8 -*-
"""RESULTS 91: the free Kaggle CPU kernel that runs the chromatin funnel and its planted-effect power calibration once.

    python orchestration/make_chromatin91_kernel.py

Pins the sha1 of every input (the staged code dataset `apexblue/lincs-chromatin-funnel` and XPert's split bundle) so the run
provably reads the bytes reviewed locally. CPU only: no GPU quota is spent.
"""
import hashlib
import io
import json
import os

ROOT = r'C:\Projects\LINCS'
SRC = os.path.join(ROOT, 'external', 'kaggle_chromatin_src')
KDIR = os.path.join(ROOT, 'external', 'kaggle_kernels', 'kern_chromatin91')
SPLITS_SHA1 = '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464'      # == external/lightning/v9payload/data/xpert_mdmt_splits.npz
FILES = ('chromatin_funnel.py', 'chromatin_power.py', 'score_dev.py', 'E_final.npy', 'E_final_mask.npy', 'lincs_cell_index.json',
         'cell_lineage.npy', 'E_final_provenance.json', 'chembl_dti_edges.tsv')   # read_chromatin91.py runs locally, not here
REPO = {'chromatin_funnel.py': r'model\v9', 'chromatin_power.py': r'model\v9', 'score_dev.py': r'model\v9'}

CODE = r'''# RESULTS 91 (pre-registered; 91.8, 91.9): the chromatin funnel and its planted-effect power calibration, run once. CPU only.
import glob, hashlib, json, os, shutil, subprocess, sys, time
PINS = __PINS__
SPLITS_SHA1 = '__SPLITS__'
hit = glob.glob('/kaggle/input/**/chromatin_funnel.py', recursive=True)
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
DATA = '/tmp/funnel_data'
os.makedirs(DATA, exist_ok=True)
for f in ('E_final.npy', 'E_final_mask.npy', 'lincs_cell_index.json', 'cell_lineage.npy'):
    shutil.copyfile(os.path.join(SRC, f), os.path.join(DATA, f))
os.symlink(spl[0], os.path.join(DATA, 'xpert_mdmt_splits.npz'))
common = ['--data_dir', DATA, '--provenance', os.path.join(SRC, 'E_final_provenance.json')]
env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
t = time.time()
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'chromatin_funnel.py')] + common +
                   ['--dti', os.path.join(SRC, 'chembl_dti_edges.tsv'), '--out', '/kaggle/working/chromatin_funnel_91.json'], env=env)
print('funnel exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: the funnel failed')
env1 = dict(env, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
t = time.time()
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'chromatin_power.py')] + common +
                   ['--out', '/kaggle/working/chromatin_power_91.json', '--draws', '5', '--workers', '4'], env=env1,
                   stdout=subprocess.DEVNULL)
print('power exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: the power calibration failed')
outs = {f: sha(os.path.join('/kaggle/working', f)) for f in ('chromatin_funnel_91.json', 'chromatin_power_91.json')}
json.dump({'pins': PINS, 'splits_sha1': SPLITS_SHA1, 'outputs': outs, 'complete': True},   # review 041 C5: the reader verifies these
          open('/kaggle/working/CHROMATIN91_COMPLETE.json', 'w'))
print('done', flush=True)
'''


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def main():
    pins = {}
    for f in FILES:
        p = os.path.join(SRC, f)
        if f in REPO:
            a = open(p, 'rb').read().replace(b'\r\n', b'\n')
            b = open(os.path.join(ROOT, REPO[f], f), 'rb').read().replace(b'\r\n', b'\n')
            assert a == b, 'staged %s differs from the repo' % f
        pins[f] = sha1(p)
    os.makedirs(KDIR, exist_ok=True)
    slug = 'lincs-chromatin91'
    io.open(os.path.join(KDIR, slug + '.py'), 'w', encoding='utf-8', newline='\n').write(
        CODE.replace('__PINS__', repr(pins)).replace('__SPLITS__', SPLITS_SHA1))
    meta = {'id': 'apexblue/' + slug, 'title': 'lincs chromatin91', 'code_file': slug + '.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
            'keywords': [], 'dataset_sources': ['apexblue/lincs-chromatin-funnel', 'apexblue/xpert-mdmt-benchmark'],
            'kernel_sources': [], 'competition_sources': [], 'model_sources': [],
            'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'}
    io.open(os.path.join(KDIR, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n').write(json.dumps(meta, indent=1) + '\n')
    print('wrote', KDIR, 'pins', pins)


if __name__ == '__main__':
    main()
