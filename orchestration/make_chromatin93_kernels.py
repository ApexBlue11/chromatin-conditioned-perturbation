# -*- coding: utf-8 -*-
"""RESULTS 93 Stage 1a: the two free Kaggle CPU kernels (H2 learning curve, H3 gradient boosting), each run once.

    python orchestration/make_chromatin93_kernels.py

Same pattern as make_chromatin91_kernel.py: every input's sha1 is pinned (the staged code dataset
`apexblue/lincs-chromatin-funnel`, new version with chromatin_h2.py + chromatin_gbm.py, and XPert's split bundle); the kernel
writes a completion marker with its output's sha1, which read_chromatin93.py (local) verifies. CPU only, no internet.
"""
import hashlib
import io
import json
import os

ROOT = r'C:\Projects\LINCS'
SRC = os.path.join(ROOT, 'external', 'kaggle_chromatin_src')
SPLITS_SHA1 = '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464'
FILES = ('chromatin_funnel.py', 'chromatin_power.py', 'score_dev.py', 'chromatin_h2.py', 'chromatin_gbm.py', 'E_final.npy',
         'E_final_mask.npy', 'lincs_cell_index.json', 'cell_lineage.npy', 'E_final_provenance.json')
REPO = {f: r'model\v9' for f in ('chromatin_funnel.py', 'chromatin_power.py', 'score_dev.py', 'chromatin_h2.py', 'chromatin_gbm.py')}
JOBS = {'h2': ('chromatin_h2.py', 'chromatin_h2_93.json', 'CHROMATIN93_H2_COMPLETE.json', []),
        'h3': ('chromatin_gbm.py', 'chromatin_gbm_93.json', 'CHROMATIN93_H3_COMPLETE.json', ['--draws', '3'])}

CODE = r'''# RESULTS 93 Stage 1a (pre-registered, amended by review 050): __JOB__, run once. CPU only.
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
import lightgbm
print('inputs verified: %d pinned files + the split bundle; lightgbm %s' % (len(PINS), lightgbm.__version__), flush=True)
DATA = '/tmp/c93_data'
os.makedirs(DATA, exist_ok=True)
for f in ('E_final.npy', 'E_final_mask.npy', 'lincs_cell_index.json', 'cell_lineage.npy'):
    shutil.copyfile(os.path.join(SRC, f), os.path.join(DATA, f))
os.symlink(spl[0], os.path.join(DATA, 'xpert_mdmt_splits.npz'))
OUT = '/kaggle/working/__OUT__'
env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
t = time.time()
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, '__MODULE__'), '--data_dir', DATA,
                    '--provenance', os.path.join(SRC, 'E_final_provenance.json'), '--out', OUT] + __EXTRA__, env=env)
print('__MODULE__ exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: __MODULE__ failed')
json.dump({'pins': PINS, 'splits_sha1': SPLITS_SHA1, 'lightgbm': lightgbm.__version__,
           'outputs': {'__OUT__': sha(OUT)}, 'complete': True}, open('/kaggle/working/__MARKER__', 'w'))
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
    for job, (module, out, marker, extra) in JOBS.items():
        kdir = os.path.join(ROOT, 'external', 'kaggle_kernels', 'kern_chromatin93_%s' % job)
        os.makedirs(kdir, exist_ok=True)
        slug = 'lincs-chromatin93-%s' % job
        code = (CODE.replace('__PINS__', repr(pins)).replace('__SPLITS__', SPLITS_SHA1).replace('__JOB__', job)
                .replace('__MODULE__', module).replace('__OUT__', out).replace('__MARKER__', marker).replace('__EXTRA__', repr(extra)))
        io.open(os.path.join(kdir, slug + '.py'), 'w', encoding='utf-8', newline='\n').write(code)
        meta = {'id': 'apexblue/' + slug, 'title': 'lincs chromatin93 %s' % job, 'code_file': slug + '.py', 'language': 'python',
                'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
                'keywords': [], 'dataset_sources': ['apexblue/lincs-chromatin-funnel', 'apexblue/xpert-mdmt-benchmark'],
                'kernel_sources': [], 'competition_sources': [], 'model_sources': [],
                'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'}
        io.open(os.path.join(kdir, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n').write(json.dumps(meta, indent=1) + '\n')
        print('wrote', kdir)
    print('pins', pins)


if __name__ == '__main__':
    main()
