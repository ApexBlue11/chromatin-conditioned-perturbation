# -*- coding: utf-8 -*-
"""RESULTS 99 / packet 084: build Phase 1's inference kernels (C2b, C3) as FREE Kaggle CPU kernels, one per seed.

Stages the private dataset apexblue/lincs-audit99 in external/kaggle_audit99/data (flat): the nine source files the
inference path imports, P7's and P9's final-snapshot checkpoints, and the slim identity files (2,000 rows each of the
saved final-snapshot deg_pred, drawn with seed 0). Writes external/kaggle_kernels/kern_audit99_s{0,1,2}/, each pinning
the sha1 of every file it uses and refusing on any mismatch. Each kernel runs, for its seed k:
    audit99_infer.py --mode swap  on v9p9_seed{k}.pt   (C3)
    audit99_infer.py --mode xcell on v9p7_seed{k}.pt   (C2b)
and writes AUDIT99_COMPLETE_s{k}.json with the sha1 of every output. Nothing is scored in the kernel.

Run:  python orchestration/make_audit99_kernels.py [--stage]   (--stage re-copies the sources into the dataset dir)
"""
import argparse, hashlib, json, os, shutil, sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS_DIR = os.path.join(REPO, 'external', 'kaggle_audit99', 'data')
KDIR = os.path.join(REPO, 'external', 'kaggle_kernels')
DATASET = 'apexblue/lincs-audit99'
MOUNTS = ['apexblue/lincs-v9-bundle', 'apexblue/xpert-mdmt-benchmark', DATASET]
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
SRC_FILES = {'audit99_infer.py': 'model/v9/audit99_infer.py', 'xpert_arm.py': 'model/v9/xpert_arm.py',
             'model_v9.py': 'model/v9/model_v9.py', 'modules_v9.py': 'model/v9/modules_v9.py',
             'config_v9.py': 'model/v9/config_v9.py', 'dp_seeding.py': 'model/v9/dp_seeding.py',
             'modules.py': 'model/modules.py', 'config_v6.py': 'model/v6/config_v6.py',
             'modules_v7.py': 'model/v7/modules_v7.py'}
RUNS = (('swap', 'v9p9'), ('xcell', 'v9p7'))
N_IDENT = 2000

TEMPLATE = r'''# lincs-audit99-s__SEED__ : RESULTS 99 Phase 1, C3 (drug swap, P9) and C2b (x_cell, P7), seed __SEED__. Kaggle CPU, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys
SEED = __SEED__
PINS = __PINS__
RUNS = __RUNS__
SRC, OUT = '/kaggle/working/src', '/kaggle/working/out'


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find_one(name):
    hits = glob.glob('/kaggle/input/**/' + name, recursive=True)
    if len(hits) != 1:
        raise SystemExit('FATAL: %s: expected one mounted copy, found %d' % (name, len(hits)))
    return hits[0]


os.makedirs(SRC, exist_ok=True)
paths = {}
for name, h in PINS.items():
    p = find_one(name)
    if sha1(p) != h:
        raise SystemExit('FATAL: %s sha1 %s != pinned %s' % (name, sha1(p), h))
    paths[name] = p
    if name.endswith('.py'):
        shutil.copy(p, os.path.join(SRC, name))
print('all %d pinned files verified' % len(PINS), flush=True)
for mode, run in RUNS:
    cmd = [sys.executable, '-u', os.path.join(SRC, 'audit99_infer.py'), '--mode', mode,
           '--ckpt', paths['%s_seed%d.pt' % (run, SEED)], '--identity', paths['%s_seed%d_ident.npz' % (run, SEED)],
           '--out', OUT, '--roots', '/kaggle/input']
    r = subprocess.run(cmd, cwd=SRC, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONUNBUFFERED='1'))
    if r.returncode:
        raise SystemExit('FATAL: %s exited %d' % (mode, r.returncode))
files = {f: sha1(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))}
want = {'%s_seed%d.%s' % (m, SEED, e) for m, _ in RUNS for e in ('npz', 'json')}
if set(files) != want:
    raise SystemExit('FATAL: outputs %s != expected %s' % (sorted(files), sorted(want)))
with open(os.path.join(OUT, 'AUDIT99_COMPLETE_s%d.json' % SEED), 'w') as f:
    json.dump({'complete': True, 'seed': SEED, 'files': files, 'pins': PINS}, f, indent=1)
shutil.rmtree(SRC)
print('AUDIT99 complete, seed %d: %d files' % (SEED, len(files)), flush=True)
'''


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stage():
    os.makedirs(DS_DIR, exist_ok=True)
    for dst, src in SRC_FILES.items():
        shutil.copy(os.path.join(REPO, src), os.path.join(DS_DIR, dst))
    rng = np.random.default_rng(0)
    for _, run in RUNS[::-1]:                       # v9p7 then v9p9: the order the staged files were first drawn in
        for k in (0, 1, 2):
            pt = os.path.join(DS_DIR, '%s_seed%d.pt' % (run, k))
            if not os.path.exists(pt):
                shutil.copy(os.path.join(REPO, 'external', 'kaggle_out', run, '%s_seed%d.pt' % (run, k)), pt)
            ident = os.path.join(DS_DIR, '%s_seed%d_ident.npz' % (run, k))
            z = np.load(os.path.join(REPO, 'external', 'kaggle_out', run, '%s_seed%d_last.npz' % (run, k)))
            sel = np.sort(rng.choice(len(z['row_index']), N_IDENT, replace=False))
            if not os.path.exists(ident):
                np.savez_compressed(ident, row_index=z['row_index'][sel], deg_pred=z['deg_pred'][sel])
    with open(os.path.join(DS_DIR, 'dataset-metadata.json'), 'w') as f:
        json.dump({'title': 'lincs audit99', 'id': DATASET, 'licenses': [{'name': 'other'}]}, f, indent=1)


def check_sources_current():
    for dst, src in SRC_FILES.items():
        if sha1(os.path.join(DS_DIR, dst)) != sha1(os.path.join(REPO, src)):
            raise SystemExit('FATAL: staged %s differs from %s; re-run with --stage' % (dst, src))


def build(seed):
    pins = {n: sha1(os.path.join(DS_DIR, n)) for n in SRC_FILES}
    for _, run in RUNS:
        for n in ('%s_seed%d.pt' % (run, seed), '%s_seed%d_ident.npz' % (run, seed)):
            pins[n] = sha1(os.path.join(DS_DIR, n))
        if pins['%s_seed%d.pt' % (run, seed)] != sha1(os.path.join(REPO, 'external', 'kaggle_out', run, '%s_seed%d.pt' % (run, seed))):
            raise SystemExit('FATAL: staged checkpoint differs from the run output')
    code = (TEMPLATE.replace('__SEED__', str(seed)).replace('__PINS__', json.dumps(pins, indent=1, sort_keys=True))
            .replace('__RUNS__', repr(RUNS)))
    d = os.path.join(KDIR, 'kern_audit99_s%d' % seed)
    os.makedirs(d, exist_ok=True)
    name = 'lincs-audit99-s%d' % seed
    with open(os.path.join(d, name + '.py'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(code)
    meta = {'id': 'apexblue/' + name, 'title': name.replace('-', ' '), 'code_file': name + '.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
            'keywords': [], 'dataset_sources': MOUNTS, 'kernel_sources': [], 'competition_sources': [], 'model_sources': [],
            'docker_image': DOCKER}
    with open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, indent=1)
    return d, pins


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', action='store_true')
    a = ap.parse_args(argv)
    if a.stage:
        stage()
    check_sources_current()
    for k in (0, 1, 2):
        d, pins = build(k)
        print(d, len(pins), 'pins')


if __name__ == '__main__':
    sys.exit(main())
