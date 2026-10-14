# -*- coding: utf-8 -*-
"""RESULTS 99 / packets 084-085: build Phase 1's inference kernels (C2b, C3) as FREE Kaggle CPU kernels.

Six kernels, one per (run, seed) so each fits Kaggle's 12 h CPU limit (review 084 ask 6):
    kern_audit99_p9_s{k}:  audit99_infer.py --mode swap  on v9p9_seed{k}.pt  (C3 + the seen-cell x_cell bound)
    kern_audit99_p7_s{k}:  audit99_infer.py --mode xcell on v9p7_seed{k}.pt  (C2b)
Each pins the sha1 of every file it uses -- the nine sources, its checkpoint and identity file, AND (review 084 C3) the
14 data files XPertData / the model read through find() -- refuses any mismatch or any name with more than one mounted
copy across the three datasets, and writes AUDIT99_COMPLETE_<tag>_s{k}.json with every output's sha1 and the pins.
The data pins come from the outcome-free probe kernel's audit99_data_pins.json (--data_pins).

The private dataset apexblue/lincs-audit99 is staged flat in external/kaggle_audit99/data.

Run:  python orchestration/make_audit99_kernels.py --data_pins <probe json> [--stage]
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
DATA_NAMES = ['xpert_mdmt_splits.npz', 'drug_feature_index.json', 'drug_descriptors.npy', 'drug_fingerprints.npy',
              'drug_unimol.npy', 'drug_atom_reprs.npy', 'drug_atom_offsets.npy', 'lincs_cell_index.json', 'E_final.npy',
              'E_final_mask.npy', 'cell_lineage.npy', 'M_pathway_v9.npy', 'STRING_adj_978_v9.npy', 'gene_vectors_978.npy']
KERNELS = {'p9': ('swap', 'v9p9'), 'p7': ('xcell', 'v9p7')}
N_IDENT = 2000

TEMPLATE = r'''# lincs-audit99-__TAG__-s__SEED__ : RESULTS 99 Phase 1, __WHAT__, seed __SEED__. Kaggle CPU, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys
SEED, TAG = __SEED__, '__TAG__'
MODE, RUN = '__MODE__', '__RUN__'
PINS = __PINS__
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
print('all %d pinned files verified, each the only mounted copy' % len(PINS), flush=True)
cmd = [sys.executable, '-u', os.path.join(SRC, 'audit99_infer.py'), '--mode', MODE,
       '--ckpt', paths['%s_seed%d.pt' % (RUN, SEED)], '--identity', paths['%s_seed%d_ident.npz' % (RUN, SEED)],
       '--out', OUT, '--roots', '/kaggle/input']
r = subprocess.run(cmd, cwd=SRC, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONUNBUFFERED='1'))
if r.returncode:
    raise SystemExit('FATAL: %s exited %d' % (MODE, r.returncode))
files = {f: sha1(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))}
want = {'%s_seed%d.%s' % (MODE, SEED, e) for e in ('npz', 'json')}
if set(files) != want:
    raise SystemExit('FATAL: outputs %s != expected %s' % (sorted(files), sorted(want)))
with open(os.path.join(OUT, 'AUDIT99_COMPLETE_%s_s%d.json' % (TAG, SEED)), 'w') as f:
    json.dump({'complete': True, 'seed': SEED, 'tag': TAG, 'mode': MODE, 'files': files, 'pins': PINS}, f, indent=1)
shutil.rmtree(SRC)
print('AUDIT99 complete, %s seed %d: %d files' % (TAG, SEED, len(files)), flush=True)
'''
WHAT = {'p9': 'C3 drug swap + the seen-cell x_cell bound on P9', 'p7': 'C2b x_cell on P7'}


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
    for run in ('v9p7', 'v9p9'):                     # the order the staged identity files were first drawn in
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


def data_pins(probe_json):
    probe = json.load(open(probe_json, encoding='utf-8'))
    pins = {}
    for n in DATA_NAMES:
        if n not in probe or probe[n]['copies'] != 1:
            raise SystemExit('FATAL: probe shows %s with %s mounted copies (need exactly 1)'
                             % (n, probe.get(n, {}).get('copies')))
        pins[n] = probe[n]['sha1'][0]
    return pins


def build(tag, seed, dpins):
    mode, run = KERNELS[tag]
    pins = {n: sha1(os.path.join(DS_DIR, n)) for n in SRC_FILES}
    for n in ('%s_seed%d.pt' % (run, seed), '%s_seed%d_ident.npz' % (run, seed)):
        pins[n] = sha1(os.path.join(DS_DIR, n))
    if pins['%s_seed%d.pt' % (run, seed)] != sha1(os.path.join(REPO, 'external', 'kaggle_out', run, '%s_seed%d.pt' % (run, seed))):
        raise SystemExit('FATAL: staged checkpoint differs from the run output')
    pins.update(dpins)
    code = (TEMPLATE.replace('__SEED__', str(seed)).replace('__TAG__', tag).replace('__MODE__', mode)
            .replace('__RUN__', run).replace('__WHAT__', WHAT[tag]).replace('__PINS__', json.dumps(pins, indent=1, sort_keys=True)))
    d = os.path.join(KDIR, 'kern_audit99_%s_s%d' % (tag, seed))
    os.makedirs(d, exist_ok=True)
    name = 'lincs-audit99-%s-s%d' % (tag, seed)
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
    ap.add_argument('--data_pins', required=True, help="the probe kernel's audit99_data_pins.json")
    ap.add_argument('--stage', action='store_true')
    a = ap.parse_args(argv)
    if a.stage:
        stage()
    check_sources_current()
    dp = data_pins(a.data_pins)
    for tag in KERNELS:
        for k in (0, 1, 2):
            d, pins = build(tag, k, dp)
            print(d, len(pins), 'pins')


if __name__ == '__main__':
    sys.exit(main())
