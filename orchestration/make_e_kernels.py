# -*- coding: utf-8 -*-
"""RESULTS 92: the two chromatin dev screens from the 91 funnel, on the P2 recipe (the 85 dev command).

    python orchestration/make_e_kernels.py e2 [--seed_start 0 --seeds 1]
    python orchestration/make_e_kernels.py e1 [--seed_start 0 --seeds 1]
    python orchestration/make_e_kernels.py e3 --seed_start 0 --seeds 3     # RESULTS 92.11: the tie-fixed encoding

E2 = + --ablate_epi: WITHOUT CELL-SPECIFIC chromatin (training-time mean ablation: every row carries the same per-gene
     dev-train mean, a gene-generic constant; review 044 C2b). Runs on the current lincs-v9-src upload or the new one.
E1 = + --chromatin_encoding clean (failed H3K27me3 missing, rank-normal per (cell, mark); needs the upload of xpert_arm.py
3495ada + E_final_provenance.json, made only after P7, 88 t1 and t2 have started). Guards as the V2 dev kernel, plus one per arm.
"""
import argparse
import io
import json
import os

ROOT = r'C:\Projects\LINCS\external\kaggle_kernels'
IMG = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
UPLOAD_NOW = '75c58f52051296b326e57613feee7be8f7097d1b'     # xpert_arm.py in the current lincs-v9-src (P7 pins it)
UPLOAD_E1 = '60bdcd48039a1ae3be4ebbfdce087c142f5d8f44'      # xpert_arm.py at 3495ada (the --chromatin_encoding flag)
UPLOAD_E3 = '50ff83da08c55f6b3ad78bf31dfc790e7cc762b4'      # xpert_arm.py with encode_chromatin + 'tie' (RESULTS 92.11)
ARMS = {'e2': {'flags': ['--ablate_epi'], 'tag': '_noepi', 'suffix': '', 'needs': ['ablate_epi'],
               'xpert_arm_sha1': [UPLOAD_NOW, UPLOAD_E1]},
        'e1': {'flags': ['--chromatin_encoding', 'clean'], 'tag': '', 'suffix': '_chromclean',
               'needs': ['chromatin_encoding', "E_final_provenance.json", 'rankdata'], 'xpert_arm_sha1': [UPLOAD_E1]},
        'e3': {'flags': ['--chromatin_encoding', 'tie'], 'tag': '', 'suffix': '_chromtie',
               'needs': ['def encode_chromatin', 'TIE_EXPECTED_MASKED', 'TIE_MIN_NONTIED', "E_final_provenance.json"],
               'xpert_arm_sha1': [UPLOAD_E3]}}

CODE = r'''# RESULTS 92 (pre-registered): __ARM__ on the P2 recipe, dev carve, seeds __SEEDS__. Guards as the V2 dev kernel, plus the arm's.
import sys, os, glob, torch, numpy as np
print('/kaggle/input contains:', sorted(os.listdir('/kaggle/input')), flush=True)
hit = glob.glob('/kaggle/input/**/xpert_arm.py', recursive=True)
if not hit:
    raise SystemExit('FATAL: xpert_arm.py not found under /kaggle/input.')
SRC = os.path.dirname(hit[0]); sys.path.insert(0, SRC)
if not torch.cuda.is_available():
    raise SystemExit('FATAL: no CUDA device. NOT falling back to CPU.')
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if any('P100' in n for n in names):
    raise SystemExit('FATAL: P100 assigned; Kaggle torch has no sm_60 kernels.')
print('GPUs:', names, flush=True)
# GUARD 1: the NaN-quantiser fix (RESULTS 32).
from modules_v9 import BinnedExpression
_q = BinnedExpression(8, n_bins=128, n_genes=978, mode='global')
_x = np.random.default_rng(0).uniform(2, 15, size=(2000, 978)).astype('float32'); _x[3] = np.nan
try:
    _q.fit(_x)
    raise SystemExit('FATAL: mounted code is PRE-FIX -- fit() accepted a NaN row. Refusing.')
except ValueError:
    pass
# GUARD 2-3: their metric, the dev-cell mode, distinct seeding.
import inspect, xpert_arm
_src = inspect.getsource(xpert_arm)
need = ('their_pearson', 'seed_start', 'n_dropped_test', 'ablate_epi', 'def carve_dev', 'dev_row_index_sha1',
        'cuda_rng_states_equal_by_epoch', 'sign_head_w', 'snapshot_cycles')
miss = [k for k in need if k not in _src]
if miss:
    raise SystemExit('FATAL: mounted xpert_arm.py lacks %r. Refusing.' % miss)
_dp = glob.glob(os.path.join(SRC, 'dp_seeding.py'))
if not _dp or 'def seed_devices' not in open(_dp[0], encoding='utf-8').read():
    raise SystemExit('FATAL: mounted dp_seeding.py missing or lacks seed_devices. Refusing.')
# GUARD ARM: the mounted code implements this arm.
ARM_NEEDS = __NEEDS__
_miss = [k for k in ARM_NEEDS if k not in _src and not glob.glob('/kaggle/input/**/' + k, recursive=True)]
if _miss:
    raise SystemExit('FATAL: the mounted code cannot run this arm, missing %r. Refusing.' % _miss)
# GUARD PIN (review 044 C2a): the arm's code identity, not just strings.
import hashlib
_h = hashlib.sha1(open(os.path.join(SRC, 'xpert_arm.py'), 'rb').read()).hexdigest()
if _h not in __PINS__:
    raise SystemExit('FATAL: mounted xpert_arm.py sha1 %s is not an allowed version %r. Refusing.' % (_h, __PINS__))
print('mounted code verified for __ARM__ (xpert_arm.py %s)' % _h[:12], flush=True)
sys.argv = ['xpert_arm.py', '--bundle', 'xpert_mdmt_splits.npz', '--split', 'split_cold_cell_1',
            '--dev_cells', '6', '--dev_seed', '0', '--dp_seed_mode', 'distinct',
            '--seeds', '__NSEEDS__', '--seed_start', '__SEED0__', '--epochs', '12', '--d_model', '256',
            '--save_pred', '/kaggle/working/v9dev___ARM__.npz', '--save_ckpt', '/kaggle/working/v9dev___ARM__.pt'] + __FLAGS__
xpert_arm.main()
# GUARD 4 (RESULTS 85.4): the carve is the recorded one, row for row.
import json as _json
outs = glob.glob('/kaggle/working/**/v9_xpert_arm_*__TAG___dev6s0__SUFFIX__.json', recursive=True) + \
       glob.glob(os.path.join(os.path.dirname(SRC), '**', 'v9_xpert_arm_*__TAG___dev6s0__SUFFIX__.json'), recursive=True)
if not outs:
    raise SystemExit('FATAL: no dev result JSON written.')
rec = _json.load(open(outs[0]))
want = '51e7e4ab8b9c3c3709d43da7fa4a8c80b77d5980'
got = rec['dev']['dev_row_index_sha1']
print('GUARD 4 dev rows sha1', got, 'OK' if got == want else 'MISMATCH', flush=True)
if got != want:
    raise SystemExit('FATAL: dev carve differs from RESULTS 85.4.')
# GUARD 5 (85.5): distinct seeding desynchronises the devices.
for r in rec['runs']:
    eq = r.get('cuda_rng_states_equal_by_epoch') or []
    print('GUARD 5 seed', r['seed'], r.get('device_seeds'), 'equal-by-epoch', eq, flush=True)
    if r.get('n_gpu', 0) > 1 and any(eq):
        raise SystemExit('FATAL: distinct seeding did not desynchronise the devices.')
import shutil
if os.path.dirname(os.path.abspath(outs[0])) != '/kaggle/working':
    shutil.copy(outs[0], '/kaggle/working/')
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', choices=sorted(ARMS))
    ap.add_argument('--seed_start', type=int, default=0)
    ap.add_argument('--seeds', type=int, default=1)
    a = ap.parse_args()
    arm = ARMS[a.arm]
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    tag = ('' if (a.seeds == 3 and a.seed_start == 0) else '_seed%d' % a.seed_start) + arm['tag']
    code = (CODE.replace('__ARM__', a.arm).replace('__SEEDS__', str(seeds)).replace('__NEEDS__', repr(arm['needs']))
            .replace('__NSEEDS__', str(a.seeds)).replace('__SEED0__', str(a.seed_start)).replace('__FLAGS__', repr(arm['flags']))
            .replace('__TAG__', tag).replace('__SUFFIX__', arm['suffix']).replace('__PINS__', repr(arm['xpert_arm_sha1'])))
    slug = 'lincs-v9dev-%s-s%d' % (a.arm, a.seed_start)
    d = os.path.join(ROOT, 'kern_v9dev_%s_s%d' % (a.arm, a.seed_start))
    os.makedirs(d, exist_ok=True)
    io.open(os.path.join(d, slug + '.py'), 'w', encoding='utf-8', newline='\n').write(code)
    meta = {'id': 'apexblue/' + slug, 'title': slug.replace('-', ' '), 'code_file': slug + '.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'keywords': ['gpu'], 'dataset_sources': ['apexblue/lincs-v9-bundle', 'apexblue/lincs-v9-src', 'apexblue/xpert-mdmt-benchmark'],
            'kernel_sources': [], 'competition_sources': [], 'model_sources': [], 'docker_image': IMG, 'machine_shape': 'NvidiaTeslaT4'}
    io.open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n').write(json.dumps(meta, indent=1) + '\n')
    print('wrote', d, a.arm, 'seeds', seeds, 'flags', arm['flags'], 'json glob', 'v9_xpert_arm_*%s_dev6s0%s.json' % (tag, arm['suffix']))


if __name__ == '__main__':
    main()
