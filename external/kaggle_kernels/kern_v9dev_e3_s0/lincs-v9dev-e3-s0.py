# RESULTS 92 (pre-registered): e3 on the P2 recipe, dev carve, seeds [0, 1, 2]. Guards as the V2 dev kernel, plus the arm's.
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
ARM_NEEDS = ['def encode_chromatin', 'TIE_EXPECTED_MASKED', 'TIE_MIN_NONTIED', 'E_final_provenance.json']
_miss = [k for k in ARM_NEEDS if k not in _src and not glob.glob('/kaggle/input/**/' + k, recursive=True)]
if _miss:
    raise SystemExit('FATAL: the mounted code cannot run this arm, missing %r. Refusing.' % _miss)
# GUARD PIN (review 044 C2a): the arm's code identity, not just strings.
import hashlib
_h = hashlib.sha1(open(os.path.join(SRC, 'xpert_arm.py'), 'rb').read()).hexdigest()
if _h not in ['50ff83da08c55f6b3ad78bf31dfc790e7cc762b4']:
    raise SystemExit('FATAL: mounted xpert_arm.py sha1 %s is not an allowed version %r. Refusing.' % (_h, ['50ff83da08c55f6b3ad78bf31dfc790e7cc762b4']))
print('mounted code verified for e3 (xpert_arm.py %s)' % _h[:12], flush=True)
sys.argv = ['xpert_arm.py', '--bundle', 'xpert_mdmt_splits.npz', '--split', 'split_cold_cell_1',
            '--dev_cells', '6', '--dev_seed', '0', '--dp_seed_mode', 'distinct',
            '--seeds', '3', '--seed_start', '0', '--epochs', '12', '--d_model', '256',
            '--save_pred', '/kaggle/working/v9dev_e3.npz', '--save_ckpt', '/kaggle/working/v9dev_e3.pt'] + ['--chromatin_encoding', 'tie']
xpert_arm.main()
# GUARD 4 (RESULTS 85.4): the carve is the recorded one, row for row.
import json as _json
outs = glob.glob('/kaggle/working/**/v9_xpert_arm_*_dev6s0_chromtie.json', recursive=True) + \
       glob.glob(os.path.join(os.path.dirname(SRC), '**', 'v9_xpert_arm_*_dev6s0_chromtie.json'), recursive=True)
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
