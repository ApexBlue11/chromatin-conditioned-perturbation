import sys, os, glob, torch, numpy as np
print('/kaggle/input contains:', sorted(os.listdir('/kaggle/input')), flush=True)
hit = glob.glob('/kaggle/input/**/xpert_arm.py', recursive=True)
if not hit:
    raise SystemExit('FATAL: xpert_arm.py not found under /kaggle/input.')
SRC = os.path.dirname(hit[0]); sys.path.insert(0, SRC)
print('source dir:', SRC, flush=True)
if not torch.cuda.is_available():
    raise SystemExit('FATAL: no CUDA device. NOT falling back to CPU.')
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if any('P100' in n for n in names):
    raise SystemExit('FATAL: P100 assigned; Kaggle torch has no sm_60 kernels.')
print('GPUs:', names, flush=True)

# GUARD 1: the mounted code must post-date the NaN-quantiser fix. A stale dataset version would silently
# reproduce the bug that voided an entire round: fit() succeeded, every value bucketed to 0, and the model
# trained to convergence with a constant expression embedding.
from modules_v9 import BinnedExpression
_q = BinnedExpression(8, n_bins=128, n_genes=978, mode='global')
_x = np.random.default_rng(0).uniform(2, 15, size=(2000, 978)).astype('float32'); _x[3] = np.nan
try:
    _q.fit(_x)
    raise SystemExit('FATAL: mounted code is PRE-FIX -- fit() accepted a NaN row. Refusing.')
except ValueError:
    pass
_q.fit(_x[np.isfinite(_x).all(1)])
if not _q.discriminates(torch.as_tensor(_x[:64][np.isfinite(_x[:64]).all(1)])):
    raise SystemExit('FATAL: quantiser does not discriminate after a clean fit.')

# GUARD 2: the arm must be the version that reports XPert's OWN metric (mean of per-row Pearson). The
# earlier version reported our median, which on these data flatters by 0.02-0.05 -- comparing our median
# against their mean is exactly the kind of mismatch this arm exists to remove.
import inspect, xpert_arm
_src = inspect.getsource(xpert_arm)
if not all(k in _src for k in ('their_pearson', 'seed_start', 'n_dropped_test', 'ablate_epi')):
    raise SystemExit('FATAL: mounted xpert_arm.py lacks --ablate_epi. A stale mount would run the ABLATED arm with chromatin still present, making it identical to the full arm and manufacturing a null. Also predates the mean-Pearson / per-seed / unfeaturisable-row fixes. Refusing -- it would KeyError on the 164 test compounds we cannot featurise.')
print('mounted code verified: quantiser fix + their-metric reporting present', flush=True)

# GUARD 3 (RESULTS 85): the mounted arm must carry the dev-cell mode (W15, 69e623f).
if not all(k in _src for k in ('def carve_dev', 'dev_cells', 'dev_row_index_sha1', 'original_test_row_indices',
                              'def seed_devices', 'cuda_rng_states_equal_by_epoch', 'no_atoms', 'deg_adapt_k', 'post_pathway', 'listnet_w', 'sign_head_w')):
    raise SystemExit('FATAL: mounted xpert_arm.py lacks the dev-cell mode. Refusing.')

# RESULTS 85.7 candidate C1: --no_atoms, seeds 0..0, the P2 command otherwise.
sys.argv = ['xpert_arm.py', '--bundle', 'xpert_mdmt_splits.npz', '--split', 'split_cold_cell_1',
            '--dev_cells', '6', '--dev_seed', '0', '--dp_seed_mode', 'distinct',
            '--seeds', '1', '--seed_start', '0', '--epochs', '12', '--d_model', '256',
            '--save_pred', '/kaggle/working/v9dev_c1.npz', '--save_ckpt', '/kaggle/working/v9dev_c1.pt'] + ['--no_atoms']
xpert_arm.main()

# GUARD 4 (RESULTS 85.4): the carve must be the recorded one, row for row.
import json as _json
outs = glob.glob('/kaggle/working/**/v9_xpert_arm_*_dev6s0_noatoms.json', recursive=True) +        glob.glob(os.path.join(os.path.dirname(SRC), '**', 'v9_xpert_arm_*_dev6s0_noatoms.json'), recursive=True)
if not outs:
    raise SystemExit('FATAL: no dev result JSON written.')
rec = _json.load(open(outs[0]))
want = '51e7e4ab8b9c3c3709d43da7fa4a8c80b77d5980'
got = rec['dev']['dev_row_index_sha1']
print('GUARD 4 dev rows sha1', got, 'OK' if got == want else 'MISMATCH', flush=True)
if got != want:
    raise SystemExit('FATAL: dev carve differs from RESULTS 85.4.')
# GUARD 5 (85.5): with distinct seeding the devices must NOT be in lockstep after any epoch.
for r in rec['runs']:
    eq = r.get('cuda_rng_states_equal_by_epoch') or []
    print('GUARD 5 seed', r['seed'], r.get('device_seeds'), 'equal-by-epoch', eq, flush=True)
    if r.get('n_gpu', 0) > 1 and any(eq):
        raise SystemExit('FATAL: distinct seeding did not desynchronise the devices.')
import shutil
if os.path.dirname(os.path.abspath(outs[0])) != '/kaggle/working':
    shutil.copy(outs[0], '/kaggle/working/')
