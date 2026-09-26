import sys, os, glob, hashlib, torch, numpy as np
print('/kaggle/input contains:', sorted(os.listdir('/kaggle/input')), flush=True)
hit = glob.glob('/kaggle/input/**/train_v9_gpu.py', recursive=True)
if not hit:
    raise SystemExit('FATAL: train_v9_gpu.py not found under /kaggle/input.')
SRC = os.path.dirname(hit[0]); sys.path.insert(0, SRC)
print('source dir:', SRC, flush=True)
if not torch.cuda.is_available():
    raise SystemExit('FATAL: no CUDA device. NOT falling back to CPU.')
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if any('P100' in n for n in names):
    raise SystemExit('FATAL: P100 assigned; Kaggle torch has no sm_60 kernels.')
if len(names) < 2:
    raise SystemExit('FATAL: fewer than 2 GPUs; the distinct-seeding guard would be vacuous.')
print('GPUs:', names, flush=True)

# GUARD 1: the NaN-quantiser fix must be in the mounted code (a stale mount voided a whole round, RESULTS 32).
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

# GUARD 2 (RESULTS 88.1, 88.4): the mounted trainer must carry W20's flags and the model the C8b layer.
import inspect, train_v9_gpu, model_v9
_t, _m = inspect.getsource(train_v9_gpu), inspect.getsource(model_v9)
if not all(k in _t for k in ('--post_pathway', '--dp_seed_mode', '--tf32', 'train_meta')):
    raise SystemExit('FATAL: mounted train_v9_gpu.py lacks the W20 flags. Refusing.')
if 'post_pathway_activations' not in _m:
    raise SystemExit('FATAL: mounted model_v9.py lacks the C8b post-pathway layer. Refusing.')
print('mounted code verified: quantiser fix, W20 trainer flags, C8b layer', flush=True)

# RESULTS 88.6 item 1 (review 027 C1): V9Config defaults otherwise (l_control 2, drug self-attention off), batch 48,
# + post_pathway, distinct seeding, TF32 off -- the SCREENED C8b's architecture, differing only in data and trainer.
SEED = 0
sys.argv = ['train_v9_gpu.py', '--post_pathway', '--seed', str(SEED),
            '--epochs', '12', '--d_model', '256',
            '--budget_h', '8.5', '--gpus', '2', '--dp_seed_mode', 'distinct', '--tf32', 'off']
train_v9_gpu.main()

# GUARD 3: the checkpoint must say what was trained.
src = '/kaggle/working/ckpt_v9_fold0_seed%d.pt' % SEED
ck = torch.load(src, map_location='cpu', weights_only=False)
cfg, tm = ck['cfg'], ck.get('train_meta', {})
print('GUARD 3 cfg.post_pathway', cfg.get('post_pathway'), '| drug_self_attn', cfg.get('drug_self_attn'),
      '| train_meta', {k: tm.get(k) for k in ('dp_seed_mode', 'tf32', 'post_pathway', 'device_seeds')}, flush=True)
eq = tm.get('cuda_rng_states_equal_by_epoch')
print('GUARD 3 equal-by-epoch', eq, flush=True)
# RESULTS 88.6 item 1: architecture fields must equal the screened C8b's (v9dev_c8b_dev6s0_seed0.pt).
WANT = {'d_model': 256, 'n_heads': 8, 'd_ff': 1024, 'l_control': 2, 'l_base': 2, 'l_perturb': 4, 'stoch_depth': 0.1,
        'dropout': 0.1, 'd_pathway': 32, 'drug_self_attn': False, 'use_ppi': True, 'use_aux': True,
        'use_gene_vectors': True, 'epi_as_gene_embedding': True, 'expr_encoder': 'binned', 'post_pathway': True}
bad = {k: (cfg.get(k), v) for k, v in WANT.items() if cfg.get(k) != v}
print('GUARD 3 architecture vs screened C8b:', 'OK' if not bad else bad, flush=True)
if bad:
    raise SystemExit('FATAL: fold-0 C8b architecture differs from the screened C8b: %r' % bad)
if ck['tcfg'].get('fold') != 0 or ck['tcfg'].get('batch') != 48:
    raise SystemExit('FATAL: not fold 0 / batch 48: %r' % {k: ck['tcfg'].get(k) for k in ('fold', 'batch')})
if tm.get('dp_seed_mode') != 'distinct' or tm.get('tf32') != 'off' or not eq or any(eq):
    raise SystemExit('FATAL: checkpoint train_meta does not show distinct seeding / TF32 off / unequal device states.')
if ck.get('epoch', -1) + 1 != 12:
    print('WARNING: checkpoint epoch', ck.get('epoch'), '-- the budget guard may have cut the schedule', flush=True)
dst = '/kaggle/working/c8b_ckpt_v9_fold0_seed%d.pt' % SEED
os.replace(src, dst)
print('checkpoint', dst, 'sha1', hashlib.sha1(open(dst, 'rb').read()).hexdigest(), flush=True)
for f in sorted(glob.glob('/kaggle/working/*')):
    print(f'{os.path.getsize(f)/1e6:9.2f} MB  {f}', flush=True)
