import sys, os, glob, torch, numpy as np
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
print('GPUs:', names, flush=True)

# ---------------------------------------------------------------------------------------------
# GUARD 1 -- the NaN quantiser. A stale dataset version silently reproduces the bug that voided an
# entire round [RESULTS 32]: fit() succeeded, every value bucketed to 0, and the model trained to
# convergence with a constant expression embedding while reporting sensible metrics.
# ---------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------
# GUARD 2 -- THE ARM ITSELF MUST BE PRESENT AND MUST DISCRIMINATE.
#
# This is the whole point of the run. A stale mount without _DrugBlock would train an ordinary
# SA-off model for 5.7 h; interaction_2x2.py would then refuse on the resulting checkpoint and the
# session would be a total loss, discovered at the END rather than the start. Worse, a _DrugBlock
# present but with a broken diagonal path would let the run finish and produce an interaction of
# exactly 0.0 -- indistinguishable from a real null, which is retraction-class 7 [RESULTS 32].
#
# So this does not check that the module EXISTS. It checks that it DISCRIMINATES, both ways:
#   * with contextualisation on, perturbing atom 3 must MOVE atom 1;
#   * under diagonal, the same perturbation must move it by EXACTLY 0.0.
# ---------------------------------------------------------------------------------------------
from config_v9 import V9Config
from modules_v9 import _DrugBlock
_c = V9Config(); _c.d_model, _c.n_heads, _c.d_ff, _c.dropout = 64, 4, 128, 0.0
if not hasattr(_c, 'drug_self_attn'):
    raise SystemExit('FATAL: mounted config_v9 has no drug_self_attn field. Stale mount.')
torch.manual_seed(0)
_b = _DrugBlock(_c, 0.0).eval()
_D = torch.randn(2, 6, _c.d_model)
_km = torch.zeros(2, 6, dtype=torch.bool); _km[:, 4:] = True
with torch.no_grad():
    _a0 = _b(_D, key_mask=_km)
    _D2 = _D.clone(); _D2[:, 3, :] += 5.0
    _a1 = _b(_D2, key_mask=_km)
    _d0 = _b(_D, key_mask=_km, diagonal=True)
    _d1 = _b(_D2, key_mask=_km, diagonal=True)
_ctx = (_a0[:, 1] - _a1[:, 1]).abs().max().item()
_dia = (_d0[:, 1] - _d1[:, 1]).abs().max().item()
if _ctx <= 1e-4:
    raise SystemExit(f'FATAL: _DrugBlock does NOT contextualise (atom 3 -> atom 1 moved {_ctx:.2e}). '
                     'Training this would test nothing.')
if _dia != 0.0:
    raise SystemExit(f'FATAL: diagonal ablation leaks cross-atom flow ({_dia:.2e} != 0). The 2x2 '
                     'interaction would be contaminated and its null would be uninterpretable.')
# Review 005 C6: this line used to print the PROBE block's parameter count (32,868 at d_model=64,
# d_ff=128) while being cited as evidence about the trained model, which is 24x larger. The probe count
# is still useful -- it says which module was actually instantiated -- but the line must not be readable
# as a statement about the run. Print both, labelled.
_c_train = V9Config()
n_probe = sum(p.numel() for p in _DrugBlock(_c, 0.0).parameters())
n_train = sum(p.numel() for p in _DrugBlock(_c_train, 0.0).parameters())
print(f'mounted code verified: quantiser fix present; _DrugBlock contextualises (|d|={_ctx:.4f}) '
      f'and diagonal is exact (|d|={_dia:.1e}). PROBE block (d_model={_c.d_model}): {n_probe:,} params; '
      f'block at TRAINING shape (d_model={_c_train.d_model}, d_ff={_c_train.d_ff}): {n_train:,} params',
      flush=True)

# ---------------------------------------------------------------------------------------------
# THE EXPERIMENT [RESULTS 58.7]
#
# Ablating the per-atom drug tokens IMPROVES accuracy by -0.0146 (paired mean -0.0173) on unseen
# compounds -- measured over 3 seeds at n=1500, every interval excluding zero, cross-seed range
# smaller than the within-seed interval [RESULTS 57]. The hypothesis is that atoms hurt because
# they are never re-contextualised inside the block loop: XPert's crossEncoder runs drug_SA before
# each cross-attention, v9 built D once outside the loop and reused it [RESULTS 47.2].
#
# NOT the original premise. The atom vectors are Uni-Mol atomic_reprs -- already contextualised by a
# transformer with a 3D distance bias -- so what is under test is a SECOND, in-loop pass that
# co-evolves with the cell embedding, not "adding structure where there was none" [RESULTS 56.1].
#
# This run produces ONE SA-on checkpoint. The decisive quantity is computed afterwards, within
# these weights, by interaction_2x2.py:
#     INTERACTION = (S11 - S01) - (S10 - S00)
# where S01/S00 mean-ablate the atom tokens and S10/S00 run drug_sa under diagonal attention.
# Read against S10-S00 from THIS run, never against the historical -0.0146 [review 004 C5].
#
# Reading rule PRE-COMMITTED before this spend [RESULTS 58.4]: >=3 sigma on the paired-mean
# interval -> a one-seed result; 1.5-3 sigma -> explicitly inconclusive; <1.5 sigma with both
# ablations confirmed fired -> informative null; either |dY|max ~ 0 -> VOID, not null.
# ---------------------------------------------------------------------------------------------
import train_v9_gpu
SEED = 0
sys.argv = ['train_v9_gpu.py', '--drug_self_attn', '--seed', str(SEED),
            '--epochs', '12', '--d_model', '256', '--l_control', '1',
            '--budget_h', '7.5', '--gpus', '2']
train_v9_gpu.main()

print('\n--- artefacts ---', flush=True)
for f in sorted(glob.glob('/kaggle/working/*')):
    print(f'{os.path.getsize(f)/1e6:9.2f} MB  {f}', flush=True)
