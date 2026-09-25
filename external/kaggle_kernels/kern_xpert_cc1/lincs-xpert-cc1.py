# -*- coding: utf-8 -*-
"""
Train XPert on split_cold_cell_1 AS PUBLISHED, then predict its test rows with our row-indexed harness.
[RESULTS 46.5, 66.8, 68, 69, 70; packets 006-008]

WHAT THIS RUN IS, decided in advance and recorded in RESULTS 69.1 before any spend
  * Their code, their data, their split column, their config, their seed (2024), their recipe -- including
    utils.py:133's `val_data = test_data` fallback. The released h5ad has NO `valid` level on any split, so
    that fallback fires and their early stopping MONITORS TEST LOSS [review 007 C1]. The asymmetry is
    disclosed and its reading is fixed: a v9 win is CONSERVATIVE, an XPert win is UNINTERPRETABLE.
  * Option A attention: the unmasked flash semantics their executed path uses, via our shim. Confirmed as
    their published behaviour by MEASUREMENT on the released checkpoint [RESULTS 70.1: masking it costs
    -0.0144, CI excluding zero]. The shim's optional mask stays OFF; GUARD C checks that.
  * Fold 1 only [review 007 C5]: v9 is fold 1 at one seed.

WHAT IS CHANGED, ALL DECLARED IN RECORD['deviations'] [reviews 008b, 009 C1]
  The uploaded dataset keeps their files verbatim (sha1-checked). The staged copy in /kaggle/working gets:
    1. the flash_attn shim on PYTHONPATH (their model imports it at module scope; not on this image);
    2. the dense unimol array their config names but never released, rebuilt from the npz they did [RESULTS 68];
    3. empty __init__.py in datasets/ and models/, so HuggingFace `datasets` cannot shadow theirs [RESULTS 71.9];
    4. ONE executed line of MyDataset.py: the per-row drug tensor copy made once per drug instead [RESULTS 73].
       Correct by construction (the cache key is the same expression as the lookup it replaces), and proven
       on 6,000 tensors. Without it their loader needs ~24.7 GB of dataset RAM on this fold.
  Items 1-3 change no executed line. Item 4 changes storage, not any value. None bears on "as published".

WHY ONE GPU OF TWO
  train_xpert.py is single-device, and utils.py:155-157 hardcode num_workers=10, which already oversubscribes
  this image's CPUs about 2.5x for ONE process. A second fold on the idle GPU would contend for the same
  CPUs and slow fold 1 -- the only fold the comparison needs. A deliberate exception to "use every unit".

Never read their reported metrics [review 007 C7]: train_xpert.py:573-574 overwrites test with val.
Predictions are scored later by head_to_head_mdmt.py against v9's split_cold_cell_1 predictions.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
BUDGET_H = 8.3          # wall-clock guard for TRAINING; prediction needs ~10 min after it
# Review 011 C1: checkpointing slows every step, so fewer epochs fit in BUDGET_H, and a guard stop while test
# loss is still improving supports no v9-win claim [RESULTS 71.7]. Nobody knows XPert's convergence epoch, so
# the real T4 step time is measured FIRST, in a run that stops after GUARD F, and the full run is decided
# with that number in hand rather than discovered through stopped_by == 'watchdog'.
MEASURE_ONLY = False
# O2 PRODUCTION [RESULTS 84, review 015]. GUARD G / H (RESULTS 81) were proved in v7 [81.7] and are not rerun.
SESSION = 2
HORIZON_EPOCHS = 297          # 84.1: FINAL at 297 completed epochs; the only other final end is their early stop
PREV = {'cuda': '12.8', 'final_epoch': 65, 'sha1': {'best.pth': '8b216a57aecbca8d8a7ccb03f71ec241dc8b1eb1', 'full_state.pt': '59c5e9603de2a741e0d397a48f11475e94a2b087', 'resume_from.pt': '0f6989e25adfdc3b71ee86ca6a1198914a64693f'}, 'torch': '2.10.0+cu128'}
# PREV = session k-1's handoff, pasted as a literal and committed to git before this push (015 C3).
# Production session 2. The v7 proof and resume tests are in the v7 kernel (git da3c5ac).
FOLD = 'split_cold_cell_1'
W = '/kaggle/working'
RECORD = {'fold': FOLD, 'budget_h': BUDGET_H, 'guards': {}, 'decision': 'option (a) as published, disclosed'}


def log(*a):
    print('[%6.1f min]' % ((time.time() - T0) / 60), *a, flush=True)


# HOST MEMORY TRACE [review 009 C5]. v3 died with no traceback and, until GUARD F was fixed, no return code.
# A training-time OOM at hour five must leave a number behind, so MemAvailable is sampled every 60 s for the
# whole run -- system-wide, because the trainer and its 20 DataLoader workers are separate processes.
import threading  # noqa: E402

MEM = []


def _mem_available_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None


def _mem_sampler():
    while True:
        MEM.append((round((time.time() - T0) / 60, 1), round(_mem_available_gb() or -1, 2)))
        time.sleep(60)


threading.Thread(target=_mem_sampler, daemon=True).start()


def mem_summary():
    ok = [m for _, m in MEM if m >= 0]
    return {'min_available_gb': min(ok) if ok else None, 'samples': len(MEM), 'trace_min_gb': MEM[-400:]}


def fatal(msg):
    RECORD['fatal'] = msg
    RECORD['host_memory'] = mem_summary()
    json.dump(RECORD, open(os.path.join(W, 'run_record.json'), 'w'), indent=2)
    raise SystemExit('FATAL: ' + msg)


def find_one(pattern):
    hits = glob.glob(pattern, recursive=True)
    if len(hits) != 1:
        fatal('expected exactly one match for %s, got %d: %s' % (pattern, len(hits), hits[:5]))
    return hits[0]


# --------------------------------------------------------------------------------------------------------
# 0. Hardware
# --------------------------------------------------------------------------------------------------------
import torch  # noqa: E402

if not torch.cuda.is_available():
    fatal('no CUDA device. NOT falling back to CPU.')
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if any('P100' in n for n in names):
    fatal('P100 assigned; this image has no sm_60 kernels.')
TORCH_BEFORE = torch.__version__
RECORD['gpus'] = names
RECORD['torch'] = TORCH_BEFORE
log('GPUs:', names, '| torch', TORCH_BEFORE)

# --------------------------------------------------------------------------------------------------------
# 1. Their training dependencies. Installed into THIS disposable image, never into a local env: the local
#    dry-run pulled ~50 packages including a numpy change [RESULTS 70.4]. torch must not move.
# --------------------------------------------------------------------------------------------------------
# torch_geometric: models/model_XPert.py imports HeteroConv and SAGEConv at module scope. Pure-python install
# only -- the optional compiled extensions are not needed for those two layers, and pulling them would risk
# a torch-version-matched wheel that moves torch.
pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scanpy', 'torchmetrics',
                      'unimol-tools', 'torch_geometric'], capture_output=True, text=True)
log('pip exit', pip.returncode)
if pip.returncode != 0:
    print(pip.stdout[-3000:], pip.stderr[-3000:])
    fatal('pip install of their dependencies failed.')
torch_after = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'],
                             capture_output=True, text=True).stdout.strip()
if torch_after != TORCH_BEFORE:
    fatal('pip changed torch %s -> %s. Refusing: the run would not be on the image we tested.'
          % (TORCH_BEFORE, torch_after))
for mod in ('scanpy', 'torchmetrics', 'unimol_tools', 'torch_geometric', 'anndata', 'h5py', 'yaml'):
    r = subprocess.run([sys.executable, '-c', 'import %s' % mod], capture_output=True, text=True)
    if r.returncode != 0:
        fatal('cannot import %s after install: %s' % (mod, r.stderr[-500:]))
RECORD['guards']['deps'] = 'scanpy, torchmetrics, unimol_tools importable; torch unchanged'

# --------------------------------------------------------------------------------------------------------
# 2. Stage their code into a WRITABLE dir (their trainer writes experiment/ relative to cwd) and link the
#    large read-only data rather than copying it.
# --------------------------------------------------------------------------------------------------------
SRC = os.path.dirname(find_one('/kaggle/input/**/XPert/train_xpert.py'))
V9_RO = os.path.dirname(find_one('/kaggle/input/**/xpert_native_eval.py'))
# Our harness writes its JSON to `<its own dir>/../results`. Under /kaggle/input that is read-only, so it would
# crash AFTER writing a valid profile and this kernel would fatal on a good run. Run a writable copy.
V9 = os.path.join(W, 'v9src')
shutil.copytree(V9_RO, V9)
os.makedirs(os.path.join(W, 'results'), exist_ok=True)
X = os.path.join(W, 'XPert')
shutil.copytree(SRC, X, symlinks=False, ignore=shutil.ignore_patterns('processed_data', 'HG_data'))
os.makedirs(os.path.join(X, 'processed_data'), exist_ok=True)
os.makedirs(os.path.join(X, 'HG_data', 'saved_embedding'), exist_ok=True)
for f in os.listdir(os.path.join(SRC, 'processed_data')):
    os.symlink(os.path.join(SRC, 'processed_data', f), os.path.join(X, 'processed_data', f))
os.symlink(os.path.join(SRC, 'HG_data', 'saved_embedding', 'HG_drug_embeddings.npy'),
           os.path.join(X, 'HG_data', 'saved_embedding', 'HG_drug_embeddings.npy'))
H5AD = os.path.join(X, 'processed_data', 'l1000_mdmt_68830_subset.h5ad')
NPZ = os.path.join(X, 'processed_data', 'unimol_mdmt_1970.npz')

# Version 1 of this kernel passed every guard below and then died on the trainer's first import:
#     ModuleNotFoundError: No module named 'datasets.MyDataset'
# Their `datasets/` and `models/` directories have no __init__.py, so they are NAMESPACE packages, and Python
# lets a REGULAR package anywhere later on sys.path beat a namespace one. This image ships HuggingFace's
# `datasets`, so `import datasets` resolved to it. It never showed locally because .venv-cuda has no HF
# datasets. Reproduced locally with a stand-in regular package, and the fix verified the same way.
# The fix adds two EMPTY files to the staged copy; no executed line of their code changes. Declared.
for pkg in ('datasets', 'models'):
    init = os.path.join(X, pkg, '__init__.py')
    if not os.path.exists(init):
        open(init, 'w').close()
# MEMORY PATCH [RESULTS 73, packet 009]. Launch v3's one-batch probe was killed by the OS at row ~18,500 of the
# SECOND test-set build: their MyDataset.load_data copies each drug's (122, 514) float32 block (245 KB) into
# EVERY row, and the val = test fallback builds the test rows twice -- measured 268 KB/row, 24.7 GB for
# train + val + test, against ~29 GB on this image. The patch computes that same tensor once per drug and
# reuses it. model/v9/prove_mydataset_patch.py builds their dataset both ways on identical rows: 6,000 tensors,
# every one torch.equal with identical dtype and shape. Values unchanged; storage shared.
MYDS = os.path.join(X, 'datasets', 'MyDataset.py')
_ORIG = 'drug_feat = tensor(drug_feat, dtype=torch.float32) if self.args.drug_feat != \'smi\' else drug_feat'
_PATCH = ('# [LINCS memory patch, RESULTS 73] one tensor per drug, reused: same values, shared storage.\n'
          '            if self.args.drug_feat != \'smi\':\n'
          '                _k = pert_id if self.args.dataset == \'transigen_sdst\' else pert_idx\n'
          '                _c = self.__dict__.setdefault(\'_drug_tensor_cache\', {})\n'
          '                if _k not in _c:\n'
          '                    _c[_k] = tensor(drug_feat, dtype=torch.float32)\n'
          '                drug_feat = _c[_k]')
_src = open(MYDS, encoding='utf-8').read()
if _src.count(_ORIG) != 1:
    fatal('memory patch: the target line occurs %d times in their MyDataset.py, expected exactly 1.'
          % _src.count(_ORIG))
open(MYDS, 'w', encoding='utf-8').write(_src.replace(_ORIG, _PATCH, 1))
import hashlib  # noqa: E402
RECORD['memory_patch'] = {'file': 'datasets/MyDataset.py', 'line_replaced': _ORIG,
                          'sha1_before': hashlib.sha1(_src.encode('utf-8')).hexdigest()[:12],
                          'sha1_after': hashlib.sha1(open(MYDS, 'rb').read()).hexdigest()[:12],
                          'proof': 'model/v9/prove_mydataset_patch.py: 6000 tensors torch.equal'}
# ACTIVATION CHECKPOINTING [RESULTS 77, packet 011]. v4 died of CUDA OOM at batch 128: stored activations measure
# ~0.117 GiB/sample, ~14.95 GiB at 128, on a 14.56 GiB T4. Gradient accumulation is NOT exact here (their
# batch_weighted_loss takes square roots of whole-batch means) and DataParallel needs edits to their forward. The
# patch below is applied at RUNTIME by a wrapper, so their files stay verbatim. Its source is generated from
# model/v9/xpert_ckpt_patch.py, the module prove_checkpoint_exact.py tests: gradients within the run-to-run noise
# floor, loss equal to 8 decimals, ~3.7 GiB at batch 128.
CKPT_PATCH_SRC = '# -*- coding: utf-8 -*-\n"""Activation checkpointing for XPert\'s encoder layers, applied at RUNTIME so their source files stay verbatim.\n[RESULTS 77, packet 011]\n\nTheir published recipe (batch 128, 978 gene tokens) needs ~0.117 GiB of stored activations per sample in training --\n~14.95 GiB at batch 128 -- against a T4\'s 14.56 GiB. Checkpointing stores only layer-boundary activations and recomputes\nthe rest in the backward pass: same batch, same loss, same gradients (model/v9/prove_checkpoint_exact.py: loss equal to\n8 decimals, gradient difference 4.353e-05 inside the 4.630e-05 run-to-run noise floor, measured by the run that\nimports THIS module; an earlier run with an inline copy gave 5.9e-05 within 8.7e-05), ~3.7 GiB at batch 128.\n\nActive only in training with grad enabled, so inference and evaluation are untouched. use_reentrant=False, which\npreserves the RNG state for dropout replay and the autocast state for recomputation.\n"""\nimport sys\n\nimport torch\nimport torch.utils.checkpoint as cp\n\n\ndef apply(model_utils=None):\n    """Patch models.model_utils.Encoder and .crossEncoder in place. Returns the list of patched class names."""\n    MU = model_utils or sys.modules[\'models.model_utils\']\n    patched = []\n    for cls in (MU.Encoder, MU.crossEncoder):\n        if getattr(cls.forward, \'_lincs_checkpointed\', False):\n            continue\n        orig = cls.forward\n\n        def make(orig):\n            def fwd(self, *a, **k):\n                if self.training and torch.is_grad_enabled():\n                    return cp.checkpoint(orig, self, *a, use_reentrant=False, **k)\n                return orig(self, *a, **k)\n            fwd._lincs_checkpointed = True\n            return fwd\n\n        cls.forward = make(orig)\n        patched.append(cls.__name__)\n    return patched\n'
open(os.path.join(X, 'xpert_ckpt_patch.py'), 'w', encoding='utf-8').write(CKPT_PATCH_SRC)
open(os.path.join(X, 'run_train_ckpt.py'), 'w', encoding='utf-8').write(
    'import sys\n'
    'sys.argv = ["train_xpert.py"] + sys.argv[1:]\n'
    'import models.model_utils\n'
    'import xpert_ckpt_patch\n'
    'print("LINCS activation checkpointing applied to:", xpert_ckpt_patch.apply(), flush=True)\n'
    'import train_xpert\n'
    'train_xpert.main()\n')
# DATAPARALLEL MEASUREMENT [RESULTS 78.4, 80; review 012]. Sources generated from model/v9/xpert_dp_patch.py and
# model/v9/xpert_dp_probe.py. The probe is kept for provenance; production does not run it (proved in v7, 81.7).
DP_PATCH_SRC = '# -*- coding: utf-8 -*-\n"""DataParallel for XPert over every visible GPU, applied at RUNTIME inside XPertNet.forward so their files stay\nverbatim. [RESULTS 78, review 012]\n\nWhy inside forward rather than wrapping the model in nn.DataParallel: the object their train_xpert.py holds stays an\nXPertNet. So state_dict() keys carry NO \'module.\' prefix -- review 012 C1 showed their --resume_from filter would match\nno key of a prefixed checkpoint and silently restart from random weights while logging success -- and every attribute\ntheir code touches keeps working.\n\nWhy it is exact for their recipe, where gradient accumulation was not [RESULTS 77.2]: torch.nn.parallel.data_parallel\nscatters the batch, runs one replica per GPU, and GATHERS the outputs to GPU 0 before their train() computes the loss,\nso batch_weighted_loss\'s sqrt(loss / num_samples) terms see all 128 samples exactly as on one GPU. Gradients from the\nreplicas are summed into the original parameters. XPert has LayerNorm only, no BatchNorm, so no statistic depends on\nthe per-replica batch. Dropout masks differ per replica: i.i.d. draws of the same Bernoulli, a different realisation,\nas a different seed is [review 012 ask 3].\n\nTwo things in their forward are pinned to one device and are localised on each replica:\n  * `self.device`, which forward uses to move every input (model_XPert.py:188-198);\n  * `self.drug_HG_embed`, a plain tensor attribute created on `device` (model_XPert.py:135). DataParallel replicates\n    parameters and buffers only; a plain tensor stays on GPU 0. It is constant (never trained), so a cached copy per\n    device is exact. The localiser is generic over every plain tensor attribute of every submodule, and\n    `plain_tensor_attributes()` lists them so a kernel guard can assert the list is exactly the one expected.\n\nActive only in training with grad enabled and at least one row per GPU, so validation, prediction and a ragged last\nbatch smaller than the GPU count run unchanged on one device.\n"""\nimport sys\n\nimport torch\nfrom torch.nn.parallel import data_parallel\n\n# (id(source), device) -> (source, copy). The source is kept and checked by IDENTITY, so a reused id() can never\n# serve a stale copy [RESULTS 80.5, Amendment F].\n_CACHE = {}\n# Only these plain tensor attributes are moved to a replica\'s device; any other tensor attribute found off-device\n# is an error, not something to copy. On a replica the parameter copies are plain attributes too (replicate()\n# sets them that way), already on the device -- they must never be moved.\nLOCALISE = frozenset([\'drug_HG_embed\'])\n# Set by the probe: True asserts autocast is ON inside every replica forward, False asserts it is OFF, None skips.\nEXPECT_AUTOCAST = None\nREPLICA_CALLS = {\'n\': 0, \'autocast_on\': 0}\n\n\ndef plain_tensor_attributes(model):\n    """Every tensor held as a plain attribute (not a parameter or buffer), as \'module.path.attr\'."""\n    out = []\n    for mname, m in model.named_modules():\n        for name, val in vars(m).items():\n            if torch.is_tensor(val):\n                out.append((mname + \'.\' if mname else \'\') + name)\n    return sorted(out)\n\n\ndef _localise(replica, dev):\n    for m in replica.modules():\n        for name, val in list(vars(m).items()):\n            if not torch.is_tensor(val) or val.device == dev:\n                continue\n            if name not in LOCALISE:\n                raise RuntimeError(\'xpert_dp_patch: tensor attribute %r is on %s, not the replica device %s, and is not \'\n                                   \'in the localise set %s\' % (name, val.device, dev, sorted(LOCALISE)))\n            key = (id(val), dev)\n            hit = _CACHE.get(key)\n            if hit is None or hit[0] is not val:\n                hit = (val, val.to(dev))\n                _CACHE[key] = hit\n            setattr(m, name, hit[1])      # replicas hold a shallow copy of __dict__: the original is untouched\n    replica.device = dev\n\n\ndef apply(model_XPert=None, device_ids=None):\n    """Patch models.model_XPert.XPertNet.forward in place. Returns the device ids it will use."""\n    MX = model_XPert or sys.modules[\'models.model_XPert\']\n    cls = MX.XPertNet\n    ids = list(device_ids) if device_ids is not None else list(range(torch.cuda.device_count()))\n    if getattr(cls.forward, \'_lincs_dp\', False):\n        return ids\n    orig = cls.forward\n\n    def fwd(self, data, *a, **k):\n        if getattr(self, \'_is_replica\', False):          # set by torch.nn.parallel.replicate\n            # NOT next(self.parameters()).device: replicate() turns a replica\'s parameters into plain non-leaf\n            # attributes, so parameters() is EMPTY on a replica (StopIteration -- caught by the local smoke test,\n            # 2026-09-24, before any GPU spend). parallel_apply runs each replica under\n            # torch.cuda.device(<its device>), so the current device is the replica\'s.\n            _localise(self, torch.device(\'cuda\', torch.cuda.current_device()))\n            on = torch.is_autocast_enabled()\n            REPLICA_CALLS[\'n\'] += 1\n            REPLICA_CALLS[\'autocast_on\'] += int(on)\n            if EXPECT_AUTOCAST is not None and on != EXPECT_AUTOCAST:\n                raise RuntimeError(\'xpert_dp_patch: autocast is %s inside a replica, expected %s [RESULTS 81.5]\'\n                                   % (on, EXPECT_AUTOCAST))\n            return orig(self, data, *a, **k)\n        if len(ids) > 1 and self.training and torch.is_grad_enabled() and data[0].shape[0] >= len(ids):\n            if a:\n                raise TypeError(\'xpert_dp_patch: positional arguments after data are not scattered; pass mode= by name\')\n            return data_parallel(self, (data,), device_ids=ids, output_device=ids[0], module_kwargs=k or None)\n        return orig(self, data, *a, **k)\n\n    fwd._lincs_dp = True\n    cls.forward = fwd\n    return ids\n'
DP_PROBE_SRC = '# -*- coding: utf-8 -*-\n"""One configuration of the v7 DataParallel proof (RESULTS 81, final form 81.5 + 81.6), run as its own process inside\nthe staged XPert copy.\n\n    python xpert_dp_probe.py MODE SHARED_DIR FOLD \'THEIR_ARGV_AS_JSON\'\n    MODE in {single_ckpt, single_ckpt_repeat, dp, split}\n\nThe first process (`single_ckpt`) writes the initial weights and one recipe batch to SHARED_DIR, and finds the fp16\nGradScaler scale; every later one loads them strictly. The ten parameters their loss never uses are frozen in EVERY\nmode (RESULTS 80.5, Amendment A), and after every gradient capture the set with `grad is None` must equal them exactly.\n\nTags, all through THEIR train() with an lr-0 SGD step and dropout zeroed on every module:\n  f64_e0 / f64_e70    float64, 16 rows (8 per GPU)         81.1a  semantics      gradients KEPT in float64 (014 C2)\n  f32m_e0 / f32m_e70  fp32, math SDPA, 16 rows              81.1b  dp vs split    (and reported vs single)\n  f16_e0 / f16_e70    fp16 autocast, 128 rows, one scale    81.1c  reported only; autocast asserted inside replicas\n16 rows, not 32 (RESULTS 81.6): on the math backend each of the 8 gene self-attention layers stores a 979x979x8\nprobability tensor for backward, ~245 MB per sample in fp32 and twice that in float64, so the uncheckpointed 32-row\nsplit and the float64 DP at 16 per GPU would not fit a T4.\n\nModes: single_ckpt (reference; every tag), single_ckpt_repeat (floor; every tag), dp (no checkpointing -- the\nproduction variant -- every tag, plus a short timing), split (one GPU, the batch as two halves concatenated before their\nloss, no checkpointing; f32m tags only).\n"""\nimport contextlib\nimport itertools\nimport json\nimport logging\nimport os\nimport sys\nimport time\n\nMODE, SHARED, FOLD = sys.argv[1], sys.argv[2], sys.argv[3]\nsys.argv = json.loads(sys.argv[4])\n\nimport torch  # noqa: E402\nimport yaml  # noqa: E402\n\nimport train_xpert as T  # noqa: E402\nfrom utils import load_dataloader  # noqa: E402\nimport models.model_XPert as MX  # noqa: E402\nimport models.model_utils  # noqa: E402,F401\nimport xpert_ckpt_patch  # noqa: E402\nimport xpert_dp_patch  # noqa: E402\n\nFROZEN = sorted([\'attnEncoder_trt.crossEncoders.0.LayerNorm.beta\', \'attnEncoder_trt.crossEncoders.0.LayerNorm.gamma\',\n                 \'attnEncoder_trt.crossEncoders.1.LayerNorm.beta\', \'attnEncoder_trt.crossEncoders.1.LayerNorm.gamma\',\n                 \'cell_emb.linear.bias\', \'cell_emb.linear.weight\', \'ctl_fc.0.bias\', \'ctl_fc.0.weight\',\n                 \'ctl_fc.3.bias\', \'ctl_fc.3.weight\'])\nROWS_SMALL = 16\nreport = {\'mode\': MODE, \'torch\': torch.__version__, \'gpus_visible\': torch.cuda.device_count(), \'tags\': {}}\n\nargs = T.arg_parse()\nconfig = yaml.safe_load(open(\'configs/%s.yaml\' % args.config))\nlogger = logging.getLogger(\'dp_probe\')\ntr, val, te, adata = load_dataloader(args, config, logger, nfold=FOLD, return_rawdata=True)\n\nif MODE in (\'single_ckpt\', \'single_ckpt_repeat\'):\n    report[\'ckpt_patched\'] = xpert_ckpt_patch.apply()\nif MODE == \'dp\':\n    report[\'dp_device_ids\'] = xpert_dp_patch.apply()\n    assert len(report[\'dp_device_ids\']) == 2, \'DataParallel needs both T4s, saw %r\' % report[\'dp_device_ids\']\n\n# ---- split: the same-GPU reference of 81.1b -- DataParallel\'s computation minus the second device ----------------\nORIG_FWD = MX.XPertNet.forward\n\n\ndef _cat(outs):\n    a = outs[0]\n    if torch.is_tensor(a):\n        return torch.cat(outs, 0)\n    if a is None:\n        return None\n    if isinstance(a, dict):\n        return {k: _cat([o[k] for o in outs]) for k in a}\n    if isinstance(a, (tuple, list)):\n        return type(a)(_cat(list(z)) for z in zip(*outs))\n    raise TypeError(type(a))\n\n\nif MODE == \'split\':\n    def _split_fwd(self, data, *a, **k):\n        if self.training and torch.is_grad_enabled():\n            h = data[0].shape[0] // 2\n            return _cat([ORIG_FWD(self, [t[:h] for t in data], *a, **k), ORIG_FWD(self, [t[h:] for t in data], *a, **k)])\n        return ORIG_FWD(self, data, *a, **k)\n    MX.XPertNet.forward = _split_fwd\n\n# ---- float64: their get_unimol_drug_feat hard-casts atom features with .float() (model_XPert.py:14) ------------------\nORIG_GUDF = MX.get_unimol_drug_feat\n\n\ndef _gudf_keep_dtype(x):\n    m = x[:, :, 0].long()\n    return x[:, :, 2:], x[:, :, 1].long(), (1.0 - m.unsqueeze(1).unsqueeze(2)) * -10000.0\n\n\ndev = torch.device(\'cuda:0\')\n\n\ndef build(dtype=torch.float32):\n    torch.manual_seed(0)\n    m = MX.XPertNet(args, config, dev, logger)\n    m.init_weights()\n    m.to(dev)\n    if dtype == torch.float64:\n        m.double()\n    params = dict(m.named_parameters())\n    assert all(n in params for n in FROZEN), \'a frozen name is not a parameter\'\n    for n in FROZEN:\n        params[n].requires_grad_(False)\n    for mod in m.modules():                        # dropout off on EVERY module, in train mode [013 C4, 80.2]\n        if isinstance(mod, torch.nn.Dropout):\n            mod.p = 0.0\n        if hasattr(mod, \'dropout_p\'):\n            mod.dropout_p = 0.0\n    return m\n\n\ninit_path, batch_path, scale_path = (os.path.join(SHARED, f) for f in (\'init.pt\', \'batch.pt\', \'fp16_scale.json\'))\nif MODE == \'single_ckpt\':\n    m0 = build()\n    torch.save({k: v.detach().cpu() for k, v in m0.state_dict().items()}, init_path)\n    torch.save(next(iter(tr)), batch_path)\n    del m0\ninit_sd = torch.load(init_path)\nbatch = torch.load(batch_path)\nassert batch[0].shape[0] == 128, \'not the recipe batch size\'\n\n\ndef capture(model, rows, epoch, precision, scale=2.0 ** 6):\n    model.load_state_dict(init_sd, strict=True)\n    b = [t[:rows] for t in batch]\n    if precision == \'f64\':\n        b = [t.double() if t.is_floating_point() else t for t in b]\n    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)\n    scaler = torch.cuda.amp.GradScaler(init_scale=scale)\n    saved_ac, saved_gudf = T.autocast, MX.get_unimol_drug_feat\n    if precision in (\'f64\', \'f32m\'):\n        T.autocast = contextlib.nullcontext\n    if precision == \'f64\':\n        MX.get_unimol_drug_feat = _gudf_keep_dtype\n    xpert_dp_patch.EXPECT_AUTOCAST = (precision == \'f16\') if MODE == \'dp\' else None\n    try:\n        losses = T.train(model, opt, [b], args, config, scaler=scaler, epoch=epoch)\n    finally:\n        T.autocast, MX.get_unimol_drug_feat = saved_ac, saved_gudf\n        xpert_dp_patch.EXPECT_AUTOCAST = None\n    keep = torch.float64 if precision == \'f64\' else torch.float32\n    g = {n: p.grad.detach().to(keep).cpu().clone() for n, p in model.named_parameters() if p.grad is not None}\n    none_set = sorted(n for n, p in model.named_parameters() if p.grad is None)\n    finite = all(bool(torch.isfinite(v).all()) for v in g.values())\n    for p in model.parameters():\n        p.grad = None\n    return [float(x) for x in losses], g, none_set, finite\n\n\nout = {}\nplan = []\nif MODE != \'split\':\n    plan += [(\'f64_e0\', ROWS_SMALL, 0, \'f64\'), (\'f64_e70\', ROWS_SMALL, 70, \'f64\')]\nplan += [(\'f32m_e0\', ROWS_SMALL, 0, \'f32m\'), (\'f32m_e70\', ROWS_SMALL, 70, \'f32m\')]\nif MODE != \'split\':\n    plan += [(\'f16_e0\', 128, 0, \'f16\'), (\'f16_e70\', 128, 70, \'f16\')]\n\nmodels = {}\nscales = json.load(open(scale_path)) if os.path.exists(scale_path) else {}\nfor tag, rows, epoch, prec in plan:\n    torch.backends.cuda.enable_mem_efficient_sdp(prec != \'f32m\')     # math backend for 81.1b; float64 forces it anyway\n    dtype = torch.float64 if prec == \'f64\' else torch.float32\n    if dtype not in models:\n        models = {dtype: build(dtype)}                                 # one model resident at a time\n        torch.cuda.empty_cache()\n    model = models[dtype]\n    before = dict(xpert_dp_patch.REPLICA_CALLS)\n    if prec == \'f16\' and MODE == \'single_ckpt\':\n        scale = 2.0 ** 6\n        while True:\n            losses, g, none_set, finite = capture(model, rows, epoch, prec, scale)\n            if finite or scale < 2.0 ** -12:\n                break\n            scale /= 2.0\n        scales[tag] = scale\n        json.dump(scales, open(scale_path, \'w\'))\n    else:\n        scale = scales.get(tag, 2.0 ** 6) if prec == \'f16\' else 2.0 ** 6\n        losses, g, none_set, finite = capture(model, rows, epoch, prec, scale)\n    report[\'tags\'][tag] = {\'losses\': losses, \'grads_finite\': finite, \'scale\': scale, \'rows\': rows,\n                           \'none_grad_set_equals_frozen\': none_set == FROZEN, \'none_grad_set\': none_set,\n                           \'replica_calls\': xpert_dp_patch.REPLICA_CALLS[\'n\'] - before[\'n\'],\n                           \'replica_calls_autocast_on\': xpert_dp_patch.REPLICA_CALLS[\'autocast_on\'] - before[\'autocast_on\']}\n    out[tag] = {\'grads\': g, \'losses\': losses}\n    torch.save(out, os.path.join(SHARED, MODE + \'_grads.pt\'))     # after every tag: a crash cannot lose what ran\n    json.dump(report, open(os.path.join(SHARED, MODE + \'_report.json\'), \'w\'), indent=1)\ntorch.backends.cuda.enable_mem_efficient_sdp(True)\nreport[\'plain_tensor_attributes\'] = xpert_dp_patch.plain_tensor_attributes(models[next(iter(models))])\n\n# Timing, dp only: the recipe itself -- published dropout, Adam, their GradScaler, loader included, frozen ten.\nif MODE == \'dp\':\n    del models\n    torch.cuda.empty_cache()\n    torch.manual_seed(0)\n    model = MX.XPertNet(args, config, dev, logger)\n    model.init_weights()\n    model.to(dev)\n    params = dict(model.named_parameters())\n    for n in FROZEN:\n        params[n].requires_grad_(False)\n    opt = torch.optim.Adam(model.parameters(), lr=config[\'train\'][\'train_lr\'], weight_decay=config[\'train\'][\'weight_decay\'])\n    scaler = torch.cuda.amp.GradScaler()\n    it = iter(tr)\n    T.train(model, opt, itertools.islice(it, 1), args, config, scaler=scaler, epoch=0)\n    for d in range(torch.cuda.device_count()):\n        torch.cuda.synchronize(d)\n        torch.cuda.reset_peak_memory_stats(d)\n    t0 = time.time()\n    T.train(model, opt, itertools.islice(it, 5), args, config, scaler=scaler, epoch=0)\n    for d in range(torch.cuda.device_count()):\n        torch.cuda.synchronize(d)\n    report[\'s_train_step\'] = (time.time() - t0) / 5\n    report[\'peak_gib_per_gpu\'] = [torch.cuda.max_memory_allocated(d) / 2 ** 30 for d in range(torch.cuda.device_count())]\n    report[\'train_batches\'], report[\'val_batches\'] = len(tr), len(val)\n\njson.dump(report, open(os.path.join(SHARED, MODE + \'_report.json\'), \'w\'), indent=1)\nprint(\'DPPROBE \' + json.dumps({k: report[k] for k in report if k != \'tags\'}), flush=True)\n'
open(os.path.join(X, 'xpert_dp_patch.py'), 'w', encoding='utf-8').write(DP_PATCH_SRC)
open(os.path.join(X, 'xpert_dp_probe.py'), 'w', encoding='utf-8').write(DP_PROBE_SRC)
# FULL-STATE RESUME [RESULTS 81.2, 81.5; reviews 012 C1/C5, 014 C1/C4]. Generated from model/v9/xpert_resume_patch.py.
RESUME_PATCH_SRC = '# -*- coding: utf-8 -*-\n"""Full-state checkpoint and resume for XPert\'s train_xpert.main(), applied at RUNTIME so their files stay verbatim.\n[RESULTS 81.2, 81.7, 84; reviews 012 C1/C5, 014 C1/C4, 015 C2-C4; 78.5]. Proven exact in v7 (81.3a/b).\n\nTheir own --resume_from reloads model weights only: the Adam load is commented out (train_xpert.py:487), the LambdaLR is\nrebuilt so its epoch count restarts at 0 (:460), EarlyStopping is constructed fresh (:524), and its key filter\n(:484-486) silently matches nothing on a prefixed checkpoint while logging success (review 012 C1). A multi-session run\nchained through it would not be their continuous training. This module makes it so:\n\n  * Registration. XPertNet, torch.optim.Adam, LambdaLR, GradScaler and EarlyStopping record their instance at\n    construction. Their main() builds exactly one of each per fold; a second one is refused.\n  * Freezing [RESULTS 80.5, Amendment A]. The parameters named in LINCS_FROZEN_PARAMS -- those that receive grad None in\n    the single-GPU recipe -- are set requires_grad=False at XPertNet construction, before the optimizer exists.\n  * Save, at every epoch boundary: after lr_scheduler.step() (train_xpert.py:545), which runs after stopper.step()\n    (:538), so every piece of state belongs to the same finished epoch. Written atomically (tmp + os.replace):\n      full_state.pt   model / Adam / GradScaler / LambdaLR state dicts; stopper counter, best_score, early_stop;\n                      torch CPU, CUDA (every device), numpy and python RNG states; the finished epoch index\n      best.pth        the bytes of the stopper\'s on-disk best checkpoint, if it exists\n      resume_from.pt  {\'epoch\', \'model_state_dict\'} -- what their --resume_from reads, and only to set start_epoch\n  * Restore, in the EarlyStopping.__init__ hook -- the last construction before their epoch loop, when every object\n    exists: a STRICT model load (the loaded key set must equal the model\'s), then Adam, GradScaler, LambdaLR, the\n    stopper\'s fields, the best checkpoint under THIS session\'s time-stamped folder, and the RNG states last, so the\n    next draw is the next epoch\'s shuffle. Their filtered load ran first and is overwritten. The first train() call\n    is asserted to be the epoch after the saved one.\n\nEnvironment: LINCS_STATE_DIR (write), LINCS_RESUME_DIR (read; absent on session 1), LINCS_FROZEN_PARAMS (JSON list).\nTest mode only [RESULTS 81.5, 81.3]: LINCS_STOP_AFTER_EPOCH (exit cleanly after saving that many finished epochs),\nLINCS_TRUNCATE_BATCHES (train and validate on the first N batches of each epoch), LINCS_DUMP_AFTER_RESTORE (write the\nlive state right after restore, for the exact round-trip test), LINCS_DETERMINISTIC=1\n(torch.use_deterministic_algorithms; the caller sets CUBLAS_WORKSPACE_CONFIG before CUDA initialises).\n\nProduction, RESULTS 84 [review 015]: LINCS_DEADLINE (unix time; after each save, stop cleanly if the slowest epoch\nso far would pass it -- so a session only ever ends at an epoch boundary, 015 C2), LINCS_HORIZON_EPOCHS (stop, FINAL,\nonce that many epochs have completed, 84.1), LINCS_EXPECT_TORCH / LINCS_EXPECT_CUDA (the stack must not move between\nsessions, 015 C3). Test only: LINCS_TEST_PATIENCE (the C4 chain test).\n\nReview 014 C4: if --resume_from is on the command line and the strict restore has not run by the first train() call,\nthe trainer fails hard -- a hook that silently failed to fire would reproduce exactly review 012 C1\'s failure.\n"""\nimport hashlib\nimport io\nimport itertools\nimport json\nimport os\nimport random\nimport shutil\nimport sys\nimport time\n\nimport numpy as np\nimport torch\n\nREG = {}\n_STATE = {\'first_train_checked\': False, \'expect_first_epoch\': None, \'restored\': False, \'resume_flag\': False,\n          \'epoch_t0\': None, \'epoch_durations\': []}\n\n\ndef _register(cls, key, after=None):\n    orig = cls.__init__\n    if getattr(orig, \'_lincs_resume\', False):\n        return\n\n    def init(self, *a, **k):\n        orig(self, *a, **k)\n        if key in REG and REG[key] is not self:\n            raise RuntimeError(\'xpert_resume_patch: a second %s was constructed; one per fold is expected\' % key)\n        REG[key] = self\n        if after is not None:\n            after(self)\n\n    init._lincs_resume = True\n    cls.__init__ = init\n\n\ndef _atomic_save(obj, path):\n    tmp = path + \'.tmp\'\n    torch.save(obj, tmp)\n    os.replace(tmp, path)\n\n\ndef _freeze(model):\n    names = json.loads(os.environ.get(\'LINCS_FROZEN_PARAMS\', \'[]\'))\n    params = dict(model.named_parameters())\n    missing = [n for n in names if n not in params]\n    if missing:\n        raise RuntimeError(\'xpert_resume_patch: frozen names not in the model: %r\' % missing)\n    for n in names:\n        params[n].requires_grad_(False)\n    print(\'LINCS froze %d unused parameters\' % len(names), flush=True)\n\n\ndef _sha1(path):\n    if not os.path.exists(path):\n        return None\n    h = hashlib.sha1()\n    with open(path, \'rb\') as f:\n        for chunk in iter(lambda: f.read(1 << 20), b\'\'):\n            h.update(chunk)\n    return h.hexdigest()\n\n\ndef collect_state():\n    """Everything restorable, in one structure -- used by the save AND by the post-restore dump, so the exact\n    round-trip test of RESULTS 81.3a compares like with like, field for field."""\n    m, opt, sch, sc, st = REG[\'model\'], REG[\'opt\'], REG[\'sched\'], REG.get(\'scaler\'), REG[\'stopper\']\n    return {\'epoch\': int(sch.last_epoch) - 1,       # LambdaLR counts the step just taken; the finished epoch is one less\n            \'model\': {k: v.detach().cpu() for k, v in m.state_dict().items()},\n            \'opt\': opt.state_dict(), \'sched\': sch.state_dict(),\n            \'scaler\': sc.state_dict() if sc is not None else None,\n            \'stopper\': {\'counter\': st.counter, \'best_score\': st.best_score, \'early_stop\': st.early_stop},\n            \'best_sha1\': _sha1(st.filepath),\n            \'rng\': {\'torch\': torch.get_rng_state(), \'cuda\': torch.cuda.get_rng_state_all(),\n                    \'numpy\': np.random.get_state(), \'python\': random.getstate()}}\n\n\ndef compare_states(saved, live, path=\'state\'):\n    """RESULTS 81.3a: every restorable field BITWISE equal. Returns the list of paths that differ (empty = pass).\n    Tensors must match in dtype, shape and every bit; numpy arrays likewise; containers element for element."""\n    bad = []\n    if torch.is_tensor(saved) or torch.is_tensor(live):\n        if not (torch.is_tensor(saved) and torch.is_tensor(live) and saved.dtype == live.dtype\n                and saved.shape == live.shape and torch.equal(saved.cpu(), live.cpu())):\n            bad.append(path)\n    elif isinstance(saved, np.ndarray) or isinstance(live, np.ndarray):\n        if not (isinstance(saved, np.ndarray) and isinstance(live, np.ndarray) and saved.dtype == live.dtype\n                and np.array_equal(saved, live)):\n            bad.append(path)\n    elif isinstance(saved, dict):\n        if not isinstance(live, dict) or set(saved) != set(live):\n            bad.append(path + \' (keys)\')\n        else:\n            for k in saved:\n                bad += compare_states(saved[k], live[k], \'%s.%s\' % (path, k))\n    elif isinstance(saved, (list, tuple)):\n        if not isinstance(live, (list, tuple)) or len(saved) != len(live):\n            bad.append(path + \' (length)\')\n        else:\n            for i, (x, y) in enumerate(zip(saved, live)):\n                bad += compare_states(x, y, \'%s[%d]\' % (path, i))\n    elif saved != live:\n        bad.append(path)\n    return bad\n\n\ndef save_state():\n    d = os.environ[\'LINCS_STATE_DIR\']\n    os.makedirs(d, exist_ok=True)\n    st = REG[\'stopper\']\n    state = collect_state()\n    epoch = state[\'epoch\']\n    _atomic_save(state, os.path.join(d, \'full_state.pt\'))\n    _atomic_save({\'epoch\': epoch, \'model_state_dict\': state[\'model\']}, os.path.join(d, \'resume_from.pt\'))\n    if os.path.exists(st.filepath):\n        tmp = os.path.join(d, \'best.pth.tmp\')\n        shutil.copyfile(st.filepath, tmp)\n        os.replace(tmp, os.path.join(d, \'best.pth\'))\n    now = time.time()\n    if _STATE[\'epoch_t0\'] is not None:\n        _STATE[\'epoch_durations\'].append(now - _STATE[\'epoch_t0\'])\n    _STATE[\'epoch_t0\'] = now\n    last_dur = _STATE[\'epoch_durations\'][-1] if _STATE[\'epoch_durations\'] else float(\'nan\')\n    print(\'LINCS STATE SAVED epoch %d | best_score %r | counter %d | epoch_s %.1f\'\n          % (epoch, st.best_score, st.counter, last_dur), flush=True)\n    horizon = os.environ.get(\'LINCS_HORIZON_EPOCHS\')\n    if horizon is not None and epoch + 1 >= int(horizon):\n        print(\'LINCS HORIZON REACHED %s\' % json.dumps({\'epoch\': epoch, \'best_score\': st.best_score,\n                                                       \'counter\': int(st.counter)}), flush=True)\n        raise SystemExit(0)\n    deadline = os.environ.get(\'LINCS_DEADLINE\')\n    if deadline is not None and _STATE[\'epoch_durations\'] and now + max(_STATE[\'epoch_durations\']) > float(deadline):\n        print(\'LINCS SESSION BOUNDARY %s\' % json.dumps({\'epoch\': epoch, \'best_score\': st.best_score,\n                                                         \'counter\': int(st.counter),\n                                                         \'max_epoch_s\': max(_STATE[\'epoch_durations\'])}), flush=True)\n        raise SystemExit(0)\n    stop_after = os.environ.get(\'LINCS_STOP_AFTER_EPOCH\')\n    if stop_after is not None and epoch + 1 >= int(stop_after):\n        print(\'LINCS_STOP_AFTER_EPOCH reached after epoch %d; exiting cleanly (test mode)\' % epoch, flush=True)\n        raise SystemExit(0)\n\n\ndef restore_state():\n    d = os.environ[\'LINCS_RESUME_DIR\']\n    # map_location=\'cpu\', independently of their load at train_xpert.py:483 [review 014 C4]; torch.set_rng_state needs a\n    # CPU ByteTensor.\n    st = torch.load(os.path.join(d, \'full_state.pt\'), map_location=\'cpu\', weights_only=False)\n    m = REG[\'model\']\n    want, have = set(st[\'model\']), set(m.state_dict())\n    if want != have:\n        raise RuntimeError(\'xpert_resume_patch: key sets differ (missing %d, unexpected %d) -- refusing [review 012 C1]\'\n                           % (len(have - want), len(want - have)))\n    m.load_state_dict(st[\'model\'], strict=True)\n    REG[\'opt\'].load_state_dict(st[\'opt\'])\n    REG[\'sched\'].load_state_dict(st[\'sched\'])\n    if st[\'scaler\'] is not None:\n        REG[\'scaler\'].load_state_dict(st[\'scaler\'])\n    stopper = REG[\'stopper\']\n    stopper.counter, stopper.best_score, stopper.early_stop = (st[\'stopper\'][\'counter\'], st[\'stopper\'][\'best_score\'],\n                                                               st[\'stopper\'][\'early_stop\'])\n    best = os.path.join(d, \'best.pth\')\n    # 015 C2: the three per-epoch files are each atomic but not atomic together; refuse an inconsistent set.\n    if _sha1(best) != st[\'best_sha1\']:\n        raise RuntimeError(\'xpert_resume_patch: best.pth sha1 %s != saved best_sha1 %s -- inconsistent state set\'\n                           % (_sha1(best), st[\'best_sha1\']))\n    rf = torch.load(os.path.join(d, \'resume_from.pt\'), map_location=\'cpu\', weights_only=False)\n    if int(rf[\'epoch\']) != int(st[\'epoch\']):\n        raise RuntimeError(\'xpert_resume_patch: resume_from epoch %s != full_state epoch %s\' % (rf[\'epoch\'], st[\'epoch\']))\n    if os.path.exists(best):\n        os.makedirs(os.path.dirname(stopper.filepath), exist_ok=True)\n        shutil.copyfile(best, stopper.filepath)\n    torch.set_rng_state(st[\'rng\'][\'torch\'])\n    torch.cuda.set_rng_state_all(st[\'rng\'][\'cuda\'])\n    np.random.set_state(st[\'rng\'][\'numpy\'])\n    random.setstate(st[\'rng\'][\'python\'])\n    _STATE[\'expect_first_epoch\'] = st[\'epoch\'] + 1\n    # 015 C3: the exact round-trip of RESULTS 81.3a, IN PROCESS, every session: any difference is fatal.\n    live = collect_state()\n    diff = compare_states(st, live)\n    if diff:\n        raise RuntimeError(\'xpert_resume_patch: restored state differs from the saved state in %r\' % diff[:10])\n    print(\'LINCS RESTORE ROUND-TRIP exact: every field bitwise equal\', flush=True)\n    _STATE[\'restored\'] = True\n    dump = os.environ.get(\'LINCS_DUMP_AFTER_RESTORE\')\n    if dump:\n        live = collect_state()\n        live[\'start_epoch_expected\'] = _STATE[\'expect_first_epoch\']\n        _atomic_save(live, dump)\n    print(\'LINCS RESUME restored epoch %d | best_score %r | counter %d | best checkpoint %s\'\n          % (st[\'epoch\'], stopper.best_score, stopper.counter, \'restored\' if os.path.exists(best) else \'none yet\'),\n          flush=True)\n\n\ndef apply():\n    """Install the hooks. Must run after their modules are importable and before train_xpert.main()."""\n    import models.model_XPert as MX\n    import utils as U\n    import train_xpert as T\n    from torch.optim.lr_scheduler import LambdaLR\n    from torch.cuda.amp import GradScaler\n\n    _STATE[\'resume_flag\'] = any(a == \'--resume_from\' or a.startswith(\'--resume_from=\') for a in sys.argv)\n    if _STATE[\'resume_flag\'] and not os.environ.get(\'LINCS_RESUME_DIR\'):\n        raise RuntimeError(\'xpert_resume_patch: --resume_from given without LINCS_RESUME_DIR; their filtered load \'\n                           \'must never be the only restore [review 012 C1, 014 C4]\')\n    for var, have in ((\'LINCS_EXPECT_TORCH\', torch.__version__), (\'LINCS_EXPECT_CUDA\', str(torch.version.cuda))):\n        want = os.environ.get(var)\n        if want is not None and want != have:\n            raise RuntimeError(\'xpert_resume_patch: %s is %s, this session has %s -- the stack moved [015 C3]\'\n                               % (var, want, have))\n    if os.environ.get(\'LINCS_DETERMINISTIC\') == \'1\':\n        torch.use_deterministic_algorithms(True)\n    _register(MX.XPertNet, \'model\', after=_freeze)\n    _register(torch.optim.Adam, \'opt\')\n    _register(GradScaler, \'scaler\')\n\n    # The save hook wraps LambdaLR.step on the CLASS and acts only for the registered instance. Wrapping it on the\n    # instance would put a local function into __dict__, which LambdaLR.state_dict() copies -- and torch.save cannot\n    # pickle it. The constructor\'s own initial step runs before registration, so it never saves.\n    _register(LambdaLR, \'sched\')\n    if not getattr(LambdaLR.step, \'_lincs_resume\', False):\n        orig_step = LambdaLR.step\n\n        def step(self, *a, **k):\n            r = orig_step(self, *a, **k)\n            if self is REG.get(\'sched\') and os.environ.get(\'LINCS_STATE_DIR\'):\n                save_state()\n            return r\n        step._lincs_resume = True\n        LambdaLR.step = step\n\n    def after_stopper(self):\n        tp = os.environ.get(\'LINCS_TEST_PATIENCE\')\n        if tp is not None:                       # TEST ONLY -- the kernel asserts it absent in production\n            self.patience = int(tp)\n        if os.environ.get(\'LINCS_RESUME_DIR\'):\n            restore_state()\n    _register(U.EarlyStopping, \'stopper\', after=after_stopper)   # T.EarlyStopping is this same class object\n\n    # EARLY-STOP MARKER [packet 015]. When THEIR stopper reports early stopping, the session is the FINAL one: print a\n    # marker line the kernel\'s watchdog acts on, and write it to the state dir, so a final session can never be\n    # mistaken for a resume point. Wrapped on the class, acting only for the registered instance.\n    if not getattr(U.EarlyStopping.step, \'_lincs_resume\', False):\n        orig_es_step = U.EarlyStopping.step\n\n        def es_step(self, score, model, current_epoch, optimizer, *a, **k):\n            stop = orig_es_step(self, score, model, current_epoch, optimizer, *a, **k)\n            if stop and self is REG.get(\'stopper\'):\n                info = {\'epoch\': int(current_epoch), \'best_score\': self.best_score, \'counter\': int(self.counter)}\n                d = os.environ.get(\'LINCS_STATE_DIR\')\n                if d:\n                    os.makedirs(d, exist_ok=True)\n                    with open(os.path.join(d, \'early_stop.json\'), \'w\') as f:\n                        json.dump(info, f)\n                print(\'LINCS EARLY STOP %s\' % json.dumps(info), flush=True)\n            return stop\n        es_step._lincs_resume = True\n        U.EarlyStopping.step = es_step\n\n    orig_train, orig_validate = T.train, getattr(T, \'validate\', None)\n    trunc = os.environ.get(\'LINCS_TRUNCATE_BATCHES\')\n\n    def train(*a, **k):\n        if not _STATE[\'first_train_checked\']:\n            _STATE[\'first_train_checked\'] = True\n            _STATE[\'epoch_t0\'] = time.time()\n            if _STATE[\'resume_flag\'] and not _STATE[\'restored\']:\n                raise RuntimeError(\'xpert_resume_patch: --resume_from is set but the strict restore did not run before \'\n                                   \'the first train() call -- refusing [review 014 C4]\')\n            exp = _STATE[\'expect_first_epoch\']\n            if exp is not None and k.get(\'epoch\') != exp:\n                raise RuntimeError(\'xpert_resume_patch: first epoch %r, expected %r\' % (k.get(\'epoch\'), exp))\n        if trunc:\n            a = list(a)\n            a[2] = itertools.islice(a[2], int(trunc))       # train(model, opt, dataloader, ...)\n        return orig_train(*a, **k)\n\n    def validate(*a, **k):\n        a = list(a)\n        a[1] = itertools.islice(a[1], int(trunc))           # validate(model, dataloader, ...)\n        return orig_validate(*a, **k)\n    T.train = train\n    if trunc and orig_validate is not None:\n        T.validate = validate\n    return sorted(REG.keys())\n'
open(os.path.join(X, 'xpert_resume_patch.py'), 'w', encoding='utf-8').write(RESUME_PATCH_SRC)
# The trainer wrapper for DataParallel + full-state resume: the PRODUCTION trainer [RESULTS 84].
open(os.path.join(X, 'run_train_dp.py'), 'w', encoding='utf-8').write(
    'import sys\n'
    'sys.argv = ["train_xpert.py"] + sys.argv[1:]\n'
    'import models.model_utils, models.model_XPert\n'
    'import xpert_dp_patch\n'
    'print("LINCS DataParallel devices:", xpert_dp_patch.apply(), flush=True)\n'
    'import xpert_resume_patch\n'
    'print("LINCS resume hooks:", xpert_resume_patch.apply(), flush=True)\n'
    'import train_xpert\n'
    'train_xpert.main()\n')
RECORD['activation_checkpointing'] = {'patched': ['Encoder', 'crossEncoder'], 'how': 'runtime wrapper; '
                                     'their files verbatim', 'proof': 'model/v9/prove_checkpoint_exact.py'}
RECORD['deviations'] = ['flash_attn shim on PYTHONPATH (their model imports it at module scope)',
                        'MyDataset: one tensor per drug instead of one per row -- values proven identical, '
                        'storage shared; the unpatched recipe needs ~24.7 GB of dataset RAM on this fold',
                        'activation checkpointing of Encoder/crossEncoder, applied at runtime by a wrapper -- '
                        'same batch 128, same loss, gradients within the noise floor; the unpatched recipe '
                        'needs ~14.95 GiB of activations on a 14.56 GiB T4',
                        'all_drugs_unimol_arr.npy rebuilt from the released npz (config names it, never released)',
                        'DataParallel over both T4s, a runtime patch inside XPertNet.forward: loss on the gathered '
                        'batch of 128; float64 gradients equal to one GPU to 5e-16 [RESULTS 81.7]',
                        'the ten parameters their loss never uses are frozen (requires_grad=False): a no-op on one '
                        'GPU, and it stops DataParallel zero-gradients from letting weight decay move them [80.5]',
                        'full-state checkpoint and resume hooks across Kaggle sessions (model, Adam, GradScaler, '
                        'LambdaLR, stopper, RNG), exact bitwise [81.7]; their lossy --resume_from sets start_epoch only',
                        'empty __init__.py in datasets/ and models/ so their packages are not shadowed by '
                        "the image's HuggingFace `datasets`"]
log('staged', SRC, '->', X)

# --------------------------------------------------------------------------------------------------------
# GUARD A -- the split is the one we think, and the val=test fallback WILL fire.
# The expected counts are measured on the released h5ad [RESULTS 69.1], not assumed. A different or stale
# subset would silently change the comparison's rows.
# --------------------------------------------------------------------------------------------------------
import h5py  # noqa: E402
import numpy as np  # noqa: E402

with h5py.File(H5AD, 'r') as f:
    d = f['obs'][FOLD]
    if isinstance(d, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in d['categories'][:]]
        codes = d['codes'][:]
        counts = {c: int((codes == i).sum()) for i, c in enumerate(cats)}
    else:
        v = [x.decode() if isinstance(x, bytes) else str(x) for x in d[:]]
        u, n = np.unique(v, return_counts=True)
        counts = dict(zip(u.tolist(), [int(c) for c in n]))
if counts != {'train': 47509, 'test': 21321}:
    fatal('%s levels are %s, expected {train: 47509, test: 21321}.' % (FOLD, counts))
RECORD['guards']['split'] = counts
RECORD['disclosed_asymmetry'] = ('no valid level -> utils.py:133 sets val_data = test_data -> early stopping '
                                 'and checkpoint selection monitor TEST loss4')
log('GUARD A', counts, '-- no valid level: their early stopping will monitor TEST loss (disclosed)')

# --------------------------------------------------------------------------------------------------------
# GUARD B -- the dense unimol array their config names, rebuilt from the npz they released. The builder
# refuses on an uncovered reachable pert_idx, duplicate idx, non-finite or all-zero features: a dead drug
# input trains to convergence and reports plausible metrics [RESULTS 32].
# --------------------------------------------------------------------------------------------------------
ARR = os.path.join(X, 'processed_data', 'all_drugs_unimol_arr.npy')
b = subprocess.run([sys.executable, os.path.join(V9, 'build_xpert_unimol_arr.py'), '--npz', NPZ,
                    '--h5ad', H5AD, '--out', ARR], capture_output=True, text=True)
print(b.stdout[-2000:], b.stderr[-2000:])
if b.returncode != 0 or not os.path.exists(ARR):
    fatal('unimol array build failed.')
RECORD['guards']['unimol_array'] = b.stdout.strip().splitlines()[-3:]
log('GUARD B unimol array built:', ARR)

# --------------------------------------------------------------------------------------------------------
# GUARD C -- flash_attn resolves to OUR shim, and the shim's optional mask is OFF (option A).
# Checked in a subprocess with exactly the PYTHONPATH the trainer will get, because that is the import that
# matters, not this process's.
# --------------------------------------------------------------------------------------------------------
SHIMS = os.path.join(V9, '_shims')
ENV = dict(os.environ, PYTHONPATH=SHIMS + os.pathsep + X, PYTHONUNBUFFERED='1')
probe = ('import flash_attn, flash_attn.flash_attn_interface as s, os; '
         'print(os.path.abspath(s.__file__)); print(s.KEY_PAD_MASK is None)')
r = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, env=ENV, cwd=X)
out = r.stdout.strip().splitlines()
if r.returncode != 0 or len(out) != 2:
    fatal('flash_attn import probe failed: %s' % r.stderr[-800:])
if not os.path.abspath(out[0]).startswith(os.path.abspath(SHIMS)):
    fatal('flash_attn resolved to %s, not our shim. A real package would behave identically only if it is '
          'the same kernel; refusing rather than guessing.' % out[0])
if out[1] != 'True':
    fatal('shim KEY_PAD_MASK is not None -- the run would be MASKED (option B), which RESULTS 70.1 showed is '
          'a different model.')
RECORD['guards']['attention'] = 'our shim, unmasked (option A)'
log('GUARD C flash_attn ->', out[0], '| unmasked')

# --------------------------------------------------------------------------------------------------------
# GUARD D -- every module of THEIRS that the trainer imports must resolve INSIDE the staged copy.
# This is the guard version 1 lacked. It imports exactly what train_xpert.py and utils.py import, in a
# subprocess with the trainer's PYTHONPATH and cwd, and refuses if any of them comes from anywhere else --
# shadowing is silent when the impostor happens to have a matching attribute, and loud only by luck.
# --------------------------------------------------------------------------------------------------------
probe = ('import os, importlib\n'
         'for m in ("datasets.MyDataset", "models.model_XPert", "models.model_utils", "metrics", "utils"):\n'
         '    mod = importlib.import_module(m)\n'
         '    print(m, os.path.abspath(mod.__file__))\n')
r = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, env=ENV, cwd=X)
if r.returncode != 0:
    fatal('GUARD D: importing their modules failed: %s' % r.stderr[-1500:])
resolved = dict(line.split(' ', 1) for line in r.stdout.strip().splitlines())
bad = {m: f for m, f in resolved.items() if not os.path.abspath(f).startswith(os.path.abspath(X))}
if len(resolved) != 5 or bad:
    fatal('GUARD D: modules resolved outside the staged copy (shadowed): %s' % (bad or resolved))
RECORD['guards']['their_modules'] = 'datasets.MyDataset, models.*, metrics, utils all resolve inside the staged copy'
log('GUARD D their modules resolve inside', X)

# --------------------------------------------------------------------------------------------------------
# 3. TRAIN, their recipe, under a wall-clock guard. Their stopper writes the best checkpoint to disk on
#    every improvement (utils.py save_checkpoint), so a guard kill cannot lose it.
#    NOTE: their boolean flags use argparse `type=bool`, so passing the string "False" ENABLES them. Only
#    --output_profile is passed, and only as True.
# --------------------------------------------------------------------------------------------------------
# THE PUBLISHED INVOCATION, not argparse defaults. Version 2 of this kernel used defaults and crashed on
# `configs/config.yaml`, which their release does not contain. Their own scripts/train.sh:15 is, for mdmt:
#   python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol
#          --nfold split_cold_drug_1,split_cold_cell_1,split_1 --dataset l1000_mdmt
#          --use_gradscaler True --include_cell_idx True
# and the README gives the same --config and --use_gradscaler. v2 differed in THREE flags, not one, and
# --include_cell_idx adds cls_token + class_fc: trained without it, the checkpoint could not have been loaded
# by our harness (strict=True) -- a failure that would have arrived AFTER eight hours of training. Our own
# harness's docstring (xpert_native_eval.py:21) already said this flag is non-default; it was not read.
# Only the fold list is reduced, to split_cold_cell_1: their loop builds a fresh model, init_weights() and
# optimizer PER FOLD (train_xpert.py:425-455), so the folds are independent. Their seed is set once at
# :402, before that loop, so our fold starts from a fresh seed-2024 state rather than the state their
# split_cold_drug_1 run left -- a seed difference, not a recipe difference. Disclosed.
PUBLISHED = {'model': 'XPert', 'config': 'config_l1000', 'drug_feat': 'unimol', 'dataset': 'l1000_mdmt',
             'use_gradscaler': 'True', 'include_cell_idx': 'True'}
cmd = [sys.executable, '-u', 'run_train_ckpt.py', '--mode', 'train', '--nfold', FOLD, '--device', 'cuda:0',
       '--output_profile', 'True']
for k, v in PUBLISHED.items():
    cmd += ['--' + k, v]
# What the trainer must PRINT in its args block. Checked against the executed namespace, not our argv:
# their booleans are argparse type=bool, so bool("False") is True and intent and effect can diverge.
EXPECT_ARGS = {'mode': 'train', 'nfold': FOLD, 'model': 'XPert', 'config': 'config_l1000',
               'drug_feat': 'unimol', 'dataset': 'l1000_mdmt', 'seed': '2024', 'use_gradscaler': 'True',
               'include_cell_idx': 'True', 'output_attention': 'False', 'output_profile': 'True'}
# --------------------------------------------------------------------------------------------------------
# GUARD F -- one real batch through THEIR loader and ONE forward pass, in the trainer's environment.
# Launches v1 and v2 each failed one stage later than the last (import shadowing, then the config path).
# The next stage is data loading and the model, and it carries a known hazard: their MyDataset indexes
# raw_data_items['col'][idx] with an integer idx, which pandas treats as positional only by a deprecated
# fallback -- our own harness re-indexes obs to avoid it. This probe uses their arg_parse(), their config,
# their load_dataloader() and XPertNet exactly as train_xpert.py does, so a failure here is the failure the
# trainer would hit, found in two minutes instead of after the first epoch.
# --------------------------------------------------------------------------------------------------------
probe_argv = ['train_xpert.py', '--mode', 'train', '--nfold', FOLD, '--device', 'cuda:0']
for k, v in PUBLISHED.items():
    probe_argv += ['--' + k, v]
probe = ('import sys, logging, yaml, torch, pandas\n'
         'sys.argv = %r\n'
         'import train_xpert as T\n'
         'from utils import load_dataloader\n'
         'from models.model_XPert import XPertNet\n'
         'args = T.arg_parse()\n'
         'config = yaml.safe_load(open("configs/%%s.yaml" %% args.config))\n'
         'logger = logging.getLogger("probe")\n'
         'tr, val, te, adata = load_dataloader(args, config, logger, nfold=%r, return_rawdata=True)\n'
         'batch = next(iter(tr))\n'
         'dev = torch.device("cuda:0")\n'
         'model = XPertNet(args, config, dev, logger)\n'
         'model.init_weights(); model.to(dev)\n'
         'import time, json\n'
         'import xpert_ckpt_patch\n'
         'xpert_ckpt_patch.apply()\n'
         'model.train()\n'
         'opt = torch.optim.Adam(model.parameters(), lr=config["train"]["train_lr"],\n'
         '                       weight_decay=config["train"]["weight_decay"])\n'
         'scaler = torch.cuda.amp.GradScaler()\n'
         'def train_step(b):\n'
         '    opt.zero_grad(set_to_none=True)\n'
         '    with torch.cuda.amp.autocast():\n'
         '        o = model(b)\n'
         '        l = sum(x.float().mean() for x in o[:3])\n'
         '    scaler.scale(l).backward(); scaler.step(opt); scaler.update()\n'
         '    return o\n'
         'torch.cuda.reset_peak_memory_stats()\n'
         'out = train_step(batch)\n'
         'it = iter(tr)\n'
         'train_step(next(it))\n'
         'torch.cuda.synchronize(); t0 = time.time()\n'
         'for _ in range(5):\n'
         '    train_step(next(it))\n'
         'torch.cuda.synchronize(); s_train = (time.time() - t0) / 5\n'
         'peak = torch.cuda.max_memory_allocated() / 2 ** 30\n'
         'model.eval(); vit = iter(val)\n'
         'with torch.no_grad(), torch.cuda.amp.autocast():\n'
         '    model(next(vit))\n'
         '    torch.cuda.synchronize(); t1 = time.time()\n'
         '    for _ in range(5):\n'
         '        model(next(vit))\n'
         '    torch.cuda.synchronize(); s_val = (time.time() - t1) / 5\n'
         'print("TIMING " + json.dumps({"s_train_step": s_train, "s_val_step": s_val, "peak_gib": peak,\n'
         '      "train_batches": len(tr), "val_batches": len(val)}), flush=True)\n'
         'print("GUARD F training step at batch", batch[0].shape[0], "peak GPU GiB", round(peak, 2))\n'
         'assert batch[0].shape[0] == 128, "not the recipe batch size"\n'
         'assert peak < 13.0, "training step peak %%.2f GiB leaves no headroom on a T4" %% peak\n'
         'ok = all(torch.isfinite(o).all().item() for o in out[:3])\n'
         'print("pandas", pandas.__version__, "| train batches", len(tr), "| val rows", len(val.dataset),\n'
         '      "| test rows", len(te.dataset), "| out", [tuple(o.shape) for o in out[:3]], "| finite", ok)\n'
         'assert ok, "non-finite forward output"\n'
         'assert len(val.dataset) == len(te.dataset), "val is not the test set: the disclosed fallback did not fire"\n'
         % (probe_argv, FOLD))
r = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, env=ENV, cwd=X)
print(r.stdout[-1500:])
if r.returncode != 0:
    # v3's GUARD F recorded THAT it failed and not why: no return code, no stderr in the record, and the log came
    # back through a client that could not encode tqdm's block characters. A negative return code is a signal --
    # -9 is the OS killer, which is what v3's silent death was. Put both in the record itself.
    tb = [l for l in r.stderr.splitlines() if l.strip() and '%|' not in l and 'it/s]' not in l]
    RECORD['guard_f_failure'] = {'returncode': r.returncode,
                                 'killed_by_signal': r.returncode < 0,
                                 'stderr_tail_without_progress_bars': tb[-40:]}
    print(r.stderr[-4000:])
    fatal('GUARD F: one batch through their loader and one forward pass failed (returncode %d%s); the trainer '
          'would too.' % (r.returncode, ', killed by signal -- out of memory' if r.returncode == -9 else ''))
RECORD['guards']['one_batch_forward'] = r.stdout.strip().splitlines()[-1][:400]
log('GUARD F one real batch + forward OK:', RECORD['guards']['one_batch_forward'])
_t = [l for l in r.stdout.splitlines() if l.startswith('TIMING ')]
if len(_t) != 1:
    fatal('GUARD F did not report its steady-state timing.')
TIMING = json.loads(_t[0][len('TIMING '):])
setup_s = time.time() - T0
epoch_s = TIMING['train_batches'] * TIMING['s_train_step'] + TIMING['val_batches'] * TIMING['s_val_step']
TIMING.update({'setup_s': round(setup_s, 1), 'epoch_s_projected': round(epoch_s, 1),
               'epochs_within_budget_projected': round((BUDGET_H * 3600 - setup_s - 900) / epoch_s, 1),
               'budget_h': BUDGET_H, 'note': 'steady state, checkpointed, optimizer step included; 900 s reserved '
                                            'for prediction and artefacts'})
RECORD['timing'] = TIMING
log('TIMING', TIMING)
import torch
FROZEN = ['attnEncoder_trt.crossEncoders.0.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.0.LayerNorm.gamma',
          'attnEncoder_trt.crossEncoders.1.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.1.LayerNorm.gamma',
          'cell_emb.linear.bias', 'cell_emb.linear.weight', 'ctl_fc.0.bias', 'ctl_fc.0.weight',
          'ctl_fc.3.bias', 'ctl_fc.3.weight']       # RESULTS 80.5 Amendment A; the grad-None set in every v7 mode

if MEASURE_ONLY:
    RECORD['stopped_by'] = 'measure_only'
    RECORD['host_memory'] = mem_summary()
    json.dump(RECORD, open(os.path.join(W, 'run_record.json'), 'w'), indent=2)
    log('MEASURE_ONLY: stopping before training. Projection recorded; the full run is decided from it.')
    shutil.rmtree(X, ignore_errors=True)      # includes the 2.24 GB rebuilt array; keeps the output small
    shutil.rmtree(V9, ignore_errors=True)
    raise SystemExit(0)

RECORD['train_cmd'] = ' '.join(cmd)
RECORD['published_command_source'] = ('scripts/train.sh:15 (mdmt) and README; fold list reduced to %s; '
                                      'folds independent per train_xpert.py:425-455' % FOLD)
RECORD['seed_note'] = ('seed 2024 set once at train_xpert.py:402 before the fold loop, so this fold starts '
                       'from a fresh seed-2024 state, not the state their 3-fold sequential run would reach')
# ========================================================================================================
# O2 PRODUCTION SESSION [RESULTS 84; review 015]
#   FINAL terminations are exactly two (84.1): their early stopping (the resume patch prints LINCS EARLY STOP), or
#   the 297-epoch horizon (LINCS HORIZON REACHED). A session otherwise ends ONLY at an epoch boundary, cleanly
#   (LINCS SESSION BOUNDARY), before the deadline. Anything else -- crash, watchdog backstop, quota kill -- is
#   INCOMPLETE and is never read through 71.7. Between sessions only timing, state integrity and quota may inform a
#   decision; never the logged loss.
# ========================================================================================================
import glob
import hashlib
import threading
assert not MEASURE_ONLY
STATE = os.path.join(W, 'state')
os.makedirs(STATE, exist_ok=True)
TEST_VARS = ('LINCS_STOP_AFTER_EPOCH', 'LINCS_TRUNCATE_BATCHES', 'LINCS_DUMP_AFTER_RESTORE', 'LINCS_DETERMINISTIC',
             'LINCS_TEST_PATIENCE')
STACK = {'torch': torch.__version__, 'cuda': str(torch.version.cuda)}
RECORD['session'] = {'index': SESSION, 'horizon_epochs': HORIZON_EPOCHS, 'stack': STACK, 'prev': PREV}
cmd_prod = [sys.executable, '-u', 'run_train_dp.py'] + cmd[3:]
DEADLINE = T0 + BUDGET_H * 3600
ENV_PROD = {k: v for k, v in ENV.items() if k not in TEST_VARS}
ENV_PROD.update(LINCS_STATE_DIR=STATE, LINCS_FROZEN_PARAMS=json.dumps(FROZEN),
                LINCS_HORIZON_EPOCHS=str(HORIZON_EPOCHS), LINCS_DEADLINE=str(DEADLINE),
                LINCS_EXPECT_TORCH=(PREV or STACK)['torch'], LINCS_EXPECT_CUDA=(PREV or STACK)['cuda'])


def _sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


STATE_FILES = ('full_state.pt', 'resume_from.pt', 'best.pth')
if SESSION > 1:
    # 015 C3: the attached state must be byte-for-byte the state session k-1 handed off, whose sha1s are literals here.
    if STACK != {'torch': PREV['torch'], 'cuda': PREV['cuda']}:
        fatal('the stack moved between sessions: %s -> %s. Stop and return [015 C3].' % (PREV, STACK))
    fs = glob.glob('/kaggle/input/**/full_state.pt', recursive=True)
    if len(fs) != 1:
        fatal('expected exactly one attached full_state.pt, found %d' % len(fs))
    RESUME_DIR = os.path.dirname(fs[0])
    got = {f: _sha1(os.path.join(RESUME_DIR, f)) for f in STATE_FILES}
    if got != PREV['sha1']:
        fatal('attached state sha1s %s != session %d handoff literals %s' % (got, SESSION - 1, PREV['sha1']))
    ENV_PROD['LINCS_RESUME_DIR'] = RESUME_DIR
    cmd_prod = cmd_prod + ['--resume_from', os.path.join(RESUME_DIR, 'resume_from.pt')]
    RECORD['session']['resume_dir'] = RESUME_DIR
    RECORD['session']['attached_sha1_verified'] = True
    log('SESSION %d resumes from epoch %d; attached state verified against the git-committed handoff'
        % (SESSION, PREV['final_epoch']))


def run_trainer(argv, env, tag, backstop):
    """Run their main() through run_train_dp.py, parse the marker lines, and return what ended it. GUARD E (executed
    args) applies to every run."""
    rec = {'tag': tag, 'outcome': None, 'marker': None, 'saved': [], 'epoch_s': [], 'last_epoch_index': None,
           'last_counter_logged': None}
    logf = os.path.join(W, 'train_%s.log' % tag)
    with open(logf, 'w') as lf:
        p = subprocess.Popen(argv, cwd=X, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        # The line loop can only check the clock when the trainer prints; a SILENT hang would run into Kaggle's
        # hard limit, whose kill discards /kaggle/working -- the state included. A timer kills it regardless.
        killer = threading.Timer(max(1.0, backstop - time.time()), p.kill)
        killer.daemon = True
        killer.start()
        in_args, seen_args, args_checked = False, {}, False
        for line in p.stdout:
            lf.write(line)
            if not args_checked:
                if '---------args-----------' in line:
                    in_args = True
                    continue
                if in_args:
                    if ':' in line:
                        k, v = line.split(':', 1)
                        seen_args[k.strip()] = v.strip()
                    elif not line.strip() and seen_args:
                        in_args, args_checked = False, True
                        bad = {k: (seen_args.get(k), v) for k, v in EXPECT_ARGS.items() if seen_args.get(k) != v}
                        rec['executed_args'] = seen_args
                        if bad:
                            p.terminate()
                            fatal('GUARD E (%s): executed args differ from the published recipe: %s' % (tag, bad))
            if line.startswith('LINCS STATE SAVED'):
                rec['saved'].append(line.strip())
                try:
                    rec['epoch_s'].append(float(line.rsplit('epoch_s', 1)[1]))
                except (IndexError, ValueError):
                    pass
            if 'Valid Total Loss' in line:
                try:
                    rec['last_epoch_index'] = int(line.split('Epoch ')[1].split(',')[0])
                except (IndexError, ValueError):
                    pass
                n = len(rec['saved']) + 1
                if n <= 3 or n % 10 == 0:
                    log(tag, 'epoch', rec['last_epoch_index'], '|', line.strip()[-120:])
            if 'EarlyStopping counter:' in line:
                try:
                    rec['last_counter_logged'] = int(line.split('EarlyStopping counter:')[1].split('out of')[0])
                except (IndexError, ValueError):
                    pass
            for key, outcome in (('LINCS EARLY STOP ', 'early_stop'), ('LINCS HORIZON REACHED ', 'horizon'),
                                 ('LINCS SESSION BOUNDARY ', 'boundary')):
                if line.startswith(key):
                    rec['outcome'], rec['marker'] = outcome, json.loads(line[len(key):])
                    log(tag, line.strip())
            if 'Traceback' in line or 'Error' in line:
                log(tag, line.strip()[-300:])
            if rec['outcome'] == 'early_stop':
                # 015 ask 2: terminate at the marker; their post-loop test pass is never read (decisions_locked).
                p.terminate()
                break
            if time.time() > backstop:
                rec['outcome'] = 'watchdog_backstop'
                log(tag, 'WATCHDOG BACKSTOP fired: the epoch-boundary stop did not act in time.')
                p.terminate()
                break
        try:
            rc = p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
    killer.cancel()
    if rc is not None and rc < 0 and time.time() >= backstop and rec['outcome'] is None:
        rec['outcome'] = 'watchdog_backstop'
    rec['returncode'] = rc
    if rec['outcome'] is None:
        # C4: rc 0 without a marker means their main() ran to its end unseen -- never a silent "finished".
        rec['outcome'] = 'exited_without_marker' if rc == 0 else ('killed_by_signal' if rc < 0 else 'crashed')
    return rec


def best_checkpoint():
    cks = glob.glob(os.path.join(X, 'experiment', '**', '%s_fold_early_stop.pth' % FOLD), recursive=True)
    if not cks:
        fatal('no best checkpoint written by their stopper')
    return max(cks, key=os.path.getmtime)


def finalize(rec, patience, tag):
    """The two FINAL terminations only (84.1). Convergence record, 71.7, and our row-indexed prediction."""
    assert rec['outcome'] in ('early_stop', 'horizon'), rec['outcome']
    ck_path = best_checkpoint()
    best_epoch = int(torch.load(ck_path, map_location='cpu', weights_only=False).get('epoch', -1))
    last_epoch_index = int(rec['marker']['epoch'])          # from the MARKER (015 ask 2), not from the state dir
    counter_at_end = last_epoch_index - best_epoch
    out = {'stopped_by': 'finished' if rec['outcome'] == 'early_stop' else 'horizon',
           'best_checkpoint': ck_path, 'best_epoch': best_epoch, 'last_epoch_index': last_epoch_index,
           'counter_at_end': counter_at_end, 'marker': rec['marker'], 'patience': patience,
           'best_selected_before_init_epoch_70': best_epoch < 70}
    if rec['outcome'] == 'early_stop':
        # 84.1(4): by construction, and verified against utils.py's counter >= patience
        if not (counter_at_end == patience == int(rec['marker']['counter'])):
            fatal('%s: counter_at_end %d, patience %d, marker counter %s disagree' % (tag, counter_at_end, patience,
                                                                                    rec['marker']['counter']))
        out['admissible_for_v9_win'] = True
    else:
        out['admissible_for_v9_win'] = counter_at_end >= 45           # 71.7 applied to the final horizon stop
    prof_path = os.path.join(W, 'xpert_trained_%s_%s_test_profile.npy' % (FOLD, tag))
    env2 = dict(ENV, XPERT_DIR=X, XPERT_CKPT=ck_path)
    pr = subprocess.run([sys.executable, '-u', os.path.join(V9, 'xpert_native_eval.py'), '--nfold', FOLD,
                         '--rows', 'test', '--device', 'cuda', '--batch', '128', '--out', prof_path],
                        capture_output=True, text=True, env=env2)
    if pr.returncode != 0 or not os.path.exists(prof_path):
        print(pr.stdout[-3000:], pr.stderr[-3000:])
        fatal('%s: prediction with the best checkpoint failed' % tag)
    prof = np.load(prof_path, allow_pickle=True).item()
    if 'row_index' not in prof or len(prof['row_index']) != 21321:
        fatal('%s: profile has %s rows / row_index present=%s' % (tag, len(prof.get('y_pred', [])), 'row_index' in prof))
    out['profile'] = {'path': prof_path, 'n': int(len(prof['row_index']))}
    return out


# ---- C4 (015): the one new path, end to end, in test mode, BEFORE real training -- session 1 only ------------------
if SESSION == 1:
    chain_state = os.path.join(W, 'chain_state')
    env_c = dict(ENV_PROD, LINCS_STATE_DIR=chain_state, LINCS_TRUNCATE_BATCHES='5', LINCS_TEST_PATIENCE='1',
                 LINCS_STOP_AFTER_EPOCH='25')
    for k in ('LINCS_DEADLINE', 'LINCS_HORIZON_EPOCHS'):
        env_c.pop(k)
    before = set(glob.glob(os.path.join(X, 'experiment', '*', '*')))
    rc_chain = run_trainer(cmd_prod, env_c, 'chaintest', backstop=time.time() + 1800)
    if rc_chain['outcome'] != 'early_stop':
        RECORD['chain_test'] = rc_chain
        fatal('C4 chain test: expected an early stop with patience 1 in <= 25 truncated epochs, got %s'
              % rc_chain['outcome'])
    ft = finalize(rc_chain, patience=1, tag='chaintest')
    RECORD['chain_test'] = {'trainer': {k: rc_chain[k] for k in ('outcome', 'marker', 'returncode', 'last_epoch_index')},
                            'finalize': ft, 'PASS': True}
    log('C4 chain test PASSED: marker -> termination -> counter_at_end == patience -> 21321-row prediction')
    # remove every chain-test artefact so the real run's checkpoint can never be confused with it
    for d in set(glob.glob(os.path.join(X, 'experiment', '*', '*'))) - before:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(chain_state, ignore_errors=True)
    os.remove(ft['profile']['path'])

# ---- the real training session ----------------------------------------------------------------------------------
for k in TEST_VARS:
    assert k not in ENV_PROD, 'test-mode variable %s in the production environment' % k
log('TRAIN (session %d):' % SESSION, ' '.join(cmd_prod))
res = run_trainer(cmd_prod, ENV_PROD, 'session%d' % SESSION, backstop=DEADLINE + 1800)
RECORD['host_memory'] = mem_summary()
sess = {'index': SESSION, 'outcome': res['outcome'], 'returncode': res['returncode'], 'marker': res['marker'],
        'epochs_this_session': len(res['saved']), 'first_saved': res['saved'][:1], 'last_saved': res['saved'][-1:],
        'epoch_s': res['epoch_s'], 'wall_h': round((time.time() - T0) / 3600, 3),
        'executed_args': res.get('executed_args')}
if SESSION == 1 and len(res['epoch_s']) >= 2:
    first2 = sum(res['epoch_s'][:2]) / 2
    sess['amendment_E'] = {'first_two_mean_s': round(first2, 1), 'projection_s': 482.2,
                           'reprice_before_session_2': first2 > 1.25 * 482.2}
RECORD['session_log'] = sess
log('session %d ended:' % SESSION, res['outcome'], '| epochs this session', len(res['saved']))

if res['outcome'] in ('early_stop', 'horizon'):
    fin = finalize(res, patience=50, tag='final')
    RECORD.update(fin)
    shutil.copy(fin['best_checkpoint'], os.path.join(W, 'xpert_trained_%s.pth' % FOLD))
    RECORD['framing'] = ('XPert trained to its published recipe on %s. NOT a reproduction of their cold-cell '
                         'run: an independent draw of the recipe.' % FOLD)
    log('FINAL (%s): best epoch %d, last %d, counter %d, admissible %s' % (fin['stopped_by'], fin['best_epoch'],
        fin['last_epoch_index'], fin['counter_at_end'], fin['admissible_for_v9_win']))
elif res['outcome'] == 'boundary':
    RECORD['stopped_by'] = 'session_boundary'
    RECORD['handoff'] = {'sha1': {f: _sha1(os.path.join(STATE, f)) for f in STATE_FILES},
                         'final_epoch': int(res['marker']['epoch']), 'torch': STACK['torch'], 'cuda': STACK['cuda']}
    log('HANDOFF for session %d:' % (SESSION + 1), RECORD['handoff'])
else:
    # crash, backstop, exit without a marker: INCOMPLETE (84.1). The state dir holds the last completed boundary.
    RECORD['stopped_by'] = 'incomplete_' + res['outcome']
    if os.path.exists(os.path.join(STATE, 'full_state.pt')):
        RECORD['handoff_candidate'] = {'sha1': {f: _sha1(os.path.join(STATE, f)) for f in STATE_FILES
                                                if os.path.exists(os.path.join(STATE, f))}}
    os.system('tail -60 %s' % os.path.join(W, 'train_session%d.log' % SESSION))

# The staged copy is several GB and rebuildable; everything needed is in the state dir, the logs and (final
# session) the copied best checkpoint and the profile.
shutil.rmtree(X, ignore_errors=True)
for junk in (ARR,):                       # 2.24 GB derived file; rebuildable, not an output
    try:
        os.remove(junk)
    except OSError:
        pass
RECORD['total_hours'] = round((time.time() - T0) / 3600, 3)
json.dump(RECORD, open(os.path.join(W, 'run_record.json'), 'w'), indent=2, default=str)
log('DONE', json.dumps({k: RECORD.get(k) for k in ('stopped_by', 'total_hours')}))
if RECORD['stopped_by'].startswith('incomplete'):
    raise SystemExit(1)
