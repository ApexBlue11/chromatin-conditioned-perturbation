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
MEASURE_ONLY = True
# v6: MEASURE_ONLY also runs GUARD G, the DataParallel proof and timing of RESULTS 80, before stopping.
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
# model/v9/xpert_dp_probe.py. Used ONLY by GUARD G in this version: the trainer command is unchanged.
DP_PATCH_SRC = '# -*- coding: utf-8 -*-\n"""DataParallel for XPert over every visible GPU, applied at RUNTIME inside XPertNet.forward so their files stay\nverbatim. [RESULTS 78, review 012]\n\nWhy inside forward rather than wrapping the model in nn.DataParallel: the object their train_xpert.py holds stays an\nXPertNet. So state_dict() keys carry NO \'module.\' prefix -- review 012 C1 showed their --resume_from filter would match\nno key of a prefixed checkpoint and silently restart from random weights while logging success -- and every attribute\ntheir code touches keeps working.\n\nWhy it is exact for their recipe, where gradient accumulation was not [RESULTS 77.2]: torch.nn.parallel.data_parallel\nscatters the batch, runs one replica per GPU, and GATHERS the outputs to GPU 0 before their train() computes the loss,\nso batch_weighted_loss\'s sqrt(loss / num_samples) terms see all 128 samples exactly as on one GPU. Gradients from the\nreplicas are summed into the original parameters. XPert has LayerNorm only, no BatchNorm, so no statistic depends on\nthe per-replica batch. Dropout masks differ per replica: i.i.d. draws of the same Bernoulli, a different realisation,\nas a different seed is [review 012 ask 3].\n\nTwo things in their forward are pinned to one device and are localised on each replica:\n  * `self.device`, which forward uses to move every input (model_XPert.py:188-198);\n  * `self.drug_HG_embed`, a plain tensor attribute created on `device` (model_XPert.py:135). DataParallel replicates\n    parameters and buffers only; a plain tensor stays on GPU 0. It is constant (never trained), so a cached copy per\n    device is exact. The localiser is generic over every plain tensor attribute of every submodule, and\n    `plain_tensor_attributes()` lists them so a kernel guard can assert the list is exactly the one expected.\n\nActive only in training with grad enabled and at least one row per GPU, so validation, prediction and a ragged last\nbatch smaller than the GPU count run unchanged on one device.\n"""\nimport sys\n\nimport torch\nfrom torch.nn.parallel import data_parallel\n\n_CACHE = {}   # (id(original tensor), device) -> copy on that device\n\n\ndef plain_tensor_attributes(model):\n    """Every tensor held as a plain attribute (not a parameter or buffer), as \'module.path.attr\'."""\n    out = []\n    for mname, m in model.named_modules():\n        for name, val in vars(m).items():\n            if torch.is_tensor(val):\n                out.append((mname + \'.\' if mname else \'\') + name)\n    return sorted(out)\n\n\ndef _localise(replica, dev):\n    for m in replica.modules():\n        for name, val in list(vars(m).items()):\n            if torch.is_tensor(val) and val.device != dev:\n                key = (id(val), dev)\n                if key not in _CACHE:\n                    _CACHE[key] = val.to(dev)\n                setattr(m, name, _CACHE[key])     # replicas hold a shallow copy of __dict__: the original is untouched\n    replica.device = dev\n\n\ndef apply(model_XPert=None, device_ids=None):\n    """Patch models.model_XPert.XPertNet.forward in place. Returns the device ids it will use."""\n    MX = model_XPert or sys.modules[\'models.model_XPert\']\n    cls = MX.XPertNet\n    ids = list(device_ids) if device_ids is not None else list(range(torch.cuda.device_count()))\n    if getattr(cls.forward, \'_lincs_dp\', False):\n        return ids\n    orig = cls.forward\n\n    def fwd(self, data, *a, **k):\n        if getattr(self, \'_is_replica\', False):          # set by torch.nn.parallel.replicate\n            # NOT next(self.parameters()).device: replicate() turns a replica\'s parameters into plain non-leaf\n            # attributes, so parameters() is EMPTY on a replica (StopIteration -- caught by the local smoke test,\n            # 2026-09-24, before any GPU spend). parallel_apply runs each replica under\n            # torch.cuda.device(<its device>), so the current device is the replica\'s.\n            _localise(self, torch.device(\'cuda\', torch.cuda.current_device()))\n            return orig(self, data, *a, **k)\n        if len(ids) > 1 and self.training and torch.is_grad_enabled() and data[0].shape[0] >= len(ids):\n            if a:\n                raise TypeError(\'xpert_dp_patch: positional arguments after data are not scattered; pass mode= by name\')\n            return data_parallel(self, (data,), device_ids=ids, output_device=ids[0], module_kwargs=k or None)\n        return orig(self, data, *a, **k)\n\n    fwd._lincs_dp = True\n    cls.forward = fwd\n    return ids\n'
DP_PROBE_SRC = '# -*- coding: utf-8 -*-\n"""One configuration of the DataParallel proof and timing, run as its own process inside the staged XPert copy.\n[RESULTS 80, review 012 C4]\n\n    python xpert_dp_probe.py MODE SHARED_DIR FOLD \'THEIR_ARGV_AS_JSON\'\n    MODE in {single_ckpt, single_ckpt_repeat, dp, dp_ckpt}\n\nEach configuration is its own process because both patches act on classes and cannot be undone in place. The first\n(`single_ckpt`) writes the initial weights and one recipe batch to SHARED_DIR; every later one loads them, strictly,\nso all four see identical weights and identical rows.\n\nEvery gradient is taken through THEIR `train()` (train_xpert.py:62), so the loss that is differentiated is their\nbatch_weighted_loss (epoch 0) or weighted_loss (epoch 70), under their GradScaler, in `.train()` mode. The optimizer is\nSGD with lr 0, so the step leaves the weights untouched and the gradients -- unscaled in place by `scaler.step` --\nare read afterwards.\n\nDropout is switched off on EVERY module, not only through the four config rates: `model_XPert.py:144, 158, 164, 170`\nhard-code `nn.Dropout(p=0.1)` in the output heads, which the config does not reach, and the attention layers carry their\nown `dropout_p` (`model_utils.py:183`). The probe asserts none is left.\n\nTwo precisions, for two different questions:\n  * fp32 (their autocast replaced by a null context), first 32 rows: is DataParallel the SAME FUNCTION? Reduction order\n    is the only difference, so the tolerance is fp32 rounding. 32 rows because unpatched fp32 at 64 per GPU would not fit.\n  * fp16 autocast, all 128 rows, their precision: is the difference smaller than the precision noise the recipe already\n    accepts, measured as the same model\'s fp16-versus-fp32 gradient difference on one GPU?\nThen timing: their train() with Adam and their GradScaler, one warm-up and five timed batches drawn from their loader.\n"""\nimport contextlib\nimport itertools\nimport json\nimport logging\nimport os\nimport sys\nimport time\n\nMODE, SHARED, FOLD = sys.argv[1], sys.argv[2], sys.argv[3]\nsys.argv = json.loads(sys.argv[4])\n\nimport torch  # noqa: E402\nimport yaml  # noqa: E402\n\nimport train_xpert as T  # noqa: E402\nfrom utils import load_dataloader  # noqa: E402\nfrom models.model_XPert import XPertNet  # noqa: E402\nimport models.model_utils  # noqa: E402,F401\nimport xpert_ckpt_patch  # noqa: E402\nimport xpert_dp_patch  # noqa: E402\n\nCKPT = MODE in (\'single_ckpt\', \'single_ckpt_repeat\', \'dp_ckpt\')\nDP = MODE in (\'dp\', \'dp_ckpt\')\ndev = torch.device(\'cuda:0\')\nreport = {\'mode\': MODE, \'gpus_visible\': torch.cuda.device_count()}\n\nargs = T.arg_parse()\nconfig = yaml.safe_load(open(\'configs/%s.yaml\' % args.config))\nlogger = logging.getLogger(\'dp_probe\')\ntr, val, te, adata = load_dataloader(args, config, logger, nfold=FOLD, return_rawdata=True)\n\nif CKPT:\n    report[\'ckpt_patched\'] = xpert_ckpt_patch.apply()\nif DP:\n    report[\'dp_device_ids\'] = xpert_dp_patch.apply()\n    assert len(report[\'dp_device_ids\']) == 2, \'DataParallel needs both T4s, saw %r\' % report[\'dp_device_ids\']\n\n\ndef build():\n    torch.manual_seed(0)\n    m = XPertNet(args, config, dev, logger)\n    m.init_weights()\n    m.to(dev)\n    return m\n\n\ninit_path, batch_path = os.path.join(SHARED, \'init.pt\'), os.path.join(SHARED, \'batch.pt\')\nmodel = build()\nif MODE == \'single_ckpt\':\n    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, init_path)\n    torch.save(next(iter(tr)), batch_path)\ninit_sd = torch.load(init_path)\nmodel.load_state_dict(init_sd, strict=True)\nbatch = torch.load(batch_path)\nassert batch[0].shape[0] == 128, \'not the recipe batch size\'\n\n# The object their code holds must still be an XPertNet with unprefixed keys [review 012 C1].\nreport[\'state_dict_keys_equal_unpatched\'] = sorted(model.state_dict().keys()) == sorted(init_sd.keys())\nreport[\'any_module_prefix\'] = any(k.startswith(\'module.\') for k in model.state_dict().keys())\nreport[\'plain_tensor_attributes\'] = xpert_dp_patch.plain_tensor_attributes(model)\n\n# Dropout off everywhere, staying in train mode [review 012 C4 -- and the four config rates are not enough].\nDROP_KEYS = (\'attention_probs_dropout_prob\', \'hidden_dropout_prob\', \'cell_input_hidden_dropout_prob\',\n             \'drug_input_hidden_dropout_prob\')\npublished_dropout = {k: config[\'model\'][\'ATTN\'][k] for k in DROP_KEYS}\nreport[\'published_dropout\'] = published_dropout\nfor k in DROP_KEYS:\n    config[\'model\'][\'ATTN\'][k] = 0.0\nn_drop = 0\nfor m in model.modules():\n    if isinstance(m, torch.nn.Dropout):\n        n_drop += int(m.p > 0)\n        m.p = 0.0\n    if hasattr(m, \'dropout_p\'):\n        n_drop += int(m.dropout_p > 0)\n        m.dropout_p = 0.0\nreport[\'dropout_sites_zeroed\'] = n_drop\nassert all(m.p == 0.0 for m in model.modules() if isinstance(m, torch.nn.Dropout))\nassert all(m.dropout_p == 0.0 for m in model.modules() if hasattr(m, \'dropout_p\'))\n\n\ndef grads_through_their_train(rows, epoch, fp32):\n    model.load_state_dict(init_sd, strict=True)\n    b = [t[:rows] for t in batch] if rows < 128 else batch\n    opt = torch.optim.SGD(model.parameters(), lr=0.0)\n    scaler = torch.cuda.amp.GradScaler()\n    saved = T.autocast\n    if fp32:\n        T.autocast = contextlib.nullcontext          # their `with autocast():` becomes a no-op: fp32 forward\n    try:\n        losses = T.train(model, opt, [b], args, config, scaler=scaler, epoch=epoch)\n    finally:\n        T.autocast = saved\n    g = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}\n    finite = all(torch.isfinite(v).all().item() for v in g.values())\n    for p in model.parameters():\n        p.grad = None\n    return [float(x) for x in losses], g, finite, float(scaler.get_scale())\n\n\nout_grads = {}\n# fp16 at 32 rows as well, so the precision-noise scale (fp16 vs fp32, one GPU) is measured on the SAME rows.\nfor tag, rows, epoch, fp32 in ((\'fp32_e0\', 32, 0, True), (\'fp32_e70\', 32, 70, True),\n                               (\'fp16_e0\', 32, 0, False), (\'fp16_e70\', 32, 70, False),\n                               (\'fp16_e0_128\', 128, 0, False), (\'fp16_e70_128\', 128, 70, False)):\n    losses, g, finite, scale = grads_through_their_train(rows, epoch, fp32)\n    report[tag] = {\'losses\': losses, \'grads_finite\': finite, \'scale\': scale, \'n_grad_tensors\': len(g)}\n    out_grads[tag] = g\ntorch.save(out_grads, os.path.join(SHARED, MODE + \'_grads.pt\'))\n\n# Timing, the recipe itself: dropout back to its published values, Adam, their GradScaler, loader included.\nif MODE != \'single_ckpt_repeat\':\n    for k in DROP_KEYS:\n        config[\'model\'][\'ATTN\'][k] = published_dropout[k]\n    del model\n    torch.cuda.empty_cache()\n    model = build()\n    model.load_state_dict(init_sd, strict=True)\n    opt = torch.optim.Adam(model.parameters(), lr=config[\'train\'][\'train_lr\'], weight_decay=config[\'train\'][\'weight_decay\'])\n    scaler = torch.cuda.amp.GradScaler()\n    it = iter(tr)\n    T.train(model, opt, itertools.islice(it, 1), args, config, scaler=scaler, epoch=0)\n    for d in range(torch.cuda.device_count()):\n        torch.cuda.reset_peak_memory_stats(d)\n    for d in range(torch.cuda.device_count()):\n        torch.cuda.synchronize(d)\n    t0 = time.time()\n    T.train(model, opt, itertools.islice(it, 5), args, config, scaler=scaler, epoch=0)\n    for d in range(torch.cuda.device_count()):\n        torch.cuda.synchronize(d)\n    report[\'s_train_step\'] = (time.time() - t0) / 5\n    report[\'peak_gib_per_gpu\'] = [torch.cuda.max_memory_allocated(d) / 2 ** 30 for d in range(torch.cuda.device_count())]\n    report[\'train_batches\'] = len(tr)\n    report[\'val_batches\'] = len(val)\n\nprint(\'DPPROBE \' + json.dumps(report), flush=True)\n'
open(os.path.join(X, 'xpert_dp_patch.py'), 'w', encoding='utf-8').write(DP_PATCH_SRC)
open(os.path.join(X, 'xpert_dp_probe.py'), 'w', encoding='utf-8').write(DP_PROBE_SRC)
RECORD['activation_checkpointing'] = {'patched': ['Encoder', 'crossEncoder'], 'how': 'runtime wrapper; '
                                     'their files verbatim', 'proof': 'model/v9/prove_checkpoint_exact.py'}
RECORD['deviations'] = ['flash_attn shim on PYTHONPATH (their model imports it at module scope)',
                        'MyDataset: one tensor per drug instead of one per row -- values proven identical, '
                        'storage shared; the unpatched recipe needs ~24.7 GB of dataset RAM on this fold',
                        'activation checkpointing of Encoder/crossEncoder, applied at runtime by a wrapper -- '
                        'same batch 128, same loss, gradients within the noise floor; the unpatched recipe '
                        'needs ~14.95 GiB of activations on a 14.56 GiB T4',
                        'all_drugs_unimol_arr.npy rebuilt from the released npz (config names it, never released)',
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
# --------------------------------------------------------------------------------------------------------
# GUARD G -- is DataParallel over both T4s the same computation, and how fast is it? [RESULTS 80]
# Four processes, one per configuration, because both patches act on classes and cannot be undone in place.
# Every gradient is taken through THEIR train(); criteria are RESULTS 80.3, committed before this launch.
# --------------------------------------------------------------------------------------------------------
import torch
DPS = os.path.join(W, 'dp_shared')
os.makedirs(DPS, exist_ok=True)
DP_MODES = ['single_ckpt', 'single_ckpt_repeat', 'dp', 'dp_ckpt']
DPR = {}
for mode in DP_MODES:
    r = subprocess.run([sys.executable, 'xpert_dp_probe.py', mode, DPS, FOLD, json.dumps(probe_argv)],
                       capture_output=True, text=True, env=ENV, cwd=X)
    lines = [l for l in r.stdout.splitlines() if l.startswith('DPPROBE ')]
    if r.returncode != 0 or len(lines) != 1:
        tb = [l for l in r.stderr.splitlines() if l.strip() and '%|' not in l and 'it/s]' not in l]
        RECORD['guard_g_failure'] = {'mode': mode, 'returncode': r.returncode, 'stderr_tail': tb[-40:]}
        print(r.stderr[-4000:])
        fatal('GUARD G: DataParallel probe %s failed (returncode %d).' % (mode, r.returncode))
    DPR[mode] = json.loads(lines[0][len('DPPROBE '):])
    log('GUARD G probe', mode, 'ok:', {k: DPR[mode].get(k) for k in ('s_train_step', 'peak_gib_per_gpu',
                                                                     'dropout_sites_zeroed')})

G = {m: torch.load(os.path.join(DPS, m + '_grads.pt')) for m in DP_MODES}


def _cmp(ga, gb):
    assert ga.keys() == gb.keys(), 'different parameters received gradients'
    num = sum(float(((ga[n] - gb[n]) ** 2).sum()) for n in ga)
    den = sum(float((gb[n] ** 2).sum()) for n in gb)
    return {'max_abs': max(float((ga[n] - gb[n]).abs().max()) for n in ga), 'rel_l2': (num / den) ** 0.5}


def _lrel(a, b):
    return max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip(a, b))


cmpr = {}
for tag in ('fp32_e0', 'fp32_e70', 'fp16_e0', 'fp16_e70', 'fp16_e0_128', 'fp16_e70_128'):
    ref = G['single_ckpt'][tag]
    cmpr[tag] = {m: _cmp(G[m][tag], ref) for m in ('single_ckpt_repeat', 'dp', 'dp_ckpt')}
    cmpr[tag]['loss_rel'] = {m: _lrel(DPR[m][tag]['losses'], DPR['single_ckpt'][tag]['losses'])
                             for m in ('single_ckpt_repeat', 'dp', 'dp_ckpt')}
prec = {e: _cmp(G['single_ckpt']['fp16_' + e], G['single_ckpt']['fp32_' + e]) for e in ('e0', 'e70')}
chk = {}
for v in ('dp', 'dp_ckpt'):
    chk[v] = {
        'fp32_same_function': all(cmpr['fp32_' + e][v]['rel_l2'] < 1e-5 and cmpr['fp32_' + e]['loss_rel'][v] < 1e-6
                                  for e in ('e0', 'e70')),
        'fp16_within_precision_noise': all(cmpr['fp16_' + e][v]['rel_l2'] <= prec[e]['rel_l2'] for e in ('e0', 'e70')),
        'structure': (DPR[v]['state_dict_keys_equal_unpatched'] and not DPR[v]['any_module_prefix']
                      and DPR[v]['plain_tensor_attributes'] == ['drug_HG_embed']
                      and all(DPR[v][t]['grads_finite'] for t in cmpr)),
        'memory_below_13': max(DPR[v]['peak_gib_per_gpu']) < 13.0}
    chk[v]['pass'] = all(chk[v].values())
s_val = TIMING['s_val_step']
proj = {}
for v in ('single_ckpt', 'dp', 'dp_ckpt'):
    ep = DPR[v]['train_batches'] * DPR[v]['s_train_step'] + DPR[v]['val_batches'] * s_val
    proj[v] = {'s_train_step': round(DPR[v]['s_train_step'], 3), 'epoch_s': round(ep, 1),
               'gpu_h_for_210_epochs': round(210 * ep / 3600, 1),
               'sessions_at_7.95h': round(210 * ep / 3600 / 7.95, 1),
               'peak_gib_per_gpu': [round(x, 2) for x in DPR[v]['peak_gib_per_gpu']]}
passing = [v for v in ('dp', 'dp_ckpt') if chk[v]['pass']]
chosen = min(passing, key=lambda v: proj[v]['epoch_s']) if passing else None
RECORD['dp'] = {'probe': DPR, 'comparisons': cmpr, 'precision_noise_fp16_vs_fp32': prec, 'checks_80_3': chk,
                'projection_80_4': proj, 'o2_variant': chosen,
                'o2_price_gpu_h': proj[chosen]['gpu_h_for_210_epochs'] if chosen else None}
log('GUARD G checks (RESULTS 80.3):', chk)
log('GUARD G projection (RESULTS 80.4):', proj, '| O2 variant:', chosen)
shutil.rmtree(DPS, ignore_errors=True)

if MEASURE_ONLY:
    RECORD['stopped_by'] = 'measure_only'
    RECORD['host_memory'] = mem_summary()
    json.dump(RECORD, open(os.path.join(W, 'run_record.json'), 'w'), indent=2)
    log('MEASURE_ONLY: stopping before training. Projection recorded; the full run is decided from it.')
    shutil.rmtree(X, ignore_errors=True)      # includes the 2.24 GB rebuilt array; keeps the output small
    shutil.rmtree(V9, ignore_errors=True)
    raise SystemExit(0)

RECORD['train_cmd'] = ' '.join(cmd)
RECORD['published_command_source'] =('scripts/train.sh:15 (mdmt) and README; fold list reduced to %s; '
                                      'folds independent per train_xpert.py:425-455' % FOLD)
RECORD['seed_note'] = ('seed 2024 set once at train_xpert.py:402 before the fold loop, so this fold starts '
                       'from a fresh seed-2024 state, not the state their 3-fold sequential run would reach')
log('TRAIN:', RECORD['train_cmd'])
deadline = T0 + BUDGET_H * 3600
TRAIN_LOG = os.path.join(W, 'train_xpert.log')
fired = None
epochs_seen, last_epoch_line = 0, ''
last_epoch_index = None       # parsed from "Epoch {epoch}, Valid Total Loss", not inferred from a line count
last_counter_logged = None    # the N in the most recent "EarlyStopping counter: N out of 50"
with open(TRAIN_LOG, 'w') as lf:
    p = subprocess.Popen(cmd, cwd=X, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1)
    in_args, seen_args, args_checked = False, {}, False
    for line in p.stdout:
        lf.write(line)
        # GUARD E -- the EXECUTED arguments must be the published ones. Their trainer prints its namespace
        # between "---------args-----------" and a blank line. Read it, compare, and kill the trainer at once
        # on any mismatch, before a single epoch is paid for.
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
                    RECORD['guards']['executed_args'] = seen_args
                    if bad:
                        p.terminate()
                        fatal('GUARD E: executed args differ from the published recipe {key: (got, want)}: %s'
                              % bad)
                    log('GUARD E executed args match the published recipe:', {k: seen_args[k] for k in EXPECT_ARGS})
        if 'EarlyStopping counter:' in line:
            try:
                last_counter_logged = int(line.split('EarlyStopping counter:')[1].split('out of')[0])
            except (IndexError, ValueError):
                pass
        if 'Valid Total Loss' in line:
            epochs_seen += 1
            last_epoch_line = line.strip()
            try:
                last_epoch_index = int(line.split('Epoch ')[1].split(',')[0])
            except (IndexError, ValueError):
                pass
            if epochs_seen <= 3 or epochs_seen % 10 == 0:
                log('epoch', epochs_seen, '|', line.strip()[-160:])
        if 'train_time' in line or 'Traceback' in line or 'Error' in line:
            log(line.strip()[-300:])
        if time.time() > deadline:
            fired = 'watchdog'
            log('WALL-CLOCK GUARD FIRED after %d epochs; terminating trainer.' % epochs_seen)
            p.terminate()
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
            break
    rc = p.wait()
if fired is None:
    fired = 'finished' if rc == 0 else ('killed_by_signal' if rc < 0 else 'crashed')
RECORD['host_memory'] = mem_summary()
RECORD.update({'train_returncode': rc, 'stopped_by': fired, 'epochs_seen': epochs_seen,
               'last_epoch_line': last_epoch_line, 'train_hours': round((time.time() - T0) / 3600, 3)})
log('training ended:', fired, '| rc', rc, '| epochs', epochs_seen)
if fired in ('crashed', 'killed_by_signal'):
    os.system('tail -60 %s' % TRAIN_LOG)
    fatal('trainer crashed; see train_xpert.log.')

# The per-fold record review 007 C6 asked for: whether early stopping or the guard ended the run -- and,
# per review 008 C2, HOW CLOSE TO CONVERGED it was when it ended.
#
# Review 008 C2: counting "EarlyStopping counter" lines measures the TOTAL number of non-improving epochs
# over the whole run, because an improving epoch resets the counter to 0 SILENTLY (utils.py step(): the
# improving branch sets self.counter = 0 with no log line). 150 scattered non-improving epochs followed by an
# improvement 5 epochs before a guard kill would read as "long converged" when the true state was 5/50.
# The exact end state needs no log parsing: every epoch after the best one was by definition non-improving,
# so the counter at the end is last_epoch_index - best_epoch. The last logged counter is kept only as a
# cross-check.
es = open(TRAIN_LOG).read()
CKPT = find_one(os.path.join(X, 'experiment', '**', '%s_fold_early_stop.pth' % FOLD))
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
best_epoch = int(ck.get('epoch', -1))
RECORD['best_checkpoint'] = {'path': CKPT, 'epoch': best_epoch}
if last_epoch_index is None or best_epoch < 0:
    fatal('could not establish last_epoch_index (%s) or best_epoch (%s); convergence is unreadable.'
          % (last_epoch_index, best_epoch))
counter_at_end = last_epoch_index - best_epoch
RECORD['last_epoch_index'] = last_epoch_index
RECORD['counter_at_end'] = counter_at_end              # epochs since the best test-loss4 checkpoint
RECORD['patience'] = 50
RECORD['counter_cross_check'] = {'last_logged_counter': last_counter_logged,
                                 'consistent': counter_at_end == 0 or last_counter_logged == counter_at_end}
RECORD['nonimproving_epochs_total'] = es.count('EarlyStopping counter')   # descriptive ONLY, not convergence
RECORD['best_selected_before_init_epoch_70'] = best_epoch < 70           # accelerated objective, disclosed
# RESULTS 71.7, pre-committed before launch: a guard-stopped run cut off while test loss was still improving
# leaves XPert UNDER-trained, so a v9 win would be inflated rather than conservative.
RECORD['admissible_for_v9_win'] = not (fired == 'watchdog' and counter_at_end < 45)
log('best checkpoint epoch', best_epoch, '| last epoch', last_epoch_index, '| counter at end',
    counter_at_end, '/ 50 | admissible for a v9-win claim:', RECORD['admissible_for_v9_win'])

# --------------------------------------------------------------------------------------------------------
# 4. PREDICT the test rows with OUR harness, which carries row_index -- their predict_profile does not, and
#    head_to_head_mdmt.py refuses to pair without it. Same shim, same unmasked path.
# --------------------------------------------------------------------------------------------------------
PROFILE = os.path.join(W, 'xpert_trained_%s_test_profile.npy' % FOLD)
env2 = dict(ENV, XPERT_DIR=X, XPERT_CKPT=CKPT)
pr = subprocess.run([sys.executable, '-u', os.path.join(V9, 'xpert_native_eval.py'), '--nfold', FOLD,
                     '--rows', 'test', '--device', 'cuda', '--batch', '128', '--out', PROFILE],
                    capture_output=True, text=True, env=env2)
print(pr.stdout[-3000:], pr.stderr[-3000:])
if pr.returncode != 0 or not os.path.exists(PROFILE):
    fatal('prediction with the trained checkpoint failed.')
prof = np.load(PROFILE, allow_pickle=True).item()
if 'row_index' not in prof or len(prof['row_index']) != 21321:
    fatal('profile has %s rows / row_index present=%s; expected 21321 with row_index.'
          % (len(prof.get('y_pred', [])), 'row_index' in prof))
RECORD['profile'] = {'path': PROFILE, 'n': int(len(prof['row_index']))}

# --------------------------------------------------------------------------------------------------------
# 5. Artefacts
# --------------------------------------------------------------------------------------------------------
shutil.copy(CKPT, os.path.join(W, 'xpert_trained_%s.pth' % FOLD))
for junk in (ARR,):                       # 2.24 GB derived file; rebuildable, not an output
    try:
        os.remove(junk)
    except OSError:
        pass
RECORD['total_hours'] = round((time.time() - T0) / 3600, 3)
RECORD['host_memory'] = mem_summary()
# Review 009 C3: the seed is set once at train_xpert.py:402, before the fold loop, and split_cold_cell_1 is
# the SECOND fold of train.sh:15 -- so this run starts from a fresh seed-2024 state, not the RNG state
# their run reached after split_cold_drug_1. Same recipe and seed value, different trajectory.
RECORD['framing'] = ('XPert trained to its published recipe on %s. NOT a reproduction of their cold-cell '
                     'run: an independent draw of the recipe.' % FOLD)
json.dump(RECORD, open(os.path.join(W, 'run_record.json'), 'w'), indent=2)
log('DONE', json.dumps({k: RECORD[k] for k in ('stopped_by', 'epochs_seen', 'total_hours')}))
shutil.rmtree(X, ignore_errors=True)      # the staged copy; outputs are already in /kaggle/working
for f in sorted(glob.glob(os.path.join(W, '*'))):
    print('%9.2f MB  %s' % (os.path.getsize(f) / 1e6, f), flush=True)
