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

WHAT IS NOT OURS TO CHANGE, AND IS NOT CHANGED
  Their source files are copied verbatim. The two things added are the flash_attn shim on PYTHONPATH (their
  model imports flash_attn at module scope and it is not on this image) and the dense unimol array their
  config names but never released, rebuilt from the npz they did release [RESULTS 68].

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
FOLD = 'split_cold_cell_1'
W = '/kaggle/working'
RECORD = {'fold': FOLD, 'budget_h': BUDGET_H, 'guards': {}, 'decision': 'option (a) as published, disclosed'}


def log(*a):
    print('[%6.1f min]' % ((time.time() - T0) / 60), *a, flush=True)


def fatal(msg):
    RECORD['fatal'] = msg
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
# 3. TRAIN, their recipe, under a wall-clock guard. Their stopper writes the best checkpoint to disk on
#    every improvement (utils.py save_checkpoint), so a guard kill cannot lose it.
#    NOTE: their boolean flags use argparse `type=bool`, so passing the string "False" ENABLES them. Only
#    --output_profile is passed, and only as True.
# --------------------------------------------------------------------------------------------------------
cmd = [sys.executable, '-u', 'train_xpert.py', '--mode', 'train', '--nfold', FOLD, '--dataset',
       'l1000_mdmt', '--drug_feat', 'unimol', '--device', 'cuda:0', '--output_profile', 'True']
RECORD['train_cmd'] = ' '.join(cmd)
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
    for line in p.stdout:
        lf.write(line)
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
    fired = 'finished' if rc == 0 else 'crashed'
RECORD.update({'train_returncode': rc, 'stopped_by': fired, 'epochs_seen': epochs_seen,
               'last_epoch_line': last_epoch_line, 'train_hours': round((time.time() - T0) / 3600, 3)})
log('training ended:', fired, '| rc', rc, '| epochs', epochs_seen)
if fired == 'crashed':
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
json.dump(RECORD, open(os.path.join(W, 'run_record.json'), 'w'), indent=2)
log('DONE', json.dumps({k: RECORD[k] for k in ('stopped_by', 'epochs_seen', 'total_hours')}))
shutil.rmtree(X, ignore_errors=True)      # the staged copy; outputs are already in /kaggle/working
for f in sorted(glob.glob(os.path.join(W, '*'))):
    print('%9.2f MB  %s' % (os.path.getsize(f) / 1e6, f), flush=True)
