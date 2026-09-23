# -*- coding: utf-8 -*-
"""One configuration of the DataParallel proof and timing, run as its own process inside the staged XPert copy.
[RESULTS 80, review 012 C4]

    python xpert_dp_probe.py MODE SHARED_DIR FOLD 'THEIR_ARGV_AS_JSON'
    MODE in {single_ckpt, single_ckpt_repeat, dp, dp_ckpt}

Each configuration is its own process because both patches act on classes and cannot be undone in place. The first
(`single_ckpt`) writes the initial weights and one recipe batch to SHARED_DIR; every later one loads them, strictly,
so all four see identical weights and identical rows.

Every gradient is taken through THEIR `train()` (train_xpert.py:62), so the loss that is differentiated is their
batch_weighted_loss (epoch 0) or weighted_loss (epoch 70), under their GradScaler, in `.train()` mode. The optimizer is
SGD with lr 0, so the step leaves the weights untouched and the gradients -- unscaled in place by `scaler.step` --
are read afterwards.

Dropout is switched off on EVERY module, not only through the four config rates: `model_XPert.py:144, 158, 164, 170`
hard-code `nn.Dropout(p=0.1)` in the output heads, which the config does not reach, and the attention layers carry their
own `dropout_p` (`model_utils.py:183`). The probe asserts none is left.

Two precisions, for two different questions:
  * fp32 (their autocast replaced by a null context), first 32 rows: is DataParallel the SAME FUNCTION? Reduction order
    is the only difference, so the tolerance is fp32 rounding. 32 rows because unpatched fp32 at 64 per GPU would not fit.
  * fp16 autocast, all 128 rows, their precision: is the difference smaller than the precision noise the recipe already
    accepts, measured as the same model's fp16-versus-fp32 gradient difference on one GPU?
Then timing: their train() with Adam and their GradScaler, one warm-up and five timed batches drawn from their loader.
"""
import contextlib
import itertools
import json
import logging
import os
import sys
import time

MODE, SHARED, FOLD = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = json.loads(sys.argv[4])

import torch  # noqa: E402
import yaml  # noqa: E402

import train_xpert as T  # noqa: E402
from utils import load_dataloader  # noqa: E402
from models.model_XPert import XPertNet  # noqa: E402
import models.model_utils  # noqa: E402,F401
import xpert_ckpt_patch  # noqa: E402
import xpert_dp_patch  # noqa: E402

CKPT = MODE in ('single_ckpt', 'single_ckpt_repeat', 'dp_ckpt')
DP = MODE in ('dp', 'dp_ckpt')
dev = torch.device('cuda:0')
report = {'mode': MODE, 'gpus_visible': torch.cuda.device_count()}

args = T.arg_parse()
config = yaml.safe_load(open('configs/%s.yaml' % args.config))
logger = logging.getLogger('dp_probe')
tr, val, te, adata = load_dataloader(args, config, logger, nfold=FOLD, return_rawdata=True)

if CKPT:
    report['ckpt_patched'] = xpert_ckpt_patch.apply()
if DP:
    report['dp_device_ids'] = xpert_dp_patch.apply()
    assert len(report['dp_device_ids']) == 2, 'DataParallel needs both T4s, saw %r' % report['dp_device_ids']


def build():
    torch.manual_seed(0)
    m = XPertNet(args, config, dev, logger)
    m.init_weights()
    m.to(dev)
    return m


init_path, batch_path = os.path.join(SHARED, 'init.pt'), os.path.join(SHARED, 'batch.pt')
model = build()
if MODE == 'single_ckpt':
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, init_path)
    torch.save(next(iter(tr)), batch_path)
init_sd = torch.load(init_path)
model.load_state_dict(init_sd, strict=True)
batch = torch.load(batch_path)
assert batch[0].shape[0] == 128, 'not the recipe batch size'

# The object their code holds must still be an XPertNet with unprefixed keys [review 012 C1].
report['state_dict_keys_equal_unpatched'] = sorted(model.state_dict().keys()) == sorted(init_sd.keys())
report['any_module_prefix'] = any(k.startswith('module.') for k in model.state_dict().keys())
report['plain_tensor_attributes'] = xpert_dp_patch.plain_tensor_attributes(model)

# Dropout off everywhere, staying in train mode [review 012 C4 -- and the four config rates are not enough].
DROP_KEYS = ('attention_probs_dropout_prob', 'hidden_dropout_prob', 'cell_input_hidden_dropout_prob',
             'drug_input_hidden_dropout_prob')
published_dropout = {k: config['model']['ATTN'][k] for k in DROP_KEYS}
report['published_dropout'] = published_dropout
for k in DROP_KEYS:
    config['model']['ATTN'][k] = 0.0
n_drop = 0
for m in model.modules():
    if isinstance(m, torch.nn.Dropout):
        n_drop += int(m.p > 0)
        m.p = 0.0
    if hasattr(m, 'dropout_p'):
        n_drop += int(m.dropout_p > 0)
        m.dropout_p = 0.0
report['dropout_sites_zeroed'] = n_drop
assert all(m.p == 0.0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
assert all(m.dropout_p == 0.0 for m in model.modules() if hasattr(m, 'dropout_p'))


def grads_through_their_train(rows, epoch, fp32):
    model.load_state_dict(init_sd, strict=True)
    b = [t[:rows] for t in batch] if rows < 128 else batch
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.cuda.amp.GradScaler()
    saved = T.autocast
    if fp32:
        T.autocast = contextlib.nullcontext          # their `with autocast():` becomes a no-op: fp32 forward
    try:
        losses = T.train(model, opt, [b], args, config, scaler=scaler, epoch=epoch)
    finally:
        T.autocast = saved
    g = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    finite = all(torch.isfinite(v).all().item() for v in g.values())
    for p in model.parameters():
        p.grad = None
    return [float(x) for x in losses], g, finite, float(scaler.get_scale())


out_grads = {}
# fp16 at 32 rows as well, so the precision-noise scale (fp16 vs fp32, one GPU) is measured on the SAME rows.
for tag, rows, epoch, fp32 in (('fp32_e0', 32, 0, True), ('fp32_e70', 32, 70, True),
                               ('fp16_e0', 32, 0, False), ('fp16_e70', 32, 70, False),
                               ('fp16_e0_128', 128, 0, False), ('fp16_e70_128', 128, 70, False)):
    losses, g, finite, scale = grads_through_their_train(rows, epoch, fp32)
    report[tag] = {'losses': losses, 'grads_finite': finite, 'scale': scale, 'n_grad_tensors': len(g)}
    out_grads[tag] = g
torch.save(out_grads, os.path.join(SHARED, MODE + '_grads.pt'))

# Timing, the recipe itself: dropout back to its published values, Adam, their GradScaler, loader included.
if MODE != 'single_ckpt_repeat':
    for k in DROP_KEYS:
        config['model']['ATTN'][k] = published_dropout[k]
    del model
    torch.cuda.empty_cache()
    model = build()
    model.load_state_dict(init_sd, strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=config['train']['train_lr'], weight_decay=config['train']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler()
    it = iter(tr)
    T.train(model, opt, itertools.islice(it, 1), args, config, scaler=scaler, epoch=0)
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    t0 = time.time()
    T.train(model, opt, itertools.islice(it, 5), args, config, scaler=scaler, epoch=0)
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    report['s_train_step'] = (time.time() - t0) / 5
    report['peak_gib_per_gpu'] = [torch.cuda.max_memory_allocated(d) / 2 ** 30 for d in range(torch.cuda.device_count())]
    report['train_batches'] = len(tr)
    report['val_batches'] = len(val)

print('DPPROBE ' + json.dumps(report), flush=True)
