# -*- coding: utf-8 -*-
"""One configuration of the v7 DataParallel proof (RESULTS 81, final form 81.5 + 81.6), run as its own process inside
the staged XPert copy.

    python xpert_dp_probe.py MODE SHARED_DIR FOLD 'THEIR_ARGV_AS_JSON'
    MODE in {single_ckpt, single_ckpt_repeat, dp, split}

The first process (`single_ckpt`) writes the initial weights and one recipe batch to SHARED_DIR, and finds the fp16
GradScaler scale; every later one loads them strictly. The ten parameters their loss never uses are frozen in EVERY
mode (RESULTS 80.5, Amendment A), and after every gradient capture the set with `grad is None` must equal them exactly.

Tags, all through THEIR train() with an lr-0 SGD step and dropout zeroed on every module:
  f64_e0 / f64_e70    float64, 16 rows (8 per GPU)         81.1a  semantics      gradients KEPT in float64 (014 C2)
  f32m_e0 / f32m_e70  fp32, math SDPA, 16 rows              81.1b  dp vs split    (and reported vs single)
  f16_e0 / f16_e70    fp16 autocast, 128 rows, one scale    81.1c  reported only; autocast asserted inside replicas
16 rows, not 32 (RESULTS 81.6): on the math backend each of the 8 gene self-attention layers stores a 979x979x8
probability tensor for backward, ~245 MB per sample in fp32 and twice that in float64, so the uncheckpointed 32-row
split and the float64 DP at 16 per GPU would not fit a T4.

Modes: single_ckpt (reference; every tag), single_ckpt_repeat (floor; every tag), dp (no checkpointing -- the
production variant -- every tag, plus a short timing), split (one GPU, the batch as two halves concatenated before their
loss, no checkpointing; f32m tags only).
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
import models.model_XPert as MX  # noqa: E402
import models.model_utils  # noqa: E402,F401
import xpert_ckpt_patch  # noqa: E402
import xpert_dp_patch  # noqa: E402

FROZEN = sorted(['attnEncoder_trt.crossEncoders.0.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.0.LayerNorm.gamma',
                 'attnEncoder_trt.crossEncoders.1.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.1.LayerNorm.gamma',
                 'cell_emb.linear.bias', 'cell_emb.linear.weight', 'ctl_fc.0.bias', 'ctl_fc.0.weight',
                 'ctl_fc.3.bias', 'ctl_fc.3.weight'])
ROWS_SMALL = 16
report = {'mode': MODE, 'torch': torch.__version__, 'gpus_visible': torch.cuda.device_count(), 'tags': {}}

args = T.arg_parse()
config = yaml.safe_load(open('configs/%s.yaml' % args.config))
logger = logging.getLogger('dp_probe')
tr, val, te, adata = load_dataloader(args, config, logger, nfold=FOLD, return_rawdata=True)

if MODE in ('single_ckpt', 'single_ckpt_repeat'):
    report['ckpt_patched'] = xpert_ckpt_patch.apply()
if MODE == 'dp':
    report['dp_device_ids'] = xpert_dp_patch.apply()
    assert len(report['dp_device_ids']) == 2, 'DataParallel needs both T4s, saw %r' % report['dp_device_ids']

# ---- split: the same-GPU reference of 81.1b -- DataParallel's computation minus the second device ----------------
ORIG_FWD = MX.XPertNet.forward


def _cat(outs):
    a = outs[0]
    if torch.is_tensor(a):
        return torch.cat(outs, 0)
    if a is None:
        return None
    if isinstance(a, dict):
        return {k: _cat([o[k] for o in outs]) for k in a}
    if isinstance(a, (tuple, list)):
        return type(a)(_cat(list(z)) for z in zip(*outs))
    raise TypeError(type(a))


if MODE == 'split':
    def _split_fwd(self, data, *a, **k):
        if self.training and torch.is_grad_enabled():
            h = data[0].shape[0] // 2
            return _cat([ORIG_FWD(self, [t[:h] for t in data], *a, **k), ORIG_FWD(self, [t[h:] for t in data], *a, **k)])
        return ORIG_FWD(self, data, *a, **k)
    MX.XPertNet.forward = _split_fwd

# ---- float64: their get_unimol_drug_feat hard-casts atom features with .float() (model_XPert.py:14) ------------------
ORIG_GUDF = MX.get_unimol_drug_feat


def _gudf_keep_dtype(x):
    m = x[:, :, 0].long()
    return x[:, :, 2:], x[:, :, 1].long(), (1.0 - m.unsqueeze(1).unsqueeze(2)) * -10000.0


dev = torch.device('cuda:0')


def build(dtype=torch.float32):
    torch.manual_seed(0)
    m = MX.XPertNet(args, config, dev, logger)
    m.init_weights()
    m.to(dev)
    if dtype == torch.float64:
        m.double()
    params = dict(m.named_parameters())
    assert all(n in params for n in FROZEN), 'a frozen name is not a parameter'
    for n in FROZEN:
        params[n].requires_grad_(False)
    for mod in m.modules():                        # dropout off on EVERY module, in train mode [013 C4, 80.2]
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, 'dropout_p'):
            mod.dropout_p = 0.0
    return m


init_path, batch_path, scale_path = (os.path.join(SHARED, f) for f in ('init.pt', 'batch.pt', 'fp16_scale.json'))
if MODE == 'single_ckpt':
    m0 = build()
    torch.save({k: v.detach().cpu() for k, v in m0.state_dict().items()}, init_path)
    torch.save(next(iter(tr)), batch_path)
    del m0
init_sd = torch.load(init_path)
batch = torch.load(batch_path)
assert batch[0].shape[0] == 128, 'not the recipe batch size'


def capture(model, rows, epoch, precision, scale=2.0 ** 6):
    model.load_state_dict(init_sd, strict=True)
    b = [t[:rows] for t in batch]
    if precision == 'f64':
        b = [t.double() if t.is_floating_point() else t for t in b]
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=scale)
    saved_ac, saved_gudf = T.autocast, MX.get_unimol_drug_feat
    if precision in ('f64', 'f32m'):
        T.autocast = contextlib.nullcontext
    if precision == 'f64':
        MX.get_unimol_drug_feat = _gudf_keep_dtype
    xpert_dp_patch.EXPECT_AUTOCAST = (precision == 'f16') if MODE == 'dp' else None
    try:
        losses = T.train(model, opt, [b], args, config, scaler=scaler, epoch=epoch)
    finally:
        T.autocast, MX.get_unimol_drug_feat = saved_ac, saved_gudf
        xpert_dp_patch.EXPECT_AUTOCAST = None
    keep = torch.float64 if precision == 'f64' else torch.float32
    g = {n: p.grad.detach().to(keep).cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    none_set = sorted(n for n, p in model.named_parameters() if p.grad is None)
    finite = all(bool(torch.isfinite(v).all()) for v in g.values())
    for p in model.parameters():
        p.grad = None
    return [float(x) for x in losses], g, none_set, finite


out = {}
plan = []
if MODE != 'split':
    plan += [('f64_e0', ROWS_SMALL, 0, 'f64'), ('f64_e70', ROWS_SMALL, 70, 'f64')]
plan += [('f32m_e0', ROWS_SMALL, 0, 'f32m'), ('f32m_e70', ROWS_SMALL, 70, 'f32m')]
if MODE != 'split':
    plan += [('f16_e0', 128, 0, 'f16'), ('f16_e70', 128, 70, 'f16')]

models = {}
scales = json.load(open(scale_path)) if os.path.exists(scale_path) else {}
for tag, rows, epoch, prec in plan:
    torch.backends.cuda.enable_mem_efficient_sdp(prec != 'f32m')     # math backend for 81.1b; float64 forces it anyway
    dtype = torch.float64 if prec == 'f64' else torch.float32
    if dtype not in models:
        models = {dtype: build(dtype)}                                 # one model resident at a time
        torch.cuda.empty_cache()
    model = models[dtype]
    before = dict(xpert_dp_patch.REPLICA_CALLS)
    if prec == 'f16' and MODE == 'single_ckpt':
        scale = 2.0 ** 6
        while True:
            losses, g, none_set, finite = capture(model, rows, epoch, prec, scale)
            if finite or scale < 2.0 ** -12:
                break
            scale /= 2.0
        scales[tag] = scale
        json.dump(scales, open(scale_path, 'w'))
    else:
        scale = scales.get(tag, 2.0 ** 6) if prec == 'f16' else 2.0 ** 6
        losses, g, none_set, finite = capture(model, rows, epoch, prec, scale)
    report['tags'][tag] = {'losses': losses, 'grads_finite': finite, 'scale': scale, 'rows': rows,
                           'none_grad_set_equals_frozen': none_set == FROZEN, 'none_grad_set': none_set,
                           'replica_calls': xpert_dp_patch.REPLICA_CALLS['n'] - before['n'],
                           'replica_calls_autocast_on': xpert_dp_patch.REPLICA_CALLS['autocast_on'] - before['autocast_on']}
    out[tag] = {'grads': g, 'losses': losses}
    torch.save(out, os.path.join(SHARED, MODE + '_grads.pt'))     # after every tag: a crash cannot lose what ran
    json.dump(report, open(os.path.join(SHARED, MODE + '_report.json'), 'w'), indent=1)
torch.backends.cuda.enable_mem_efficient_sdp(True)
report['plain_tensor_attributes'] = xpert_dp_patch.plain_tensor_attributes(models[next(iter(models))])

# Timing, dp only: the recipe itself -- published dropout, Adam, their GradScaler, loader included, frozen ten.
if MODE == 'dp':
    del models
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    model = MX.XPertNet(args, config, dev, logger)
    model.init_weights()
    model.to(dev)
    params = dict(model.named_parameters())
    for n in FROZEN:
        params[n].requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=config['train']['train_lr'], weight_decay=config['train']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler()
    it = iter(tr)
    T.train(model, opt, itertools.islice(it, 1), args, config, scaler=scaler, epoch=0)
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
        torch.cuda.reset_peak_memory_stats(d)
    t0 = time.time()
    T.train(model, opt, itertools.islice(it, 5), args, config, scaler=scaler, epoch=0)
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    report['s_train_step'] = (time.time() - t0) / 5
    report['peak_gib_per_gpu'] = [torch.cuda.max_memory_allocated(d) / 2 ** 30 for d in range(torch.cuda.device_count())]
    report['train_batches'], report['val_batches'] = len(tr), len(val)

json.dump(report, open(os.path.join(SHARED, MODE + '_report.json'), 'w'), indent=1)
print('DPPROBE ' + json.dumps({k: report[k] for k in report if k != 'tags'}), flush=True)
