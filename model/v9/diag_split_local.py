# -*- coding: utf-8 -*-
"""POST-HOC DIAGNOSTIC for RESULTS 80.6, one laptop GPU, 0 GPU-hours. Not a pre-registered test; it decides only what
the redesigned proof should be.

Hypothesis: v6's fp32 DP-vs-single gradient difference (3.5e-4) is batch-SHAPE rounding amplified by an ill-conditioned
loss, not DataParallel semantics. Test without DataParallel: on ONE GPU, compare their train() gradient on 32 rows with
the same 32 rows run as two 16-row forwards whose outputs are concatenated before their loss -- exactly the computation
DataParallel performs, minus the second device. If split-vs-full is also ~1e-4, the hypothesis holds.

Run at two points in weight space: fresh init (as v6) and the released trained checkpoint, where predictions vary
across genes and the loss should be better conditioned.

Their train_xpert imports scanpy and unimol_tools at module scope through utils.py; neither is used by train() or the
loss functions. They are stubbed as EMPTY modules here so the real train() can be imported without installing them into
the CUDA venv (which would downgrade numpy/pandas under every v9 regression). Any use of them would raise.
"""
import contextlib
import sys
import types

import numpy as np
import torch

for name in ('scanpy', 'unimol_tools'):
    m = types.ModuleType(name)
    if name == 'unimol_tools':
        m.UniMolRepr = None
    sys.modules[name] = m

sys.path.insert(0, r'C:\Projects\LINCS\model\v9')
import xpert_native_eval as X  # noqa: E402

E = X.load_everything('cuda')
from torch.utils.data import DataLoader  # noqa: E402
import train_xpert as T  # noqa: E402
import models.model_XPert as MX  # noqa: E402
import xpert_ckpt_patch  # noqa: E402

torch.backends.cuda.enable_flash_sdp(False)                       # as on a T4
import os as _os
if _os.environ.get('DIAG_MATH') == '1':
    torch.backends.cuda.enable_mem_efficient_sdp(False)           # force the math SDPA backend
    print('SDPA: math backend only', flush=True)
args = X.their_args(nfold='split_cold_cell_1', device='cuda')
cfg = E['cfg']
adata = E['ad'].read_h5ad(X.H5AD)
feats = X.drug_feat_dict(X.UNIMOL)
tr = np.where(adata.obs['split_cold_cell_1'].values == 'train')[0][:64]
sub = X.subset(E, adata, np.isin(np.arange(adata.n_obs), tr))
ds = E['MyDataset'](sub, feats, args, cfg, E['logger'], max_value=cfg['dataset']['max_value'],
                    min_value=cfg['dataset']['min_value'])
batch = next(iter(DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)))
import os
FP64 = os.environ.get('DIAG_FP64') == '1'
if os.environ.get('DIAG_ROWS16') == '1' and not FP64:
    batch = [t[:16] for t in batch]
if FP64:
    batch = [t[:16].double() if t.is_floating_point() else t[:16] for t in batch]

xpert_ckpt_patch.apply()

# fp64 only: their get_unimol_drug_feat hard-casts atom features with .float() (model_XPert.py:14); keep the input
# dtype instead, for this diagnostic alone. Identical otherwise.
if FP64:
    def _gudf(x):
        m = x[:, :, 0].long()
        return x[:, :, 2:], x[:, :, 1].long(), ((1.0 - m.unsqueeze(1).unsqueeze(2)) * -10000.0)
    MX.get_unimol_drug_feat = _gudf
ORIG = MX.XPertNet.forward
SPLIT = {'on': False}


def cat(outs):
    a = outs[0]
    if torch.is_tensor(a):
        return torch.cat(outs, 0)
    if a is None:
        return None
    if isinstance(a, dict):
        return {k: cat([o[k] for o in outs]) for k in a}
    if isinstance(a, (tuple, list)):
        return type(a)(cat(list(z)) for z in zip(*outs))
    raise TypeError(type(a))


def fwd(self, data, *a, **k):
    if SPLIT['on'] and self.training and torch.is_grad_enabled():
        h = data[0].shape[0] // 2
        return cat([ORIG(self, [t[:h] for t in data], *a, **k), ORIG(self, [t[h:] for t in data], *a, **k)])
    return ORIG(self, data, *a, **k)


MX.XPertNet.forward = fwd


def zero_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if hasattr(m, 'dropout_p'):
            m.dropout_p = 0.0


def grads(model, sd, epoch, split):
    model.load_state_dict(sd, strict=True)
    SPLIT['on'] = split
    saved = T.autocast
    T.autocast = contextlib.nullcontext
    try:
        T.train(model, torch.optim.SGD(model.parameters(), lr=0.0), [batch], args, cfg,
                scaler=torch.cuda.amp.GradScaler(), epoch=epoch)
    finally:
        T.autocast = saved
        SPLIT['on'] = False
    g = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    for p in model.parameters():
        p.grad = None
    return g


def rel(ga, gb):
    num = sum(float(((ga[n] - gb[n]) ** 2).sum()) for n in gb)
    return (num / sum(float((gb[n] ** 2).sum()) for n in gb)) ** 0.5


dev = torch.device('cuda')
for label in (('released trained checkpoint, FLOAT64, 16 rows',) if FP64 else ('fresh init (seed 0), as v6', 'released trained checkpoint')):
    if label.startswith('fresh'):
        torch.manual_seed(0)
        model = E['XPertNet'](args, cfg, dev, E['logger'])
        model.init_weights()
        model.to(dev)
    else:
        model, _ = X.build_model(E, args, 'cuda')
    zero_dropout(model)
    if FP64:
        model.double()
    model.train()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in (0, 70):
        full = grads(model, sd, epoch, False)
        rep = grads(model, sd, epoch, False)
        spl = grads(model, sd, epoch, True)
        print('%-45s epoch %2d | rel_L2: repeat vs full %.2e | SPLIT 16+16 vs full %.2e'
              % (label, epoch, rel(rep, full), rel(spl, full)), flush=True)
    del model
    torch.cuda.empty_cache()
