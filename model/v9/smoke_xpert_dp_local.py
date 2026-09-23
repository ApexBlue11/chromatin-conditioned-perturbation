# -*- coding: utf-8 -*-
"""Local smoke test of xpert_dp_patch and of the probe's gradient mechanics, on the one laptop GPU. It cannot test the
two-GPU scatter (that is GUARD G's job on Kaggle); it tests everything else the probe relies on, so a Kaggle launch does
not fail on a mechanical error: the replica path and its device localisation, the plain-tensor inventory, their
train() driven with an lr-0 SGD step and a GradScaler in both precisions and both loss branches, and dropout zeroing."""
import contextlib
import sys

import numpy as np
import torch

sys.path.insert(0, r'C:\Projects\LINCS\model\v9')
import xpert_native_eval as X  # noqa: E402

E = X.load_everything('cuda')
from torch.utils.data import DataLoader  # noqa: E402
import train_xpert as T  # noqa: E402  -- their script; main() is guarded by __name__
import xpert_ckpt_patch  # noqa: E402
import xpert_dp_patch  # noqa: E402

torch.backends.cuda.enable_flash_sdp(False)
args = X.their_args(nfold='split_cold_cell_1', device='cuda')
model, _ = X.build_model(E, args, 'cuda')
adata = E['ad'].read_h5ad(X.H5AD)
feats = X.drug_feat_dict(X.UNIMOL)
tr = np.where(adata.obs['split_cold_cell_1'].values == 'train')[0][:16]
sub = X.subset(E, adata, np.isin(np.arange(adata.n_obs), tr))
ds = E['MyDataset'](sub, feats, args, E['cfg'], E['logger'],
                    max_value=E['cfg']['dataset']['max_value'], min_value=E['cfg']['dataset']['min_value'])
batch = next(iter(DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)))
print('batch type', type(batch).__name__, 'len', len(batch), '| first', tuple(batch[0].shape))

ids = xpert_dp_patch.apply(device_ids=[0])
print('dp ids', ids)
pta = xpert_dp_patch.plain_tensor_attributes(model)
print('plain tensor attributes:', pta)
assert pta == ['drug_HG_embed'], pta
assert not any(k.startswith('module.') for k in model.state_dict())

n = 0
for m in model.modules():
    if isinstance(m, torch.nn.Dropout):
        n += int(m.p > 0); m.p = 0.0
    if hasattr(m, 'dropout_p'):
        n += int(m.dropout_p > 0); m.dropout_p = 0.0
print('dropout sites zeroed:', n)
model.train()
with torch.no_grad():
    rep = torch.nn.parallel.replicate(model, [0])[0]
    assert getattr(rep, '_is_replica', False)
    o_rep = rep(batch)
    o_top = model(batch)
d = max((a - b).abs().max().item() for a, b in zip(o_rep[:3], o_top[:3]))
print('replica path vs top-level path, max |d| over the three outputs: %.3e' % d)
assert d == 0.0 or d < 1e-6, d
assert rep.device == torch.device('cuda:0')

xpert_ckpt_patch.apply()
sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
cfg = E['cfg']
for fp32 in (True, False):
    for epoch in (0, 70):
        model.load_state_dict(sd, strict=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        scaler = torch.cuda.amp.GradScaler()
        saved = T.autocast
        if fp32:
            T.autocast = contextlib.nullcontext
        try:
            losses = T.train(model, opt, [batch], args, cfg, scaler=scaler, epoch=epoch)
        finally:
            T.autocast = saved
        g = [p.grad for p in model.parameters() if p.grad is not None]
        fin = all(torch.isfinite(x).all().item() for x in g)
        moved = max((model.state_dict()[k].float() - sd[k].float()).abs().max().item() for k in sd)
        print('fp32' if fp32 else 'fp16', 'epoch', epoch, '| losses', [round(float(x), 5) for x in losses],
              '| grads', len(g), 'finite', fin, '| weights moved by the lr-0 step: %.1e' % moved)
        assert fin and len(g) > 100 and moved == 0.0
        for p in model.parameters():
            p.grad = None
print('SMOKE PASSED')
