# -*- coding: utf-8 -*-
"""Where does XPert's TRAINING GPU memory go? [RESULTS 77]

v4 died with CUDA OOM at batch 128 on a T4 (14.56 GiB). The T4 is sm_75, so PyTorch's flash SDPA backend (sm_80+) is
unavailable there, as is the real flash_attn their published run used. This measures peak allocated memory for one
training step (autocast fp16 forward + backward, their recipe's precision under --use_gradscaler True) at small batch
sizes, under three SDPA backend settings, and extrapolates to 128. Local GPU, seconds of compute.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r'C:\Projects\LINCS\model\v9')
import xpert_native_eval as X  # noqa: E402

E = X.load_everything('cuda')
from torch.utils.data import DataLoader  # noqa: E402

args = X.their_args(nfold='split_cold_cell_1', device='cuda')
model, _ = X.build_model(E, args, 'cuda')   # released weights; architecture = include_cell_idx True
model.train()
adata = E['ad'].read_h5ad(X.H5AD)
feats = X.drug_feat_dict(X.UNIMOL)
tr = np.where(adata.obs['split_cold_cell_1'].values == 'train')[0][:64]
sub = X.subset(E, adata, np.isin(np.arange(adata.n_obs), tr))
ds = E['MyDataset'](sub, feats, args, E['cfg'], E['logger'],
                    max_value=E['cfg']['dataset']['max_value'], min_value=E['cfg']['dataset']['min_value'])

BACKENDS = {
    'T4-like default (flash off; mem-efficient or math chosen by PyTorch)': dict(flash=False, mem=True, math=True),
    'math only (materialises the full attention matrix)': dict(flash=False, mem=False, math=True),
    'mem-efficient only': dict(flash=False, mem=True, math=False),
}


def step_peak(bs):
    batch = next(iter(DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)))
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(batch)
        loss = sum(o.float().mean() for o in out[:3] if torch.is_tensor(o) and o.dtype.is_floating_point)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30


print('GPU', torch.cuda.get_device_name(0), '| weights resident %.3f GiB' % (torch.cuda.memory_allocated() / 2 ** 30))
res = {}
for name, b in BACKENDS.items():
    torch.backends.cuda.enable_flash_sdp(b['flash'])
    torch.backends.cuda.enable_mem_efficient_sdp(b['mem'])
    torch.backends.cuda.enable_math_sdp(b['math'])
    try:
        peaks = {bs: step_peak(bs) for bs in (2, 4, 8)}
        slope = (peaks[8] - peaks[2]) / 6
        res[name] = (peaks, slope)
        print('\n%s\n   step peak above resident: %s GiB | per sample %.3f GiB | projected at batch 128: %.1f GiB'
              % (name, {k: round(v, 3) for k, v in peaks.items()}, slope, peaks[2] + slope * 126))
    except RuntimeError as e:
        print('\n%s\n   NOT RUNNABLE here: %s' % (name, str(e).splitlines()[0][:160]))
print('\nT4 capacity 14.56 GiB, of which weights + optimizer state take part.')
