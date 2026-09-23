# -*- coding: utf-8 -*-
"""Proof that activation checkpointing of XPert's encoder layers changes no gradient, and how much memory it saves.
[RESULTS 77, packet 011]

v4 died of CUDA OOM at batch 128 on a T4: measured activations are ~0.114 GiB/sample -> ~14.6 GiB at 128 against a
14.56 GiB card. Gradient accumulation is NOT exact for their recipe (epochs < 70 train on batch_weighted_loss, which takes
square roots of WHOLE-BATCH means), and DataParallel needs edits to their forward (it hard-codes self.device).
Checkpointing recomputes activations in the backward pass instead of storing them: same batch, same loss, same
gradients. Applied as a RUNTIME patch to models.model_utils.Encoder / crossEncoder, so their files stay verbatim.

Test: gradients of one training step with and without the patch, same weights, same inputs, same RNG seed, under the
recipe's autocast fp16. The comparison is against the NOISE FLOOR of two unpatched runs, because GPU backward kernels
need not be bitwise deterministic.
"""
import sys

import numpy as np
import torch
import torch.utils.checkpoint as cp

sys.path.insert(0, r'C:\Projects\LINCS\model\v9')
import xpert_native_eval as X  # noqa: E402

E = X.load_everything('cuda')
from torch.utils.data import DataLoader  # noqa: E402

import xpert_ckpt_patch  # noqa: E402  -- the module the kernel deploys, tested as-is

torch.backends.cuda.enable_flash_sdp(False)          # as on a T4 (sm_75): PyTorch then picks mem-efficient SDPA
args = X.their_args(nfold='split_cold_cell_1', device='cuda')
model, _ = X.build_model(E, args, 'cuda')
model.train()
adata = E['ad'].read_h5ad(X.H5AD)
feats = X.drug_feat_dict(X.UNIMOL)
tr = np.where(adata.obs['split_cold_cell_1'].values == 'train')[0][:64]
sub = X.subset(E, adata, np.isin(np.arange(adata.n_obs), tr))
ds = E['MyDataset'](sub, feats, args, E['cfg'], E['logger'],
                    max_value=E['cfg']['dataset']['max_value'], min_value=E['cfg']['dataset']['min_value'])


def one_step(bs, seed=0):
    batch = next(iter(DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)))
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(batch)
        loss = sum(o.float().mean() for o in out[:3] if torch.is_tensor(o) and o.dtype.is_floating_point)
    loss.backward()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss), grads, peak


def maxdiff(g1, g2):
    assert g1.keys() == g2.keys(), 'different parameters received gradients'
    return max((g1[n] - g2[n]).abs().max().item() for n in g1)


BS = 8
l_a1, g_a1, _ = one_step(BS)
l_a2, g_a2, _ = one_step(BS)
mem_unpatched = {bs: one_step(bs)[2] for bs in (2, 4, 8)}
patched = xpert_ckpt_patch.apply()
print('patched classes:', patched)
assert patched == ['Encoder', 'crossEncoder'], patched
l_b, g_b, _ = one_step(BS)
mem_patched = {bs: one_step(bs)[2] for bs in (2, 4, 8)}

floor = maxdiff(g_a1, g_a2)
diff = maxdiff(g_a1, g_b)
print('parameters with gradients: %d' % len(g_a1))
print('loss  unpatched run 1 %.8f | run 2 %.8f | checkpointed %.8f' % (l_a1, l_a2, l_b))
print('max |grad| difference: unpatched vs unpatched (noise floor) %.3e | unpatched vs checkpointed %.3e' % (floor, diff))
ok = diff <= floor or diff == 0.0
print('checkpointing within the noise floor:', ok)
for tag, p in (('unpatched', mem_unpatched), ('checkpointed', mem_patched)):
    slope = (p[8] - p[2]) / 6
    print('%-13s step peak %s GiB | per sample %.4f GiB | projected at batch 128: %.2f GiB'
          % (tag, {k: round(v, 3) for k, v in p.items()}, slope, p[2] + slope * 126))
assert ok, 'checkpointing changed gradients beyond the noise floor'
print('PROOF PASSED')
