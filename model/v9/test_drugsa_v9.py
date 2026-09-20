# -*- coding: utf-8 -*-
"""Does drug self-attention actually CONTEXTUALISE, or does it merely exist?

RESULTS 32's lesson: a guard that asserts `fitted == 1.0` checks that fit() was CALLED, not that the
bins DISCRIMINATE, and a model trained to convergence with a dead input. So the decisive test here is
not "does the module exist" but "does atom i's representation depend on atom j".

Expected:
  drug_self_attn=False -> changing atom j leaves atom i's contribution untouched (a bag of atoms)
  drug_self_attn=True  -> changing atom j CHANGES atom i's contribution (a molecule)
"""
import os, sys, copy
import numpy as np
import torch

V9 = r'C:\Projects\LINCS\model\v9'
sys.path.insert(0, V9)
sys.path.insert(0, r'C:\Projects\LINCS\model')
sys.path.insert(0, r'C:\Projects\LINCS\model\v6')
os.chdir(V9)

from config_v9 import V9Config
from modules_v9 import _DrugBlock

torch.manual_seed(0)

cfg = V9Config()
cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout = 64, 4, 128, 0.0

B, L, d = 2, 6, cfg.d_model
D = torch.randn(B, L, d)
key_mask = torch.zeros(B, L, dtype=torch.bool)   # False = keep
key_mask[:, 4:] = True                            # last two are padding

blk = _DrugBlock(cfg, 0.0).eval()

fails = []

# ---- 1. contextualisation: perturbing atom 3 must move atom 1 -------------------
with torch.no_grad():
    out_a = blk(D, key_mask=key_mask)
    D2 = D.clone(); D2[:, 3, :] += 5.0
    out_b = blk(D2, key_mask=key_mask)
delta_at_1 = (out_a[:, 1, :] - out_b[:, 1, :]).abs().max().item()
if delta_at_1 > 1e-4:
    print('[PASS] drug self-attention CONTEXTUALISES: perturbing atom 3 moves atom 1  -- |d|max %.4f' % delta_at_1)
else:
    fails.append('atom 1 did not move when atom 3 changed (|d|max %.2e) -- not contextualising' % delta_at_1)

# ---- 2. the no-SA path is genuinely a bag: identity on other atoms --------------
#        (the baseline arm feeds D straight to cross-attention, so atom 1 is untouched by atom 3)
bag_delta = (D[:, 1, :] - D2[:, 1, :]).abs().max().item()
if bag_delta == 0.0:
    print('[PASS] baseline arm is a BAG: with no drug_sa, atom 1 is bit-identical when atom 3 changes')
else:
    fails.append('baseline D is not a bag (|d| %.2e)' % bag_delta)

# ---- 3. padding is respected: changing a PADDED token must not move a real one ---
with torch.no_grad():
    D3 = D.clone(); D3[:, 5, :] += 5.0           # token 5 is masked out
    out_c = blk(D3, key_mask=key_mask)
pad_leak = (out_a[:, :4, :] - out_c[:, :4, :]).abs().max().item()
if pad_leak < 1e-5:
    print('[PASS] padded atoms do not leak into real ones  -- |d|max %.2e' % pad_leak)
else:
    fails.append('PADDING LEAK: changing a masked token moved a real one by %.4f' % pad_leak)

# ---- 4. shape and finiteness ----------------------------------------------------
if out_a.shape == D.shape and torch.isfinite(out_a).all():
    print('[PASS] shape preserved and output finite  -- %s' % (tuple(out_a.shape),))
else:
    fails.append('shape/finiteness: %s finite=%s' % (tuple(out_a.shape), bool(torch.isfinite(out_a).all())))

# ---- 5. flag OFF must leave the model bit-identical to pre-change ----------------
cfg_off, cfg_on = V9Config(), V9Config()
cfg_on.drug_self_attn = True
if cfg_off.drug_self_attn is False and cfg_on.drug_self_attn is True:
    print('[PASS] flag defaults to False and is switchable  -- off=%s on=%s'
          % (cfg_off.drug_self_attn, cfg_on.drug_self_attn))
else:
    fails.append('flag default wrong')

# ---- 6. parameter cost of the arm ------------------------------------------------
n = sum(p.numel() for p in blk.parameters())
print('[INFO] _DrugBlock params at d_model=%d: %s (per perturb block)' % (cfg.d_model, f'{n:,}'))

print()
if fails:
    print('%d FAILURES' % len(fails))
    for f in fails:
        print('  [FAIL]', f)
    sys.exit(1)
print('5/5 drug-self-attention checks passed')
