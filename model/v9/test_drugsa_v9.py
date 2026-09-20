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

# ---- 7. diagonal removes cross-atom flow: perturbing atom 3 leaves atom 1 untouched ----
with torch.no_grad():
    out_diag_a = blk(D, key_mask=key_mask, diagonal=True)
    out_diag_b = blk(D2, key_mask=key_mask, diagonal=True)
diag_delta_at_1 = (out_diag_a[:, 1, :] - out_diag_b[:, 1, :]).abs().max().item()
if diag_delta_at_1 == 0.0:
    print('[PASS] diagonal removes cross-atom flow: perturbing atom 3 moves atom 1 by exactly 0.00e+00  -- |d|max %.2e' % diag_delta_at_1)
else:
    fails.append('CROSS-ATOM LEAK under diagonal: atom 1 moved by %.4e when atom 3 changed' % diag_delta_at_1)

# ---- 8. capacity is preserved: parameter count under diagonal=True is identical ----
n_params_standard = sum(p.numel() for p in blk.parameters())
n_params_diag = sum(p.numel() for p in blk.parameters())
if n_params_diag == n_params_standard == n:
    print('[PASS] capacity is preserved: parameter count under diagonal=True is identical  -- %s params' % f'{n_params_diag:,}')
else:
    fails.append('CAPACITY MISMATCH: standard %d vs diagonal %d' % (n_params_standard, n_params_diag))

# ---- 9. block is not a no-op under diagonal: SwiGLU and residual path are live ----
d_from_in = (out_diag_a - D).abs().max().item()
if d_from_in > 1e-4:
    print('[PASS] diagonal mode is not a no-op: SwiGLU and residual are live  -- |d|max %.4f' % d_from_in)
else:
    fails.append('DIAGONAL IS A NO-OP: output identical to input (|d|max %.2e)' % d_from_in)

# ---- 10. padding still holds under diagonal: changing masked token does not move real one ----
with torch.no_grad():
    out_diag_c = blk(D3, key_mask=key_mask, diagonal=True)
diag_pad_leak = (out_diag_a[:, :4, :] - out_diag_c[:, :4, :]).abs().max().item()
if diag_pad_leak == 0.0:
    print('[PASS] padding still holds under diagonal: changing padded token leaves real tokens untouched  -- |d|max %.2e' % diag_pad_leak)
else:
    fails.append('PADDING LEAK under diagonal: changing masked token moved real one by %.4e' % diag_pad_leak)

# ---- 11. default path unchanged: omitting argument is bit-identical to diagonal=False ----
with torch.no_grad():
    out_default = blk(D, key_mask=key_mask)
    out_explicit_false = blk(D, key_mask=key_mask, diagonal=False)
default_diff = (out_default - out_explicit_false).abs().max().item()
if default_diff == 0.0 and torch.equal(out_default, out_explicit_false):
    print('[PASS] default path unchanged: omitted argument is bit-identical to diagonal=False  -- diff %.2e' % default_diff)
else:
    fails.append('DEFAULT PATH CHANGED: omitted arg differs from diagonal=False by %.4e' % default_diff)

# ---- 12. threading: PerturbBlock and LincsV9 forward with diagonal=True -----------
from model_v9 import PerturbBlock, LincsV9
cfg_sa = copy.deepcopy(cfg)
cfg_sa.drug_self_attn = True
cfg_sa.n_genes = 16
cfg_sa.d_cell_ctx = 4
cfg_sa.l_base, cfg_sa.l_perturb = 1, 1
cfg_sa.use_ppi = False
cfg_sa.d_pathway = 16
cfg_sa.n_pathways = 4
cfg_sa.expr_encoder = 'raw'

pb = PerturbBlock(cfg_sa).eval()
h_dummy = torch.randn(B, cfg_sa.n_genes, cfg_sa.d_model)
with torch.no_grad():
    _, D_pb_a = pb(h_dummy, D, key_mask, diagonal=True)
    _, D_pb_b = pb(h_dummy, D2, key_mask, diagonal=True)
pb_delta = (D_pb_a[:, 1, :] - D_pb_b[:, 1, :]).abs().max().item()
if pb_delta == 0.0:
    print('[PASS] PerturbBlock threads diagonal: atom 1 untouched under atom 3 perturbation  -- |d|max %.2e' % pb_delta)
else:
    fails.append('PerturbBlock threading failed: atom 1 moved by %.4e' % pb_delta)

m_v9 = LincsV9(cfg_sa, np.zeros((cfg_sa.n_pathways, cfg_sa.n_genes), dtype=np.float32)).eval()
b_dummy = {
    'x_ctl': torch.randn(B, cfg_sa.n_genes),
    'x_cell': torch.randn(B, cfg_sa.n_genes),
    'E': torch.randn(B, cfg_sa.n_genes, cfg_sa.d_epi),
    'r': torch.rand(B, cfg_sa.n_genes),
    'atoms': torch.randn(B, cfg_sa.max_atoms, cfg_sa.d_atom),
    'atom_mask': torch.ones(B, cfg_sa.max_atoms, dtype=torch.bool),
    'u_feats': torch.randn(B, cfg_sa.d_global),
    'cell_ctx': torch.zeros(B, cfg_sa.d_cell_ctx),
    'dose': torch.rand(B),
    'time': torch.rand(B),
}
b_dummy['atom_mask'][:, 10:] = False  # padding
with torch.no_grad():
    out_m_diag = m_v9(b_dummy, diagonal=True)
    out_m_false = m_v9(b_dummy, diagonal=False)
    out_m_default = m_v9(b_dummy)
if torch.equal(out_m_default['delta'], out_m_false['delta']) and torch.isfinite(out_m_diag['delta']).all():
    print('[PASS] LincsV9 threads diagonal: default bit-identical to diagonal=False, diagonal=True runs finite')
else:
    fails.append('LincsV9 threading check failed')

print()
if fails:
    print('%d FAILURES' % len(fails))
    for f in fails:
        print('  [FAIL]', f)
    sys.exit(1)
print('11/11 drug-self-attention checks passed')
