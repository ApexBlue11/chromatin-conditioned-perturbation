import torch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from modules_v9 import _DrugBlock
from model_v9 import LincsV9
from config_v9 import V9Config

def moved(a, b, pos): return (a[:, pos] - b[:, pos]).abs().max().item()

def test_residual_cuts():
    torch.manual_seed(42)
    d = 16
    cfg = V9Config(d_model=d, n_heads=2, dropout=0.0, use_aux=False, expr_encoder='linear')
    cfg.drug_self_attn = True
    cfg.l_base = 1
    cfg.l_perturb = 2        # V9Config's real fields; the >= 2 blocks T7/T8 need, set explicitly
    cfg.n_genes = 10
    
    blk = _DrugBlock(cfg).eval()
    
    D = torch.randn(2, 7, d)
    key_mask = torch.zeros(2, 7, dtype=torch.bool)
    key_mask[:, 5:] = True
    
    M_pathway = torch.randn(5, cfg.n_genes)
    model = LincsV9(cfg, M_pathway).eval()
    assert len(model.perturb) == 2 and all(b.drug_sa is not None for b in model.perturb), 'T7 needs 2 SA blocks'

    # T1 Default path unchanged
    out_def_1 = blk(D, key_mask=key_mask, atom_alpha=1.0)
    out_gs_1 = blk(D, key_mask=key_mask, atom_alpha=1.0, global_self_only=False)
    assert torch.equal(out_def_1, out_gs_1), "T1 failed: _DrugBlock atom_alpha=1.0"
    
    out_def_0 = blk(D, key_mask=key_mask, atom_alpha=0.0)
    out_gs_0 = blk(D, key_mask=key_mask, atom_alpha=0.0, global_self_only=False)
    assert torch.equal(out_def_0, out_gs_0), "T1 failed: _DrugBlock atom_alpha=0.0"

    batch = {
        'E': torch.zeros(2, cfg.n_genes, 3),
        'r': torch.zeros(2, cfg.n_genes),
        'x_ctl': torch.zeros(2, cfg.n_genes),
        'atoms': torch.randn(2, 6, cfg.d_atom),
        'atom_mask': torch.ones(2, 6, dtype=torch.bool),
        'u_feats': torch.randn(2, cfg.d_global),
        'dose': torch.randn(2),
        'time': torch.randn(2),
        'cell_ctx': torch.zeros(2, cfg.d_cell_ctx)
    }
    batch['atom_mask'][:, 4:] = False
    
    out_lincs_def = model(batch)
    out_lincs_flags = model(batch, drug_global_self_only=False, drug_xattn_global_only=False)
    assert torch.equal(out_lincs_def['delta'], out_lincs_flags['delta']), "T1 failed: LincsV9"

    # T2 Single block, atom_alpha=0.0, global_self_only=True: perturb atom 3
    D2 = D.clone()
    D2[:, 3] += 5
    o = blk(D, key_mask=key_mask, atom_alpha=0.0, global_self_only=True)
    o2 = blk(D2, key_mask=key_mask, atom_alpha=0.0, global_self_only=True)
    assert moved(o, o2, 0) == 0.0, "T2 failed: global token moved"
    
    o_noflag = blk(D, key_mask=key_mask, atom_alpha=0.0)
    o2_noflag = blk(D2, key_mask=key_mask, atom_alpha=0.0)
    assert moved(o_noflag, o2_noflag, 0) > 1e-4, "T2 failed: contrast check"

    # T3 Single block, atom_alpha=1.0, global_self_only=True: perturb atom 3
    o_t3 = blk(D, key_mask=key_mask, atom_alpha=1.0, global_self_only=True)
    o2_t3 = blk(D2, key_mask=key_mask, atom_alpha=1.0, global_self_only=True)
    assert moved(o_t3, o2_t3, 0) == 0.0, "T3 failed: global token moved"
    assert moved(o_t3, o2_t3, 1) > 1e-4, "T3 failed: atoms didn't communicate"

    # T4 Single block, atom_alpha=0.0, global_self_only=True: perturb global token
    D4 = D.clone()
    D4[:, 0] += 5
    o_t4_base = blk(D, key_mask=key_mask, atom_alpha=0.0, global_self_only=True)
    o_t4_pert = blk(D4, key_mask=key_mask, atom_alpha=0.0, global_self_only=True)
    assert moved(o_t4_base, o_t4_pert, 1) > 1e-4, "T4 failed: atoms didn't read global token"

    # T5 _last_rowsum_maxdev
    for a in [0.0, 0.5, 1.0]:
        blk(D, key_mask=key_mask, atom_alpha=a, global_self_only=True)
        assert blk.attn._last_rowsum_maxdev < 1e-6, f"T5 failed for atom_alpha {a}"

    # T6 Padding with global_self_only=True: perturb padded token 5
    D_pad = D.clone()
    D_pad[:, 5] += 5
    o_pad_base = blk(D, key_mask=key_mask, global_self_only=True)
    o_pad_pert = blk(D_pad, key_mask=key_mask, global_self_only=True)
    assert (o_pad_base[:, 0:5] - o_pad_pert[:, 0:5]).abs().max().item() == 0.0, "T6 failed"

    # T7 Acceptance test: full LincsV9, drug_atom_alpha=0.0, both flags True
    batch_rand = dict(batch)
    batch_rand['atoms'] = torch.randn_like(batch['atoms'])
    out_t7_base = model(batch, drug_atom_alpha=0.0, drug_global_self_only=True, drug_xattn_global_only=True)
    out_t7_rand = model(batch_rand, drug_atom_alpha=0.0, drug_global_self_only=True, drug_xattn_global_only=True)
    diff_rand = (out_t7_base['delta'] - out_t7_rand['delta']).abs().max().item()
    
    batch_mean = dict(batch)
    # over the BATCH, exactly as alpha_sweep.make_atom_ablated_batch does
    batch_mean['atoms'] = batch['atoms'].mean(dim=0, keepdim=True).expand_as(batch['atoms'])
    out_t7_mean = model(batch_mean, drug_atom_alpha=0.0, drug_global_self_only=True, drug_xattn_global_only=True)
    diff_mean = (out_t7_base['delta'] - out_t7_mean['delta']).abs().max().item()
    
    print(f"T7 diff_rand: {diff_rand}")
    print(f"T7 diff_mean: {diff_mean}")
    assert diff_rand == 0.0, "T7 failed on rand"
    assert diff_mean == 0.0, "T7 failed on mean"

    # T8 Full LincsV9, drug_atom_alpha=0.0, one flag True
    out_t8_xattn_rand = model(batch_rand, drug_atom_alpha=0.0, drug_xattn_global_only=True)
    out_t8_xattn_base = model(batch, drug_atom_alpha=0.0, drug_xattn_global_only=True)
    diff_xattn = (out_t8_xattn_base['delta'] - out_t8_xattn_rand['delta']).abs().max().item()
    
    out_t8_g_rand = model(batch_rand, drug_atom_alpha=0.0, drug_global_self_only=True)
    out_t8_g_base = model(batch, drug_atom_alpha=0.0, drug_global_self_only=True)
    diff_g = (out_t8_g_base['delta'] - out_t8_g_rand['delta']).abs().max().item()
    
    print(f"T8 diff_xattn: {diff_xattn}")
    print(f"T8 diff_g: {diff_g}")
    assert diff_xattn > 1e-6, "T8 failed on xattn only"
    assert diff_g > 1e-6, "T8 failed on global only"

    # T9 No non-finite outputs
    key_mask_all_pad = torch.ones(2, 7, dtype=torch.bool)
    key_mask_all_pad[:, 0] = False
    
    flags_list = [(False, False), (True, False), (False, True), (True, True)]
    for g, x in flags_list:
        o_fin = blk(D, key_mask=key_mask, global_self_only=g)
        assert torch.isfinite(o_fin).all(), "T9 failed on blk normal"
        o_fin_pad = blk(D, key_mask=key_mask_all_pad, global_self_only=g)
        assert torch.isfinite(o_fin_pad).all(), "T9 failed on blk padded"
        
        o_lincs = model(batch, drug_global_self_only=g, drug_xattn_global_only=x)
        assert torch.isfinite(o_lincs['delta']).all(), "T9 failed on model"

    # T10 ValueError
    try:
        blk(D, key_mask=key_mask, global_self_only=True, alpha=0.5)
        assert False, "T10 failed to raise ValueError for alpha=0.5"
    except ValueError:
        pass

    try:
        blk(D, key_mask=key_mask, global_self_only=True, diagonal=True)
        assert False, "T10 failed to raise ValueError for diagonal=True"
    except ValueError:
        pass
        
    cfg_no_sa = V9Config(d_model=d, n_heads=2, dropout=0.0, use_aux=False, expr_encoder='linear')
    cfg_no_sa.drug_self_attn = False
    cfg_no_sa.n_genes = 10
    model_no_sa = LincsV9(cfg_no_sa, M_pathway).eval()
    try:
        model_no_sa(batch, drug_global_self_only=True)
        assert False, "T10 failed to raise ValueError for drug_global_self_only on no_sa model"
    except ValueError:
        pass

    # T11 Threading
    batch_flags = dict(batch)
    batch_flags['drug_global_self_only'] = True
    batch_flags['drug_xattn_global_only'] = True
    
    out_kwargs = model(batch, drug_global_self_only=True, drug_xattn_global_only=True)
    out_batch = model(batch_flags)
    
    assert torch.equal(out_kwargs['delta'], out_batch['delta']), "T11 failed"

if __name__ == '__main__':
    test_residual_cuts()
    print("All tests passed.")
