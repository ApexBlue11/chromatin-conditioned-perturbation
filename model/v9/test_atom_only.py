import torch
import os
import sys

# Ensure imports work when run as script
sys.path.insert(0, os.path.dirname(__file__))

from modules_v9 import _DrugBlock
from model_v9 import LincsV9
from config_v9 import V9Config

def moved(out_a, out_b, pos): return (out_a[:, pos] - out_b[:, pos]).abs().max().item()

def test_atom_only():
    # Setup
    torch.manual_seed(42)
    d = 16
    cfg = V9Config(d_model=d, n_heads=2, dropout=0.0, use_aux=False, expr_encoder='linear')
    cfg.drug_self_attn = True
    cfg.n_layers = 1
    cfg.n_perturb = 2
    cfg.n_genes = 10
    
    blk = _DrugBlock(cfg).eval()
    
    # inputs D of shape (2, 7, d) with key_mask[:, 5:] = True
    D = torch.randn(2, 7, d)
    key_mask = torch.zeros(2, 7, dtype=torch.bool)
    key_mask[:, 5:] = True
    
    # T1 atom_alpha=1.0 equals the default call exactly
    out_default = blk(D, key_mask=key_mask)
    out_1 = blk(D, key_mask=key_mask, atom_alpha=1.0)
    assert torch.equal(out_default, out_1), "T1 failed"
    
    # T2 single block, atom_alpha=0.0: perturb atom 3
    D2 = D.clone()
    D2[:, 3] += 5
    o = blk(D, key_mask=key_mask, atom_alpha=0.0)
    o2 = blk(D2, key_mask=key_mask, atom_alpha=0.0)
    assert moved(o, o2, 1) == 0.0, "T2 failed"
    
    # T3 single block, atom_alpha=0.0: perturb global token
    D3 = D.clone()
    D3[:, 0] += 5
    o3 = blk(D3, key_mask=key_mask, atom_alpha=0.0)
    assert moved(o, o3, 1) > 1e-4, "T3 failed"
    
    # T4 single block, atom_alpha=0.0: perturb atom 3 -> global token moves
    assert moved(o, o2, 0) > 1e-4, "T4 failed"
    
    # T5 the contrast that justifies the operator: with the retired alpha=0.0, T3's and T4's quantities are exactly 0.0
    o_old = blk(D, key_mask=key_mask, alpha=0.0)
    o2_old = blk(D2, key_mask=key_mask, alpha=0.0)
    o3_old = blk(D3, key_mask=key_mask, alpha=0.0)
    assert moved(o_old, o3_old, 1) == 0.0, "T5 (T3 check) failed"
    assert moved(o_old, o2_old, 0) == 0.0, "T5 (T4 check) failed"
    
    # T6 _last_rowsum_maxdev < 1e-6 at atom_alpha in {0, 0.25, 0.5, 0.75}
    for a in [0.0, 0.25, 0.5, 0.75]:
        blk(D, key_mask=key_mask, atom_alpha=a)
        assert blk.attn._last_rowsum_maxdev < 1e-6, f"T6 failed for {a}"
        
    # T7 parameter count asserted equal at all five atom_alpha values
    counts = []
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        c = sum(p.numel() for p in blk.parameters() if p.requires_grad)
        blk(D, key_mask=key_mask, atom_alpha=a)
        c2 = sum(p.numel() for p in blk.parameters() if p.requires_grad)
        counts.append(c2)
    assert len(set(counts)) == 1, "T7 failed"
    
    # T8 padding: perturb padded token -> real tokens 0..4 move by exactly 0.0 at every atom_alpha
    D_pad = D.clone()
    D_pad[:, 5] += 5
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        o_base = blk(D, key_mask=key_mask, atom_alpha=a)
        o_pad = blk(D_pad, key_mask=key_mask, atom_alpha=a)
        diff = (o_base[:, 0:5] - o_pad[:, 0:5]).abs().max().item()
        assert diff == 0.0, f"T8 failed for {a}"
        
    # T9 no non-finite values at any atom_alpha, including sample whose atom tokens are all padded
    key_mask_all_pad = torch.ones(2, 7, dtype=torch.bool)
    key_mask_all_pad[:, 0] = False
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        o_fin = blk(D, key_mask=key_mask, atom_alpha=a)
        assert torch.isfinite(o_fin).all(), f"T9 failed for {a} regular"
        o_fin_pad = blk(D, key_mask=key_mask_all_pad, atom_alpha=a)
        assert torch.isfinite(o_fin_pad).all(), f"T9 failed for {a} all padded"
        
    # T10 single block, influence of atom 3 on atom 1 at atom_alpha in {0, 0.25, 0.5, 0.75, 1}: exactly 0.0 at 0 and strictly increasing after
    print("T10 influence of atom 3 on atom 1:")
    influences = []
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        oa = blk(D, key_mask=key_mask, atom_alpha=a)
        oa2 = blk(D2, key_mask=key_mask, atom_alpha=a)
        infl = moved(oa, oa2, 1)
        print(f"  atom_alpha={a}: {infl}")
        influences.append(infl)
    assert influences[0] == 0.0, "T10 failed at 0"
    for i in range(1, len(influences)):
        assert influences[i] > influences[i-1], "T10 failed strictly increasing"
        
    # T11 full LincsV9, drug_atom_alpha=0.0: perturb atom 3 and measure how much the model's delta output moves
    M_pathway = torch.randn(5, cfg.n_genes)
    model = LincsV9(cfg, M_pathway).eval()
    
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
    
    out_base = model(batch, drug_atom_alpha=0.0)
    batch2 = dict(batch)
    batch2['atoms'] = batch['atoms'].clone()
    batch2['atoms'][:, 2] += 5.0 # perturb atom 3 (index 2 in atoms)
    out_pert = model(batch2, drug_atom_alpha=0.0)
    
    delta_moved = (out_base['delta'] - out_pert['delta']).abs().max().item() if isinstance(out_base, dict) else (out_base - out_pert).abs().max().item()
    print(f"T11 delta output moved: {delta_moved} (Because atom 3 reaches the global token in perturb block 1, and the global token informs genes in perturb block 2)")
    assert torch.isfinite(torch.tensor(delta_moved)), "T11 failed"
    
    # T12 atom_alpha=0.5 together with alpha=0.5 raises ValueError; likewise with diagonal=True
    try:
        blk(D, key_mask=key_mask, atom_alpha=0.5, alpha=0.5)
        assert False, "T12 failed to raise ValueError for alpha=0.5"
    except ValueError:
        pass
        
    try:
        blk(D, key_mask=key_mask, atom_alpha=0.5, diagonal=True)
        assert False, "T12 failed to raise ValueError for diagonal=True"
    except ValueError:
        pass

if __name__ == '__main__':
    test_atom_only()
    print("All tests passed.")
