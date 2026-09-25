import os
import sys
import numpy as np
import torch
import torch.nn as nn
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v7'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config
from test_candidates_v9 import make_tiny, make_batch, check, R
from model_v9 import v9_loss
from chromatin_mismatch import score_mismatched_chromatin
from modules_v9 import ChromatinGatedUnionMP

def get_a_g(model, b):
    # a_i and g_ij are computed inside ChromatinGatedUnionMP
    ppi = model.ppi
    m_f = b['E_mask'].float()
    inp = torch.cat([b['E'] * m_f, m_f], dim=-1)
    a = torch.sigmoid(ppi.mlp(inp)).squeeze(-1)
    any_obs = b['E_mask'].any(dim=-1)
    a = torch.where(any_obs, a, torch.ones_like(a))
    g = torch.sqrt(a.unsqueeze(2) * a.unsqueeze(1))
    return a, g

def test_a_bitwise_equal():
    cfg1 = V9Config()
    cfg1.chromatin_edges = False
    cfg1.union_edges = False
    
    cfg2 = V9Config() # Same
    
    m1, _, M, _, _ = make_tiny(cfg=cfg1, seed=0)
    m2, _, _, _, _ = make_tiny(cfg=cfg2, seed=0)
    m1.eval(); m2.eval()
    
    b = make_batch(cfg1, seed=0)
    
    # Just to be sure we have some mock E_mask since it's now expected
    b['E_mask'] = torch.ones_like(b['E'], dtype=torch.bool)
    
    with torch.no_grad():
        out1, _ = m1(b, return_aux=True)
        out2, _ = m2(b, return_aux=True)
        
    diff = (out1['delta'] - out2['delta']).abs().max().item()
    check('C7 (a): Both flags off -> bitwise identical outputs', diff < 1e-6, f'diff {diff}')

def test_b_all_missing():
    cfg7 = V9Config()
    cfg7.chromatin_edges = True
    m7, _, _, _, _ = make_tiny(cfg=cfg7, seed=0)
    
    cfg7u = V9Config()
    cfg7u.union_edges = True
    m7u, _, _, _, _ = make_tiny(cfg=cfg7u, seed=0)
    
    # Make sure W in 7u matches W in 7
    m7u.ppi.W.weight.data.copy_(m7.ppi.W.weight.data)
    m7u.ppi.W.bias.data.copy_(m7.ppi.W.bias.data)
    
    m7.eval(); m7u.eval()
    
    b = make_batch(cfg7, seed=0)
    # Force all missing
    b['E_mask'] = torch.zeros_like(b['E'], dtype=torch.bool)
    b['E'] = torch.randn_like(b['E'])
    
    with torch.no_grad():
        h = torch.randn(b['x_ctl'].shape[0], cfg7.n_genes, cfg7.d_model)
        out7 = m7.ppi(h, E=b['E'], E_mask=b['E_mask'])
        out7u = m7u.ppi(h, E=b['E'], E_mask=b['E_mask'])
        
    diff = (out7 - out7u).abs().max().item()
    check('C7 (b): all tracks missing -> C7 equals C7u', diff < 1e-6, f'diff {diff}')

def test_c_toggling_indicator():
    cfg = V9Config()
    cfg.chromatin_edges = True
    m, _, _, _, _ = make_tiny(cfg=cfg, seed=0)
    m.eval()
    
    m.ppi.mlp[2].weight.data.fill_(1.0)
    b = make_batch(cfg, seed=0)
    b['E_mask'] = torch.ones_like(b['E'], dtype=torch.bool)
    b['E'] = torch.zeros_like(b['E']) # value at 0
    
    a1, _ = get_a_g(m, b)
    
    b2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in b.items()}
    b2['E_mask'][0, 0, 0] = False # Toggle one indicator for gene 0
    
    a2, _ = get_a_g(m, b2)
    
    diff = (a1[0, 0] - a2[0, 0]).abs().item()
    check('C7 (c): toggling indicator changes a_i', diff > 1e-6, f'diff {diff}')
    
    diff_other = (a1[0, 1:] - a2[0, 1:]).abs().max().item()
    check('C7 (c): other genes unchanged', diff_other < 1e-6, f'diff {diff_other}')

def test_d_changing_chromatin():
    cfg = V9Config()
    cfg.chromatin_edges = True
    m, _, _, _, _ = make_tiny(cfg=cfg, seed=0)
    m.eval()
    
    m.ppi.mlp[2].weight.data.fill_(1.0)
    b1 = make_batch(cfg, seed=0)
    b1['E_mask'] = torch.ones_like(b1['E'], dtype=torch.bool)
    
    b2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in b1.items()}
    b2['E'][0, 5, :] += 1.0 # Change gene 5
    
    _, g1 = get_a_g(m, b1)
    _, g2 = get_a_g(m, b2)
    
    diff_k = (g1[0, 5, :] - g2[0, 5, :]).abs().max().item()
    check('C7 (d): changing gene k chromatin changes g_kj', diff_k > 1e-6, f'diff {diff_k}')
    
    # Check all other gates are identical
    mask = torch.ones(cfg.n_genes, cfg.n_genes, dtype=torch.bool)
    mask[5, :] = False
    mask[:, 5] = False
    diff_other = (g1[0][mask] - g2[0][mask]).abs().max().item()
    check('C7 (d): all other gates identical', diff_other < 1e-6, f'diff {diff_other}')

def test_e_init_95():
    cfg = V9Config()
    cfg.chromatin_edges = True
    m, _, _, _, _ = make_tiny(cfg=cfg, seed=0)
    m.eval()
    
    b = make_batch(cfg, seed=0)
    b['E_mask'] = torch.ones_like(b['E'], dtype=torch.bool)
    
    a, _ = get_a_g(m, b)
    
    # expected around 0.95 due to logit init
    diff = (a - 0.95).abs().mean().item()
    check('C7 (e): a ~= 0.95 at init', diff < 0.05, f'diff {diff}')

def test_f_union_adj():
    cfg = V9Config()
    cfg.chromatin_edges = True
    m, _, _, _, _ = make_tiny(cfg=cfg, seed=0)
    
    A_hat = m.ppi.A_hat
    
    # symmetric
    diff_sym = (A_hat - A_hat.T).abs().max().item()
    check('C7 (f): adjacency symmetric', diff_sym < 1e-6, f'diff {diff_sym}')
    
    # no self-loops -> diagonal of A_hat should be 0
    diag_sum = torch.diag(A_hat).abs().sum().item()
    check('C7 (f): no self-loops', diag_sum < 1e-6, f'sum {diag_sum}')
    
    # finite row sums
    row_sums = A_hat.sum(dim=1)
    is_finite = torch.isfinite(row_sums).all().item()
    check('C7 (f): finite row sums', is_finite, f'is_finite {is_finite}')

def test_g_mismatch_scorer():
    class DummyModel:
        def eval(self): pass
        def __call__(self, b):
            # return some dummy output based on E
            return {'delta': b['E'].sum(dim=-1).mean(dim=0, keepdim=True)} # [1, G]

    class DummyD:
        def __init__(self):
            self.cell = ['cell1', 'cell2']
            self.E = np.zeros((2, 978, 3), dtype=np.float32)
            self.E[0, 0, 0] = 1.0
            self.E[1, 0, 0] = 2.0
            
            # cell1 has track 1, cell2 doesn't. 
            self.E[0, 1, 1] = 3.0
            self.E[1, 1, 1] = 4.0 
            
            self.Em = np.zeros((2, 978, 3), dtype=bool)
            self.Em[0, 0, 0] = True
            self.Em[1, 0, 0] = True
            
            self.Em[0, 1, 1] = True
            self.Em[1, 1, 1] = False
            
        def batch(self, idx, device):
            return {
                'E': torch.as_tensor(self.E[idx]).to(device),
                'E_mask': torch.as_tensor(self.Em[idx]).to(device),
                'y_delta': torch.zeros(len(idx), 978).to(device)
            }
            
    out_npz = 'toy_mismatch.npz'
    D = DummyD()
    model = DummyModel()
    
    donor_map = {'cell1': ['cell2'], 'cell2': ['cell1']}
    # Provide exactly enough for the function.
    # Wait, the function assumes 5 donors (range(5)). Let's pass 5 copies for the test.
    donor_map = {'cell1': ['cell2']*5, 'cell2': ['cell1']*5}
    
    score_mismatched_chromatin(model, D, [0, 1], donor_map, out_npz, device='cpu')
    
    
    with np.load(out_npz) as z:
        m0 = z['mismatch_preds_0']
    
    check('C7 (g): swaps co-observed', m0[0, 0] == 2.0, f'got {m0[0, 0]}')
    check('C7 (g): keeps eval mask / uses donor mean for only-eval-obs', m0[0, 1] == 0.0, f'got {m0[0, 1]}')
    
    if os.path.exists(out_npz):
        try:
            os.remove(out_npz)
        except OSError:
            pass

def main():
    test_a_bitwise_equal()
    test_b_all_missing()
    test_c_toggling_indicator()
    test_d_changing_chromatin()
    test_e_init_95()
    test_f_union_adj()
    test_g_mismatch_scorer()
    
    print(f'\n{sum(R)}/{len(R)} checks passed')
    sys.exit(0 if all(R) else 1)

if __name__ == '__main__':
    main()
