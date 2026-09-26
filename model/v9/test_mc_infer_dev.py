import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config
from model_v9 import LincsV9
from modules_v7 import StochasticDepth
from xpert_arm import wsd_factor

# Import the code to test
from mc_infer_dev import set_mc_mode, mc_predict

def _tiny_model(dropout=0.0, sd=0.0):
    torch.manual_seed(0)
    cfg = V9Config(d_model=16, d_pathway=8, post_pathway=True, expr_encoder='raw')
    cfg.dropout = dropout
    cfg.stoch_depth = sd
    M = np.zeros((5, 978), np.float32)
    for k in range(5):
        M[k, k * 50:(k + 1) * 50] = 1
    model = LincsV9(cfg, M, None, None)
    return model, cfg

def _mock_D():
    class DummyData:
        def __init__(self):
            pass
        def batch(self, idx, dev):
            B = len(idx)
            g = torch.Generator().manual_seed(0)
            return {'x_ctl': torch.randn(B, 978, generator=g).to(dev),
                    'x_cell': torch.randn(B, 978, generator=g).to(dev),
                    'E': torch.randn(B, 978, 3, generator=g).to(dev),
                    'r': torch.ones(B, 978).to(dev),
                    'E_mask': torch.ones(B, 978, 3).to(dev),
                    'cell_ctx': torch.randn(B, 16, generator=g).to(dev),
                    'atoms': torch.randn(B, 4, 512, generator=g).to(dev),
                    'atom_mask': torch.ones(B, 4, dtype=torch.bool).to(dev),
                    'u_feats': torch.randn(B, 2580, generator=g).to(dev),
                    'dose': torch.randn(B, generator=g).to(dev),
                    'time': torch.randn(B, generator=g).to(dev)}
    return DummyData()

def test_a_set_mc_mode():
    model, _ = _tiny_model(0.1, 0.1)
    
    # det
    counts = set_mc_mode(model, 'det')
    assert counts['Dropout'] == 0
    assert counts['StochasticDepth'] == 0
    assert not model.training
    
    # drop
    counts = set_mc_mode(model, 'drop')
    assert counts['Dropout'] > 0
    assert counts['StochasticDepth'] == 0
    assert not model.training
    
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            assert m.training
        if isinstance(m, StochasticDepth):
            assert not m.training
            
    # full
    counts = set_mc_mode(model, 'full')
    assert counts['Dropout'] > 0
    assert counts['StochasticDepth'] > 0
    assert not model.training
    
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            assert m.training
        if isinstance(m, StochasticDepth):
            assert m.training

def test_b_mc_predict_drop():
    model, _ = _tiny_model(dropout=0.1, sd=0.0)
    D = _mock_D()
    dev = 'cpu'
    
    # det
    (det_pa, det_pd), _ = mc_predict(model, D, [0, 1, 2], dev, 'det', K=8, seed=0)
    
    # drop pass 1
    (drop_pa1, drop_pd1), _ = mc_predict(model, D, [0, 1, 2], dev, 'drop', K=8, seed=0)
    
    # drop pass 2
    (drop_pa2, drop_pd2), _ = mc_predict(model, D, [0, 1, 2], dev, 'drop', K=8, seed=0)
    
    assert np.allclose(drop_pa1, drop_pa2)
    assert np.allclose(drop_pd1, drop_pd2)
    
    assert not np.allclose(det_pa, drop_pa1)
    assert not np.allclose(det_pd, drop_pd1)

def test_c_mc_predict_full():
    model, _ = _tiny_model(dropout=0.0, sd=0.5)
    D = _mock_D()
    dev = 'cpu'
    
    # det
    (det_pa, det_pd), _ = mc_predict(model, D, [0, 1], dev, 'det', K=8, seed=0)
    
    # drop
    (drop_pa, drop_pd), _ = mc_predict(model, D, [0, 1], dev, 'drop', K=8, seed=0)
    
    # full
    (full_pa, full_pd), _ = mc_predict(model, D, [0, 1], dev, 'full', K=8, seed=0)
    
    assert np.allclose(det_pa, drop_pa)
    assert np.allclose(det_pd, drop_pd)
    
    assert not np.allclose(det_pa, full_pa)
    assert not np.allclose(det_pd, full_pd)

def test_d_wsd_factor():
    import math
    def old_wsd(s, steps):
        warm = max(1, int(0.03 * steps))
        if s < warm:
            return s / warm
        return max(0.0, 1 - math.sqrt(max(0, s - 0.8 * steps) / max(1, 0.2 * steps)))
        
    for s in range(100):
        assert wsd_factor(s, 100) == old_wsd(s, 100)
        
    for s in range(120):
        assert wsd_factor(s % 40, 40) == old_wsd(s % 40, 40)

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
