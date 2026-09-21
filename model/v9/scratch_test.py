import torch
import torch.nn as nn
from modules_v9 import _DrugAttention

def test_drug_attn():
    attn = _DrugAttention(32, 4, 0.0)
    x = torch.randn(2, 5, 32)
    key_mask = torch.tensor([[False, False, False, True, True], [False, False, True, True, True]])
    
    # Check default path
    out_default = attn(x, key_mask=key_mask)
    
    # Check alpha=1.0 path
    out_1 = attn(x, key_mask=key_mask, alpha=1.0)
    assert torch.equal(out_default, out_1), "alpha=1.0 not identical to default"
    
    # Check diagonal path
    out_diag = attn(x, key_mask=key_mask, diagonal=True)
    out_0 = attn(x, key_mask=key_mask, alpha=0.0)
    assert torch.equal(out_diag, out_0), "alpha=0.0 not identical to diagonal=True"
    
test_drug_attn()
print("test_drug_attn passed")
