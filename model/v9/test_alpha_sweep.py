import torch
import numpy as np

from config_v9 import V9Config
from model_v9 import LincsV9

def main():
    cfg = V9Config(drug_self_attn=True, n_genes=10, d_model=32, d_atom=10, d_global=5, d_pathway=5, d_cell_ctx=5, l_base=1, l_perturb=1, use_ppi=False, expr_encoder='raw')
    M = np.zeros((2, 10))
    model = LincsV9(cfg, M).eval()
    
    B, G = 2, 10
    # True means valid in atom_mask
    atom_mask = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
    batch = {
        'E': torch.randn(B, G, cfg.d_epi) if cfg.epi_as_gene_embedding else None,
        'r': torch.randn(B, G) if cfg.epi_as_gene_embedding else None,
        'x_ctl': torch.randn(B, G),
        'atoms': torch.randn(B, 5, cfg.d_atom),
        'atom_mask': atom_mask,
        'u_feats': torch.randn(B, cfg.d_global),
        'dose': torch.randn(B),
        'time': torch.randn(B),
        'cell_ctx': torch.randn(B, cfg.d_cell_ctx),
    }

    # 1. alpha=1.0 bit-identical to default
    out_default = model(batch)
    out_1 = model(batch, drug_alpha=1.0)
    diff1 = (out_default['delta'] - out_1['delta']).abs().max().item()
    assert diff1 == 0.0, f"Test 1 Failed: alpha=1.0 diff {diff1}"
    print(f"Test 1 passed: alpha=1.0 difference is {diff1:.2e}")

    # 2. alpha=0.0 bit-identical to diagonal=True
    out_diag = model(batch, diagonal=True)
    out_0 = model(batch, drug_alpha=0.0)
    diff2 = (out_diag['delta'] - out_0['delta']).abs().max().item()
    assert diff2 == 0.0, f"Test 2 Failed: alpha=0.0 diff {diff2}"
    print(f"Test 2 passed: alpha=0.0 difference is {diff2:.2e}")

    # 3. Monotone info flow
    drug_block = model.perturb[0].drug_sa
    D = torch.randn(2, 6, cfg.d_model) # 1 global + 5 atoms
    key_mask = ~torch.cat([torch.ones(B, 1, dtype=torch.bool), atom_mask], 1) # True where pad
    
    # Perturb atom 3 -> index 3 in D
    D_pert = D.clone()
    D_pert[:, 3, :] += torch.randn(cfg.d_model)
    
    print("Test 3: Monotone information flow")
    prev_diff = -1
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out_base = drug_block(D, key_mask=key_mask, alpha=alpha)
        out_pert = drug_block(D_pert, key_mask=key_mask, alpha=alpha)
        # Measure how much atom 1's output moves -> index 1 in D
        diff = (out_base[:, 1, :] - out_pert[:, 1, :]).abs().sum().item()
        print(f"  alpha={alpha:.2f}: diff={diff:.4e}")
        if alpha == 0.0:
            assert diff == 0.0, f"Test 3 Failed: alpha=0 diff is {diff}, expected 0.0"
        else:
            assert diff > prev_diff, f"Test 3 Failed: not strictly increasing (prev {prev_diff}, now {diff})"
        prev_diff = diff

    # 4. Capacity is preserved.
    # The delegated version counted parameters ONCE, printed the number, and asserted nothing, on the
    # argument that the property is structurally guaranteed. It is -- alpha is not a parameter -- but that
    # is an argument, not a measurement, and this codebase has already been burned by a guard that
    # asserted a quantiser's `fitted == 1.0` while its bins were NaN. A test that cannot fail cannot
    # catch the future change that breaks the property. Count at every alpha and assert equality.
    counts = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _ = drug_block(D, key_mask=key_mask, alpha=alpha)   # exercise the path before counting
        counts[alpha] = sum(p.numel() for p in model.parameters())
    assert len(set(counts.values())) == 1, f"Test 4 FAILED: capacity varies with alpha: {counts}"
    print(f"Test 4 passed: capacity is {counts[1.0]} at every alpha, asserted at {len(counts)} values")

    # 5. Padding holds at every alpha
    # Changing a padded token moves a real token by exactly 0.00e+00
    # In batch, atom 5 (index 4) is padded for both sequences. D index 5.
    D_pad_pert = D.clone()
    D_pad_pert[:, 5, :] += torch.randn(cfg.d_model)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out_base = drug_block(D, key_mask=key_mask, alpha=alpha)
        out_pad = drug_block(D_pad_pert, key_mask=key_mask, alpha=alpha)
        # Check all real tokens (indices 0, 1, 2)
        diff_real = (out_base[:, 0:3, :] - out_pad[:, 0:3, :]).abs().max().item()
        assert diff_real == 0.0, f"Test 5 Failed at alpha={alpha}: diff={diff_real}"
    print("Test 5 passed: padding holds at every alpha")

    # 6. No NaN at any alpha
    # fully-masked padded query row
    D_fully_masked = D.clone()
    # Mask all tokens in sequence 1
    key_mask_full = torch.ones(B, 6, dtype=torch.bool)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out_full_mask = drug_block(D_fully_masked, key_mask=key_mask_full, alpha=alpha)
        assert torch.isfinite(out_full_mask).all(), f"Test 6 Failed at alpha={alpha}: NaNs found"
    print("Test 6 passed: no NaN at any alpha")

if __name__ == '__main__':
    main()
