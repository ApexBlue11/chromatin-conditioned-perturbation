# -*- coding: utf-8 -*-
"""Verification tests for the 2x2 interaction harness [TASK W5].

Validates the harness on a small untrained LincsV9 model:
  1. All four cells (S11, S01, S10, S00) return finite scores.
  2. S00 != S01 and S10 != S11 (the diagonal flag actually changes outputs/scores).
  3. The interaction equals (S11 - S01) - (S10 - S00) recomputed independently, to 1e-12.
  4. With drug_self_attn=False, the harness reports diagonal-unavailable rather than producing numbers.
"""
import copy
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
os.chdir(HERE)

from config_v9 import V9Config
from model_v9 import LincsV9
from interaction_2x2 import (
    pearson_rows,
    make_atom_ablated_batch,
    check_diagonal_available,
    evaluate_chunks_2x2,
    boot_2x2,
    run_interaction_2x2,
    DiagonalUnavailableError,
)


def build_test_model(drug_self_attn=True, seed=42):
    """Construct a lightweight untrained LincsV9 model for testing."""
    torch.manual_seed(seed)
    cfg = V9Config()
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.d_ff = 128
    cfg.dropout = 0.0
    cfg.drug_self_attn = drug_self_attn
    cfg.n_genes = 16
    cfg.d_cell_ctx = 4
    cfg.l_base = 1
    cfg.l_perturb = 1
    cfg.use_ppi = False
    cfg.d_pathway = 16
    cfg.n_pathways = 4
    cfg.expr_encoder = 'raw'

    # Non-trivial pathway projection matrix [P, G]
    M_dummy = np.zeros((cfg.n_pathways, cfg.n_genes), dtype=np.float32)
    for p in range(cfg.n_pathways):
        M_dummy[p, p * 4 : (p + 1) * 4] = 1.0

    model = LincsV9(cfg, M_dummy).eval()
    return model, cfg


def build_synthetic_chunks(cfg, n_chunks=2, batch_size=16, seed=42):
    """Construct synthetic batches matching the schema of collate_v9."""
    rng = np.random.default_rng(seed)
    chunks = []
    for c_idx in range(n_chunks):
        B = batch_size
        atoms = torch.from_numpy(rng.standard_normal((B, cfg.max_atoms, cfg.d_atom)).astype(np.float32))
        atom_mask = torch.ones(B, cfg.max_atoms, dtype=torch.bool)
        atom_mask[:, 10:] = False  # Ragged padding: tokens 10..max_atoms are masked

        chunk = {
            'x_ctl': torch.from_numpy(rng.standard_normal((B, cfg.n_genes)).astype(np.float32)),
            'x_cell': torch.from_numpy(rng.standard_normal((B, cfg.n_genes)).astype(np.float32)),
            'E': torch.from_numpy(rng.standard_normal((B, cfg.n_genes, cfg.d_epi)).astype(np.float32)),
            'r': torch.from_numpy(rng.uniform(0.0, 1.0, (B, cfg.n_genes)).astype(np.float32)),
            'atoms': atoms,
            'atom_mask': atom_mask,
            'u_feats': torch.from_numpy(rng.standard_normal((B, cfg.d_global)).astype(np.float32)),
            'cell_ctx': torch.zeros(B, cfg.d_cell_ctx, dtype=torch.float32),
            'dose': torch.from_numpy(rng.uniform(0.1, 10.0, B).astype(np.float32)),
            'time': torch.from_numpy(rng.uniform(6.0, 48.0, B).astype(np.float32)),
            'y_delta': torch.from_numpy(rng.standard_normal((B, cfg.n_genes)).astype(np.float32)),
        }
        chunks.append(chunk)
    return chunks


def main():
    print("=== Testing 2x2 Interaction Harness [TASK W5] ===\n", flush=True)
    fails = []

    # Build model with drug_self_attn=True
    model_sa, cfg_sa = build_test_model(drug_self_attn=True)
    chunks = build_synthetic_chunks(cfg_sa, n_chunks=2, batch_size=16)

    # Run harness on drug_self_attn=True model
    res = run_interaction_2x2(model_sa, chunks, n_boot=500, seed=0)
    scores = res['scores']
    s11, s01, s10, s00 = scores['S11'], scores['S01'], scores['S10'], scores['S00']

    # --------------------------------------------------------------------------
    # Check 1: all four cells return finite scores
    # --------------------------------------------------------------------------
    all_finite = all(np.isfinite(s) for s in [s11, s01, s10, s00]) and np.isfinite(res['interaction'])
    if all_finite:
        print(f"[PASS] Check 1: all four cells return finite scores "
              f"(S11={s11:+.5f}, S01={s01:+.5f}, S10={s10:+.5f}, S00={s00:+.5f}, "
              f"interaction={res['interaction']:+.5f})", flush=True)
    else:
        fails.append(f"Check 1 failed: non-finite scores found: {scores}")

    # --------------------------------------------------------------------------
    # Check 2: S00 != S01 and S10 != S11 (diagonal flag actually changes something)
    # --------------------------------------------------------------------------
    diag_changes_intact = (s10 != s11)
    diag_changes_ablated = (s00 != s01)
    dY_ctx = res['dY_max']['dY_max_context']
    if diag_changes_intact and diag_changes_ablated and dY_ctx > 1e-4:
        print(f"[PASS] Check 2: diagonal flag changes scores and activations "
              f"(S10 != S11: {s10:+.5f} != {s11:+.5f}, S00 != S01: {s00:+.5f} != {s01:+.5f}, "
              f"|dY|max_context={dY_ctx:.4f})", flush=True)
    else:
        fails.append(f"Check 2 failed: diagonal flag did not produce different scores "
                     f"(S10==S11: {s10==s11}, S00==S01: {s00==s01}, dY_max_context={dY_ctx})")

    # --------------------------------------------------------------------------
    # Check 3: interaction equals (S11 - S01) - (S10 - S00) independently, to 1e-12
    # --------------------------------------------------------------------------
    recomputed_inter = (s11 - s01) - (s10 - s00)
    diff = abs(res['interaction'] - recomputed_inter)
    if diff < 1e-12:
        print(f"[PASS] Check 3: interaction equals (S11 - S01) - (S10 - S00) to 1e-12 "
              f"(harness={res['interaction']:.12f}, recomputed={recomputed_inter:.12f}, diff={diff:.2e})", flush=True)
    else:
        fails.append(f"Check 3 failed: interaction mismatch: harness={res['interaction']}, "
                     f"recomputed={recomputed_inter}, diff={diff}")

    # --------------------------------------------------------------------------
    # Check 4: with drug_self_attn=False, harness reports diagonal-unavailable
    # --------------------------------------------------------------------------
    model_no_sa, cfg_no_sa = build_test_model(drug_self_attn=False)
    avail, reason = check_diagonal_available(model_no_sa)

    refused_correctly = False
    refusal_msg = ""
    try:
        run_interaction_2x2(model_no_sa, chunks, n_boot=500, seed=0)
        fails.append("Check 4 failed: run_interaction_2x2 did not raise DiagonalUnavailableError")
    except DiagonalUnavailableError as e:
        refusal_msg = str(e)
        if "diagonal mode unavailable for this checkpoint" in refusal_msg:
            refused_correctly = True
        else:
            fails.append(f"Check 4 failed: error message did not contain expected phrase: '{refusal_msg}'")

    if not avail and "diagonal mode unavailable for this checkpoint" in reason and refused_correctly:
        print(f"[PASS] Check 4: with drug_self_attn=False, harness reports diagonal-unavailable "
              f"rather than producing numbers:\n         '{refusal_msg}'", flush=True)
    else:
        fails.append(f"Check 4 failed: check_diagonal_available={avail}, reason='{reason}'")

    print(flush=True)
    if fails:
        print(f"{len(fails)} TEST(S) FAILED:", flush=True)
        for f in fails:
            print(f"  [FAIL] {f}", flush=True)
        sys.exit(1)
    else:
        print("ALL 4 CHECKS PASSED: harness verified on small untrained model.", flush=True)
        sys.exit(0)


if __name__ == '__main__':
    main()
