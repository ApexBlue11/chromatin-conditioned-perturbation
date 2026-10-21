# PACKET 096 — CODE: T-E2 A1–A5 (RESULTS §101.4) before its one run
packet_id: 096
created: 2026-10-21
type: **CODE REVIEW** (T-E2 has not run; B is not coded and runs only after T-C's reading)

**Files:** `model/v10c/te2_tests.py` and `test_v10c_te2.py` (c1b28f1). `te_tests.per_cell_r` gains an optional `extra`
covariates argument: with the default, nothing changes, and T-E's 16 tests pass.

## Mapping to §101.4
- **The guard:** H-E2 is recomputed from scratch and must equal `te_reading.json`'s mean r to 1e-9, or the run stops.
- **A1 (CCLE):**
  - **The data:** CCLE log2(TPM + 1) landmarks (`ccle_expr_v11.npz`), with the landmark Entrez order asserted equal to
    the chromatin genes', ranked within cell over finite genes.
  - **The cells:** H-E2's cells with CCLE on > 50 % of genes.
  - **The statistics:**
    - (i) ACT with [CCLE rank, its LCO gene mean] as extra covariates;
    - (ii) CCLE rank as m, its mask = finite;
    - (iii) CCLE as m over genes with both measured, with [ACT, its LCO mean] as extra;
    - the ACT-only baseline on the same cells.
  - **`a1_rule`** follows §101.4: beyond / explained / inconclusive (tested).
- **A2:**
  - per cell, the distinct controls are permuted (`default_rng(0)`) and split in half to give b1 and b2 (rank01);
  - cells with < 2 controls are out;
  - (i) m = b2, mask all true, basal = b1;
  - (ii) H-E2 with basal b1;
  - plus H-E2 with the full basal on the same cells.
- **A3:** H-E2 with `flexible_basal=True` (decile dummies).
- **A4:**
  - transcription inhibitors (frozen IDs in `te2_comparators.json`); proteostasis (class); a magnitude-matched REF set;
  - **the matching:** per cell, REF compounds whose mean row ‖contrast‖ lies in BET rows' IQR there, rows sampled to BET's
    count (`default_rng(0)`);
  - each × ACT tss_50kb, with the paired BET − comparator difference.
- **A5:**
  - E / C-only ACT cells, MINO excluded;
  - leave the six B-lymphoid cells out;
  - plates with ≥ 5 allowed mates (the contrast recomputed with `min_mates=5`).
- **Every A uses 95 % CIs,** cell-cluster bootstrap, 20,000 draws.

## ASKS
1. Is it faithful to §101.4?
   - The CCLE cell rule (> 50 % of genes finite) wasn't registered. Is it acceptable?
   - A2(i) uses "mask all true" for b2. Fine?
2. In A1(iii), ACT enters as a covariate only on genes with both measures. Is that the right construction?
3. Anything to change before the one run?

## Also, outcome-free status (no review needed)
- **The genome-wide chromatin kernels are done:** 4 signal shards plus methylation. Assembly into a v11 chromatin array is
  next.
- **The v11 STRING graph:** experimental + database channels, ≥ 0.400, prior-corrected; 345,962 edges (1633131).
- **CCLE:** expression for 131 cells (07a7d00).
