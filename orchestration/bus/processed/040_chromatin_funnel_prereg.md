# PACKET 040 — PRE-REGISTRATION: the chromatin funnel (RESULTS §91), before any code or data
packet_id: 040
created: 2026-10-02
repo_commit: b81cf8f
type: **PRE-REGISTRATION of a CPU-only screen (0 GPU-h).** It decides which chromatin strategies earn a one-seed GPU dev screen,
each priced in its own later packet. Principal's request: *"try to work on making the chromatin information somehow important,
brainstorm multiple strategies and try to funnel the best ones by using some tests; don't spend too much resources on these"*.
Nothing here touches a test row or P7.

## A. The diagnosis (§91.1)
- **v9's chromatin terms are drug-independent.** They are a per-gene embedding summed into the gene token, plus the signed additive
  head `MLP(E[c,g])·r` (`modules_v9.py` `GeneRepresentation`, `SignedChromatinHead`). That is a fixed per-(cell, gene) offset, the
  same for every drug.
- **§44's ridge couldn't express a drug-dependent or gene-local rule either.** It used chromatin as a 24-SVD-component cell-level
  block over ~40 cells, entering additively. A gene-local rule (each gene's own chromatin, one rule shared across genes) instead
  learns from hundreds of thousands of (cell, gene) points, and applies to a new cell by construction.
- **Coverage on `split_cold_cell_1`:** 18 of 32 training cells have chromatin, and all 6 dev cells do:
  - HEK293T, HL60, LNCAP and SKBR3 have all three marks;
  - VCAP has H3K27ac and H3K27me3;
  - U937 has ATAC only.
  Of the 26 dev-training cells, 12 have chromatin.

## B. Definitions (§91.2, binding)
- **Rows:** `split_cold_cell_1` train rows only (47,509); test rows are never loaded. **Dev rows** are the §85 dev cells: 4,043 rows,
  sha1 `51e7e4ab…`. **Dev-train rows** are the other 43,466, from 26 cells.
- **Truth:** `y = X − X_ctl`.
- **Condition mean μ:** key (pert, dose, time), backing off to (pert, time), then pert, then the global mean. Each cell counts once.
  - For dev rows, μ comes from all dev-train cells.
  - For dev-train fit targets, μ^(−c) excludes the row's own cell.
- **Gene-local features:**
  - `b̃` is basal expression (the cell's mean `X_ctl`), z-scored across genes within the cell;
  - `Ẽ_m` is `E_final` z-scored per (cell, mark), exactly as `xpert_arm.py` does, and 0 where the mark is missing.
- **Score S:** `score_dev.row_pearson`, imported. The headline is the mean over dev rows. Also reported: `centred_r`, per cell, and
  the top tercile of `‖y‖₂`.
- **B0 = μ.**
- **Hyper-parameters** (ridge penalties, τ, β) are chosen by leave-one-cell-out over the 12 covered dev-train cells, never on dev cells.

## C. The three tests and their rules
**T1 — drug-conditioned gene-local rule (S1; S3, S4, S6 inside it).**
- **Model:** `ŷ[r,g] = μ[k,g]·(1 + v_d·f[c,g]) + w_d·f[c,g]`, with `w_d = w + δ_d` and `v_d = v + ε_d`. That is a global rule plus a
  ridge-shrunk per-drug deviation, with δ = ε = 0 when a drug has fewer than 3 covered dev-train cells. It is fitted against
  `y − μ^(−c)`.
- **Feature sets:** FB = {b̃}; FBC = {b̃, Ẽ_A, Ẽ_K, Ẽ_M}; FBC⊥ (chromatin residualised on b̃); FC.
- **Nulls:**
  - **N1:** every cell's Ẽ is replaced by the mean Ẽ over covered dev-train cells, i.e. a gene property identical in every cell;
  - **N2:** each dev cell gets another dev cell's chromatin, via a fixed derangement.
- **Estimand: Δ_T1 = S(FBC) − S(FB).**
- **Advance to "C9" (a chromatin interaction head) requires all of:**
  - Δ_T1 ≥ +0.003 on all dev rows, **or** ≥ +0.006 on the top tercile;
  - positive in ≥ 4 of 6 dev cells;
  - S(FBC) − S(N1) > 0 in ≥ 4 of 6 dev cells.
- **Reported:** the epigenetic-drug stratum (ChEMBL targets HDAC*, BRD2/3/4, EZH2/EED/SUZ12, DNMT*, KDM*, EP300/CREBBP, DOT1L,
  SIRT*), centred scores, and the global rule alone.

**T2 — retrieval by chromatin similarity (S2).**
- **Model:** `ŷ_r = Σ α(c,c')·ȳ[k,c']` over covered dev-train cells, with α = softmax(s/τ).
- **Similarities:**
  - s_C: mean over shared marks of the across-gene Pearson of Ẽ;
  - s_B: the same Pearson on b̃;
  - s_BC = s_B + β·s_C;
  - uniform.
  All variants use the same neighbour set.
- **Estimand: Δ_T2 = S(s_BC) − S(s_B).**
- **Advance to "C10" (a neighbour-response token) requires both, each positive in ≥ 4 of 6 cells:** Δ_T2 ≥ +0.002, and
  S(s_C) − S(uniform) ≥ +0.003.

**T3 — which genes can move in this cell (S5).**
- **Model:** `v[c,g]` is the SD of `y[r,g]` over the cell's rows, z-scored within the cell. It is predicted gene-locally as
  `a·v̄_g + β·f`.
- **Estimand, per dev cell:** ρ_T3 = Pearson_g(v̂_FBC, v) − Pearson_g(v̂_FB, v).
- **Advance to "C11" (a chromatin-conditioned response scale):** ρ_T3 ≥ +0.02 in ≥ 4 of 6 dev cells.

**T0 (reported):** chromatin–b̃ correlations; between-cell chromatin pairs with r > 0.99 (duplicated sources were found in 2026-07);
coverage.

## D. The funnel's exit (§91.7)
- **One script, CPU only:** `model/v9/chromatin_funnel.py`, written by a worker to the PI's contract, with tests that call the code.
  It runs once. A full run longer than ~2 min goes to a free Kaggle CPU session.
- **Survivors:** each passing test advances one strategy to a one-seed dev screen, priced in its own packet.
- **If nothing passes:** a three-way measured negative is reported, beyond §44 and §55.
- **Later adoption** would be development after P7, and needs its own test-cell registration.

The brainstorm (IDEAS A12) also lists S7–S11. S7 is a chromatin direction head (deferred). S8, chromatin edges, was C7 and not
accepted. S9 is enhancer and TF-motif features, which need peak-level data not on disk. S10, imputation, is rejected. S11,
`x_ctl` dropout, runs only if T1 shows non-redundant information.

## ASKS
1. **Estimands and nulls.** Is N1 the right "cell-specific, not a gene prior" null for T1? Is Δ_T2's basal-similarity comparator
   the right bar?
2. **Leakage.** μ^(−c) for fit targets; LOCO selection over covered dev-train cells only; dev cells' own `X_ctl` used as an input,
   since it is available at test time. Is anything missed? In particular, does the per-drug deviation (δ_d) need the ≥ 3-cell floor,
   or a stronger one?
3. **Thresholds.** Is +0.003 (or +0.006 on the top tercile) a sensible necessary condition, given that a GPU screen advances at
   +0.0034?
4. **Is anything mis-ranked or missing from the brainstorm?**
5. **Is any test redundant with what §44/§55/§82.7 already measured?**
