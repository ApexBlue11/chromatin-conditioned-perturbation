# PACKET 041 — CODE REVIEW before the one real run: the §91 chromatin funnel + the §91.9 power calibration; two amendments; a disclosure
packet_id: 041
created: 2026-10-02
repo_commit: 3dcdcd6
type: **CODE + AMENDMENTS, before any real reading.** CPU only (a free Kaggle CPU session), 0 GPU-h.

## A. What is to be reviewed (all PI-written; the workers' versions were rejected — B)
- **`model/v9/chromatin_funnel.py`** (§91 as amended by §91.8 / §91.9). Main pieces:
  - `condition_means`: sparse group sums, leave-own-cell-out, back-off (pert,dose,time) → (pert,time) → pert → global, each cell
    counting once;
  - `suff_stats` / `build_HB` / `solve_hier`: T1's normal equations and the profiled hierarchical ridge, δ_d = 0 below 3 covered
    cells. Tested equal to the stacked least squares;
  - `run_t1`: LOCO over the 11 covered dev-train cells. Each fold recomputes μ with the held-out cell removed from the pool, and
    each (spec, variant) picks its own (κ, κ_d);
  - `run_t2`: each similarity variant tunes its own τ (β for s_BC), with §91.8's no-shared-mark rule;
  - `run_t3` / `t3_fit_predict`: the held-out cell is excluded from the prior and from N1 inside LOCO;
  - `t0`, `m1`, `readings_on` (T1/T2 conjuncts, M3 flags, M4), `score` (centring within the scored rows).
  Every target-derived quantity is recomputed from the `y` passed in.
- **`model/v9/chromatin_power.py`** (§91.9 M2):
  - `f* = ρ·b̃ + √(1−ρ²)·ξ`, with ξ = the cell's own track (K, else A, else M) under one gene permutation shared across cells, and
    ρ = the median within-cell r(track, b̃) over the covered dev-train cells;
  - P1–P4 scaled to π·Var(e) over covered dev-train rows;
  - five draws; `t1_case` / `t3_case`; MDE = the smallest π with ≥ 4 of 5 passes; the instrument faults.
- **Tests:** `test_chromatin_funnel.py` 12 passed and `test_chromatin_power.py` 5 passed, all on synthetic worlds that call the
  code. Two **mutation checks** both fail as they should: removing the own-cell exclusion, and breaking the ridge's profile term.
- **Kernel:** `orchestration/make_chromatin91_kernel.py` generates `kern_chromatin91`, a CPU kernel with no internet. It pins the
  sha1 of all 9 staged files (code + inputs, private dataset `apexblue/lincs-chromatin-funnel`) and of the split bundle
  `1444b253…`, which equals the local payload. It runs the funnel with 4 threads, then the power calibration with 4 workers.

## B. Disclosure: one partial look at real data, from a rejected worker
W27's brief allowed a real-data smoke. Its output, which I read, came from a defective implementation:
- **The defects:** a wrong fit set, LOCO leaks, and T2 comparators tuned with s_BC's τ.
- **The coverage:** 37 of 1,849 drugs, 39 of 4,043 dev rows.
- **The numbers:** T1 Δ +0.0003 (top −0.0001); cells 2 / N1 1 / centred 1; T2 Δ 0.000; s_C − uniform −0.0032; no pass.

The §91 rules had been committed before; nothing has been changed because of it. Every later brief forbade real-data runs.
W27b also failed review (AGENT_REGISTRY): μ was not recomputed from the passed y, unseen drugs were predicted as 0 (NaN), there
was no `main()`, and a test was a `pass` stub. Per ORCHESTRATION §6b, the final module is PI-written.

## C. Structural smoke on a REAL-STRUCTURE, RANDOM-VALUE bundle (no information about the real reading)
The bundle keeps every real row, cell, drug, split and chromatin track, with **random X / X_ctl**:
- **Run:** 400 s locally, no NaN rows; every score ≈ 0; nothing advances. The M3 positive-control flags fire (as they should on
  noise).
- **Coverage:** 3 dev neighbour pairs share no mark, as you counted; no chromatin pair has r > 0.99.
- **Power timing:** one planted T1 case takes 101 s locally and a T3 case 1 s, so the full calibration is ≈ 80 min on a 4-core
  Kaggle CPU.

It found two things that bear on the reading:
1. **969 of 4,043 dev rows (24 %) are drugs absent from every dev-train cell.** Their μ is the global mean, at back-off level 3.
2. **M1(b) reads 0.069 on random y.** Neighbours chosen by μ-similarity share e's own −μ term. M1(a) reads 0.009.

## D. Proposed amendments, before any real reading
- **A1 (row set).** The estimand of record for T1 and T2 becomes the **3,074 drug-known dev rows**: back-off level ≤ 2, i.e. the
  drug has a response in at least one dev-train cell. All 4,043 rows stay reported. The power calibration computes the MDE on both
  row sets, and the reading uses the row set of record.
  - *Why:* a level-3 row carries no information on whether chromatin modulates a **known** drug response. It only dilutes Δ_T1
    and lowers power.
  - *Both are already computed* (`row_sets.all_dev_rows` / `drug_known_rows`). Your ruling fixes which is read.
- **A2 (M1 floor).** M1 is read against its random-y structural floor: (a) 0.009, (b) 0.069.
- **A3 (power design note).** Planting uses μ^(−c), the drug mean T1 fits, rather than a μ that includes the row's own cell, so the
  planted gain has exactly T1's form. The W28 brief said "no own-cell exclusion"; W28 was never run.

## E. After your review
Push the dataset and the CPU kernel. Read the funnel and the power calibration mechanically by §91.3–91.9. T4 (P2's 3 dev
checkpoints with `mc_infer_dev.py --ablate_epi`, local GPU inference) runs after the funnel. Record everything in RESULTS §91.11.

## ASKS
1. **Correctness.** Is anything in `condition_means`, the T1 algebra / LOCO / variant tuning, the T2 neighbour tables, T3, M1,
   or the subset scoring wrong? Is any exclusion missed?
2. **Power design.** Is `f*`'s construction right, given that the gene permutation is shared and ρ comes from real tracks? Are
   the P1–P4 scaling and the MDE rule right? Is the T1 null-check (5 draws, π = 0) enough?
3. **A1–A3:** accept, amend or reject. Which row set is of record?
4. Is the disclosure (B) adequate, and does it require anything (e.g. excluding those 39 dev rows from the reading)?
5. Anything before the run?
