# REVIEW OF PACKET 041
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 3dcdcd6

**The code is correct on every path I traced.**
- **The paths:** condition means, T1's normal equations and profiled hierarchical ridge, every LOCO exclusion, N1, the
  T2 neighbour tables, T3 and subset scoring.
- **The tests:** I reran `test_chromatin_funnel.py` and `test_chromatin_power.py` and got 17 passed in 16 s.
- **The pins:** all 9 pinned files and the split bundle match the repo and the staged `external/kaggle_chromatin_src`.

**Two design problems should be fixed before the one run.** I confirmed both on your own synthetic world:
- **C1:** the centred conjunct is a cell-sign count. A purely drug-independent planted shift passes T1's full advance
  rule in **10 of 36** cases, because its centred Δ is ≈ 0 to five decimals and the per-cell signs are numerical dust.
  This is my review 040 C2's settle, and I concede it was incomplete.
- **C2:** the power calibration measures the MDE of an easier test than the one that will be read.

Also: A1 should scale its thresholds (C3), M1's random-y floors don't transfer to real data (C4), and the reading should
require the completion marker (C5).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **The centred conjunct counts cell signs, so a drug-independent effect passes it on noise.** I planted P3 (drug-independent shift) at π = 5 % and 10 %, over 6 worlds × 3 draws, using `test_chromatin_funnel.world` and your `plant` / `t1_case`. T1's **full advance rule passed in 10 of 36 cases**: raw Δ was +0.007 to +0.016, and `centred_cells_pos` was ≥ 4 by chance. Under P3, Δ_centred is **+0.00000** (\|Δ\| < 5e-6), with per-cell signs that are coin flips. Under P1 and P2, Δ_centred ≈ raw Δ (+0.001 to +0.007) on **6 of 6** cells. **Two consequences.** (a) In the real reading, a drug-independent chromatin effect would advance to C9 about 28 % of the time. (b) In the calibration, the fault "P3 passes at any π in any draw" fires with probability ≈ 1 − 0.72⁵ ≈ 80 %, voiding T1 spuriously. The sign count alone was my settle in review 040 C2. A magnitude was needed too. | Give the centred conjunct a **magnitude** in both T1 and T2: **Δ_centred ≥ half the raw bar** on the rows of record (+0.0015 on all rows; +0.002 on drug-known rows under C3), **and** > 0 in ≥ 4 of 6 cells. With it, P3's centred ≈ 0 always fails, so the any-pass P3 fault becomes appropriate. Add a test asserting that P3 at π = 5–10 % fails across ≥ 10 world × draw combinations. |
| 2 | MAJOR | stats | **The MDE is computed for an easier test than the one that will be read, and the reading rule binds on it.** The calibration gives T1 `FBf = [b̃, f*]`: 2 columns and 4 per-drug parameters. The real test is `FBC = [b̃, A, K, M]`: 4 columns and 8 per-drug parameters, with real missingness. Under LOCO-chosen ridge, the extra columns add variance and shrinkage, so the real test's MDE is larger. The planted form is also exactly linear in the feature supplied (oracle form). Since MDE ≤ 2 % licenses *"an informative negative"*, an optimistic MDE could license a negative the real test can't support. The same applies to T3 (FBf against FBC). | Calibrate the test **as it will be run**. **FBC\*** is each covered cell's real feature set, with its primary mark replaced by f\* and each other available mark replaced by an independently permuted copy (one shared permutation per mark, as for ξ). Mark availability is unchanged. Build N1 from FBC\*, and do the same for T3. Word the negative as *"… explaining ≥ MDE of the cell-specific residual, of a form linear in the encoded tracks, on these dev cells."* |
| 3 | MINOR | stats | **A1 removes dilution, and with an unchanged bar that's a ×1.32 relaxation.** On the 969 level-3 dev rows, μ is the global mean and δ = 0, so T1's FBC − FB prediction difference is the same for all of a cell's level-3 rows. It carries no drug-conditioned signal; T2 behaves the same way at level 3. If the effect sits on the 3,074 drug-known rows, the all-rows Δ is ≈ 0.76 × the known-rows Δ. So keeping +0.003 on known rows corresponds to ≈ +0.0023 on all rows, below the registered bar and the GPU screen's +0.0034. | Accept A1 with the bars restated for the row set (× 4,043 / 3,074): **T1** Δ ≥ +0.004 or top tercile ≥ +0.008, centred ≥ +0.002 (C1); **T2** Δ_T2 ≥ +0.0026 and s_C − uniform ≥ +0.004. The cell counts stay as registered. `t1_reading` and `t2_reading` currently hardcode 0.003 / 0.006 / 0.002 / 0.003. Pass the bars in per row set, so the funnel and the power calibration read identically. |
| 4 | MINOR | confound | **A2's random-y floors don't transfer, because M1's artefact grows with μ's share of the variance.** M1(a) pairs adjacent doses of one drug, and M1(b) pairs μ-similar drugs. In both, the two residuals `e = y − μ^(−c)` share the −μ term, since the μ's are similar by construction. On random X / X_ctl, μ (a mean over about 25 cells) has ≈ 1/25 of y's variance, hence the small floors (0.009 / 0.069). On real data, μ carries the shared drug response, a far larger share, so the artefact is larger and of unknown size. Reading real M1 against random-y floors would overstate the cell-specific structure. | Use a **split pool**. Fix a split of the other dev-train cells into halves A and B. M1(b) picks neighbours by μ^A and forms every e with μ^B. M1(a) forms the two residuals of a pair with μ from disjoint halves. The shared term is then absent by construction. Report the random-y values as a check: they should be ≈ 0. |
| 5 | MINOR | provenance | **The funnel JSON lands before the power run.** If the power calibration fails, `chromatin_funnel_91.json` sits in `/kaggle/working` without the MDE the reading rule needs. `CHROMATIN91_COMPLETE.json` lists pins but not output hashes. | Write both output files' sha1s into the marker. The reading refuses unless the marker exists and both files match it, as P7's scorer does. |

## Answers to the asks

**Ask 1 — I found no error. What I verified:**
- **`condition_means`:** it removes the row's own cell when that cell is in the pool, counts each cell once, backs off
  only when no other cell has the key, and gives dev rows the full dev-train pool. The LOCO pools exclude the held-out
  cell for fit rows too, mirroring the final fit.
- **T1 algebra:** `build_HB`'s blocks are Σ n f fᵀ, Σ S1 f fᵀ and Σ S2 f fᵀ. Its B is [T0 f, T1 f].
- **`solve_hier`:** I re-derived the profiled closed form: θ solves (ΣH − Σₐ H(H + λ_d)⁻¹H + λ_g)θ = ΣB − Σₐ H(H + λ_d)⁻¹B.
  The δ_d floor is counted per fold, as review 040 asked.
- **N1:** refitted, per available mark, excluding the held-out cell (`feature_builder`). N2 is *applied*, which is right
  for a reported ablation.
- **T2:** the neighbour tables exclude the own cell, the no-shared-mark median fill is per target, and each variant
  tunes its own τ (and β). Since s_BC's β grid includes 0, it nests s_B, so Δ_T2 asks whether a β > 0 chosen by LOCO
  carries to dev. That's fine.
- **T3:** LOCO excludes the held-out cell from `v̄` and from N1.
- **`score`:** it centres and takes terciles within the scored subset.

**Ask 2 — f\* is right for what it targets: it keeps the redundancy with b̃ (ρ) and the between-cell structure, and
destroys the gene link.**
- **What it doesn't match:** the shape of the test, which is C2.
- **The rest is right:**
  - P1–P4 are scaled to π·Var(e) over covered dev-train rows;
  - P4 is implemented as y + α·e⊙f\*;
  - the MDE rule is ≥ 4 of 5 draws on the row set of record.
- **The π = 0 null:** five draws are enough to detect a gross fault, because the raw magnitude conjunct protects it. It
  isn't a false-positive-rate estimate, and shouldn't be called one.
- **The P3 fault:** see C1.

**Ask 3 — my rulings on the amendments:**
- **A1:** accept, with C3's scaled bars. **The drug-known rows are of record**, and all rows stay reported.
- **A2:** replace with C4's split pool.
- **A3:** accept. Planting through μ^(−c) gives the planted gain exactly T1's form. That also makes it maximally
  learnable, which is part of why C2's shape-matching matters.

**Ask 4 — the disclosure is adequate, and no exclusion is needed.**
- **The look carries almost no information.** It covers 39 dev rows (1 %) from code with a wrong fit set, so its
  numbers say essentially nothing about the correct reading.
- **The rules came first.** §91 was committed before the look.
- **Nothing proposed since moves toward a pass.** With C3's scaling, A1 no longer relaxes the bar. C1 and C2 both make
  passes and "informative negative" labels *harder*.
- **Record it** verbatim in §91.11.

**Ask 5 — before the run:**
- implement C1–C3 and C5, and C4 if M1 stays reported;
- add the P3 test from C1 and a shape-matched calibration test;
- regenerate the kernel with new pins and rerun both test files.
- **Cost:** about 80 CPU-minutes on Kaggle, on the free CPU session, which is right.

## What I checked and found sound

- **The P3 measurement:** across 36 synthetic P3 cases, raw Δ is +0.0061 to +0.0156, Δ_centred is ≈ 0, and 10 of 36
  pass. Under P1 and P2 (3 worlds × π 1 % and 5 %), Δ_centred is +0.0010 to +0.0071, on 6 of 6 cells every time. So
  C1's magnitude separates the two cleanly.
- **The kernel:**
  - it's CPU only with no internet;
  - it pins 9 files plus the bundle;
  - the funnel prints no readings, and the power run's stdout is discarded;
  - `run_funnel` asserts 11 covered dev-train cells, without PHH.
- **The real-structure random smoke:** your 969 level-3 rows and the 3 dev neighbour pairs without a shared mark agree
  with what the code and the coverage imply.

## What I could not assess, and why

- **The real data's centred noise under P3.** My demonstration uses your synthetic world, where the decomposition is
  clean. On real data, P3's Δ_centred won't be exactly 0, which is why C1 sets the magnitude at half the raw bar rather
  than at "> 0".
