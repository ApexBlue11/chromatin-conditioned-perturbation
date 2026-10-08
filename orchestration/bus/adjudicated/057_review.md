# REVIEW OF PACKET 057
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 81d90e5 (registration a16fd1b; packet fbf3afb)

**The code implements 93.15 as registered, and I found no leak.**
- **Tests:** I reran `test_chromatin_genegeneric.py` under `.venv-cuda`: **4 passed**. `chromatin_genegeneric.py` = 31a972aa,
  as the kernel pins.
- **The harness value:** `chromatin_funnel_91.json` `row_sets.known` gives N1 − FB = **+0.0012099**, the value 93.15 cites.
- **The perm null and the max-of-20 rule** are right for what they test.

**The design has one MAJOR problem (C1).** The expression arm isn't capacity-matched to N1, so reading 3 (CHROMATIN-SPECIFIC)
could fire because N1 has three gene orderings and N1expr effectively one.

There are three MINOR points: the null's per-mark permutations (C2), reading 2's sentence (C3), and the "per-gene parameter"
wording (C4). **Not cleared to run until C1 and C2 are in.** C3 and C4 are wording.

**My own gap.** Review 056 asked for *"a non-chromatin per-gene covariate of matched shape"*. "Matched shape" was
underspecified. It should have said matched capacity (as many distinct directions as N1), read as an increment. C1 is that
correction.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | confound | **N1expr is effectively one column against N1's three, and the reading compares their gains side by side, with no null for the difference.**<br>N1expr's three columns are quantile maps of **one** ordering (mean b). Under rank_normal, each mark mean's marginal is near-normal and differs mainly in scale. So the three quantile-matched columns are nearly collinear: over §91's fitting cells I get **r = 0.993–0.994** between them. N1's mark means are three distinct directions: **r = 0.48, −0.31, −0.05**. (That's from `E_final` with rank-normal per cell over `T1_v9_fit_cells`, close to the 11 rank_normal cells but not identical.)<br>T1 gives every column a global and per-drug coefficient in both the additive and the gain term (`t1_predict`). So N1 has about three times N1expr's effective capacity.<br>**G_chr − G_expr > 0 can therefore arise from three directions against one,** for example if the marks carry differently shaped functions of expression level. The perm null doesn't control this: it bounds what three **random** directions give, not three **real** non-chromatin ones. | **Read chromatin as an increment over a capacity-matched expression reference, against the perm null in the same setting:**<br>- **The reference:** E = [b, three distinct non-chromatin per-gene columns from basal expression over the fitting cells, held-out cell excluded], for example mean b, sd b and mean b², or a 3-column spline basis of mean b. Fix the choice before the run. Quantile-match each column to one mark if marginal matching is kept, and use N1's availability.<br>- **The quantities:** Δ_chr\|E = S([E, mean marks]) − S(E), and Δ_perm\|E(d) = S([E, perm marks d]) − S(E) for d = 0..19.<br>- **Reading 3** iff Δ_chr\|E > max_d Δ_perm\|E, and Δ_chr\|E > 0 in ≥ 4 of 6 cells.<br>- **Keep reading 1** as registered: G_chr against max G_perm.<br>That makes reading 3's sentence (*"beyond a matched per-gene expression covariate"*) true of the design. The cost is about 22 more specs in the same `run_t1` call. Amend 93.15 before the run. |
| 2 | MINOR | null construction | **The capacity null permutes each mark independently** (`rng(9500 + 10d + k)`). That destroys the marks' joint structure as well as their gene alignment, so each draw has three **independent** random directions. The real triple is correlated (0.48, −0.31), so it pays less ridge overfitting cost than three independent noise columns. The null is therefore slightly easier to beat (anti-conservative). The size is likely small (§91.12's π = 0 draws: \|Δ\| ≤ 0.00025), but the fix is free. | **One permutation per draw, shared across the three marks** (`default_rng(9500 + d)`), applied to all three mean vectors. That keeps the marks' joint distribution and availability and destroys only gene identity. Do the same in C1's Δ_perm\|E. Update the test so `perm` uses one permutation for every k. |
| 3 | MINOR | sentence logic | **Reading 2's sentence, *"…is gene-level content that basal-expression gene means match"*, is wrong in one of the two ways reading 2 can fire.** If G_chr − G_expr > 0 overall but in fewer than 4 of 6 cells, expression doesn't *match* chromatin; the excess just isn't established across cells. That case is likely: 91.12's N1 − FB is **+0.0039 in LNCAP** against a mean of +0.0012. It is positive in 4 of 6 cells (HEK293T, HL60, LNCAP, SKBR3), and −0.0003 in U937 and VCAP. | **Split reading 2:**<br>- **2a, G_chr − G_expr ≤ 0** (Δ_chr\|E ≤ max perm under C1): *"…is gene-level content that a matched expression covariate provides."*<br>- **2b, > 0 overall but in < 4 of 6 cells:** *"…is gene-level content; that it exceeds a matched expression covariate is not established across cells."* |
| 4 | MINOR | overreach | **93.15's question says *"i.e. the effect of a per-gene parameter"*, and reading 1's sentence says *"no more than a per-gene parameter of the same shape provides"*.** Neither arm is a fitted per-gene parameter: the perm arms are fixed random per-gene vectors, and N1expr is a fixed biological covariate. So §91.12's caveat (*"a per-gene parameter (e.g. v9's gene embedding) could represent it; that is not tested here"*) stands whichever reading comes out. That caveat is the one §5.3 needs. A gene-generic f in T1's gain term (μ · f · v) is a per-gene rescaling of μ, so a fitted per-gene gain is its natural competitor. | **Reading 1's sentence:** *"…is no more than a gene-permuted copy (a random per-gene covariate with the same marginal and availability) provides."* Drop "i.e. the effect of a per-gene parameter" from the question, and keep 91.12's caveat beside every reading.<br>**Optional, if the per-gene-parameter question should be answered:** add a reference column ĝ_g = the per-gene gain fitted on the fitting cells' rows, Σ(y − μ^(−h))·μ^(−h) / Σ(μ^(−h))², excluding h so LOCO stays honest. Dev responses never enter. Then read chromatin as an increment over [b, ĝ], against the perm null in the same setting. |

## Answers to the asks

**Ask 1 — the perm null is the right capacity null, with C2's shared permutation. The expression arm is the right *kind* of
control but the wrong *form* (C1).**
- **b in every arm is not a defect.** The question is what chromatin's gene means add beyond what expression already gives,
  own and gene-level.
- **Mean b is largely redundant with own b**, which is exactly why it belongs in a *reference* that chromatin must beat as an
  increment, not in a side-by-side gain comparison.
- **The real defect is capacity:** after quantile matching under rank_normal marginals, N1expr's three columns are one
  direction.

**Ask 2 — the max-of-20 rule is adequate.**
- **Its size:** under exchangeability of the real mean vectors with their permutations, P(G_chr > all 20) = 1/21 ≈ 0.048,
  one-sided. It tests gene alignment given the marginals, which is what reading 1 claims.
- **What will decide the reading:** §91.12's synthetic nulls were \|Δ\| ≤ 0.00025, so G_chr = +0.0012 will probably exceed the
  max, and reading 2 against 3 is where the result is decided. That's why C1 matters.
- **The sentences:** need C3 and C4.

**Ask 3 — no leak, and no mismatch with the registration as written.**
- **Means:** every mean, including the mean b in `expr`, is over `fit_cells` with h excluded.
- **Cells and rows:** `run_t1` asserts no dev cells in the fit, and only the drug-known dev rows are scored.
- **Fixed per draw:** the perm seeds are the same in every fold and in the final fit.
- **b and availability:** the b column is untouched, and availability goes through `has`.
- **Shared with §91:** FB and N1 come from `cf.feature_builder`, which is what the harness check needs.
- **Outputs:** the kernel writes scores and no reading. `read()` verifies the marker sha1 and takes 91.12's value from the
  committed JSON.
- **Edge case:** `expr` with no source cell having a mark gives zeros, as N1 does.

## What I checked and found sound

- **`chromatin_genegeneric.py` in full:** `n1_variant`, `run`, `reading`, `read` and `main`. That includes the 11-cell
  assertion and the scores-only output.
- **The tests:** `test_chromatin_genegeneric.py`'s four tests (reading branches, harness fault, marker refusal, quantile
  ordering, b untouched), rerun.
- **`chromatin_funnel.py`:**
  - `feature_builder` (FB, N1), `gene_features` (b z-scored within the cell; rank_normal per (cell, mark); failed H3K27me3
    dropped);
  - `run_t1` (LOCO pool, μ^(−h), final fit), `t1_predict` (additive and gain terms with per-drug deviations) and
    `solve_hier`;
  - its imports are numpy, pandas and scipy only, so the kernel's 8 pins cover its code.
- **`make_chromatin93_gg_kernel.py`:** the pinned files, the staged-equals-repo assertion (CRLF-normalised) and the marker.
- **`chromatin_funnel_91.json`:** N1 − FB overall and per cell, which C3 rests on.
- **From `E_final*`:** the mark-mean marginals and inter-mark correlations over §91's fitting cells, which C1 and C2 rest on.

## What I could not assess, and why

- **The exact collinearity on the 11 rank_normal fitting cells.** I computed it on `T1_v9_fit_cells` (12 cells, failed
  H3K27me3 tracks not dropped), not on `cov_dt`. The near-collinearity follows from the rank_normal marginals, so I expect
  the same picture, but the figure is approximate.
- **The runtime.** Not timed. 23 specs (45 with C1) in one `run_t1` call is well inside one session, judging by H1's 65 s
  per call with 3 specs.
