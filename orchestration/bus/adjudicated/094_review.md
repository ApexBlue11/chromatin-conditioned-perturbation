# REVIEW OF PACKET 094 (T-E code, §101 / §101.1, before its one run)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: c4aeb12 (code 059875a)

**`te_tests.py` implements §101 as amended by §101.1.**
- **The contrast** excludes epigenetic, bet_inactive_control and sirtuin_or_natural_product mates, and needs ≥ 2. Excluded-class
  rows are contrasted against the allowed mates.
- **The partial r** removes basal (deciles within the silent tercile for H-E3) and the leave-cell-out gene means of U **and**
  m, over mask = 1 genes only.
- **`decide`** requires all three of §101.1's conditions, with the correct tail for a negative sign.
- **The inputs check out:**
  - the feature indices match `v10c_chromatin.npz`'s names;
  - the mark coverage gives §101's cells (ACC 67, ACT 82, PCG 49, METH 114);
  - the sub-labels survived the class re-freeze (DNMTi 433, BETi 506, JQ1(−) 87, HDACi 2,431 training rows before
    eligibility).
- **Tests:** 13 / 13 pass on my run.

**The caveat:** one change should precede the single run (C1).
- **The problem:** the cell-swap null that enters `decide()` recomputes the m covariate from the swapped matrix. That puts a
  share of each cell's own m into its covariate.
- **The size:** the effect is small, about 1/(n − 1).
- **Why fix it now:** it touches a pass-rule input, and the fix is one argument.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | nulls (Ask 2; a `decide()` input) | **Both nulls recompute the covariates from the permuted m instead of holding them fixed.**<br>**Cell-swap** (in the pass): `per_cell_r(Ms, ms, …)` recomputes `mbar = lco_mean(Ms, ms)`. Row i now holds m_{d(i)}, so its leave-cell-out mean *excludes* m_{d(i)* but *includes* cell i's **own** true m, sitting in another row: about 1/(n − 1) of it, roughly 3–5 % for 22–31 cells. The swapped analysis then partly adjusts for the true m, which nudges the null's tail toward 0 and makes the pass slightly easier.<br>**Gene-permute** (reported): `mbar` becomes a hybrid of permuted eligible cells and unpermuted non-eligible cells.<br>**What a partial correlation calls for:** covariates are part of the design. Only the variable of interest is permuted (Freedman–Lane permutes m's residuals on the covariates). | Compute `ubar` and `mbar` **once from the real data**, and pass them fixed into `per_cell_r` for both nulls. Keep recomputing `ubar` in the calibration, where U really changes.<br>Optionally, make gene-permute the Freedman–Lane form: permute the residuals of m on [basal, ubar, mbar] within mask = 1 genes.<br>Add a test showing that the null's covariates are the real-data ones. |
| 2 | MINOR | calibration wording | **The calibration's "detected" uses only the CI condition. The pass also needs Δspec and the cell-swap condition.**<br>Under a cell-specific plant the cell-swap null sits near 0, and Δspec against an unplanted REF should follow the CI. So the CI is probably binding, but that is untested. | Word it as *"the smallest β at which the CI condition is met"*.<br>Optionally, also report Δspec's CI (planted U against the unplanted REF) per β, so the limit covers two of the three conditions. |
| 3 | MINOR | reporting and implementation choices | **Eligibility is implicit, and two covariate choices aren't in the registration.**<br>• `per_cell_r` silently drops cells with < 30 genes, and `evaluate` returns no `passes` key when fewer than 3 cells remain.<br>• `ubar` averages over every cell with a finite profile, whether or not it has the mark or a basal.<br>• `mbar` averages over **all** cells with the mark measured at the gene, including T-C's test cells' chromatin. That is an input, not an outcome, so it isn't a leak. | **Record per hypothesis:**<br>• the eligible cells;<br>• the skipped cells, with the reason (genes < 30, no basal, no mark) and their gene counts;<br>• an explicit `passes: false, reason: fewer than 3 eligible cells` when that happens.<br>**State in §101.1:** the `ubar` / `mbar` cell sets as implementation choices made before the run. |

## Answers to the asks

**Ask 1 — yes, faithful.**
- **Skipping cells with < 30 eligible genes is acceptable.**
  - The skip is outcome-free: it depends on masks, basal and profile finiteness, never on r.
  - With H-E3's 12 parameters (intercept, 9 decile dummies, ubar, mbar), 30 genes leave 18 df, a sensible floor.
  - Report the skipped cells (C3).
- **The 4,000-draw bootstrap inside the calibration is acceptable** for a reported limit.
  - At 98.33 % that is about 33 draws per tail.
  - The Monte Carlo error in the bound is small next to the β grid's spacing (0.02 → 0.05 → 0.1).

**Ask 2 — not quite.** For a partial correlation, hold the covariates fixed and permute only m (or its residuals). See C1.
- It matters most for the cell-swap null, because it enters the pass.

**Ask 3 — make C1, then run once.**
- C2 and C3 are wording and reporting.
- Also record the wall time and the per-class mate report, which the code already writes.

## What I checked and found sound

- **`te_tests.py` in full:**
  - **`signed_contrast`:** allowed mates, same-compound exclusion, plates with < 2 allowed rows skipped, excluded-class rows
    contrasted against the allowed mates;
  - **`lco_mean`:** the by-hand formula;
  - **`partial_r`:** an intercept plus covariates, residual correlation;
  - **`decile_dummies`:** within the selected genes, one level dropped;
  - **`per_cell_r`:** the gene set mask ∧ finite U / ubar / mbar ∧ gene_sel, ≥ 30;
  - **`boot_ci`, `derangement`;**
  - **`nulls`:** gene-permute within mask = 1; cell-swap by derangement, with the mask intersected;
  - **`calibrate`:** a plant of sign · β · sd(U) · z(m) on mask genes, `ubar` recomputed;
  - **`decide`:** both signs, with p95 / p05;
  - **`evaluate`:** Δspec paired over common cells; H-E3 against R;
  - **`class_profiles`:** ≥ 5 finite rows;
  - **`main`:** training rows only; basal = rank of the mean distinct-control profile; the silent tercile over all genes; I /
    R over all of c's rows; REF = the non-excluded classes; the reported set with the registered signs.
- **The feature names and indices; mask coverage per axis; the sub-labels and sensitivity sets** in the re-frozen
  `compound_classes.json`.
- **`test_v10c_te.py`:** 13 / 13, including the gene-generic effect not passing and masks that differ by cell.

## What I could not assess, and why

- **The run itself.** T-E hasn't run.
- **The 12 / 12 mutation testing.** Not rerun.
