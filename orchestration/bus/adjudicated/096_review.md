# REVIEW OF PACKET 096 (T-E2 code, A1–A5 of §101.4, before its one run)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: e540c14 (code c1b28f1)

**`te2_tests.py` implements A1, A3, A4 and A5 as registered.**
- **The guard** (H-E2 reproduced to 1e-9) is in place.
- **The `extra` covariates in `per_cell_r`** change nothing by default. T-E's 16 tests pass, and with T-E2's 3 that makes
  **19 / 19** on my run.

**One defect must be fixed before the read-once run: A2(i) can't score any cell as written** (C1, demonstrated).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | code bug (A2(i); Ask 1) | **"Mask all true" for b2 makes A2(i) empty.**<br>• b2 is NaN for every chromatin cell without ≥ 2 distinct training controls: **53 of 217** (the T-C test cells, NCIH1975, and so on).<br>• With `mask = allm` (all true), `lco_mean(b2, allm)` keeps those NaN rows in the sums, so the leave-cell-out mean of b2 is **NaN for every gene**.<br>• `per_cell_r` then finds no gene with a finite mbar, and every cell is skipped.<br>• **My synthetic check with `te_tests`:** a 6-cell b2 with one NaN row scores **0 cells** with the all-true mask, and **5** with `mask = isfinite(b2)`.<br>• **The cost:** the run would report A2(i) as "fewer than 3 cells" and use up the read-once on it. None of the three tests exercises `main()`. | Use `mask = np.isfinite(b2)` for A2(i). Add a test with a NaN b2 row asserting that A2(i) scores the finite cells. |
| 2 | MINOR | A1 rule's gene sets | **The ACT-only baseline and (i) run on slightly different genes.** (i) also requires finite CCLE and its leave-cell-out mean (971 of 978 landmarks), and the ACT-only baseline doesn't.<br>The rule compares \|(i)\| with ½ \|ACT-only\|, so both should be on the same genes. The difference is about 1 %. | Compute the ACT-only baseline with `gene_sel = isfinite(E) & isfinite(Ebar)`. |
| 3 | MINOR | A1 validity diagnostic (report only) | **"Beyond two expression measurements" assumes CCLE measures these lines' expression well.**<br>A poor CCLE match (a different subline or culture) would add little, and ACT would "survive" for that reason alone. | **Report, per A1 cell, corr(CCLE rank, L1000 basal)** over the landmark genes. Flag cells below, say, 0.5. The rule is unchanged. |
| 4 | MINOR | A4 reading | **§101.4's reading** ("a comparator whose r is within the BET CI means not BET-specific") isn't computed. Which CI is meant isn't stated either: T-E's 98.33 % one over 19 cells, or a 95 % one on the comparator's cells. | Fix it now: **BET's 95 % CI over the cells common to BET and that comparator.** Write the reading per comparator into the JSON. Report each comparator's n cells. |

## Answers to the asks

**Ask 1 — faithful, except A2(i)'s mask.**
- **The CCLE cell rule (> 50 % of genes finite) is acceptable.** It is outcome-free. With CCLE covering 971 of 978 landmarks,
  it effectively means "the line has CCLE".
- **"Mask all true" for b2 isn't acceptable.** It breaks A2(i) (C1). Use the finite mask: b2 is defined for every gene of a
  cell that has it.
- **An interpretation note for A2(i):** it uses **half** the controls as its covariate. H-E2 uses all of them, with roughly
  half the noise variance. So A2(i) **overstates** the leakage from random L1000 noise present in H-E2. A near-zero A2(i) is
  strong evidence against noise; a non-zero one should be scaled (Spearman–Brown) before it is compared with −0.071.

**Ask 2 — yes.** A covariate must be defined for every gene in the regression, so (iii) on genes where both measures exist is
the right complete-case construction. ACT covers about 976 of 978 genes in these cells, so nothing material is lost.

**Ask 3 — fix C1 (with its test), then run once.** C2–C4 are small and outcome-free. Do them in the same commit, before the
run.

## What I checked and found sound

- **`te2_tests.py` in full:**
  - **the guard;**
  - **A1:**
    - CCLE loaded on the landmark Entrez order (asserted);
    - within-cell ranks over finite genes;
    - E-bar and A-bar as leave-cell-out means with the right validity masks;
    - (i) through `extra`, (ii) with CCLE's own m-mean, (iii) on mask_both with ACT and A-bar as extra;
    - `a1_rule` = §101.4's rule verbatim;
  - **A2:** the per-cell split of the distinct controls (`default_rng(0)`, deterministic cell order), rank01 halves, c12, and
    (ii) with b1 basal;
  - **A3:** decile basal;
  - **A4:** the frozen 10 transcription-inhibitor IDs, the proteostasis class, and magnitude matching (per cell, per compound
    mean ‖contrast‖ within BET's IQR, rows sampled to BET's count, `default_rng(0)`), paired with BET;
  - **A5:** tier E / C via `chipatlas_selection.json` (MINO out, missing selections excluded), leave-lineage-out, and the
    ≥ 5-mate contrast recomputed.
- **The `te_tests.per_cell_r` diff:** extra covariates also filter genes to finite values; the default is unchanged.
- **`te2_comparators.json` against §101.4's list.**
- **Tests:** 19 / 19 (T-E2's 3 and T-E's 16).

## What I could not assess, and why

- **The run.** T-E2 hasn't run, and I didn't run any part of A1–A5 on real data, to keep them blind.
- **Part B.** It isn't coded yet.
