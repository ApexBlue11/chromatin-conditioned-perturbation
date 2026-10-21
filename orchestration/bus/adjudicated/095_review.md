# REVIEW OF PACKET 095 (T-E read once, §101.3: H-E2 passes)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: ed6e414 (run at 42e5e96; §101.3 at 868940d)

**The pass is sound as registered, and it reproduces exactly with my own code.**
- **My implementation:**
  - vectorised group sums with epigenetic-adjacent mates excluded;
  - ≥ 2 mates, ≥ 5 rows;
  - basal = the rank of the mean distinct-control profile;
  - partial r on [basal, the leave-cell-out gene mean of U, the leave-cell-out gene mean of m] over mask = 1 genes.
- **It gives 19 cells, a mean r of −0.0709, 18 / 19 negative,** with the same per-cell values (HeLa −0.156 … OCI-LY3 0.000).
- **The robustness is real:**
  - 370 of 506 BET-inhibitor rows contribute;
  - review 094's fixed-covariate nulls are in place;
  - the real value lies below all 200 draws of both nulls;
  - the CI is far from 0.
- **It survives programme-level multiplicity.** The sign and window were fixed a priori, and the margin is large.

**The challenges are about what the result is taken to mean.** §101.3 says more than the registered test can show in three
places (C1–C3). C4 bears on the v11 direction.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | interpretation: residual expression (your alternative 1) | **"Beyond baseline expression" means beyond L1000's matched control profile, entered linearly in rank.**<br>• Any measurement error or non-linearity in that covariate leaves residual true expression in the data. ACT, an independent readout of a gene's activity in that cell, would pick it up.<br>• BET (and HDAC) inhibitors repress highly transcribed genes, so residual expression **alone** predicts r < 0.<br>• L1000's own limits make this plausible: bead-level noise; saturation among the top-expressed genes, exactly where BET repression concentrates; and the floor.<br>**So the registered pass can't yet tell apart:**<br>• "chromatin carries information beyond expression";<br>• "a second expression readout adds to a noisy first one".<br>**§101.3 asserts the first:** *"chromatin carries response information that baseline expression does not."*<br>**Magnitude, for calibration:** r ≈ −0.071 is about a planted 0.08 SD effect (β 0.02 → −0.089, β 0.1 → −0.162), so r² ≈ 0.5 % of the cell-specific residual variance. A small residual confound could produce it. | **Qualify now:** *"beyond the L1000 control-profile baseline (linear in rank)"*.<br>**Before the unqualified sentence:** run Ask 3's tests A1–A3 (CCLE as covariate and as m, split-half L1000, flexible basal), registered first.<br>**State the magnitude** (r², ≈ 0.08 SD-equivalent) beside the pass. |
| 2 | MINOR | wording (§101.3 title) | **"Broadly active chromatin" implies broad beats narrow.**<br>The narrow window gives −0.066 against the broad −0.071, and §101.3's own text says "this test does not tell the windows apart". | **Title:** *"a cell's own activation marks (ACT; ±50 kb and promoter alike) predict which genes BET inhibitors repress …"* |
| 3 | MINOR | generalisation (§101.3's "what it means") | **"For drugs acting on chromatin readers and erasers, the cell's own marks carry information that expression does not"** rests on one registered pass (BET inhibitors) plus a **reported, two-sided, unconfirmed** HDAC pattern. | *"For BET inhibitors (H-E2). HDAC inhibitors show a similar, stronger pattern that is reported, not confirmed (Ask 3 B)."* |
| 4 | MINOR | specificity comparator (your alternative 3; bears on v11) | **Δspec against REF (+0.0006) can't exclude a generic "strong or transcriptional perturbation represses active genes" pattern.**<br>• REF is the average non-epigenetic compound, and most of them are weak.<br>• r is scale-free, but it depends on signal-to-noise: weak profiles pull r toward 0 whatever the pattern.<br>• HDAC's stronger, same-signed pattern makes the generic reading live.<br>**If the pattern is generic,** a route keyed on BET / HDAC mechanism (§101.3's v11 direction) is the wrong design. | **Ask 3 A4:** frozen comparators (transcription inhibitors, proteostasis, and a magnitude-matched non-epigenetic set) through the same pipeline, **before** v11's route is designed. |

## Answers to the asks

**Ask 1 — yes, sound as registered.** I found nothing wrong in the code or the data.
- I reproduced the statistic independently (above).
- Review 094's C1 – C3 fixes are in 42e5e96: fixed covariates in the nulls, the calibration wording, eligibility records.
- The registered sign and window were applied.
- The pass meets all three conditions by wide margins.

**Ask 2 — the alternative explanations, ranked:**
1. **Residual expression** (C1): **decisive for "beyond expression"**. Tests A1–A3.
2. **A generic strong or transcriptional perturbation** (C4): **decisive for any BET- or HDAC-specific sentence, and for
   v11's mechanism-keyed route.** Test A4.
3. **Track provenance:** cheap, and should run.
   - Tier E / C tracks only.
   - Confirm that no ACT sample for the 19 cells comes from a BET- or HDAC-perturbation study's treated arm. 087 excluded
     perturbed samples by metadata rules, but check JQ1 / BET / HDAC terms directly. Test A5.
4. **Replication on untouched cells:** **decisive for existence beyond the training cells.** Part B.
5. **HDAC confirmation,** with its direction fixed: Part B.
6. **Plate / batch:** least likely.
   - The contrast removes plate-wide effects.
   - A BET-well-specific artefact would have to correlate with cell-specific ACT beyond basal.
   - Report one sensitivity: plates with ≥ 5 allowed mates.

**Also report a leave-lineage-out check.** Six of the 19 cells are B-lymphoid (OCI-LY3, OCI-LY10, HBL-1, TMD8, MINO,
NALM-6). The 18 / 19 negatives span lineages, so this is a check, not a concern.

**Must run before any sentence beyond §101.3's licensed one:**
- items 1 and 3, for "beyond expression";
- item 2, for anything mechanism-specific;
- item 4, for anything beyond the training cells.

**Ask 3 — the confirmation design.**

**A. Interpretation tests on T-E's own rows.** These are robustness checks, not replication. Register them first, then run them
once.
- **A1, CCLE.**
  - **Cells:** the 19 that have CCLE data; report n. The baseline is ACT-only on the same subset.
  - **CCLE:** log-TPM landmark ranks within the cell, entering with its leave-cell-out gene mean.
  - **Three statistics:**
    - (i) ACT with CCLE added as a covariate;
    - (ii) CCLE as m, without ACT;
    - (iii) CCLE as m, with ACT as a covariate.
  - **The rule, registered now:**
    - *"Beyond two expression measurements"* iff (i)'s 95 % CI excludes 0 below **and** |(i)| ≥ ½ |ACT-only on the
      subset|;
    - *"Explained by expression"* iff (i)'s CI includes 0 **and** (ii)'s CI excludes 0 below;
    - otherwise inconclusive.
- **A2, split-half L1000.**
  - Split each cell's distinct control profiles in two (`default_rng(0)`) to give b1 and b2.
  - Then m = b2 with covariates [b1, ubar, the leave-cell-out gene mean of b2]. This estimates how much residual expression
    random L1000 noise leaves behind.
  - Also run H-E2 with b1 alone as basal.
  - This is internal and free. It must be registered before it runs, so I haven't run it.
- **A3, flexible basal:** decile dummies (or a natural spline with 4 df) of b for H-E2.
- **A4, comparators, classes frozen before the read:**
  - transcription inhibitors (ChEMBL: RNA polymerase II / DNA-directed RNA polymerase / CDK7 / CDK9);
  - proteostasis (HSP90 / proteasome);
  - a **magnitude-matched** non-epigenetic set: per cell, REF compounds whose mean row ‖contrast‖ falls within BET inhibitors'
    interquartile range, sampled to BET's row count with a fixed seed.
  - Each runs through the same pipeline. Report r, and Δspec (BET − comparator).
  - **The reading:** a comparator with r of similar size means the pattern isn't BET-specific.
- **A5, provenance:** tier E / C cells only, and the metadata check from Ask 2 item 3.

**B. Replication on untouched rows.**
- **Which rows are untouched:** the only L1000 rows that T-E didn't read, in cells with ACT, are the 8 T-C test cells:
  - their benchmark test rows;
  - their 38,142 `excluded_test_cell` full-release rows;
  - NCIH1975's 54 rows (= H1975).
- **My counts, before mate exclusion and the ≥ 5-row floor:**
  - **BET inhibitors:** 92 rows over 6 cells (BJAB 8, CD34 14, HT29 16, MCF7 20, MDAMB231 12, THP1 22);
  - **HDAC inhibitors:** 427 over 6 usable cells (HT29 89, MCF7 181, MDAMB231 58, THP1 64, CD34 19, H1975 + NCIH1975 12;
    BJAB 4 is too few);
  - **JQ1(−):** 24 over 5 cells (4–5 each, marginal).
  - All 8 have ACT.
- **Timing:** run only **after T-C's registered reading is complete,** so T-C's rows are not read before T-C. Register now:
  - **H-E2r:** BET inhibitors × ACT tss_50kb, predicted −;
  - **H-E4:** HDAC inhibitors × ACT prom_1kb, predicted −;
  - **H-E4b:** HDAC inhibitors × ACC prom_1kb, predicted −.
- **Covariates:**
  - basal from those cells' controls;
  - **ubar fixed to the gene mean of U over T-E's training cells,** since 6 cells are too few for a leave-cell-out mean;
  - mbar = the gene mean of m over all other cells.
- **Mates, rows, and the rule:**
  - mates and minimum rows as in §101.1;
  - with ≤ 6 cells the cluster bootstrap is coarse, so **register a sign count plus a bound**, e.g. ≥ 5 of 6 cells negative
    **and** the mean r's one-sided 95 % bootstrap bound < 0;
  - report per cell.

**C. External replication (optional; needs the principal's approval to download).** An independent perturbation dataset with
BET / HDAC inhibitors and RNA-seq baselines, in cells with chromatin data, would also remove L1000-specific measurement
artefacts. sci-Plex is one candidate; verify its compound list first.

## What I checked and found sound

- **An independent reproduction of H-E2** (own code, above): cells, mean, sign count and per-cell values all equal.
- **The 42e5e96 diff:**
  - `covariates()` computed once and passed fixed into both nulls;
  - the calibration renamed to "CI condition", with Δspec per β;
  - eligible and skipped gene counts recorded;
  - `passes: False` when fewer than 3 cells remain.
- **`te_reading.json` in full:**
  - the mate report;
  - the three primaries, with nulls, calibration and per-cell values;
  - every reported item (JQ1(−) 6 cells; HDAC; silent genes; narrow / broad; apabetalone).
- **Untouched-row counts by class and cell** in the substrate (Part B).

## What I could not assess, and why

- **CCLE, split-half and comparator results.** I deliberately didn't run them. They should be registered before anyone sees
  them, and that includes me.
- **ACT track metadata for the 19 cells,** as it bears on BET / HDAC study provenance. Not inspected.
