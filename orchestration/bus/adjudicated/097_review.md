# REVIEW OF PACKET 097 (T-E2 read once, §101.6: H-E2 is explained by expression and is not BET-specific)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: fb1483d (run at 9af31b8; §101.6 at c3cbff9)

**The registered reading is correct, and A1 reproduces exactly with my own code.**
- **My implementation:**
  - CCLE log2(TPM + 1) landmark ranks within cell;
  - its leave-cell-out gene mean;
  - the like-for-like gene set (finite CCLE and E-bar).
- **On the same 11 cells, with the same per-cell values:**
  - ACT-only **−0.0640**;
  - (i) ACT given CCLE **−0.0051**;
  - (ii) CCLE as m **−0.1385**.
- **So the rule gives "explained by expression":** (i)'s CI includes 0, and (ii)'s excludes it below.

**The other checks:**
- A3 and A5 are robust.
- A4 shows the pattern isn't BET-specific: transcription inhibitors and proteostasis are stronger, and magnitude-matched REF is
  null.
- §101.6's withdrawal of the mechanism-keyed route follows.

**The caveats concern three sentences** that say a little more than the data (C1–C3).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording: "with an RNA-seq baseline, chromatin adds nothing" | **(i) is −0.005 [−0.034, +0.024], so a residual ACT association of up to −0.034 isn't excluded.** That is about half of ACT-only's −0.064 on these cells.<br>**Per cell** (descriptive, post hoc), the reduction is uneven:<br>• 7 / 11 stay negative;<br>• the three B-lymphoid cells flip positive (OCI-LY3 +0.064, MINO +0.082, NALM-6 +0.036). MINO is also the DLL1-stimulated track.<br>• The other 8 average **−0.030** (7 / 8 negative), against their ACT-only **−0.082**. | *"With an RNA-seq baseline, chromatin adds nothing detectable on these 11 cells: ACT given CCLE is −0.005 [−0.034, +0.024]. So at least about half, and on the point estimate over 90 %, of the marks' association is RNA-seq expression."*<br>Report the per-cell values. Make no claim from the lymphoid split. |
| 2 | MINOR | wording: A2 "systematic, not random noise" | **The better evidence is A2(ii), not A2(i).**<br>• **A2(ii):** H-E2 is **unchanged** with half the controls (−0.0706 against −0.0709). If random L1000 noise drove it, halving the controls would increase the leakage.<br>• **A2(i):** −0.001 [−0.039, +0.040] is too wide to exclude up to about −0.02 at full basal after the 0.5 scaling.<br>• **ρ ≈ 0.999 is the across-gene split-half reliability.** It is dominated by between-gene levels, so the Spearman–Brown factor isn't the cell-specific reliability. Because the scaled value is bounded within 2× of the raw one, this doesn't change anything. | *"Random L1000 noise doesn't drive it: halving the controls leaves H-E2 unchanged. The residual is systematic: L1000's cross-gene levels against RNA-seq (within-cell corr 0.62–0.71)."* |
| 3 | MINOR | wording: the comparators | **"Global transcription- and proteostasis-perturbing drugs repress the genes most expressed in the cell" extrapolates A1 to the comparators.**<br>• A4 measured the comparators with **ACT** as m.<br>• The CCLE decomposition was run only for BET. | *"show the same ACT pattern, more strongly (expression-driven too, if A1 carries over; untested)"*.<br>Or run A1 for the two comparators on T-E's rows, labelled post hoc and reported. |

## Answers to the asks

**Ask 1 — yes, the reading is fair, and "explained by expression" is correctly applied.**
- **CCLE validity of 0.62–0.71 doesn't weaken it.** CCLE as m (−0.139) is about twice as strong as the marks on the same cells
  and genes, so CCLE carries the relevant information whatever its agreement with L1000's control profile.
- **Eleven cells are enough for the registered CI.**
- Apply C1's wording to the "adds nothing" sentence.

**Ask 2 — as registered, B now has limited value. Amend it before any of its rows are read.**
- **The value problem:** H-E2r would replicate, beyond L1000's basal, a pattern now known to be RNA-seq expression.
- **Add the A1 statistics to B:** ACT given CCLE, and CCLE as m, on the B cells that have CCLE. My counts:
  - **BET inhibitors:** B's 6 cells include HT29, MCF7, MDAMB231 and THP1 with CCLE, which is 4 cells, below B's 5-cell floor.
    So *report only.*
  - **HDAC inhibitors:** HT29, MCF7, MDAMB231, THP1 and H1975 have CCLE, which is 5 cells, so the A1 rule can apply.
- **HDAC's pattern has never been CCLE-tested,** so B on untouched rows is the clean place to do it.
- **A cheaper option before then:** A1 for HDAC on T-E's rows now, labelled post hoc and reported. That would inform v11 sooner.
- Record whichever you choose before T-C's reading.

**Ask 3 — what the data support for v11, as an assessment:**
- **Supported:** at the landmark genes in training cells, an RNA-seq baseline carries response information that L1000's control
  profile doesn't. That holds for transcription-, proteostasis- and BET-class drugs (A1, A4), with a modest effect (CCLE as m
  r ≈ −0.14 for BET).
- **What it supports most directly** is the already-planned **CCLE on / off arm**, with drug-class strata.
- **A hard-coded class × expression route is a separate design choice.** The model's drug-conditioned attention over
  expression-bearing gene tokens can represent that interaction, so a dedicated route needs its own justification.
- **The scope of the proxy role:**
  - only **17 of 217** chromatin cells lack CCLE (16 with ACT);
  - they include **2 of the 7 clean T-C cells (BJAB, CD34)**;
  - and the marks are a weaker stand-in than RNA-seq on the cells where both exist (−0.064 against −0.139).
- **Missingness:** for unseen cells without CCLE, the pre-registered missingness design (§101.1) is what matters.
- **T-C is unaffected:** v10-C uses chromatin, not CCLE.

## What I checked and found sound

- **An independent reproduction of A1** (ACT-only, (i), (ii)) on the 11 CCLE cells, with identical per-cell values.
- **`te2_reading.json` in full:**
  - the guard (−0.07087);
  - A1 with its validity correlations;
  - A2 (split-half reliabilities, the scaled value, (ii));
  - A3;
  - A4 (rows, n cells, BET on common cells, the mechanical readings);
  - A5's three subsets.
- **Coverage counts:** CCLE against chromatin cells, and against the T-C and B cells.

## What I could not assess, and why

- **A1 for the comparators, or for HDAC.** It wasn't run. That is C3 and Ask 2's options.
- **A2 and A4 independently.** I didn't recompute them; I read their JSON.
