# REVIEW OF PACKET 109 (stage 1 read once; §103.4)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 9d69e81 (run at 4e49ffa)

**Every registered number reproduces exactly with my own code,** written from the registration rather than from
`stage1.json`.

| rung | mine | `stage1.json` |
|---|---|---|
| P0 | median 0.0041 over 4,918 | 0.004, 4,918 |
| VD shares (gene / cell / gene × cell) | 0.336 / 0.026 / 0.637 | 0.336 / 0.026 / 0.637 |
| VD noise test | MS_gxc 0.380, bound 0.3253, ratio 1.168 (2,770 pairs) | 1.17 |
| ceiling | ρ̂ 0.1824 (PC3 0.129, MCF7 0.233), √ρ̂ 0.427 (2,764) | 0.18 / 0.43 |
| D1, hairpin (primary) | PC 0.1005 (6 / 6); ACT partial +0.0013 (3 / 6); rung 1 +0.054 | 0.101; +0.001; +0.054 |
| D1, consensus | PC 0.079; ACT −0.0069 (2 / 6) | 0.079; −0.007 |
| B1 | 3,312 pairs, 502 drugs; median 0.4294; top 5 % 0.099 | the same |
| B1 matched vs mismatched | +0.00048 | +0.0005 |

**The code changes after review 108 are correct:**
- `corr_rows_own`: every knockdown loses its own coordinate, by the sum-minus-own form.
- The multi-covariate rank partial (intercept plus the ranks of both covariates).
- `d1_analysis` with the primary's positive control gating.
- Hub candidates from graph genes only, with 0 skips.
- The secondary covariate taken from the substrate's per-cell control profiles.

**The registered verdicts stand:** P0 passes; D1 fails with its positive control passing; B1 passes; B1 is not cell-specific.

**The caveats are three sentences** (C1–C3). One of them rests on a decomposition the packet doesn't report.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording: the ceiling | ***"A model's cell-specific r is bounded below by about √ρ̂ = 0.43"*** says models score at least 0.43. What is bounded below is the **ceiling**: the best achievable r is at least about 0.43. The reading's own later sentence (*"the ceiling … is at least 0.43"*) has it right. | *"The best achievable cell-specific r is at least about √ρ̂ = 0.43 (row-wise 0.39), on PC3 / MCF7."* |
| 2 | MINOR | what D1's positive control shows | **r = log(s_kd / s_ctl) behaves like a signal-to-noise ratio, and part of the positive control is the noise side.**<br>Decomposed, with my centred within-cell Spearman against expression:<br>• **−log s_ctl:** +0.053, **6 / 6 cells**. Higher relative expression means lower relative control spread, i.e. L1000 measurement noise falls with expression.<br>• **log s_kd (hairpin) alone:** +0.032, 5 / 6 cells.<br>• **The ratio:** +0.101.<br>• s_kd and s_ctl correlate 0.59–0.77 per cell.<br>So *"the information is in the data, but expression already holds it"* overstates. The ratio's link to expression is partly lower noise at higher expression and partly a larger knockdown spread. D1's null (ACT adds nothing beyond expression) is unaffected. | **Report the decomposition beside the positive control.**<br>Word the rung: *"ACT's rung-1 association (+0.054) is accounted for by expression; D1's quantity is a knockdown-to-control spread ratio, and its association with expression is partly measurement noise (expression vs control spread −0.053, 6 / 6 cells) and partly knockdown spread (+0.032)."* |
| 3 | MINOR | wording: "not cell-specific" (B1) | **The test can only detect a large cell-specific correspondence.**<br>• The knockdowns' cell-specific part has reliability ρ̂ ≈ 0.18, so a knockdown signature is mostly gene-generic plus noise.<br>• The matched / mismatched CI [−0.008, +0.009] excludes a matched advantage above ≈ 0.009 in percentile.<br>• **"Above degree-matched genes"** is descriptive: the hub median is 0.490 against the target's CI [0.407, 0.457]. There is no paired test. | *"No detectable cell specificity (matched − mismatched +0.0005 [−0.008, +0.009]); with the knockdowns' cell-specific part at ρ̂ ≈ 0.18, only a large cell-specific correspondence could have shown."* |

## Answers to the asks

**Ask 1 — yes, exactly, with my own implementation** (table above).
- **For B1,** my code re-derives everything from the inputs:
  - the train / train compound and cell filter;
  - inhibitory ChEMBL targets;
  - each drug's highest dose at 24 h, else 6 h;
  - correlations with every knockdown's own coordinate excluded;
  - percentiles and the cross-cell comparison.
- **I didn't recompute the bootstrap CIs.** The point estimates and counts match to the last digit, and the CI code was
  reviewed in 108.

**Ask 2:**
- **"A real, modest cell-specific component": right.**
  - MS_gxc / bound = 1.17;
  - σ²_gxc ≥ 0.380 − 0.325 = 0.055, which is 14.5 % of MS_gxc;
  - ρ̂ = 0.18.
- **The VD arithmetic is stated correctly.** E[MS_gxc] = σ²_gxc + σ²_noise, and the bound is ≥ σ²_noise. So the difference is
  a lower bound on σ²_gxc. As stated, it rests on PC3 / MCF7 and a selected gene subset.
- **"Chromatin's null at rung 2":** right. Make the information sentence precise (C2).
- **"B1: a gene-level correspondence, mostly carried by active compounds; not cell-specific":** right, as descriptions.
  - The tas tertiles (0.474 / 0.471 / 0.343) are reported strata, with no test between them.
  - Make "not cell-specific" carry its sensitivity (C3).

**Ask 3 — the two aims follow. Four points for stage 2's registration:**
1. **LOCO on d:** per-row r against d will be small, because d's reliability is ≈ 0.18 and the ceiling ≥ 0.43.
   - Register each candidate's gain against the gene-mean, nearest-cell, k-NN-cell and ridge baselines, with 6-cell
     uncertainty (per-cell signs plus a cell-level CI).
   - Report it relative to the ceiling as well as raw.
2. **Cold-gene with the full network** is the aim that B1 connects to drugs (gene level). It deserves the clearest rule.
3. **Chromatin:**
   - keep rung 3's synthetic-injection positive control before any chromatin gain is read;
   - given C2, include each cell × landmark's control spread (s_ctl) as a covariate or stratum wherever chromatin is read, so
     a noise channel can't masquerade as a chromatin effect.
4. **The drug bridge is gene-level only.** Any claim that a cell representation helps drugs will need its own bridge or test.
   B1 can't supply it.

## What I checked and found sound

- **The 4e49ffa diff:** `corr_rows_own`, the multi-covariate `partial_spearman`, `d1_analysis`, the hairpin-primary wiring,
  the secondary covariate, the graph-only hub.
- **An independent recomputation** of P0, VD, the ceiling, D1 (hairpin and consensus: PC, ACT partial, rung 1) and B1 (pairs,
  drugs, median, top-5 %, matched vs mismatched) from the extracts, the Level-5 drug matrix, the DTI edges, the sign map and
  the substrate split.
- **The D1 positive control's decomposition** into its knockdown and control parts, and the per-cell s_kd / s_ctl
  correlation.

## What I could not assess, and why

- **The bootstrap CIs.** I read them, not recomputed them (C2–C3 don't depend on them).
- **Whether L1000 noise falls with expression for a technical or a biological reason.** C2 rests only on the observed
  association with the control spread.
