# REVIEW OF PACKET 071 (P9 against ridge, RESULTS 96.13)
verdict: SOUND
reviewed_commit: 66d18e0 (result 34a20ae)

**The reading is mechanical and right, and my independent rescore reproduces it.** I wrote my own scorer, sharing no code with
`score_p9.py`, and ran it to scratch on the downloaded files:
- **the inputs:** v9 seed files 0–2 (`deg_pred`), the bundle's `ridge_pred − ctl_true`, the units file and the clean list;
- **the row score:** per-row Pearson on y − ctl, v9 = the mean of the three per-seed Pearsons;
- **the estimand:** per-molecule median difference, unweighted mean, 20,000-draw molecule bootstrap with **my own** RNG seed,
  and a two-sided sign test.

| | packet | my rescore |
|---|---|---|
| clean, mean d_m | +0.1186 [+0.1113, +0.1260] | **+0.1186 [+0.1113, +0.1260]** |
| clean, molecules favouring v9 | 338 / 346, p 6.7e-89 | **338 / 346, p 6.7e-89**, 0 ties |
| full | +0.1160 [+0.1094, +0.1230], 376 / 384, p 5.7e-100 | +0.1160 [+0.1093, +0.1229], 376 / 384, p 5.7e-100 |
| seeds 0 / 1 / 2 (clean) | +0.1187 / +0.1181 / +0.1184 | identical; 338 / 338 / 339 favour |
| ensemble | +0.1242 | +0.1242 |
| row-pooled | clean 0.6417 / 0.5217; full 0.6467 / 0.5294 | identical |
| strata (clean) | +0.127 / +0.096 / +0.085 | +0.1265 / +0.0960 / +0.0848 (≥ 0.999 full: +0.0947) |
| per cell | 34 / 40 | 34 / 40 (clean and full) |
| duplicate lift (row-pooled) | v9 +0.049, ridge +0.075, DiD −0.026 | +0.0487, +0.0748, −0.0261 |

**The inputs check out:**
- all 20 `P9_COMPLETE.json` sha1s match the files;
- the snapshot identities hold for every seed (main = mean of snapshots to 1e-6; `_last` = `_snap2`);
- `deg_pred` = `y_pred − ctl_true`;
- the ridge targets agree exactly;
- the row sets are `5f85ef0b` and `6024dbf8`;
- the log shows `--no_test_metrics`, GUARD 5 and GUARD 6.

The four points below are all MINOR, about wording and reporting, and none touches the reading.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | sentence scope | **The licensed sentence says "on XPert's cold-drug split", but this is fold 1 of 5.** The "not licensed" list excludes other folds. The sentence itself, quoted alone, reads as the whole benchmark. | *"… on fold 1 of XPert's cold-drug split (`split_cold_drug_1`) …"* |
| 2 | MINOR | strata framing | **"The margin is largest for the compounds least like any training compound" is accurate, but it comes mostly from ridge falling.** From the 0.6–0.8 stratum to < 0.6 (clean, molecule level): v9 goes 0.656 → 0.642 (−0.014) and ridge 0.551 → 0.509 (−0.042). v9's own score is lower on the least-similar compounds too. The 0.8–0.999 stratum (n 14, CI 0.052–0.128) overlaps both. | Add the per-model levels beside the margins. Say the larger margin at low similarity reflects mainly ridge's decline, so it isn't evidence that v9 does better in absolute terms on novel chemistry. |
| 3 | MINOR | duplicate lift | (a) **The lifts are row-pooled.** With molecule weighting (mean of per-molecule means): v9 **+0.030**, ridge **+0.055**. The **DiD is stable** at −0.025 against −0.026, but the individual lifts depend on the weighting.<br>(b) **"Ridge, which reads ECFP4 directly, profits more from seeing the molecule than v9 does" is a causal reading.** The composition caveat applies to the DiD too: a model × compound-type interaction can't be separated from the leak. | (a) Report both weightings, or label the lifts "row-pooled".<br>(b) Reword: *"ridge's score rises more on duplicate rows than v9's (DiD −0.026 [−0.044, −0.007]). That is consistent with ridge exploiting exact-structure matches; the contrast also reflects which compounds are duplicates."* |
| 4 | MINOR | per-cell wording | **The 6 non-favouring cells are one molecule.** H1975, NCIH1975, NCIH2073, NCIH508, NCIH596 and SUDHL4 form a panel that profiled only afatinib and erlotinib. On the clean subset each holds erlotinib (`BRD-K70401845`) alone, 5–6 rows. So "6 of 40 cells" is one molecule's evidence, not six independent cells. | Add *"(all six are erlotinib alone, one panel)"*. |

## Answers to the asks

**Ask 1 — yes, mechanical and right.** The table above is my independent reproduction. The verdict follows 96.3 / 96.7 for each
subset:
- the mean is > 0;
- the CI excludes 0 and is 0.015 wide, so not uninformative;
- 97.7 % of molecules favour v9, sign p ≪ 0.01;
- each seed passes alone.

**Ask 2 — the sentence is within the numbers (C1 aside), and the "not licensed" list is complete for 96.3's scope.**
- **One check I ran,** because the project's own 81e7075 concern is drug use: the **cell-centred** version of the same estimand
  (90.2's centring per test cell; not registered for P9, so a diagnostic only).
  - **The result:** clean **+0.1224 [+0.1154, +0.1297], 341 / 346**; full +0.1202, 378 / 384. Centred row-pooled: v9 0.625,
    ridge 0.500.
  - **The scale of the cell-average component:** a test-derived oracle that predicts each cell's mean true Δ scores only
    **0.164** per row on the clean subset.
- **So the margin isn't a cell-average artefact,** and the sentence doesn't need a "not drug-specific" limit. If you report this,
  label it the critic's post hoc diagnostic, not a registered reading.

**Ask 3 — nothing overreaches materially.** Three framings need the qualifiers in C2–C4:
- the strata margin is mainly ridge's decline;
- the duplicate-lift magnitudes depend on the weighting, and the causal wording needs the hedge;
- the 6 cells are one molecule.

The row-pooled 0.6467 placed next to XPert's published 0.645 is explicitly "not compared", which is correct. It should stay so
until O9.

## What I checked and found sound

- **My own scorer,** to scratch: the row scores, molecule mapping, clean mask (sha1 `6024dbf8`), per-molecule medians, bootstrap,
  sign test, per seed, ensemble, row-pooled, strata, per cell, and duplicate lift both ways.
- **The cell-centred diagnostic** and the oracle cell-mean score.
- **The inputs:** `P9_COMPLETE.json` against the files (20 / 20), the snapshot identities, `deg_pred` = `y_pred − ctl_true`, and
  the ridge target agreement.
- **The run:** the log's code-verification, GUARD 5 / 6 and `--no_test_metrics` lines.
- **The record:** the output file's sha1 against its marker (`d7519f60`); the 96.13 text against 96.3 / 96.7 / 97.6 item 5.

## What I could not assess, and why

- **Whether the Kaggle run used exactly the reviewed kernel (937696eb).** I relied on the log's "6 pinned sha1s" line and the
  committed `kern_v9p9`. The remote kernel source isn't in the output.
- **The 20,000-draw CIs to the last digit.** I used my own RNG seed, so the full-split CI differs from the packet's in the fourth
  decimal. That is bootstrap noise.
