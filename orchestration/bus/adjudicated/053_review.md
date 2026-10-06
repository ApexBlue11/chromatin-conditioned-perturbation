# REVIEW OF PACKET 053
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 7b16141

**Both readings are mechanical, and I reproduced them exactly.**
- **The outputs:** each equals its marker's sha1 (`d90fa910…`, `faef65c3…`) and the committed copy.
- **The marker pins:** they are d4b3a4e's (chromatin_h2 3a1c8d00, chromatin_gbm 7f7c3bb5).
- **The rerun:** `read_chromatin93.py h2|h3` gives readings **identical** to the committed JSONs: H2 NOT RISING; H3 DOES NOT
  ADVANCE, labelled *"informative null for the gain form (MDE ≤ 0.005 …)"*, with `G_check_informative` true.
- **The numbers:** every value in 93.7 that I checked matches `chromatin_gbm_93.json` and `chromatin_h2_93.json`: real C − B,
  centred, C − N1 and their per-cell values; chosen min-child; all calibration ranges; the faults.
- **The G check came out informative**, as review 052's follow-up predicted. Raw Δ was +0.030 to +0.056, while C − N1 was
  −0.026 to −0.052, so it fails correctly.

There are two MINOR points, both in the interpretation:
- **C1:** item 2's "gene-generic beats the cell's own, in every dev cell" is the size and cell count the **nothing-planted
  calibration** also gives.
- **C2:** item 4's "neither a linear nor a tree learner" needs the gain-form scope, because the trees are blind to
  drug-specific shifts.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **Item 2 reads as a finding, but the instrument produces the same gap when the cell's own feature carries no signal.** In H3's **null** calibration draws (shape-matched synthetic features, nothing planted), **C − N1 is −0.0021 / −0.0044 / −0.0022, with C > N1 in 1 / 1 / 0 of 6 cells**, and N1 − B is +0.0025 to +0.0037. The real run's C − N1 = **−0.0034, 0 of 6**, sits **inside that null range**. So "gene-generic chromatin beats the cell's own, in every dev cell" is the trees' default outcome: a cell-specific feature with no transferable signal costs about 0.003 against its own gene means. It is not "the strongest form yet" of 91.12's gene-generic reading. What *is* above the null: the cell's own chromatin adds **+0.0019** over B (null: about 0), and real N1 adds **+0.0053** (null: +0.003). | **Item 2, reworded:** *"C − N1 = −0.0034 (0 of 6 cells) lies within the nothing-planted calibration's range (−0.0044 to −0.0021; 0–1 of 6), so the cell's own promoter chromatin behaves like a feature without transferable cell-specific signal; its deficit against N1 is the learner's cost of fitting cell-specific variation that does not transfer. Above the null: own chromatin +0.0019 and gene-mean chromatin +0.0053 over B (null ≈ 0 and +0.003)."* Drop "the strongest form yet". Readings (a) and (b) stay as stated; Ask 5 has what would separate them. |
| 2 | MINOR | overreach | **Items 1 and 4 state the null without its form.** Item 4 says the marks *"do not carry a cell-specific signal that either a linear or a tree learner can transfer"*. But **H3 is blind to drug-specific shifts: P2 passes 0 of 3 at every π up to 5 %** (raw Δ ≤ +0.0031). T1's bound on that form is 5 % (§91.12). The trees' negative covers the gain form only. H2's null also has a useful property worth stating: the planted 0.5 % gain is detected at full size from k = 4 (median FBC − N1 +0.0042 at k = 4, +0.0039 at all). So the flat real curve is not a small-k limitation of the estimator. | **Item 1:** *"… more fitting cells did not lift the curve over k = 4–11 (a monotone +0.0007, below 2 sd; the planted 0.5 % gain is already detected at k = 4); a non-linear learner found no cell-specific gain of the gain form (MDE ≤ 0.5 %)."* **Item 4:** *"… do not carry a cell-specific signal of the gain form that a linear or tree learner transfers (drug-specific shifts: bounded at 5 % linearly, undetected by the trees at any planted size)."* |

## Answers to the asks

**Ask 1 — yes, mechanical** (93.2, 93.5 C2–C4, 93.6), and reproduced from the committed outputs.

**Ask 2 — no action. 93.7 already gives the all-rows instrument no weight, which is right.**
- **Why it voids:** P3 on all rows has a centred Δ of **+0.0041 to +0.0063** (known rows −0.0001 to +0.0012), which clears the
  all-rows centred bar (0.0015).
- **The likely mechanism** (plausible, not verified): on the level-3 rows μ is the global mean. The covered-share absorption of
  a covered-cell plant therefore differs between known and level-3 rows. Pooling both row types within a cell leaves a
  row-type-dependent part of a "drug-independent" plant that centring doesn't remove.
- **This is why** review 041 C3 made the drug-known rows the rows of record. Report the all-rows VOID beside the reading as you
  do, and add one clause on the level-3 mechanism.

**Ask 3 — items 1 and 4 need C2's scope, item 2 needs C1. Item 3 is fine** (not pre-registered, not read).

**Ask 4 — adequate, with one addition.**
- **What's there:** the bench timed one fit on synthetic data; the real run added μ, prediction and Kaggle's slower CPUs. The
  fit count (1,209) was as stated.
- **The addition:** it ran 9.4 h of a 12 h session, so any H3-style run with more cases must be split across kernels.
- **The versions:** lightgbm 4.6.0 on Kaggle against 4.7.0 locally doesn't matter here, since nothing is compared across
  versions.

**Ask 5 — yes. One addition would actually separate item 2's readings; a shrinkage arm would only partly.**
- **Replicate reliability (separates (a) from (b)).** Stage 1b downloads **per-sample** peak files. For every cell with ≥ 2
  usable samples, compute F_prom from each half of its samples (a fixed split, committed now). Report the correlation, across
  genes, of the two halves' **deviations from the gene mean** over the other cells.
  - High split-half reliability means the deviation is reproducible, so measurement noise isn't what kills transfer (reading
    (a)).
  - Low reliability supports (b).
  - Either way it's reported, not read, and outcome-free.
- **A shrinkage arm** (own features shrunk toward N1 by λ chosen by grouped CV on the fitting cells) can be reported beside
  H1's binding comparison. A null at every λ doesn't distinguish (a) from (b) with low signal-to-noise. It's only informative
  if it finds a gain.
- **Keep H1's binding conjunct unchanged:** the linear T1 form, with S(C) − S(N1) ≥ 0.002. H3's null calibration shows the
  ≈ −0.003 overfitting cost is a tree property. T1's linear ridge doesn't pay it at that size: §91's real T1 C − N1 was
  +0.00023.

## What I checked and found sound

- **Provenance:** the output sha1s against the markers and the committed copies, and the marker pins against d4b3a4e.
- **The readers:** `read_chromatin93.py` rerun on `external/kaggle_out/chromatin93_{h2,h3}`, identical to the committed
  readings.
- **H3 from the raw output:**
  - the real known-row conjuncts and per-cell values; chosen min-child and CV scores;
  - known and all faults, and MDEs;
  - per-draw null, P2 5 %, P3 5 % and G cases on both row sets;
  - N1 − B in the null draws.
- **H2:** the per-subset real FBC − N1 values; the planted P1 medians and ranges per k; the k = all harness check
  (`k_all_reproduces_91` true).

## What I could not assess, and why

- **Whether the level-3 mechanism in Ask 2 is the actual cause of the all-rows P3 pass.** It would need the per-row-type
  centred decomposition, which isn't in the output. It doesn't bear on the reading of record.
