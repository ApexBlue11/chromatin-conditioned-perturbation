# REVIEW OF PACKET 089 (T-A, the CPU gate: registered no pass; post hoc within-cell diagnostic, §100.6 / §100.6a)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 41d83cd (code 9c235f9 / c230f6e; result 40f4295)

**The registered reading is mechanical and correct: no pass. The diagnosis of why is right too.**
- **The mechanism:** the registered target keeps each cell's overall responsiveness level, and the chromatin channels that are
  constant within a cell act as cell intercepts:
  - the 4 mask channels;
  - the all-zero features of the 113 cells without chromatin;
  - each cell's feature levels.
  Those intercepts can't transfer to a held-out cell.
- **The confirmation:** the real gain (−0.054) equals the gene-permute null's (−0.051), and that null keeps exactly those
  per-cell summaries.
- **The design error is mine.** My 086 Ask 4 design asked "which genes respond in this cell" without centring within cell. I
  concede it.

**The post hoc diagnostic is the meaningful measurement.** It was flagged before it ran, and it is properly labelled. Its
wording should state what the calibration bounds (C1).

**My rerun of `ta_gate.py`** (the same code and data; 20 null draws instead of 200):

| | record (200 draws) | my rerun (20 draws) |
|---|---|---|
| registered gain | −0.054 | **−0.0541**; cell-swap p95 +0.010, gene-permute mean −0.0515 → no pass |
| centred (post hoc) gain | +0.00001 | **+0.0000113**; calibration detects β = 0.02 |
| centred cell-swap p95 | +0.00004 | **+0.0000009**, so with 20 draws the real gain clears p95 (C3) |

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | the scope of the null (Ask 2) | **"Below 0.02 SD" bounds a narrower effect than "the drug-averaged linear mechanism".**<br>• **One feature:** the calibration plants β · z(accessibility) along **one model feature** (feature 0, ACC at the promoter, centred), which a linear model captures exactly. The limit is for an effect of that shape. Effects spread across features, or nonlinear in a feature, aren't bounded by it.<br>• **Magnitude, not direction:** R is the mean **\|contrast\|**, a response *magnitude*. A chromatin effect on the *direction* of response (for example, closed genes induced less but repressed as much) can cancel or hide in it. | Word it: *"a linear effect of 0.02 SD along promoter accessibility on drug-averaged response magnitude would have been detected; none was. Effects on response direction, nonlinear effects and drug-conditioned effects are not tested."* Use "not detected", not "absent". |
| 2 | MINOR | design note (no change) | **The 113 cells without chromatin stay in the leave-cell-out training set.**<br>• After the training-cell gene means are subtracted, their chromatin columns become a fixed pattern (−gene mean) that the ridge must also fit. That dilutes w.<br>• The calibration (planted in chromatin cells only) still detects 0.02 SD, so power is shown *as implemented*. | Nothing to change for the record. If T-A is ever re-run, fit the chromatin model on chromatin cells only (or give them their own gene means) for a cleaner limit. |
| 3 | MINOR | robustness of the post hoc null | **The post hoc "no" depends on the null's tail, not on the effect size.**<br>• My 20-draw rerun (a different RNG path) gives a cell-swap p95 of **+0.0000009**, which the real gain (+0.0000113) exceeds. Your 200 draws give **+0.00004**, which it doesn't.<br>• **What is robust is the magnitude.** The real gain is **≈ 1.4 %** of the gain a planted 0.02 SD effect produces (+0.0008). Gain grows roughly as β², so an effect along that feature is **≲ 0.003 SD**.<br>• "Not detected" is right. The bound is the stronger statement. | Report the real gain's **empirical rank** among the 200 cell-swap draws (and save the draws). Lead the post hoc statement with the magnitude bound: *"the within-cell gain is about 1.4 % of a 0.02 SD planted effect's (≲ 0.003 SD along promoter accessibility)"*. Report significance second. |

## Answers to the asks

**Ask 1 — yes, the cell-intercept explanation is right, and yes, the registered design should have centred within cell.** The
gene-permute equality is the decisive evidence. The omission was mine (086 Ask 4).

**Ask 2 — fair once scoped as in C1:** single-feature, linear, magnitude only, drug-averaged. Keep the explicit "no claim about
drug-conditioned effects" sentence.

**Ask 3 — yes. Report T-A beside T-B as the prior:** the registered no pass, and the post hoc null at 0.02 SD.
- **It doesn't change T-B's mechanical rule.**
- **It does shape how a T-B result is read:**
  - **if T-B passes:** the drug-averaged linear pattern is absent, so the gain must come from drug-conditioned or nonlinear use.
    The reported gene-shuffle / cell-swap permutations (review 086 C4) then become the key interpretation;
  - **if T-B fails:** T-A and T-B agree.

## What I checked and found sound

- **`ta_gate.py` in full:**
  - responsiveness (mean \|contrast\| over training plate-mates, ≥ 2; cells ≥ 200 rows);
  - per-gene standardisation;
  - the leave-cell-out ridge with gene fixed effects by training-cell demeaning;
  - λ chosen on the base model;
  - both nulls, the planted calibration, and the `--centre` path (R, basal and chromatin centred; accessibility taken after
    centring);
  - touch-once.
- **The 100.6 registration against the code.**
- **The 100.6a text against the output JSONs.**

## What I could not assess, and why

- **The full 200-draw nulls.** My rerun used 20 draws per null (the calibration's 50 are fixed in the code). The gains don't
  depend on the null count.
