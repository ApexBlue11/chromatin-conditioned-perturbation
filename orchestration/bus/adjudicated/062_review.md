# REVIEW OF PACKET 062
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 5832d97 (packet 1dd0e68)

**The readings are mechanical, and I reproduced them exactly.**
- **Provenance:** all seven result files match their markers (measured 62f7d676, v9 seeds 76f4c109 / 2042e870 / 6d44282f,
  seed-mean 2772235f, μ e7ad047d, ridge 1cf86cc4).
- **The rerun:** `--read_b` on them gives a reading **identical** to `stage_b_reading.json` (1645eb9b): reading 1 TRUE (μ also
  expresses), 2a FALSE, 2b TRUE.
- **The numbers:** every value in 94.11's table and the packet matches, including B3c and its raw counterpart, the v9 − μ A1
  differences, and B3 T.

**One MAJOR point, on interpretation (C1).** 2b's pass is mechanical. But its licensed sentence's *"slightly better than the
compound's training-cell average does"* is not distinguishable from chance. I computed the paired null that 94.9 didn't
register, and the margin falls inside it. v9's across-cell pattern is also mostly μ's. C2 is MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | overreach | **2b's margin over μ sits within a paired swap null.** Per unit, swapping v9's and μ's standardised d at random (10,000 draws; centred within class as B3c does) gives the null for B3c_v9 − B3c_μ:<br>- **the seed-mean:** +0.088, **p = 0.29**;<br>- **the seeds:** +0.066 (p 0.36), +0.142 (p 0.24), +0.010 (p 0.47).<br>**v9's own within-class pattern is largely μ's:** r(v9 centred d_std, μ centred d_std) = **0.79–0.84** across runs, and μ has no cell information.<br>**On raw d** the difference is −0.20 to −0.24.<br>So the 2b pass is a registered threshold crossed by a margin that random assignment of v9 and μ to units reproduces about a third of the time. *"…slightly better than the compound's training-cell average does"* states a gain the data can't distinguish from none. | **Keep the reading as registered** (2b TRUE), and replace the licensed sentence's use with: *"Reading 2b passed its registered rule (v9 B3c 0.81 against μ 0.72 on within-cell standardised pathway activity). Its margin over μ is within a paired swap null (post hoc; p 0.29 for the seed-mean, 0.24–0.47 by seed). v9's across-cell pattern correlates 0.8 with μ's, and on raw activity the order reverses. It is not evidence that v9 adds cell-specific pathway information beyond the compound's average."* Record the swap null as **post hoc, descriptive**. Reword "What Stage B says" bullet 2 to match: *"…a registered standardised-scale reading passed, by a margin within noise."* The paper should not state 2b's sentence as a finding. |
| 2 | MINOR | clarity | **"This matches §85.14's cell-level readout, the accuracy gains over μ, and §88's internal null" is ambiguous on its middle term.** If v9 is more accurate than μ on the test cells, that's cell-specific information in accuracy terms. These mechanism readouts then *don't see* it, which isn't the same as "matches". | Name the comparison and its direction. If v9 beats μ in accuracy: *"whatever v9 adds over μ in accuracy is not visible at these mechanism readouts."* |

## Answers to the asks

**Ask 1 — yes, mechanical**, reproduced from the seven marker-verified files.

**Ask 2 — the beside-text is accurate, but not enough on its own.**
- **Accurate:** "small margin", "μ at 0.72 through composition", "raw reverses" and "sd includes members".
- **Missing:** the margin's swap null, and v9's r ≈ 0.8 with μ (C1).
- **With those, the honest content of 2b is null-equivalent.** "Indicates … slightly better than μ" shouldn't be stated.

**Ask 3 — yes.** *"Nothing here shows the model adds mechanism-relevant, cell-specific information robustly beyond what the
drug's average response already carries"* is licensed. The swap null makes it firmer.
- **2a:** v9 − μ retrieval mean −0.004, 2 of 5 cells positive.
- **2b:** the margin is within noise, and the pattern is mostly μ's.
- **Bullet 1** ("essentially the compound's identity") is licensed.
- **The "As does μ" qualifier** on reading 1 is right.
- **The ceiling note:** noise-free predictions exceeding the measured reference is as anticipated (94.10 C3).

**Ask 4 — the shortest licensed statement of §94:**
> *"The measured landmark responses in unseen cells carry drug mechanism: mechanism-mate retrieval AUROC 0.60–0.67 against a
> 0.50 null in 4 of 5 cells, and pathway activity in the expected direction in 15 of 17 class-by-cell units. v9's predictions
> reproduce that structure, but no better than the compound's average response across the training cells: retrieval doesn't
> exceed it, and a registered cell-specific pathway readout passed by a margin within a paired swap null that reverses on raw
> activity. Nothing here shows v9 adds mechanism-relevant, cell-specific information beyond the drug's average."*

Scope it with "compounds seen in training; model predictions, not internal attributions".

## What I checked and found sound

- **Provenance:** marker sha1s for all seven Stage B result files; `--read_b` rerun, identical to the committed reading.
- **From the result files:**
  - per-run B3c (standardised and raw) with p's;
  - v9 − μ A1 per cell for every run;
  - μ's B1 and B3;
  - B3 T for every source.
- **A paired swap null for B3c_v9 − B3c_μ** (standardised and raw), and the correlation between v9's and μ's centred
  standardised d. Both computed post hoc from the committed d's.
- **The RESULTS text:** 94.11 against the reading.

## What I could not assess, and why

- **v9's accuracy relative to μ on these rows** (C2's middle term). It isn't in the packet, and I didn't recompute it, since it
  needs reading the predictions against the truth beyond what Stage B reads.
