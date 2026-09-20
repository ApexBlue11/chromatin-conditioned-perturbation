# REVIEW OF PACKET 002
verdict: SOUND-WITH-CAVEATS
reviewed_commit: efbfaa3

Every number in this packet reproduces exactly from the raw `.npz` files on an independent
re-implementation. The guards are real and they execute. The placebo stratum and the cluster bootstrap
were the right two things to build.

The caveats are not about the computation — they are about what the two headline constructions license.
**Both of them read as evidence that chromatin *hurts*, and neither supports that.** The retraction of
+0.0042 is correct; replacing it with −0.0141 would be the same error with the sign reversed.

The defensible number is in C1.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | wrong-quantity | **The −0.0141 cluster mean comes entirely from cells with no chromatin.** The three no-track cells (HS578T, BJAB, H1975) contribute **102 %** of the unweighted cluster mean — remove them and it is −0.0038. Those are cells where the treatment does not exist, so they cannot carry information about whether chromatin helps; including them as 3 of 8 equally-weighted clusters is what produces the negative point estimate. Restricted to the **5 cells where chromatin is actually present**, the unweighted per-cell mean is **+0.000360**, cluster bootstrap CI **[−0.005433, +0.006153]**, 2 of 5 positive, sign-test p = 1.000. That is a tight, well-centred null — a better result than either headline, and it is the estimand the objective asks for. | Adopt the 5-cell restriction as the primary analysis and report the 8-cell version as a sensitivity. The no-track cells belong in the placebo, not in the treatment estimate. |
| 2 | MAJOR | code-vs-intent | **"BOTH arms see identical zeros there" is false**, in the script docstring and in ask 2's premise. I checked the executed path: `E_final[HS578T]` is all-zero and BJAB/H1975 are absent from the cell index, so the **ON** arm feeds zeros — but `xpert_arm.py` sets `self.E[:] = self.E[m_tr].mean(0)` and `self.r[:] = self.r[m_tr].mean(0)` for **every** row, so the **ABLATED** arm feeds a non-zero training-mean constant on those same rows. The inputs differ. Direction matters and it runs **against** the ON arm: zeros are out-of-distribution for a model trained mostly on real chromatin, while the training mean is in-distribution. So −0.022625 is a noise floor **plus a handicap**, not a clean floor, and it overstates the floor's magnitude. Answering ask 2 directly: not valid as a pure training-noise estimate; valid only as "how much these two models differ where chromatin cannot inform", which is the weaker claim. | Re-score an ON-arm variant in which no-track rows receive the training mean rather than zeros — inference-time only on the saved checkpoint, no retraining. The residual difference on those rows is then the clean floor. |
| 3 | MAJOR | stats | **The sign of the headline is an artefact of the weighting choice.** On the same 8 per-cell numbers: size-weighted (= pooled) **+0.004237**; unweighted **−0.014105**; trimmed **−0.005964**; drop-H1975 **−0.003760**; treated-cells-only **+0.000360**. Five defensible estimators, two signs, a 4× spread in magnitude. With 8 clusters no directional statement survives the choice of estimator, so the estimand has to be fixed **before** the number is quoted, not chosen after seeing it. | State the estimand explicitly ("mean effect per cell line, over cells where chromatin exists") and pre-commit to its estimator. Then report the others as a sensitivity table, which is what they are. |
| 4 | MINOR | stats | Ask 4, answered: **H1975 is 77 % of the −0.0141**, at n = 54 rows and 1/8 of the weight. One caveat against my own point — its row-level \|t\| is 17.2, so −0.0865 is a genuine property of these two models on that cell, not row-sampling noise. It is still a no-track cell, so what it measures with that precision is the C2 handicap, not chromatin. | Covered by C1 and C2; no separate experiment needed. |
| 5 | MINOR | stats | Ask 1, answered: the construction is right, the count is not. A percentile cluster bootstrap with 8 clusters undercovers — resampling 8 values with replacement yields a lumpy, discrete distribution (4,586 distinct means in 20,000 draws), and the interval should be read as descriptive rather than as 95 % coverage. Printing the per-cell values, which you did, is what makes the result assessable at all. | Keep the per-cell table as the primary object. If an interval is quoted, say it is descriptive, or use a small-cluster correction. |
| 6 | MINOR | confound | Ask 3, answered: **the pattern tracks cell-line size, not chromatin coverage.** Spearman(n_rows, delta) = **0.762, p = 0.028** — the two positive cells are the two largest. Track count shows no dose-response: 3 tracks −0.001825 (MCF7, MDAMB231, CD34), 2 tracks +0.003636 (HT29, THP1), 0 tracks −0.038213. If chromatin were doing the work, 3-track cells should lead 2-track cells; they do not. So the per-cell pattern is **not** consistent with "an effect present only in well-covered cells", and the leading alternative is a size-dependent artefact rather than biology. | Nothing cheap distinguishes these at 5 cells. It is a reason to stop, not a reason for another experiment. |

## What I checked and found sound

- **Full independent reproduction.** I re-scored both arms from the raw `.npz` with my own chunked
  per-row Pearson and got, to every digit shown: ON 0.4734, ABLATED 0.4692, pooled paired +0.004237;
  chromatin-present n=19,950 → +0.005855; no-track n=1,201 → −0.022625; all eight per-cell deltas; and
  cluster mean −0.014105 with CI [−0.036380, +0.001021] against your [−0.036660, +0.001017] (bootstrap
  RNG draw order, immaterial). 0 degenerate rows confirmed.
- **The guards are real and they run.** `row_index` identical, `y_true`/`ctl_true` max|diff| exactly 0.0,
  all scored rows are `test` rows of the split, test/train cell intersection empty. I re-asserted each
  independently. This is the C2 failure from 001 properly closed, not papered over.
- **The packet-defect disclosure is accurate.** Both `.npz` files exist, dated 2026-08-31 and 09-01, and
  `external/` is gitignored — so C2c was wrong for the reason you state, and the fault was the packet's.
  Recording it as 11 of 12 is the correct adjustment and I would have made the same error again given
  001's contents.
- **A bug I went looking for and did not find.** HS578T is in the cell index with an empty mask, so the
  normalisation loop in `xpert_arm.py` skips it — if `E_final[HS578T]` held raw values they would reach
  the model un-z-scored. It is all-zero (`absmax = 0`, 0/2934 non-zero), so this does not occur. Noting it
  because it would have been a silent input corruption confined to one arm.
- **A check that cuts against my own C2.** HT29 and THP1 have 2 of 3 tracks, so one track is zeros for
  them and the C2 handicap should apply partially. HT29 is nonetheless the *most positive* cell
  (+0.009459). So the handicap does not obviously dominate on partially-covered cells, and C2 should not
  be read as explaining the whole per-cell pattern.
- **Artefact paths and `002_gitstat.txt` are present.** I did not need `git log` at any point in this
  review. The packet-side fix works.

## What I could not assess, and why

- **Anything that separates "chromatin" from "these two particular training runs."** Still one seed per
  arm. Everything in this packet — placebo, per-cell, cluster bootstrap — is a property of one fixed pair
  of models. The cluster bootstrap resamples cell lines, not training runs, so it answers "would another
  panel of 8 cell lines show this?" and not "would another seed?". Given C1 now returns a null at
  +0.0004, I do not think buying seeds is worth it; I am recording the limit, not requesting the spend.
- **Whether the size–effect correlation is causal or coincidental.** ρ = 0.762 at n = 8 is p = 0.028, which
  is suggestive and badly underpowered simultaneously. I can rule out that it tracks track count; I cannot
  say what it does track.
- **The −0.0226 placebo as a magnitude.** Per C2 it is contaminated in a known direction but by an
  unknown amount, so I can say it overstates the floor without saying by how much.

## One line, if a summary sentence is wanted

> On the five unseen cell lines where chromatin data exists, ablating it changes accuracy by
> +0.0004 (cluster CI [−0.0054, +0.0062]). The earlier +0.0042 was row-weighting; the −0.0141 is the
> three cell lines that have no chromatin at all.
