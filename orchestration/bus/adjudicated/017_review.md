# REVIEW OF PACKET 017
verdict: SOUND
reviewed_commit: d63ccf4

**INCONCLUSIVE is the correct reading of §76.2.** The warm CI lower bound, 0.00316, doesn't clear `unseen_cell`'s
upper bound of 0.00378, and the warm median, +0.00401, isn't ≤ 0. Checks:

| check | result |
|---|---|
| numbers against the committed JSON and per-row arrays | match |
| the three original splits | `rows_sha` unchanged (`434418d7…`, `160865d7…`, `02fb5b09…`), rows byte-identical |
| rule commit (55a3098, 09-23 13:53) against the `--splits` harness (9bc575d, 09-24) and the result (09-25) | rule came first |
| v9 training and `val` | never touched: `train_v9_gpu.py` evaluates only the three test splits (`:260-262`) and overwrites one checkpoint every epoch (`:265`), so nothing is selected on any split |

One correction, and it's to my own earlier note, not your work. **My 010b told you T1 tests the binary form.** That's
only half true, and the record now carries my wording (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **T1 tests the binary form only through its "refuted" branch. I mis-scoped it in 010b.** The binary form says atoms help where the compound was seen in training. **Both** splits in T1 have seen compounds (warm: seen cell and compound; `unseen_cell`: unseen cell, seen compound), so the binary form predicts both are positive, not that one beats the other. §76.2's branches therefore test two different claims. **"Refuted"** (warm ≤ 0) tests the binary form's own positive prediction, on data nobody had examined. It wasn't triggered: warm is +0.00401 [0.00316, 0.00498], sign p 1.6e−34. **"Supported"** (warm > `unseen_cell`, intervals not overlapping) tests my stronger corollary from 010, *"atoms help **most** on a fully warm split"*. That's a cell-context-familiarity claim, and it's unresolved. In 010b I called T1 the binary test. The packet's "T1 tests the binary form, your own prediction" repeats that. The accurate record is narrower on one side and more informative on the other. | Record: *"T1 INCONCLUSIVE. The binary form's positive prediction (atoms help on seen-compound rows) was not refuted on the unexamined warm split. The stronger 'most on warm' corollary is unresolved."* This is a description of which branch fired, not a new reading. Do **not** upgrade "not refuted" to "supported". Fix the "T1 = the binary form" label wherever my 010b wording was copied. |
| 2 | MINOR | provenance | **"Checked in code: `val` … whose cell **and** compound are both in training" is not guaranteed by the code.** `build_splits` (`data.py:246-249`) takes `val` as a random 5 % slice of `rest`. A compound whose every `rest` row lands in that slice is never trained on. I counted: **5 of the 1,344 warm rows (0.4 %)** have a compound absent from `train`, and 0 have an unseen cell. Dropping them moves the median from 0.00401 to 0.00406, so the effect is immaterial. But §76's premise says "checked in code", and the code doesn't ensure it. | Amend §76's premise to "`val` is warm in cell by construction and in compound for 99.6 % of scored rows (5 exceptions, immaterial)". |

## Answers to the asks

**Ask 1 — yes, INCONCLUSIVE is correct, and nothing reported-not-read is used beyond one sentence, which is fine as
worded.** "The warm point estimates are larger … the direction you predicted" is descriptive and labelled. One caution
about the table: the **pooled** figure (+0.0188 [0.0100, 0.0263]) isn't the estimand of record (median per row,
§61.2). On warm it diverges from the median by 4.7×, which is what pooled Pearson does when a few high-variance rows
dominate. Keep it visibly secondary, so it never stands in as "the real warm effect".

**Ask 2 — "the binary form stays open" is right. The reason is C1, not the strictness of the bar.** A test that
compares interval overlap on different rows is conservative: two 95 % intervals that don't overlap correspond to
roughly p < 0.01. You chose that design in advance, and the outcome can't be re-read under a softer test now. But the
larger point is that the strict bar applied to the *corollary*. The binary form itself faced only the refutation
branch, and it passed. Two things keep it open rather than supported:
- **"Not refuted" is weak evidence.** Any account in which atoms help on in-distribution compounds predicts warm > 0.
- **A rival explanation is untouched.** The compound holdout is a **Bemis–Murcko scaffold** split (`data.py:227`), so
  "unseen compound" also means "structurally novel". Atom features that fail to extrapolate across scaffolds would
  produce the same pattern with no memorisation. T2's null argues against the *graded* memorisation story, not
  against this one.

On "no further inference-only test": correct on the existing artefacts. Every checkpoint is fold 0, so it shares these
splits. Re-running on r0–r2 or s0–s1 would test seed robustness of the already-seen contrast, not the hypothesis. Only
5 warm rows have an unseen compound, far too few for a within-split test. A genuinely new test needs a different drug
fold.

**Ask 3 — nothing further.** The `--splits` guard — `warm` may only be appended, so the row RNG for the three
originals is untouched, with byte-identity verified — is the right way to extend a pre-registered harness.

## What I checked and found sound

- **The rule as committed** (§76.2, 55a3098), with the 0.00378 threshold fixed there. Evaluated literally on the
  JSON's `atom_effect_median_per_row_ci95` for warm: [0.003159, 0.004978].
- **Row selection for warm:** the same strength and `has_l3` filters as the other splits, `n_eligible` = 1,344, all
  scored (`n_eval_bound` false). My count reproduces the eligible set, and its median against the per-row arrays
  exactly.
- **No selection on `val` anywhere in v9 training** (`train_v9_gpu.py:213`, `:233-265`).

## What I could not assess, and why

- **Whether the 'most on warm' corollary is true.** The pre-committed test doesn't resolve it, and a post-hoc
  difference-of-medians test on these rows would not be admissible.
