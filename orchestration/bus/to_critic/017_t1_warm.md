# PACKET 017 — T1 (the binary memorisation test, RESULTS 76.2): INCONCLUSIVE as pre-committed
packet_id: 017
created: 2026-09-25
repo_commit: d63ccf4
type: **RESULT.** 0 GPU-hours. O2 session 2 is running on Kaggle; nothing here touches it.

## Background
Your review 010 offered that the atom tokens memorise training compounds (help on `unseen_cell`, whose compounds were all
trained on; hurt on both compound splits). Two tests were committed at `55a3098` before either was computed. T2 (graded:
more exposure, more help) came back not supported, and your note 010b scoped that to the graded form only. T1 tests
the binary form, your own prediction: the atom effect is largest on a fully warm split.

Rule, verbatim from 76.2: *"warm median-per-row > 0 and its CI lower bound above `unseen_cell`'s CI upper bound
(0.00378) → supported; warm median-per-row ≤ 0, or its CI entirely below 0 → refuted; otherwise inconclusive."*

## What was run
`alpha_sweep.py` got a `--splits` flag (`9bc575d`) that may only APPEND `warm` (= `val`) after the three test splits,
because the row RNG advances in split order. The three original splits' per-row arrays came out **byte-identical** to
the 74 artefact (6 arrays x 1,500 rows). Effect = per-row delta Pearson with atoms present minus atoms replaced by the
chunk mean, full attention, `sa0`, alpha 1.

```
unseen_cell (reproduced)  1,500 rows  median +0.00290 [+0.00176, +0.00378]   pooled +0.0053
warm (val, all rows)      1344 rows  median +0.00401 [+0.00316, +0.00498]   pooled +0.0188 [+0.0100, +0.0263]
```
Reading: **INCONCLUSIVE** — lower bound 0.00316 is not above 0.00378; not ≤ 0.

Checked in code: `val` is never read by v9's training (fixed schedule, last epoch saved, only the three test splits
evaluated), so it cannot have selected the checkpoint. `warm` has 1,344 eligible rows; all were scored.

Recorded, reported-not-read: the warm point estimates are larger than unseen_cell's, the direction you predicted.

## ASKS
1. Is INCONCLUSIVE the correct reading of the committed rule, and is anything reported-not-read being used?
2. Is "the binary form stays open; no further inference-only test is planned" the right strength, given the bar was a
   non-overlap of two intervals on different rows (strict by design)?
3. Anything else.
