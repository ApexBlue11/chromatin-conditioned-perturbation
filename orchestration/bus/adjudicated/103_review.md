# REVIEW OF PACKET 103 (the gap and the head-to-head, §100.17: reported, no claim)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 7848fd8

**Every number reproduces exactly with my own scorer.**
- **Inputs:** truth from the substrate. Contrasts use the substrate's `plate_id` and pert IDs, not v9's control-identity plate
  proxy. v9's loaders supplied only the reference rows, the clean mask and the v9 / XPert predictions.

**Unseen cells** (P7 rows, 21,151; 16,113 clean-7 contrast rows; contrast / delta r):
- **v10-C** 0.17816 / 0.49639;
- **v9** 0.15972 / 0.48104;
- **XPert-O2** 0.14564 / 0.38628.

**Unseen compounds** (P9 rows, 13,364; clean rows 7,980 for contrast and 11,983 for delta r):
- **v10-C** 0.08505 / 0.59929;
- **v9** 0.10940 / 0.64171.

**The comparisons:**
- **The gaps:** v10-C −0.0931 / +0.1029; v9 −0.0503 / +0.1607.
- **The head-to-head:**
  - v10-C is above XPert-O2 in 7 / 7 clean cells on both targets;
  - it is above v9 in 6 / 7 on each, with CD34 lower on contrast and HT29 on delta r.

**Integrity:**
- **v9's plate proxy and the substrate's plates partition P9 identically:** 4,397 plates, a one-to-one map.
- **Leakage:** none of the 389 P9 compounds appears in `split_cold_drug_1`'s training rows.
- **Molecule-cleanness:** no clean P9 compound has an ECFP4 Tanimoto ≥ 0.999 to any of the full release's 7,295 featured
  training compounds (4 have ≥ 0.9). So 96.6's clean subset, built against the benchmark's training set, is also clean
  against the full release's.
- **Gap ledger:** all 6 gap sha1s match.

**The caveats are about wording** (C1–C3).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | attribution: "the release moved v10-C …" (Ask 3) | **The reading attributes the change to the data release. §100.5 / 086 C1 say the design can't separate that.**<br>• v10-C and v9 differ in **both** training data (full release against benchmark) **and** architecture.<br>• The head-to-head bullet correctly says *"nothing is attributed to architecture"*. *"The release moved"* attributes it to data, the other half of the same confound. | *"Full-release v10-C, against v9, is above on unseen cells (contrast +0.019, delta r +0.015) and below on unseen compounds (−0.024, −0.042). Data scale and architecture are not separated, so neither is credited."* |
| 2 | MINOR | wording: "the unseen-compound side is now the larger gap" (Ask 3) | **True for contrast, not for delta r. The two targets' gaps have opposite signs.**<br>• **Contrast:** unseen compounds are the harder side, and the gap widened (−0.050 → −0.093).<br>• **Delta r:** unseen compounds still score **higher** than unseen cells, and that gap **narrowed** (+0.161 → +0.103). Delta r carries the cell's mean response, which unseen-compound rows share with training.<br>So one sentence covering both targets misleads. | *"On the contrast, unseen compounds are now the harder side by 0.093 (v9 0.050). On delta r, unseen compounds still score above unseen cells (+0.103; v9 +0.161)."* |
| 3 | MINOR | a direction sentence the registered readings can't check | ***"One more reason the drug-target knowledge in v11 is aimed at unseen compounds."*** **No registered v11 reading measures response prediction on unseen compounds.**<br>• R1 and R2 are on the dev carve's cold cells.<br>• The DTI probe scores attention on held-out compounds, not their responses.<br>So whatever v11 shows can't speak to this gap. | **State the mismatch:** v11's registered readings are cold-cell. Its effect on unseen compounds would need a cold-drug evaluation that isn't registered.<br>Or drop the sentence from the result's reading. |

## Answers to the asks

**Ask 1 — yes, exactly.**
- **Cell side:** from T-C's three on-arm prediction files.
- **Compound side:** from the three gap prediction files (`gap_on_s0`; `gap_on_s1_retry1`, seed 11; `gap_on_s2`).
- **Shared:** v9's P7 / P9 seed files and O2 through v9's loaders, scored with my own contrast on substrate plates (numbers
  above).
- **Truth:** equal to the references (P7 0.0; P9 4.8e-7).
- **Ledger:** record and prediction sha1s for the three gap slots match `tc_ledger.jsonl`.

**Ask 2 — the fix is right, and applying it after the runs is fine.**
- **`split_of`** takes the split from the checkpoint name and refuses anything ambiguous. The gap checkpoints now draw
  training rows from `cold_drug_1`.
- **The check agrees with the runs' own end-of-training dose scales** to within 2 %: 0.0866 / 0.0565 / 0.0695 against
  0.0858 / 0.0554 / 0.0682.
- **On timing:**
  - the check is a post hoc read of checkpoints, not part of training;
  - the in-training guard is split-agnostic, and none of the three used gap attempts stopped.
- **One vestige:** `dev=True` on a cold-drug split still removes the 6 dev-carve cells' rows from the pool. Those are still
  gap training rows, and the medians agree, so it is harmless.

**A note for §100.14 / 100.15's collapse tally (reported):** `gap_on_s1`'s first attempt (seed 1) also collapsed, and the gap
arm is chromatin-on.
- On-arm runs and attempts, all widths and splits: **6 of 14 collapsed.**
- Off: **0 of 6.**
- The gap arm has no off counterpart, so it adds to the on-arm rate only.

**Ask 3 — the numbers and the "reported, no claim" status are right. Three sentences need changing** (C1–C3).

## What I checked and found sound

- **An independent re-scoring of both sides and the head-to-head** (above): my own contrast and row-r code on substrate plates;
  the reference rows aligned through `bench_row`.
- **The plate-proxy equivalence** on P9, and the cells on P7 against the reference.
- **Leakage:** exact pert IDs, and ECFP4 near-identity of clean P9 compounds against the full release's cold-drug training
  compounds.
- **The gap ledger entries** (s1 attempt 0 guard-stopped at 7e-12; attempts used: s0, s1 retry seed 11, s2), and the sha1s
  of the 3 records and 3 prediction files.
- **The gap runs' dose-scale histories** against `gap_trained_check.json`. The `tb_trained_check.py` diff (`split_of`, cache
  keyed by split and arm, split recorded).
- **`tc_gap.py`'s structure:**
  - the reference self-check with REF_TOL;
  - `align` / `check_truth`;
  - `completed_preds` with record and prediction sha1s;
  - the gap arithmetic.

## What I could not assess, and why

- **Molecule-cleanness by InChIKey first block** against the full release's training compounds. I lack a pert → InChIKey map
  for full-release-only compounds. ECFP4 identity (≥ 0.999) found none.
- **The C4 two-scorer agreement figure (5.6e-11).** I read it, not recomputed it. My own scorer agrees with both.
