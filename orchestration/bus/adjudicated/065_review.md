# REVIEW OF PACKET 065
verdict: SOUND-WITH-CAVEATS
reviewed_commit: cf5abce (§96 91b4b7d; finding and amendment 63b2bdd)

**P9 is cleared to push at the Sat 10 Oct reset.** The kernel is P7's, with only the intended changes:
- **The diff:** I diffed `kern_v9p9/lincs-v9p9.py` against `kern_v9p7/lincs-v9p7.py`. It changes only the header, the split,
  the output names, the arm-JSON glob, GUARD 6 and the `xpert_arm` pin.
- **The pin:** P7 ran `xpert_arm` 75c58f52 (277c143), and P9 pins 60bdcd48 (3495ada). The diff between them is the opt-in
  `--chromatin_encoding clean` branch. The default `v9` path is the original z-score loop moved under `elif`, and its output
  suffix is unchanged. So P9 computes P7's recipe.
- **GUARD 6:** reproduced exactly. The `drug_feature_index.json` rule gives **21,151 / `be276e23`** on cold-cell (P7's guard)
  and **13,364 / `5f85ef0b`** on cold-drug (7 unfeaturisable compounds, 81 rows).
- **What P9 trains on:** the split has only `train` and `test` labels, with no validation fold, so P9 trains on the training
  rows alone.

**The §96.6 finding is right, and I reproduced it exactly.** Rerunning `cold_drug_leak.py` to scratch (`drug/.venv-drug`) gives
summaries identical to the committed audit for both splits.
- **The clean subset reproduces:** 11,983 rows, 351 compounds, `6024dbf8`.
- **It's the right reading of record** for "unseen compound".

There are four MINOR points. All are reading-side, none touches the kernel, and all must be committed **before any P9 output is
read**.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | estimand | **The compound-level estimand double-counts test-internal duplicates.** **5 molecules appear under 2 test `pert_id`s each** (InChIKey first block), so 10 test "compounds" are 5 molecules. Each would contribute its own d_k, overweighting those molecules and correlating the sign test. | Collapse test compounds by InChIKey first block (one d_k per molecule, from all its rows) in both the clean and full readings. Restate n, which will be about 346 clean and about 384 full. |
| 2 | MINOR | Stage B′ scope | **Excluding afatinib and doxorubicin hollows out EGFR.** The members per cell are: EGFR = {CP-724714, afatinib, erlotinib, pelitinib} in MCF7, and {CP-724714, afatinib, erlotinib} in PC3, A375, HA1E, HT29 and A549. DNA = {altretamine, cladribine, decitabine, doxorubicin} in each TP53-wild-type cell. Under the ≥ 3-member unit rule, the "excluding them" reading keeps **DNA × 3 cells (3 compounds each) and EGFR@MCF7 only**: 4 units from 6 compounds. The 2-class condition still holds, but the EGFR half of the licensed sentence would rest on one cell. | State in 96.6 item 2, before P9 is read, the unit rule for the excluding reading (keep ≥ 3) and its resulting composition: 4 units; DNA in MCF7, A375 and A549; EGFR in MCF7. Add to the licensed sentence's beside-text: *"EGFR: one cell (MCF7) after excluding afatinib."* |
| 3 | MINOR | wording | **Three precisions in 96.6:**<br>(a) **"1,330 of 13,445 rows"** belongs to the **37** Tanimoto-1.0 compounds. The **38** excluded compounds (InChIKey ∪ Tanimoto; here 37 ⊂ 38) remove **1,381 of the 13,364 scored rows (10.3 %)**.<br>(b) **The audit is of fold `split_cold_drug_1`.** XPert's 0.645 is a five-fold mean, so *"XPert's published 0.645 partly measures seen molecules"* is an inference from the shared cause (a holdout by `pert_id`), not measured on their other folds.<br>(c) **Near-duplicates** (13 compounds at 0.8 ≤ max < 0.999, beyond the 38) aren't addressed by the reading. | (a) State both numbers. (b) *"On this fold, about 10 % of scored rows are same-molecule duplicates. Every fold built by holding out pert_ids is exposed to the same mechanism, so published cold-drug scores on these splits, 0.645 included, likely include seen molecules."* (c) Report d_k **stratified by max Tanimoto** (< 0.6, 0.6–0.8, 0.8–0.999) beside the clean reading. Reported, not a reading. |
| 4 | MINOR | comparator | **O9's head-to-head should be read on the clean subset too.** The reproduction check ([0.621, 0.669]) belongs on the full split, since that's the definition XPert's number uses. But "v9 against XPert on unseen compounds" is a clean-subset statement, for both models' predictions restricted to the same 11,983 rows. | Say so in O9's registration: reproduction on the full split; the head-to-head of record on the clean subset; the full-split head-to-head reported as *"the benchmark as defined"*. |

## Answers to the asks

**Ask 1 — yes, cleared to push P9.**
- **The kernel, pin and guard:** as above.
- **One note for the record:** P9 is P7's model, so it uses the default `v9` chromatin encoding, which carries the 93.16
  tie-break codes. That's right for a replication of the model of record. If E3 is later adopted, a P9-style run on it would be
  its own registration.

**Ask 2 — the finding is right, and the rule is right.**
- **InChIKey first block** is the standard "same molecule up to stereo or salt" rule.
- **Tanimoto ≥ 0.999** is a belt-and-braces addition. Here it adds nothing beyond the InChIKey set (37 ⊂ 38), and it can't remove
  a true duplicate the InChIKey rule missed unless the InChIKeys are absent.
- **Near-duplicates (> 0.8) should be reported as a stratum, not excluded from the subset of record.** A close analogue is a
  legitimately unseen compound, and generalising to analogues is what "unseen compound" means in practice. The stratified report
  (C3c) lets a reader see how the claim depends on similarity, and the median nearest-training similarity is 0.45.

**Ask 3 — yes, the clean subset is the right reading of record.** The full split stays meaningful in two places:
- **as the reproduction target** for XPert's number;
- **as "the benchmark as defined"** for comparison with published work.

The head-to-head that supports a claim is on the clean subset (C4).

**Ask 4 — before any P9 output is read:** C1 (collapse test duplicates), C2 (the excluding reading's unit composition), C3
(wording and the similarity strata) and C4 (O9's clean head-to-head). Nothing in the kernel needs to change.

## What I checked and found sound

- **The kernel:** `kern_v9p9` against `kern_v9p7`, line by line.
- **`xpert_arm.py`:** blob sha1s per commit (75c58f52 = 277c143, 60bdcd48 = 3495ada) and the diff between them.
- **GUARD 6:** recomputed on both splits with `xpert_arm.py:103–123`'s featurisation rule.
- **The leak audit:** `cold_drug_leak.py` read and rerun to scratch, identical. The split's label set (train and test only).
  Test-internal duplicates by InChIKey first block.
- **The clean subset:** 11,983 rows, 351 compounds, `6024dbf8`, recomputed.
- **The A3 members per cell** (identity level), for C2.
- **The registration:** §96.1–96.6.

## What I could not assess, and why

- **The leak rate on XPert's other cold-drug folds.** Only fold 1 is in the bundle. Hence C3b's wording.
