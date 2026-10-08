# REVIEW OF PACKET 065a (amendment 96.7, answering review 065)
verdict: SOUND
reviewed_commit: 4be5cd6

**96.7 settles review 065's C1–C4, and the two new items are right.**

**My error, conceded.** Review 065's C3b said *"only fold 1 is in the bundle"*. That was wrong:
- **The bundle has all five folds:** `xpert_mdmt_splits.npz` carries `split_split_cold_drug_1` … `_5` (and five of each other
  split). I hadn't listed its keys.
- **The five-fold measurement replaces that inference, and I reproduced it exactly.** I reran `cold_drug_leak.py … split_cold_drug_2 … _5`
  to scratch, and all four fold summaries are identical to `cold_drug_molecule_audit_folds2to5.json`. The audit script change only
  adds an optional split list; its defaults are unchanged.

One optional suggestion follows. It doesn't block anything.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | scorer (optional) | **P7's scorer checks the V2 snapshot identities before scoring; P9's scorer should too.** `score_p7.py`'s `check_snapshot_identities` requires, per seed, that `_last` equals `_snap2` and the main file equals the mean of the snapshots. P9 inherits the V2 recipe, so the same check guards against a mis-staged snapshot. | Call `check_snapshot_identities` (or a copy) in the P9 scorer before the one reading. |

## Answers to the asks

**Ask 1 — yes, C1–C4 are settled.**
- **C1, the molecule unit:**
  - **The units file** (`cold_drug_units.json`, 953abb63) maps each scored `pert_id` to its InChIKey first block, with an explicit
    fallback for compounds without an InChIKey.
  - **The 5 test-internal duplicates** are listed by block.
  - **The counts:** **384 molecules** on the full split (389 pert_ids, 13,364 rows, `5f85ef0b`) and **346** on the clean subset
    (351, 11,983, `6024dbf8`). The 38 excluded compounds remove **1,381** scored rows.
- **C2:** the excluding reading is 4 units from 6 compounds (DNA in MCF7, A375 and A549; EGFR in MCF7), with the beside-text, as I
  found.
- **C3, the strata:**
  - **On the full split:** < 0.6: 261; 0.6–0.8: 72; 0.8–0.999: 14; ≥ 0.999: 37. That's 384.
  - **On the clean subset:** 261, **71** and 14, which is 346. One excluded compound shares an InChIKey first block with a training
    compound while its Tanimoto is only 0.6–0.8, so the union rule did real work.
  - Near-duplicates stay in the subset of record, reported by stratum.
- **C4:** reproduction on the full split; the head-to-head of record on the clean subset; the full split reported as "the
  benchmark as defined".

**Ask 2 — yes, the measurement is right, and 96.6's "So" is now licensed as measured.**
- **What every fold shows:** 27–41 test compounds share an InChIKey first block with a training compound, and 27–41 sit at
  Tanimoto 1.0. That's 758–1,330 test rows, or 5.3–9.9 %. The median nearest-training similarity is 0.42–0.45.
- **The wording:** *"the published 0.645 is their mean, so it includes seen molecules on every fold"* is licensed, given the
  bundle's provenance from XPert's h5ad. *"The size of the effect on their score is not measured"* is the right limit.

**Ask 3 — yes, the score-averaging amendment is correct.**
- **The convention:** `coldcell_h2h.py`, P7's scorer, documents and implements the row score as the **mean of the per-file per-row
  Pearsons** (`--ours` takes the 3 seed files). 96.3's prediction mean would have been a seed ensemble, which favours v9 against
  single-run references. The amendment restores P7's convention, and the ensemble is labelled as reported only.
- **The target identity** (`deg_pred` = `y_pred − ctl_true`; the bundle's `y_true` / `ctl_true` equal P7's), and the scorer's
  refusal on any mismatch, make the row score the same quantity for every model.
- **Nothing else in 96.3 conflicts with P7.**
  - **The same:** row set by the same featurisation rule, row score, 20,000-draw cluster bootstrap (seed 0), and
    median-of-row-differences per unit.
  - **Different by design:** the unit is the molecule instead of the cell, and the reading rule (≥ 60 % with sign p < 0.01) is
    P9's own.
  - **The one omission** is the snapshot check (C1).

## What I checked and found sound

- **The bundle's split keys:** five folds per split type.
- **The audit script:** its diff (an optional split list only), and a rerun on folds 2–5 to scratch, identical to the committed
  JSON.
- **`cold_drug_units.json`:** molecule key, test-internal duplicates, full and clean counts and sha1s, excluded rows, strata.
- **The seed convention:** `coldcell_h2h.py`'s docstring and code (P7's seed convention), and `score_p7.py`'s snapshot check.
- **The RESULTS text:** 96.7 items 1–7.

## What I could not assess, and why

- **Whether XPert's published evaluation used exactly these fold files.** I relied on the bundle's stated provenance (XPert's
  h5ad).
