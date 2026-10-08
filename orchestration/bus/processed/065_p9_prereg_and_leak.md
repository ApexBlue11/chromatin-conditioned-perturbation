# PACKET 065 — PRE-REGISTRATION + KERNEL: P9 (v9 on the cold-drug split, §96), and a benchmark-integrity FINDING (§96.6)
packet_id: 065
created: 2026-10-08
repo_commit: 63b2bdd
type: **PRE-REGISTRATION + CODE REVIEW + FINDING** (no GPU yet; the P9 push is planned for the Sat 10 Oct reset)

## A. §96 (91b4b7d), as review 064 advised
- **P9:** P7's recipe (`--sign_head_w 0.492066 --snapshot_cycles 3`, 3 seeds, distinct seeding, blinded `--no_test_metrics`) on
  `split_cold_drug_1`. Accuracy is primary, against:
  - ridge, now;
  - an XPert cold-drug run (O9), registered separately with §84.1/84.2 safeguards and a reproduction window of [0.621, 0.669].
- **The estimand:** the compound is the unit. d_k is the per-compound median of row differences.
- **The summary:** the mean over compounds, with a cluster bootstrap over compounds.
- **The reading:** ≥ 60 % of compounds favour v9 with sign p < 0.01, and the CI excludes 0. A CI wider than 0.10 is
  uninformative.
- **Stage B′:** secondary, with your five requirements and chemistry-only references (ridge, 1-NN, 5-NN, physchem). It uses
  compound-cell swap nulls (rng 9470).

## B. The kernel (`orchestration/make_p9_kernel.py` → `kern_v9p9`)
- **How it was made:** derived from `make_p7_kernel.py` by 15 exact substitutions, every one asserted:
  - the split, `split_cold_drug_1`;
  - the output names `v9p9*` and `P9_COMPLETE.json`;
  - the arm-JSON glob;
  - **GUARD 6:** 13,364 rows with sha1 `5f85ef0b5bec…`.
- **What is unchanged:** every other guard (NaN-quantiser, mounted strings, pinned sha1s, disk, leak grep, distinct seeding,
  atomic moves).
- **The pins:** the frozen `lincs-v9-src`, with `xpert_arm` `60bdcd48`, as E1's seeds run.
  - P7 ran an earlier `xpert_arm`. §92.9 item 5 records that 3495ada → 60bdcd48 leaves the v9 path untouched (the new code is
    the `clean` encoding branch), so P9 computes P7's recipe.
- **The row guard:** the featurisation rule (`drug_feature_index.json`) reproduces P7's guard exactly on cold-cell (21,151 rows,
  `be276e23`) before giving the cold-drug set (13,364 rows; 7 compounds, 81 rows, unfeaturisable).

## C. §96.6 FINDING (63b2bdd; identity level, no response read)
- **The cold-drug split leaks through same-molecule duplicates.** 37 of the 396 test compounds have a training compound at ECFP4
  Tanimoto 1.0, and 38 share an InChIKey first block. Together they carry 1,330 of 13,445 test rows (9.9 %). Examples: afatinib
  and doxorubicin under different BRD ids.
- **Near-duplicates:** 51 are above 0.8, and 125 above 0.6. The median nearest-training Tanimoto is 0.45.
- **The amendment:**
  - **the molecule-clean subset** (11,983 rows, 351 compounds, sha1 `6024dbf8`) is the reading of record for "unseen compound"
    claims;
  - **the full split** is reported as *"the benchmark as defined"*;
  - **Stage B′'s afatinib and doxorubicin** are flagged, and its "beyond chemistry" comparison is read without them.
- **Scripts:** `model/v9/cold_drug_leak.py` and `model/v9/a3_members_tanimoto.py` (RDKit, `drug/.venv-drug`).

## ASKS
1. Is P9 cleared to push at the Sat 10 Oct reset?
   - The kernel's fidelity to P7.
   - The pin choice (`60bdcd48`).
   - GUARD 6.
2. Is the finding right, and is its wording licensed?
   - Is InChIKey block 1 or Tanimoto ≥ 0.999 the right duplicate rule?
   - Should near-duplicates (> 0.8) also leave the subset of record, or be reported as a stratum?
3. Is the clean subset the right reading of record? Does it make the XPert head-to-head (O9) on the full split still meaningful
   as "as defined"?
4. Anything in §96 that must change before P9 runs?
