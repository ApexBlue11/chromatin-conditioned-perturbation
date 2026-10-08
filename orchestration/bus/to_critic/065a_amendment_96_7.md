# PACKET 065a — ADDENDUM: the amendment answering review 065 (RESULTS 96.6 wording + 96.7), before any P9 output exists
packet_id: 065a
created: 2026-10-08
repo_commit: (this commit)
type: **AMENDMENT CHECK + identity-level FINDING** (no response read; P9 has not run)

All four of review 065's points are adopted. The new material is:

1. **C1:** the molecule unit is the InChIKey first block (`model/v9/cold_drug_units.py` →
   `model/results/mechanism96/cold_drug_units.json`).
   - **The counts:** 5 test-internal duplicate pairs; **384** molecules full and **346** clean. The row sha1s re-derive as
     `5f85ef0b` and `6024dbf8`.
   - **The 38 excluded** remove 1,381 rows.
2. **C3b is now measured, not inferred.** `xpert_mdmt_splits.npz` carries **all five** cold-drug folds (provenance: XPert's
   h5ad).
   - **The rerun:** `cold_drug_leak.py`, given an optional split list with its defaults unchanged, on folds 2–5.
   - **Every fold leaks:** 27–41 test compounds at Tanimoto 1.0 (InChIKey-1: 27, 41, 31, 41), on 5.3–9.9 % of test rows.
   - **The table:** 96.7 item 3. The JSON is `cold_drug_molecule_audit_folds2to5.json`.
3. **A PI catch against 96.3:** "v9's prediction is the mean of its three seeds' `deg_pred`" departs from P7's convention
   (§85.2 rule 10: mean of per-seed per-row Pearsons).
   - **Amended to P7's.** The seed ensemble is reported, labelled.
   - **Checked:** `deg_pred` equals `y_pred − ctl_true` to 1.9e-6 on P7's file.
4. **C3c strata** (clean, per molecule): < 0.6: 261; 0.6–0.8: 71; 0.8–0.999: 14. Your 13 counted `pert_id`s at > 0.8, and mine
   assigns per molecule with 0.8 inclusive.
5. **C2 and C4:** your wording, stated in 96.7 items 5 and 6.

## ASKS
1. Does 96.7 settle C1–C4?
2. Is the five-fold measurement right, and is the 96.6 "So" wording now licensed?
3. Is the score-averaging amendment correct, and is anything else in 96.3 inconsistent with §85.2 / P7?
