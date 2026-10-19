# REVIEW OF PACKET 092 (the T-C readers: rule, gap / head-to-head, ledger; before any T-C run)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: c946a9f (`tc_read.py` 846d3db + c3d2f6c; `tc_gap.py`, `tc_gap_repro.py`, `tc_ledger.py` c3d2f6c)

**Both readers are faithful to §100.5 / §100.8 / §100.11.**
- **`tc_read.py`:**
  - the first completed attempt per slot;
  - contiguous attempts;
  - the record checked against the ledger's sha1;
  - per-cell 3-seed means over the 7 clean cells;
  - ≥ 5 / 7 plus a percentile cluster CI, and the mirror rule;
  - a missing slot gives no claim;
  - a per-attempt report.
- **`tc_gap.py`** uses v9's own loaders and scorers, so rows, plates, mates, the zero rule and seed-averaging are v9's by
  construction.
  - It refuses any other row set.
  - It checks v10-C's truth against the reference truth through the substrate.
- **Tests:** 17 / 17 pass on my run.

**My own code reproduces the new reference values.** I used the substrate's **real `plate_id`** rather than
`control_ids(X_ctl)`, and my own group sums:

| | §092 / `tc_gap_repro.json` | my recompute |
|---|---|---|
| v9 contrast, clean 7 / all 8 | 0.1597 / 0.1602 (16,167 rows, all 8) | **0.1597 (16,113) / 0.1602 (16,167)** |
| XPert contrast, clean 7 / all 8 | 0.1456 / 0.1460 | **0.1456 / 0.1460** |
| v9 delta r, clean 7 / all 8 | 0.4810 / 0.4812 | **0.4810 (21,097) / 0.4812 (21,151)** |
| XPert delta r, clean 7 / all 8 | 0.3863 / 0.3868 | **0.3863 / 0.3868** |

**What I'd tighten:** provenance in `tc_gap.py` (C1), one asymmetry (C2, Ask 1), and two read-time checks (C3, C4). All are
cheap and none blocks the push.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | provenance (`tc_gap.py`) | **`tc_gap.py` reads records and prediction files that nothing verifies.**<br>• `completed_preds` takes the first completed attempt and loads `v10c_pred_{tag}.npz` with **no** ledger check, for both the `tc` and the `gap` slots, and with **no** check of the file against the record's `pred_sha1`.<br>• `tc_read.py` verifies only the record, and it doesn't read prediction files at all.<br>**So the gap / head-to-head numbers are the one place an unlogged or changed file could enter.** | In `completed_preds`:<br>(i) check the record's sha1 against the ledger (`tc_ledger.ledger_sha1s`, the slot keyed `tc_on_s{s}` / `gap_on_s{s}`);<br>(ii) check the npz's sha1 against the record's `pred_sha1`, which the ledger also stores.<br>Refuse either mismatch. Add a tampered-npz test. |
| 2 | MINOR | consistency of the reported items (Ask 1) | **Missing slots are treated asymmetrically on the two reported sides.**<br>• **The gap arm:** reported over its completed slots, with n stated (§100.11 C3a).<br>• **The T-C on side:** dropped entirely if any on slot is missing.<br>• §100.8's "no claim" binds **T-C's rule** (on against off). The gap and head-to-head are reported, never a claim, on both sides.<br>• Suppressing v10-C's unseen-cell side hides the arm most relevant to the head-to-head, and only when on runs collapse. That is exactly when the head-to-head is informative. | **Register before T-C runs:** both reported sides use completed slots, with n stated per side and per model. T-C's rule still says no claim when a slot is missing.<br>If you prefer suppression, state the asymmetry and its reason in §100.11. |
| 3 | MINOR | robustness of the reuse (Ask 2) | **The reuse of `audit99_phase1`'s private functions is validated once, by the repro, not each time the reader runs.**<br>A later edit to `audit99_phase1.py`, `score_p10.py` or `coldcell_h2h.py` would change the reported numbers silently. | **Self-validate at read time.** `tc_gap.main()` already scores v9 and XPert in the same run, so assert they equal the registered references within 1e-4:<br>• all 8: C2c 0.16024 / 0.14602, C2a 0.48121 / 0.38678;<br>• clean 7: 0.15972 / 0.14564, 0.48104 / 0.38628;<br>• C1 0.10940 (7,980 rows) and 0.64171.<br>Record the sha1s of the three modules and the P7 / P9 / O2 inputs in `tc_gap.json`. |
| 4 | MINOR | the rule against the reported numbers (report only) | **The rule and the reported per-cell values are computed by two different scorers.**<br>• **`tc_read`** uses the trainer's `heldout_scores`. Its `row_pcc` uses a 1e-6 clamp, not `ZERO_REL`, and it averages per-seed per-cell means.<br>• **`tc_gap`** uses `rowr_pred`, the relative 1e-4 zero rule, on seed-averaged per-row r.<br>**What I checked is identical:** rows (by assertion), plates (8349ba8), and the compound key. Substrate `meta_pert_id` equals the bundle's on every test row of both splits.<br>**So** the two should agree to about 1e-4 for trained runs, differing only on near-zero predictions. | In `tc_gap.json`, report v10-C's clean-7 per-cell contrast from both computations and their maximum difference. Then the numbers in the rule and in the head-to-head are visibly the same quantity. Report it; don't rule on it. |
| 5 | MINOR | edge case (a kernel death after a guard failure) | **If a kernel dies for a non-guard reason after an earlier guard-failed attempt in the same kernel, the readers stop.**<br>• Example: attempt 0 guard-fails, then attempt 1 runs out of memory.<br>• The kernel writes no `V10C_KERNEL_COMPLETE.json`. So `slot_attempts` refuses the directory if it is present, and if it is absent, the re-run's attempts aren't contiguous from 0.<br>• `tc_ledger death` logs only the death, not the dead kernel's earlier attempts. Their `V10C_FAILED_*.json` files are still in its output. | Have `death` also log the dead kernel's FAILED attempts from their files. Have `slot_attempts` accept a death-logged directory, taking its attempts from the ledger. Or register that such a kernel is re-run from its first unfinished attempt with the earlier attempts logged by hand. Test one of these. |

## Answers to the asks

**Ask 1 — yes, faithful, with C2 for the on side.**
- **`tc_read.py` matches:**
  - §100.5's rule;
  - §100.8's first-completed attempt;
  - §100.11's ledger refusal and per-attempt report (seed, guard level, the §100.9 check, the other arm's same-seed status).
- **`tc_gap.py` matches §100.11 C2:**
  - clean 7 primary, 8 labelled;
  - the same rows, mates and truth;
  - mates among all P7 / P9 rows;
  - C1's clean ≥ 2-mate rows;
  - 97.10's clean delta r.
- **The gap-arm rule matches C3a.**

**Ask 2 — the reuse is acceptable and the repro is sufficient for comparability, with C3 added.**
- **Reusing v9's scorers is the strongest guarantee** of identical definitions.
- **The repro exercises the right things:**
  - a shuffled row order;
  - the bench_row → substrate mapping;
  - truth equality through the substrate, derived independently of the stand-in;
  - reproduction of every reference;
  - the stand-in scoring identically to v9.
- **What it can't exercise is a real trainer file.** I read the trainer's `savez`: `sub_row = D.sub_row[D.ev]` (substrate
  indices), `bench_row`, and float32 `delta`, all consistent with `align` and `check_truth`.
- §100.11's row-identity check (21,151 / 13,364) covers the row sets, and `align` re-asserts them at read time.

**Ask 3 — yes, record them now,** before any v10-C T-C number exists.
- **The values:**
  - v9 / XPert, clean 7: contrast 0.1597 / 0.1456, delta r 0.4810 / 0.3863;
  - v9's gap (clean 7): contrast −0.050, delta r +0.161.
- **The labels:** they are recomputations of predictions already read, so nothing is selected. Both v9 and XPert trained
  with the NCIH1975 alias, which matters only for the 8-cell values.
- **A note for the record:** H1975 contributes only 54 rows, so 7 against 8 cells moves the row-pooled values by ≤ 0.0005.
  The cell-cluster summaries and the per-cell head-to-head are where it matters.

## What I checked and found sound

- **`tc_gap.py` in full:**
  - `align` (exact set equality, reorder by bench_row), `check_truth` (substrate X − ctl[ctl_id] against the reference truth,
    1e-4);
  - `contrast_rows` / `delta_rows` (= `test_c1` / `test_c2c` logic: the truth contrast's mates mask, the zero rule, NaN off
    the mask, seed-averaged per-row r);
  - `summarise`, `cell_side`, `compound_side` (C1 on have ∧ clean, delta r on clean);
  - `completed_preds`, `gap`, and the head-to-head block.
- **The reused `audit99_phase1` functions:** `rowr_pred`, `plate_contrast`, `_plates_for`, `_load_p9_side`, `_load_p7_side`.
- **`tc_gap_repro.py`:** the stand-in construction (independent `sub_of` through the substrate's test rows, a shuffled order)
  and its JSON.
- **The `tc_read.py` diff:** the ledger refusal keyed (slot, attempt); `attempt_report`; trained flags.
- **`tc_ledger.py`:** `attempt_entries` (seed from args, record / pred sha1, guard level), idempotent `add`, `death`,
  `ledger_sha1s`.
- **The trainer's compound key and plate key** against the bundle's on both test splits: identical.
- **Tests:** `test_v10c_tc_read.py` + `test_v10c_tc_gap.py`, **17 / 17**.
- **The reference values,** recomputed independently: the table above.

## What I could not assess, and why

- **A real T-C prediction file through `tc_gap.py`.** None exists yet.
- **`tc_gap.py` end to end.** I didn't re-run it, and recomputed its references with my own code instead.
- **The 6 / 6 mutation testing.** Not rerun.
