# PACKET 092 — CODE: the T-C readers (rule, gap / head-to-head, ledger) before any T-C run (RESULTS §100.5, §100.8, §100.11)
packet_id: 092
created: 2026-10-19
type: **CODE REVIEW** (no T-C prediction exists)

**Commits:**
- 846d3db: `tc_read.py`, the rule;
- c3d2f6c: `tc_gap.py`, `tc_gap_repro.py`, `tc_ledger.py`, the per-attempt report and ledger verification in `tc_read.py`.

**The status of T-C:** all nine kernels are cleared (review 091), but the first push was refused: *"Maximum weekly GPU quota of
30.00 hours reached"*. They push at the Sat 24 Oct 00:00 UTC reset.

## `tc_read.py` (the 100.5 rule)
- **What it reads:** per slot, the first completed attempt, across new-kernel retries `_r1` / `_r2`.
  - Attempts must be contiguous from 0.
  - The record's sha1 must equal the T-C ledger's, else FATAL (100.11 Ask 4 i).
- **The rule:**
  - per clean cell, the 3-seed mean of the per-row mean contrast r, and Δ_c;
  - ≥ 5 of 7 plus a percentile cluster CI (20,000 draws, `default_rng(0)`; lower > 0), and the mirror rule;
  - a missing slot gives no claim.
- **Reported:**
  - attempts per (arm, seed slot, attempt), with the seed used and the guard level (at a stop, from the FAILED record; at
    completion, the last epoch's mean batch-median s);
  - the §100.9 check, when `--trained` is given;
  - whether the other arm's attempt at the same seed completed;
  - delta r per cell, the row-pooled means, H1975, the permutations, the single-control counts, the seeds used, and the
    L6 wording.
- **Tests:** 14, three of them end to end (ledger + main), covering:
  - a guard retry being used;
  - a tampered record being refused;
  - an exhausted slot giving no claim;
  - an idempotent ledger;
  - a logged death.
- **Mutation testing:** 6 / 6 mutants of the rule and slot logic killed.

## `tc_gap.py` (the gap and the head-to-head, reported)
- **It imports `audit99_phase1`** and uses its `_load_p7_side`, `_load_p9_side`, `plate_contrast` and `rowr_pred`. So the rows,
  plates, mate sets, zero-variance rule and seed-averaging are v9's own, by construction.
- **The unseen-cell side:**
  - the P7 rows; mates among all of them;
  - contrast (C2c) and delta r (C2a);
  - 7 clean cells (primary) and 8 (labelled);
  - models: v10-C (T-C on), v9 (P7), XPert (O2).
- **The unseen-compound side:**
  - the P9 rows; mates among all of them;
  - contrast on clean rows with ≥ 2 mates (C1), and delta r on clean rows (97.10);
  - models: v10-C (the gap arm, over its completed slots, n stated) and v9.
- **The gap** is compound minus cell, per target, for v10-C and v9. **The head-to-head** is per clean cell, v10-C against
  XPert O2 and v9, with no claim.
- **The asserts:**
  - v10-C's `bench_row` set equals the reference rows (refused otherwise);
  - its truth (substrate X − ctl at `sub_row`) equals the reference truth to 1e-4.
- **The outcome-free validation** (`tc_gap_repro.py` → `model/results/v10c/tc_gap_repro.json`):
  - v9's own P7 / P9 seed predictions are written as stand-in v10-C T-C and gap kernels, with the kernel layout, `bench_row`
    and `sub_row`, the row order shuffled. They are pushed through `tc_gap.main()` end to end.
  - **It reproduces every reference:**

    | reference | expected | reproduced |
    |---|---|---|
    | C2c, v9 / XPert | 0.160 / 0.146 | 0.1602 / 0.1460, 16,167 rows |
    | C2a, v9 / XPert | 0.481 / 0.387 | 0.4812 / 0.3868 |
    | C1, v9 | 0.109 | 0.1094, 7,980 rows |
    | 97.10, v9 | 0.6417 | 0.6417 |

  - **The stand-in scores identically to v9** on both sides, so the alignment, truth check and mate sets are exact.
  - **New reference values** (v9 / XPert, never seen with any v10-C number):
    - the 7 clean cells: v9 contrast 0.1597 and delta r 0.4810; XPert 0.1456 and 0.3863;
    - v9's gap (clean 7): contrast **−0.050** and delta r **+0.161**. Unseen compounds are *harder* than unseen cells on the
      contrast, and easier on delta r.

## `tc_ledger.py`
- `add <kernel_dir>` logs every attempt: the slot, attempt, returncode and seed; the record sha1, pred sha1 and end dose
  level; or the FAILED record's sha1, stop step and guard level. It is idempotent.
- `death <slot> --attempt k --reason …` logs a non-guard death, which is re-run at the same seed and attempt.

## ASKS
1. Are the readers faithful to 100.5 / 100.11?
   - **The gap arm:** an exhausted gap slot is dropped and n is stated.
   - **The T-C on arm:** with a missing on slot, v10-C's unseen-cell side isn't reported at all.
   - Is that the right reading of 100.11 C3(a) and 100.8?
2. Is reusing `audit99_phase1`'s private functions acceptable? Is the reproduction a sufficient check of comparability?
3. Should the new v9 reference values (clean 7, and v9's gap) be recorded in RESULTS now, before any v10-C T-C number exists?
