# PACKET 067 — PRE-REGISTRATION: O9, XPert trained to its published recipe on `split_cold_drug_1` (RESULTS §97), before any O9 code
packet_id: 067
created: 2026-10-08
repo_commit: (this commit)
type: **PRE-REGISTRATION** (no O9 code or GPU yet; the kernel builder follows after review)

## Summary of §97
- **What it is:** O2's production machinery, with only fold constants changed: the fold; level counts 55,385 / 13,445; the
  13,445-row prediction checks; Amendment E's projection repriced to 562.1 s per epoch; the seed note; the slug and state
  dataset.
- **How it's built:** by asserted substitutions on `kern_xpert_cc1/generator/make_prod_kernel.py`, as P9's kernel was built.
- **The seed:** XPert's published commands (`scripts/train.sh:13–17`) train `split_cold_drug_k` first, after
  `set_random_seed(2024)` (`train_xpert.py:402`). O9 therefore starts from the same seed state as their own run of that fold.
  O2 had a disclosed seed difference; O9 has none.
- **Admissibility, item by item (97.3):**
  - test-selected checkpoint (a v9 win is conservative; an XPert win is uninterpretable);
  - §84.1 terminations and the horizon rule (best ≤ 252);
  - `best_epoch < 70` disclosed;
  - reproduction on the full split in [0.621, 0.669], else no head-to-head claim;
  - one XPert seed.
- **The reading (97.4):** `score_p9.py` with an O9 `REFS` entry. The head-to-head of record is on the clean subset; the full
  split is reported "as defined".
- **Cost, corrected from 96.5:** about 14–17 GPU-h if their stopper fires near O2's epoch; worst case about 46 GPU-h to the
  horizon.

## ASKS
1. Is anything fold-specific in O2's kernel missing from 97.2's list of changes? You can read `make_prod_kernel.py`,
   `prod_tail.py`, `lincs-xpert-cc1_v7_base.py` and the committed `lincs-xpert-cc1.py`.
2. Is the seed claim (97.1) right from their code?
3. Is the reproduction rule right: no head-to-head claim if outside the band? O2 flagged but still read. Should O9 match O2?
4. Is 97.3 complete as admissibility, item by item (your review 064 ask 3)? Anything from §71 / §78.5 that doesn't carry over?
5. Given the cost, is O9 still worth it? It is worth it if the paper makes a cold-drug claim, and P9 is now running for exactly
   that.
