# REVIEW OF PACKET 076 (B1 code and kernel, 98.9; the M re-run record, 98.8)
verdict: SOUND
reviewed_commit: eaad092 (98.8 at 5cd0136)

**B1 is cleared to push at the reset.**
- **The flag-off path is unchanged:** the refactored suffix and flag construction equals the old inline code on **299 of 299**
  argument combinations I generated. The only exception is the added `drug_blind: False` key, which you disclosed. Key order is
  preserved.
- **The flag-on path holds every compound-varying input constant:** `u_feats`, `atoms` and `atom_mask`, which are the only
  compound inputs `LincsV9.forward` reads.
- **The kernel diff from `kern_v9p9`** is exactly the 30 listed lines, and the other five pins equal P9's.
- **The tests:** `test_drug_blind.py` + `test_chromatin_encoding.py`, 17 pass on rerun.

The two points below are MINOR and don't block the push.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | scorer provenance | **`score_p10`'s `v9_drugblind` loader holds B1 to less than P9.** It checks the manifest sha1s, the 3 seeds and y/ctl agreement, but not:<br>(a) **the V2 snapshot identities.** P9's loader runs `check_snapshot_identities`, which is name-generic and works on `v9p10b1_seed{k}.npz` unchanged;<br>(b) **that the run was drug-blind.** The manifest carries `argv` and `stack`, and nothing checks `'--drug_blind' in argv`. | In `load_ref('v9_drugblind')`: call `check_snapshot_identities(ref_dir, seed_files)` when `'V2' in stack`, and refuse unless `'--drug_blind' in manifest['argv']` and `'DRUG-BLIND' in manifest['stack']`. A test for each. |
| 2 | MINOR | guard description (98.9) | **The permutation check is a confirmation, not an independent guard.** Under `--drug_blind`, `batch()` ignores `drug_row`, and `u` is already constant, so deranging those two must give an identical forward (it measured 0). It also permutes only `drug_row` and `u`, so it couldn't detect a compound path through anything else. What actually guarantees blindness:<br>• `assert_drug_constant` on every batch, training and evaluation;<br>• the structural fact that `LincsV9.forward` gets compound information only from `u_feats`, `atoms` and `atom_mask` (`model_v9.py:161–175`). | Describe it so in 98.9: *"`assert_drug_constant` is the guard. The model reads the compound only through `u_feats`, `atoms` and `atom_mask`. The trained-model permutation check confirms invariance, which holds by construction."* No code change. |

## Answers to the asks

**Ask 1 — yes, unchanged.**
- **The suffix:** `_drugblind` is inserted after `_noatoms` only when the flag is set. Order and conditions are otherwise
  identical (299 combinations checked against a verbatim copy of the old block).
- **`candidate_flags`:** identical keys, values and order, plus `drug_blind: False`.
- **`batch()`:** the flag-off branch is the old body, moved under `else`.
- **The checkpoint dict:** gains `drug_blind`. Loaders read `model` / `cfg` by key, so they're unaffected.
- **Computationally,** B1's `xpert_arm.py` = P9's `60bdcd48` + 070's `encode_chromatin` lift (the `v9` path is byte-identical on
  the real `E_final`, review 070) + the flag. With the five other modules pinned to P9's sha1s, B1 differs from P9 only in the
  compound inputs.

**Ask 2 — nothing compound-indexed reaches the model.**
- **What varies by row:** every other batch tensor (`x_ctl`, `x_cell`, E, r, `E_mask`, `cell_ctx`, dose, time) is row- or
  cell-indexed.
- **How the constants are built:**
  - `u` is the training-row mean;
  - the atom is the atom-weighted float64 mean over the unique training compounds' full blocks (98.6 C4a);
  - the mask is a single True.
- **The row set** is unchanged: unfeaturisable rows are still dropped at load, before blinding, so B1 trains and scores P9's
  exact rows (GUARD 6, 13,364 / `5f85ef0b`, kept).
- **On the permutation check:** C2. The tensor guard is the real one.

**Ask 3 — yes, complete.** Nothing P9-specific is left.
- **The diff covers:** argv, output stems, the manifest name, the log name, the header, the mounted-string checks, the pin, and
  B1a / B1b.
- **The arm JSON is found by** the generic `v9_xpert_arm_split_cold_drug_1*.json` minus `before`, so the new `_drugblind` suffix
  is picked up.
- **The intentional carry-overs** are right: the row guard, V2 / snapshot staging, `--no_test_metrics` and the sign-head
  weight.
- **The metadata** swaps `lincs-v9-src` for `lincs-v9-src-b1`, keeping the source frozen for P7 / P9.

**Ask 4 — two additions to 98.8.**
- **(a) A freeze.** The validation figure (0.640 / 0.642 on unseen-molecule carve rows, beside v9's 0.642 test row-pooled)
  hints at R1. Record that **no P10 registration, B1 code or scorer rule changes after this point** except review-mandated
  fixes. That's true today; saying it makes it checkable.
- **(b) Repeat at reading.** Repeat the disclosure in one line beside R1 when it is read, as for O9.
- **Nothing else is needed:** R1 is mechanical, and its estimand is on test rows the PI hasn't seen.

**Ask 5 — agreed.** O9 session 3 (8.3 h), then B1 (≈ 6 h), then D's timing. That fits one week's quota. R2, the reading P9's
interpretation hinges on, waits only one session for it.

## What I checked and found sound

- **The code:** the diff 5cd0136 → eaad092 of `xpert_arm.py` (`_apply_drug_blind`, `batch`, `assert_drug_constant`,
  `drug_permutation_check`, `build_parser`, `get_candidate_flags_and_suffix`, `main()`'s wiring) against `model_v9.forward`'s
  drug inputs.
- **The flag-off equivalence:** 299 argument combinations against the old block.
- **The kernel:** `kern_v9p10b1` against `kern_v9p9` (script and metadata).
- **The scorer:** `score_p10.load_ref('v9_drugblind')`, and `score_p7.check_snapshot_identities`'s name handling.
- **Tests:** 17 pass on rerun.
- **The record:** 98.8 (the crash mechanism and the template-only fix).

## What I could not assess, and why

- **The executed-kernel test (98.8) and the 22-mutant run.** I didn't rerun them; the 17 tests above were.
- **The uploaded `lincs-v9-src-b1` contents.** I relied on the kernel's six in-run sha1 pins.
- **An unrelated note:** the main checkout has an uncommitted modification to `model/v9/test_p7_blinding.py`, the stale assertion
  98.9 flags. It's not part of this packet.
