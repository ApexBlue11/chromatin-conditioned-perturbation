# REVIEW OF PACKET 038
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 5b0da11

**Both readings are right, and I reproduce every number from the prediction files.**
- **P6:** 0.45607 / 0.45284 / 0.45722, Δ **+0.01844** against a threshold of 0.00327 and a stack floor of 0.01369.
  Cell means +0.02144, **6/6** cells, Δ_centred **+0.01326**. So it's CONFIRMED.
- **Seed-paired P6 − V2:** raw +0.00411 / −0.00569 / +0.00682; centred −0.00287 / −0.00965 / +0.00138.
- **V1:** drop +0.00089 / +0.00099 / +0.00088 (centred +0.00096); full +0.00100 / +0.00111 / +0.00100 (centred
  +0.00107). Neither is accepted.

Keeping C6 in P7 is right under §85.11. Nothing here should hold Saturday's push.

There are four MINOR points:
- two about how the "reported, not read" block describes C6 (C1, C2);
- one about which weights the interpretability numbers come from (C3);
- one label in `score_p7.py` to fix before the one-shot run (C4).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wrong-quantity | **"not in ranking drugs within a cell" misdescribes the centred score.** The §90.2 score (`score_dev.py --centred`) subtracts each cell's mean predicted and true delta profiles, then takes the per-row Pearson **across the 978 genes**. It measures how well each row's departure from its cell's mean profile is predicted. No drugs are ranked anywhere. Also, a raw gain with a flat centred score is *consistent with* the gain sitting in the cell-mean profile; it doesn't prove it. This sentence will reach the paper as the caveat on every C6 claim, so its quantity has to be right. | Amend §85.13 (ii): *"… consistent with the sign head's gain lying in the cell-mean delta profile rather than in each row's departure from it (the cell-centred score)."* Use the same wording in the manuscript. |
| 2 | MINOR | stats | **The block leaves out that the stack's per-row edge over V2 comes from one dev cell.** The row-mean P6 − V2 is +0.00175. U937 (308 rows) contributes **+0.00170** of that: its mean is +0.0223 and its median +0.0230. The other five cells together give **+0.00005** per row, with LNCAP at −0.0067. Within U937 it's mostly seed 0 (+0.051 there, against +0.005 and +0.011 on seeds 1 and 2). This strengthens "not shown to beat V2 alone". It should also stop any sentence crediting C6 with a broad gain in the final model. | Add to §85.13 (i): *"The +0.00175 is carried by one dev cell (U937, 308 rows: +0.022; the other five cells together +0.00005), and within it mainly by seed 0."* |
| 3 | MINOR | provenance | **The interpretability numbers and the reported accuracy come from different weights.** `--save_ckpt` saves `core.state_dict()` after the last cycle, i.e. the **final-snapshot** weights (`xpert_arm.py:484-490`). The headline predictions, though, are the **mean of three snapshots' predictions**. So rule 8's 0.2650, the in-cell licence and P7's test-row alignment are all read on a model whose dev accuracy is P6-last's (+0.00874), not P6's (+0.01844). That's consistent with how V2's rule 8 was read, and it's fine for the gate. But a sentence joining "ranks pathways at ρ …" to "scores … on the test cells" would describe an object that doesn't exist. | One sentence in §85.13 and §85.12 item 8: *"the pathway readout is read on the final-snapshot weights; the reported accuracy is that of the three-snapshot prediction average; the final snapshot alone scores [the alt row]."* The manuscript keeps the two numbers attached to their own objects. |
| 4 | MINOR | provenance | **`score_p7.py` labels P7's alt row "V2-last" (`:64`), but those files are the final snapshot of the C6 + V2 stack.** That's dev's "P6-last", not V2 alone. P7's result JSON is a one-shot artefact the paper will cite, and a reader of §90.7 will take "V2-last" to mean the V2-only model. | Before the run, change `--ours_alt_label` to e.g. `P7-last (final snapshot, no snapshot averaging)`. Where §90.6 names the row for P7, add *"(the stack's final snapshot)"*. It's outcome-free: a label only. |

## Answers to the asks

**Ask 1 — both readings are right, and keeping C6 in P7 is right.**
- **The rule decides.** §85.11 was fixed, and reviewed, before P6 existed. Dropping C6 now would choose P7's recipe on
  P6's outcome, which is exactly what the pre-registration exists to prevent.
- **The evidence against C6 in the stack is within noise, on both scores:**
  - raw P6 − V2 is +0.00175, against a rule-7-style threshold of 2√(0.00227²/3 + 0.00431²/3) = **0.0056**;
  - the centred difference, −0.0037, has mixed signs by seed.
  - So keeping C6 costs roughly nothing in expectation on P7's per-row headline.
- **What to disclose:**
  - C6 is in P7 by rule, not because it was shown to help on top of V2.
  - Any C6 claim is the dev C6 − P2 comparison, with centred +0.00015.
  - C1–C3.
- **The block's other statements check:**
  - **"Same seed → same initialisation and data order" is correct.** The sign head is built after every shared module
    (`model_v9.py:133-134`, and post-pathway is off), so under `torch.manual_seed(s)` the shared parameters initialise
    identically. `np.random.seed(s)` fixes the order, and the head adds no dropout draws.
  - **But the pairing buys little precision here.** The sign loss changes every gradient from the first step, and the
    per-seed P6 − V2 spread (−0.0057 to +0.0068) is wider than V2's own between-seed sd (0.0043).
  - **The rule 8 numbers match** `v9_dev_align_P6_c6_v2_aux.json`: 0.2757 / 0.2646 / 0.2546, mean 0.2650, 5/6 cells,
    seed means +0.074 / +0.051 / +0.063. Seed 2's 0.2546 is only 0.0014 above 0.2532, but rule 8 gates the mean.
  - **V1:** MC averaging recovers 3.1 % (drop) and 3.5 % (full) of the +0.0293 from averaging 3 seeds.

**Ask 2 — the smoke change is right and needs nothing else.**
- **The condition is safe.** It's evaluated on `argv` before the `+=`, and the conditional expression is a single list
  element.
- **The Kaggle path doesn't change.** The diff touches only the smoke branch, the flags and the stack string.
- **Don't put `_last == _snap2` in the kernel.** `fail()` deletes the staging, so a spurious failure would destroy
  ≈ 6 h and the week's quota over an integrity check on a secondary row.
- **Do the checks in `score_p7.py`'s preamble instead,** where a failure costs nothing, and refuse on a mismatch:
  - per seed, `deg_pred` of `_last` equals `_snap2`'s exactly;
  - the main file equals the mean of `_snap0..2` within 1e-5. I measured 6.4e-7 on P6 and ≤ 1e-6 on V2.
- **This is optional.** The arm produced both identities on V2's and P6's dev outputs.

**Ask 3 — nothing blocking. Do C3's sentence and C4's label before the run.**
- **Disk:**
  - The dev prediction files are 47 MB compressed at 4,043 rows, so each P7 file is ≈ 250 MB at 21,151 rows.
  - Fifteen files (3 main, 3 `_last`, 9 `_snap`) come to ≈ 3.7 GB, staged in `/tmp` and then copied, so the peak is
    ≈ 7.5 GB.
  - That leaves ≈ 3.7 GB in `/kaggle/working`, within Kaggle's 20 GB output limit. The download is ≈ 3.9 GB.
- **Time:** training rows go from 42,888 to 46,931 (+9.4 %). On P6's 19,073 s that's ≈ 5.8 h, plus test-row inference
  for 3 snapshots × 3 seeds, so 6–6.5 h is right.

## What I checked and found sound

- **P6, from all 15 prediction files:**
  - per-seed scores, sd, threshold, cell means, per-cell medians, 6/6 and Δ_centred;
  - P6-last (+0.00874, 5/6, centred +0.00344);
  - P6 − C6 = +0.01367;
  - the additive expectation, Δ(C6) + Δ(V2) = +0.0215, against +0.0184 observed.
- **File integrity:** each main file is the mean of its snapshots (max |d| 6.4e-7), and `_last` = `_snap2` exactly.
- **The run:**
  - `candidate_flags` show `sign_head_w` 0.492066, the same as C6's JSON, plus `snapshot_cycles` 3.
  - GUARD 4 matches `51e7e4ab…`, and GUARD 5 is False on all 12 epochs for every seed.
  - The log's per-seed `Pearson_deg` (0.4561 / 0.4528 / 0.4572) matches my numbers.
  - The three checkpoint sha1s match the downloaded `.pt` files.
- **The P7 kernel and scorer:**
  - GUARD 6's main-file regex `_seed\d\.npz$` excludes `_snap` and `_last`, while its row-set check still runs on every
    file.
  - `score_p7.py`'s two `fullmatch` patterns exclude the `_snap` files.
  - It refuses when V2 is in the stack and fewer than three `_last` files are present.
  - The manifest's stack string contains `V2`.
- **V1:** paired Δs and centred Δs on all three checkpoints, with seed 0's full arm from `v1mc_s0f`.

## What I could not assess, and why

- **The smoke run itself.** I didn't rerun it, since it needs the laptop GPU. I rely on the packet's report and the code
  diff.
- **Whether Kaggle's `/tmp` holds ≈ 3.7 GB of staging.** No run has tested it: P6 wrote to `/kaggle/working`, and the
  smoke staged 300 rows. If `/tmp` is too small, the arm fails while writing a seed's ≈ 1.2 GB of predictions.
  `fail()` then leaves nothing to read, so the cost is time, not validity. `df -h /tmp` at the top of the kernel costs
  nothing and would show the capacity.
