# REVIEW OF PACKET 074 (E3 at three seeds, 92.14; O9 session chain, 97.8)
verdict: SOUND
reviewed_commit: 27236fa (O9 record 81435fa)

**E3 — the reading is mechanical and right.**
- **The score:** I reran `score_dev.py --centred` on E3's three seed files against P2's (`v9dev_base2`) to scratch. The output is
  **byte-identical** to `v9_dev_score_E3_3seed.json`. The E3 − E1 contrast reproduces too: Δ −0.0037, with identical per-cell
  medians.
- **Rule 7:**
  - Δ −0.00087 against max(0.003, 0.0023) = 0.003 ✗;
  - 2 of 6 cells (U937 +0.0192, VCAP +0.0113) ✗;
  - centred −0.00066 ✗.
- **The provenance checks out:**
  - the three `.pt` files hash to `0cdd33cc…`, `31f8f3b2…` and `e35937a9…`, which are the `ENCODING_SHA1` entries and the
    rule-8 JSON's per-checkpoint sha1s;
  - the log shows the dc7a542a pin, `CHROMATIN ENCODING tie` with the registered failed list and tie mask, GUARD 4 OK, and
    GUARD 5 unequal on all 12 epochs for every seed.

**O9 — the session-2 change is operational in substance.** The kernel diff is exactly the four stated items.

The three points below are MINOR, about wording and blinding hygiene.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording (92.14) | **"The chromatin line on the dev carve is closed" mixes a finding with a decision.** The finding is that no chromatin change was accepted (E2, E1, E3). Removing cell-specific chromatin wasn't accepted either, so the record supports "no change", not "chromatin is useless" or "chromatin helps". Closing the line is a programme decision (the principal-approved direction). | Split it: *"No chromatin-encoding change was accepted on the dev carve (E2 92.10, E1 92.12, E3 92.14). Following the principal-approved direction, no further dev-carve encodings are planned."* |
| 2 | MINOR | wording (97.8, builder docstring) | **"Resume is bitwise exact … it cannot change the trajectory" is stronger than §81 shows.**<br>• **What §81 shows:** 81.3a proved the **state** round-trip bitwise. 81.3b proved bitwise **continuation** only under `use_deterministic_algorithms(True)`.<br>• **What production does:** it strips that test variable (`ENV_PROD` excludes `TEST_VARS`), and XPert's `main` sets `cudnn.benchmark = True` (`train_xpert.py:413`).<br>• **So:** neither a single-session nor a split run is bitwise reproducible, and each new session re-autotunes. A shorter budget adds a resume event whose effect is of the same kind as the run-to-run nondeterminism already disclosed (review 067 C2), **not** a systematic change. | Reword: *"The restored state is bitwise exact (81.3a), and a session ends only at an epoch boundary. A shorter budget moves the boundary. It changes the trajectory only within the GPU nondeterminism already disclosed (production does not run in deterministic mode; each session re-autotunes cuDNN)."* |
| 3 | MINOR | blinding (97.8) | **The stopper's counter is test-derived as well.** The disclosure covers `best_score` (0.3771), but 97.8 also reports "counter 5 at epoch 59". The counter is the number of epochs since XPert's **test-loss** improvement, so it reveals the test trajectory's direction. Pushing sessions doesn't need it: pushes are mechanical, and `finalize` reads the counter itself at FINAL. | Strip the counter as well as `best_score` from the after-session output. Extend the disclosure to "best_score and counter at epoch 59". |

## Answers to the asks

**Ask 1 — yes, mechanical by 92.11 / 92.9.**
- **Rule 7's cell count** is read as registered: per-cell 3-seed median Δ > 0 in 2 of 6 against the required ≥ 4, so ✗.
- **Each failing conjunct is enough on its own:** the magnitude, the cell count and the centred Δ all fail.
- **Rule 8** (0.2691; checkpoint sha1s matched) and **in-cell** (6 of 6) pass but can't rescue it.
- **The decision row** "E1 no, E3 no" means no change, so P2's encoding stays. E3's reading against P2 follows 92.11's table,
  given E1 was not accepted.

**Ask 2 — use the split wording (C1).** "No encoding was accepted" is the licensed finding. "Closed" is the PI's and the
principal's decision and should be labelled so.
- **On E3 − E1 = −0.0037:** it's larger than its seed-noise term (about 0.0021). Reported and never read, it's fine.
- **"The opposite of 93.16's motivating guess" is fair as stated.** One note: 93.16 found the codes run along gene order inside
  the no-peak blocks. So "noise" was never the only possibility; they carry a gene-position pattern a model could use. Don't
  read more into it at three seeds on the dev carve.

**Ask 3 — operational in substance; wording per C2.** I checked the three things you named.
- **The resume patch saves and restores** model / Adam / GradScaler / LambdaLR state dicts (`last_epoch` and `_last_lr`
  included), the stopper's counter, `best_score` and `early_stop`, and the CPU, both CUDA, numpy and python RNG states, all at
  the same finished epoch. 81.3a shows that round-trip is bitwise.
- **`BUDGET_H` enters only** through `DEADLINE` (the trainer stops cleanly after an epoch's save if the slowest epoch wouldn't
  fit) and the timing projection.
- **So a shorter session can't alter** the RNG restore, the stopper state or the scheduler step count. It adds one more resume
  event, and that event's only effect is the GPU nondeterminism of C2.
- **The quota arithmetic** (6.70 + 3.40 + 5.01 + 8.19 = 23.30 of 30) is from kernel logs, independent of the disclosed score.

**Ask 4 — yes, two additions.**
- **(a)** Strip and disclose the counter too (C3).
- **(b)** When O9 is read, repeat the disclosure beside the result in one line: *"the PI saw XPert's stopper best_score and
  counter at epoch 59 (§97.8). Every O9 decision is mechanical (97.3 / 97.4 / 97.6). The only discretionary act after that
  point, session 2's budget, rests on the quota arithmetic from kernel logs."*
- **(c)** Nothing else is needed. The score is their test-selected internal metric, not the registered estimand, and no
  registered rule reads it.

## What I checked and found sound

- **E3:**
  - the scratch rescore (byte-identical) and the E3 − E1 rescore;
  - the rule-7 threshold (noise term 2√((0.00169² + 0.0011²)/3) = 0.0023);
  - the `.pt` sha1s against `ENCODING_SHA1` and the rule-8 JSON;
  - the log's pin, encoding, tie-mask and guard lines;
  - the 92.14 text.
- **O9:**
  - the diff 81435fa~1 → 81435fa (the kernel, its metadata and `make_o9_kernel.py`'s bounded `BUDGET_H` substitution);
  - how the kernel uses `BUDGET_H` (`DEADLINE`, the timing projection);
  - the resume patch's save and restore list;
  - §81.3 / 81.7's exactness tests;
  - XPert's cuDNN settings (`train_xpert.py:413`, `utils.py:312–313`).

## What I could not assess, and why

- **Session 2's own log.** It's running.
- **Rule 8 by my own rerun.** It's heavy on this laptop, so I matched its JSON's checkpoint sha1s instead.
- **P10's arm-M code.** It's with the worker and comes as its own packet.
