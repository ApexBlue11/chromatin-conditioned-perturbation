# PACKET 022 — RESULT: the pre-registered v9-versus-XPert cold-cell head-to-head (O2 complete)
packet_id: 022
created: 2026-09-25
repo_commit: fa0d24d
type: **RESULT.** Read mechanically by `model/v9/coldcell_h2h.py`; nothing here was chosen after seeing it.

## The rules this is read under, verbatim
From the current RESULTS.md:

### 71.3 The reading, as inequalities, with the asymmetry of §69.1 built in
| result | reading |
|---|---|
| cluster mean **> 0**, cluster CI **excluding 0**, and **≥ 7 of 8** cells favour v9 | **v9 generalises to these unseen cell lines better than XPert as published.** Conservative, because XPert selected its checkpoint on these test rows. One training run each. |
| cluster mean **< 0**, CI excluding 0, ≥ 7 of 8 favour XPert | **Uninterpretable as a model comparison** — the win may be the test-guided checkpoint selection. Reported plainly, not explained away. |
| anything else | **No cell-level claim.** The per-cell table and the row-pooled number are reported, the latter explicitly as MCF7-dominated. |

**Informative width, fixed in advance:** a cluster CI wider than **0.10** is uninformative whatever its
sign. (For scale: v9's fold-1 number is 0.4734 against XPert's published cold-cell 0.383, a gap of about
0.09, so an interval narrower than 0.10 is the smallest that could separate a gap of that size from zero.)

**RESULTS 71.7, verbatim as committed at `08b0a90`:**

### 71.7 🔴 AMENDMENT, required by review 008 as the condition of GO — committed before launch, no data seen
**What a guard-stopped run licenses.** Their early stopping monitors test `loss4` [§69.1], which normally
*favours* XPert — the reason a v9 win is "conservative". But if the 8.3 h guard fires while test loss is still
improving, the kernel loads XPert's best-*so-far* checkpoint. XPert is then **under-trained**, and a v9 win is
**inflated**, not conservative. The asymmetry reverses exactly in that case, and §71 never said so.

Pre-committed, as an inequality:

| training ended by | `counter_at_end` | reading |
|---|---|---|
| their early stopping (`finished`) | 50 by construction | §71.3 applies unchanged |
| **the wall-clock guard** | **≥ 45** of patience 50 | effectively converged; §71.3 applies |
| **the wall-clock guard** | **< 45** | **supports NO v9-win claim**, whatever the numbers — XPert was still improving when cut off |
| crash | — | void; the kernel fatals |

The **45** is my resolution of the review's *"a watchdog stop with the counter near patience is effectively
converged"*: 90 % of patience without improvement. The review's own inequality, taken literally, would forbid
any guard-stopped claim, since a guard stop always has counter < 50; I have taken its stated intent instead and
say so here so the choice is visible. The kernel writes `admissible_for_v9_win` from exactly this rule.

**How convergence is measured** (review 008 C2). `counter_at_end = last_epoch_index − best_epoch`, where
`last_epoch_index` is parsed from their `Epoch {n}, Valid Total Loss` line and `best_epoch` is read from the
checkpoint their stopper wrote. This is exact: every epoch after the best is by definition non-improving. The
kernel's first draft recorded `early_stop_counter_hits` = the count of `EarlyStopping counter` lines, which the
reviewer showed is the **total** number of non-improving epochs over the run, because an improving epoch
resets the counter **silently** (`utils.py` `step()`: `self.counter = 0`, no log line). Verified in the code.
That field is renamed `nonimproving_epochs_total` and marked descriptive-only; the last logged counter is kept
as a cross-check against `counter_at_end`.

**Disclosed, not vetoed:** if `best_epoch < 70`, XPert's checkpoint was selected on the "accelerated"
`batch_weighted_loss` objective before the switch at `init_epoch`, and never benefited from the full one. The
kernel records `best_selected_before_init_epoch_70`.

**Per-cell uncertainty** (review 008 C4). Each of the 8 `d_c` is reported with its own bootstrap CI over that
cell's rows and its scored-row n, so that BJAB (73 scored of 84) and H1975 (54) are visibly noisy rather than
eight equal-looking numbers. The reviewer's point about the conjunction is recorded as the reason the
unweighted mean is kept: requiring **both** the cluster CI **and** ≥ 7/8 cells means a noisy small cell can
only cause a false **no-claim**, never a false v9 win.

**On 71.6.** The reviewer noted `n_dropped_test_unfeaturisable = 170` was already on disk in
`model/results/v9_xpert_arm_split_cold_cell_1_seed0.json`. It then corrected its own tally — 008a went out
before its review, so the ask was answered by us, not by it. Both are true: I answered it before review, and I
did so by recomputing a number the repo already held. A small method-rule-20 miss on my side, not a packet
defect. The reviewer's independent check agrees in full, and my converse check (0 of 21,151 kept rows
unfeaturisable) is the half that shows the exclusion is *exactly* the compound rule.

## What happened
**How O2 ended (84.1):** their early stopping, in session 2 (kernel v9, 3.24 h): marker epoch **90**, counter 50,
best epoch **40**, `counter_at_end == 50 == marker counter` asserted. 91 epochs over two sessions, ~11.5 GPU-h.
`best_epoch < 70`: the checkpoint was selected on their "accelerated" objective before the switch at `init_epoch` 70
— disclosed, not vetoed (71.7). The final checkpoint is byte-identical to session 1's `best.pth` (sha1 `8b216a57`),
as it must be. Prediction profile sha1 `69484323`; read by `model/v9/coldcell_h2h.py` →
`model/results/coldcell_h2h_split_cold_cell_1_O2.json`.

## The result
| cell | rows | d_c = median(r_v9 − r_XPert) [95 % CI] | v9 | XPert |
|---|---|---|---|---|
| MCF7 | 10815 | +0.07513 [+0.07278, +0.07716] | 0.4459 | 0.3686 |
| HT29 | 5837 | +0.12246 [+0.11917, +0.12611] | 0.5371 | 0.4179 |
| MDAMB231 | 2188 | +0.06965 [+0.06569, +0.07366] | 0.4728 | 0.4099 |
| HS578T | 1074 | +0.10606 [+0.10198, +0.11017] | 0.5128 | 0.4038 |
| THP1 | 815 | +0.07535 [+0.06357, +0.08384] | 0.4044 | 0.3345 |
| CD34 | 295 | -0.03545 [-0.04248, -0.02198] | 0.3037 | 0.3289 |
| BJAB | 73 | -0.00122 [-0.01675, +0.01346] | 0.3251 | 0.3246 |
| H1975 | 54 | -0.04019 [-0.07018, -0.03042] | 0.5240 | 0.5826 |

| | value |
|---|---|
| **cluster (estimand of record):** mean d_c | **+0.04647** [+0.00477, +0.08613], width 0.081 (informative: < 0.10) |
| cells favouring v9 / XPert | **5 / 3** (sign p 0.727) |
| row-pooled (labelled: MCF7 = 51.1 % of rows) | +0.08667 [+0.08527, +0.08811]; v9 0.4734 vs XPert 0.3868; v9 better on 81.8 % of rows |
| **reproduction (71.4):** XPert on all 21321 test rows | **0.3862**, inside the band [0.302, 0.464] around their published cold-cell 0.383 ± 0.027 |
| admissible for a v9-win claim (71.7) | True |

⇒ **Pre-committed reading, 71.3 row 3: NO CELL-LEVEL CLAIM.** The cluster mean favours v9 with a CI excluding 0
and informative, but **5 of 8** cells favour it where 71.3 requires ≥ 7. Reported as 71.3 prescribes: the per-cell
table and the row-pooled number, the latter labelled as MCF7-dominated. **XPert trained to its published recipe on
`split_cold_cell_1`** — not "their cold-cell run reproduced" — but its score on these rows, 0.386, lands inside the
band of their published five-fold cold-cell mean (0.383 ± 0.027).
*Reported, not read:* the three cells favouring XPert are the three smallest (CD34 295 rows, BJAB 73, H1975 54). 71.3
said in advance that its conjunction lets a noisy small cell cause a false no-claim, never a false win; that is not
re-read. The registered second comparison is §85's P7 (a dev-selected v9, 3 seeds), read under the same rule.

## Deviations and disclosures
**Deviations of the XPert run, as recorded, with one correction.** (1) flash_attn shim; (2) `MyDataset`: one tensor
per drug instead of per row, values proven identical (`prove_mydataset_patch.py`, sha1 `8d8c50f24a9c` →
`31b5ef68355a`); (3) `all_drugs_unimol_arr.npy` rebuilt from the released npz; (4) DataParallel over two T4s inside
`XPertNet.forward`, float64 gradients equal to one GPU to 5e−16 — with the two GPUs' dropout masks **identical in
epoch 0** (review 018 C1); (5) the ten never-used parameters frozen; (6) full-state resume across the session
boundary, exact. **Correction:** the run record's list also names "activation checkpointing"; it was carried over
from the v7 kernel's list and is **wrong for the trained model** — the production trainer applied only the
DataParallel patch and the resume hooks (its log shows `LINCS DataParallel devices: [0, 1]` and no checkpointing
line); checkpointing ran only inside GUARD F's one-batch probe. Provenance only, not a deviation: empty
`__init__.py` in `datasets/` and `models/` (008b).
**v9 side, disclosed:** the committed model (`v9_cc1_epi_seed0`, trained on Kaggle) very likely trained under
DataParallel with lockstep masks for its whole run (85.5, review 021 C1; verification running in P2 v2). The budget
asymmetry runs in XPert's favour: 91 epochs with test-loss checkpoint selection against v9's fixed 12.

## ASKS
1. Is NO CELL-LEVEL CLAIM the correct mechanical reading, and is anything reported-not-read being used?
2. Is "XPert trained to its published recipe on `split_cold_cell_1`" plus the reproduction-band sentence the right
   strength, or does the reproduction sentence overreach?
3. Do the v9-side lockstep caveat (pending verification) or the corrected checkpointing entry change admissibility?
4. What may the paper say about v9 versus XPert on unseen cell lines from this result alone?
5. Anything else.
