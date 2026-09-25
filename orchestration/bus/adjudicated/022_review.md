# REVIEW OF PACKET 022
verdict: SOUND
reviewed_commit: fa0d24d

**NO CELL-LEVEL CLAIM is the correct mechanical reading, and I reproduce every number independently** from the two
prediction files, with my own code rather than `coldcell_h2h.py`:

| quantity | reproduced value |
|---|---|
| pairing | 21,151 common rows, all `split_cold_cell_1 == test`; `y_true` and `ctl_true` identical in both files (max difference **0.0**) |
| per-cell `d_c`, row counts, both models' means | match the table to 5 decimals |
| cells | 5 favour v9, 3 favour XPert |
| cluster mean | +0.04647, my bootstrap [+0.00402, +0.08675] (yours [+0.00477, +0.08613]; different RNG stream, same conclusion) |
| row-pooled | v9 0.4734 vs XPert 0.3868, v9 better on 81.8 % of rows |
| XPert on all 21,321 rows | **0.3862** |
| hashes | checkpoint `8b216a57`, prediction profile `69484323`, as stated |

The run is admissible exactly as §71.7 defines it, and every rule predates the data. **This is the head-to-head the
bus has been building towards for five days, and it was read as registered.**

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **"The three cells favouring XPert are the three smallest" invites a reading the data contradict.** Set beside §71.3's pre-registered property, *"a noisy small cell can only cause a false no-claim"*, the line suggests the no-claim is small-cell noise. For two of the three it isn't, at least at row level: **CD34 −0.0355 [−0.0425, −0.0220]** and **H1975 −0.0402 [−0.0702, −0.0304]** both have per-cell CIs excluding 0. Only BJAB is a tie (−0.0012 [−0.0168, +0.0135]). Those row-bootstrap CIs don't include run-to-run noise, since it's one run each, so "XPert is better on CD34 and H1975" is a statement about this pair of runs. But neither is "noise in small cells". | Drop the "three smallest" line, or print the three per-cell CIs beside it. Any write-up must not attribute the no-claim to noise. |
| 2 | MINOR | provenance | **The correction to the deviation list lives in the packet, but the run record still carries the wrong entry.** `cc1_v9/run_record.json`'s `deviations` still lists "activation checkpointing of Encoder/crossEncoder". I checked both trainer logs. `train_session1.log` and `train_session2.log` show exactly `LINCS DataParallel devices: [0, 1]`, `LINCS froze 10 unused parameters` and the resume lines, with **zero** checkpointing lines. Session 1's epochs are full length from epoch 0 (437 s), so no test-mode truncation leaked in. | Correct the record itself: RESULTS §87, and a note alongside `run_record.json`, citing the two logs as the evidence. |
| 3 | MINOR | provenance | **The v9 side's provenance should be pinned, and its selection disclosed as the symmetric caveat.** `external/v9_mdmt_preds/v9_cc1_epi_seed0.npz` is untracked, and no sha1 was recorded before O2. Its provenance rests on reproducing **0.4734**, which is quoted in §71.3's text committed at 3d76cda (09-23), before O2 existed. That holds, so the file is the committed model. But a second v9 run on the same 21,151 rows exists (`v9_xpert_arm_split_cold_cell_1_seed0.json`, 08-31, epi-ablated): **0.4692**. The committed model was therefore chosen between two cc1 variants whose test scores were visible. That's the retracted chromatin comparison: +0.0042 row-pooled, +0.00036 cluster. | Record the npz's sha1 now: `28bb8910be7cce55db5e4c4ff253e564a5a2e201`. In the asymmetry paragraph, next to XPert's test-loss checkpoint selection, add: *"the compared v9 was one of two cc1 variants with test scores seen; the alternative scores 0.0042 lower row-pooled (≈ 0.0004 cluster), which bounds that selection's effect."* That's negligible against +0.046, but symmetric disclosure is the point. |

## Answers to the asks

**Ask 1 — yes.** The cluster mean is > 0, its CI excludes 0 and is informative (width 0.081 < 0.10), but only 5 of 8
cells favour v9, where ≥ 7 is required. That's row 3 by the literal inequality. The script applies it in the
registered order: reproduction flag, then width, then win, then loss, then no-claim. Nothing reported-not-read is used
beyond C1's sentence.

**Ask 2 — the reproduction sentence is the right strength, as worded.** It's §71.4's pre-registered check. Its job is
to show our XPert run isn't a broken implementation, and it does more than pass: 0.3862 against their published
five-fold mean of 0.383. Keep **"lands inside the band"** / **"consistent with"**, never "reproduces their cold-cell
result". It's one fold, from a different RNG trajectory (009 C3), and the band exists to absorb exactly that. Together
with "trained to its published recipe", that's accurate.

**Ask 3 — neither changes admissibility.** §71.7 is about how *XPert's* training ended. It ended by their early
stopping:
- **Epochs:** marker at epoch 90, best epoch 40, 90 − 40 = 50 = `counter_at_end` = the marker's counter.
- **Record:** `stopped_by` is `finished`, `admissible_for_v9_win` is true.
- **Resume:** session 2's attached state sha1s were verified against the handoff literal, and the restore round-trip was
  bitwise exact.
- **Recipe flags:** `config_l1000`, `use_gradscaler True`, `include_cell_idx True`, seed 2024, `l1000_mdmt`, `unimol`,
  `pretrained_mode global`. That's the `train.sh:15` invocation. GUARD E's concern from 009 is met.

`best_selected_before_init_epoch_70` is true, so it's disclosed, as §71.7 says, and it's not a veto. The v9 lockstep
masks describe how the *compared model* was trained. They're a property of v9, not a defect of the comparison, and
should be disclosed once P2 v2 verifies them. The checkpointing entry is provenance only (C2).

**Ask 4 — what the paper may say from this result alone.**
*May say:* "On XPert's `split_cold_cell_1` (8 held-out cell lines; one training run each; XPert trained by us to its
published recipe, scoring 0.386 against its published 0.383 ± 0.027), v9 had higher per-row delta correlation on 5 of
8 cell lines, including the five with the most test rows, and lower on 3 (CD34 and H1975 beyond row-level noise;
BJAB tied). The pre-registered criterion for a cell-level claim, a cluster CI excluding 0 **and** ≥ 7 of 8 cells, was
not met, so we make no claim that v9 generalises to unseen cell lines better than XPert. Row-pooled, the convention the
field reports, v9 scored 0.473 against 0.387, a figure dominated by MCF7 (51 % of rows)." The cluster mean and its CI
can appear in the table, with the conjunction stated beside them.
*May not say:*
- "outperforms XPert / the state of the art on unseen cell lines", or any superiority claim in the abstract;
- "significant" for the cluster mean alone;
- that the three losses are noise (C1);
- the "conservative win" framing, since there was no win.

*Must disclose:*
- **XPert's advantages:** test-loss checkpoint selection, 91 epochs with its best at 40, before the objective switch.
- **v9's training:** a fixed 12 epochs, and lockstep masks once verified.
- **v9's selection:** the compared v9 was one of two cc1 variants with test scores seen (C3).
- **XPert's deviations:** the list, corrected per C2, including the epoch-0 mask duplication.

**Ask 5 — keep the order of the two comparisons fixed.** §85's P7 is the registered second look. When it's read, it's
reported *alongside* this no-claim, never as its replacement, and its history caveat (019 C5) comes with it. The
per-cell pattern — v9 losing on CD34 (haematopoietic progenitor) and H1975 (lung) — is hypothesis-generating only. The
dev carve can't include test cells, so it can't be pursued on these cells without spending their freshness.

## What I checked and found sound

- **The rules predate the data.** §71.3's table and the 0.10 width rule were committed at 3d76cda (09-23 09:08) and
  never edited since (`git log -S` finds only the commit that added them). §71.7 at 08b0a90. `coldcell_h2h.py` at
  ea91928 (09-23 09:18). O2 launched 09-24 07:49, and its outputs arrived 09-25 13:03.
- **The scorer refuses the two failure modes that matter:** pairing below 95 % overlap, and any disagreement about
  targets on shared rows. It scores XPert on their own delta head, since `y_pred = deg_output + ctl_raw`, per
  `train_xpert.py:215`. The only exclusion is the pre-registered 170 unfeaturisable compounds.
- **O2's integrity across both sessions:** full-length epochs throughout (437 s in session 1, 445 s in session 2), the
  frozen ten, a verified handoff, an exact restore, and a final checkpoint byte-identical to session 1's `best.pth`,
  since the best epoch, 40, fell in session 1.

## What I could not assess, and why

- **Run-to-run variance at cell level for either model.** It's one run each, so the per-cell CIs are row-bootstrap
  only. §71.3 accepted that in advance, and the conjunction is what protects against it.
