# PACKET 034 — PRE-REGISTRATION: P7's execution (the final v9, the registered second comparison with XPert), before V2's data
packet_id: 034
created: 2026-09-30
repo_commit: 14f8e58
type: **DESIGN / execution pre-registration.** Nothing in P7 exists. It will run once, after the quota reset (Sat 00:00 UTC), with
the stack §85.11 decides. The scorer it uses is committed (84ffd1e; single-file output reproduces §87's JSON exactly).

## 1. What P7 is (§85.2 rule 10, §85.11)
The stack on all 32 training cells of `split_cold_cell_1`, seeds 0–2, scored once on its 8 test cells against O2 (XPert trained to
its published recipe, §87). The stack is **P2 + C6** (accepted), plus **V2** iff V2 is accepted and P6 confirms (§85.11), plus
**V1-full** iff it is accepted and confirmed on C6's checkpoints (§85.11). Branches, all fixed now:
| V2 | V1-full | P7 command adds |
|---|---|---|
| not accepted / P6 not confirmed | not accepted | `--sign_head_w 0.492066` |
| accepted and confirmed | — | `--sign_head_w 0.492066 --snapshot_cycles 3` |
| — | accepted and confirmed | the above, then `mc_infer_dev`-style MC inference (arm and K as accepted) on the 8 test cells |

## 2. Command and kernel guards
`xpert_arm.py --bundle xpert_mdmt_splits.npz --split split_cold_cell_1 --dp_seed_mode distinct --seeds 3 --seed_start 0 --epochs
12 --d_model 256 <stack flags> --save_pred /kaggle/working/v9p7.npz --save_ckpt /kaggle/working/v9p7.pt` — §87's v9 command
(`kern_cc1_epi_s0`) plus distinct seeding, the stack flags and 3 seeds; **no `--dev_cells`** (all training cells). Guards: 1–3 as
the dev kernels (dry-run locally against the staged upload before the push); GUARD 5 distinct; **GUARD 6: each seed's saved
`row_index` has 21,151 rows whose sorted sha1 is `be276e23…` — exactly §87's scored v9 rows.** ~1.9 GPU-h per seed (≈ 5.7 h).

## 3. Test cells touched once
The predictions are scored once, locally, by `coldcell_h2h.py --theirs <O2 profile> --ours v9p7_dev…_seed{0,1,2}.npz [--ours_alt
…_last.npz --ours_alt_label "V2-last"] --h5ad … --split split_cold_cell_1 --run_record <O2 run_record> --n_boot 20000 --seed 0`.
A kernel that fails a guard **before** writing test predictions may be relaunched (logged). A kernel that fails **after** writing
any test prediction is not rerun until those predictions are deleted unread and the failure is recorded; the rerun is disclosed.

## 4. Readings (mechanical; no new rule)
- **P7's verdict:** `coldcell_h2h.py`'s §71.3 / §71.7 reading of the seed-averaged estimand, labelled *"dev-selected increments on a
  baseline partly chosen with test-cell knowledge"* (rule 10).
- **Joint with §71/§87 (rule 10's table):** §87 read "no claim", so only rows 3–4 apply — P7 "v9 wins" → *"a dev-selected v9
  generalises better than XPert as published"* (secondary, labelled; §87 reported as no claim); anything else → no claim.
- **Secondary, never in the verdict:** per-seed blocks; the cell-centred score for **both** models (85.10); `ensemble_own` (v9's own
  row, no paired difference, §90.6); the V2-last row if V2 is stacked (90.6 amended); C6's §85.10 wording inherited.
- **Interpretability (reported):** the §85.10 measurements on P7's checkpoints on the 8 test cells — aux-readout alignment, the
  training-row prior, and the in-cell rule — by `align_dev.py` extended with a `--rows test` mode, written and committed **before P7
  runs**. The licensed sentence is §85.10's, with P7's numbers.

## ASKS
1. Is anything in 1–4 inconsistent with §85.2 rule 10, §71.3/71.7, §85.11 or §90.6 as amended?
2. The "touched once" rule in 3: is deleting unread predictions of a failed-after-writing run the right handling, or should any such
   event void P7 (it would then be a second look)?
3. The interpretability measurement on the test cells: is running §85.10's instrument on test rows a new reading that needs its own
   rule, or is reporting it with §85.10's licensed wording enough?
