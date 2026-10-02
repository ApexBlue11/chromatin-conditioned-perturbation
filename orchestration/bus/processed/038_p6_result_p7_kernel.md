# PACKET 038 — RESULT: V1 not accepted; P6 (C6 + V2) CONFIRMED under §85.11; P7 kernel regenerated for `c6_v2`
packet_id: 038
created: 2026-09-30
repo_commit: 5b0da11
type: **RESULT ×2 + GPU plan (P7, next week's first push).** Both results are read mechanically by the rules fixed before
their data (§90.3 for V1, §85.11 as amended by review 037 C1 for P6). P7's recipe is then fixed by §85.12 item 1.

## A. V1 (MC inference, §90.3) — RESULTS §90.8: **neither arm accepted**
Inference only, on the three C8b dev checkpoints (identity checks: min per-row r 0.9999998 on every checkpoint); K = 8; paired
against the recomputed deterministic arm. Seed 0's full arm comes from the rerun `lincs-v1mc-s0f` (the first kernel was
cancelled at 6.2 h, cause unknown); its deterministic arm is bitwise identical to the original (max |diff| 0.0).

| arm | paired Δ (3-ckpt mean) | per checkpoint | Δ cell means | cells | Δ_centred | §90.3 |
|---|---|---|---|---|---|---|
| V1-drop | +0.00092 | +0.00089 / +0.00099 / +0.00088 | +0.00118 | 4 / 6 | +0.00096 | NOT ACCEPTED |
| V1-full | +0.00104 | +0.00100 / +0.00111 / +0.00100 | +0.00136 | 4 / 6 | +0.00107 | NOT ACCEPTED |

Both fail the 0.003 floor; every other conjunct passes. MC averaging recovers ≈ 3–4 % of the +0.029 that averaging 3 seeds
gives. V1 does not enter the stack; P7 uses no MC inference.

## B. P6 (§85.11) — RESULTS §85.13: **CONFIRMED** → P7 stack `c6_v2`
Kernel `lincs-v9dev-p6`: the P2 command + `--sign_head_w 0.492066 --snapshot_cycles 3`, seeds 0–2, dev carve (rows sha1
`51e7e4ab…`). `candidate_flags` in the arm JSON record both flags. GUARD 4 OK; GUARD 5 distinct, device RNG states unequal on
all 12 epochs for every seed. 19,073 s. Scored by `score_dev.py --centred` against P2 (μ0 0.43693, s0 0.00169; P2 centred 0.47486).
Rule 8 by `align_dev.py --readout aux --no_nulls` on the three final-snapshot checkpoints (sha1 `1c1a5d92…`, `7aaa6451…`, `d07b0164…`).

| arm | seeds | mean (sd) | Δ vs μ0 | Δ cell means | cells | Δ_centred | aux alignment |
|---|---|---|---|---|---|---|---|
| **P6 = C6 + V2** (mean of 3 snapshots) | 0.45607 / 0.45284 / 0.45722 | 0.45537 (0.00227) | **+0.01844** | +0.02144 | 6 / 6 | **+0.01326** | **0.2650** |
| P6-last (final snapshot alone) | 0.44606 / 0.44360 / 0.44737 | 0.44568 (0.00191) | +0.00874 | +0.00986 | 5 / 6 | +0.00344 | — |
| V2 alone (§90.7) | 0.45196 / 0.45852 / 0.45039 | 0.45363 (0.00431) | +0.01669 | +0.01796 | 6 / 6 | +0.01697 | 0.2675 |
| C6 alone (§85.8) | 0.44179 / 0.44007 / 0.44327 | 0.44171 (0.00160) | +0.00477 | +0.00746 | 4 / 6 | +0.00015 | 0.2624 |

§85.11's conjuncts:
- rule 7: Δ +0.01844 ≥ max(0.003, 2√(s0²/3 + 0.00227²/3)) = 0.00327. OK.
- rule 7: cell means +0.02144 > 0. OK.
- rule 7: 6 of 6 cells (≥ 4 needed). OK.
- rule 8: aux 0.2650 ≥ 0.2532 and > 0.2292. OK.
- stack floor: Δ(stack) 0.01844 ≥ max(0.00477, 0.01669) − 0.003 = 0.01369. OK.
- centred: Δ_centred +0.01326 > 0. OK.

Result: **CONFIRMED**. Aux alignment by seed 0.2757 / 0.2646 / 0.2546. In-cell licensed at 5 / 6 cells (HL60 −0.0001; V2 alone had 6 / 6).
Seed means +0.074 / +0.051 / +0.063.

**Reported, not read** (as written in §85.13):
1. Per row, the stack adds +0.00175 over V2 alone, less than C6's own +0.00477 over P2. Seed-paired P6 − V2 is +0.00411 / −0.00569 / +0.00682,
   mixed in sign, so **the stack is not shown to beat V2 alone**. §85.11 only asks that it not fall more than 0.003 below its best component.
2. On the cell-centred score the stack is **0.00371 below V2 alone** (seed-paired −0.00287 / −0.00965 / +0.00138), and C6 alone moved it by
   only +0.00015. The sign head's gain sits in the per-cell mean profile, not in ranking drugs within a cell. The paper will state this
   beside any C6 claim.
3. Per-cell Δ (P6, median): HEK293T +0.0165, HL60 +0.0202, LNCAP +0.0086, SKBR3 +0.0096, U937 +0.0403, VCAP +0.0248.

## C. P7 kernel (§85.12), regenerated: `python orchestration/make_p7_kernel.py c6_v2`
- The command is §85.12 item 2 with the bracket filled: `... --epochs 12 --d_model 256 --sign_head_w 0.492066 --snapshot_cycles 3
  --no_test_metrics --save_pred /tmp/p7stage/v9p7.npz --save_ckpt /tmp/p7stage/v9p7.pt`. There is no `--dev_cells`, so all 32
  training cells are used; seeds 0–2 with distinct seeding. The manifest's stack string is `P2 + C6 + V2`.
- The generator re-asserted that all six uploaded files equal the repo up to line endings; the kernel pins their sha1s
  (GUARD 2b) and refuses to run on a mismatch.
- **One change to the kernel since review 035: the local smoke only.** With `--snapshot_cycles 3` the arm asserts
  `epochs % snapshot_cycles == 0`, so the smoke override now runs `--epochs 3` when `--snapshot_cycles` is in the argv, and 1 otherwise.
  The Kaggle path does not change.
- **Smoke passes end to end** (dev carve, 300 rows, 1 seed, laptop GPU; never test rows). GUARD 6 and the manifest now cover
  `v9p7_dev6s0_seed0{,_last,_snap0,_snap1,_snap2}.npz`; `_last` is byte-identical to `_snap2` (sha1 `2f6c1f2b…`), as it should be.
  The captured log has 0 lines matching `pearson`. The smoke's arm JSON was moved out of `model/results`.
- **Naming checked against `score_p7.py`:** without `--dev_cells` the arm writes `v9p7_seed{k}.npz` and `v9p7_seed{k}_last.npz`.
  These are exactly `score_p7.py`'s two regexes. The manifest stack contains `V2`, so `score_p7.py` adds the three V2-last files as
  `--ours_alt … --ours_alt_label V2-last` (§90.6 as amended), and it refuses if fewer than 3 are present.
- The test-row alignment (`align_dev.py --rows test`, once, after `P7_SHA1` is pinned) reads the saved `.pt`, i.e. the final-snapshot
  weights, the same weights §85.13's rule 8 was read on.
- **Price and timing:** P6 took 5.3 h on 26 training cells. P7 has all 32, so ≈ 6–6.5 h on one 2×T4 session (under the 12 h limit).
  This week's quota is spent (≈ 29 h of 30), so the push happens right after the reset (Sat 00:00 UTC). §88 t1 / t2 (≈ 7.2 h each)
  follow in the same week: ≈ 21 h total.

## ASKS
1. Are both readings (V1, P6) right under their rules, and is anything in §85.13's "reported, not read" block mis-stated or
   missing? In particular: is it correct to keep C6 in P7 on §85.11's rule while reporting that the stack does not beat V2
   alone, and that C6 adds nothing on the centred score?
2. Does the smoke-only epochs change need anything else before the push? For example, should the smoke also assert
   `_last == _snap2`?
3. Anything to change in P7 before it runs? After it runs, the test cells are touched exactly once.
