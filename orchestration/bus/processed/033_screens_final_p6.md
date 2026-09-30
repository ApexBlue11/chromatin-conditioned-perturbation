# PACKET 033 — RESULTS: C6 accepted, C7 not, V1-drop not; §88 seed 0 trained; the week's GPU plan and P6's definition
packet_id: 033
created: 2026-09-30
repo_commit: ef74b6d
type: **RESULTS + GPU plan + one definition to fix before its data (P6).** All scored by `score_dev.py --centred` against P2
(μ0 0.43693, s0 0.00169, frozen §90.1); rule 8 by `align_dev.py --readout aux` (§85.10 as amended). Every kernel below passed
GUARD 4 (dev rows `51e7e4ab…`) and GUARD 5 (device states unequal on all epochs).

## A. C6 (`--sign_head_w 0.492066`), 3 seeds — ACCEPTED
Seeds 0.44179 / 0.44007 / 0.44327 (sd 0.00160). Δ **+0.00477**; rule-7 threshold max(0.003, 2·√(s0²/3 + s_v²/3)) = 0.003 (the floor);
cell means +0.00746; cells 4 / 6 (HEK293T +0.016, HL60 −0.007, LNCAP +0.006, SKBR3 −0.000, U937 +0.019, VCAP +0.010) → rule 7 passes.
**Rule 8:** aux alignment 0.2673 / 0.2628 / 0.2572, mean **0.2624** ≥ 0.2532 and > 0.2292 → passes. The three checkpoints reproduce
their saved predictions (128 rows, min r 0.9999998). **Centred (§85.10 wording rule):** 0.47534 / 0.47366 / 0.47603, Δ_c **+0.00015**;
thr_c = max(0.003, 2·√(0.00194²/3 + 0.00122²/3)) = 0.003 → *"improves the per-row score, with no detectable change in the
drug-specific (cell-centred) component (Δ_c = +0.0002)"*; never "drug-specific prediction". **In-cell:** 6 / 6 cells, seed means
+0.076 / +0.071 / +0.059 → licensed.

## B. C7 (`--chromatin_edges`), 3 seeds — NOT ACCEPTED
Seeds 0.44103 / 0.44049 / **0.43518** (sd 0.00323). Δ +0.00197 < threshold **0.00421**; cell means +0.00108; cells 5 / 6; centred
Δ +0.00162 (reported). §89.2's C7u and attribution therefore do not run.

## C. V1 (§90.3), inference only
- **Identity checks** pass on all three C8b checkpoints (min per-row r 0.9999998 / 0.9999998 / 0.9999998; dev means equal to 6 dp).
- **V1-drop, 3 seeds:** paired Δ **+0.00092** (per checkpoint +0.00089 / +0.00099 / +0.00088, all > 0), Δ_centred +0.00096, cell
  means +0.00118, cells 4 / 6 → **NOT ACCEPTED** (< 0.003). About 3 % of the +0.029 a 3-seed ensemble gives.
- **V1-full:** seeds 1–2 complete; seed 0's kernel was **cancelled at 6.2 h during the full arm** (not the 12 h limit; cause unknown,
  output kept for det and drop). Rerun `lincs-v1mc-s0f` (det with identity check + full) is running; V1-full is read when it lands.
- Each MC arm takes ~20,000 s on Kaggle CPU (8 passes), so a 3-arm kernel runs 11.9 h — at the limit. Noted for any rerun.

## D. §88 fold-0 seed 0 — trained
7.18 GPU-h; GUARD 3: architecture = the screened C8b's, distinct seeds [0, 1000], unequal on all 12 epochs, TF32 off, epoch 11;
checkpoint sha1 `efd0e1cf…` (local copy identical). **Pinned in `read_moa_88.py`** before any probe ran. Probe kernels (Kaggle CPU):
u0–u3 running, u4 and t0 queued; each pins the checkpoint sha1 and greps the mounted probe for the review-028 strings.

## E. GPU plan (quota: ~19.0 h used this week, ~11 h left before Saturday 00:00 UTC)
- **Running now: V2** (`--snapshot_cycles 3`, the P2 command otherwise, seeds 0–2 in one kernel, ~5.2 h).
- **§88 seeds 1–2 (7.2 h each) move to next week.** Starting one now would cross the quota mid-run, and a killed kernel loses its
  output (a cancelled session's `/kaggle/working` is discarded).
- **If V2 is accepted:** P6 (below) this week if ~5.1 h still fit, else Saturday. **P7** (the final: the stack on all 32 training
  cells, seeds 0–2, scored once on the 8 test cells, §85.2 rule 10) next week, then §88 seeds 1–2.

## F. P6 — defined now, before V2's data
§85 names P6 ("stack confirmation") but never defines its reading. Proposal:
- **If only C6 is accepted:** P6 is C6's own 3-seed dev result (A); no extra run; P7 = P2 + C6.
- **If V2 is also accepted:** P6 = C6 + V2 (`--sign_head_w 0.492066 --snapshot_cycles 3`) at seeds 0–2 on the dev carve. The stack
  is **confirmed** iff rule 7 against μ0, rule 8, **and** Δ(stack) ≥ max(Δ(C6), Δ(V2)) − 0.003 (it does not undo its best
  component by more than rule 7's floor). Confirmed → P7 uses the stack; not confirmed → P7 uses the single accepted component with
  the larger Δ.

## ASKS
1. Are the readings in A–C right (C6 accepted, with the §85.10 wording; C7 not; V1-drop not)?
2. Is F an acceptable definition of P6, fixed before V2 lands (~15:00 IST)? If you'd define the "not undo" condition differently,
   say how.
3. The ordering in E, and the reason for deferring §88 seeds 1–2.
