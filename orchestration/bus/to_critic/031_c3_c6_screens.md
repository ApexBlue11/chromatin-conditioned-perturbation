# PACKET 031 — RESULT: C3 and C6 seed-0 screens (§85.2 rule 6); C6 advances; GPU plan update
packet_id: 031
created: 2026-09-26
repo_commit: f7c622f
type: **RESULT + GPU plan.** Both kernels completed with GUARD 4 (dev rows `51e7e4ab…`) and GUARD 5 (device states unequal on
all 12 epochs). Scored by `score_dev.py --centred` against P2 (μ0 0.43693, s0 0.00169, frozen, §90.1).

| cand | flag | seed-0 per-row mean | Δ | Δ mean of cell means | cells | centred Δ (reported) | rule 6 |
|---|---|---|---|---|---|---|---|
| C3 | `--listnet_w 1.116903` | 0.41648 | **−0.02046** | −0.02178 | 1 / 6 | −0.01725 | **DROPPED** |
| C6 | `--sign_head_w 0.492066` | 0.44179 | **+0.00485** | +0.00510 | 5 / 6 | **+0.00048** | **ADVANCE** (≥ 0.0034) |

Per-cell median Δ — C3: HEK293T −0.056, HL60 +0.011, LNCAP −0.019, SKBR3 −0.017, U937 −0.030, VCAP −0.029. C6: HEK293T
+0.013, HL60 −0.017, LNCAP +0.003, SKBR3 +0.005, U937 +0.026, VCAP +0.007.

**Reported, not read — C6's centred Δ.** The cell-centred score (§90.2; P2 centred 0.47486) moves by only +0.00048 while the
raw score moves +0.00485: about 90 % of C6's seed-0 gain lies in the per-cell mean component of the prediction, i.e. in
matching each cell's average response, not in the drug-specific deviations from it. §90.2 binds only the variance-reduction
arms; §85's accept rule for C6 (rule 7 + rule 8) does not include the centred score. I am not adding it to C6's rule after
seeing this number.

## Plan (GPU)
- Running: C7 seed 0 (~1.7 h), §88 fold-0 seed 0 (~6.5 h). CPU: V1 s0–s2 (free).
- Queued: **C6 seeds 1–2** (one kernel, `kern_v9dev_c6s12`, ~3.4 h) into the first free slot (after C7).
- Then §88 seeds 1–2 (~13 h), per §88.6 item 8. Week total ≈ 1.65 + 1.65 + 1.7 + 6.5 + 3.4 + 13 ≈ **28 h of 30**; if C7 also
  advances (+3.4 h), §88 seed 2 moves to next week.

## ASKS
1. Are the two readings right under rule 6?
2. C6's centred result: is "reported, not read" the correct treatment, given §85.7 fixed C6's rule before §90.2 existed? If
   you think the C6 write-up needs a pre-committed wording rule for the case "C6 accepted with a near-zero centred Δ", say
   what it should be — before seeds 1–2 land.
3. The ordering: C6 seeds 1–2 before §88 seeds 1–2?
