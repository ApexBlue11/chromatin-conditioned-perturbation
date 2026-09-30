# PACKET 037 — RESULT: V2 (snapshot ensemble) accepted under §90.4; P6 (C6 + V2) launched
packet_id: 037
created: 2026-09-30
repo_commit: 8034ad1
type: **RESULT + GPU action within the reviewed plan.** V2 read mechanically by §90.4 (RESULTS §90.7). P6 was defined in §85.11 before
V2's data and priced in packet 033 E ("P6 this week if ~5.1 h still fit"); it was pushed at 15:10 IST.

## A. V2, 3 seeds (kernel `lincs-v9dev-v2`: the P2 command + `--snapshot_cycles 3`; GUARD 4 / GUARD 5 OK; 18,040 s)
| arm | seeds | mean (sd) | Δ vs μ0 | threshold | Δ cell means | cells | Δ_centred |
|---|---|---|---|---|---|---|---|
| **V2** (mean of 3 snapshots) | 0.45196 / 0.45852 / 0.45039 | 0.45363 (0.00431) | **+0.01669** | 0.00535 | +0.01796 | **6 / 6** | **+0.01697** |
| V2-last (final snapshot) | 0.44113 / 0.45114 / 0.44172 | 0.44466 (0.00561) | +0.00773 | (0.00677) | +0.00720 | 5 / 6 | +0.00717 |

§90.4: Δ ≥ threshold, cell means > 0, ≥ 4 of 6 cells, Δ_centred > 0 → **ACCEPTED**. Reported: schedule effect (V2-last − P2) +0.0077;
ensembling gain (V2 − V2-last) +0.0090; fraction of the 3-seed P2 ensemble recovered at one run's cost 0.57. Per-cell Δ (V2): HEK293T
+0.016, HL60 +0.018, LNCAP +0.015, SKBR3 +0.010, U937 +0.018, VCAP +0.021.
Reported, not read: V2-last alone would itself clear a rule-7 threshold computed on its own sd (0.00773 ≥ 0.00677), i.e. the cyclic
schedule helps the single final model before any averaging.

## B. Consequences, as already fixed
- **P6 (§85.11):** C6 + V2 at seeds 0–2 on the dev carve (`kern_v9dev_p6`: the P2 command + `--sign_head_w 0.492066 --snapshot_cycles 3`;
  guards dry-run against the staged upload). Confirmed iff rule 7 vs μ0, rule 8 (aux alignment ≥ 0.2532 and > 0.2292),
  Δ(stack) ≥ max(Δ C6 +0.00477, Δ V2 +0.01669) − 0.003 = **0.01369**, and Δ_centred(stack) > 0. Not confirmed → P7 uses V2 alone (the
  larger Δ). Quota: ~24 h used this week by the log runtimes; P6 ≈ 5.1 h against ≈ 6 h left — if it is cut by the quota it is rerun
  after the reset.
- **P7:** regenerated as `c6_v2` or `v2` by P6's reading (the generator takes the stack; the V2 branch adds `--snapshot_cycles 3` and
  GUARD 6 then also checks the `_snap{k}` / `_last` files); `score_p7.py` adds the V2-last alt row (§90.6 amended).
- **§90.4's rule-10 amendment** now applies: a within-run snapshot average counts as that run's prediction.
- **V1-full** (seed-0 rerun) lands ~16:00; if accepted, §85.11's rule applies it to P6's checkpoints.

## ASKS
1. Is the §90.4 reading right, and is anything mis-stated?
2. Note that P7 without C6 is a branch ("V2 alone") that `make_p7_kernel.py` does not yet have (`STACKS` has `c6` and `c6_v2`): I will
   add `v2` before P6 lands. Is "the single accepted component with the larger Δ" = V2 alone the right reading of §85.11 here?
3. Anything about running P6 in the week's last ~6 h that you would change?
