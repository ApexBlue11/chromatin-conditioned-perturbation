# PACKET 090 — OBSERVATION + DESIGN/CODE: T-B on seed 0's drug path did not train; a guard for T-C (RESULTS §100.7, §100.8)
packet_id: 090
created: 2026-10-18
type: **OBSERVATION REVIEW + DESIGN/CODE REVIEW** (T-B is not read; nothing here selects on dev scores)

**Commits:**
- 42ae31e: the ledger;
- 70f5b4a: §100.7 and §100.8, registered before any guard code;
- 8c2c714: the guard code.

**Files:**
- `model/results/v10c/tb_ledger.jsonl`;
- `model/v10c/diag_dose_collapse.py` → `model/results/v10c/tb_collapse_diag.json`;
- `model_v10c.py`, `train_v10c.py`, `orchestration/make_v10c_kernels.py`, `test_v10c_model.py`.

## What happened
Three of the six d = 256 T-B runs are complete: off s0, on s0 and on s1. Off s1 and on s2 are running; off s2 and the d384
screen are queued.

**`tb_on_s0`:**
- The training contrast loss (1 − PCC) averaged 0.999, 0.993, 0.994 and 1.001 over the four epochs.
  - The same seed's off arm: 0.908, 0.867, 0.804, 0.832.
  - On seed 1: 0.921, 0.873, 0.817, 0.820.
- The held-out contrast r is −0.0001, and exactly 0.0 on LNCAP.
- Delta r is 0.404: the basal path trained.

**The CPU diagnosis** (480 random dev rows and 480 random training rows per checkpoint):
- **on s0:** the dose scale s = σ(a_d(ld − b_d)) is **0.000000 on 100 % of rows**; the mean γ is 0.989, near its init of 1.
- **on s1 and off s0:** median s 0.116 and 0.134 (10–90 %: 0.05–0.26); mean γ 0.43.
- **The on s0 drug-path weights** sit near init scale and shrank under weight decay:
  - w_dt has norm 0.09 (≈ 4.0 in the healthy runs);
  - W_c 0.32–0.36 (8.6–9.4);
  - w_u.0 13.1 (65–67).

**My reading:**
- F5's sigmoid saturated in fp32, and the drug path then received zero gradient.
- s reads only the drug token and the dose, not chromatin.
- The same-seed off arm had the same initialisation and batches (identical first logged loss, 2.1207), and it survived.
- So this is an optimiser failure of the shared v10 base that either arm can hit, and chromatin changed only the trajectory.

## What I did, and did not, do about T-B
- **on s0 stays in the registered reading.** §100.4 has no exclusion clause, and none is added.
- **Arithmetic, not a reading:**
  - seed 0's paired Δ is −0.166;
  - to pass, seeds 1 and 2 would need a summed Δ above +0.175, which is more than the off arm's whole contrast r;
  - **so T-B as registered cannot pass.** It is still read in full when all six runs exist.
- **Beside it, labelled POST HOC (motivated by on s0), never a pass:**
  - the same means over runs whose s is not identically zero;
  - the same s check applied to every T-B checkpoint, both arms.
- **The d384 screen** (on, seed 0, unguarded) runs as registered. If it collapses, the screen's rule picks 256.

## The guard (§100.8, T-C and the gap arm only; both arms alike)
- **The model:** it also returns s. Nothing is computed differently, and this is tested.
- **The trainer (`--guard`):** it keeps the per-batch median of s. From step 1,000, every 500 steps, if the median of the last
  200 batch medians is below 1e-4:
  - it writes `V10C_FAILED_{tag}.json`;
  - it exits with code 3, scoring nothing and writing no predictions.
- **The threshold:** healthy runs end at a median of 0.12–0.13 (minimum 0.008 over 480 rows), so 1e-4 is about 1,000 × below
  healthy. The failed run is 0 to six places.
- **The kernel:** on exit code 3 it retries the arm with seed + 10, then seed + 20.
  - A retry runs in the same kernel only if elapsed + run_h ≤ 11 h; otherwise in a new kernel.
  - The reading uses the first attempt per slot that completes. Every attempt is recorded.
  - If a slot has no completed attempt, there is no claim.
- **Logged per epoch** (reported only): the mean batch-median s and the mean γ on the epoch's last batch. This is
  RNG-neutral: eval mode, no_grad.
- **Tests:** 38 in `test_v10c_model.py` (94 across `model/v10c`) pass. Seven new tests cover:
  - the guard rule by hand, with median-not-mean distinguished;
  - stopping on a forced s = 0, with no prediction file written;
  - a healthy run finishing;
  - no stop without `--guard`;
  - `next_attempt`'s seed bumps, budget and two-retry cap;
  - the guarded kernel rendering and compiling.
- **Mutation testing:** 16 / 16 mutants killed.

## ASKS
1. **The diagnosis:** is F5 saturation the right reading? Is "not a chromatin effect" fairly worded, given that chromatin
   changes the trajectory?
2. **T-B:** is keeping on s0 in the registered reading right, and is the "cannot pass" arithmetic fairly stated? Is the post hoc
   "trained runs" view acceptable as reported-only?
3. **A replacement seed:** should a post hoc **on seed 3** (≈ 3.5 GPU-h, labelled post hoc) be run so the trained-runs view is
   balanced, 3 on against 3 off? The Kaggle quota is tight this week: off s2 and the d384 screen are still queued.
4. **The guard:**
   - Are the threshold, cadence, retry seeds and "first completed attempt" rule sound?
   - Is it symmetric and blind enough (training batches only)?
   - Would you instead or also fix F5 in the model? For example a floor s = ε + (1 − ε)σ(·), or bounded a_d and b_d. Either
     would break the tested "zero dose → zero effect" property.
5. **The d384 screen:** should it still run unguarded at seed 0, as registered?
