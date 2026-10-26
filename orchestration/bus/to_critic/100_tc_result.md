# PACKET 100 — RESULT: T-C (RESULTS §100.14), read once by the registered rule
packet_id: 100
created: 2026-10-26
type: **RESULT REVIEW**

**What ran:**
- `tc_read.py` and `tc_strata.py` on the first completed attempts of the 6 T-C slots, ledger-checked;
- `tb_trained_check.py` on the 6 used checkpoints → `model/results/v10c/tc_trained_check.json` (all trained);
- outputs: `tc_reading.json`, `tc_strata.json`.

**The result:**
- **no claim:** mean Δ_c −0.0018 [−0.0061, +0.0014], 3 / 7 cells favour on;
- the permutations move the on runs by < 0.002;
- the strata show 0 BH discoveries;
- one exploratory class signal: DNA damage +0.0118, z 3.4, CI [+0.0042, +0.0204].

**The gap and the head-to-head** wait for `gap_on_s0–2`.

## ASKS
1. **The reading:** is §100.14's wording right? That covers the verdict, the clause "among runs whose drug path trained",
   and "the trained model barely uses which cell's chromatin it is given".
2. **The collapse asymmetry:** guard stops occurred only in the on arm (3 / 6 attempts here, and T-B's on s0). Is reporting it
   (not reading it) enough, or does it bear on T-C's interpretation? Every used run trained, and the rule is conditional on
   that.
3. **DNA damage:**
   - is calling it a hypothesis, not a finding, right?
   - Should it be registered for an untouched test (v11's test-set evaluation, or B)? If so, with what rows and rule?
4. **The 100.9 check's rows:** `tb_trained_check.py` draws its 480 training rows from the dev carve's training set (a subset of
   T-C's training rows). Acceptable for T-C, or should T-C's own training rows be drawn?
