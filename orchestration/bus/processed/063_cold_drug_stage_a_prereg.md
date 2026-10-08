# PACKET 063 — PRE-REGISTRATION + CODE: §95 Stage A′ (measured mechanism on the cold-drug split's test rows), before its run
packet_id: 063
created: 2026-10-08
repo_commit: 7000c59
type: **PRE-REGISTRATION + CODE REVIEW** (local CPU; measured data only)

## A. Why (RESULTS §95.1, 40ba31b)
§94 showed v9's cold-cell predictions express mechanism only as the compound's training-cell average μ does, and every
compound there was seen in training. The test of mechanism from chemistry is the cold-drug split, where no test compound is in
training and μ doesn't exist. Stage A′ first measures whether these test rows' **measured** responses carry the readouts, as
review 059 C5 asked.

## B. The design (§95.2)
- **Rows:** `split_split_cold_drug_1 == 'test'`, 13,445 rows, 396 compounds, 40 cells.
- **Everything else** follows §94 / §94.7 / §94.10.
- **Scored cells,** from identity counts: MCF7 56, PC3 54, A375 50, HA1E 43, HT29 40, A549 39.
- **A1:** ≥ 4 of 6 cells.
- **A3 gates:**
  - RAF in HT29 and A375 (BRAF-V600);
  - MDM2 in MCF7 and A549 (TP53-wild-type). HA1E is excluded, since its SV40 large T inactivates p53;
  - ER in MCF7;
  - AR dropped (PC3 is AR-negative).
- **§95.3 sketches** P9 (v9 on cold-drug, GPU) and Stage B′ with chemistry-only references (ridge and a structural nearest
  neighbour). Each is registered later.

## C. Code (`mechanism_stage_a.py` 2bf65909, PI glue)
- **`SPLITS`** holds the per-split constants: scored cells, row count, A1 minimum and gates. Cold-cell, the default, is §94's
  unchanged.
- **The gates:** an `ACTIVE` split dict sets the gated MAPK, p53 and ER rows' gates. `run_pipeline(split=…)` sets it.
- **The rest:** `load_rows(split=…)` and the reader's threshold follow the result's `split`.
- **Tests:** 25 pass. They include the cold-drug gates, a check that the cold-cell gates are restored, and the 4-of-6 threshold.

## ASKS
1. Are the scored cells, the gates and the 4-of-6 A1 rule right for these cells? Is anything missing from the genotype
   gating, for example A549's KRAS G12S, or PC3 and HA1E?
2. **The design of the later stages:** is a chemistry-only reference set of ridge (ECFP + descriptors) plus a Tanimoto
   nearest-neighbour training compound the right "beyond structure" bar for Stage B′? What else should it include?
3. Any defect in the split refactor (cold-cell behaviour must be unchanged)? Cleared to run Stage A′ once?
