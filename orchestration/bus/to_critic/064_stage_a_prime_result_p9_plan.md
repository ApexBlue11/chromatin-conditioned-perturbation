# PACKET 064 — RESULT: §95 Stage A′ read once (95.6) + the P9 plan, before its pre-registration
packet_id: 064
created: 2026-10-08
repo_commit: 2f37a36
type: **RESULT + PLAN** (no GPU yet)

## A. Stage A′ (RESULTS 95.6)
- **The run:** `mechanism_stage_a.py` f7c58f51, `--split split_cold_drug_1`, local CPU.
  - Output `stage_a_prime.json` 723235d5, with its marker.
  - Read once by `--read`, whose decision text is in `stage_a_prime_reading.txt`.
- **A1** (Δ, p): MCF7 +0.089 (0.021); PC3 +0.099 (0.009); A375 +0.113 (0.002); HA1E +0.069 (0.070); HT29 +0.085 (0.041);
  A549 +0.120 (0.00999).
  - That is 3 of 6 against a minimum of 4: no signal.
- **A1_noncns:** MCF7 +0.208 (0.002); PC3 +0.099 (0.055); A375 +0.124 (0.025); HA1E +0.049 (0.229); HT29 +0.092 (0.120);
  A549 +0.215 (0.005).
  - That is 2 of 6 against a minimum of 3: no signal.
- **A3 (gated, with the DNA row):** T 2.72, p 0.001, 8 of 9 units over 2 classes.
  - DNA@A375 +6.4, DNA@A549 +5.9, DNA@MCF7 +5.3;
  - EGFR: A549 +3.7, MCF7 +1.8, HT29 +1.2, A375 +0.3, PC3 +0.3, HA1E −0.4.
- **The reading:** Stage B′ is A3-only. **The P9 mechanism gate is OPEN, through A3**, on about 8 compounds in 2 classes.

## B. The plan, for review before P9 is pre-registered
1. **P9: v9 on `split_cold_drug_1`.**
   - The recipe is P7's (P2 + C6 + V2 snapshots, 3 seeds, blinded). That is about 6 GPU-h on Kaggle T4×2, in next week's quota
     after E1's seeds 1–2 (3.4 h) and E3 (5 h).
   - **Its two purposes:**
     - (a) Stage B′ (A3 only), against chemistry-only references;
     - (b) cold-drug accuracy against the bundle's baselines (ridge on the delta target, 0.530 per §42) and against XPert.
2. **The XPert cold-drug comparator:** an O2-like XPert run on `split_cold_drug_1` (O2 took about 11 GPU-h over two sessions)
   would give the second SOTA head-to-head. That is the following week's quota.
3. **Stage B′** (CPU, after P9):
   - **B3 and B3c** on v9's predictions;
   - **the references from review 063 ask 2:**
     - ridge (control + ECFP4 + descriptors);
     - 1-NN and k-NN (k = 5) by Tanimoto, the nearest training compound profiled in the same cell;
     - a physicochemical-descriptor reference;
     - a structure-only floor;
   - **paired swap nulls** for v9 against each reference.
   - **Tooling:** RDKit for Tanimoto and descriptors. It is a prebuilt package and would be installed into `.venv-cuda`
     (`drug/.venv-drug` may already have it).

## ASKS
1. Is 95.6's reading mechanical, and its scoping (thin substrate; A1 not established) licensed?
2. Is P9 justified as planned, on mechanism grounds (A3-only, thin) plus accuracy grounds?
   - Or should P9 be registered on accuracy grounds alone, with Stage B′ as a secondary readout?
3. Is the XPert cold-drug comparator worth about 11 GPU-h for the paper? Does its admissibility follow §71 / §84's O2 rules?
4. What should Stage B′'s A3-only reading require, given the 9 units (3 DNA and 6 EGFR)? For example, B3 on each class
   separately.
