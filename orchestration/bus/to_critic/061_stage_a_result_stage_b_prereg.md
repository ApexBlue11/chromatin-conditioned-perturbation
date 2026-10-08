# PACKET 061 — RESULT (§94 Stage A, 94.8) + PRE-REGISTRATION AND CODE (§94.9 Stage B), before any prediction is read
packet_id: 061
created: 2026-10-08
repo_commit: ce75800
type: **RESULT + PRE-REGISTRATION + CODE REVIEW**

## A. Stage A, read once (RESULTS 94.8)
- **The run:** `mechanism_stage_a.py` (review 060 fixes in), local CPU, 241 s, on P7's 21,151 rows. Only the `row_index` key of
  `v9p7_seed0.npz` was read.
- **Two disclosed events:**
  - the first attempt stopped at the 21,151 guard (the split's test set has 21,321), computing nothing; the rows were then
    restricted to §94.2's set (a1d2372);
  - the first `--read` crashed printing "μ" to cp1252, before any output; it was rerun with UTF-8 output.
- **Output:** `model/results/mechanism94/stage_a_result.json` d7f3856a, with its marker and `stage_a_reading.txt`.
- **A1, per cell** (A1 / null mean / p / compounds scored):

  | cell | A1 | null mean | p | scored |
  |---|---|---|---|---|
  | MCF7 | 0.653 | 0.500 | 0.001 | 400 |
  | HT29 | 0.598 | 0.500 | 0.001 | 288 |
  | MDAMB231 | 0.646 | 0.498 | 0.001 | 48 |
  | HS578T | 0.671 | 0.502 | 0.001 | 31 |
  | THP1 | 0.665 | 0.499 | 0.022 | 17 |

  4 of 5 cells pass. The self-retrieval ceilings are 0.989–1.000.
- **A3 (gated):** T 3.92, p 0.001, 15 of 17 units with d > 0. Ungated: T 2.81, 16 of 20.
- **The reader's output:** BOTH, so Stage B tests both.

## B. Stage B pre-registration (RESULTS 94.9, 7dfdd16, before any prediction value is read)
- **The predictions:**
  - v9: P7's three seeds (`deg_pred`, the three-snapshot averages) and their mean;
  - μ: `mean_drug_delta_pred` from `baselines_split_cold_cell_1.npz`;
  - measured: the ceiling.
- **The readouts** use Stage A's code with the predicted Δ: B1 (A1), B3 (A3), and B3c (within-class-centred Spearman of
  measured against predicted d, with a within-class permutation null).
- **The readings:**
  - 1, expresses mechanism: B1 and B3 by Stage A's rules;
  - 2a, beyond μ in retrieval: v9 − μ > 0 in ≥ 4 of 5 cells, mean ≥ 0.02;
  - 2b, beyond μ in cell-specific pathway magnitude: B3c > 0 with p < 0.05, and > μ's.
  - Each must hold on the seed-mean **and** every seed.

## C. Code (`mechanism_stage_a.py` 079b17ba, PI glue)
- `--delta 'a.npz[,b.npz]:key'`, aligned by `row_index` and asserted to cover Stage A's rows.
- `b3c()`. A constant within-class prediction gives ρ = 0, since Spearman is undefined there.
- `read_stage_b()` over marker-verified result files.
- **Tests:** 22 pass. The new Stage B tests catch 5 of 5 mutants (seed-mean only, no centring, 2b without μ, 2a's mean bar,
  alignment).
- **The planned run:** 6 local CPU runs (3 seeds, the seed-mean, μ, ridge) at about 4 min each, then `--read_b` once.

## ASKS
1. Is Stage A's reading mechanical and its 94.8 wording licensed? That includes "the earlier nulls cannot be blamed on mechanism
   being absent at these readouts".
2. Is Stage B's design sound?
   - Is μ the right reference, and is B3c's within-class centring the right way to isolate cell-specific magnitude?
   - Are the bars sensible: 2a's 4 of 5 with a 0.02 mean; 2b's p < 0.05 and > μ?
   - Is "seed-mean and every seed" the right robustness rule?
3. Any defect in the Stage B code? Cleared to run?
