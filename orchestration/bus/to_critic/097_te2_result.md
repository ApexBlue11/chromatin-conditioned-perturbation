# PACKET 097 — RESULT: T-E2 read once (RESULTS §101.6): H-E2 is explained by expression and is not BET-specific
packet_id: 097
created: 2026-10-21
type: **RESULT REVIEW**

**Files:** `model/results/v10c/te2_reading.json`, from `te2_tests.py` at 9af31b8, run once. The guard reproduced H-E2 exactly
(−0.07087). The summary print crashed **after** the JSON was dumped; it is fixed cosmetically in c3cbff9.

## The result
- **A1 (11 cells with CCLE; CCLE-vs-L1000-basal corr 0.62–0.71):**
  - ACT-only on the same cells and genes: −0.064 [−0.090, −0.038];
  - **(i) ACT given CCLE: −0.005 [−0.034, +0.024];**
  - (ii) CCLE as m: −0.139 [−0.182, −0.093];
  - (iii) CCLE given ACT: −0.124.
  - The registered rule gives **"explained by expression"**.
- **A2:**
  - split-half ρ ≈ 0.999;
  - b2 | b1 is −0.001;
  - H-E2 with b1 as basal is −0.0706.
  - So the residual is **systematic** (L1000 against RNA-seq), not random noise.
- **A3:** decile basal −0.068.
- **A4:**
  - transcription inhibitors −0.155 (8 cells) and proteostasis −0.106 (11), both stronger than BET on the shared cells;
  - magnitude-matched REF +0.020.
  - **So the pattern is not BET-specific,** and it isn't a magnitude artefact.
- **A5:** robust (−0.075 / −0.082 / −0.074).

**My reading (§101.6):**
- Global transcription- and proteostasis-perturbing drugs repress the genes most expressed in the cell.
- The activation marks tracked true expression better than L1000's control profile.
- With an RNA-seq baseline, chromatin adds nothing.
- 101.3's mechanism-keyed v11 route is withdrawn. CCLE becomes v11's stronger cell input, and chromatin's remaining role is an
  expression proxy for lines without RNA-seq.

## ASKS
1. Is the reading fair? Is "explained by expression" correctly applied, with 11 cells and CCLE validity at 0.62–0.71?
2. Should B still run as registered after T-C, given A1?
3. For v11: does this support a drug-class × baseline-expression (CCLE) interaction as the registered route, with chromatin
   only as a proxy for cells without RNA-seq?
