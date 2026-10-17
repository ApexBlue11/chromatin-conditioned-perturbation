# PACKET 089 — RESULT: T-A, the CPU gate (RESULTS §100.6 as registered on review 086 Ask 4; §100.6a)
packet_id: 089
created: 2026-10-17
type: **RESULT REVIEW** (CPU; it gates nothing)

**Code:** `model/v10c/ta_gate.py` (6 tests). Outputs: `model/results/v10c/ta_gate.json` (registered) and
`ta_gate_centred_posthoc.json` (post hoc, flagged before it ran, c230f6e).

**As registered** (173 cells, 60 with chromatin, λ = 100):
- the leave-cell-out R² gain is **−0.054**;
- the cell-swap null has mean −0.009 and a 95th percentile of +0.021;
- the gene-permute null has mean −0.051;
- **no pass**; the planted calibration detects β ≥ 0.2 SD only.

**My reading:**
- per-gene standardisation across cells leaves each cell's overall responsiveness level in R;
- so the within-cell-constant chromatin channels act as cell intercepts that don't transfer;
- the real gain equals the gene-permute null's.

**Post hoc** (R, basal and chromatin centred within cell):
- the gain is **+0.00001**, against a cell-swap 95th percentile of +0.00004 and a gene-permute mean of −0.00022;
- the calibration detects **β = 0.02 SD**.

**So a linear, drug-averaged, cell-specific chromatin effect on responsiveness is below 0.02 SD.**

## ASKS
1. Is the "cell intercept" explanation of the registered negative gain right? Should the registered design have centred
   within cell (my error and yours)?
2. Is the post hoc reading fairly worded: "absent at 0.02 SD" for the drug-averaged linear effect, with no claim about
   drug-conditioned effects (which L4's gate targets)?
3. Should T-B's reading carry T-A as its prior explicitly (low)?
