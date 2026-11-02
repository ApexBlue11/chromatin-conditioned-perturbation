# PACKET 109 — RESULT: stage 1, read once (RESULTS §103.4)
packet_id: 109
created: 2026-11-02
type: **RESULT REVIEW**

**The run:** `model/cellrep/stage1.py` at 4e49ffa, once → `model/results/cellrep/stage1.json`. The leak test on the real index
structures gave fixed ≤ 0.0008 and leaky −1.000.

| rung | result |
|---|---|
| P0 | **passes**: median own-gene percentile 0.004 (per cell 0.000–0.010); overexpression 0.237 (reported) |
| VD (reported) | gene 0.336, cell 0.026, gene × cell 0.637; **MS_gxc / noise bound 1.17** |
| ceiling (reported) | ρ̂ 0.18 (PC3 0.13, MCF7 0.23); √ρ̂ 0.43 |
| D1 positive control | **passes**: 0.101 [0.071, 0.130], 6 / 6 cells |
| D1 (ACT:prom_1kb partial on CCLE) | **fails**: +0.001 [−0.028, +0.031], 3 / 6 cells. Rung 1 +0.054. With the L1000 control covariate −0.004 |
| D1 consensus (reported) | PC 0.079 [0.049, 0.108]; ACT −0.007 [−0.036, +0.023] |
| B1 | **passes**: median 0.429 [0.407, 0.457]; top 5 % 0.099; hub control 0.490; tas tertiles 0.474 / 0.471 / 0.343 |
| B1 matched against mismatched | **not cell-specific**: +0.0005 [−0.008, +0.009] |

## ASKS
1. **Does it reproduce from `stage1.json`** (the per-gene P0 values are saved), or by re-running the script (minutes on CPU)?
2. **Is §103.4's reading right?**
   - "a real, modest cell-specific component";
   - "chromatin's null at rung 2: the information is in the data, but expression already holds it";
   - "B1: a gene-level correspondence, mostly carried by active compounds; not cell-specific".
   - Is the VD arithmetic (σ²_gxc ≥ 0.055, about 14 % of MS_gxc) stated correctly?
3. **What follows for stage 2?** §103.4 proposes:
   - LOCO, on the cell-specific part against a ceiling of about 0.43;
   - cold-gene, where the full network predicts unseen genes (the route B1 supports);
   - chromatin tested with a positive control, not assumed.
