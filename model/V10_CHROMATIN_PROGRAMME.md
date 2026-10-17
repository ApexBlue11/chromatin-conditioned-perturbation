# v10-C: the chromatin programme — every lever at once, judged once

**Opened:** 2026-10-17, at the principal's direction: *"do whatever it takes for chromatin to work … however far you have
to go in terms of architecture overhaul … and try to close the gap between unseen compounds and unseen cells"*.
**Logged as:** RESULTS §100, which registers this design before any of its code or data. **Critic:** packet 086.

**The rule that makes a positive result count:** every lever below is built, then judged ONCE against an identical
model without chromatin, at 3 seeds, by a rule fixed here. No lever is tuned after its result is seen. If chromatin
still does not contribute with all of them in place, that is a strong negative and goes in the paper as such.

## 1. Why chromatin failed, and the lever for each cause

| cause (evidence) | lever |
|---|---|
| **Too few cells** to learn a transferable chromatin rule: 32 training cells on the unseen-cell split, 18 with chromatin (§99 audit; chromatin acts as a cell ID) | **L1, the full XPert release:** 336,852 rows, 217 cells, 8,416 compounds. Minus the 8 unseen-cell test lines: **209 training cells, 35 with chromatin, 273,486 rows** (measured 2026-10-17, identity only) |
| **Coverage:** 17 of the 40 benchmark cells have no chromatin; the 2026-07 substitute search rescued 4 of 121 slots | **L2, coverage:** a fresh ChIP-Atlas search over all 217 lines (ATAC, H3K27ac, H3K27me3 and the same-sign substitutes of the 2026-07 menu); plus **DNA methylation** (CCLE RRBS promoter methylation) as a fourth, repression axis: **137 of the 217 lines are DepMap/CCLE lines** (by name, 109 of them without our chromatin) |
| **Representation:** peak calls only; H3K27me3 forced into narrow peaks; tie-break artefacts (§93.16); failed tracks read at full weight (T4b) | **L3, quantitative signal:** bigWig coverage per gene in several windows (promoter ±1 kb; ±10 kb; gene body; an upstream enhancer window), read by HTTP range queries on Kaggle CPU (bandwidth was the 2026-07 blocker); broad H3K27me3 domains captured by coverage; per-(cell, mark) normalisation; failed tracks dropped; an explicit missingness channel |
| **Design:** in v9 chromatin can only shift a gene's identity or add a fixed per-gene offset — it cannot change how a DRUG moves a gene (M6) | **L4, a chromatin stream that interacts with the drug** (§2) |
| **Measurement:** the standard target is plate-dominated and registers only a quarter of v9's real compound use (C3: +0.070 on the contrast vs +0.017 on the standard target); chromatin's role is to modify drug effects | **L5, the plate-contrast target** in training (F1) and as the primary score |
| **The unseen-cell lead's input** cannot be attributed without a retrain (C2b, §99.8) | **L6, an inductive cell baseline:** x_cell = the mean of the cell's controls on OTHER plates (leave-own-plate-out), identical in training and test, so x_cell ≠ x_ctl is always in distribution; XPert is given the same input in the head-to-head |

## 2. The architecture (v10-C)

Built on the v10 base (F4–F7, F9 of `V10_AUDIT_AND_PLAN.md`) plus:

- **A separate chromatin stream.** Per gene, c_g = ChromEnc(features_g, missingness_g): a small MLP over the L3 window × mark
  features (+ methylation), its own 256-d stream, **not summed into the gene identity** (the identity keeps the STRING
  vector and the learned embedding; the principal's "keep them separate").
- **Chromatin-conditioned drug attention.** In each perturb block the gene's query to the drug tokens is
  q_g = W_q h_g + W_c c_g, so which drug features a gene reads depends on its chromatin state.
- **A chromatin gate on the drug's effect.** The cross-attention update to gene g is multiplied by
  γ_g = σ(MLP([c_g ; drug_global ; dose])), "can this drug move this gene in this cell", initialised at 1 (no-op).
- **Chromatin-gated STRING propagation.** Edge weights w_ij = σ(f(c_i, c_j)) on the 978-gene STRING graph, so signal
  spreads only between genes whose chromatin permits it (C7, rebuilt on L2/L3 data).
- **Width:** d_model 256 as default; 384 screened once (5.8× more rows may lift the capacity limit).

Every chromatin component has a matched "off" switch that removes the information, not the machinery (the §45
ablation convention: training-mean chromatin and mask), so the comparison of record changes exactly one thing.

## 3. Tests, in order (each registered in full in §100 before its code)

1. **T-A, a CPU gate before any GPU** (closed form, training rows of the full release, the plate-contrast target):
   does L2/L3 chromatin predict which genes a drug moves in a cell beyond the cell's expression? Calibrated with
   planted effects as §91.9 did. **If T-A finds nothing, L4's GPU screens still run** (the principal's direction), but the
   prior expectation is recorded.
2. **T-B, the dev carve** (6 cells inside the training cells, 3 seeds, the §85 rules on the contrast target):
   v10-C against v10-C with chromatin off. One arm per lever group (L4 components together, then the width).
3. **T-C, the unseen-cell test** (`split_cold_cell_1`, 3 seeds, the cluster rule):
   - **chromatin's contribution:** v10-C vs chromatin-off, per cell, ≥ 6 of 8 cells and a cluster CI excluding 0;
   - **the gap:** unseen-cell vs unseen-compound score, both on the contrast target;
   - **the head-to-head:** v10-C vs XPert under the L6 input.
4. **Mechanism check (reported):** the learned gate's correlation with accessibility, by sign, per mark.

## 4. Cost and order

- **This week (free CPU):** L1 substrate (local, no download); L2 / L3 data kernels (need the downloads in §5); T-A;
  v10-C code + tests + mutation; packets to the critic.
- **GPU:** the full release is ~5.8× the rows, so about 1 h per epoch on 2 × T4; 4 epochs ≈ 4 GPU-h per seed per arm.
  T-B ≈ 2 arms × 3 seeds × 4 h ≈ 24 GPU-h (about one Kaggle week); T-C ≈ 24 GPU-h. **About three weeks of free quota**
  end to end, alongside O9's remaining sessions.

## 5. Downloads this needs (principal's permission requested 2026-10-17; run inside Kaggle CPU kernels)

- ChIP-Atlas experiment metadata (`experimentList.tab`, chip-atlas.dbcls.jp, a few hundred MB) — to search all 217 lines.
- ChIP-Atlas / ENCODE bigWig signal for the selected tracks — read by gene window over HTTP (a few MB per track, not whole files).
- CCLE RRBS promoter methylation (DepMap portal, `CCLE_RRBS_TSS1kb_20181022`, tens of MB).
- `pyBigWig` from PyPI inside those kernels.
