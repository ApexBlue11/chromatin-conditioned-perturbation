# PACKET 106 — DESIGN REVIEW: the cell-representation programme, stage 1 (RESULTS §103)
packet_id: 106
created: 2026-11-01
type: **DESIGN REVIEW** (before any code or any genetic signature value is read)

**Context:**
- **The principal's direction** (IDEAS §E, 2026-11-01): choose a cell representation on targets that test whether a model
  understands the cell, before any drug-response run.
- **Every idea climbs a diagnostic ladder,** and each null is assigned to its first failed rung (§E9):
  data → beyond the baseline → positive control → usage → gain.
- **The principal asked explicitly** that the critic point out mistakes in the experimental design.

**The data** (outcome-free, §103):
- L1000 genetic perturbations from the GEO GSE92742 Level 5 GCTX, extracted by a Kaggle kernel.
- **Integrity:** 50 trt_cp signatures equal our `Y_target_level5_978.npy` rows exactly.
- **The core set:** `trt_sh.cgs` at 96 h in PC3, MCF7, A375, A549, HEPG2, HT29 — 20,696 signatures, 3,927 genes, 2,100 in all six.

**What §103 registers:**
- **The harness:**
  - target y(g, c); cell-specific part d = y − the gene's mean over the other core cells;
  - per-row r on d (primary) and on y;
  - splits: LOCO, cold-gene, both;
  - baselines: gene mean, nearest cell by CCLE, linear ridge with and without chromatin;
  - a noise ceiling from 96 h against 120 / 144 h pairs;
  - the drug-test-cell rule: MCF7 and HT29 are transductive in any drug evaluation.
- **P0:** on-target knockdown visible (median own-gene percentile ≤ 0.10).
- **D1:** chromatin and a gene's cell-specific response spread.
  - Centred by landmark, partial Spearman with ACT:prom_1kb controlling for CCLE expression.
  - Its positive control: expression itself must associate first.
  - Landmark-cluster bootstrap; ≥ 5 of 6 cells.
- **B1:** does a drug's response resemble its target's knockdown in the same cell?
  - Training cells and training compounds only.
  - Rank percentile among all knockdowns in the cell; median < 0.45, drug-cluster CI.
  - Reported: cell-matched against mismatched, and a degree-matched hub control.

## ASKS
1. **The harness.** Is d (the deviation from the gene's mean over the other core cells) the right cell-specific target? With 6
   core cells, is LOCO adequate, or should the non-core knockdown cells enter LOCO with chromatin masked?
2. **Are the baselines strong enough?** Is any obvious one missing, e.g. a cell-mean offset estimated from vector controls?
3. **P0's threshold and D1's design:**
   - is SD over knockdowns the right "capacity to move"?
   - is the centring correct for an unseen-cell question?
   - is the positive control the right gate, given §91 T3's failure?
   - is landmark-cluster bootstrapping right when only 6 cells vary?
4. **B1:**
   - is the training-cells / training-compounds restriction enough to keep every test row unread?
   - is averaging a drug over doses and times right, or should dose and time be matched to the knockdown's 96 h?
   - is the threshold of 0.45 sensible, given CMap's published connectivity?
5. **What is missing** that the principal's "find out why it works or doesn't" requires at stage 1, before candidate models?
