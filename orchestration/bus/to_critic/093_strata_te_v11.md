# PACKET 093 — DESIGN: T-C's stratified readings (§100.13), T-E (§101), and v11's outline for early challenge
packet_id: 093
created: 2026-10-21
type: **DESIGN REVIEW** (registrations committed bc5a34e; no T-C run exists; no T-E code exists)

## 1. §100.13 — T-C's stratified readings (reported, never a claim)
- **Why:** the principal asked whether chromatin helps specific drugs or cells rather than on average.
  - Epigenetic compounds are 2 % of T-C's test rows (470 of 21,151). A +0.05 effect on them would move the average by ≈ 0.001.
- **The first look on T-B dev** (`tb_strata.py`, post hoc):
  - nothing stood out;
  - the dev carve is too small (17 compounds with ≥ 20 rows).
- **Classes frozen before any T-C output** (`compound_classes.py` / `.json`):
  - ChEMBL mechanism text plus name lists, one class per compound, by priority;
  - the T-C row and compound counts are in §100.13.
- **The strata:**
  - per cell (7 clean, H1975 apart);
  - per class, with a compound-cluster bootstrap and a seed-SE z;
  - per compound (≥ 10 rows, 1,014 compounds), with BH q = 0.10 on z;
  - epigenetic × cell.
- **The quantity:** the trainer's saved per-row contrast r, slot-averaged per arm, Δ_row = on − off.

## 2. §101 — T-E, three CPU tests (training cells of cold_cell_1 only)
- **H-E1:** DNMT inhibitors × promoter methylation, r > 0 (251 rows, 31 cells).
- **H-E2:** BET inhibitors × broad ACT (tss_50kb), r < 0 (339 rows, 22 cells). The JQ1(−) control is reported (55 rows).
- **H-E3:** induction of silent genes × ACC prom_1kb, r > 0, specific against repression.
- **The estimand:**
  - per cell, a partial correlation across genes after regressing out basal and a leave-cell-out gene mean;
  - the mean over cells, with a cell-cluster CI at 98.33 % (Bonferroni over 3);
  - specificity against the non-epigenetic reference (paired by cell);
  - gene-permute and cell-swap nulls, and planted calibration.

## 3. v11 outline (not registered; I'm asking for early challenge before it is)
The principal's direction (2026-10-19): the full STRING graph, chromatin on all its genes, a chromatin gate, CCLE baselines,
and knowledge priors.

- **Today's state:**
  - the model uses the induced 978 × 978 STRING subgraph (13,001 edges);
  - the full graph (19,496 genes, 929,472 edges) and 128-d gene vectors for all of them exist locally;
  - of the 1,363 compounds with ChEMBL mechanism targets (546 target genes), **only 314 hit a landmark gene**. So in the
    subgraph most drug targets never touch the measured genes.
- **The module:**
  - **What each of the 19,496 nodes starts with:**
    - its gene vector;
    - the cell's chromatin at that gene (genome-wide windows, needing a ChIP-Atlas re-read);
    - the cell's CCLE expression where the line is in CCLE (needing a download);
    - the current drug's target signal.
  - **Message passing:** K rounds over the full graph, with chromatin-gated edges √(a_i a_j).
  - **Readout:** into the 978 landmark tokens.
  - **An auxiliary head:** molecule tokens → ChEMBL targets / Reactome-GO pathways, trained on training compounds only.
- **The knowledge rules I'd register:**
  1. **Independent sources only:** ChEMBL mechanisms, Reactome, GO, and STRING's experimental / database channels. Nothing
     derived from L1000 / CMap responses, whether signatures, gene sets or inferred MoA.
  2. **Unseen compounds get two settings:** "known targets as input" and "targets predicted from structure".
  3. **The same-molecule exclusion** (96.6) stays.
- **The dev-screen arms:**
  - full graph against the subgraph;
  - the gate on against off;
  - CCLE on against off;
  - known against predicted targets.
  - Every arm gets the §100.13-style strata.
- **The prior:** §83's fixed diffusion over this graph carried no drug-specific signal (NONE − RANDOM +0.0019). A trained
  propagation is a different test.

## ASKS
1. **§100.13:**
   - Is the seed-SE z with BH adequate for per-compound reporting? Would you prefer a slot-permutation null?
   - Are the class definitions acceptable as frozen?
2. **§101:**
   - Is the partial-correlation estimand with a leave-cell-out gene mean sound?
   - Are 5-row profile minimums and the > 50 % mask rule reasonable?
   - Is H-E2's sign (BET inhibitors repress broadly acetylated genes) the right a priori direction for L1000 landmarks?
3. **v11:**
   - Are the knowledge rules sufficient against leakage?
   - Is anything in STRING itself L1000-derived (the co-expression channel)? Should only experimental + database edges be
     used?
   - Which arms are essential for attribution?
