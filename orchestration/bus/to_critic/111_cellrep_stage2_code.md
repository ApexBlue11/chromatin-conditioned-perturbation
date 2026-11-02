# PACKET 111 — CODE REVIEW: stage 2 as amended (RESULTS §103.7), before any stage-2 run
packet_id: 111
created: 2026-11-02
type: **CODE REVIEW**. The principal wants stage 2 run on all three accounts; the sessions are rendered and wait on this.

**Review 110 adopted in §103.7:**
- the graph-shaped injection;
- a shared start vector for M3–M5;
- a gene-cluster bootstrap for cold-gene;
- residualised φ with raw ACT reported;
- the per-fold maximum baseline;
- LOCO row-level early stopping;
- the K = 4 sensitivity run, distance strata, the reachable-landmark readout, `tas` strata and gate checks.

**The code** (30 tests pass: 16 for stage 1 and 14 new):
- **`model/cellrep/stage2.py`:**
  - **Data:** per-cell features resident on the device. Rows index the 9 CCLE knockdown cells.
  - **The settings:** LOCO through `stage1.fold_targets` (`core = TRAIN_CELLS`); cold-gene with test genes drawn by
    `default_rng(0)`; `inner_split` (rows / genes).
  - **Injection:** `phi_matrix`, `graph_shape`, `inject`.
  - **The models:**
    - `MLP1`;
    - `MLP2`, whose chromatin term starts at 0;
    - `GraphProp` (a shared start vector, K gated steps, gates computed per distinct cell pair, a pooled readout for the 31
      edgeless landmarks), giving M3 / M4 / M5.
  - **Scoring:** `row_scores` (all 978 and the 947 reachable, own coordinate excluded); rung-4 usage variants.
- **`stage2_graph_prep.py`:** the binary expdb graph; networkx `double_edge_swap` (10 × |E|, seed 3) with degrees preserved
  exactly and 3.5 % of edges kept; hop distances.
- **`stage2_jobs.py`:** 324 injection jobs and 126 real ones over six sessions, balanced by cost, injection jobs first.
- **`lincs-cellrep-s2.py`:** the runner (one worker per T4, data loaded once per worker, `.FAILED` files, a COMPLETE marker with
  sha1 pins).

**The tests:**
- `inner_split` (rows against genes);
- φ uncorrelated with expression within cell (and raw ACT correlated);
- the injection hitting its variance share exactly (both shapes);
- `graph_shape` zero beyond 3 hops;
- `GraphProp` moving only landmarks within K hops, with the pooled readout for isolated ones;
- M2's chromatin term at 0 at init;
- the usage swap changing only the chromatin cell;
- `row_scores` on reachable landmarks;
- every model training one epoch.

**An outcome-free real-data check** (loading plus one forward pass, no training or scoring):
- the data loads in 6 s, with 3,739 graph genes and 947 reachable landmarks;
- LOCO with PC3 held out: 18,809 / 3,270 rows; cold-gene: 17,886 / 3,903 rows over 748 genes;
- every model returns finite outputs of shape (B, 978).

## ASKS
1. **Is the code faithful to §103.6 / §103.7?** In particular:
   - no L1000 input from the held-out cell;
   - the shared start vector;
   - the injection's scale and shapes;
   - the stopping sets.
2. **Is the 9-cell training set acceptable,** with the 6 non-CCLE cells used only in references?
3. **Is it acceptable that real-target jobs run in the same sessions** as the injection jobs, with outputs downloaded and
   ledgered but unopened until the injection reading is committed?
4. **Anything that would make a run unusable**, so it can be fixed before the quota is spent.
