# TASK W11 — full-graph target propagation, the prerequisite for A1 (RESULTS §83)

Precisely specified by a pre-registration; **do not redesign it.** Repo root `C:\Projects\LINCS`. Write only the two new
files named below. CPU only, numpy/scipy (use `scipy.sparse`). No git. No GPU.

## RUN EVERY COMMAND IN THE FOREGROUND AND BLOCK ON IT. USE ONLY `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`.

## What it computes
For each drug signature, spread the compound's annotated mechanism targets over the full human STRING protein graph by
a random walk with restart, read the result at the 978 landmark genes, and ask whether it predicts which landmarks
respond (|z|) better than the same number of RANDOM targets would. Nothing is fitted.

## Inputs (all on disk)
- Graph: `network/outputs/v9/string_graph_v9.npz`: `nodes` (19496,) symbols, `edge_index` (2, 929472) int32, `weight`
  (929472,) float32, `landmark_idx` (978,) int32 = node index of each landmark in canonical order.
  **Check whether each undirected edge appears once or twice** (count pairs with `i > j` whose reverse is also present)
  and build a symmetric `scipy.sparse.csr_matrix` W with each undirected edge weighted ONCE per direction (no doubling).
  Zero diagonal. Print the check.
- Canonical landmark order: `Data Info/pathway_landmark_genes.txt`. **Assert** `nodes[landmark_idx]` equals it.
- Targets: `drug/outputs/dti/chembl_dti_edges.tsv` (tab-separated; columns `pert_id`, `gene_symbol`,
  `direct_interaction`). Keep rows with `direct_interaction == 1` and a non-empty `gene_symbol`; map symbols to node
  indices by exact match; **count and report** unmapped symbols; a compound's target set = its unique mapped nodes.
- Response: `phase2_assembly/outputs/Y_target_level5_978.npy` (312438, 978), canonical landmark order. Load with
  `mmap_mode='r'`.
- Signatures: `phase2_assembly/outputs/signatures_usable.tsv` (`row` indexes Y; `cell_id`, `pert_id`).

## 1. `model/v9/a1_fullgraph_pretest.py`
Arguments `--n_boot 20000`, `--seed 0`, `--alpha 0.5`, `--max_compounds N` (smoke tests only). Output
`model/results/a1_fullgraph_pretest[_cmp<N>].json` and the same stem `_rows.npz`.

- `P = D^-1/2 W D^-1/2` as sparse (zero-degree nodes: zero rows/columns). Propagate a BLOCK of seed vectors at once
  (dense matrix T, 19496 × k): iterate `S ← alpha * (P @ S) + T` from `S = T` until `max |ΔS|` summed per column
  (L1 per column) is below 1e-10 for every column; cap at 500 iterations and **raise** if not converged.
- Conditions per compound: **NONE** (its own targets, `1/|T|` each); **RANDOM** — 5 draws of `|T|` distinct nodes
  uniformly from all 19,496 nodes with `numpy.random.default_rng(seed + compound_index)` (compound_index = position
  in the sorted list of eligible compounds), each propagated; **DEGREE** — `s` = the weighted degree `W.sum(1)`.
- Score per signature row: Spearman between `s[landmark_idx]` and `abs(Y[row])`, over the landmarks that are **not**
  targets of the compound (a target node that is a landmark is excluded). RANDOM's score = the mean of its 5 draws'
  Spearmans (each draw excludes the compound's REAL landmark targets, same mask as NONE). Constant vectors → NaN,
  counted.
- Rows: signatures whose `pert_id` has ≥ 1 mapped target.
- Estimand: per cell with ≥ 50 non-NaN rows, `d_c` = median of `score_NONE − score_RANDOM` (primary) and of
  `score_NONE − score_DEGREE` (secondary). Mean of `d_c`, cluster bootstrap over cells (`default_rng(seed)`, one
  resample matrix for both contrasts), `n_cells_positive`, two-sided sign test. Also median score per condition.
- Reading (RESULTS §83.4), into the JSON as `reading_83_4`:
  1. `median(NONE) > 0.01` and NONE−RANDOM `mean_d > 0`, `ci95[0] > 0`, `n_cells_positive >= 0.75 * n_cells` →
     `"SIGNAL"`
  2. else if NONE−RANDOM passes those three but `median(NONE) <= 0.01` → `"SPECIFIC_BUT_NEGLIGIBLE"`
  3. else → `"NO_SIGNAL"`
- Record: unmapped symbol count, compounds with targets, rows, cells used, iterations to converge (max over blocks),
  and the edge-listing check.

## 2. `model/v9/test_a1_fullgraph.py` — assertions that can fail
On a small synthetic sparse graph: (T1) alpha = 0 gives `S == T`; (T2) the iteration matches the dense solve
`inv(I − alpha P) @ T` to 1e-8; (T3) an edge list given in both directions yields the same W as given once; (T4) random
draws never repeat within a draw, are identical across calls, and have the target count; (T5) target exclusion — changing
|z| at a landmark target does not change the score; (T6) each of the three readings is produced by a synthetic input
built for it; (T7) non-convergence raises.

## 3. Run, in the foreground, and paste in full
```
cd C:\Projects\LINCS\model\v9
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_a1_fullgraph.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_fullgraph_pretest.py --help
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_fullgraph_pretest.py --max_compounds 20 --n_boot 200
```
The last is a smoke test only. **Do not run the full analysis.**

## Report
`C:\Projects\LINCS\research\W11_fullgraph_REPORT.md`: the full code, every command's output, and
`## What I was unsure about`.
