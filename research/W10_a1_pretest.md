# TASK W10 — the A1 zero-parameter diffusion pre-test (RESULTS §82)

Precisely specified by a pre-registration; **do not redesign it.** Repo root `C:\Projects\LINCS`. Write only the two new
files named below. CPU only, numpy/scipy. No git. No GPU.

## RUN EVERY COMMAND IN THE FOREGROUND AND BLOCK ON IT
Do not background anything. Write the report last, with real output in hand.

## USE THIS PYTHON, AND ONLY THIS ONE
`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`. Never bare `python`.

## What it computes, in one paragraph
For each drug signature (compound d, cell c), spread the compound's known protein targets over a gene–gene graph by a
random walk with restart, and ask whether the result predicts which of the 978 landmark genes respond (|z|). The graph's
edges are optionally weighted by the chromatin accessibility (ATAC) of the two genes in that cell. The question is
whether the cell's OWN accessibility makes the prediction better than ANOTHER cell's. Nothing is fitted.

## Inputs (all on disk; verified by the PI)
- Graph: `network/outputs/v9/union_graph_v9.npz` — `edge_index` (2, 81846) listing each undirected edge ONCE with i < j,
  `nodes` (978,) gene symbols. Use the **binary union**: `A[i,j] = A[j,i] = 1` for every listed edge; diagonal 0.
- Gene order: `Data Info/pathway_landmark_genes.txt` (978 symbols, one per line). **Assert** `list(nodes)` equals it.
- Targets: `drug/outputs/dti/dti_reference.tsv`, columns `pert_id`, `gene_symbol`, `gene_idx` (index into the order
  above). **Assert** `nodes[gene_idx] == gene_symbol` on every row. A compound's target set = unique `gene_idx`.
- Response: `phase2_assembly/outputs/Y_target_level5_978.npy` (312438, 978) float32, same gene order.
- Signatures: `phase2_assembly/outputs/signatures_usable.tsv` — `row` indexes Y; `cell_id`, `pert_id`.
- Chromatin: `phase2_assembly/outputs/E_final.npy` (83, 978, 3) and `E_final_mask.npy` (bool), same gene order;
  channel 0 = ATAC (primary), channel 1 = H3K27ac (secondary). Cell rows: `epigenetics/outputs/epigenetics_cell_index.json`
  (`cell_id_to_row`; open with encoding `utf-8-sig`).

## 1. `model/v9/a1_diffusion_pretest.py`
Arguments: `--channel {0,1}` (default 0), `--max_cells N` (default: all; for smoke tests only), `--n_boot 20000`,
`--seed 0`, `--alpha 0.5`. Output paths must carry every argument that changes the numbers:
`model/results/a1_diffusion_pretest_ch<channel>[_cells<N>].json` and the same stem `_rows.npz`.

**Eligible cells:** channel observed (`E_final_mask[c, :, channel]`) for ≥ 90 % of the 978 genes. Sort eligible cells by
`cell_id`. With `--max_cells N`, keep the first N. For a gene unobserved in cell c, use the median of c's observed values.

**Rows:** signatures whose `cell_id` is eligible and whose `pert_id` has ≥ 1 target. Record the count per cell.

**Graph operators.** For weights `W` (978×978, symmetric, zero diagonal): `deg = W.sum(1)`; `P = D^-1/2 W D^-1/2`
(a zero-degree gene gets zero rows and columns); `K = inv(I − alpha * P)` (use `numpy.linalg.solve` against the
identity, float64). Propagated score for compound d: `s = K @ t_d`, where `t_d` has `1/|targets|` at the targets.

**Conditions**, per cell c:
- `OWN`: `W = A * sqrt(outer(a_c, a_c))`, a_c = cell c's channel vector (imputed as above).
- `MISMATCH`: the same with `a_{c'}`, for 5 cells c' ≠ c drawn **without replacement** from the eligible list by
  `numpy.random.default_rng(seed + index_of_c_in_sorted_eligible_list)`. Score = the mean of the 5 Spearmans.
  (If fewer than 6 eligible cells exist, use all others.)
- `MEAN`: `a` = the mean of the eligible cells' imputed vectors.
- `NONE`: `W = A`.
- `DEGREE`: `s` = `A.sum(1)` (the same for every compound).

**Score per row:** Spearman correlation between `s` and `abs(Y[row])` over the landmarks **that are not targets of the
row's compound** (`scipy.stats.spearmanr`, or rank-then-Pearson — identical). Group rows by compound so each compound's
exclusion mask is built once. Rows where either vector is constant over the kept genes get NaN and are counted.

**Estimand:** for each contrast X − Y in {OWN−MISMATCH (primary), OWN−MEAN, OWN−NONE, NONE−DEGREE}: per cell,
`d_c` = median over the cell's non-NaN rows of `score_X − score_Y` (paired, same row). Summary: `mean_d` = unweighted
mean of `d_c`; `ci95` from a cluster bootstrap resampling CELLS with replacement (`n_boot`, `default_rng(seed)`, one
resample matrix shared by all contrasts); `n_cells_positive`; two-sided sign-test p (`scipy.stats.binomtest`).
Also report, per condition, the median score over all rows.

**The pre-registered reading** (RESULTS §82.5) — compute it and write it into the JSON under `reading_82_5`:
1. If `median_score(NONE) <= 0.01` **or** NONE−DEGREE's `ci95` includes 0 → `"UNINFORMATIVE"`.
2. Else if OWN−MISMATCH has `mean_d > 0`, `ci95[0] > 0`, and `n_cells_positive >= 0.75 * n_cells` → `"SUPPORTS"`.
3. Else if OWN−NONE has `ci95[0] > 0` and OWN−MEAN's `ci95` includes 0 → `"GENE_LEVEL_PRIOR"`.
4. Else → `"NO_SUPPORT"`.

The npz: per-row arrays of every condition's score, the row index, and the cell id index.

## 2. `model/v9/test_a1_diffusion.py` — every check an assertion that can fail
On a small synthetic graph (e.g. 12 genes), using the functions from the script (structure the script so the operator,
the score and the bootstrap are importable functions):
- **T1** `alpha = 0` ⇒ `K = I` ⇒ `s == t` exactly.
- **T2** gating with `a` = all ones gives weights identical to `A`, so OWN's `s` equals NONE's `s` (`allclose`, 1e-12).
- **T3** `K` is symmetric (`P` symmetric) to 1e-10, and `(I − alpha P) @ K == I` to 1e-10.
- **T4** a zero-degree gene: no NaN or inf in `P` or `K`.
- **T5** target exclusion: changing `|z|` at a target gene does not change the score; changing it at a non-target does.
- **T6** the MISMATCH draws for a cell never include the cell itself, are distinct, and are identical across two calls.
- **T7** the gene-order assertion raises when the symbol list is permuted.
- **T8** the cluster bootstrap with every `d_c` equal to 0.3 gives `mean_d == 0.3` and `ci95 == [0.3, 0.3]`.
- **T9** each of the four readings is produced by a synthetic input built to produce it.

## 3. Verification you must run, in the foreground, and paste in full
```
cd C:\Projects\LINCS\model\v9
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_a1_diffusion.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_diffusion_pretest.py --help
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_diffusion_pretest.py --max_cells 3 --n_boot 200
```
The last is a smoke test only; paste its printed summary and the output file names. **Do not run the full analysis
(no `--max_cells`)** — that is mine, after I verify the code.

## Report
`C:\Projects\LINCS\research\W10_a1_pretest_REPORT.md`: the full code of both files, the full output of every command,
and `## What I was unsure about`.
