# TASK W27 — `model/v9/chromatin_funnel.py`: the RESULTS §91 chromatin funnel (closed form, CPU only)

## Read first (binding)
`model/results/RESULTS.md` section `## 91.` in full, **including the §91.8 amendments** (the pre-registration — **the estimands, nulls
and rules there are the contract; do not change any of them**), `model/v9/score_dev.py` (`DEV_SHA1`, `row_pearson`, `centred_r` — **import these, never re-implement**),
`model/v9/xpert_arm.py` lines 67–76 (`carve_dev`) and 185–212 (how `E_final` is z-scored per (cell, mark) and how cells map to rows).

## Data (all local, read-only)
`DATA = C:\Projects\LINCS\external\lightning\v9payload\data` holds `xpert_mdmt_splits.npz`, `E_final.npy`, `E_final_mask.npy`,
`lincs_cell_index.json`, `cell_lineage.npy`. `DTI = C:\Projects\LINCS\external\kaggle_moa_inputs\chembl_dti_edges.tsv` (tab-separated;
columns include `pert_id`, `gene_symbol`). The split column is `split_split_cold_cell_1` (values `train` / `test`).

## What to build — functions first, each importable and tested; `main()` only wires them
1. `load_train_rows(data_dir)` → dict of numpy arrays for **train rows only** (`X`, `C` = X_ctl, `pert`, `dose`, `time`, `cell`,
   `row_index` from the bundle's `row_index` key). **Never load test rows into any array kept** (index the npz arrays with the train
   mask immediately). Assert exactly 47,509 rows. Returns also `is_dev` for cells {HEK293T, HL60, LNCAP, SKBR3, U937, VCAP}; assert
   `hashlib.sha1(np.sort(row_index[is_dev]).astype(np.int64).tobytes()).hexdigest() == score_dev.DEV_SHA1` and 4,043 dev rows.
2. `condition_means(y, keys, cell, fit_mask, exclude_own_cell)` → per-row μ [n, 978] by the §91.2 back-off chain: key (pert, dose,
   time) → (pert, time) → pert → global, the finest level that has ≥ 1 **fit** row from a cell other than the row's own when
   `exclude_own_cell=True` (any fit cell when False). At every level: **mean over cells of within-cell means** (each cell counts
   once). Also return the back-off level used per row. Only `fit_mask` rows ever contribute.
3. `gene_features(cells, C_rows_by_cell, E, Em, cidx, failed, encoding)` → `b` [n_cells, 978] (mean X_ctl over the cell's rows,
   then z-scored across genes within the cell), `Ez` [n_cells, 978, 3] and `has_mark` [n_cells, 3]. **Availability:** a mark is
   present iff `E_final_mask.npy` has it for the cell **and** it is not a failed-ChIP H3K27me3 track — `failed` = the list
   `h3k27me3_failed_chip_downweighted` in `C:\Projects\LINCS\phase2_assembly\outputs\E_final_provenance.json` (mark index 2 only).
   **`encoding='rank_normal'` (primary):** per (cell, mark), `scipy.stats.rankdata(v, method='average')` → `norm.ppf((rank − 0.5)/n)`
   across the 978 genes. **`encoding='v9'` (reported secondary):** exactly `xpert_arm.py` 190–194 (z-score per (cell, mark)) **with
   failed tracks kept** (v9's actual input). Missing marks → 0. Cells absent from `lincs_cell_index.json` have no marks.
   With the failed tracks removed, PHH has no mark: **11 covered dev-train cells** (assert this count for the primary encoding).
   The covered set is always derived from `has_mark` for the encoding in use (under `v9` it is 12, PHH included) — never hard-coded.
4. Feature-set builders: `FB`, `FBC`, `FBC_perp` (each mark residualised on `b` across genes within the cell, by least squares),
   `FC`, and nulls `N1` and `N2`. **N1:** for every cell, **each mark the cell has** is replaced by the mean of that mark over the
   covered dev-train cells **that have that mark**; marks the cell lacks stay 0; **inside LOCO the mean excludes the held-out cell**;
   N1 is **refitted** like FBC (same features and capacity), never FBC's coefficients applied to N1 features. **N2:** each dev cell
   takes the chromatin of the next cell in HEK293T→HL60→LNCAP→SKBR3→U937→VCAP→HEK293T (training cells unchanged, so FBC's fit is
   **applied** to the swapped dev features; a mark the receiving cell's donor lacks is 0).
5. **T1** `fit_t1(...)` / `predict_t1(...)`: regressors per (row, gene) = `[f_j, μ·f_j]` for each feature j; target `y − μ^(−c)`;
   global ridge + per-drug deviation ridge (`δ_d`, zero for drugs with < 3 covered dev-train cells). **Use sufficient statistics**
   (per (cell[, drug]) per-gene sums of n, Σμ, Σμ², Σt, Σμt): the design never needs to be materialised. Penalties are
   `κ·N` with N = number of (row, gene) points in the fit; κ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1} (global) and κ_d ∈ {1e-2, 1e-1, 1, 10, ∞}
   (∞ = no deviation), chosen jointly by **leave-one-cell-out over the 11 covered dev-train cells** (held-out cell scored by mean per-row
   Pearson, its μ from the other dev-train cells; **the δ_d floor of 3 covered cells is counted within each fold, excluding the
   held-out cell**). Then refit on all covered dev-train cells and predict dev rows `ŷ = μ + X·(β+δ_d)`.
   Fit rows are covered dev-train rows only; for a row whose cell lacks a mark, that mark's `f` is 0. Also report the **additive-only
   rule** (v = δ = ε = 0, w fitted alone) and the **global rule alone** (δ = ε = 0).
6. **T2** `retrieve(...)`: for a query (dev row, or a held-out covered dev-train cell's row in LOCO), neighbours = covered dev-train
   cells (excluding the query's own cell) having the row's condition at the **finest back-off level any neighbour has**; ŷ = Σ α ȳ with
   ȳ the neighbour's within-cell mean at that level, α = softmax(s/τ). `s_C` = mean over marks present in **both** cells of Pearson
   across genes of `Ez`; **if the pair shares no mark, that neighbour's s_C = the target's median s_C over its other neighbours**
   (count such pairs). `s_B` = Pearson across genes of `b`. `s_BC = s_B + β·s_C`.
   τ ∈ {0.01, 0.03, 0.1, 0.3, 1, ∞} (∞ = uniform), β ∈ {0, 0.25, 0.5, 1, 2, 4}, chosen by LOCO over covered dev-train cells. Also a
   lineage-match weighting (`cell_lineage.npy` rows equal → weight e^(1/τ), else 1) and B0 over all 26 dev-train cells (reported).
7. **T3** `t3(...)`: `v[c,g]` = SD over the cell's rows of `y[r,g]` (needs ≥ 20 rows; report cells skipped), z-scored across genes
   within the cell; gene-local ridge `v̂ = a·v̄_g + β·f + intercept`, coefficients shared over genes and cells, `v̄_g` = mean of `v`
   over covered dev-train cells **other than the held-out one**; κ by LOCO as in T1. Per dev cell: Pearson_g(v̂, v) for FB, FBC, gene
   prior only, N1 (refitted, as specified in item 4).
8. **T0** diagnostics per §91.6.
9. **Scoring and readings:** `score_dev.row_pearson` per dev row for every predictor; means over all dev rows, per cell, top tercile
   of `‖y‖₂` (terciles over dev rows), cell-centred via `score_dev.centred_r`; paired Δ per row and per-cell mean Δ; cells with Δ > 0.
   Epigenetic stratum: dev rows whose `pert` has any ChEMBL `gene_symbol` matching
   `^(HDAC\d+|BRD[234]|EZH2|EED|SUZ12|DNMT\w*|KDM\w+|EP300|CREBBP|DOT1L|SIRT\d)$` (report its row count). Apply §91.3–91.5's
   advance rules **as amended (§91.8)** mechanically in code and write each conjunct's value and pass/fail:
   T1 = (Δ_T1 all ≥ 0.003 or top-tercile ≥ 0.006) ∧ cells(Δ_T1 > 0) ≥ 4 ∧ cells(S(FBC) − S(N1) > 0) ≥ 4 ∧ cells(Δ_centred(FBC − FB) > 0) ≥ 4;
   T2 = Δ_T2 all ≥ 0.002 ∧ S(s_C) − S(uniform) all ≥ 0.003 ∧ both positive in ≥ 4 cells ∧ cells(Δ_centred(s_BC − s_B) > 0) ≥ 4;
   T3 = cells(ρ_T3 ≥ 0.02) ≥ 4 ∧ cells(ρ(FBC) − ρ(N1) > 0) ≥ 4. Every test is run under the primary encoding (the readings) and the
   `v9` encoding (reported). Report S(B0) and S(FB) beside the constant `P2_DEV_MEAN = 0.43693`.
10. **Output:** one JSON (`--out`, default `model/results/chromatin_funnel_91.json`) with every number above, chosen
    hyper-parameters, coverage, runtime, and sha1 of every input file. `--smoke K`: keep only K drugs (deterministic: sorted pert ids,
    every 50th) for a fast check and **require `--out` outside `model/results`**.

## Guards (assertions in the code, each with a test that makes it fire)
- No test row in any array (count and split label); dev sha1; **no dev row in any fit, mean, LOCO or hyper-parameter step** (check by
  row-index set disjointness at every entry point); μ^(−c) never includes the row's own cell; `row_pearson`/`centred_r` are the
  score_dev objects (`assert chromatin_funnel.row_pearson is score_dev.row_pearson`).

## Tests — `model/v9/test_chromatin_funnel.py` (synthetic data; they must call the functions above and be able to fail)
- `condition_means`: a 3-cell, 2-drug toy with hand-checked expected values (back-off levels, each-cell-once weighting,
  own-cell exclusion).
- T1 recovers a planted rule (`y = μ(1 + v·f) + w·f + noise`) and its increment vanishes when `f` is permuted across genes.
- T2: a neighbour with identical chromatin gets the largest α; τ = ∞ equals the uniform neighbour mean.
- T3 recovers a planted `β`.
- N1 makes chromatin identical across cells **for the marks each cell has** (a cell's missing mark stays 0), and its LOCO mean
  excludes the held-out cell; N2 follows the stated derangement.
- Encoding: `rank_normal` is monotone in the raw values, bounded by `norm.ppf(1 − 0.5/978)`, and gives tied raw values equal outputs;
  a failed-ChIP H3K27me3 track is absent under the primary encoding and present under `v9`; on the real files, the primary encoding
  leaves exactly 11 covered dev-train cells (PHH uncovered), HEK293T = ATAC + H3K27ac, VCAP = H3K27ac.
- Each guard fires on a violating input (a test row, a wrong dev set, a dev row passed as a fit row).

## Rules
Edit only the two new files. **Never write into `model/results/` except the final `--out` of a full run, which you do NOT run** —
you run only `--smoke 40 --out <scratchpad>` once and the tests once. Interpreter **only**
`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` by full path; set `OMP_NUM_THREADS=2`; run
`-m pytest model/v9/test_chromatin_funnel.py -q` **once** and the smoke **once**, sequentially, from `C:\Projects\LINCS`. No other suite,
never two python processes at once. No installs, no git, no GPU, no scratch files in the repository. Report to
`C:\Users\Surya\AppData\Local\Temp\claude\C--Projects-LINCS-LINCS-project-scope-review-1b6bd9\1a2a87ef-e957-4ed6-bb10-ec2e7f9e1823\scratchpad\W27_report.md`
with both outputs pasted, the smoke's runtime, and a mandatory `## What I was unsure about` section.
