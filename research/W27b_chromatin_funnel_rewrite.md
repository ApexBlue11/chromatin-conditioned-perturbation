# TASK W27b — rewrite `model/v9/chromatin_funnel.py` as a vectorized library that meets the §91 contract (W27's version fails it)

## Read first (binding)
`model/results/RESULTS.md` `## 91.` in full **including §91.8 and §91.9** (the contract; change nothing in it);
`research/W27_chromatin_funnel.md` (the original brief — every requirement there still holds unless replaced below);
the current `model/v9/chromatin_funnel.py` and `model/v9/test_chromatin_funnel.py` (W27's — reuse what is correct: the T1 normal
equations in `build_H_B` and the profiled hierarchical ridge in `solve_t1` are mathematically right; `gene_features` and
`construct_features` are right). `model/v9/score_dev.py` (`DEV_SHA1`, `row_pearson`, `centred_r` — import, never re-implement).

## The PI's review of W27 — every item must be fixed
1. **Fit set.** T1 and T3 fits must use **only covered dev-train cells** (the 11 under the primary encoding, 12 under `v9`), for
   **every** feature set including FB. W27 fitted on all 26 dev-train cells (uncovered cells entered with chromatin = 0).
   μ, however, is computed from **all 26 dev-train cells** (the drug's mean response in other cells), as §91.2 says.
2. **LOCO leak in T1.** Inside a fold that holds out covered dev-train cell h, the targets `y − μ` of the other cells must use a μ
   that excludes **both** the row's own cell **and** h. Compute μ once per fold (12 μ's: 11 folds + the final fit).
3. **LOCO leaks in T3.** Inside a fold, `v̄_g` is the mean over the fit cells **other than h**, and N1's mark means exclude h
   (§91.8 item 4).
4. **T2 tuning.** Each similarity variant gets **its own** LOCO choice: τ for `s_B` alone, τ for `s_C` alone, (τ, β) for `s_BC`,
   τ for lineage-match. W27 reused `s_BC`'s τ for the comparators, which biases Δ_T2 toward `s_BC`.
5. **Missing pieces (all required in the output):**
   - **T1:** S(B0); FB; FBC; FBC⊥; FC; N1; N2; the additive-only rule (regressors `f` only, κ_d = ∞); the global rule (κ_d = ∞).
     Δ_T1 on all dev rows, on the top tercile, per cell, and cell-centred; the epigenetic-drug stratum and the rest. Both
     encodings: `rank_normal` gives the readings, `v9` is reported for FB, FBC and N1.
   - **T2:** s_B, s_C, s_BC, uniform and lineage-match over the 11 neighbours, plus B0 over all 26 dev-train cells.
   - **T3:** prior, FB, FBC and N1 per dev cell.
   - **T0** (§91.6) and **M1** (§91.9; below).
   - **M3** positive controls: FB − B0 per cell, s_B − uniform per cell, and T3's FB − prior per cell.
   - **M4:** S(FC) − S(B0) and S(FBC) − S(FB).
   - Every advance rule's conjuncts, each with its value and pass/fail, plus the instrument-check flags.
   - Chosen hyper-parameters, coverage (dev rows by back-off level, drugs with ≥ 3 covered cells), input sha1s and runtime.
6. **Dead and duplicate code.** Delete the stub `retrieve_t2` (the first definition), the shadowed `evaluate_t2`s and the
   unused `fit_t1_loco`. Keep one definition of each function.
7. **Imports.** `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` then `import score_dev`, as the other
   `model/v9` scripts do. **Assert** `row_pearson is score_dev.row_pearson`.
8. **Guards.** Assert dev-row disjointness at every fit entry point (`fit_t1`, `fit_t3`, T2 neighbour means, μ pools): no dev row
   index may be in a fit or pool set. Each guard needs a test that makes it fire.

## Performance — required, because §91.9 M2 reruns the whole pipeline ~225 times
- **Condition means, vectorized** (`condition_means_vec(y, level_keys, cell, pool_mask, exclude_cell_of_row, extra_exclude_cell)`):
  - per level, compute per-(key, cell) mean matrices and per-key sums and counts by group sums (numpy / pandas groupby on integer codes);
  - leave-own-cell-out is `(S_k − M_kc)/(n_k − 1)`;
  - the fold exclusion of cell h removes h's groups from `S_k` and `n_k`;
  - back off where the other-cell count is 0.
  No per-row Python loop. The test compares it with a slow reference on a toy.
- **Sufficient statistics** per (cell, drug) via group sums. Build `H_d`/`B_d` **once per fold**, never inside the κ loops; for the
  per-group quadratic forms use `einsum`.
- **Batched ridge:** solve all drugs at once with `np.linalg.solve` on stacked (D, 2J, 2J) arrays.
- **Vectorized LOCO and dev predictions:** for a cell, `ŷ = μ + F_c·Wᵀ + μ ⊙ (F_c·Vᵀ)`, with W and V the per-row (θ + δ_d) halves.
  No per-row Python loops anywhere.
- **Benchmark on synthetic data of the real size** (47,509 × 978, 26 cells, 1,849 drugs, random features): report the seconds for
  one complete T1 fit (LOCO over 11 folds × 25 κ pairs + final fit), one T2 tuning and one T3 fit. Target: T1 ≤ 30 s.

## M1 — ceilings (§91.9), dev-train cells only
- **(a) Dose-neighbour agreement:** for each (cell, pert, time) with ≥ 2 doses, take adjacent dose pairs (sorted by dose). Report
  the per-pair Pearson across genes of `e = y − μ^(−c)` (μ over dev-train cells excluding the row's own cell): mean, median, per
  cell, and n pairs.
- **(b) Within-cell drug-neighbour predictability:** for each dev-train row r in cell c, take the K = 10 rows of the **same cell,
  with a different pert**, whose μ^(−c) profiles have the highest Pearson with r's. Then `ê_r` is the mean of their `e`, scored as
  per-row Pearson(`ê_r`, `e_r`). Report the mean per cell and overall. Compute in chunks; no full R × R matrix for the largest cell
  if that exceeds ~1 GB.

## API the power module (W28, next) will call — keep these signatures
```
prepare(data_dir) -> Ctx          # rows, keys, cells, μ pools, features for both encodings, masks
run_t1(ctx, y, F_by_cell, fit_cells, variant) -> dict   # variant in {'full','global','additive'}; returns dev predictions, κ, κ_d
run_t2(ctx, y, sim_variant, ...) -> dict
run_t3(ctx, y, F_by_cell, fit_cells, null_F_by_cell=None) -> dict
readings(ctx, y, results) -> dict  # every conjunct and the advance flags
```
`y` is a parameter everywhere (the power module passes planted copies). μ and every sufficient statistic are recomputed from the
`y` given.

## Tests — `model/v9/test_chromatin_funnel.py` (extend; keep W27's tests that remain valid; synthetic data only)
- `condition_means_vec` equals a slow reference on a toy, with back-off, own-cell exclusion and fold exclusion.
- T1 recovers a planted rule; its increment vanishes when `f` is permuted across genes.
- With planted P3 (a drug-independent `α·f`), FBC beats FB on the raw score but **not** on the centred score.
- T2 tuning: each variant's τ is chosen by its own LOCO (the variants can be shown to pick different τ on a constructed toy).
- T3 LOCO: `v̄_g` excludes the held-out cell (a toy where including it would change the prediction).
- Each guard fires on a violating input (a dev row passed to a fit or pool).
- The vectorized predictions equal W27's per-row `predict_t1` on a toy.

## Rules
Edit only `model/v9/chromatin_funnel.py` and `model/v9/test_chromatin_funnel.py`. **Do NOT run anything on the real data**:
no `main()` and no smoke on real files. The PI runs the real funnel only after the critic reviews the code. You run only
`-m pytest model/v9/test_chromatin_funnel.py -q` once and the synthetic benchmark once, sequentially, with
`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` by full path and `OMP_NUM_THREADS=2`. No other suite, never two python processes
at once. No installs, no git, no GPU. **No scratch files or folders anywhere in the repository** (W27 left `scratch/`; it has been
deleted — use the scratchpad below). Report to
`C:\Users\Surya\AppData\Local\Temp\claude\C--Projects-LINCS-LINCS-project-scope-review-1b6bd9\1a2a87ef-e957-4ed6-bb10-ec2e7f9e1823\scratchpad\W27b_report.md`
with the test output, the benchmark timings, and a mandatory `## What I was unsure about` section.
