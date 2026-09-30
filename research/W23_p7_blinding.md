# TASK W23 — P7 blinding: `xpert_arm.py --no_test_metrics`, and `align_dev.py --rows test` (review 034)

## Read first
`orchestration/bus/adjudicated/034_review.md` (C1, C3 — binding), `model/v9/xpert_arm.py` (`main()`: the per-seed `rec` built after
training, the `print(f'  [seed {seed}] Pearson …')` line, the final summary block using `mmr('Pearson…')`, and everything after it
that writes the output JSON), `model/v9/align_dev.py` (all), `model/v9/test_candidates_v9.py` (how the arm is smoke-tested).

## Part A — `xpert_arm.py --no_test_metrics` (store_true; default off must stay byte-for-byte today's behaviour)
When set:
1. The per-seed `rec` carries **no** `Pearson`, `Pearson_deg`, `Pearson_median`, `Pearson_deg_median` keys (keep seed, seconds, n_gpu,
   dp_seed_mode, device_seeds, cuda_rng_states_equal_by_epoch) and the per-seed `print(... Pearson ...)` line is not printed.
2. The final summary block does not print any Pearson aggregate; print instead
   `test metrics suppressed (--no_test_metrics): predictions only, scored once elsewhere (RESULTS 85.12)`.
3. Nothing written to the output JSON contains a model-based score on the evaluated rows (check every field written after the
   summary; the model-free `nulls` computed before training may stay). Record `no_test_metrics: true` in `candidate_flags`.
4. Predictions and checkpoints are saved exactly as today.

## Part B — `align_dev.py --rows {dev,test}` (default `dev` = today, unchanged)
With `--rows test`:
1. Build `XPertData` **without** the dev carve (all 32 training cells; `D.te` = the 8 test cells after the unfeaturisable drop);
   assert `sha1(np.sort(D.row_index[D.te]).astype(np.int64).tobytes()) == 'be276e2385330240c2e6237e5b67eb32185d1dde'` (21,151 rows).
2. References: the training-row prior from `D.tr` (all 32 training cells) — same function as today; the LOCO prior over the 8
   test cells (reported only).
3. The in-cell increment per test cell exactly as today's `in_cell_increment` (other-cells mean readout over the other 7 cells).
4. Licence, factored into a function `licence(per_seed_cell_increments, per_seed_alignment, prior, min_cells)` returning a dict:
   `beats_prior_every_seed` (alignment_s > prior for every seed), `cells_positive` (3-seed mean per-cell increment > 0),
   `seed_means_positive` (mean over cells > 0 on every seed), and `in_cell_licensed = cells_positive >= min_cells and
   seed_means_positive`. `--rows dev` uses `min_cells = 5` (today's rule, 6 cells); `--rows test` uses `min_cells = 7` (8 cells).
   Today's dev JSON `in_cell` fields must be computed through this function with identical values.
5. The output file name gets a `_test` suffix for `--rows test`. Default n_perm nulls may be skipped with `--no_nulls` as today.

## Tests — `model/v9/test_p7_blinding.py` (IMPORT AND CALL the code; never re-implement it)
(a) An `xpert_arm.py` smoke with `--dev_cells 6 --epochs 1 --limit_train 300 --seeds 1 --batch 2 --no_test_metrics --save_pred
    <tmp>/x.npz`: return code 0; the output JSON's `runs` entries have no key containing `Pearson`; captured stdout contains no line
    matching `Pearson_deg\s+[-0-9]` or `\[seed \d+\] Pearson`; the prediction `.npz` exists with the usual keys. Write the output
    JSON to a temp dir (point the arm's WORK dir at it if needed; never leave files in `model/results/`).
(b) The same smoke without the flag prints the per-seed Pearson line (the default path unchanged).
(c) `licence(...)`: 8 toy cells with 7 positive → licensed at min_cells 7; 6 positive → not; a seed whose cell mean is ≤ 0 → not;
    `beats_prior_every_seed` False when one seed's alignment ≤ prior.
(d) The dev path: on the committed `model/results/v9_dev_align_P2_baseline_incell_aux.json` per-checkpoint increments, `licence(...,
    min_cells=5)` returns the committed `cells_positive` and `licensed` values.
**Do NOT run `align_dev.py --rows test` on any real checkpoint** — test rows are scored once, after P7 (RESULTS 85.12). Also rerun
`model/v9/test_candidates_v9.py` and `model/v9/test_xpert_arm_dev.py`. Paste all outputs.

## Rules
Edit only `xpert_arm.py` and `align_dev.py`; create `test_p7_blinding.py`. **Never delete or rewrite existing docstrings or comments.**
Interpreter **only** `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` by full path (`-m pytest` for pytest). Run test files one
at a time, sequentially. No installs, no git, no GPU runs beyond the smokes, **no scratch files anywhere in the repository** (use a
temp dir), no writes into `model/results/`. Report anything ambiguous and what you chose.
