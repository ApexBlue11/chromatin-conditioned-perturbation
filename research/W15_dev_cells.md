# TASK W15 — A dev-cell mode for `model/v9/xpert_arm.py` that can never score the test cells

## Context
`model/v9/xpert_arm.py` trains our model ("v9") on a split of XPert's benchmark and scores it on that split's test
rows. For `split_cold_cell_1`, the test rows are 8 held-out cell lines that a pre-registered comparison will use ONCE.
Model development must choose designs on OTHER cells. Read `xpert_arm.py` fully first, and packet
`orchestration/bus/to_critic/019_v9_dev_protocol.md` (P1).

## Contract (amended by RESULTS 85.2 rules 1-2 -- read `### 85.2` in `model/results/RESULTS.md`)
The cold-cell split is in the MDMT bundle: always pass `--bundle xpert_mdmt_splits.npz --split split_cold_cell_1`.

1. Add `--dev_cells K` (int, default 0 = today's behaviour, byte-for-byte) and `--dev_seed S` (int, default 0).
2. Put the selection in a pure, importable function
   `carve_dev(rows_per_cell, K, seed, min_rows=200, max_rows=2000) -> sorted list of K cell ids`, where
   `rows_per_cell` maps each TRAINING cell id to its number of training rows. The eligible pool is the sorted list of
   cells with `min_rows <= rows <= max_rows`; selection = `np.random.RandomState(seed).choice(len(pool), K,
   replace=False)` applied to that sorted pool. Raise if the pool has fewer than K cells. Add CLI flags
   `--dev_min_rows` (200) and `--dev_max_rows` (2000).
3. When K > 0:
   - the training set becomes the training rows whose cell is NOT in the dev cells; the evaluation set becomes the
     training rows whose cell IS in the dev cells;
   - **the test rows are never passed to the model**: drop them from the dataset object before training, and add an
     assertion right before every evaluation/prediction call that no evaluated row_index is a test row_index;
   - print one line `DEV CELLS <json>` with: the eligible pool (ids and row counts), the dev cell ids, rows per dev cell, total dev rows, total remaining
     training rows, and the sha1 of the sorted dev `row_index` array;
   - **every step fitted from data uses the NEW training rows only** -- the quantiser (`fit_bins` or whatever fits
     the binned expression encoder), any normalisation, and the per-cell control means -- and the script ASSERTS
     that no dev-cell or test-cell row is among the rows fed to each fit;
   - the per-cell control mean (`x_cell`) must be computed from the NEW training rows only (as today for test cells:
     cells without training rows fall back to their own rows' controls; controls are inputs, not targets);
   - every output file name (result JSON, `--save_pred`, `--save_ckpt`) gets a `_dev{K}s{S}` tag inserted before the
     extension so dev outputs can never overwrite test-mode outputs; the result JSON gets a `"dev"` block with the
     `DEV CELLS` record and `"mode": "dev"`.
4. K = 0 must be exactly today's behaviour and file names.

## Tests — new file `model/v9/test_xpert_arm_dev.py`
- `carve_dev`: deterministic for a seed; only pool members (200-2000 rows) are ever chosen; different seeds give
  different sets on a 12-cell pool; returns K distinct members; K = 0 returns []; a pool smaller than K raises.
- A **smoke run** of the real script on the real data with `--bundle xpert_mdmt_splits.npz --split split_cold_cell_1 --dev_cells 6 --epochs 1
  --limit_train 300 --seeds 1` (it must finish in a few minutes on this laptop; if `--limit_train` does not keep it
  small, say so and stop), then assert from its JSON: `mode == "dev"`, 6 dev cells, dev rows > 0, no dev cell is a
  test cell of `split_cold_cell_1`, and the `DEV CELLS` line appeared in stdout.
- Show that K = 0 still writes today's file name (a dry check of the name-building code is enough — no second run).

## Rules
- Edit only `model/v9/xpert_arm.py`; create `model/v9/test_xpert_arm_dev.py`. Do not modify data files, results in
  `model/results/` from earlier runs, or any other code.
- Interpreter: `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`. Do not install packages.
- The smoke run is the only training allowed, and it must be tiny (1 epoch, limited rows). No other GPU use.
- Delete the smoke run's output files when the test finishes (they are tagged `_dev6s0`), and say which you deleted.
- Paste the full test output in your final message.
