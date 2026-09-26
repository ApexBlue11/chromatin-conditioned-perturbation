# TASK W21 — RESULTS §90: V1 (MC averaging at inference) script + V2 (`--snapshot_cycles`) flag in xpert_arm.py

## Read first
`model/results/RESULTS.md` section `## 90.` (binding). `model/v9/xpert_arm.py` (`main()`: how `XPertData` is built for the dev
carve, the model construction, the LR `LambdaLR`, the eval loop after training, `--save_pred` / `--save_ckpt` formats).
`model/v9/model_v9.py`, `model/v9/modules_v9.py` and `model/v7/modules_v7.py` (`StochasticDepth`). `model/v9/score_dev.py`
(what the saved `.npz` must contain). `model/v9/test_probe_moa_88.py` (how a tiny LincsV9 is built for tests).

## Part A — new file `model/v9/mc_infer_dev.py`
1. CLI: `--ckpt` (an xpert_arm dev checkpoint, e.g. `external/kaggle_out/v9dev_c8b/v9dev_c8b_dev6s0_seed0.pt`), `--arm
   {det,full,drop}`, `--K 8`, `--out PATH.npz`, `--bundle xpert_mdmt_splits.npz`, `--split split_cold_cell_1`,
   `--dev_cells 6`, `--dev_seed 0`, `--identity_check SAVED.npz` (optional), `--limit N` (tests only: first N dev rows).
2. Data: build `XPertData` **exactly as `xpert_arm.main()` does for the dev carve** (reuse its code by import; factor a small
   helper inside `xpert_arm.py` only if you must, keeping its behaviour identical). Assert `ck['split'] == --split` and that
   `sha1(np.sort(D.row_index[D.te]).astype(np.int64).tobytes())` equals `51e7e4ab8b9c3c3709d43da7fa4a8c80b77d5980` (skip
   the sha1 assert only when `--limit` is given).
3. Model: `V9Config` from `ck['cfg']` (keep only dataclass fields), `LincsV9(cfg, M, ppi, gv)` with the same M / ppi / gene
   vectors files as xpert_arm, `load_state_dict(ck['model'], strict=True)`.
4. Helpers (module level, so tests can import them):
   - `set_mc_mode(model, arm)`: `model.eval()`; then for `det` nothing; for `drop` `.train()` on every `nn.Dropout` only;
     for `full` `.train()` on every `nn.Dropout` and every `StochasticDepth`. Return a dict of how many modules of each type
     were switched.
   - `predict_rows(model, D, idx, dev)`: xpert_arm's eval loop verbatim (batches of 64, `torch.no_grad()`, **no autocast**),
     returning `(y_pred, deg_pred)` as float32 numpy.
   - `mc_predict(model, D, idx, dev, arm, K, seed)`: `set_mc_mode`; for `det` one pass; otherwise for k in 0..K−1
     `torch.manual_seed(1000 * seed + k)` (and `torch.cuda.manual_seed_all` of the same) then one `predict_rows` pass; return
     the mean over passes. `seed = ck['seed']`.
5. Output: `.npz` with exactly xpert_arm's `--save_pred` keys (`y_pred, deg_pred, y_true, ctl_true, row_index`) and a `.json`
   sidecar (arm, K, seed, switched-module counts, ckpt sha1, n rows, wall seconds).
6. `--identity_check SAVED.npz` (det arm): assert per-row Pearson(recomputed `deg_pred`, saved `deg_pred`) ≥ 0.9999 on every
   row and that the dev per-row mean delta Pearson (`score_dev.row_pearson` vs `y_true − ctl_true`) matches the saved one to
   within 5e-5; print both numbers; SystemExit on failure.

## Part B — V2 flag `--snapshot_cycles N` in `xpert_arm.py` (default 1 = today, bitwise)
1. Factor today's LR lambda into a module-level `wsd_factor(s, steps)` (identical arithmetic: warm = max(1, int(0.03·steps));
   `s / warm` if `s < warm` else `max(0, 1 − sqrt(max(0, s − 0.8·steps) / max(1, 0.2·steps)))`).
2. With N > 1: assert `epochs % N == 0`; `L = (epochs // N) * (len(D.tr) // batch)`; LR factor at step s = `wsd_factor(s % L, L)`.
   With N = 1 the schedule must be exactly today's.
3. Factor the post-training eval loop into a function used both at the end and at each cycle end. With N > 1, after the last
   epoch of each cycle: `model.eval()`, predict the dev/test rows, keep the snapshot, then `model.train()`.
4. With N > 1 and `--save_pred`: save each snapshot as `<pred>_snap{k}.npz` (same keys), the last snapshot also as
   `<pred>_last.npz`, and the main `<pred>` file = the **mean of the N snapshots'** `y_pred` / `deg_pred`. The reported metrics
   use the averaged prediction. Output-name tag `_snap{N}`; record `snapshot_cycles` in `candidate_flags`.

## Tests — `model/v9/test_mc_infer_dev.py` and additions to `model/v9/test_candidates_v9.py`. Tests must IMPORT AND CALL the
code under test — never re-implement it inside the test.
(a) `set_mc_mode` on a tiny LincsV9: `det` switches nothing; `drop` puts only `nn.Dropout` in train mode and every
    `StochasticDepth` in eval; `full` both; the top-level model stays `training == False`.
(b) With a tiny model (dropout 0.1): `mc_predict(..., 'drop', K=8, seed=0)` twice gives identical output; it differs from `det`.
(c) With a tiny model with dropout 0.0 and stochastic depth 0.5: `drop` equals `det` exactly; `full` differs.
(d) `wsd_factor` with N = 1 equals today's lambda for every s in range(100) (bitwise); with N = 3 on steps 120 the factor at s
    equals `wsd_factor(s % 40, 40)` for every s.
(e) One real identity check (tiny, allowed): `mc_infer_dev.py --ckpt external/kaggle_out/v9dev_c8b/v9dev_c8b_dev6s0_seed0.pt
    --arm det --limit 128 --identity_check external/kaggle_out/v9dev_c8b/v9dev_c8b_dev6s0_seed0.npz` (compare the first 128
    dev rows) must pass.
(f) A smoke run: `xpert_arm.py ... --dev_cells 6 --epochs 3 --snapshot_cycles 3 --limit_train 300 --seeds 1 --batch 2
    --save_pred <tmp>/x.npz` writes three `_snap` files, a `_last` file, and a main file whose `deg_pred` equals the mean of the
    three snapshots (write outputs to a temp dir, never into `model/results/`).
Also rerun `model/v9/test_candidates_v9.py`, `model/v9/test_xpert_arm_dev.py`, `model/v9/test_c7_v9.py`. Paste all outputs.

## Rules (read twice)
- Interpreter: **only** `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`, by full path, for every command. **Never** the system
  Python, never bare `python` or `pytest` (`pytest` is installed in `.venv-cuda`: use `...\python.exe -m pytest`).
- Run test files **one at a time, sequentially**. Never start a second test process while one is running; never re-launch a
  suite in the background. This is a laptop: no long CPU jobs.
- Edit only `xpert_arm.py` and `test_candidates_v9.py`; create `mc_infer_dev.py` and `test_mc_infer_dev.py`. No installs, no
  training beyond the smoke in (f), no GPU runs beyond (e) and (f), no git, no scratch files in the repository root.
- Report anything in RESULTS §90 that is ambiguous and what you chose.
