# PACKET 035 — CODE: P7's blinding (W23 + PI) and the P7 kernel, before Saturday's run
packet_id: 035
created: 2026-09-30
repo_commit: 219c99f
type: **CODE REVIEW before GPU spend (~5.7 GPU-h, after the Saturday reset).** Nothing has touched the test cells.

## What changed
- **`model/v9/xpert_arm.py` (277c143):** `--no_test_metrics` — no `Pearson*` fields in the per-seed record, no per-seed print, no
  aggregate print (a "suppressed" line instead), no `metric` key in the JSON, `candidate_flags.no_test_metrics = true`. With the flag
  off, the record and JSON keep their original key order (the worker had reordered them; PI restored). Model-free `nulls` unchanged.
- **`model/v9/align_dev.py` (277c143):** `--rows test`: no dev carve, test-row sha1 `be276e23…` asserted, the training-row prior from
  all 32 training cells, `licence()` with `min_cells` 5 (dev) / 7 (test), and `in_cell` now reports `min_cells`,
  `beats_training_prior_every_seed` and the prior (PI added; the worker computed but did not write it). The review-032 rule comment
  was deleted by the worker and restored. **Never run on a real checkpoint.**
- **Tests:** `test_p7_blinding.py` 3/3 (smoke with the flag: no `Pearson` key in `runs`, no score line in stdout; without the flag the
  per-seed line is printed; `licence()` on toys; the dev path reproduces the committed P2 in-cell JSON), `test_candidates_v9` 45/45
  (its W21 smoke no longer leaves a result JSON in `model/results`), `test_xpert_arm_dev` pass, `test_mc_infer_dev` 4/4. The worker
  ran three suites in parallel (CPU 99 % for 20 min); the PI stopped it and ran every suite alone.
- **`external/kaggle_kernels/kern_v9p7/lincs-v9p7.py` (219c99f; generator `orchestration/make_p7_kernel.py`):** the §85.12 command
  (`--sign_head_w 0.492066`; `--snapshot_cycles 3` added only if §85.11 says so) run as a **subprocess** whose stdout is captured to a
  log in `/tmp/p7stage`; any captured line matching `pearson…\d.\d` is **not echoed** and fails the kernel; after exit 0: exactly one
  new arm JSON, `no_test_metrics` recorded, no score field, GUARD 5 (distinct, every epoch), **GUARD 6** (each prediction file: 21,151
  rows, sorted sha1 `be276e23…`); only then are the staged files moved to `/kaggle/working`; `fail()` deletes the staging directory
  and the arm JSON. Guards 1–3 grep the mount for `no_test_metrics`, `snapshot_cycles`, `sign_head_w` and `dp_seeding.py`.
- **Local smoke (env `LINCS_P7_SMOKE_DIR`, dev carve, 1 seed, 1 epoch, 300 rows; never test rows):** passes end to end; the captured
  log contains **0** lines mentioning Pearson; the leak regex hits the three real score-line forms and misses the null / loss /
  "suppressed" lines. The smoke caught two Windows-only faults (an open `np.load` handle blocking the move; listing a directory),
  both fixed.

## ASKS
1. Is the blinding complete — is there any other path by which a model-based test score reaches the kernel log, the JSON, or
   `/kaggle/working` (e.g. an exception message, a warning, the arm's final JSON)?
2. Is the kernel's fail-closed staging sound on Kaggle (a `SystemExit` after a partial `shutil.move`, or a session kill mid-move)?
3. Anything in `align_dev.py --rows test` that differs from §85.12 item 8?
