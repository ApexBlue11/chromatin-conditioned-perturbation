# REVIEW OF PACKET 035
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 219c99f

**The blinding is complete on every path I can trace, and nothing here should hold Saturday's push.**
- **In the arm:** with `--no_test_metrics`, the only model-based test scores (the four `Pearson*` fields in `rec`,
  the per-seed print and the `mmr` aggregates) are never computed.
- **In the kernel:** it captures stdout **and** stderr (`stderr=STDOUT`), scans every line before echoing, verifies
  the arm JSON has no score field, and only moves files after GUARD 5 and GUARD 6 pass.
- **The upload:** the staged Kaggle upload matches 277c143 / 219c99f byte for byte (`xpert_arm.py`, `align_dev.py`,
  `model_v9.py`, `modules_v9.py`, `config_v9.py`, `dp_seeding.py`, `coldcell_h2h.py`).

Three MINOR hardenings suit a run that happens once: partial outputs (C1), a structural guard on `--rows test` (C2), and
pinning the mounted source by hash rather than by string (C3).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | code-vs-intent | **The final move isn't atomic, so a failure mid-move can leave a scorable partial set.** `/tmp` and `/kaggle/working` are different filesystems on Kaggle, so `shutil.move` is copy-then-unlink. An exception partway through the loop (disk, I/O) ends the kernel with, say, two of three seeds in `/kaggle/working`, all having passed every guard. An errored kernel's working directory is kept. §85.12 item 5 covers it procedurally ("deleted before scoring … disclosed"), but only if someone notices the set is incomplete. | Make incompleteness **unscorable**. Copy each file to `/kaggle/working/<name>.part`, `os.replace` it to its final name (same filesystem, atomic), and write **`P7_COMPLETE.json` last**, listing every file and its sha1. The local scoring preamble then refuses to run unless the marker exists and every `--ours` / `--ours_alt` file's sha1 matches it. That's alongside the O2 `69484323` assert already in §85.12. |
| 2 | MINOR | code-vs-intent | **Nothing in the code stops `align_dev.py --rows test` being run on a non-P7 checkpoint.** §85.12 item 8 says it runs once, after P7, and never before. But every dev checkpoint (P2, C6, C7, C8b) is local and passes `assert ck['split'] == a.split`. A run on any of them would read test-cell interpretability from a model the pre-registration never designated. That spends test-cell freshness outside the plan. It isn't an accuracy leak, since `align_dev` computes no accuracy, but the rule rests only on intention. | Add a pin, as in `read_moa_88.py`: `P7_SHA1 = [None, None, None]`, filled from the P7 kernel's printed sha1s, and `--rows test` refuses any checkpoint not in the list, or any run while a pin is `None`. |
| 3 | MINOR | provenance | **Guards 2–3 check strings in the mounted source, not its identity.** For a one-shot run, a stale or locally edited upload that still contains `no_test_metrics`, `sign_head_w` and the rest would pass. The staged copy matches the commit today, but the kernel can't know that. | Pin the sha1 of each mounted file that defines the run (`xpert_arm.py`, `model_v9.py`, `modules_v9.py`, `modules_v7.py`, `config_v9.py`, `dp_seeding.py`) to the reviewed commit's contents, computed from the repo when the kernel is generated. The kernel refuses on any mismatch, and costs seconds. |

## Answers to the asks

**Ask 1 — complete, on every path I can find.**
- **Score computations:** the arm has **three** model-based score computations on the evaluated rows, all gated:
  - the `rec.update({...Pearson...})` block, `xpert_arm.py:475-480`;
  - the per-seed print, `:513-514`;
  - the `mmr` aggregates, `:524-528`.
- **Model-free lines that stay:** the `nulls` line is copy-control and mean-drug, computed before any training
  (`:369-379`). The training-loss prints are on training batches.
- **Exceptions:** Python tracebacks print code, not variable values. NumPy warnings from a skipped `their_pearson`
  can't occur.
- **Other outputs:** the saved npz files hold predictions plus targets, and the checkpoints hold weights plus
  `cfg/split/seed/epochs`. Neither carries a score.
- **V2's per-cycle `eval_model`** returns predictions only.
- **The leak regex is a backstop,** not the protection. It would miss a score printed without the word "pearson", but
  no such print exists in the arm.
- **The arm JSON** is checked for any `pearson` key in `runs` and for `metric`, and `fail()` removes it.

**Ask 2 — sound for failures before the move. Partly sound during it.**
- **Before the move:** every `fail()` precedes the move loop, deletes `/tmp/p7stage`, and removes only the arm JSON
  this run created (the `before` set). A killed session loses `/tmp`, and by your observation `/kaggle/working` too.
- **During the move:** a `SystemExit` can't occur after the loop starts, since no `fail()` follows it. But an I/O
  exception can, and a session kill can land mid-copy. C1 makes both unscorable instead of merely procedural.

**Ask 3 — it matches §85.12 item 8.**
- **Rows and prior:** `dev_cells=0` gives `D.te` as the test rows, with the sha1 asserted (`be276e23…`), and `D.tr`
  as all 32 training cells, so the training-row prior is P7's.
- **Licence:** `licence()` applies "beats the prior on every seed" and "≥ `min_cells` cells on the 3-seed mean, with
  the mean of per-cell increments > 0 on every seed", with `min_cells` 7 for test and 5 for dev. The JSON records
  `min_cells`, the prior and both licences.
- **Separate outputs:** the file suffix `_test` keeps test and dev results apart.
- **The one gap** is C2.

## What I checked and found sound

- **`make_p7_kernel.py`** generates either the `c6` or the `c6_v2` argv. The committed kernel is the `c6` form, with
  `--snapshot_cycles 3` absent, as §85.11 requires until V2 is decided.
- **GUARD 6 globs `v9p7*_seed*.npz`,** so under V2 it also checks `_snap{k}` and `_last`, and it requires exactly
  `N_SEEDS` main files.
- **The smoke path** uses the dev carve (`--dev_cells 6`) and moves its arm JSON out of `model/results`, so it can
  neither touch test rows nor pollute the results directory.

## What I could not assess, and why

- **Kaggle's exact retention of `/kaggle/working` for errored versus cancelled sessions.** C1 makes the answer
  irrelevant.
