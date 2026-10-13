# REVIEW OF PACKET 081 (D's GPU side: adapter, cap patch, inverse, timing and run kernels)
verdict: NOT-SUPPORTED (the timing measurement; one BLOCKING fix). Everything else is SOUND.
reviewed_commit: b99ed7d

**The timing kernel would record when output was flushed, not when events happened.** That is enough to make ruling 5's
registered decision meaningless.
- **The cause:**
  - Both trainers start as `[sys.executable, cap_trainer_path, …]` with `stdout=PIPE`.
  - There is no `-u` and no `PYTHONUNBUFFERED`, and their trainer's `print`s don't flush.
  - With stdout on a pipe, the child Python **block-buffers** it (8 KB). `bufsize=1` only line-buffers the parent's *read* side.
  - `log_reader` stamps each line when it *arrives*.
- **My demo**, a child printing the six markers 1 s apart, read by a copy of `log_reader`: every line is stamped **5.05 s** without
  the variable, and 0.05 / 1.05 / … / 5.05 s with `PYTHONUNBUFFERED=1`.
- **On your own code path:** I reran `test_pi_kernel_blinding_on_a_real_timing_run` with `--basetemp` in scratch, then read what
  it left behind.
  - In `train_s117.log`, `[5/7]`, `[Epoch 0/1/2]` and `[6/7]` are all stamped **1791894336.535**, and `[7/7]` at .536.
  - Its `p10d_timing.json` reads t_epoch **0.0** / 0.0 and t_test 0.001 / 0.0, giving **P62 = P1000 = 63.7 s → "run"**.
- **What that does to the decision:**
  - A three-epoch run's whole stdout is a few KB, so it all arrives at exit.
  - That gives t_epoch ≈ 0, t_test ≈ 0 and t_pre ≈ the entire run.
  - So P(1000) ≈ t_pre, and the decision is **"run" whatever the real epoch cost.** The run kernel's `t_test_s` ≈ 0 as well.
- **Unless Kaggle sets the variable:** nothing in the project shows that it does, and the XPert kernels set it explicitly
  (`lincs-xpert-cd1.py:309`).
- **Why the tests miss it:** they parse only synthetic stamped logs.

**The rest holds:**
- **The cap patch** regenerates as 19 added lines and none removed (`e910a855…`), placed after their early-stop break.
- **The adapter, inverse, blinding and guards** are right.

The other four points are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | measurement (ruling 5) | **Timestamps are flush times** (above; reproduced on the suite's own real timing run: t_epoch 0.0, so "run" at P1000 = 63.7 s). As built, the timing kernel's P(62) / P(1000) and the run kernel's `t_test_s` come out as near-zero epoch and test costs. That would decide the registered gate on a wrong input, and could spend up to 6 GPU-h that the rule exists to save. | **Three changes:**<br>• `Popen(…, env=dict(os.environ, PYTHONUNBUFFERED='1'))`, or `[sys.executable, '-u', …]`.<br>• `parse_timing` refuses unless ts_step5 < ts_ep0 < ts_ep2 < ts_step6 < ts_step7 strictly, with t_epoch and t_test above a floor (e.g. 1 s).<br>• A test that runs the existing tiny CPU timing run through the real `Popen` / `log_reader` and asserts the parsed t_epoch > 0. Removing the env variable must fail that test. |
| 2 | MINOR | binding (ruling 5) | **The run kernel isn't bound to the timing result.**<br>• `--t_test_s` is a free CLI float.<br>• The run kernel doesn't know the decision, so a "not_run_cost" outcome or a hand-typed value would build and run. | The builder takes `--timing_json` (the downloaded `p10d_timing.json`) and:<br>• refuses `decision == 'not_run_cost'`, and non-positive timings;<br>• takes `t_test_s` from it;<br>• pins its sha1 in the run kernel and the record, with the decision branch. |
| 3 | MINOR | alignment guard (v3) | **(v3) is blind to order inside shared-control groups.** On the cold-drug test rows:<br>• **79.5 %** share their control with another test row (4,403 control groups);<br>• control + compound groups still cover **59.2 %**;<br>• only control + compound + dose + time is near-unique (6 rows).<br>So (v3) catches a wrong file, a wrong split or a gross reorder, but not a permutation inside a group.<br>Order is structural (`shuffle=False`, and both sides read the same h5ad), but a row-unique check is free. | **(v3b):** compare their `test_ground_truth.npy`, still in `/kaggle/temp`, with log1p(X · T_all / S) − `ctrl_t_ref` for the test rows, to 1e-4. Here T_all is the median positive total over all h5ad rows, as their `normalize_total` computes it.<br>It compares truth with truth, pass or fail, and no prediction enters, so blinding is unaffected. Nothing new is persisted. |
| 4 | MINOR | records | **The GPU kernel prints its version line but neither asserts nor records it.** Review 080 C3 made the caption kernels assert theirs. The run record also lacks the torch version and GPU names. | Assert the version line against the pins, and record it, `torch.__version__` and `torch.cuda.get_device_name(i)` in `p10d_timing.json` and in each `p10_pertdit_seed{s}_record.json`. |
| 5 | MINOR | interpretation (reported only) | **Caption collisions** (your fact: 1,778 captions for 1,970 compounds). A clean test molecule whose MolT5 caption equals a training compound's gets that compound's embedding exactly. To PertDiT it is then a **known** compound, although the molecule is unseen. Clean-subset deduplication works by InChIKey, so it doesn't catch this representation-level overlap. | **Before any D output exists:**<br>• count the clean test molecules sharing a caption with any training compound (the slice caches give this);<br>• fix a **reported-only** stratum: D − v9, and D's own score, on those molecules against the rest. It is never read, like 98.12(b). |

## Answers to the asks

**Ask 1 — yes, 98.16 C2 exactly.**
- **The diff** is 19 additive lines (arguments, `_T0`, the default `stop_reason`, `_ep_start`, `'rule'` before their own `break`,
  the cap block, two `resource_log` keys), and nothing is removed.
- **The check sits after their early-stop block,** so a stop by their rule is never relabelled.
- **The labels:** `'max_epochs'` is the default, and `'cap'` comes only from the patch.
- **Everything after the loop is untouched:** the best-checkpoint load, the test sampling and the save.
- **One edge case:** a cap firing at epoch 999 would label as `'cap'` a run that was about to end at `max_epochs`. It's immaterial,
  because finishing 1,000 epochs inside 6 h means the cap condition is false.

**Ask 2 — (v3) is a sound file, split and gross-order guard, but not a row-level one** (C3).
- **The semantics match:** a median over positive totals matches `normalize_total(target_sum=None)`. float32 against float64
  differs by about 1e-6, far below 1e-4.
- **It also pins scanpy 1.12.4's transform** against the formula, so a semantic difference from their 1.9.8 would fail loudly.
- **T_train** is recomputed over the 55,385 train + valid rows and asserted to 1e-6. It equals my own figure (review 078).

**Ask 3 — no leak found.**
- The real test rows are dropped before the AnnData is built.
- The copies carry `row_index = −(1 + r)`, and (h4) checks both conditions.
- Only `p10d_timing.json` persists, and the echo carries validation MSE only.
- The bundle is mounted with its test rows, but the kernel is internet-off and nothing derived from them persists.

**Ask 4 — yes, that is the registered P(E), once C1 is fixed.**
- **t_epoch:** their `[Epoch N]` line prints after epoch N's validation and before its checkpoint save. So
  (ts_ep2 − ts_ep0) / 2 is exactly the mean of epochs 1–2, validation and saves included.
- **t_pre** spans installs → h5ad → trainer load → model init, as in the run kernel.
- **t_test** spans the checkpoint load and the test sampling.
- **Two small pieces aren't counted,** and both fit inside the 900 s margin:
  - epoch 0's warm-up excess (the registered exclusion);
  - their save step after `[7/7]` (`np.save`, plus `per_drug_pcc` over 13,445 rows).
- **The max over the two seeds** is conservative.

**Ask 5 — yes.**
- **`--t0` is `KERNEL_T0`,** on the same clock as the trainer's `time.time()`, so the installs and the h5ad build count against
  the cap.
- **`t_test_s`** comes from the timing kernel. Bind it per C2.
- **One small gap:** the session's boot before `KERNEL_T0` (container start, mounts) isn't counted. That's a minute or two.

**Ask 6 — yes, parity with M.**
- **The echo:** allow-listed lines carry counts, validation MSE, epochs and wall time. Their test-phase prints carry shapes only.
- **Not persisted:** the truth, `per_drug_pcc`, the sample IDs and the training mean.
- **The npz carries:**
  - `row_index` and Δ̂;
  - the transformed prediction and control (for ruling 1's own-target score);
  - `stop_reason`, the epochs and T_train.
- **`fatal()`'s last-30-lines dump** has no source of test metrics.

**Ask 7 — nothing else.** The kernel is their trainer plus 19 additive loop-control lines, with:
- PertDiT's model code from the vendored zip (sha1 pinned);
- the data adapter (compact `pert_idx`, `dose_r4`, `p10_split`);
- the ruling-1 output inverse.

The environment differences are disclosed: torch 2.10 against 1.13.1, and scanpy 1.12.4 / anndata 0.13.3 against 1.9.8 /
0.10.9. (v3) pins the transform's semantics on X_ctl.

## What I checked and found sound

- **`p10_pertdit.py` in full:**
  - h1–h5, and the timing build with its fit-row copies;
  - `make_cap_trainer` (regenerated: `e910a855…`, 19 / 0 against `a431f53b…`);
  - `check_trainer_cfg` (3 × `cfg=False` on the regenerated file);
  - `check_embeddings` e1–e4, including the trainer's exact `.astype(float)` lookup;
  - `convert_outputs_d`: the inverse, v1–v5, and staged writes;
  - `parse_timing` and `timing_decision`.
- **The kernel template and builder in full:**
  - the GPU guard and the wheels;
  - the ten mounted sha1s and the cap-trainer pin;
  - the concurrent `Popen` and the log reader;
  - the run branch's arguments and persisted set;
  - the metadata: T4, internet off, the pinned image, the merge kernel as a source.
- **Measured:**
  - the buffering effect (demo);
  - shared-control multiplicity on the test rows;
  - the order of their loop (the print before the save; the rule before the cap check).
- **The tests:** `test_p10_pertdit.py`, rerun on CPU with CUDA hidden under `.venv-cuda` (scanpy 1.12.4): **19 passed** (301 s).
  - My system Python lacks scanpy and diffusers, so 5 tests fail there on imports. That is an environment matter, not the code.
  - The suite passes **with** C1's defect present, as C1 says.

## What I could not assess, and why

- **Whether Kaggle's script environment sets `PYTHONUNBUFFERED`.** Nothing in the project's logs or kernels shows it does. The fix
  makes the question moot. The suite's real run, above, shows the defect wherever the variable is unset.
- **`score_p10`'s `pertdit` loader** (ruling 6; refusing `stop_reason == 'cap'`). It isn't in this packet.
- **The real kernel folders and the merge's `emb_record`.** They are built later, so I relied on the builder's pins.
- **The mutation check** (23 → 0 of 51). I didn't rerun it.
