# REVIEW OF PACKET 082 (review 081 adopted; the real timing kernel; D's scorer; the caption-collision stratum)
verdict: SOUND
reviewed_commit: 2b819b3

**C1's fix is complete, and nothing blocks pushing the timing kernel at the reset.**
- **The code:** the trainers run with `PYTHONUNBUFFERED=1`, and `parse_timing` now refuses buffered stamps.
- **The kernel file:** I re-rendered the timing kernel from the downloaded `emb_record` with my own call to `build_kernels`. The
  script and metadata are **byte-identical** to the committed files, and no run kernel is built without a timing JSON.
- **The pins:** all 11 equal the staged `lincs-p10d-src` files and the merge's outputs (drug embeddings `eba75fdc`, dose
  embeddings `c42bce8e`).
- **The merged embeddings check out:**
  - the keys are 0..1969 plus `negative_ctrl` (all NaN, (2, 1024));
  - everything is float32 and finite;
  - all ten dose keys are present;
  - compounds that share a caption have identical tensors.
- **The collision stratum reproduces** from my own clean mask and molecule labels (review 071's): **46 / 346 molecules, 1,577
  rows**, the identical row set, and no molecule with mixed rows.
- **Tests:** `test_p10_pertdit.py` + `test_score_p10.py`, rerun under `.venv-cuda` with CUDA hidden: **112 passed** (22 + 90,
  304 s).

The four points below are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | binding (C2 / d5) | **The timing JSON is bound by its decision, not by what was timed.**<br>• Neither the builder nor (d5) checks that `p10d_timing.json`'s `verified_sha1s` equal the run's pins.<br>• So a timing measured with different code, data or embeddings (an earlier build, say) would still bind the run. | In `build_kernels(timing_json_path=…)` and in (d5), require `timing['verified_sha1s'] == PIN_D`, the run kernel's own 11 pins. |
| 2 | MINOR | provenance (d2) | **The records now carry `versions`, `torch_version` and `gpu_names`, but the scorer doesn't check them.** | In `check_d_provenance`, require `versions == PINNED_GPU_VERSIONS` and two non-P100 GPU names, for both seeds' records and the timing JSON. |
| 3 | MINOR | wording (ruling 1 block) | **The "own-target" block is P9's per-row Pearson over 978 genes, on PertDiT's target.**<br>• Bai et al.'s Fig. 2a figure is PCC_DEG: the top 100 genes by \|truth\| per sample (`evaluate_all.pcc_top_k_deg`).<br>• So "as Bai et al. score it" (98.15 ruling 1) describes the **target**, not the **metric**. | Label it *"per-row Pearson on PertDiT's own (transformed) target"*. If comparability with their figure is wanted, add their vendored `pcc_top_k_deg` on the same arrays as a second reported number. Never read. |
| 4 | MINOR | wording (98.20) | **"192 share a caption … with another compound" undercounts.** 1,970 − 1,778 = 192 is the *surplus*. The compounds that share a caption with at least one other are **326, in 134 groups** (checked on the merged cache and embeddings). | *"1,778 distinct captions for 1,970 compounds: 326 compounds fall in 134 shared-caption groups, and so share their drug embedding."* |

## Answers to the asks

**Ask 1 — yes, C1 is complete.**
- **Stderr:** it is merged into the same pipe (`stderr=STDOUT`) and is unbuffered under the variable too. Only stdout markers are
  parsed anyway.
- **Partial lines:** `readline()` waits for the newline, so a line is stamped when it completes. With an unbuffered child, that
  is when it is printed.
- **The guard:** strictly increasing markers plus a 1 s floor would catch any remaining batching at the timing kernel's scale. T4
  epochs and the test pass run in minutes.
- **The mutant test** on the real tiny run (t_epoch > 1 s) is the right check.

**Ask 2 — nearly.** Re-deriving the decision in the builder and again in (d5), with `not_run_cost` building nothing, ties the run
to the *decision*. Challenge 1 here adds the tie to *what was timed*.

**Ask 3 — no objection to (v3b).**
- **T_all** is the median positive total over the h5ad's 68,830 rows, which is exactly the AnnData their `normalize_total` sees.
- **The tolerance:** float32 against float64 differs by about 1e-6, far inside 1e-4.

**Ask 4 — mostly yes.**
- **(d1)–(d6)** are the right set. Add challenge 1's timing pins and challenge 2's version and GPU checks.
- **The cap rule is the 98.16 C2 enforcement:**
  - `pertdit` with `'cap'` is fatal without `--d_truncated_reported`;
  - with the flag, `_relabel` overwrites every `LABEL` (including `of_record`'s), so `read_p10_d` refuses the JSON;
  - s118 is refused by `_load_marked`'s `ref` check.
- **The own-target transform is theirs:**
  - clip, then the median positive total over all rows, then log1p, separately for X and X_ctl;
  - zero rows are left at zero, as scanpy does;
  - the rows are the same 68,830 as their AnnData.
  The metric wording is challenge 3.
- **The reader is the registered "same rule"** (98.3):
  - `_decide` re-derives the verdict and requires it to equal the stored one;
  - UNINFORMATIVE takes precedence;
  - the two directional strings are `verdict()`'s own;
  - everything else is "no compound-level claim".

**Ask 5 — yes, within bounds as reported only.** Exact string equality against **any** training compound's caption is the
operational definition, because that is exactly when PertDiT receives an identical embedding. Training compounds have 1,448
distinct captions among 1,578. The stratum is fixed before any D output, with its file and row sha1s pinned in the scorer.

**Ask 6 — nothing blocks the timing push.**
- **Challenges 1 and 2** are scorer and builder additions for the *run* kernel. They can land before that kernel is built.
- **The timing kernel:**
  - matches its re-render;
  - passes every guard up to launch in your dry run;
  - needs two non-P100 GPUs (T4×2 shape);
  - persists only `p10d_timing.json`.

## What I checked and found sound

- **The diffs:**
  - `p10_pertdit.py`: (v3b), `MIN_TIMING_S`, the order check;
  - `make_p10_d_kernels.py`: unbuffered `Popen`, the version assert, the timing binding, the record fields;
  - `score_p10.py` in full for D: `check_d_provenance`, `load_ref` for D, the cap rule and `_relabel`, `d_own_target_truth`,
    the collision parts, `read_p10_d` with `_decide` and `_load_marked`.
- **Re-rendered** the timing kernel (identical); **sha1s** of the staged source and of the merge outputs; **`emb_record`**'s
  contents (every verified sha1, the slice-record sha1s, torch 2.10.0+cu128, the CPU model).
- **Reproduced:**
  - the collision stratum, independently;
  - that the merged cache equals the union of the five slice caches (1,970 entries, 1,778 distinct captions);
  - the embedding guards, and identical tensors within caption groups.
- **`p10_d_caption_strata.py`:** it reads `row_index` only from P9, with no predictions.

## What I could not assess, and why

- **The Kaggle runs**, including `lincs-p10d-verline`'s environment beyond its log line, and your local dry run of the kernel
  prefix. I didn't repeat the dry run; I relied on the re-render matching.
- **The mutation checks** (0 of 9; 26 of 27). I didn't rerun them.
