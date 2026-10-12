# REVIEW OF PACKET 079 (D re-ruled at its gate, 98.15)
verdict: SOUND
reviewed_commit: 4dc85ac

**The six rulings are faithful to 98.2 and to M's precedent, and every factual claim in 98.15 checks out against Bai et al.'s
code.** I fetched their code at `1d750cd098f0…`, read-only, into scratch.
- **`train_pertdit.py`** there is byte-identical to the vendored copy.
- **`bench_dataset.py`** (3,703 B) confirms three things:
  - the dose dict is indexed by each row's float;
  - `negative_ctrl` is read once in `__init__` and used only under `self.cfg`;
  - the test loader is `shuffle=False`.
- **C3's scope holds:**
  - **PRnet** applies `normalize_total` + `log1p` (`train_prnet.py:121–132`) and saves Δ in that space (383–384).
  - **The other five trainers** contain neither call, and save raw Δ.
  - **The evaluator** scores each model's saved arrays as given.
- **`p10_d_inverse_check.py`** is an independent implementation, and its numbers equal mine.

The six points below are MINOR. They are pins the worker contract needs, plus two citation fixes.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | pin (precompute script) | **"Their script requires CUDA only for speed" is true in substance, but as written it exits on CPU.** Lines 73–81 call `sys.exit` when captions remain and CUDA is absent.<br>Rulings 2–4 also need, compared with their file:<br>• compounds restricted to 1,970 (it embeds all of `idx2smi`);<br>• ten dose prompts (line 159 is one);<br>• the placeholder (lines 150–152 load upstream's archive).<br>So "their own precompute script" here is a **modified copy**. | Generate it as M+dt's trainer was: named substitutions on the vendored file, the output sha1 pinned, and the diff committed and reviewed. Keep the code changes to two (the CUDA gate and the dose loop) by doing the rest on the data side:<br>• a restricted `idx2smi` file at `IDX2SMI_PATH` (sha1);<br>• a placeholder pickle at `ORIGINAL_EMB_PATH`. |
| 2 | MINOR | pin (ruling 5, cap branch) | **Their trainer has no time stop.** The epoch loop (lines 311–385) ends only on their rule or at 1,000 epochs. The "otherwise" branch's 6 GPU-h cap therefore needs a loop-control patch, and nothing in 98.15 specifies it. It is of the same kind as O9's `BUDGET_H`: loop control, not the model. | The patch: after each epoch, stop if elapsed + t_epoch + t_test + a margin > 6 h, then fall through to their own best-checkpoint test prediction. Record `stop_reason` ∈ {rule, cap} and the stopping epoch. `score_p10`'s `pertdit` loader then refuses a **reading** when `stop_reason == 'cap'`, which enforces "not run (cost)" mechanically. |
| 3 | MINOR | blinding hygiene (ruling 5, timing kernel) | **"Samples the full test set once" produces test-row predictions** from a three-epoch model. Nothing needs them, and D's test rows can stay untouched until the run. | Time sampling on a 13,445-row block of **training** rows (same shapes, same cost), or keep everything in `/kaggle/temp` and persist only the timings. |
| 4 | MINOR | blinding parity with M | **Their trainer writes `per_drug_pcc.csv` (test PCC per drug) and `test_ground_truth.npy` into `output_dir`** (lines 442–472). M's kernels kept both in `/kaggle/temp` (075, Ask 2). | Do the same for D. Persist `test_predictions` and `test_ctrl` (the inverse needs both), `best_model.pt`, the curve and `resource_log`. Ruling 1's own-target score is then computed at scoring time from the bundle with their transform (T over the h5ad's 68,830 rows, as their code computes it). |
| 5 | MINOR | pin (Ask 6, dose keys) | **The lookup is exact.** `self.dose_emb[self.dose_list[index]]` (`bench_dataset.py:48`) indexes by the float that `adata.obs[col].values.astype(float)` produces (`train_pertdit.py:212`). The bundle's doses are float32, and float64(float32(0.1)) = 0.10000000149…, which is not `round(d, 4)`. A mismatch raises `KeyError` at the first batch, which is loud but comes after the GPU session starts. | Write the h5ad dose column as float64 of `round(d, 4)`. Build the dict keys **and** the prompt strings from that column's unique values. That gives '0.1', '0.37', '0.62', '0.77', '1.11', '2.31', '3.33', '5.0', '10.0' and '40.0'; the 10 µM prompt equals their exact string. Assert that the keys cover every row's dose before any GPU work. |
| 6 | MINOR | citation precision (C3 text, ruling 4) | **Three attributions are slightly off:**<br>(a) **The loading** of each model's arrays "as given" is in `evaluate_one_fold.py:18–19`. `evaluate_all.py` is the metric library. Its PCC_DEG (`pcc_top_k_deg`) picks the top 100 genes by \|truth\| in each model's own space, so for PertDiT and PRnet even the DEG selection is transformed.<br>(b) **MultiDCP's raw Δ** relies on its `--output_space deg` default, the unified runs. `native` writes absolute profiles.<br>(c) **The package versions** (transformers 4.39.1, tokenizers 0.15.2, sentencepiece 0.2.0) come from PertDiT's `models/PertDiT/repo/environment.yml`. The harness's `pip_freeze.txt` has none of them. | Cite them so. |

## Answers to the asks

**Ask 1 — yes, faithful. The principle is the right one.**
- **The registration's explicit word governs where it speaks.** That is what pre-registration means. "As Bai et al. ran it"
  fills the gaps where 98.2 is silent, and for M, 98.2 itself said "as published".
- **Both principles agree here anyway.** Per-dose is PertDiT's own upstream design (`dosage_prompt_emb_lincs.pkl`, a dose-keyed
  dict, read at their lines 215–219). Bai et al.'s single prompt followed from their single-dose data.
- **Ruling 1 is faithful:** the inverse uses inputs only, and the oracle bound and the own-target score are reported beside D.
- **On timing:** the rulings come after R1 and M+dt were read, but both make D runnable and **stronger** (dose information, an
  almost lossless inverse). Neither moves toward v9, and no D output exists. I see no forking-path concern.

**Ask 2 — the zero placeholder plus the pinned `cfg=False` is sufficient.**
- **Where the token is touched:** `bench_dataset.py` reads it at line 34 and uses it only under `self.cfg` (line 44).
- **Why it can't be used:** the trainer passes `cfg=False` literally in all three constructors (lines 223, 227, 231), and the
  pinned sha1 fixes that.
- **A fail-on-index guard** would mean patching their dataset class. Not needed.
- **An optional extra guard at no cost:** fill the placeholder with NaN instead of zeros. Any use would then turn the loss NaN
  and fail visibly. `pad_sequence` never sees the token.

**Ask 3 — CPU captions are acceptable, and the disclosure is enough** (with C1's pin).
- **Why disclosure suffices:** a caption is a representation produced by their model. There is no "correct" caption for a
  held-out compound, and a beam tie that resolves differently on CPU and GPU is a hardware effect, not an analytic choice.
- **Why not a GPU spot-check:** it would need MolT5's 3.13 GB weights inside a GPU kernel, against ruling 4's design, for a
  reported-only number.
- **What matters more for an outside reproducer:**
  - the caption cache's sha1;
  - the library versions and CPU thread count used;
  - C1's script pin.

**Ask 4 — the three-way rule is sound,** with C2 and C3.
- **The arithmetic, checked:**
  - **Earliest stop:** `epoch − best_epoch ≥ 10 and epoch > 60` gives epoch index 61, so 62 epochs. P(62) is right.
  - **Training batches:** 779 against the 5,000 cap.
  - **Validation:** 1,108 rows is 18 batches, against the 110 cap. So t_epoch must include each epoch's DDIM validation. It does
    if the timing loop is their loop.
- **Truncated-run reporting is within bounds** when it is labelled and the loader refuses a reading (C2).
- **The seed: keep 117 as the reading of record.**
  - **Why:** the scorers average per-seed per-row Pearsons, not predictions (`score_p9.py:424–425`; `score_p10.py:188`). So one
    seed has the same expectation as a seed mean, only more variance, and the 98.3 UNINFORMATIVE rule absorbs that.
  - **Averaging 117 and 118** would change a registered rule after R1 was read, for a variance gain only. Report 118 as the seed
    spread.

**Ask 5 — yes, within what the code shows.** I checked each trainer at `1d750cd`:
- **PRnet** transforms at lines 121–132 (no clip) and saves at 383–384.
- **The other five** have no `normalize_total` or `log1p`:
  - CIGER: Δ = `x_pert − x_ctrl` (line 148), saved at 460;
  - DeepCE: 542;
  - MultiDCP: 929, under `deg`;
  - TranSiGen: 234;
  - XPert: `y_true − ctl_true` (173).
- **The evaluator** loads them unchanged.
- The "like for like against the four other raw-Δ full-feature models" wording is right. Citations per C6.

**Ask 6 — the contract should pin C1–C5, plus these:**
- **The h5ad:** M's rows, carve and guards (a–e), with a float64 dose column (C5).
- **The embedding dict:** keyed by int `pert_idx` (their `.astype(int)`), with coverage asserted for all three splits. Record
  the caption cache's and the embeddings' sha1s.
- **The inverse:**
  - **T_train:** name the row set (all 55,385 split-train rows, or the 49,846 fit rows; either uses inputs only). Pin it and its
    value.
  - **S_ctl:** from the clipped raw control.
  - **The output:** raw Δ̂ as float32, keyed by `row_index`. The test loader is `shuffle=False` (`bench_dataset.py:91–93`), so
    the order is the test-mask order.
- **The seeds:** 117 and 118 on `cuda:0` and `cuda:1`, through their `seed_everything`.
- **The `pertdit` loader checks:**
  - the manifest and the records' input pins;
  - `n_test` 13,445;
  - seed 117 as of record;
  - `stop_reason` (C2).

## What I checked and found sound

- **Bai et al.'s repo at `1d750cd`** (commit date 2026-05-13; tree listing; files read-only in scratch):
  - `models/PertDiT/repo/src/bench_dataset.py` in full;
  - `train_pertdit.py` (byte-identical to the vendored copy);
  - `train_prnet.py`'s transform and save sections;
  - `train_{xpert,deepce,multidcp,transigen,ciger}.py` (no transform calls; their truth lines; MultiDCP's `--output_space`
    default);
  - `evaluate_one_fold.py` and `evaluate_all.py`'s PCC_DEG;
  - PertDiT's `environment.yml` and the harness's `pip_freeze.txt`.
- **HuggingFace:** `molt5-large-smiles2caption` `e307e7a1f959…` and `BioLinkBERT-large` `1eb6d81c5fc1…` are each model's
  current revision. The weights are `pytorch_model.bin`: 3.13 GB + 1.33 GB, pickle files, so recording their sha256s is the
  right guard.
- **The PI's code and text:** `p10_d_inverse_check.py` (its numbers equal mine); 98.13's C4 text; the 98.14 withdrawal; and
  98.15's arithmetic: the stopping rule, the batch counts, and the quota (8.3 + 6 + 0.5 + 6 = 20.8 of 30, leaving ≈ 9 h).

## What I could not assess, and why

- **The worker's adapter, caption and training kernels.** They don't exist yet; C1–C5 are what to pin.
- **MolT5 caption throughput on Kaggle CPU** for the bundle's unique SMILES. I made no runs.
- **PertDiT's model code** (`CrossDit`, `Sampler`). I didn't read it. None of the rulings depends on it, and it runs unmodified.
