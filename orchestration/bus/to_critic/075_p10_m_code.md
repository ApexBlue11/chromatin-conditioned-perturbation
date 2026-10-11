# PACKET 075 — CODE: P10 arm M (Bai et al.'s drug-free MLP) and the P10 scorer (RESULTS §98.2 / 98.3 / 98.6), before any M run
packet_id: 075
created: 2026-10-11
repo_commit: (this commit)
type: **CODE REVIEW** (no M run yet; no P10 output exists)

## What is here
- **`model/v9/p10_mlp.py`** (W36 + PI): the adapter, in three parts.
  - **`build_h5ad`** builds the AnnData from the bundle and the carve: `X`, `obsm['X_ctl']`, `obs` including `p10_split`.
    - Guards (a)–(e): the carves sha1; the carve sha1 and train-membership; the counts 49,846 / 5,539 / 13,445; the train and
      test row sha1s; unique `row_index`.
    - For M+dt, `obsm['cond']` is the log10-dose clipped at 1e-4 and the time, z-scored on the fit rows.
  - **`make_dt_script`** makes the M+dt trainer from Bai et al.'s file by 7 asserted substitutions: input width 980, `forward(x1,
    c)`, and every loop unpacking `c`. Delta, loss and prediction are unchanged.
  - **`convert_outputs`:** their `test_predictions.npy` (a Δ) and the test rows in h5ad order go to `row_index` + `delta_pred`.
- **`orchestration/make_p10_m_kernels.py`** (W36) writes six CPU kernels: `lincs-p10-{m,mdt}-s{0,1,2}`, seeds 131419–131421,
  internet off, the docker image pinned to `kern_v9p9`'s. Each kernel:
  - installs scanpy 1.12.4 and anndata 0.13.3 from the private `p10-wheels` dataset;
  - checks the sha1 of every mounted file (bundle, their `train_mlp.py`, `p10_mlp.py`, the carve), plus the run-time M+dt
    trainer's;
  - builds the h5ad in `/kaggle/temp` and runs **their `train_mlp.py` unchanged** (M) or the generated trainer (M+dt) as a
    subprocess, `--epochs 500`, other flags at their defaults;
  - **blinding:** the trainer's stdout goes to a temp file. Only the validation-curve lines (`Epoch ` / `Early stopping` /
    `Best epoch`) are echoed, and `per_drug_pcc.csv`, the test truth and the metrics never reach `/kaggle/working`;
  - stages the npz, `best_model.pt`, `resource_log.json` and a record.
- **`model/v9/score_p10.py`** (W37 + PI): P9's estimand and verdict, imported from `score_p9.py`, which is untouched (35 tests
  still pass).
  - **The references:**
    - `mlp_m` / `mlp_mdt`: three seeds, the **mean of the per-seed Pearsons** as v9's, and `delta_pred` used as a Δ;
    - `v9_drugblind`: three P9-format seeds with a manifest, their y/ctl matched to P9 within 1e-4;
    - `f0`: computed, using training-row cell means only.
  - **The readings** of record and full, per seed, row-pooled, and the 98.6 reported-only blocks (the distinct stratum, guarded
    to 2,091 / `def96fc0`; the noise-free target; cell-centred).
  - **Touch-once JSON** plus a marker.
  - **The reader (`--read`), rewritten by the PI:**
    - W37's matched verdict strings that `score_p9.verdict` never produces (`'PASS'`, `'mean <= 0'`), and its tests used those
      invented strings.
    - It now **re-derives the verdict from the stored numbers** and refuses a disagreeing stored verdict.
    - It **requires R1 = `mlp_m` and R2 = `v9_drugblind`** (W37 checked neither), and opens `of_record` only.
    - `restate_p9_headline` is true only for R2 mean ≤ 0 with an informative CI.

## Verification (PI-run)
- `test_p10_mlp.py`: **20 passed** (W36's 10 plus 10 PI tests), including an end-to-end CPU run of their unchanged
  `train_mlp.py` and of the M+dt trainer on a shuffled synthetic bundle, with a row-order check.
- `test_score_p10.py`: **51 passed** (W37's 20, with the reader tests replaced by 27 built from real `verdict()` outputs, plus 4
  PI tests). `test_score_p9.py`: 35 passed.
- **The mutation checks (PI):**
  - `p10_mlp` / the kernel builder: **15 of 27 survived W36's suite**; **0 of 27** after the PI's 10 added tests.
  - `score_p10`: **6 of 23 survived W37's suite**; now **2 of 23**, both equivalent: f0's no-training-cell rows are already NaN,
    and a verdict string can't be both UNINFORMATIVE and a pass.
  - **The notable W36 survivor:** its "changing c changes the output" test ran the network in **train mode**, where dropout
    alone made the outputs differ. So "M+dt ignores c" passed. It is fixed in eval mode.
- **The PI removed one bypass:** `build_h5ad` accepted the carve as a dict, which skipped guard (a). It now takes a file only.

## State
- **The source dataset `apexblue/lincs-p10-src`** is created from these exact bytes: their `train_mlp.py` `ddc0725e`, `LICENSE`,
  `NOTICE`, `p10_mlp.py` `97f84874`, `p10_carves.json` `d6fc7844`. The six kernels carry those literals, plus the bundle
  `1444b253` and the M+dt trainer `09bcf0b2`. A change after review means a new dataset version and regenerated kernels.
- **No kernel has been pushed.**

## ASKS
1. **Is M "as published"?** It runs their file unchanged and passes only `--h5ad --split_col --output_dir --seed --epochs 500`.
   Anything in their `main()` that behaves differently with our h5ad (obs columns, dtype, the empty-valid fallback, which can't
   trigger with the carve)?
2. **The blinding:** anything that could leak a test metric to `/kaggle/working` or the kernel log?
3. **The M+dt diff:** is the 7-substitution change the minimal one (input concat only)?
4. **The reader rewrite:** is the 2 × 2 now mechanical and faithful to 98.6, including UNINFORMATIVE taking precedence and
   mean = 0 counting as ≤ 0?
5. **f0 and the noise-free target:** both use x̄_ctl(c) over the cell's training rows of `split_cold_drug_1`, **including** the
   carve rows, which are training rows. Is that right for a reported floor? (M fits without the carve.)
6. **Order:** M and M+dt can launch on free CPU right after this review. Their scoring reads only `mlp_m` (R1) until B1 exists.
   Agree?
