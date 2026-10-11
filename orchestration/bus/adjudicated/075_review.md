# REVIEW OF PACKET 075 (P10 arm M code: adapter, six kernels, `score_p10.py`)
verdict: SOUND
reviewed_commit: a93ef5a

**The code is faithful to 98.2 / 98.6, and M runs Bai et al.'s file unchanged.**
- **The tests:** `test_p10_mlp.py`, `test_score_p10.py` and `test_score_p9.py` rerun: **106 passed** (20 + 51 + 35).
- **Every pinned literal equals the bytes on disk:**
  - `train_mlp.py` `ddc0725e`, `p10_mlp.py` `97f84874` (both LF, equal to their git blobs);
  - the carves `d6fc7844` (CRLF, as you disclosed);
  - the bundle `1444b253`.
- **The M+dt trainer regenerates** to `09bcf0b2`. Its diff from their file is exactly the 7 hunks: input width 978 + 2, `forward`
  concatenating `c`, the dataset's `c`, and the three loops' unpacking. That's the minimal input-only change.
- **The six kernels differ only** in `ARM`, `SEED`, `K` and the docstring.

The four points below are MINOR, and none blocks the push.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | blinding guard (Ask 2) | **The one way a test metric could reach the kernel log is their empty-valid fallback.**<br>• `train_mlp.py` sets `val_ds = test_ds` when the `'valid'` mask is empty. Its `Epoch … val_deg=` lines, which the kernel echoes, would then be **test** PCC, and the checkpoint would be test-selected.<br>• Guard (c) checks 5,539 `'valid'` rows **in the adapter before writing**. The trainer re-reads the h5ad and filters again, and nothing checks what the trainer itself saw.<br>• The risk is low (the same column, round-tripped through h5ad), but the guard costs one line. | Before echoing anything, the kernel reads `trainer.log` and refuses unless it contains `  valid: 5539 samples` and `  test: 13445 samples`, **and** does not contain `No valid set found`. This is a kernel change only; the dataset needn't change. |
| 2 | MINOR | provenance (scorer) | **`score_p10` checks M's files only for `arm` and `seed`.** It doesn't check:<br>• that each `p10_{arm}_seed{K}_record.json`'s verified input sha1s equal the pinned literals (bundle, `train_mlp`, `p10_mlp`, carves, and for `mdt` the dt script);<br>• that the npz files are the ones downloaded (no manifest analogue of `P9_COMPLETE.json`);<br>• that the run used the carve (C1).<br>The output's `inputs` also records only the reference directory, not the reference files' sha1s. | At download, write a small manifest (the sha1 of every npz and record). Have `load_ref` check the manifest, the records' sha1 literals, and `n_test == 13445`. Record the per-seed npz sha1s in `inputs.ref`. |
| 3 | MINOR | portability | **`p10_carves.json` is pinned by its CRLF on-disk bytes** (7,654 CRLF), while git stores LF. A fresh clone on Linux, including an outside reproducer, fails `CARVES_SHA1` in both `p10_mlp` and `score_p10`. It fails loudly, but it fails. | Either add a `.gitattributes` entry (`model/results/p10/p10_carves.json -text`, after re-committing the CRLF bytes), or pin the canonical content (the sha1 of the LF-normalised bytes) in both places. Do it before any output exists. |
| 4 | MINOR | reading wording (98.6; my wording in review 073) | **R2's restatement case is "mean ≤ 0", and that is mine.** With an informative CI (width ≤ 0.10), a mean of −0.002 can sit on a CI reaching about +0.05. "v9's unseen-compound accuracy does not depend on the compound" would then overstate what the data exclude. I proposed that sentence in 073 without the bound; I concede it. | Have `R2_SENTENCE['mean_le_0']` (and §5.5's restated headline) carry the CI's upper end: *"no detectable dependence on the compound: its contribution is at most {ci_hi:+.3f} per molecule (95 % cluster CI)"*. The case logic is unchanged. |

## Answers to the asks

**Ask 1 — yes, as published.** Their `main()` with our h5ad:
- **Input and target:** `obsm['X_ctl']` is the row's control (dense float32). `X` is the absolute profile, and their loss is on
  `x1 + delta` against it.
- **The split values** match their literals exactly: `'train'` / `'valid'` / `'test'`.
- **The obs columns** their tail needs (`pert_idx`, `cell_idx`) exist as int64.
- **Row order:** `test_loader` has `shuffle=False`, so prediction order is the h5ad test-mask order, which `convert_outputs` maps
  back by `obs['row_index']`.
- **The checkpoint dict** (an int epoch, a state dict, a float `val_deg_pcc`) loads under `torch.load`'s weights-only default.
- **The flags:** only `--h5ad --split_col --output_dir --seed --epochs 500` are passed. The defaults (batch 512, lr 1e-4,
  eval 1, 3 × 2,048, patience 50, dropout 0.1, clip 1.0) are 98.6's.
- **The empty-valid fallback** can't trigger given guard (c); C1 makes that checked at run time.
- **One record point:** the vendored file's `--seed` default is **42**, not 131419. "Their default" for 131419 must come from
  their harness's launch scripts, which aren't vendored. Cite that source, or call the seeds "chosen".

**Ask 2 — nothing reaches `/kaggle/working` except prediction Δs, `best_model.pt` (validation PCC only),
`resource_log.json` (epochs, wall time, `n_test`) and the record.**
- **What stays in `/kaggle/temp`:** `test_ground_truth.npy`, `test_ctrl.npy`, `per_drug_pcc.csv` and the h5ad, which aren't
  persisted. `compute_all_21` is never called.
- **What the trainer prints:** the test set only as a count.
- **The one residual path** is C1.

**Ask 3 — yes, minimal.** The diff is the 7 hunks above, and the delta, loss, output width (`X1_DIM` still 978) and prediction
are unchanged. Conditioning is standardised on the fit rows only.

**Ask 4 — yes, mechanical and faithful to 98.6:**
- **UNINFORMATIVE takes precedence:** `verdict` checks the width first, and `_decide` tests UNINFORMATIVE first.
- **Pass:** only `verdict`'s own "v9 predicts … better than" string.
- **`mean_le_0`:** any non-pass with mean ≤ 0. That includes mean = 0 exactly and the significant reverse.
- **The verdict is re-derived** from the stored numbers and must equal the stored one.
- **The references:** R1 must be `mlp_m` and R2 `v9_drugblind`. Only `of_record` is opened.
- **`restate_p9_headline`** is true only for R2's `mean_le_0`.
- **Wording:** C4.

**Ask 5 — yes, right.** The carve rows are training rows. v9 trained on them, and v9's own `x_cell` is the mean over all
training rows of the cell. F0 and the noise-free target are reported floors and references, not M, so using all training rows is
correct. The carve exists only to give M a non-test early stop.

**Ask 6 — agreed.**
- **M and M+dt on free CPU now.** Add C1's log guard first; it's a one-line kernel change.
- **Score R1 (`mlp_m`) when the runs finish.** R1 is a benchmark statement, B1's registration is fixed, and the reader handles
  "R2 pending".
- C2 and C3 can land before scoring.

## What I checked and found sound

- **The code:** `train_mlp.py` (vendored, read in full); `p10_mlp.py` in full; `make_p10_m_kernels.py`'s output (all six kernels
  and their metadata: CPU, internet off, the pinned image); `score_p10.py` in full; `score_p9.verdict`.
- **Rerun:** the three test suites (106 pass).
- **Hashes:** the pinned sha1s against the files and the git blobs; the line endings by bytes.
- **The real bundle's keys,** loaded without pickle (all unicode or numeric).
- **The dt-script regeneration** and its diff.

## What I could not assess, and why

- **Their harness's launch scripts** (the seed provenance) and **the Kaggle dataset `lincs-p10-src` itself.** I relied on the
  kernels' in-run sha1 guards, which pin the reviewed bytes.
- **The mutation runs.** I didn't rerun them.
