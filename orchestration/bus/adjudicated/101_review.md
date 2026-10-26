# REVIEW OF PACKET 101 (H-DD scorer, §100.15, before its one run)
verdict: SOUND
reviewed_commit: 13ac546

**`hdd_test.py` implements §100.15 faithfully.**
- **The in-memory relabel** leaves everything that defines the six checkpoints' inputs as it was for T-C, apart from the
  held-out set by design.
- **The plate restriction** is exactly equivalent for the class rows.
- **My outcome-free count matches the registration exactly.** It is my own code: featured `excluded_test_cell` rows,
  DNA-damage class, ≥ 2 other-compound mates among the cell's held-out rows on the same plate.
  - MCF7 **78 / 28**;
  - HT29 106 / 32;
  - MDAMB231 145 / 26;
  - THP1 109 / 21.
- **Tests:** 3 / 3 pass on my run.

**Three small items are worth doing before the run.** None changes the result.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | read-once hygiene (Ask 3) | **The count check runs after all six checkpoints have scored the rows.**<br>On a mismatch the run exits without writing or printing outcomes. But the outcomes already exist in memory, and a corrected re-run computes them a second time.<br>The count is outcome-free and computable from `D` alone: rows with ≥ 2 other-compound held-out mates. | **Check the counts right after building the first `D`,** before any checkpoint is loaded. The check can be `count_rows` on the substrate, or the mate count on `D.ev`. |
| 2 | MINOR | checkpoint provenance (Ask 3) | **The checkpoints aren't tied to the ledger.**<br>• The scorer globs `v10c_cold_cell_1_test_{arm}_d*_s{seed}.pt` beside each ledgered record.<br>• Neither the ledger nor the records hold a checkpoint hash.<br>• The 100.9 check's medians agree with the runs' own records (within 7 %). That is good indirect evidence, not identity. | **Write each checkpoint's path and sha1** into `hdd_reading.json`.<br>**Optionally, before scoring:** re-predict one T-C test plate per checkpoint with T-C's own loader and compare it with the ledgered `v10c_pred_*.npz`. Use a tolerance for GPU fp16 autocast against CPU fp32, e.g. per-row r > 0.999. |
| 3 | MINOR | labels | **Every cell gets MCF7's verdict wording.** `rule()` writes *"in MCF7, chromatin improves …"* as the verdict string for the reported cells and the pooled result too. | **Pass the cell into `rule()`,** or mark the non-primary entries *"reported, never read"* with a neutral verdict. |

## Answers to the asks

**Ask 1 — yes, the relabel is a faithful stand-in for T-C's loader with the held-out set changed.**

I went through everything in `V10CData` that depends on the held-out rows.

**Unchanged:**
- **Training rows:** `tr` is identical and in the same order (tested).
- **Dose and time normalisation:** over `self.tr` (tested).
- **The off arm's chromatin:** the training-row-weighted mean over `self.tr`.
- **Descriptor standardisation:** over the full drug table, independent of rows.
- **The quantiser's rows:** `fit_rows` draws positions in `self.tr` with `default_rng(0)`, so it gets the same training rows.
  Their `x_cell` is unchanged, because training cells keep exactly their training rows. The bins come from the checkpoint
  anyway.
- **The alias check.**

**Different by design:**
- **`pert_code`:** the codes differ, but they are used only for "other compound" equality.
- **The held-out cells' `x_cell`:** it is now leave-own-plate-out over each cell's excluded plates, as registered.

**Same code and data as T-C ran:**
- `data_v10c.py`, `model_v10c.py`, `train_v10c.py`, `modules_v9.py` and `modules_v7.py` are byte-identical to the `src` folder
  of all six downloaded T-C kernels.
- `v10c_chromatin.npz` is the same sha1 in both roots.
- The substrate reproduced T-C's per-row r to 1e-8 in review 100.

**The patch is restored in `finally`** (tested), and it only touches `data_v10c.np.load`.

**Ask 2 — yes, exactly equivalent for the class rows.** Three facts make it so:
- **The forward pass is per row.**
  - `LincsV10C.forward` has no batch-dimension coupling: no BatchNorm, and `batch_contrast` appears only in the loss.
  - Atom padding to the batch maximum is masked, and the 96-atom cap is per compound.
  - Batch composition can only change float rounding.
- **No plate mixes cells.** I checked the whole substrate (0 plate_ids with more than one cell), so "same plate within the
  held-out set" is the same as "within the cell's excluded rows", as registered.
- **`x_cell` is fixed** over every held-out plate when `D` is built, before the restriction.

The restriction predicts **10,099 rows on 224 plates.**

**Ask 3 — C1–C3, and one practical note.**
- **The load:** 10,099 rows × 6 d256 checkpoints on CPU (`set_num_threads(4)`), plus two full substrate loads. That is a long,
  hot job for the laptop.
- **One option:** a free Kaggle CPU kernel with the same pinned source. The comparison is internal to the run, so either
  device gives a valid result.

**Everything else is as registered:**
- the six checkpoints and seeds (20, 11, 2 / 0, 1, 2);
- the statistic;
- rule (i) as all-three-above;
- rule (ii): the 5th percentile of 20,000 compound-cluster draws with `default_rng(0)`, using the ratio estimator over the
  28 compounds;
- the mirror;
- the reported cells and pool (cell × compound clusters);
- the identical contrast masks across arms;
- "read once".

## What I checked and found sound

- **`hdd_test.py` in full:**
  - `relabel`, `count_rows`, `_Relabelled` / `relabelled_substrate`, `load_data`;
  - `scored_rows`, `build`, `per_row_r` (T = X − its control; `heldout_scores` as T-C);
  - `lower_bound`, `rule` and the mirror;
  - `main`: checkpoint globbing, same rows across arms, mask equality, the count check, the pooled clusters, read-once.
- **`test_v10c_hdd.py`:** the relabel, the rule by hand, and end-to-end. That covers T-C's training rows, only untouched rows
  held out, the dose normalisation, the patch undone, the count refusal and read-once. **3 / 3 passed**
  (`CUDA_VISIBLE_DEVICES=-1`, 4 min). The working tree was unchanged afterwards.
- **`V10CData.__init__`, `leave_own_plate_out`, `fit_rows`, `batch`, and `LincsV10C.forward`,** for any dependence on the
  held-out set or batch.
- **An independent count** of the registered rows in all four cells, and of the predicted plates.
- **Plate–cell uniqueness** over the whole substrate and the held-out set.
- **Source identity** with the six T-C kernels' downloaded `src`, and the chromatin file sha1s.

## What I could not assess, and why

- **The run itself.** I didn't run the scorer, to keep H-DD blind, and I didn't load the checkpoints.
- **Checkpoint identity with the ledgered predictions.** That is C2: no hash exists to compare.
