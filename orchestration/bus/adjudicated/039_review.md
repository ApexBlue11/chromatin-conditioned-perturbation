# REVIEW OF PACKET 039
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 1cd614c

**`kern_v9p7` at 1cd614c is cleared for the 05:30 IST push, with §88 t1 in the second slot.**
- **The adjudication:** C1–C4 are implemented faithfully, in RESULTS §85.12 items 5, 7 and 8, §85.13 (i), (ii) and (iv),
  §90.6, `score_p7.py` and the manuscript's P6 paragraph.
- **GUARD 0** runs before any staging or training, and it passes on the probed layout.
- **`check_snapshot_identities`** refuses the cases it should. It can't refuse correct output, because both identities
  hold by construction in `xpert_arm.py`.
- **The tests:** I reran `test_score_p7.py` and got 8 passed in 3.8 s. The `--dry_check` tests ran against the real O2
  profile, not a skip.

One MINOR point, for the manuscript only: the pathway-readout numbers in the abstract and §5.4 don't say which model
they describe (C1). It doesn't touch the push.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | provenance | **The abstract's (iv) and §5.4's pathway claim don't name their model, and the final model differs.** *"ranks which pathways move in held-out cells (ρ 0.273 …)"* and *"its own readout beats the same model's readout for other cells in all 6 dev cells"* are §85.10's numbers for **P2**, the dev baseline: one 12-epoch model, no sign head, no snapshots. The model the paper's second comparison scores is the C6 + V2 stack. On dev, its readout (final-snapshot weights) is **0.265**, licensed "in this cell" at **5 of 6** (HL60 −0.0001), as §85.13 already states. A reader will attach 0.273 and "all 6" to the model compared against XPert. That's the same object-attribution issue as review 038 C3, which you've already fixed in §5.3. | Name the model in both places until P7's test-cell reading replaces them. For example: *"… ranks which pathways move in held-out cells (dev baseline: ρ 0.273, +0.044 above a cell-agnostic prior; the final model's final-snapshot weights: 0.265, 'in this cell' on 5 of 6 dev cells)"*. In §5.4, write *"in all 6 dev cells (dev baseline; 5 of 6 for the final model)"*. |

## Answers to the asks

**Ask 1 — yes, the adjudication is right, apart from C1 above.**
- **C1 (review 038):** the centred score is now defined correctly, ranks no drugs, and is stated as "consistent with".
- **C2:** your recomputation matches mine to the digit.
- **C3:** §85.12 item 8 and §85.13 (iv) attach 0.2650 and the in-cell licence to the final snapshot, and +0.01844 to
  the prediction average. The manuscript's *"pathway alignment 0.265, read on the final-snapshot weights (whose own
  accuracy is +0.0087)"* is exact.
- **C4:** `ALT_LABEL` is used at the one place the alt row is built (`score_p7.py:100`). §85.12 items 5 and 7 and §90.6
  agree with it.
- **The manuscript's P6 paragraph:** each number sits with its object. It has +0.004 / −0.006 / +0.007 and "0.004
  below V2 alone", credits U937 for the edge, and says "in the final model by the pre-registered rule".

**Ask 2 — GUARD 0 is correct and harmless.**
- **Placement:** it sits after the pin checks and before `shutil.rmtree(STAGE)` and the `Popen`. So it can only refuse
  before anything is staged or trained, and its `SystemExit` has nothing to clean up.
- **On the probe's layout it passes.** `/tmp` has 1.2 TB free and `/kaggle/working` 20.9 GB, on different `st_dev`s.
  The minimum is 20.9 GB ≥ 5 GB, and the same-device branch doesn't apply. In the smoke it only prints.
- **The thresholds fit the expected output, not the incompressible worst case:**

  | case | per directory | if `/tmp` and `/kaggle/working` share a disk |
  |---|---|---|
  | expected: dev files are 47.4 MB compressed at 4,043 rows, so ≈ 248 MB each at 21,151 rows; 15 of them plus 3 checkpoints of 53 MB | ≈ 3.9 GB | ≈ 7.8 GB |
  | worst case: 4 float32 arrays × 21,151 × 978 = 331 MB per file, uncompressed | ≈ 5.1 GB | ≈ 10.2 GB |
  | GUARD 0 threshold | 5 GB | 9 GB |

  If you want the guard to be sufficient whatever the compression, use **6 GB / 11 GB**. On the probed layout that
  changes nothing, so it's optional.

**Ask 3 — yes, it refuses every case it should and nothing it shouldn't.**
- **What it refuses:**
  - a row-set mismatch in any snapshot or `_last`;
  - any difference between `_last` and `_snap2` on either scored array;
  - a main file more than 1e-5 from the snapshot mean on either array;
  - a missing snapshot file. Its count comes from the manifest, and the files are sha1-checked before the identities run.
- **Correct output can't trip it:**
  - `_last` and `_snap2` are written from the same array, `snapshots_pa[-1]` and `snapshots_pd[-1]`
    (`xpert_arm.py:499-505`), so exact equality holds by construction.
  - The main file is `np.mean(snapshots, 0)` (`:473`) cast to float32. A float32 mean recomputed in any order differs
    by a few ulp. The largest |y_pred| on P6's dev files is 16.0, where an ulp is 1.9e-6, so the worst case is
    ≈ 4e-6 < 1e-5. My float64 recomputation gave 6.4e-7.
  - A wrong main file, such as one snapshot or the mean of two, differs by about 0.01–1, so it's caught.
- **`row_index` dtype:** `np.array_equal` compares values, so int32 against int64 passes when the rows agree. That's
  correct.
- **One optional improvement: NaN.** A NaN prediction makes both `array_equal` and `d <= tol` false. The refusal is
  right, since `coldcell_h2h.py` would otherwise score only rows finite in every file, shrinking the row set. But the
  message would wrongly blame `_last` or the mean. An `np.isfinite(...).all()` check first, with its own message, would
  name the real cause.

**Ask 4 — cleared.** Push `kern_v9p7` at 1cd614c at 05:30 IST.
- **§88 t1 is safe in the second slot:**
  - `kern_moa88_t1` differs from `t0` by one line (`SEED = 1`).
  - The code it trains (`train_v9_gpu.py`, `model_v9.py`, `modules_v9.py`, `config_v9.py`, `dp_seeding.py`) last
    changed on 09-26 at 09:37 or earlier, before t0 ran.
  - The staged upload (`external/kaggle_v9_src`) equals the repo on all five files, ignoring line endings.
  - So t1 trains exactly t0's code with another seed.
- **Quota:** P7 (6–6.5 h) + t1 + t2 (≈ 7.2 h each) ≈ 21 h of 30.
- **After P7:** in order, run `score_p7.py --dry_check`, pin `P7_SHA1`, score once, and run `align_dev.py --rows test`
  once.

## What I checked and found sound

- **The diff f527d7f..1cd614c:**
  - The kernel and generator change by GUARD 0 only, identical in both.
  - `score_p7.py` gains the check and the label.
  - The disk probe is a CPU kernel with no data and no internet.
- **The tests:** the fixtures build main files with the same float32 `np.mean` as the arm. Tampering a `_last` and
  re-hashing the manifest is the right adversarial case. The no-V2 stack skips the guard.
- **RESULTS §85.13 (i), (ii) and (iv), and §90.6:** the wording is as adjudicated.

## What I could not assess, and why

- **The GPU machine's disk layout.** Only a CPU session was probed. GUARD 0 makes this safe: a different layout either
  passes the thresholds or refuses before any GPU time is spent.
