# REVIEW OF PACKET 052
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 3103adb

**The Stage 1a code implements 93.2 / 93.5 / 93.6, and I found no leak.**
- **Tests:** I reran the three test files under `.venv-cuda` (lightgbm 4.7.0): **11 passed** (143 s).
- **Pins:** the kernels' 10 pins equal the repo files: `chromatin_funnel` c56bb6f1, `chromatin_power` c501cba2, `score_dev`
  a9002bce, `chromatin_h2` e75d1281, `chromatin_gbm` 3f3ee97d, and `E_final*` and `cell_lineage` as in §91.

**H3's folds don't leak.**
- **Validation cells:** each fold's training pairs exclude the fold's cells (asserted).
- **Fold features:** come from `make_B(train_cells)`, so N1's means exclude the fold.
- **The target:** uses μ from a pool without the fold.
- **The final fit:** uses all 11 covered cells, with dev cells never in a fit (`assert_no_dev`).

**The rest matches the registration.**
- **The reading:** T1's five conjuncts plus `vs_N1_all ≥ t1/2` (0.002 on known rows).
- **The calibration:** re-runs fold choice and final fit inside every case (null; P1/P2 × 3 π; P3 5 %; G 2 % and 5 %), giving
  the 1,209 fits stated.
- **MDE:** 3 of 3 draws.
- **The reader:** orders VOID → ADVANCES → label, with "informative iff MDE_P1 ≤ 2 %".

**H2:** `run_t1` with a subset keeps μ on the full dev-train pool and restricts the fit rows, LOCO and N1 to the subset, which
is what H2 asks.

There are three MINOR points. One must be fixed before the run: the gene-generic check can be vacuous (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | test-coverage | **The gene-generic check (93.5 item 2) can pass vacuously, both in the test and in the real run.** (a) `test_a_gene_generic_planted_effect_does_not_pass` plants μ ⊙ fbar with **fbar random**. Neither C (the world's chromatin) nor N1 can represent it, so raw Δ ≈ 0 and both asserts hold trivially. The test never shows N1 neutralising a gene-generic effect that C **can** capture. (b) 93.5 requires G "at a size whose raw Δ clears 0.004". `faults_and_mde` records `G_raw_delta`, but neither it nor `read_chromatin93.py` checks it. If both G cases' raw Δ on known rows fall below 0.004, VOID_G = False certifies nothing. | (a) In the test, plant G with fbar = the gene-mean of the world's primary mark over covered cells, so C can capture it. Assert raw Δ ≥ `BARS['known']['t1']` **and** `vs_N1_all` < t1/2 **and** no pass. (b) In the reader, report `G_check_informative` = (some G case has raw Δ ≥ 0.004 on known rows in every draw). If false, add to the H3 reading: *"the gene-generic check is uninformative (G's raw Δ below the bar)"*. Not VOID: the magnitude conjunct still protects the reading itself. Commit before the run, with the regenerated pins. |
| 2 | MINOR | provenance | **H2's k = all fit is §91's T1 exactly, but nothing checks that it reproduces §91.** Same specs, same 11 cells, same μ, LOCO and N1: its FBC − FB and FBC − N1 on known rows should equal §91.12's +0.00144 and +0.00023 (to float tolerance). That's a free end-to-end check of the harness on real data. | Record `k_all_reproduces_91` in `chromatin_h2_93.json`: \|Δ − 0.00144\| and \|Δ − 0.00023\| < 1e-6, against the committed `chromatin_funnel_91.json`. The reader prints it. A false value means a harness fault, not a reading. |
| 3 | MINOR | efficiency | **The "≈ 1 h" estimate counts fit time only.** `run_loocv_arm` recomputes `condition_means(ctx, y, pool)` for every (arm, min-child, fold), 36 times per `run_h3`, where 4 distinct pools would do. It also predicts the fold's full 978-gene validation block each time. Over 31 `run_h3` calls this may dominate the runtime. It isn't a correctness issue: 93.5's "re-planned, never silently shortened" covers a timeout. | Optional: cache μ per fold within `run_h3`. If not, expect 2–4 h and say so. |

## Answers to the asks

**Ask 1 — no leak. One mismatch with 93.5 (C1(b)), and the reading rules are right.**
- **Missing marks enter as 0** (the rank-normal median) in C and N1 alike. That's T1's convention, and the right choice:
  NaN would let trees split on mark availability, a cell-level identity proxy.
- **One harmless overlap:** `chromatin_gbm.py` writes its own `CHROMATIN93_H3_COMPLETE.json`, and the kernel then overwrites
  it with pins and the output sha1. The reader needs only `complete` and `outputs`.

**Ask 2 — both amendments are acceptable.**
- **Grouped 4-fold over cells** keeps the cell-level hold-out for a single regularisation knob over 3 values.
- **The 5 % pair sample** (about a million pairs against ≤ 6 features) is ample for these trees. Because the calibration uses
  the same sample, the MDE is the MDE of the test as run.
- Both were made before any run and are disclosed in 93.6.

**Ask 3 — yes, use 3 draws if it's cheap, which it is: closed form, about 30 more T1 fits.**
- **Why:** the planted curve is the reference for how much of a slope is the estimator's own small-k behaviour. With one draw,
  that reference carries the draw's own idiosyncrasy at every k.
- **How:** report the per-k median over draws × subsets.
- **Not required:** H2 is reported only.

## What I checked and found sound

- **`chromatin_gbm.py` in full:**
  - `get_fitting_pairs` (covered, dev-train, drug-known; seeded 5 %; the same in every case);
  - `get_folds` (rng 9302, round-robin);
  - `run_loocv_arm` (fold pool, fold features, fold target, assertion);
  - `fit_predict_arm`;
  - `run_h3` (known and all, `BARS` per row set);
  - `calibrate` (per-draw `cp.builders`; G = mean f\* in every covered cell; P2's per-perturbation w; P3);
  - `faults_and_mde`;
  - the marker.
- **`chromatin_h2.py`:** `subsets` (rng 9300 + k), `t1_deltas` (known rows), and `run_t1`'s subset semantics
  (`chromatin_funnel.py:311-336`).
- **`read_chromatin93.py`:** H2 strict monotone plus a 2-sd rise on FBC − N1; H3's VOID, ADVANCES and label logic; refusal on
  marker sha1.
- **The kernels:** pins, bundle sha1, `OMP_NUM_THREADS` 4, and the completion marker after the run. The H2 and H3 kernels
  differ only in script, output and marker names.
- **`feature_builder` and `cp.builders`:** they return features for every cell, so `make_dataset`'s `np.empty` is always
  filled.

## What I could not assess, and why

- **Runtime on Kaggle's CPUs.** I didn't run `run_h3` on the real data locally (laptop thermal limit). C3's estimate is from
  reading the code.
