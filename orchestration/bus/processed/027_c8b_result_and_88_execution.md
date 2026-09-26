# PACKET 027 — RESULT: C8b's dev screen (3 seeds) read by §85.2 rule 7 and the §88.5 gate; §88's execution plan
packet_id: 027
created: 2026-09-26
repo_commit: 5af16eb
type: **RESULT + GPU plan.** The accuracy numbers are final. The §88 code is being written (worker W20), and the
operationalisations in part B are committed before any §88 model is trained or probed.

## A. C8b (`--post_pathway`), 3 seeds, dev rows (RESULTS §85.4 carve, sha1 51e7e4ab… checked in-kernel, GUARD 4 OK)
Kernel `lincs-v9dev-c8b`, Kaggle 2×T4, distinct seeding (GUARD 5: device CUDA states unequal after all 12 epochs, all three
seeds), ~6,000 s per seed. Scored by `model/v9/score_dev.py` against P2 (μ0 0.43693, s0 0.00169).

| seed | per-row mean | mean of cell means |
|---|---|---|
| 0 | 0.4376 | 0.4315 |
| 1 | 0.4393 | 0.4418 |
| 2 | 0.4417 | 0.4395 |
| **mean** (sd 0.00206) | **0.43955** | |

Δ per-row mean **+0.00262**; Δ mean of cell means **+0.00046**; per-cell median Δ: HEK293T −0.0102, HL60 −0.0065,
LNCAP +0.0088, SKBR3 +0.0008, U937 −0.0005, VCAP +0.0020 → **3 of 6** favour C8b.

**Rule 7, read mechanically:** threshold = max(0.003, 2·√(s0²/3 + s_v²/3)) = **0.00308**. Δ 0.00262 < 0.00308 (fails);
cell means > 0 (passes); 3 of 6 < 4 (fails) → **NOT ACCEPTED.** C8b does not enter the stack.
**§88.5 gate:** 3-seed mean Δ ≥ −s0 (−0.00169) → +0.00262 **passes → §88 executes.**
Reported, not read: seed 0 alone (+0.0007) would have been "one more seed" under rule 6; C8b ran 3 seeds by §88.5.
Recorded: `model/results/v9_dev_score_C8b_postpath.json`, ledger, RESULTS §85.8 row.

## B. §88 execution — operationalisations committed now (before any §88 model or probe exists)
1. **Training (88.1 item 1):** `train_v9_gpu.py --post_pathway --seed S --epochs 12 --d_model 256 --l_control 1
   --budget_h 7.5 --gpus 2 --dp_seed_mode distinct --tf32 off`, S = 0, 1, 2 — the fold-0 recipe of the last fold-0
   run (`kern_drugsa_s0`: 12 epochs, d 256, l_control 1) with drug self-attention off (V9Config default) and **batch 48
   (V9TrainConfig default)**. The r-series used `--batch 96`; nothing here is compared with it (§88.4). The two new flags
   default to today's behaviour; a CPU test asserts the default parse and the distinct seed list `[s, s+1000]`, and the
   trainer refuses to continue if the device states are equal after epoch 0 with `distinct`.
2. **Untrained inits (88.1 item 5):** built from the seed-0 C8b checkpoint's `cfg` (so `post_pathway` on and every shape
   identical), weights **not** loaded, `torch.manual_seed(0…4)`, quantiser fitted on 20,000 training rows exactly as §86.
   The probe refuses a cfg without `post_pathway`.
3. **Rows:** §86's code path; asserted row sha1 = `3e59a7ba…` (§86's: 496 scored compounds, 156 unseen).
4. **Null 1s (88.2 item 4):** compound response strength = median over the compound's scored rows of the row's mean over
   the 978 genes of `|y_Δ|`; quintile edges by `np.quantile` over the D scored compounds; positives are permuted only
   among compounds in the same quintile; 1,000 permutations; `p_s = mean(perm ≤ S)`.
5. **Responsiveness strata (88.1 item 7):** per compound, rank the 978 genes by the mean over its rows of `|y_Δ|`
   (percentile in [0, 1), 0 = most responsive); the compound's value = mean percentile of its landmark targets; no
   landmark target → the third, reported group; the rest split at their median (≤ median = `target_responsive`).
6. **Reader:** `model/v9/read_moa_88.py`, applying §88.3 mechanically to the 3 trained + 5 untrained JSONs, committed
   **before** the probe runs on any trained model. `m_u`, `sd_u` = mean and sd (ddof 1) over the 5 untrained `diff`s.
7. **Probe on Kaggle CPU** (free), one kernel per checkpoint, as §86.

## C. This week's GPU plan (Kaggle's 30 h quota reset today; ~0 h used before these)
| item | GPU-h | status |
|---|---|---|
| C3 (`--listnet_w 1.116903`), C6 (`--sign_head_w 0.492066`), seed 0 each | 2 × ~1.7 | running |
| C7 (`--chromatin_edges`), seed 0 | ~1.7 | queued behind C3/C6 |
| §88: 3 fold-0 C8b seeds | 3 × ~5.7 | after W20 is verified |
| follow-up seeds for any candidate that advances; C7u × 3 if C7 is accepted | ≤ ~5 | as rule 6 / §89.2 decide |
Total ≈ 27 h. P6/P7 (the stacked final) would then fall in next week's quota.

## ASKS
1. Is part A's reading of rule 7 and §88.5 correct, and is anything about C8b's accuracy result mis-stated?
2. Are the five operationalisations in part B faithful to §88 as amended by review 025 — in particular the response-
   strength definition for Null 1s and building the untrained inits from the trained checkpoint's cfg?
3. Batch 48 vs the r-series' 96: agree that §88.1's "V9Config defaults otherwise" means 48?
4. The GPU ordering in part C: §88 (~17 h) before any P6/P7 spend. Given C8b is not accepted for accuracy, is spending
   ~17 GPU-h on its MoA test this week the right priority over the final model (P7), or should §88 wait until P7 is
   scored? (§88.5 says §88 executes; it does not say when.)
