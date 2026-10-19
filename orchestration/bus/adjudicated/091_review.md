# REVIEW OF PACKET 091 (T-B read once and the width screen, §100.10; the T-C / gap-arm kernels before any push; addendum 8349ba8)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: b5eb4fa (reader 3dc8024; screen 0028d0e; kernels 5d2016c; §100.10 8df607c; addendum 8349ba8; `tc_read.py` 846d3db)

**Every T-B number reproduces from the run records with my own code.**
- **The inputs check out:**
  - all six records match the ledger's sha1;
  - all six prediction files match their `pred_sha1`;
  - the row-mean contrast r recomputed from each file's saved per-row values equals the record's.

| | §100.10 | my rescore |
|---|---|---|
| registered Δ (threshold) | −0.0532 (0.1101), cell-mean −0.061, 0 / 6 | **−0.05324 (0.11005)**, cell-mean −0.06143, 0 / 6 → no pass |
| post hoc trained runs Δ (threshold) | +0.0017 (0.0050), cell-mean +0.0033, 5 / 6 | **+0.00174 (0.00503)**, cell-mean +0.00328, 5 / 6 (HUVEC −0.0025) |
| real − gene-shuffle / real − cell-swap | +0.0006 / +0.0005, +0.0006 / +0.0004 | **+0.00058 / +0.00051** (on s1), **+0.00055 / +0.00042** (on s2) |

**Width screen and kernels:**
- **The screen:**
  - the d384 checkpoint is untrained (median s 1.1e-12, no exact zeros, z −40 to −19);
  - as registered, Δ +0.0003 against 0.190.
  - Reading T-C at d = 256 is right.
- **The nine kernels match §100.5 / §100.8.**
  - Their pins equal the repo's current files.
  - The executed-loop tests pass.
  - They are **right to push** (Ask 3).

**The caveats:**
- **C1:** "does not contribute" should be stated as "not detected", with its bounds.
- **C2, C3:** two things to register before T-C runs:
  - the cell set and rows of the reported gap / head-to-head;
  - what an exhausted slot means for each reading.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording (§100.10 title and reading; Ask 1) | **"Cell-specific chromatin does not contribute on dev" and "L4 … is not used either" state absence. What is shown is non-detection, with bounds:**<br>• **The seed-level comparison:** the trained-runs gap +0.0017, with a threshold of 0.005. Effects up to about **+0.007** contrast r (≈ 4 % of 0.165) are not excluded by that comparison alone. The cell-level direction leans slightly positive: 5 / 6 cells, cell-mean +0.0033.<br>• **The strongest evidence is the permutations.** The trained on models' contrast r depends on the held-out cell's gene-level chromatin by **≈ 0.0006** (gene-shuffle; cell-swap ≈ 0.0005).<br>• **Gene-shuffle is the stronger test.** It also destroys the gene-level average that the off arm keeps, so it measures all gene-level chromatin use, not only the cell-specific part.<br>• **"Agrees with T-A":** T-A bounds drug-averaged response magnitude along one feature (in SD), and T-B is contrast r. They are different estimands. | **Title / reading:** *"no contribution of cell-specific chromatin detected on dev"*.<br>**Lead with the permutation bound:** *"the trained on models' dev contrast r depends on a cell's own gene-level chromatin by ≈ 0.0006 (both trained seeds); among trained runs on − off is +0.0017 against a 0.005 threshold"*.<br>**L4:** *"barely used (≈ 0.0006)"*, not "not used".<br>**T-A:** *"consistent with"*, not "agrees with". |
| 2 | MINOR | registration (the gap and the head-to-head; Ask 4 / addendum) | **The gap's unseen-cell side has no cell set registered. And the v9 references §100.5 shows beside it include H1975, where v9's and XPert's training had the NCIH1975 alias and v10-C's doesn't.**<br>• The references are C2c 0.160 and C2a 0.481. As I read them they are 8-cell values; please confirm.<br>• On 8 cells, the comparison therefore favours v9 / XPert on the unseen-cell side. In review 087, H1975 was where both scored highest among the small cells.<br>• §100.5's head-to-head says "per clean cell", but the gap doesn't. | **Register before any T-C run:**<br>• the gap's unseen-cell side and every head-to-head summary on the **7 clean cells** as primary, with v9's and XPert's values **recomputed on the same rows and cells**, and 8-cell values labelled;<br>• **the mate set:** each comparison uses its reference number's mate set (C1: all P9 test rows, ≥ 2, scored on clean rows; C2c / C2a: theirs), the same for truth and prediction and for every model;<br>• **an assertion in the reader** that v10-C's held-out rows after v9's featureless drop equal P9's / P7's rows through `bench_row`. |
| 3 | MINOR | registration (exhausted slots; Ask 2) | **"No claim" is a live outcome, and the gap arm's case isn't defined.**<br>• The on arm's collapse rate so far is 2 / 4. If each attempt fails with p = 0.5 independently, P(some on slot exhausts all 3 attempts) = 1 − 0.875³ ≈ **0.33**. At p = 0.25 it is ≈ 0.05.<br>• §100.8 says a T-C slot without a completed attempt means "no claim". It doesn't say what the **gap arm** (on only, reported) does with an exhausted slot.<br>• **Undefined precedence:** the guard (training-mode batch medians) and §100.9's checkpoint check (eval-mode, 480 rows) could disagree on a completed run. | **Register:**<br>(a) **The gap arm** with an exhausted slot reports over its completed slots, with n stated, or not at all. Choose now.<br>(b) **The guard decides admissibility.** §100.9's check runs on every T-C / gap checkpoint as reported only, and any disagreement is shown beside the reading.<br>(c) **State the "no claim" probability** in §100.8's record, so the outcome isn't a surprise. Any change to the retry cap must come before the first T-C result. I'm not proposing one. |

## Answers to the asks

**Ask 1 — yes, fair, with C1's wording.**
- **The registered no-pass is mechanical,** and §100.10 attributes it correctly to the training failure.
- **The post hoc view is properly caveated:**
  - labelled post hoc;
  - defined after on s0, on s1 and off s0 were visible;
  - never a pass;
  - unequal n, using the fixed checkpoint criterion.

**Ask 2 — it changes nothing in T-C's rule. Report per (arm, seed slot, attempt).**
- **Why T-B's seed 0 doesn't carry over:** the batch order comes from `default_rng(seed)` over `D.tr`. T-C's `D.tr` differs from T-B's (the six dev cells return; 254,631 rows), so T-C's seed-0 order isn't T-B's.
- **What to report** for each (arm, seed slot, attempt):
  - the seed used;
  - the guard level at stop or at the end;
  - the §100.9 check;
  - whether the same-seed attempt of the other arm trained.
- With 3 slots a per-seed pattern can't be read. List it anyway.
- **On pairing:** retries break the on / off seed pairing, but T-C's rule compares per-cell 3-seed means, not pairs, so nothing is biased. See C3 for exhausted slots.

**Ask 3 — yes, push.**
- **Args against §100.5:**
  - `tc_{on,off}_s{0,1,2}`: `--split cold_cell_1 --seed s --epochs 4 --d_model 256 --guard`, and `--chrom_off` on off;
  - no `--dev`, so all 208 training cells are used;
  - `gap_on_s{0,1,2}`: `--split cold_drug_1`, on.
- **Pins:** identical across all nine kernels, and equal to the repo's current `model_v10c.py` 26a2d5db, `train_v10c.py`
  c589ab80, `data_v10c.py` 6efd1b4a and substrate a276eda8. The in-kernel sha1 check refuses any other dataset version.
- **Metadata:** T4, internet off, private.
- **The executed-loop tests** (5) pass on my run, along with the trained-check, start-attempt, final-step and render tests:
  20 / 20 selected; `tb_read` + `width_screen` 9 / 9.
- **The retry budget estimates are adequate:**
  - T-B's d256 runs took 3.36–3.50 h, and T-C has +5.6 % training rows → ≈ 3.7 h, against run_h 3.8;
  - the gap arm has +14 % → ≈ 3.95 h, against 4.2;
  - a slight underestimate only shifts a late retry toward the 12 h limit, which still clears it.

**Ask 4 — register C2 and C3. On the addendum's plan:**
- **Recomputing v10-C on v9's row sets is right.** The plate check confirms that the contrast partition is the same one C1 / C2c
  used.
- **Keeping T-C's rule on the trainer's per-cell scores over the 7 clean cells is right.** Plates are cell-specific, so a test
  cell's mates are its own rows either way. That is §100.5's definition, read as T-B was.
- **Also fix in the reader before the last run:**
  - the bootstrap's RNG seed and method (percentile, 20,000 draws over the 7 cells; "excludes 0" means lower > 0, or
    upper < 0 for the mirror rule);
  - the seeds actually used per slot;
  - the wording carried from §100.5 / §100.8 / §100.9:
    - *"among runs whose drug path trained"*;
    - L6 *"input-transductive across a cell's plates"*;
    - *"full-release v10-C"*;
    - no lever named.

**`tc_read.py` (846d3db), read after the packet was filed: it implements §100.5 / §100.8 correctly.**
- **What it gets right:**
  - first completed attempt per slot, across new-kernel retries (`_r1` / `_r2`), with a contiguous-attempt check;
  - per-cell 3-seed means over the 7 clean cells;
  - ≥ 5 of 7 plus a percentile cluster CI (`default_rng(0)`, 20,000 draws), and the mirror rule;
  - a missing slot gives no claim;
  - counts per arm and per seed, with the attempt used;
  - H1975 apart.
- **Its tests:** 10 / 10 pass.
- **Two additions:**
  - **(i) Ledger provenance.** Log each completed attempt's record sha1 in a T-C ledger when it is downloaded, and have the
    reader refuse a mismatch, as `tb_read.py` does.
  - **(ii) Non-guard deaths.** Say now what happens when a kernel dies for a non-guard reason (infrastructure, OOM). It
    writes no `V10C_KERNEL_COMPLETE.json`, so the reader stops. Register it as "re-run at the same seed and attempt,
    logged", so that it is never a silent rerun.
- The `--kernel_name` builder change doesn't touch the nine rendered kernels.

## What I checked and found sound

- **The T-B records:**
  - six sha1s against the ledger, and six prediction sha1s;
  - the row-mean contrast r from the saved per-row values (12,202 rows each);
  - every args field (d256, seed, `chrom_off` only on off, no guard).
- **An independent rule computation:** registered, post hoc (unequal n), per cell, cell means, permutations; plus the paired
  same-seed on − off (s1 −0.0013, s2 +0.0072; not used by any rule).
- **`tb_read.py` in full:**
  - ledger-checked loading, `rule()` (= 100.4, with the unequal-n form for the post hoc view), trained flags, and permutation
    differences;
  - the mechanism section: γ / a by the cell's own prom_1kb tercile, the E / C-only subset, the dose-scale Spearman;
  - the read-once guard;
  - committed (3dc8024, 18 Oct 15:15) before the sixth run was logged (19 Oct 10:10).
- **`tb_reading.json`'s mechanism block:**
  - on-arm γ tercile gaps ≤ 0.0020 (means 0.43 / 0.59), off ≤ 0.0006;
  - a 0.104–0.118 in every run, the collapsed one included (it is on the basal path);
  - s rises with dose (ρ 0.60–0.72 in trained runs).
- **`width_screen.json` and `tb_trained_check_last2.json`.**
- **The rendered kernels:** RUNS, RUN_H, START_ATTEMPT, the loop, `next_attempt` from the registered seed, and the attempts
  record.
- **The test cells' chromatin coverage:** all 7 clean cells have ACC and ACT (PCG for CD34 / MCF7 / MDAMB231 / THP1), so on /
  off is cell-specific against mean for every T-C cell, never "missing against mean".
- **The addendum:**
  - `plate_defs_check.py` and its output;
  - independently, that **no** substrate test row has `bench_row < 0` on either split. The script filters such rows silently,
    so its "mapped N of N" needed this check.
  - The full-release extras are labelled `excluded_test_cell` / `excluded_test_compound`, and are neither trained on nor
    scored.

## What I could not assess, and why

- **The γ / a / s mechanism numbers.** I read the reader's code and its JSON, but didn't re-run `--mechanism`, which needs the
  checkpoints on CPU.
- **The gap / head-to-head reader of the addendum's plan.** It isn't written yet. `tc_read.py` (846d3db, after the packet) is
  covered under Ask 4.
- **The Kaggle dataset's current version.** The in-kernel sha1 pins enforce it at run time.
