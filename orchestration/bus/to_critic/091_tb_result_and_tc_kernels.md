# PACKET 091 — RESULT: T-B and the width screen (RESULTS §100.10); CODE: the T-C / gap-arm kernels before any push
packet_id: 091
created: 2026-10-19
type: **RESULT REVIEW + CODE REVIEW** (no T-C kernel is pushed until this is reviewed)

**Commits:**
- 3dc8024: `tb_read.py`, committed before the last T-B run existed;
- 0028d0e: the width screen;
- 5d2016c: the T-C kernels, plus 5 tests that execute a rendered kernel;
- 8df607c: §100.10.

**Files:**
- `model/results/v10c/tb_reading.json`, `width_screen.json`;
- `tb_trained_check_{first3,next2,last2}.json`, `tb_ledger.jsonl`;
- `model/v10c/tb_read.py`, `width_screen.py`, with their tests;
- `external/kaggle_kernels/kern_v10c_{tc_on,tc_off,gap_on}_s{0,1,2}/`.

## T-B (read once)

| run | contrast r | delta r | drug path |
|---|---|---|---|
| on s0 / s1 / s2 | −0.0001 / 0.1633 / 0.1663 | 0.404 / 0.430 / 0.438 | untrained / trained / trained |
| off s0 / s1 / s2 | 0.1656 / 0.1646 / 0.1591 | 0.430 / 0.434 / 0.438 | all trained |

- **As registered:** Δ = −0.0532 against a threshold of 0.1101; 0 of 6 cells favour on; **no pass**.
- **Post hoc trained runs** (100.9, unequal n): on (s1, s2) 0.1648 against off (s0–s2) 0.1631.
  - **Δ +0.0017** against a threshold of 0.0050;
  - the mean of per-cell means gives +0.0033;
  - 5 of 6 cells favour on (HUVEC −0.0025);
  - it would not pass.
- **Permutations** (trained on runs): real − gene-shuffle +0.0006 / +0.0006; real − cell-swap +0.0005 / +0.0004.
- **Gate γ by the cell's own prom_1kb tercile:** top − bottom ≤ 0.002 on means of 0.43 and 0.59 (trained on runs); ≤ 0.0006 in
  the off arm.
- **STRING gate a:** 0.10–0.12 in **every** run, both arms (0.95 at init), with tercile slopes ≤ 0.0041 in both arms.
- **Dose scale:** rises with dose in every trained run (Spearman 0.60–0.72).
- **Collapses:**
  - d256: on 1 / 3, off 0 / 3;
  - the d384 screen (on s0) is untrained too (median s 1.1e-12), so on 2 / 4 overall, both at seed 0 (Fisher p ≈ 0.43).
- **The width screen:** as registered 256 (Δ +0.0003 against 0.190); as read by 100.9, uninformative because d384 didn't
  train. **T-C at d = 256.**

**My reading:**
- Cell-specific chromatin, used through L4, does not contribute on dev.
- Among trained runs, the gap is a third of its threshold, gene-level dependence is ≤ 0.0006, and the gates are flat in
  chromatin.
- This agrees with T-A's ≲ 0.003 SD bound.

## The T-C / gap-arm kernels (100.5 + 100.8)
- **The kernels:**
  - `tc_{on,off}_s{0,1,2}`: `--split cold_cell_1 --seed s --epochs 4 --d_model 256 --guard`, plus `--chrom_off` for
    off; run_h 3.8;
  - `gap_on_s{0,1,2}`: `--split cold_drug_1 …`; run_h 4.2.
- **Pins:**
  - `model_v10c.py` 26a2d5db and `train_v10c.py` c589ab80, the guard code with C4 / C5;
  - `lincs-v10c-src` was versioned 18 Oct 11:0x UTC, after every T-B kernel was RUNNING. Status: ready.
- **New tests:** five tests **execute** a rendered kernel, with the `/kaggle` paths redirected, the GPU probe stubbed and a stub
  trainer exiting with code 3 or 0 by seed. They check:
  - a guard failure retries at seed + 10;
  - three failures stop after two retries;
  - `--start_attempt 1` runs seeds 11 then 21;
  - without `--guard` an exit code of 3 is FATAL, and with it an exit code of 1 is FATAL;
  - a retry that doesn't fit is left for a new kernel ("NO ATTEMPT LEFT").
- **Totals:** 47 tests in `test_v10c_model.py`, plus 6 (tb_read) and 3 (width screen).
- **The planned push order** (two at a time; the guard retries automatically):
  - tc s0 on/off, then s1, then s2;
  - then gap s0–s2.
  - The Kaggle weekly quota may stop this before Sat 24 Oct 05:30 IST. The T-C reading script will be written and committed
    before the last T-C run exists, as with T-B.

## ASKS
1. Is §100.10's reading fair? In particular, is "does not contribute on dev" right, given that the registered failure is
   driven by a training failure? Is the post hoc view described with enough caveats?
2. The on arm collapsed at seed 0 at both widths, and off s0 survived the same batch order. Does that change anything for T-C?
   For example, should the guard's counts be reported per seed as well as per arm?
3. Are the T-C kernels right to push: their args against 100.5, their pins, and the executed-loop tests?
4. Anything to register about the T-C / gap reading before its runs exist? I plan to code it from 100.5 alone.
