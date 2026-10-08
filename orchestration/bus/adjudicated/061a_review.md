# REVIEW OF PACKET 061, ADDENDUM: clearance check on bbefd48 (94.10)
verdict: SOUND
reviewed_commit: bbefd48

**Cleared for the Stage B runs** (measured rerun, v9 seeds 0/1/2, the seed-mean, μ and ridge) and one `--read_b`.
- **Code and tests:** `mechanism_stage_a.py` = 6602e762. I reran `test_mechanism_stage_a.py` under `.venv-cuda`: **24 passed**.
- **The fixes:** review 061's C1–C5 are implemented as 94.10 records.

There are two optional notes below, neither blocking.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | construct (optional) | **The z-score's sd includes the class's own members.** In a cell where a class responds strongly, its members widen that pathway's sd and so compress their own `d_std` somewhat. It applies to measured and predicted alike, and with 48–466 labelled compounds per cell against 3–16 members, it's modest. It slightly flattens across-cell contrasts rather than creating them. | Optional: standardise by the sd over the cell's **non-member** labelled compounds per unit. If not, leave it as is and note it in 94.10. |
| 2 | MINOR | wording (optional) | **With standardised d, 2b's licensed sentence is most exact as a relative statement.** | *"…predict in which cells a pathway inhibitor's pathway effect stands out most, relative to that pathway's spread across the cell's compounds."* |

## Answers to the asks

**Clearance: yes.**

**C1, standardised d:**
- **What `run_a3` does:** it z-scores each pathway's ULM activity within each cell over the cell's labelled compounds (ddof 1;
  a zero sd → NaN), recomputes d exactly as registered (`d_std_per_unit`), and records each cell's per-pathway mean and sd.
- **What `read_stage_b` uses:** `d_std_per_unit` for B3c of record on both sides, with raw B3c reported.
- **A3's own reading** stays on raw d, as Stage A registered.
- **The scale-invariance test** is present and passes.

**C5, the baselines in expression space, confirmed at source:** `xpert_mdmt_baselines.py` lines 11–13 and 155 build:
- `mean_drug_delta` = `x_ctl` + the compound's mean Δ over **training** rows (the global mean if unseen). So
  `mean_drug_delta_pred − ctl_true` is exactly the compound's training-row mean Δ: one vector per compound, **with no
  cell-specific information**, as μ must be.
- `ridge` = `x_ctl` + W[x_ctl, ECFP4, descriptors, log dose, time]. Its Δ uses the cell's control as an input, which is fine for
  a reported baseline.
- **`load_predicted_delta`'s `key-ctl_true`:** it splits only the key part (after the last `:`), so Windows paths are safe.
- **Means read:** reading means only, for the scale check, is disclosed and reads no mechanism.

**C2–C4:** in place, in 94.8 and the reader's output (`reading_1_mu_also_expresses`, the reading-1 sentence, "fraction of the
measured reference", and the scope string).

**The measured rerun:** asserting that it reproduces 94.8's A1 and A3 numbers exactly is the right guard. Record the assertion's
outcome with the Stage B result.

## What I checked and found sound

- **The code:** the diff ce75800 → bbefd48 of `mechanism_stage_a.py` (`run_a3` standardisation and `activity_scale`;
  `load_predicted_delta`'s `-` key; `read_stage_b`'s B3c-of-record and reporting).
- **The tests:** 24, rerun.
- **The baselines:** `xpert_mdmt_baselines.py`'s construction of `copy_control`, `mean_drug_delta` and `ridge`.
- **The RESULTS text:** 94.10 against the code.

## What I could not assess, and why

- **The Stage B outputs.** None exist yet.
