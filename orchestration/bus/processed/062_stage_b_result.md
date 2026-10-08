# PACKET 062 — RESULT: §94 Stage B read once (RESULTS 94.11)
packet_id: 062
created: 2026-10-08
repo_commit: 5832d97
type: **RESULT** (local CPU; no GPU)

## A. What ran
- **The runs:** `mechanism_stage_a.py` 6602e762 (cleared by review 061a), seven sequential local runs.
  - **measured rerun:** asserted equal to 94.8 on every A1 field and on A3's T, p, frac and d; adds `d_std`;
  - **v9 seeds 0, 1, 2:** `deg_pred`;
  - **v9 seed-mean;**
  - **μ:** `mean_drug_delta_pred − ctl_true`;
  - **ridge:** `ridge_pred − ctl_true`.
- **Every run** ended rc 0, with a marker.
- **The read:** `--read_b` once → `stageB/stage_b_reading.json` (1645eb9b).
- **Input sha1s:**

  | file | sha1 |
  |---|---|
  | measured | 62f7d676 |
  | v9_seed0 | 76f4c109 |
  | v9_seed1 | 2042e870 |
  | v9_seed2 | 6d44282f |
  | v9_seedmean | 2772235f |
  | mu | e7ad047d |
  | ridge | 1cf86cc4 |

## B. The numbers
**A1** (MCF7 / HT29 / MDAMB231 / HS578T / THP1):

| source | A1 |
|---|---|
| measured | 0.653 / 0.598 / 0.646 / 0.671 / 0.665 |
| μ | 0.661 / 0.651 / 0.741 / 0.735 / 0.715 |
| v9 seed-mean | 0.675 / 0.647 / 0.751 / 0.699 / 0.709 |
| seed0 | 0.672 / 0.644 / 0.753 / 0.702 / 0.722 |
| seed1 | 0.672 / 0.643 / 0.754 / 0.692 / 0.720 |
| seed2 | 0.678 / 0.656 / 0.741 / 0.699 / 0.694 |
| ridge | 0.702 / 0.646 / 0.736 / 0.742 / 0.674 |

**v9 − μ** (A1):

| run | MCF7 / HT29 / MDAMB231 / HS578T / THP1 |
|---|---|
| seed-mean | +0.014 / −0.004 / +0.010 / −0.036 / −0.006 |
| seed0 | +0.011 / −0.007 / +0.012 / −0.033 / +0.007 |
| seed1 | +0.011 / −0.007 / +0.013 / −0.043 / +0.005 |
| seed2 | +0.017 / +0.005 / −0.000 / −0.036 / −0.021 |

**B3 T (gated):** measured 3.92; μ 8.23; v9 seed-mean 5.81, seeds 5.75 / 5.50 / 5.68; ridge 7.71. B1 and B3 carry signal for
every v9 run and for μ.

**B3c** (of record, standardised d; reported, raw d):

| run | of record (p) | raw (p) |
|---|---|---|
| μ | 0.718 (0.013) | 0.355 |
| v9 seed-mean | 0.806 (0.002) | 0.150 (0.358) |
| seed0 | 0.784 (0.004) | 0.120 (0.392) |
| seed1 | 0.860 (0.0005) | 0.154 (0.355) |
| seed2 | 0.728 (0.007) | 0.135 (0.372) |
| ridge | 0.637 | — |

**The readings:**
- **1:** TRUE, with μ also expressing mechanism.
- **2a:** FALSE.
- **2b:** TRUE, on the seed-mean and every seed.

## ASKS
1. Is the reading mechanical?
2. Is 94.11's sentence for 2b licensed with its beside-text? The beside-text says:
   - the margin over μ is small, with no null for the difference;
   - μ reaches 0.72 through composition;
   - raw B3c reverses (v9 0.12–0.15 against μ 0.355);
   - the sd includes the class members.
3. Is the "What Stage B says" paragraph licensed? In particular, the sentence *"nothing here shows the model adds
   mechanism-relevant, cell-specific information robustly beyond what the drug's average response already carries"*, given
   that 2b passed.
4. For the paper: what is the shortest licensed statement of §94 (Stages A and B)?
