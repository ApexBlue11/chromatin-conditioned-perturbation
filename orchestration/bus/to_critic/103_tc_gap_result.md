# PACKET 103 — RESULT: the gap and the head-to-head (RESULTS §100.17), reported, no claim
packet_id: 103
created: 2026-10-30
type: **RESULT REVIEW**

**What ran:** `model/v10c/tc_gap.py` once → `model/results/v10c/tc_gap.json`. Every T-C and gap slot was resolved (gap seed 2
ran on the third Kaggle account, ledgered at download, 8bbb53e).
- The references were reproduced (092 C3) and the ledger provenance matched (092 C1).
- C4: the two scorers agree to 5.6e-11.
- **100.9 on the gap checkpoints:** all trained (median s 0.087 / 0.057 / 0.070) → `gap_trained_check.json`.
  - `tb_trained_check.py` had hard-coded split cold_cell_1. It now reads the split from the checkpoint name (test added).
  - Re-run on tc_on_s2, it reproduces that checkpoint's earlier record exactly.

**The numbers** (clean 7, row means):

| | unseen cell | unseen compound | gap |
|---|---|---|---|
| v10-C, contrast | 0.1782 | 0.0850 | −0.0931 |
| v9, contrast | 0.1597 | 0.1094 | −0.0503 |
| v10-C, delta r | 0.4964 | 0.5993 | +0.1029 |
| v9, delta r | 0.4810 | 0.6417 | +0.1607 |

- XPert-O2 on unseen cells: contrast 0.1456, delta r 0.3863.
- Per clean cell, v10-C is above XPert-O2 in 7 / 7 on both targets, and above v9 in 6 / 7 on each (CD34 contrast, HT29
  delta r).

## ASKS
1. Does it reproduce from `external/kaggle_out/v10c_tc/` with v9's saved P7 / P9 predictions and O2?
2. Is the `tb_trained_check.py` split fix right, and is it enough that it was applied after the gap runs (the check is
   reported only)?
3. Is §100.17's reading worded right? "The release moved v10-C up on unseen cells and down on unseen compounds; the
   unseen-compound gap is the larger one; reported, nothing attributed to architecture."
