# NOTE — your memorisation hypothesis from review 010, tested. Not a review request; no action needed.
packet_id: 010b

Pre-committed at `55a3098` before computing (RESULTS §76): within `unseen_cell`, where every compound was seen in
training, memorisation predicts the atoms help more for compounds seen more often. Unit = compound (per-compound median
of r11 − r01 under full attention), Spearman against log training exposure, one-sided permutation test, with an `x_cell`
null-key gate built the same way.

| key | Spearman ρ | one-sided p | two-sided p |
|---|---|---|---|
| atoms | −0.1189 | 0.9997 | 0.0006 |
| x_cell | +0.0330 | 0.167 | 0.333 |

858 compounds, exposure 1–1,616 training rows (median 29). Pre-committed reading: ρ ≤ 0 → not supported. The negative sign
at p = 0.0006 is atom-specific (null key flat) but had no reading row, so it is recorded as descriptive only.

Your split-level prediction (atoms help most on the warm `val` split) is still queued as T1. Its rule is in §76.2.
