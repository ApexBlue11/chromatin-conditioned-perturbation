# PACKET 051 — RESULT: §88 read once (RESULTS 88.8): NULL
packet_id: 051
created: 2026-10-06
repo_commit: cc29c25
type: **RESULT** (no GPU; the probes ran on Kaggle CPU)

## A. What ran
- **pt1 and pt2:** pushed after 832f270's pins. Their logs show the pinned sha1s (`9a6e8a81…`, `aa82e17d…`), and both probes
  exited 0.
- **The read:** the 8 probe JSONs were assembled as `external/kaggle_out/moa88_read/{t0..t2,u0..u4}`, then
  `read_moa_88.py` ran once → `model/results/moa_88_reading.json`. No probe JSON was opened before.

## B. The reading
- **Validity:** every check true; no void probe.
- **The untrained calibration:** all-rows m_u +0.0006, sd_u 0.0144, so the bar is −0.0282.
- **Per seed:** seed 0 −0.0210 (p 0.013), failing the 2-sd condition only; seed 1 −0.0011; seed 2 −0.0147 (p 0.042).
- **Verdict:** **NULL.**
- **Permitted wording:** *"does not align with annotated drug mechanism beyond five untrained initialisations"*. Seed 0's
  alignment is reported, not read.
- **Combined with §86.4 and C 4.1a:** no readout recovers drug mechanism beyond calibrated nulls. The cell-level readout
  (§85.14) is what holds.

## ASKS
1. Is 88.8 read mechanically? Is the "reported, not read" seed-0 sentence licensed?
2. The combined sentence ("none of the three ways … recovers annotated mechanism beyond its calibrated nulls"): is it
   licensed for the paper, and how should it read?
