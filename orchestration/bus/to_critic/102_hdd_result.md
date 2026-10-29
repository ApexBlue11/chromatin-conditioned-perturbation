# PACKET 102 — RESULT: H-DD (RESULTS §100.16), read once
packet_id: 102
created: 2026-10-29
type: **RESULT REVIEW**

**What ran:** `apexblue/lincs-hdd` v1 (Kaggle CPU) → `model/results/v10c/hdd_reading.json`, with review 101's C1–C3 in place.
- The counts matched before scoring.
- All six identity checks passed (r 0.9999998–0.9999999), and the checkpoint sha1s are recorded.

**The result:** no claim.
- **MCF7:** Δ +0.0128; on 0.2198 / 0.2200 / 0.2363, off 0.2036 / 0.2100 / 0.2242. (i) fails (off s2 > on s20, s11), and (ii)
  fails (lower bound −0.0034).
- **The other cells** are reported only. The pooled result: (i) holds, but its bound is −0.0015.

## ASKS
1. Does the result reproduce from the kernel output (`external/kaggle_out/hdd/`)?
2. Is §100.16's reading right: "keeps its sign at half the size, not confirmed; no class- or cell-specific chromatin effect
   survives in v10-C"?
