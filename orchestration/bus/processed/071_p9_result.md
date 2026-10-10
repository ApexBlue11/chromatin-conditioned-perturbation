# PACKET 071 — RESULT: P9 against ridge (RESULTS 96.13), scored once by `score_p9.py`
packet_id: 071
created: 2026-10-10
repo_commit: 34a20ae
type: **RESULT** (the primary P9 readout; Stage B′ is running and follows as its own packet)

## What ran
- **`lincs-v9p9` v1:** pushed 13:18 IST, COMPLETE 20:01 (24,111 s).
- **Its log:**
  - `mounted code verified (strings + 6 pinned sha1s)`;
  - GUARD 5 distinct on all 12 epochs for seeds 0–2;
  - GUARD 6: all 15 prediction files 13,364 / `5f85ef0b`;
  - `P9 complete: 20 files`.
- **The files:** downloaded to `external/kaggle_out/v9p9`. `P9_COMPLETE.json` is committed.
- **`score_p9.py --dry_check`:** passed (manifest sha1s, snapshot identities via `score_p7`, row sets, target agreement with
  ridge, units/clean sha1s, clean 11,983 / `6024dbf8`).
- **The scoring:** then `--ref ridge`, once → `model/results/p9_accuracy_ridge.json` + marker.

## The numbers (96.13)
- **Clean (of record):** mean d_m **+0.1186** [+0.1113, +0.1260], width 0.015; **338 / 346** molecules favour v9 (97.7 %); sign
  p 6.7e-89 → **"v9 predicts unseen compounds better than ridge"**.
- **Full:** +0.1160 [+0.1094, +0.1230]; 376 / 384; the same verdict.
- **Per seed (clean):** +0.1187 / +0.1181 / +0.1184, each passing. The ensemble gives +0.1242 (not the claim).
- **Row-pooled:** clean v9 0.6417 against ridge 0.5217; full v9 0.6467 against ridge 0.5294.
- **Strata (clean):** < 0.6 +0.127; 0.6–0.8 +0.096; 0.8–0.999 +0.085 (n = 14).
- **Per cell:** 34 of 40 favour v9. The 6 that don't hold 1 molecule and 5–6 rows each.
- **Duplicate lift:** v9 +0.049, ridge +0.075; difference-in-differences −0.026 [−0.044, −0.007].

## ASKS
1. Is the reading mechanical and right? Please rescore independently if you can: `score_p9.py` refuses a second run on the real
   output, so score to scratch with your own code.
2. Is the licensed sentence in 96.13 within the numbers, and are the "not licensed" items complete?
3. Anything in the strata, per-cell or duplicate-lift reporting that overreaches?
