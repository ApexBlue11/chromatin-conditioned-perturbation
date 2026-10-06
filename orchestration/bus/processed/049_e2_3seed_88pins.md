# PACKET 049 — RESULT: E2 at three seeds (RESULTS 92.10), and §88's t1/t2 pins and probes
packet_id: 049
created: 2026-10-06
repo_commit: 832f270
type: **RESULT + PROCEDURE CHECK** (no GPU; the §88 probes run on free Kaggle CPU)

## A. A gap in the loop, disclosed
- **What happened:** the PI's session-scheduled checks died with the earlier session on 3 Oct. Nothing was read between 3 Oct
  ~19:00 and 6 Oct 06:15 IST.
- **Effect on order:** none. E2 s1, t1 and t2 had all COMPLETED by the time they were read, and nothing was pushed, re-run or
  re-read in the gap.

## B. E2 at three seeds (92.10; 92.9's procedure)
- **Seeds 1–2:**
  - mount `60bdcd48`, GUARDs 4 and 5 OK;
  - `score_dev --centred --preds <E2 ×3> --baseline <P2 ×3>`.
- **Rule 7:**
  - Δ **+0.0073** ≥ bar 0.0036 (E2 sd 0.0026);
  - mean of per-cell means +0.0095;
  - **cells 3 of 6**, so the rule fails.
- **Other conjuncts:**
  - centred +0.0045;
  - rule 8: 0.2891 (0.2979 / 0.2866 / 0.2830);
  - in-cell 6 of 6.
- **Verdict:** **NOT ACCEPTED.**
- **Per cell:** HEK293T +0.0217, LNCAP +0.0365 and VCAP +0.0220 gain; U937 −0.0055, SKBR3 −0.0077 and HL60 −0.0123 lose. This is
  T4's split.
- **Decision table (92.9):** the reading is now set by E1 alone, and the E1-vs-E2 comparator no longer binds.

## C. §88 (88.6 items 1, 11)
- **The trained runs:** t1 and t2 COMPLETE. Both logs show the last epoch is 11, GUARD 3's architecture vs the screened C8b is
  OK, and distinct seeding holds.
- **The checkpoints:** sha1s from the logs are t1 `9a6e8a81…` and t2 `aa82e17d…`. They were verified against the downloaded
  files and **pinned in `read_moa_88.py` at 832f270, before the probes ran**.
- **The probe kernels:** `kern_moa88_pt1` and `pt2` were generated from pt0. The diff is the seed, the checkpoint name, the sha1
  and the kernel source. They were pushed to Kaggle CPU after the commit.
- **The new upload:** the probes mount the new `lincs-v9-src` (3 Oct). `probe_moa_88.py` there is byte-identical (`66dd9531`) to
  the version pt0 and pu0–pu4 ran on, and it does not import `xpert_arm.py`, the only code file that changed.

## ASKS
1. Is 92.10 read mechanically? Is its "what it suggests" line licensed?
2. Is anything missing in the §88 procedure before the reader runs, once pt1 and pt2 land?
