# PACKET 077 — RESULT: R1, v9 against Bai et al.'s drug-free MLP on unseen compounds (RESULTS §98.11)
packet_id: 077
created: 2026-10-12
repo_commit: (this commit)
type: **RESULT** (mechanical, by 98.6 / 98.7; scorer committed in 23d1f75 before the run)

## The reading
- **Of record (clean, 346 molecules):** −0.0038 [−0.0086, +0.0016]; 96 molecules favour v9 and 250 favour M. The verdict is NO
  COMPOUND-LEVEL CLAIM; the reader's case is mean ≤ 0, so *"v9 does not beat a drug-free model on unseen compounds (benchmark
  statement only)"*. R2 is pending.
- **Per seed:** −0.0040 / −0.0038 / −0.0034. Row-pooled: v9 0.642, M 0.635. Full split: +0.0045, NO CLAIM.
- **Reported, never read:**
  - **F0:** 0.391 row-pooled (your number, reproduced); v9 beats it by +0.241, 344 of 346.
  - **The distinct-control stratum:** **+0.044 [0.033, 0.056], 208 of 267 molecules favour v9.**
  - **The noise-free target:** −0.0049 [−0.0075, −0.0022], with M better.
  - **Cell-centred:** −0.0004, NO CLAIM.
- **The freeze and the disclosure** are repeated beside the result, as you asked.

## Provenance
- **The run:** M v2, three seeds, with GUARD 2 / GUARD 3 in each log; early stops at epochs 291 / 298 / 259.
- **The download:** logs first, then each npz and record, then `P10_M_COMPLETE.json`. `--dry_check` passed (manifest, the
  records' pins, arm / seed / n_test, P9's guards).

## ASKS
1. Is the reading mechanical and correctly worded? In particular:
   - NO CLAIM with 250/346 against v9 is reported as "does not beat", not "M beats v9";
   - the row-pooled sign is v9's;
   - is the sentence "not a margin over the strongest simple baseline" licensed by R1 alone?
2. **The distinct-control pattern** is described as a reported block with a post hoc reading (shared plate controls let M recall
   training responses) and labelled untested. Is that wording within bounds?
   - Is the suggested confirmatory design right: another cold-drug fold, with the distinct-control rows as the primary unit?
3. **The noise-free target favours M.** Is it right that it can't separate the shared-plate route from compound use, because it
   keeps the shared plate component?
4. **Manuscript §5.5's new bullet:** within the record?
5. **Anything to add to B1's reading** (R2) now that R1 is known, without changing its registration (the freeze)?
