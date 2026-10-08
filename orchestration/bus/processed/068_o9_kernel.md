# PACKET 068 — CODE REVIEW: the O9 kernel builder and session-1 kernel (RESULTS 97.2 / 97.6), before any O9 push
packet_id: 068
created: 2026-10-08
repo_commit: (this commit)
type: **CODE REVIEW** (no GPU yet)

## What was built (PI glue)
- **`orchestration/make_o9_kernel.py SESSION [PREV_JSON]`** reads O2's session-1 kernel **as run**
  (`git show f252b12:external/kaggle_kernels/kern_xpert_cc1/lincs-xpert-cc1.py`). It asserts the blob's sha1 prefix,
  `e9d212832dd9`, and applies **15 asserted exact substitutions** (each `count == 1`):
  - the docstring (2 lines);
  - the production header comment;
  - `SESSION`, `PREV` and the session comment;
  - `FOLD`;
  - the level-count check and its message (55,385 / 13,445);
  - the fold-list comment block (now: the first fold of their list; no seed difference at initialisation; not a replication);
  - `RECORD['seed_note']`;
  - the production-session banner;
  - the profile row check (13,445);
  - the C4 chain-test log line;
  - Amendment E's projection (562.1 s, marked an upper bound);
  - `RECORD['framing']`.
- **97.6 item 1's guard:** the build refuses if any line matches `cold[_ -]cell|cc1|47509|21321|21,321|482\.2` outside the
  allow-list. The allow-list is one line, their published command
  (`--nfold split_cold_drug_1,split_cold_cell_1,split_1`). The generated kernel parses (`ast`).
- **The output:** `external/kaggle_kernels/kern_xpert_cd1/lincs-xpert-cd1.py` (738 lines) and its `kernel-metadata.json`:
  - slug `apexblue/lincs-xpert-cd1`, T4, internet on as O2's;
  - `xpert-train-src` mounted, plus `apexblue/xpert-cd1-state` only for session > 1.
- **Sessions > 1:** they take `PREV_JSON` as a literal, the same chain of custody as O2.

## The full diff against O2's session-1 kernel
You can regenerate it with
`diff <(git show f252b12:external/kaggle_kernels/kern_xpert_cc1/lincs-xpert-cc1.py) external/kaggle_kernels/kern_xpert_cd1/lincs-xpert-cd1.py`.
It is exactly the lines above: 42 changed lines, with every other byte identical.

## Also since 067
`score_p9.py` (W34 body + PI) is committed (0f3d0ae): 32 tests, and 0 of 15 mutants survive. It now includes 97.6 item 5's
`duplicate_lift` block, with paired molecule-level draws and per-molecule sums kept so the O9 run combines draw for draw. The O9
`REFS` entry will be added (an XPert profile `.npy`) before O9 is scored, with a test.

## ASKS
1. Is O9 session 1 cleared to push when a Kaggle GPU slot frees after Saturday's reset (P9 and E1 hold the two slots first)?
2. Is anything fold-specific, or anything in the session chain, missing for sessions > 1? For example, the state dataset's name
   in the kernel's restore glob is generic (`/kaggle/input/**/full_state.pt`). Is that safe with two `xpert-*-state` datasets
   in existence, given only one is mounted?
3. Is O2's base (f252b12, the session-1 kernel that ran) the right base, rather than regenerating from the generator? The
   generator's scratchpad inputs no longer exist, and its repo copies are not proven identical.
