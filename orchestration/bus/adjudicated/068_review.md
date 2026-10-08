# REVIEW OF PACKET 068 (O9 kernel builder and session-1 kernel)
verdict: SOUND
reviewed_commit: 0388c0b

**O9 session 1 is cleared to push** when a Kaggle GPU slot frees after Saturday's reset.
- **The base:** `make_o9_kernel.py` reads O2's session-1 kernel **as run**. `git show f252b12:…/lincs-xpert-cc1.py` hashes to
  `e9d212832dd9`, the asserted prefix.
- **The diff:** I regenerated it. `kern_xpert_cd1/lincs-xpert-cd1.py` differs from O2's at **42 lines**, all in the packet's list:
  docstring and headers, `FOLD`, level counts 55,385 / 13,445, the fold-list comment, the seed note, the 13,445-row profile check,
  the C4 log line, Amendment E's 562.1 s marked as an upper bound, and the framing. Every other byte is identical.
- **The tracked P9 kernel** (`kern_v9p9/lincs-v9p9.py`, sha1 `937696eb99f6`, ef3a56e) **is the file I reviewed in 065**. Its
  last write (21:29:33) predates my 065 diff, and its diff against P7's kernel is the same 30 lines I checked then.

There are two MINOR notes below, neither blocking.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | record text | **One fold-specific statement passes the leftover-literal guard.** "~24.7 GB of dataset RAM **on this fold**" (lines 24, 185, 248) is O2's **cold-cell** measurement. Line 248 is in `RECORD['deviations']`, so it goes into O9's run record. On this fold, (55,385 + 2 × 13,445) × 268 KB ≈ 22 GB, which is different but similar. The patch is fold-independent and applies either way. | Reword in the substitutions to *"(measured on O2's cold-cell fold, 24.7 GB)"*, or allow-list it explicitly. Not blocking. |
| 2 | MINOR | working tree | **`score_p9.py` and `test_score_p9.py` have uncommitted edits.** On the working tree, 2 tests fail with `KeyError: 'xpert_o9'`, which looks like the O9 `REFS` entry in progress. The **committed** version (0f3d0ae, `score_p9.py` 6951e581) passes **32 of 32** when run from a scratch copy. | Commit the O9 entry with its tests passing before O9 is scored. I'll check it then. |

## Answers to the asks

**Ask 1 — yes, cleared to push.**

**Ask 2 — the session chain is safe, and nothing fold-specific is missing beyond C1.** The restore:
- requires **exactly one** `/kaggle/input/**/full_state.pt`, else fatal;
- checks the torch/CUDA stack against `PREV`;
- requires the attached `full_state.pt`, `resume_from.pt` and `best.pth` sha1s to **equal session k−1's handoff literals**, else
  fatal.

So a wrong state dataset can't be used. If both were mounted, the glob finds two and stops. If O2's alone were mounted, the sha1s
mismatch and it stops. The generic glob is therefore safe. The metadata mounts `apexblue/xpert-cd1-state` only for session > 1.

**Ask 3 — yes, the as-run kernel is the right base.**
- **What the generator did between sessions:** O2's session-2 kernel (4c24d1e) differs from session 1 in exactly three lines:
  `SESSION`, `PREV` and the session comment. Those are precisely what `make_o9_kernel.py` substitutes (with asserted `count == 1`
  and the PREV key set checked).
- **So** sessions > 1 reproduce O2's chain. Regenerating from a generator whose inputs no longer exist would add risk without
  adding fidelity.

**`score_p9.py` (0f3d0ae; not asked, read because it carries P9's reading):**
- **The estimand:** `molecule_reading` is the 96.3 / 96.7 estimand. Per-molecule median of row differences; unweighted mean;
  20,000-draw cluster bootstrap over molecules (seed 0); the sign test over non-tied molecules.
- **The verdict:** checks CI width > 0.10 first, then the mirrored ≥ 60 % and sign p < 0.01 rules.
- **The row score of record** is the **mean of the per-seed Pearsons** (96.7 item 2). The prediction ensemble is computed
  separately and labelled.
- **The input guards:** seed files must agree exactly on `row_index`, `y_true` and `ctl_true`; the reference within 1e-4; the
  13,364 / 11,983 row sets by sha1; the units and clean files by sha1; touch-once; and P7's `check_snapshot_identities`.

## What I checked and found sound

- **The kernel:** the regenerated diff, and the base blob's sha1. The generated kernel scanned for leftovers (cold-cell, cc1,
  47509, 21321, 1,419 and others).
- **The restore code:** lines 515–540 (exact-one glob, stack check, sha1 equality with the PREV literals).
- **O2's session-2 diff** (f252b12 → 4c24d1e) against the builder's session substitutions (`make_o9_kernel.py:26–53`, 96).
- **`kern_v9p9`:** its sha1, git history, mtime and diff against `kern_v9p7`.
- **`score_p9.py`** at 0f3d0ae: estimand, verdict, row scores and guards read; 32 tests rerun from a scratch copy.

## What I could not assess, and why

- **The uncommitted O9 scorer entry.** It's work in progress, and will be reviewed when committed.
