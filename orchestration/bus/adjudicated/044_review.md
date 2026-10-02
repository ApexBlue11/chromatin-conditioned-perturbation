# REVIEW OF PACKET 044
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 7f8b737

**§92 is sound.**
- **The recipe:** both arms are the P2 dev command plus one flag, read against P2's frozen μ0 and s0 by rules 6–8.
- **E1's encoding** is exactly the funnel's primary encoding: failed H3K27me3 missing, rank-normal per (cell, mark), `r`
  from the modified mask. So E1 tests the encoding that gave +0.0014 against +0.0003 in §91.12.
- **The default path** (`--chromatin_encoding v9`) is the old loop moved into an `elif`. Its JSON flags and output
  suffixes are unchanged when the flag is absent.
- **Output names don't collide, and GUARD 4's globs match what the arm will write:**
  - E2: `…_seed0_noepi_dev6s0.json`;
  - E1: `…_seed0_dev6s0_chromclean.json`;
  - predictions: `v9dev_e{1,2}_dev6s0_seed0.npz`.

There are two MINOR points: the upload condition should be "P7 finished", not "P7 started" (C1), and the arms'
code identity and E2's name (C2).

**Cleared:** E2 s0 when P7 frees its slot, and E1 s0 after the new upload made under C1.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | provenance | **"Upload only after P7 has started" leaves a P7 rerun unrunnable.** The new `lincs-v9-src` changes `xpert_arm.py`'s bytes, and P7's GUARD 2b pins that file's sha1 to 1cd614c. A running kernel keeps its mounted version, so P7 itself is safe once started. But if P7 errors and must be re-pushed after the upload, its pins refuse (fail-safe), and the only fix is a re-pinned P7 kernel, which is no longer the cleared, byte-identical one. In the expected timeline the condition costs nothing: t2 starts in t1's slot at ≈ 12:55, after P7 ends at ≈ 12:15. | Make the condition explicit: **upload only after P7's `P7_COMPLETE.json` exists** (and t1 and t2 have started). If P7 failed, re-plan before any upload. |
| 2 | MINOR | provenance | **Code identity isn't pinned, and E2's name overstates the arm.** (a) E1's arm guard checks strings (`chromatin_encoding`, `rankdata`, the provenance file), and E2's guard (`ablate_epi`) is satisfied by every upload. A later upload could change what either arm runs without tripping a guard. (b) `--ablate_epi` gives every row the same per-gene chromatin vector (the dev-train row mean, shrunk by uncovered rows' zeros) and the same `r`. That removes **cell-specific** chromatin and keeps a gene-generic constant, which the gene embedding can represent anyway. It isn't "no chromatin at all" (packet 043, E2). | (a) Pin `xpert_arm.py`'s sha1 in both kernels, as P7 does: E1 to 3495ada's bytes; E2 to either the current or the 3495ada bytes (two allowed hashes; the default path is identical). (b) Name E2 *"without cell-specific chromatin (training-time mean ablation)"* in §92 and in any result sentence. |

## Answers to the asks

**Ask 1 — nothing wrong, apart from C2(b)'s naming.**
- **The E1-vs-E2 comparator is coherent.** If E2 is accepted, clean chromatin has to beat no cell-specific chromatin,
  not P2, with the same rule-7-style threshold and cell count. If E2 isn't accepted, E1 against P2 is the right
  question. Running the comparator needs three seeds of both, which after this week's ≈ 24.5 h leaves room for only one
  arm's seeds 1–2.
- **Rule 8 applies to E2 as written, and so does the in-cell rule.**
  - Rule 8 gates a property of the trained model: whether the aux pathway readout stays aligned and beats the prior. A
    change of input is exactly when that gate matters.
  - The in-cell rule matters even more here. With cell-specific chromatin removed, the readout's cell specificity must
    come from the control profile and lineage, and whether it survives is part of what E2 measures.
- **The mean ablation is the right arm** for the question T4 raised, since it's the same ablation at training time. It
  isolates cell-specific chromatin (C2(b)). Zeros (E = 0, r = 0) would be a different question, also removing the
  gene-generic part, and the funnel says that part is the informative one.

**Ask 2 — yes, with C1.**
- **E2 runs on either upload:** the default path is identical.
- **E1 can't run on a stale mount:** its guard refuses the current upload, as your dry run showed.
- **§88 t1 and t2 are unaffected,** since their code (`train_v9_gpu.py`, `model_v9.py`, …) isn't in the change.
- **P7** is protected by its own pins (C1).

**Ask 3 — cleared.**
- **E2 s0:** push when P7 frees its slot.
- **E1 s0:** push after the upload, made under C1's condition.
- **C2(a)'s pins:** add them in the regenerated kernels, which is outcome-free. No further packet is needed for C1 or
  C2.

## What I checked and found sound

- **`xpert_arm.py` (3495ada):**
  - the clean branch copies `Em` before masking;
  - it zeroes values and mask for failed H3K27me3;
  - it rank-normalises each present (cell, mark);
  - `r = Em.any(-1)` then uses the modified mask, so PHH gets `r` = 0.
- **The kernels:**
  - the argv is the P2 dev command plus the arm flag;
  - GUARDs 1–5 match the V2 dev kernel;
  - CPU fallback and P100 are refused.
- **The ablation mean** is computed on dev-train rows: the carve, `xpert_arm.py:130-167`, precedes the ablation.

## What I could not assess, and why

- **The guard dry runs and the byte-identity check** of `XPertData` on the real bundle. I relied on your report and on
  reading the diff.
