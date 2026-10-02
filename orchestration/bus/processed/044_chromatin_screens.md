# PACKET 044 — GPU PLAN: the two chromatin screens from the funnel (RESULTS §92), before any E-arm data
packet_id: 044
created: 2026-10-03
repo_commit: 7f8b737
type: **PRE-REGISTRATION + GPU SPEND** (≈ 3.6 GPU-h this week): two one-seed rule-6 screens on the P2 recipe.

## A. The arms (§92.2)
- **E2:** the P2 recipe + `--ablate_epi`, a training-time mean ablation (all rows carry the dev-train mean chromatin and track mask).
  It runs on the current `lincs-v9-src` upload.
- **E1:** the P2 recipe + `--chromatin_encoding clean` (`xpert_arm.py` 3495ada): failed H3K27me3 missing, rank-normal per (cell, mark).
  - **Default path byte-identical:** on the real bundle with the dev carve, `XPertData`'s E, r, Em, ctx, X and C are identical between
    HEAD and 3495ada.
  - **Clean mode verified:** HEK293T keeps [ATAC, K27ac], VCAP [K27ac], PHH has no marks (r = 0), max |E| 3.28.
  - **The upload:** needs a new `lincs-v9-src` (that file plus `E_final_provenance.json`), uploaded **only after P7, §88 t1 and t2 have
    started**.
- **Kernels:** `orchestration/make_e_kernels.py` generates `kern_v9dev_e2_s0` and `kern_v9dev_e1_s0`, with the V2 dev kernel's
  guards plus one guard per arm.
- **Guard dry-runs** (exec'd against staged uploads):
  - E2 passes on the current upload;
  - **E1 refuses the current upload** (missing `chromatin_encoding`, `E_final_provenance.json`, `rankdata`), so a stale mount cannot
    silently run E2-like code;
  - E1 passes on the staged new upload.

## B. Rules (§92.3; §85.2 unchanged, baseline P2 μ0 0.43693, s0 0.00169)
- **Rule 6 (seed 0):** Δ ≥ 0.0034 → seeds 1–2; Δ < s0 → dropped; otherwise seed 1, then re-read.
- **Accept (3 seeds):** rule 7 + Δ_centred > 0 + rule 8 + the in-cell rule.
- **E1 vs E2:** if E2 is accepted, E1 must also beat E2, by a seed-paired Δ ≥ max(0.003, 2√(s_E1²/3 + s_E2²/3)) in ≥ 4 of 6 cells.

## C. Spend and order (§92.5)
- **Cost:** ≈ 1.8 GPU-h per seed-0 screen.
- **Week total:** P7 + t1 + t2 + E2 s0 + E1 s0 ≈ 24.5 of 30 GPU-h.
- **Order:** E2 s0 takes the slot P7 frees (≈ 12:15 IST); E1 s0 follows the new upload. Seeds 1–2 for an advancing arm only if
  quota remains.

## ASKS
1. Is anything in §92 wrong or missing? In particular:
   - the E1-vs-E2 comparator;
   - whether rule 8 is meaningful for E2, given that the pathway readout's input changes;
   - whether E2's `--ablate_epi` mean (all dev-train rows, including uncovered cells' zeros) is the right "no chromatin" arm.
2. Is the upload sequencing safe, and do the guards protect the right things?
3. Clearance to push E2 s0 when P7 frees a slot, and E1 s0 after the new upload.
