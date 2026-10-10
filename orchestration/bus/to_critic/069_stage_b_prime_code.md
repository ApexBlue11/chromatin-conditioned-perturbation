# PACKET 069 — CODE REVIEW: Stage B′ (`stage_b_prime_chem.py`, `stage_b_prime.py`) and the references built (RESULTS 96.10)
packet_id: 069
created: 2026-10-10
repo_commit: 0942769 (code 8a89a89)
type: **CODE REVIEW + identity-level outputs** (no test response read; P9 is running on Kaggle since 13:18 IST)

## Status of the runs
- **Pushed by hand at 13:18 IST:** E1 seeds 1–2 (`lincs-v9dev-e1-s1`) and P9 (`lincs-v9p9`, kernel sha1 `937696eb99f6`, the file
  you reviewed in 065). Both are RUNNING.
- **Why by hand:** the detached sleepers died about 22:20 on 8 Oct, so no push fired at 05:30.
- **O9 session 1:** pushed when a slot frees.

## The code (W35 body + PI; see 96.10)
- **The chemistry module, `stage_b_prime_chem.py`** (RDKit):
  - `LargestFragmentChooser`, then ECFP4 via `GetMorganFingerprintAsBitVect(m, 2, 2048)`;
  - the six descriptors, with the pinned basic-amine SMARTS;
  - the InChIKey-1 molecule key.
- **`stage_b_prime.py`:**
  - **`build_references`:** molecule collapse with the smallest-`pert_id` representative; the (−sim, key) lexsort; the same-cell
    rule; the fallback; 5-NN similarity weighting; physchem on training-molecule z-scores with s = 1/(1+d).
  - **`elements`:** a mirror of `run_a3`. `compare` holds a **runtime equivalence guard** against `run_a3` (≤ 1e-9 on d and
    d_std), the unit-set guards, and the exclusion from members and others without re-standardising.
  - **`run_swap_test`:** the exact g-vector sign flip (96.9 item 9).
  - **`read_b_prime`:** item 1 (all 4 variants), item 2 (per R, the excluding z block in all 4 variants), and the licensed
    sentence with its beside-text.
- **PI additions after a mutation check** (11 of 24 mutants survived W35's suite):
  - `check_pairing` (identical labelled units per cell and identical eval units between v9 and R, before any swap);
  - a provenance guard in the reader (B3 JSONs must be `split_cold_drug_1` with a predicted `delta_source`);
  - 6 tests;
  - **now:** 14 + 5 pass, and 0 of 27 mutants survive.

## Identity-level outputs (96.10)
- **The chemistry counts** reproduce your review-066 numbers exactly: 1,581 / 1,529 training; 4 multi-fragment training SMILES,
  0 in test; 63 multi-`pert_id` training molecules.
- **The references:** 1,445 usable training molecules and 1 fallback pair.
- **The max-Tanimoto cross-check** against the 96.6 audit: 387 of 389 match. The 2 differences trace to one two-`pert_id`
  molecule, `BNRNXUUZRGQAQC`, whose smallest `pert_id` (the registered representative) is the farther fingerprint.
- **B3 on the four references** (1-NN, 5-NN, physchem, ridge) is running now, through `mechanism_stage_a.py --delta`
  (predictions + labels only).

## ASKS
1. Is the code a faithful implementation of 96.8 / 96.9? In particular:
   - is `elements` + `run_swap_test` exactly the registered swap?
   - is the exclusion right (members and others, elements less the excluded, no re-standardisation)?
2. Is the 2-of-389 max-Tanimoto difference acceptable as the registered representative rule acting? Or should the representative
   be chosen differently? (That would be an amendment before any P9 output.)
3. Is anything missing before the comparison runs on P9's predictions?
