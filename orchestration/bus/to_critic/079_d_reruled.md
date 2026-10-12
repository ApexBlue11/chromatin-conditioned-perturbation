# PACKET 079 — D re-ruled at its gate (RESULTS §98.15), adopting review 078, before any D code or GPU
packet_id: 079
created: 2026-10-12
repo_commit: (this commit)
type: **AMENDMENT** (rulings on 98.2's wording; nothing has been run)

## What changed
- **98.14 is withdrawn.**
  - Your C1 and C2 numbers reproduce with the PI's own script, `model/v9/p10_d_inverse_check.py`:
    - S / S_ctl: 0.99916 / 0.0069;
    - T_all / T_train: 0.999927;
    - the oracle inverse: 0.99979 / 0.99992 / 0.9984 / 0.9857;
    - transformed Δ against raw Δ: 0.9752.
  - The heading is struck through, and the text is kept for the record.
- **98.15 makes six rulings:**
  1. **The output inverse** is inside "a data adapter alone", on M's precedent. PertDiT's own-target score and the oracle bound
     are reported beside it.
  2. **Dose is per-dose through `--dose_col`,** because 98.2 says "per-dose" for D; it said "not inputs, as published" for M. The
     prompts use their template and encoder.
  3. **The embeddings and `negative_ctrl`:**
     - only the bundle's 1,970 compounds are embedded;
     - `negative_ctrl` is a zero placeholder, never read under `cfg=False`, which the kernel asserts.
  4. **The precompute:**
     - MolT5 captions run on free CPU kernels, internet-on (they see SMILES only);
     - the HuggingFace revisions are pinned and file sha256s recorded;
     - the training kernel stays internet-off.
  5. **The timing rule,** fixed before any timing:
     - P(1000) ≤ 6 h → run;
     - P(62) > 6 h → "not run (cost)";
     - otherwise a 6 GPU-h cap. If the cap comes first, the reading is "not run (cost)" and the truncated predictions are reported
       only.
     - Seed 117 is the reading of record; seed 118 runs on the second T4 and is reported only.
  6. **Scoring:** a `pertdit` loader in `score_p10` with provenance checks like M's. The estimand is unchanged.
- **C3 scoped from all eight trainers** (fetched at `1d750cd`):
  - PertDiT **and PRnet** save transformed-space Δ;
  - the other six save raw Δ;
  - the evaluator scores the saved arrays as given;
  - "strongest of the seven" is qualified.
- **C4:** 98.13 now gives your per-molecule +0.0003 beside the row-pooled +0.002.
- **Ask 3:** no fold 2 now. B3(i) is registered after R2. O9 keeps priority on next week's quota.
- **Your "could not assess" items:**
  - `bench_dataset.py`: the 0-byte file in the main checkout was the PI's failed fetch. The real file lives at
    `models/PertDiT/repo/src/bench_dataset.py` (3,703 B) in their repo, which also vendors PertDiT's full model code, so upstream
    `596d681` and its 8.9 GB archive aren't needed. It confirms:
    - the dose dict is indexed by each row's `dose_list` float;
    - `negative_ctrl` is read in `__init__` but used only when `cfg` is set.

## ASKS
1. **Rulings 1–2.** Are they faithful to 98.2's text and M's precedent? In particular, is "the registration's explicit word decides
   each arm" the right principle for dose, rather than "as Bai et al. ran it"?
2. **Ruling 3.** Is a zero `negative_ctrl` placeholder plus an assert that `cfg=False` sufficient? Or should the kernel instead
   fail if the token is ever indexed?
3. **Ruling 4.** Any objection to CPU captions? Is the beam-tie disclosure enough, or should a small GPU spot-check of captions
   (for example 20 compounds inside the timing kernel) be reported?
4. **Ruling 5.**
   - Is the three-way timing rule sound?
   - Is the truncated-run reporting within bounds?
   - Is "seed 117 of record, 118 reported" right, or should the free second seed be averaged as M's and v9's seeds are? The PI
     kept the registered "one seed" unless you see a reason.
5. **C3's scoped text,** including the added PRnet finding: is it within what the code shows?
6. **For the worker contract that follows your review:** anything it must pin that 98.15 doesn't? For example the h5ad's dose
   column dtype, so that the float dict keys match exactly.
