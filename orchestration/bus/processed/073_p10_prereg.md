# PACKET 073 — PRE-REGISTRATION: P10, the drug-free controls on `split_cold_drug_1` and the newest published model with code (RESULTS §98), before any P10 code
packet_id: 073
created: 2026-10-10
repo_commit: (this commit)
type: **PRE-REGISTRATION** (no P10 code, data or GPU yet)

## Why now
- **The principal asked** whether XPert is old and whether a newer SOTA should be compared, if it is cheap.
- **The survey (98.1):** XPert (Nat Mach Intell, Jan 2026) is still at or near the top of the open models on L1000 in both 2026
  benchmarks: Bai et al. (May 2026) and Bison (arXiv 2609.32467, 26 Sep 2026; XPert 29.6 against State 27.2 on L1000 P1).
- **The newest model with code** is PertDiT (Quant Biol 2026), vendored in Bai et al.'s MIT harness. It scores at or below XPert
  there.
- **The real gap is §50.5's drug-free MLP**, which was required and never run:
  - it matched all seven published models on unseen compounds;
  - P9's +0.119 is against ridge, a model that *uses* the compound.

## The arms (98.2)
- **M — Bai et al.'s MLP, from their code, as published:**
  - X_ctl only → Δ; 3 × 2,048, ReLU, dropout 0.1;
  - Adam 1e-4, batch 512, at most 500 epochs, early stop on validation PCC_DEG;
  - a 10 % molecule-disjoint validation carve, `default_rng(9810)`, with its sha1 committed first;
  - 3 seeds; Kaggle CPU.
- **B1 — v9 drug-blind:** P9's command plus `--drug_blind`.
  - No input varies with the compound except dose and time: `u` is set to its training mean, and every row gets one atom token,
    the mean of all training atoms.
  - The functional test: permuting compound identities leaves the predictions unchanged.
  - ≈ 6 GPU-h, next week.
- **D — PertDiT via their harness:** only if a data adapter alone suffices **and** a one-epoch timing projects ≤ 6 GPU-h.
  Otherwise it is reported as not run.

## The readings (98.3)
- **P9's estimand and verdict rule, unchanged.** The clean subset is of record.
- **R1, v9 against M:**
  - PASS → "beats a drug-free model";
  - M at or above v9 → P9's headline is restated as a basal-profile margin, and manuscript §5.5 is rewritten;
  - otherwise "not separable at this power".
- **R2, v9 against B1:** the same rule.
- **D:** read with the same rule, labelled "our run of their harness".
- **Reported only:** ridge against M; M beside Bai et al.'s 0.637 (different data); the cell-centred versions.

## ASKS
1. **Is M "as published"?** In particular:
   - the validation carve (their 3:1:1 split has a validation set; this split doesn't);
   - dropping dose and time;
   - scoring with our `score_p9.py` rather than their evaluator.
   Should M also get a dose/time-aware variant, as a reported row?
2. **Is B1's `--drug_blind` specification complete for v9?** You know `xpert_arm.py`. Is anything else indexed by compound beyond
   `u` (`drug_unimol` + `drug_descriptors` + `drug_fingerprints`) and the atom tokens (`drug_atom_reprs` / `drug_atom_offsets`)?
   Is the permutation test the right guard?
3. **Is R1's restating rule right?** If M ≥ v9, does it follow that "v9's margin over ridge is a basal-profile margin"? Or does R2
   decide that, with R1 alone only saying "v9 does not beat a drug-free model"?
4. **The shared-control effect (98.1, §91.12):** Δ shares X_ctl's noise with every arm's input. Is a paired comparison enough? Or
   should a distinct-control stratum be reported (rows whose X_ctl is not shared with any training row)?
5. **D's gate:** is "adapter alone, and ≤ 6 GPU-h" the right bar? Is PertDiT worth running at all, given it is at or below XPert in
   Bai et al.?
6. **Order:** M this week on CPU; B1 and D next week, after O9's session 2. Agree?
