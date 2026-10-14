# PACKET 084 — CODE: review 083 adopted; Phase 1's code (C1–C4), before any of it runs (RESULTS §99.4; `V10_AUDIT_AND_PLAN.md` §8)
packet_id: 084
created: 2026-10-14
repo_commit: (this commit)
type: **CODE REVIEW** (no Phase 1 test has run; no test-row score has been computed)

## 1. Review 083, adopted (all six, plus asks 2 and 5)

**Verified by the PI before adoption** (training rows; now A2b in `model/results/audit99/target_sources.json`):
- over F0 the drug term adds +0.070 (31,722 rows with plate-mates) and +0.156 (9,125 rows without, best w = 2.0; yours was
  +0.152);
- the dose-only floor on the contrast is 0.027;
- 60 % of plate-mates are at another dose.

**The plan:** `V10_AUDIT_AND_PLAN.md` §8 records the amendments and **governs where §1–§7 differ**. The superseded lines are
struck through in place, not rewritten. M3 is now *partly confirmed*.

**B1** runs at the reset as registered (ask 5).

## 2. The code

**`model/v9/audit99_phase1.py`: C1, C1b, C1c, C2a, C2c, C4 rescoring, plus the C2b / C3 readers.**

- **The contrast:**
  - **Definition:** `plate_contrast(V, plate, pert)`: c_i = V_i − the mean of V over the scored rows on i's plate with a
    different compound, ≥ 2 of them.
  - **Plate proxy:** the control group. `_plates_for` refuses if the bundle's `X_ctl` differs from the scored file's
    `ctl_true` by more than 1e-4.
- **`rowr_pred`:** a prediction row whose centred norm is below 1e-4 of the truth row's scores 0 and is counted. This
  operationalises "zero variance" so that a plate-constant model is not scored on float rounding (please check).
- **C1:**
  - **Primary:** v9 (P9, 3 seeds; row score = the mean of per-seed r) against M+dt, per molecule, on clean rows with ≥ 2
    mates. The statements are benchmark-level, and the no-claim carries `ci_hi`.
  - **Reported:**
    - per seed;
    - v9 against 0;
    - M, which is 0 by construction (the zero-variance rows are counted);
    - the magnitude ratio;
    - **the dose-only floor:** `loo_mean` over **training** rows at the same (cell, time, dose) on another control,
      contrasted over the scored plate-mates;
    - the strata by the plate-mates' mean ECFP4 Tanimoto (`drug/outputs/drug_fingerprints.npy`; all 389 test compounds
      covered).
- **C1b:** the oracle plate (P_meas over all bundle rows). **C1c:** the training-plate predictor on the original target.
- **C2a / C2c:** P7 paired with O2 on `row_index`. Both refuse if the targets disagree by more than 1e-4.
- **C4:** Spearman of dose against ‖y − P_meas‖ and ‖ŷ − P_meas‖ per (compound, cell, time) with ≥ 3 doses.
- **`read_c2b`:**
  - **Variant of record:** `deg_ctl` (`x_cell :=` the row's own `X_ctl`). `deg_global` is reported.
  - **Precondition:** the intact cluster lead must be > 0 with its CI excluding 0, else "not assessable".
  - **Label:** final-snapshot basis.
- **`read_c3`:** the variant of record is `tani`, with `free` reported. The statement carries the CI (or `ci_hi` for a no-claim).
- **`_load_infer`, per seed:**
  - the manifest must be complete, for the right seed, and carry the file's sha1;
  - the record's mode and seed must match, and a `--limit` smoke output is refused;
  - the seeds must agree on `row_index`, `y` and `ctl`.
- **`main`:** each test writes `phase1_<T>.json` and refuses to run twice.

**`model/v9/audit99_infer.py`: the inference for C2b and C3 (CPU, deterministic, `model.eval()`).**

- **Identity before anything is modified:** 2,000 rows of the saved final-snapshot `deg_pred` (drawn with seed 0) must
  match, with min r ≥ 0.9999 and every row finite.
- **xcell:** passes for `ctl`, then `global`. `x_cell` is restored between them.
- **swap:**
  - `substitutes(cell, pert, fp, rng, tani_max)` gives one substitute compound per (compound, cell), drawn uniformly from the
    allowed set (other compounds in that cell, and Tanimoto < 0.4 for `tani`);
  - every row of the compound takes the substitute's first-row drug inputs (`drug_row`, so atoms and the atom mask, plus
    `u`);
  - a (compound, cell) with no allowed substitute is −1, unswapped and counted.
  - Fingerprints come from `u[:, 532:2580]` (`xpert_arm.py` builds u = [Uni-Mol 512 | descriptors 20 | ECFP4 2048]).

**`orchestration/make_audit99_kernels.py`:** three free Kaggle CPU kernels, `kern_audit99_s{0,1,2}`, each with 13 pins (9
sources, 2 checkpoints, 2 identity files).
- Each kernel:
  - refuses any input whose sha1 differs or that has more than one mounted copy;
  - runs swap (P9 seed k) then xcell (P7 seed k);
  - writes `AUDIT99_COMPLETE_s{k}.json`.
- Mounts: `lincs-v9-bundle`, `xpert-mdmt-benchmark`, and the new private `apexblue/lincs-audit99`.
- **Disclosed:** the sources are the current `xpert_arm.py` (`c55ce257`), not P9's or P7's pinned `60bdcd48` / `75c58f52`. The
  identity check is the equivalence proof. On a local smoke (16 identity rows, 48 swapped rows, 8 xcell rows), min r was
  0.99999996 on both models. That smoke printed one plumbing number, the max |intact − swapped| on 48 rows (5.5), and
  nothing was scored.

## 3. Tests and mutation

- **26 tests pass on CPU** (19 in `test_audit99_phase1.py`, 7 in `test_audit99_infer.py`), on synthetic worlds only.
  They include:
  - C1 crediting a compound-using model and not a plate-only one;
  - the no-claim bound;
  - the dose floor seeing dose structure, refusing non-training rows, and excluding the row's own plate;
  - C2b carried / not carried / not assessable, plus a tampered-output and a smoke-output refusal;
  - C3's variant of record and bound;
  - the swap's within-cell, compound-level and Tanimoto rules, and determinism;
  - the identity guard;
  - the **generated kernel executed end to end** on fake mounted inputs, plus its tampered and duplicated-input refusals.
- **Mutation: 0 of 20 survive.** The mutants were:
  - same-compound exclusion; minimum 1 mate; the zero rule off or exact-only; the control guard;
  - clean rows; the M comparator; `other_mean`'s selection; rerun;
  - the bound unformatted; the floor on all rows; the floor keeping its own plate;
  - C2b's precondition; C2b's variant; C3's variant;
  - the manifest sha1; smoke records;
  - the Tanimoto constraint; a row-level swap; the identity threshold.

## ASKS
1. **The contrast code and `rowr_pred`'s relative zero rule:** is 1e-4 of the truth norm right? Is any path left where a
   plate-constant model scores non-zero?
2. **The dose-only floor as implemented:** training rows only, a different control, exact dose match, then the same
   contrast. Is this the floor you meant?
3. **C2b:** is `x_cell := X_ctl` the right inductive replacement inside this architecture? It makes the two control views
   identical.
4. **C3's substitution:**
   - one substitute per (compound, cell), uniform among those allowed;
   - −1 when none, and those rows are excluded from both variants;
   - is excluding a row unswapped under *any seed* right? `swapped` requires all three.
5. **Running the identity check against the current `xpert_arm.py`** instead of the pinned one: acceptable as disclosed?
6. **Anything that should block running C1 / C1b / C1c / C2a / C2c / C4 locally, and the three CPU kernels?**
