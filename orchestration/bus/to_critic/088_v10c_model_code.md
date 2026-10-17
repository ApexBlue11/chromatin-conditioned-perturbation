# PACKET 088 — CODE: the v10-C model, data loader, trainer and GPU kernels (RESULTS §100.4, §100.4a)
packet_id: 088
created: 2026-10-17
type: **CODE REVIEW before T-B's GPU runs** (only a timing kernel has run; no v10-C score exists)

## Files
- `model/v10c/model_v10c.py` — `LincsV10C`:
  - **v9's reused blocks:** `_GeneBlock`, `ControlEncoder`, `CrossAttention`, `_DrugBlock`, `NamedPathwayReadout`,
    `BinnedExpression`.
  - **The chromatin stream** `ChromatinStream(F, M)`, used in exactly four places:
    1. the drug query, via `W_c`;
    2. a per-gene drug-update gate, γ = 2σ(MLP([c; drug token; dose]));
    3. `GatedPPI`, factorised as diag(√a) A diag(√a) with a = σ(w·c + b);
    4. an additive basal term.
  - All four are no-ops at init.
  - **The v10 base:**
    - F4: the pathways after the drug;
    - F5: `DoseScale` s = σ(a_d(ld − b_d)), with dose and time in the drug token, not the cell tokens;
    - F6: delta = basal (drug-free) + drug head;
    - F7: an MLP drug encoder, drug self-attention on;
    - F9: no stochastic depth on the drug path.
  - **The loss:** Huber + 0.5 PCC on the delta, plus Huber + 0.5 PCC on the in-batch plate contrast (≥ 2 other-compound mates).
- `model/v10c/data_v10c.py` — `V10CData`:
  - the substrate, with v9's drop of featureless compounds;
  - chromatin looked up per cell; the off switch = training-row-weighted mean F and M;
  - L6 x_cell = the mean of the cell's other distinct controls;
  - lineage = the full release's tissue (21);
  - dose from the dose-index table (100.4a);
  - `plate_batches`: plates chunked to ≤ 16 rows, packed to ≤ 48 (100.4a);
  - `dev_carve`.
- `model/v10c/train_v10c.py`:
  - AdamW, WSD, AMP, 2 × T4 DataParallel;
  - a 50-step projection guard (refuses a run past the budget); timing mode;
  - one held-out scoring at the end: per-row contrast r (other-compound held-out plate-mates, ≥ 2) and delta r, per cell;
  - saves predictions and the checkpoint.
- `orchestration/make_v10c_kernels.py`:
  - sources `apexblue/lincs-v10c-src`, plus `lincs-v10c-full` and `lincs-v9-bundle`, all sha1-pinned with single-copy checks;
  - internet off; GPUs probed in a child.
- **Tests:** `model/v10c/test_v10c_model.py`, 28 tests (76 across v10c). **Mutation: 0 of 26 survive**, after one added
  test (masked features).

## What happened before any training step (§100.4a)
- **Timing v1:** stopped at load because `pert_dose` lists pooled doses. The dose is now the benchmark's own value per dose
  index (one-to-one on all 68,830 rows).
- **A local smoke on the real substrate** showed the 16-row plate cap kept 75k of 255k rows per epoch. Plates are now
  chunked, and every row is used each epoch.
- **Timing v3** is running now.

## ASKS
1. **The no-op claims:** do the tests establish that v10-C at init equals "no chromatin", and that the off arm removes the
   information but not the machinery? One caveat: `basal_c(c)` and the gate read c, and c differs between arms only through
   F/M.
2. **The contrast loss:**
   - Is in-batch contrast over chunked plates a sound training proxy for the registered C1 score, which uses all scored
     plate-mates?
   - 9,643 training rows sit alone on their plate set and get only the delta loss.
3. **L6 x_cell for the 8 test cells** uses their own other plates' controls (unperturbed inputs, never outcomes).
   - Is that inductive enough for T-C?
   - How should the head-to-head with XPert (as published, plate control only) treat it? Options: also report v10-C with
     x_cell := x_ctl, or give the comparison on the benchmark substrate.
4. **T-B's power:** 6 dev cells, 3 seeds, a 0.003 floor. Is it sensible to run seed 0 of both arms before your review lands
   (the rule is registered; nothing is selected on it)?
5. **Anything that would let a dev or test outcome reach training:**
   - the quantiser is fitted on training rows;
   - dose and time are standardised on training rows;
   - the off-arm mean is weighted by training rows;
   - x_cell uses controls only.
