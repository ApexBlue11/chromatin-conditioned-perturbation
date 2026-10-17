# REVIEW OF PACKET 088 (v10-C model, loader, trainer, kernels; §100.4 / §100.4a)
verdict: SOUND
reviewed_commit: 91da9df

**The code implements §100.4, and the no-op and isolation claims hold.**
- **At initialisation,** none of the four chromatin pathways depends on F / M:
  - `W_c = 0`, so the query is unchanged;
  - the gate's last layer is zero, so γ = 1;
  - `GatedPPI`'s gate weight is zero, so a = 0.95 for every gene, and its output projection is zero-initialised;
  - `basal_c = 0`.
- **The basal term is identical for plate-mates** (same `x_ctl`, the L6 `x_cell`, lineage and chromatin). So the contrast is
  carried by the drug path alone, which the tests check.
- **The plate key is the control group exactly:** 25,092 `plate_id` and 25,092 `ctl_id`, a bijection, with no plate holding two
  controls. So T-B's dev contrast is C1's definition.
- **Tests:** `model/v10c/` passes **80 / 80** under `.venv-cuda` with `CUDA_VISIBLE_DEVICES=-1`.

**Three MINOR points.** None blocks T-B. T-C needs review 087 C1 first.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | guard (loader) | **`V10CData`'s "a held-out cell is in training" check compares names,** so it passes with `NCIH1975` (= `H1975`) in training (review 087 C1). | Canonicalise cell names through an alias map before the check (`NCIH1975 → H1975`), and add a test with an aliased held-out cell. |
| 2 | MINOR | robustness (trainer) | **With `CUDA_VISIBLE_DEVICES=""` this machine's torch reports `is_available()` True but 0 devices.** The trainer then picks `cuda` and exits at the GPU guard, which is how 3 tests failed on my first run. It is fail-safe, not a defect on Kaggle. | Choose the device from `torch.cuda.device_count() > 0`, or document that tests hide CUDA with `-1`. |
| 3 | MINOR | the off arm's description (Ask 1) | **"Removes the information, keeps the machinery"** is precisely: the off arm keeps a **cell-invariant** chromatin vector per gene (the training-row-weighted mean F and M), and removes the **cell-specific** part. Gene-level average chromatin reaches both arms. | Say so in §100.4 and in T-B's statement: *"cell-specific chromatin contributes"*. Review 086 C4's gene-shuffle permutation addresses the gene-level part. |

## Answers to the asks

**Ask 1 — yes.** The tests establish that v10-C at init doesn't depend on chromatin:
- `test_chromatin_is_a_noop_at_init`;
- `test_gate_and_string_gate_start_at_their_registered_values`;
- the per-pathway "opened" tests.

**On the caveat:** `basal_c(c)` and the gate read c, and c differs between arms only through F / M. That is the intended single
difference. The basal route can't move the primary (contrast) score, because it cancels within a plate
(`test_contrast_cancels_the_basal_term_for_plate_mates`). It can only move delta r, which is reported. The wording is C3.

**Ask 2 — yes, a sound proxy.**
- Truth and prediction use the same in-batch mates, so a subset of plate-mates gives a noisier but consistent estimate of the
  plate mean. Chunks of one plate packed into the same batch share mates.
- The 9,643 single-plate rows getting only the delta loss is right.
- **Put `contrast_rows_per_step` (already logged) in the T-B record,** so the proxy's coverage is visible. Large plates split
  into ≤ 16-row chunks will often leave rows with fewer than 2 other-compound mates.

**Ask 3 — input-transductive, as review 086 C3 says. Label it, and give XPert the same input or use 086 C1(b)'s wording.**
- **Don't use "v10-C with `x_cell := x_ctl`" as the comparison.** C2b showed that configuration is off-distribution for a model
  trained with a cell mean.
- One exception: L6 already gives single-control cells `x_cell == x_ctl` in training (flagged), so for H1975 that input is
  in-distribution. H1975 is excluded anyway (087 C1).

**Ask 4 — acceptable.** The rule is registered, nothing is selected on seed 0, and none of this review's points changes a T-B
code path. Record that seed 0 of both arms started before review 088 landed.

**Ask 5 — nothing in the loader or trainer lets a dev or test outcome reach training:**
- the quantiser is fitted on training rows;
- dose and time are standardised on training rows;
- the off-arm mean is weighted by training rows (dev cells excluded);
- `x_cell` uses control profiles only;
- dev cells are removed from training;
- held-out scoring uses held-out mates only.

The one leak is the data-level alias (087 C1), which affects T-C, not T-B.

## What I checked and found sound

- **`model_v10c.py` in full:**
  - the stream, `GatedPPI` (the factorisation and the 0.95 start), `CellFiLM` (identity start, lineage only), `DoseScale`;
  - `PerturbBlockC` (query, gate, dose scale), the forward pass (basal / drug split, F4 placement);
  - `batch_contrast`, `v10c_loss`.
- **`data_v10c.py` in full:**
  - the feature drop, `dev_carve` (the registered pool and draw), `leave_own_plate_out` (distinct profiles, own excluded,
    singles flagged);
  - the off switch, dose from the index, training-row standardisation;
  - `plate_batches` (every row once, chunks whole), `batch()`.
- **`train_v10c.py` in full:**
  - the GPU guard, quantiser fit, WSD, AMP / DataParallel (no cross-sample ops in the model), the 50-step budget guard;
  - `heldout_scores` (C1 definition, per-cell), and the record and prediction file.
- **The kernel builder's pin and single-copy pattern.**
- **The plate / control bijection** on the substrate.

## What I could not assess, and why

- **The timing kernel's output and any T-B run.** None has been read.
- **The kernel builder's full pin list against the uploaded `lincs-v10c-src`.** I relied on the builder's own sha1 computation
  and the in-kernel checks.
