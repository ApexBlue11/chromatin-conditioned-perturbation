# REVIEW OF PACKET 061
verdict: SOUND-WITH-CAVEATS
reviewed_commit: ce75800 (Stage A result d68adcd; Stage B registration 7dfdd16; packet 80dcadc)

**Stage A's reading is mechanical, and I reproduced it.**
- **Provenance:** `stage_a_result.json` = d7f3856a, matching its marker.
- **The rerun:** `--read` on the current code gives output **identical** to `stage_a_reading.txt`.
- **The numbers:** every number in 94.8 matches the JSON: A1, null mean and sd, p, compounds scored, both ceilings,
  active-subset A1 and n, A3's T, p, fraction and 17 per-unit d's, and the ungated T, p and 16 of 20.
- **The row change after clearance (a1d2372)** reads only the `row_index` key of `v9p7_seed0.npz`, as §94.2 registered. It
  computed nothing before the guard.
- **The review 060 fixes are in:** the ceiling is half-against-half with same-plate units dropped plus a no-shared-plate variant,
  and `per_class_auroc_meaning` is recorded.

**Stage B's code implements 94.9** (079b17ba). The design has one MAJOR point: B3c, and so reading 2b, can be won by
**cell-level response magnitude** rather than pathway-specific cell biology (C1). Fix it, or reword 2b, before any prediction is
read. C2–C5 are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | construct | **B3c's within-class centring removes each class's mean across cells, but not a cell's overall response scale.** A3's d is in raw ULM t-units. If a model predicts larger responses overall in some cells, every class's predicted d grows in those cells. Wherever the measured responses are also larger there, the centred d's correlate. μ is the same vector in every cell, so it can't do this. So v9 could pass **B3c_v9 > B3c_μ** by predicting a cell's general responsiveness. Reading 2b would then license *"…predict in which cells a pathway inhibitor's pathway effect is strong"*, a pathway-specific claim, without pathway specificity. | **Before any prediction is read, compute B3c on standardised d:** each pathway's ULM activity z-scored within each cell over the cell's labelled compounds, then d computed as registered. Both measured and predicted are standardised the same way. This removes cell-level scale and keeps all 17 units. Record each cell's per-pathway activity sd in every result file, so `read_stage_b` can standardise. Keep raw-d B3c as reported. **Or** keep B3c as it is and reword 2b's sentence: *"…predict in which cells a pathway inhibitor's measured pathway effect is larger, which may include the cell's overall response magnitude"*. |
| 2 | MINOR | overreach | **Two of 94.8's interpretation sentences say more than Stage A shows.**<br>(a) **"The pathway magnitude is strongly cell-dependent (MAPK +20.1 in MDAMB231 against +5.6 in MCF7)":** a unit's compounds differ by cell. **MAPK** uses 10 distinct compounds across its 4 cells, only **3** in every cell. **PI3K** uses 16 across 5, with **2** in every cell. **EGFR** 9 / 5, **JAK** 8 / 3. The RAF members are in HT29 only, by design. So the across-cell spread mixes cell biology with compound composition and, as in C1, cell-level scale.<br>(b) **"The earlier model-mechanism nulls cannot be blamed on mechanism being absent from the data at these readouts":** the earlier readouts (atom→gene attention, gradient × activation on targets, a pathway layer) are different readouts. §86's data projection found no target-pathway alignment at its readout. | (a) *"…varies across cells (MAPK +20.1 in MDAMB231 against +5.6 in MCF7); the units contain different compounds in each cell, so this mixes cell biology with compound composition."*<br>(b) *"The measured responses carry mechanism at retrieval and pathway-activity readouts, so the earlier model nulls are not explained by the data lacking mechanism in general. Whether the data carry it in the form those readouts sought is separate, and §86's data projection says not at its readout."* |
| 3 | MINOR | wording | **"The measured ceiling" isn't a ceiling for predictions.** Predictions carry no measurement noise, and μ averages each compound over many training cells. So A1 on predictions **can exceed** measured A1, and `ceiling_fraction` can be > 1 without meaning better than the truth. The same applies to the self-retrieval diagnostics on prediction runs: μ's two halves are identical, so its ceiling is trivially 1. | Call it **"the measured reference"** in 94.9 and the reader's text, and note that fractions above 1 reflect noise-free predictions. Mark the C4 diagnostics as not meaningful on prediction runs. |
| 4 | MINOR | reading | **Reading 1 (EXPRESSES MECHANISM) will very likely hold for μ too,** since compounds are seen in training. Stated alone for v9, it reads as a model finding. | State reading 1 together with μ's (B1/B3 for μ, already computed in `read_stage_b`): *"v9's predictions express mechanism (as does μ)"* when μ passes. |
| 5 | MINOR | input | **The bundle's `ridge_pred` sits beside `y_true` and `ctl_true`, so it may be predicted expression, not a predicted Δ.** If so, `--delta …:ridge_pred` would score absolute expression. `mean_drug_delta_pred` and `deg_pred` are Δ by name and by `xpert_arm.py`. | Confirm `ridge_pred`'s target. If it's expression, pass `ridge_pred − ctl_true`. It's reported only. |

## Answers to the asks

**Ask 1 — Stage A is mechanical (94.3–94.7, reproduced).** The 94.8 wording is licensed except the two sentences in C2.
- **"Both readouts carry signal":** right.
- **THP1:** misses on p, with 17 compounds after the plate rule, exactly as review 060 predicted.
- **No positive-control unit:** disclosed.
- **The ungated report:** consistent with the RAF gate.

**Ask 2 — the design is sound with C1.**
- **μ is the right reference.** Under the cold-cell split it carries each compound's training-cell response and no cell
  information, so "beyond μ" is the right definition of what the model adds.
- **B3c's within-class centring** isolates across-cell variation within a class, but not cell-level scale (C1).
- **Composition:** μ controls it only through the comparison. That's adequate given C1, because μ's d varies across cells
  exactly through composition.
- **The bars:**
  - 2a (v9 − μ > 0 in ≥ 4 of 5 cells, mean ≥ 0.02) and 2b (p < 0.05 and > μ) have no null for the v9 − μ difference itself.
  - "Seed-mean and every seed" partly substitutes, and it's a sensible, conservative robustness rule.
  - Optional: report a paired swap null for 2a (per compound, swap the v9 and μ signatures, 1,000 draws).
- **The plate rule:** keeping it on prediction runs is right, since v9's inputs include the plate-matched control profile.

**Ask 3 — the code matches 94.9.**
- **`load_predicted_delta`:** aligns each file by its own `row_index` and asserts coverage of Stage A's rows. The seed-mean is
  the mean over files. The keys exist: `deg_pred` and `row_index` in `v9p7_seed*.npz`; `mean_drug_delta_pred`, `ridge_pred` and
  `row_index` in the baselines.
- **`b3c`:** within-class centring; the null permutes the predicted d across cells within each class (10,000 draws, rng 9460);
  ρ = 0 for a prediction with no within-class spread; p = (1 + #≥)/(1 + n).
- **`read_stage_b`:**
  - B1 and B3 by Stage A's rules on each prediction run's own nulls;
  - 2a and 2b as registered;
  - all readings over the seed-mean plus the 3 seeds;
  - marker-verified inputs.

**Clearance: not yet.** C1 changes how B3c is computed, and standardising needs per-cell activity sds in each run's output. So
the 6 runs should happen after C1 is decided: either implement the standardisation, or adopt the reworded 2b. With the reword
only, the code is cleared as is.

## What I checked and found sound

- **Stage A:** output and marker sha1s, `--read` reproduced, and every reported number against the JSON. The a1d2372 diff (only
  `row_index` read). The review 060 fixes are present in the result keys.
- **Stage B code:** the diff a1d2372 → ce75800 in full (`load_predicted_delta`, `b3c`, `_verified_json`, `read_stage_b`, and
  the CLI's `--delta` and `--read_b`).
- **The prediction files and baselines:** key names only, no values read.
- **Identity level:** distinct and shared member compounds per A3 class across cells (C2a).

## What I could not assess, and why

- **Whether cell-level scale is large in the predictions.** That would need reading predictions, which 94.9 forbids before
  the run. The measured d's don't show a uniform cell scale (MDAMB231 is high for MAPK and low for EGFR). Hence C1 is a fix to
  the instrument, not a claim about the result.
