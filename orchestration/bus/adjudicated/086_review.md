# REVIEW OF PACKET 086 (v10-C, the chromatin programme: design, §100; read with the model registration of §100.4)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 85c2253 (§100.4 at d2523dc)

**The design is right for the principal's question.** Every lever goes in, then one comparison changes only the chromatin
*information* (the §45 convention, machinery kept). The registered T-B rule is §85's rule 7 on the contrast target, fixed
before any run, and all four chromatin pathways are exact no-ops at initialisation.

**Two design points need settling before T-C** (C1, C2). T-B can proceed as registered: the dev carve is clean.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | head-to-head confound (T-C) | **"v10-C against XPert under L6" can't separate architecture from data.**<br>• v10-C trains on the full release (209 cells, 254,685 rows). XPert's numbers come from the benchmark (32 cells, 47,509 rows). A v10-C win would conflate L1 (data scale) with L4–L6 (architecture and inputs).<br>• **The unseen-cell / unseen-compound gap** inherits the same issue, since the compound side also trains on the full release. | **Choose one, before T-C:**<br>(a) register XPert retrained on the v10-C substrate with L6. Ask whether the L6 input is a pure data substitution in XPert's cell-input path, and pin which input;<br>(b) word T-C's head-to-head as *"v10-C trained on the full release against XPert trained on the benchmark"*, attributing nothing to architecture.<br>**A cheap anchor either way:** v9's architecture trained on the substrate exists implicitly as v10-C minus the levers. It costs a GPU arm, so only if wanted. |
| 2 | MAJOR | test-cell validity (T-C rule) | **One of the 8 unseen-cell test lines isn't unseen:** `H1975` = `NCIH1975`, in training (review 087 C1). §100.5's cell rule ("≥ 6 of 8 and a cell-cluster CI") would count a leaked cell. | Register T-C over the **7 clean cells**, with a rule scaled to 7 (e.g. ≥ 5 of 7, plus the cluster CI). Report H1975 separately. Exclude `NCIH1975` from v10-C's unseen-cell training (087 C1). |
| 3 | MINOR | L6 label (Ask 3) | **L6 is in-distribution, but for unseen cells it is still input-transductive.**<br>• For a test cell, "the cell's controls on other plates" are the controls of its *other test plates*: test-time inputs from other test rows, though not labels.<br>• It is identical in training and test, so x_cell ≠ x_ctl stays in distribution (C2b's failure mode is gone). | Label L6 *"leave-own-plate-out, input-transductive across a cell's plates"* wherever the unseen-cell result is stated. Give XPert the same input (C1a). Report how many test cells have a single control profile (where L6 = x_ctl). |
| 4 | MINOR | what T-B identifies | **On / off can't tell chromatin as a cell descriptor from chromatin as gene-level mechanism.** Unseen dev cells get cell-specific features either way. | **Report, never read, two cheap inference-time permutations on the trained "on" model** (no retrain, like C3):<br>• **gene-shuffle:** within each cell, permute the chromatin features across genes. This breaks gene-level information and keeps the cell's summary;<br>• **cell-swap:** give each dev cell another cell's chromatin.<br>On ≫ gene-shuffle means gene-level chromatin is used. |
| 5 | MINOR | attribution (Ask 1) | **"All at once" licenses "chromatin information (L2–L3), used through L4, contributes", not *which* lever.** L1, L5 and L6 change both arms equally, so they are context, not part of the contrast. | Word any positive T-B / T-C result accordingly. A "which lever" claim needs its own registered ablation later. |

## Answers to the asks

**Ask 1 — yes for the principal's goal** (does chromatin contribute in the best design we can build). A positive result can't
name the lever (C5), and a negative one, with every lever in, is the strong negative the doc already commits to reporting.

**Ask 2 — your three exclusions, plus cell aliases.**
- `NCIH1975` is the one live case (087 C1). My scan of all 217 full-release cells by mean-X correlation finds no other test-cell
  alias.
- The "(cell, compound, dose, time) matches a test row" rule is redundant once whole test cells and whole test molecules are
  excluded, so it is harmless.

**Ask 3 — inductive in the sense that matters for distribution (training equals test), but input-transductive** (C3).
- Give XPert the same input. If its cell input can't take it without a model change, use option (b) of C1's wording.

**Ask 4 — a T-A gate, with its calibration:**
- **The target:** per (cell, gene) on training rows, a responsiveness. For example, the mean over compounds of |contrast Δ|, or
  its variance.
- **The model:** responsiveness ~ gene fixed effect + the gene's basal level in that cell (x_cell) + chromatin features.
  Evaluate by **leave-cell-out** R² gain over the no-chromatin model.
- **The nulls:**
  - chromatin **swapped across cells** (keeps the gene structure, breaks cell specificity);
  - **permuted across genes within cell**.
- **Calibration:** plant a responsiveness ∝ accessibility effect at known strengths (as §91.9 did) and record recovery.
- **The pass rule:** the leave-cell-out gain exceeds the cell-swap null's 95th percentile.

**Ask 5 — an assessment, not a choice.**
- **What answers the registered question:** T-B on / off, and T-C on / off.
- **The width screen** is a single seed and cheap.
- **The arm whose value depends on an open decision is XPert under L6** (C1). Settle its training data before spending on it.
  Otherwise its result can't be attributed.

## What I checked and found sound

- **`V10_CHROMATIN_PROGRAMME.md`; §100–§100.4:** the levers, the tests, the off convention, the no-op initialisation claims, the
  T-B rule, the dev carve and its selection, and the epoch / budget rule.
- **The dev carve's cells against an alias scan:** clean.
- **The unseen-cell rescoring behind C2** (in review 087).

## What I could not assess, and why

- **The v10-C model code and its no-op tests** (`model/v10c/`), and the 100.4a loader fixes. They aren't in this packet.
- **T-A's design.** It isn't registered yet; Ask 4 offers one.
