# REVIEW OF PACKET 032
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 0e83810

**Binding rule 8 to the aux readout is the right repair, and it's outcome-free with respect to C6 and C7.** It should
be declared as an amendment, a change of instrument, not presented as a reading of the old rule (ask 1).

The bigger finding sits in ask 2. **The permutation null that every alignment claim so far has been measured against
can't license "in this cell".** A data-only, cell-agnostic ranking of pathways scores 0.214 on the same metric. The aux
readout's 0.273 is only about +0.06 above it, and §37's "8–12 sd" and the dev "40 sd" are against a null that a generic
ranking beats by more than 20 sd (C1). That changes what CLAIMS 4.16 may say, and it makes rule 8's second condition
vacuous (C2).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | overreach | **The alignment has never been compared with a generic pathway ranking, and a generic ranking gets most of it.** The column-permutation null destroys pathway identity. So it rejects "random", but not "the same ranking of pathways for every row". I built the simplest cell-agnostic prior from data alone: for each dev cell, the mean pathway target `\|Δ\| @ M_rownorm.T` over the **other five** dev cells' rows, the same vector for every row of the cell. Through `interp_v9.pathway_alignment`, on the same 4,043 rows, it scores **0.2143**. The aux readout scores 0.2732 (seeds 0.263–0.279). So about 0.21 of the 0.27 is available from "which pathways generally move", with no cell information, and the cell-level part is about **+0.06**. Two contrasts, for scale: pathway size alone scores 0.066, and the row's own `\|control\|` pooled over members scores −0.073. So the prior is not an artefact of pathway structure. CLAIMS 4.16's sentence *"the named pathway nodes rank which pathways move in this cell"* rests on exactly the part the null doesn't test. | Add to `align_dev.py` two references, computed from data or readouts only and committed before any variant is run. **(a) A training-row prior:** per dev row, the mean pathway target over the training rows, the ranking a model could learn with no cell information. **(b) A cell-shuffle null:** the readout rows permuted across dev cells, pathway columns intact, 200 times. That measures how much of the readout's alignment survives being given another cell's readout. State any interpretability sentence as the increment over (a), and require that increment to exceed (b)'s spread before "in this cell" is used. |
| 2 | MINOR | stats | **Under the aux readout, rule 8's "≥ 5 sd of its own permutation null" is vacuous.** The bar is 0.0010 + 5 × 0.0066 ≈ **0.034**, against a baseline at 0.273. By C1, a generic ranking would pass it at about 32 sd (0.214 against 0.0010 ± 0.0066). The non-inferiority half (≥ 0.2532) does all the work. | Replace the second condition, before any variant is measured, with: **the variant's 3-seed mean alignment > the training-row prior's alignment (C1a) on the same rows.** It's a data-only constant, so it can be fixed now. P2 passes on every seed against the leave-one-cell-out proxy (0.278 / 0.263 / 0.279 > 0.214). |

## Answers to the asks

**Ask 1 — declare it as an amendment. It is outcome-free.**
- **Why it's an amendment:** rule 8 was written against §37's instrument and cites its numbers. The channel mean is
  a fixed, uniform projection of the 32 channels. The aux loss trains a different direction, `aux_path`
  (`nn.Linear(d_pathway, 1)`, shared across nodes), and nothing constrains how the two relate. Under the old readout
  the baseline fails the rule's own second condition, and a rule the baseline can't pass is broken, so repairing it
  before any variant is measured is legitimate.
- **Why it's outcome-free:** no C6 or C7 checkpoint has been through `align_dev.py`. The P2 values seen under both
  readouts don't favour any variant, because the gate is relative to that same baseline.
- **Supporting evidence, suggestive only:** in all three P2 models the learned aux direction has a **negative** weight
  sum (Σw −5.8 / −6.4 / −4.3; 34–44 % of the 32 weights positive). So a uniform channel mean points largely *against*
  the trained direction, which fits the inverted sign. The fold-0 r-series is also negative (−0.8 / −3.3 / −1.1), and
  e12 is near 0 (+0.26), so Σw alone doesn't predict §37's sign. The direction simply isn't pinned.
- **State the scope in the amendment:** the aux readout is scored against **its own training target**. `aux_targets`
  (`model_v9.py:237-243`) is `M_norm·|Δ|`, the same `M` row-normalisation and the same `|y_true − ctl_true|` that
  `align_dev.py` uses. So under the amendment, rule 8 reads *"the auxiliary pathway head keeps its held-out-cell
  accuracy (non-inferiority)"*. That's a sound gate, but not an interpretability readout in any stronger sense.

**Ask 2 — yes, cite the aux readout, and caveat §37 now.**
- **§37's bullet** (*"its alignment beats its own permutation null by 8–12 sd … carries real, measurable mechanistic
  signal"*) needs three corrections before anything cites it:
  - the channel mean isn't the trained readout;
  - its sign did not reproduce in the dev-carve models (−0.0385, z −2.7 to −6.6);
  - the permutation null doesn't control for a generic pathway ranking (C1).
- **"Mechanistic signal" should go.** At most, say the named layer's channel mean aligned with pathway-level response
  magnitude above a column-permutation null in one fold-0 model.
- **For CLAIMS 4.16,** the licensable sentence, once C1's references exist, is roughly: *"a shared linear readout of the
  named pathway nodes, trained to predict each pathway's response magnitude, ranks which pathways move in held-out cells
  at ρ = x, which is y above a cell-agnostic prior"*. It stays cell-level, drug-independent and supervised, as 4.16
  already says. Until (a) and (b) are computed, drop "in this cell" from the sentence.

**Ask 3 — the function, target and null form are §37's. The rows differ, and so do the data path and trainer.**
- **Identical:** `pathway_alignment` (per-row Spearman of the readout against `|Δ| @ Mn.T`), and a null built from one
  pathway-column permutation applied to all rows, 200 times.
- **Different:**
  - **Rows:** §37 scored 480 rows per split filtered at `strength ≥ eval_min_strength` (`probe_v9.py:106`).
    `align_dev.py` scores all 4,043 dev rows with **no strength filter**, so weak signatures, whose `|Δ|` targets are
    mostly noise, pull every readout's alignment down.
  - **Data path:** `XPertData` (with `x_cell` = the training-row cell mean), not `LincsV9Dataset`.
  - **Trainer and fold:** xpert_arm vs train_v9_gpu, and cc1 dev vs fold 0.
  - **Reporting:** z rather than the +1-corrected p.
- **Consequence:** none of this matters for rule 8, since baseline and variant are scored on the same rows. But it does
  mean the dev numbers and §37's aren't comparable. The sign flip is not explained by the rows, because weak rows shrink
  a correlation toward 0 and can't invert it.

## What I checked and found sound

- **`align_dev.py`:** the rows are rebuilt through `XPertData` with the sha1 asserted, the model is loaded with
  `strict=True` from the checkpoint's own cfg, it runs in eval mode, and it reads `aux['pathway_activations']` /
  `aux['pathway_pred']` from the pre-drug layer (`model_v9.py:211-213`).
- **P2's checkpoints are the scored models:** the packet's 128-row identity checks hold, and the three `.pt` files are
  now local.

## What I could not assess, and why

- **The training-row prior's exact value.** I used the leave-one-cell-out dev proxy (0.214). The training-row version
  needs the training rows' deltas through `XPertData`, which is a small CPU job for `align_dev.py`, and I didn't run it
  on the laptop.
