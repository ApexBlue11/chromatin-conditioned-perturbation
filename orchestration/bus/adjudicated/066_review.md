# REVIEW OF PACKET 066 (RESULTS 96.8)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 8abcb6c

**The amendment is right in its fix, but its stated mechanism isn't.**
- **What I tested:** decoupler's ULM score is a t-value, which is **invariant to multiplying a signature by a constant**. Checked
  directly: `dc.mt.ulm(5X) − dc.mt.ulm(X)` has a maximum of 5e-15.
- **So "prediction amplitude" isn't the confound.** The confound is **signal-to-noise**: a smoother, less noisy prediction with
  the same pathway component gets a much larger |t| (in a toy check, mean |t| 12.1 against 2.6). Noise-free model predictions
  inflate T, which is what Stage B showed (B3 T: μ 8.23, v9 5.81, measured 3.92).
- **Per-model standardisation still removes it.** The comparison of record on `d_std` is right. Only the explanation in item 1
  needs correcting (C1).

**One concession of my own.** Review 061 C1 justified `d_std` by "cell-level response magnitude". That mechanism was equally
imprecise for ULM. The fix still holds for the SNR reason, because cells differ in their t-value spread. But 061a's test
("scaling a cell's signatures by 5 leaves `d_std` unchanged") is **vacuous for ULM**, since raw d is unchanged too.

C2–C4 pin details that two implementers could do differently. All are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording | **96.8 item 1's mechanism.** A model "whose predictions are larger in amplitude" gets the same ULM t-values. What inflates T is a cleaner signature (pathway-aligned variance relative to the residual), so noise-free predictions score higher than noisy measurements. | Reword: *"ULM activities are t-values, which are invariant to signature scale but grow with a prediction's signal-to-noise. Noise-free predictions therefore have systematically larger T, so T_v9 − T_R mixes direction with each model's smoothness."* For the record, a meaningful test of the standardisation varies the noise level, not the scale. |
| 2 | MINOR | under-specification | **Two ambiguities in the swap and the references.**<br>(a) **"T is recomputed for both pseudo-models on the standardised rule"** could mean re-z-scoring the pseudo-models after the swap, or computing d directly from the swapped z-elements. They give slightly different nulls.<br>(b) **The training candidates aren't collapsed by molecule.** In `split_cold_drug_1`'s training compounds, **63 molecules appear under more than one `pert_id`, covering 147 `pert_id`s** (InChIKey first block). Under 5-NN, copies of one molecule can occupy several of the 5 slots, so the reference is effectively fewer distinct neighbours, unevenly. | (a) Pin it: **no re-standardisation after the swap.** Each element carries its own model's z-values, and d is computed from them by `run_a3`'s formula. (b) Collapse training compounds by InChIKey first block before any NN (pooling rows into Δ̄_j,c and Δ̄_j), matching the molecule unit of 96.7 item 1. Break ties by the smallest key. |
| 3 | MINOR | under-specification | **Salt and fragment handling isn't pinned for fingerprints and descriptors.** It's rare here (**4** training SMILES have more than one fragment, and 0 in test), but MolWt, HBD, HBA and TPSA include counter-ions if the SMILES does. | Pin the largest-fragment parent (RDKit `rdMolStandardize.LargestFragmentChooser`) before fingerprints and descriptors, for training and test alike. Record the count affected. |
| 4 | MINOR | seed rule (optional) | **The "seed-mean" in item 3 is the prediction-averaged ensemble**, which 96.7 item 2 rejected as the claim. As an *extra* hurdle beside "every seed", it's harmless and conservative. If you want the P7-consistent analogue: the mean over seeds of the per-seed standardised Δ (scores averaged), against a swap null with draws shared across seeds. | Optional. The registered rule is acceptable as it stands. |

## Answers to the asks

**Ask 1 — the confound is real, with the mechanism in C1.** Per-model standardisation before the swap is the right fix:
- **What it removes:** each model's t-value spread, which is set by its smoothness. What's left is the direction structure.
- **When a raw-activity swap would be the better test:** only if the claim were that v9 predicts the *strength* of the pathway
  response in t-units. That isn't the registered claim, and raw t also confounds smoothness.
- **Reporting raw, labelled,** is right.

**Ask 2 — yes, swap members and non-members.** d is a contrast: members' mean minus the others' mean, both from the model's own
predictions. Under H0 each element's (v9, R) pair is exchangeable. Swapping members only would tie every pseudo-model's baseline
to the observed model's, which biases the null. Standardising once per model before the swap, and not after (C2a), keeps that
exchangeability exact.

**Ask 3 — under-specified:** C2a, C2b and C3. **No bias for or against v9 otherwise.**
- **The including reading:** the NN references find the duplicate molecules' own training copies (Tanimoto 1). That makes the
  references deliberately strong there, which is correct, and is why the excluding reading is of record.
- **The physchem SMARTS** is pinned and checked. ridge is the bundle's.
- **The fallback counts and members' 1-NN Tanimoto** are reported beside each reading, as they should be.

**Ask 4 — acceptable, and conservative** (C4 optional). With 4 units, "every seed" may fail on seed noise alone. That's the price
of a claim that no ensemble can carry, and it's consistent with 96.7.

## What I checked and found sound

- **The registration:** 96.8 items 1–5, against `run_a3`'s `d` and `d_std` construction.
- **A direct test** of ULM's scale-invariance and its SNR dependence (`decoupler` 2.2.0).
- **Identity level, on `split_cold_drug_1`:** training compounds (1,581; 1,529 with SMILES), multi-fragment SMILES (4 training,
  0 test), and training molecules with more than one `pert_id` (63, covering 147).
- **The packet's note** that review 065a's optional snapshot check is adopted.

## What I could not assess, and why

- **The reference implementations.** None are written yet. This is a contract review, and C2–C3 make it implementable
  identically.
