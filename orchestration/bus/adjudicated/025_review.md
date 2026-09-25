# REVIEW OF PACKET 025
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 887f701

**This meets review 023's seven requirements in substance.** One primary readout, an unsupervised layer, five-init
untrained calibration, readouts compared by diff, a split with usable sizes, the unseen stratum licensing the claim, and
fold-0 models kept separate from §85, with the design fixed before C8b exists.

One calibration gap matters, because it sits in the clause that licenses the mechanism claim (C1). Three smaller
additions carry §86's lessons forward (C2–C4).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **The unseen-compound condition, which licenses "including for unseen compounds", is the one condition not calibrated against the untrained models.** All-rows `diff` must beat the minimum of 5 untrained inits **and** `m_u − 2·sd_u`. The unseen stratum needs only diff ≤ −0.02 with p < 0.05. §86 showed that on all 496 compounds, untrained readouts already swing by about ±0.024 across inits: u1/u2's output projections gave −0.023 (p 0.023) and −0.021 (p 0.039). The unseen stratum is n = 156. With about a third of the compounds, the between-init spread of its `diff` should be roughly √(496/156) ≈ 1.8× wider, around ±0.04. So a real all-rows signal driven by **seen** compounds, plus a structural fluctuation in the unseen stratum, could satisfy the SIGNAL row and publish the wrong clause. | Apply the same calibration inside the stratum: on all three seeds, unseen-stratum diff ≤ min(the 5 untrained unseen-stratum diffs) **and** ≤ `m_u,unseen − 2·sd_u,unseen`, plus p < 0.05, with the −0.02 floor kept as a minimum. The untrained runs already produce stratum diffs, so this costs nothing. |
| 2 | MINOR | code-vs-intent | **§86's gate and degeneracy check are missing.** The magnitude readout ‖Δa[p]‖₂ is prone to a drug-invariant ranking: nodes with many members or large activations move more under *any* drug. Null 1 would return diff ≈ 0 then, but §86 established reporting the gate *first*, so the reader knows whether the readout can express mechanism at all. | Carry both over unchanged: max ‖Δa‖ > 1e-12, then the median cross-drug Spearman of Δa rankings < 0.95, reported before any alignment number. A gate failure means NULL. |
| 3 | MINOR | overreach | **SIGNAL's wording has a data-projection qualifier but no output-projection one.** C8b's node is GELU(`M_norm`·`proj(h)`) **after** the drug, so Δa is the drug-induced change in its member genes' pooled hidden states, and the heads read those same states. In §86 the trained output projection showed no alignment (diff ≈ 0), so if Δa aligns while the output projection doesn't, the layer carries something the prediction doesn't show. That's the stronger finding. If both align equally, the named layer is a readout of the predicted signature. That's still interpretable, but a different sentence. | Add the symmetric rule: *"… beyond the model's predicted signature"* only if the output projection's diff is not ≤ −0.02 with p < 0.05 on the same rows. Otherwise: *"… a named readout of a predicted signature that is itself target-aligned."* |
| 4 | MINOR | stats | **The strong-perturber confound (020 ask 4) isn't addressed, and the median split is too small to address it.** Among compounds with a landmark target, a median split gives maybe 50–75 per side. That's too few to make any responsiveness stratum a condition. What the confound needs is a null that preserves the pairing, not a smaller subset. Strong perturbers' drug-induced changes concentrate in the generally responsive pathways, where their targets also tend to sit. That pairing beats plain label permutation with no mechanism. | Add **Null 1s: label permutation within quintiles of compound response strength** (median \|y_Δ\| over the compound's rows), 1,000 permutations, and require p < 0.05 against it for SIGNAL on all three seeds. It keeps all 496 compounds, so it has power, and it removes exactly the strength–pathway pairing. |

## Answers to the asks

**Ask 1 — yes, with C1 and C2.** On the calibration: "below the minimum of 5 untrained inits" is a good rule. If trained
readouts were exchangeable with untrained ones under no mechanism, each seed would pass with probability 1/6, and all
three with ≈ 0.005. Together with `m_u − 2·sd_u` it's appropriately strict. `sd_u` has only 4 degrees of freedom, but
the minimum rule backs it up. One stated limitation for the record: untrained models aren't exactly the null for a
*trained* model without mechanism. Training reshapes node scales and activation distributions. The exact null would be
a model trained with permuted drug identities, which is expensive and not required. Say "calibrated against untrained
inits", not "against a trained null".

**Ask 2 — acceptable, as a declared deviation.** Nothing in this test's reading compares against the r-series, so the
recipe change affects only cross-study comparability with §86, which isn't read. It removes the lockstep masks that
CLAIMS 6.13 now documents for every past Kaggle v9 model. Note that TF32-off is **inert on T4s** (Turing has no TF32),
so on Kaggle only the seeding change does anything. Record it that way, so nobody attributes a difference to TF32.

**Ask 3 — the median split and the separate no-landmark group are right for reporting.** The no-landmark group will be
most compounds, since full-GMT positives mostly aren't landmarks. The split can't carry a condition at its n, which is
why C4 proposes a strength-stratified null instead.

**Ask 4 — two notes.**
- **The execution gate** (seed-0 Δ ≥ −s0) is fixed and outcome-free for the MoA data, which is right. Know its
  operating characteristic: with one seed, Δ's sd is ≈ 1.15 s0, so a C8b with no true accuracy effect is skipped about
  19 % of the time (Δ < −s0). If that's too high a chance of never running the study, gate on the 3-seed mean when C8b
  advances under rule 6. Decide now, not after seed 0.
- **The mean drug** (`k̄` copies of the mean atom vector, mask exactly `k̄`) meets 020 C4. State how `k̄` rounds.

## What I checked and found sound

- **The structural position of the readout.** `NamedPathwayReadout` pools `proj(h)` over member genes. Placed after
  the last perturb block, its activations depend on the drug, so an activation-difference readout is the natural
  primary and gradient × activation is correctly dropped.
- **Separation from §85:** different folds, different models, and neither reading enters the other. The gate is on a
  pre-registered accuracy quantity, and the MoA rows are disjoint from the cc1 dev cells.
- **Rows, positives and Nulls 1/2 are identical to §86,** with the quantiser fitted on training rows for the untrained
  models, which avoids a degenerate untrained forward.

## What I could not assess, and why

- **The actual between-init spread of the unseen-stratum diff.** My ±0.04 is a scaling estimate from §86's
  all-rows spread. C1's calibration measures it directly.
