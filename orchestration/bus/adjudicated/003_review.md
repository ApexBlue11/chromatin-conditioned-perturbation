# REVIEW OF PACKET 003
verdict: NOT-SUPPORTED
reviewed_commit: f8e185c

Scoped to what the verdict means for a design review: **the design as written is not decisive, so the
spend is not yet justified.** It is not that the idea is bad — the hypothesis is worth testing and
bringing it here before the spend was right. Four defects, three of which cost nothing to fix, and one
that may remove the need for the run altogether.

Sequencing is the whole recommendation: **C4 is free and can make C1–C3 moot.** Do it first.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | missing-null | **The 2×2 is missing its fourth cell, so this measures two main effects and calls their comparison an interaction.** You propose ablating atoms (gives S01) and ablating `drug_sa` (gives S10), both against the intact model (S11). The quantity you actually want — does `drug_sa` change the *value of atoms* — is `(S11−S01) − (S10−S00)`, and **S00, both ablated, is not in the design**. Without it, "does the atom contribution revert toward −0.025?" can only be answered by comparing against §37's SA-**off** run, which is the between-run comparison the design is built to avoid. Answering ask 1 directly: you moved the problem, you did not remove it. | Add S00 — both ablations applied in the same forward pass. Zero extra GPU, one extra eval pass. Then report the interaction term with its own `|dY|max`, not the two main effects side by side. |
| 2 | MAJOR | code-vs-intent | **Mean-ablating `drug_sa` removes capacity, not just contextualisation — which re-admits the +30 % confound inside the run.** `_DrugBlock.forward` is two residual branches: `D + attn(n1(D))` then `D + ff(n2(D))`. Neutralising the module's output kills the SwiGLU as well as the cross-atom mixing, so the ablated arm is not "the same model without contextualisation", it is a smaller-capacity model. That is precisely the confound §003 claims to have escaped by staying within one run. Answering asks 2 and 3 together: the choice of neutralisation is what decides whether capacity confounds the within-run contrast, and ablate-to-mean is the wrong one here. | Ablate to **diagonal attention** — mask the attention matrix to the identity so each atom attends only to itself. Parameters, FFN, residual scale and the per-atom pathway are all preserved; only cross-atom information flow is removed, which is exactly the hypothesis. Validate it with the harness you already have: under diagonal attention, perturbing atom 3 must move atom 1 by exactly `0.00e+00`, matching the arm-off case in `test_drugsa_v9.py`. Report `|dY|max` for the diagonal ablation so a true null stays distinguishable from a module that never fired. |
| 3 | MAJOR | overreach | **The stated mechanism — "a BAG of independent per-atom Uni-Mol vectors with no intramolecular structure" — is false**, in the `_DrugBlock` docstring and in the packet's FUNCTION section. `drug_atom_reprs.npy` holds Uni-Mol **`atomic_reprs`** (`drug/scripts/step8_integrate_atom_tokens.py`, kernel `lincs-atom-tokens`, `remove_hs=True`): the per-atom outputs of Uni-Mol's transformer encoder, which attends over every atom with a 3D distance bias. Atom *i*'s 512-d vector **already encodes its molecular environment**. `linear(atoms)` is a linear map of already-contextualised representations, not of raw atom features. What v9 lacks relative to XPert is a *second*, in-loop re-contextualisation that co-evolves with the cell embedding across blocks — real, but a far weaker deficit than "no intramolecular structure". The attached falsifiable prediction (−0.025 flips positive) is calibrated against a baseline that does not exist, and a null result would be over-read as "atom-level attribution in this architecture is dead" when it would only license "a second round of contextualisation adds little on top of Uni-Mol's". | Restate the mechanism and shrink the predicted effect accordingly, before the run rather than after. If you want the strong version of the premise, the free test is whether a molecule's `atomic_reprs` are predictable from its own CLS token plus atom identity — if they are largely not, the structure is already there and C3 stands. |
| 4 | MAJOR | stats | **The effect being explained has no error bar, and n = 480.** All three figures (−0.00671 / −0.02549 / −0.02187) come from one checkpoint, one seed (`ckpt_v9_fold0_seed0.pt`), on **480 signatures per regime** — and that is the whole split, not a cap: `--n_eval` defaults to 1500 and the code takes `min(n_eval, len(idx))`. `probe_v9.py` contains no bootstrap and emits no interval (no `bootstrap`/`ci95` anywhere in it). So ~5.6 GPU-h is being committed to explain a number whose uncertainty has never been computed. Answering ask 5: **yes, there is a free version, and this is it.** | Re-run `probe_v9.py` on the **existing** checkpoints with a paired row bootstrap over the 480 rows, and across the fold0 seeds you already hold in `external/v9_checkpoints/`. Inference only, no training. If the −0.025 interval spans zero, or if it moves across seeds by more than its own width, there is nothing to explain and the run should not be bought. This is the single highest-value action in the packet and it costs no quota. |
| 5 | MINOR | provenance | **The ~5.6 h projection is a floor, not an estimate.** It scales §29.1's 0.0091 s/row-epoch by training rows — but §29.1 was measured with the module **off**. Adding `_DrugBlock` to every `PerturbBlock` adds attention that is O(A²) in atoms per block plus a SwiGLU, on top of +30 % parameters. The direction is known even if the magnitude is not, so the headline cost of a cost review is understated by an unmeasured amount. | Time 50 optimiser steps with `drug_self_attn=True` at full width and rescale. Two minutes, and it converts the only unmeasured number in the COSTS section into a measured one. |
| 6 | MINOR | provenance | **The cheap-split option in ask 4 is not free.** `probe_v9.py` takes `--ckpt` but no `--bundle` or `--split`; it is wired to the fold0 pipeline. Running the ablations on `split_cold_drug_1` therefore costs porting work before it costs 1.7 h, and that work is not in the cost table. | Price the port. Note that C1's fix makes the design self-contained — it no longer needs to compare against the historical −0.025 — so the split choice is now a question of which regimes you want, not of matching §37. |

## What I checked and found sound

- **The XPert architectural claim is correct, and I initially thought it was not.** My first grep covered
  only `external/xpert/code/XPert/*.py` and returned nothing, which looked like retraction-class 8. It is
  in `models/`: `model_utils.py:359` constructs `self.drug_SA` inside `crossEncoder.__init__`, `:364`
  calls it in `forward` before the cross-attention at `:371`, and `model_XPert.py:44` holds a
  `ModuleList` of `crossEncoders` indexed per layer at `:65` with `drug_embed` **reassigned** each time —
  so contextualisation does compound across blocks, exactly as the packet describes. Read from the
  executed path, not the config. The claim holds.
- **The §47.6 self-correction is the right kind.** +11.38 % measured at d_model 64 was wrong for the full
  model; +30.06 % at d_model 256 is a 3× larger confound and you surfaced it against your own interest.
- **`test_drugsa_v9.py` tests discrimination, not existence.** Perturbing atom 3 moves atom 1 by
  \|d\|max 0.2754; with the arm off it is bit-identical; perturbing a **padded** atom moves a real one by
  `0.00e+00`. That last check is the one that catches a mask bug, and it is the retraction-7 lesson
  applied correctly rather than recited.
- **The probe reports `dY_max` and `fired` on every ablation.** `atom_tokens` shows dY_max 1.77–2.07 with
  `fired: True` across all three regimes, so the negative number is a live component rather than a dead
  branch. The standing ablation rule is satisfied; my C4 is about the interval, not the firing.
- **Defaulting the flag off and keeping `test_v9.py` at 55/55 with the arm off** means the existing
  results are not silently perturbed by this change.
- **Bringing a design for review before the spend.** The +30 % correction alone would have confounded an
  accuracy comparison that had already been run.

## What I could not assess, and why

- **Whether +30 % capacity changes headline accuracy.** Out of scope for a within-run design, and it is
  why a *positive* interaction would still not be conclusive. Worth noting the asymmetry you half-stated:
  a **null** interaction needs no capacity control and is decisive on its own; only a positive result
  obliges you to buy the matched-capacity arm. The design is a sound one-sided screen — once C1 and C2
  are fixed.
- **Actual wall clock with the module on.** Nothing measured exists; see C5.
- **Whether 55,385 rows (`split_cold_drug_1`) is enough to learn to use contextualisation at all.** A null
  there could mean the mechanism is absent or that the data are too few to train it, and the packet has
  no basis for separating those. That is an argument for fold0 that the packet does not make.

## Recommended order, since the asks are about spend

1. **C4 first — costs nothing.** Bootstrap the existing probe, across the seeds already on disk. If
   −0.025 is not solidly non-zero and stable, stop; there is no phenomenon.
2. **C3 — restate the mechanism** and re-derive what effect size would actually count as confirmation.
3. **C2 and C1 — fix the ablation operator and add S00.** Both are code changes, no quota.
4. **C5 — measure the real step time**, then choose a split.
5. Only then spend.
