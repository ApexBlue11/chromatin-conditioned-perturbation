# REVIEW OF PACKET 024
verdict: SOUND-WITH-CAVEATS
reviewed_commit: c3a42ba

**The batch is well formed.** One flag per candidate, byte-identical when off, and a regression test asserting that.
Seed 0 first, the rules from §85.2, C5 and C7 held back until their coverage is settled, and C8b's MoA test ordered in
advance.

Two definitions need fixing before build:
- **C2 would test a different hypothesis from the one it cites (C1).**
- **C4's weighting isn't written down.** It says "as in their eq." (C2).

The rest are exact enough, provided each carries an acceptance test a worker's code can be checked against (C4).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **C2 removes the control profile entirely. It doesn't test "raw control, no learned encoder", which is what TxPert's Fig. 7 is cited for.** `ControlEncoder` (`modules_v9.py:169-188`) is `gene_tok + expr(x)` followed by `l_control` self-attention `_GeneBlock`s. Those blocks are the **learned basal-state encoder**. Setting `use_matched_ctl = use_cell_ctl = False` zeros both control views. The delta head reads only the gene tokens (`MultiTaskHeads.forward`, `modules_v9.py:412-417`: `d = self.delta(h)`), and `x_ctl` enters only as `abs = x_ctl + d`. So under C2 the **delta prediction sees no control information at all**. That tests "is the basal state needed", which §37 already answers: ablating the matched control costs +0.105 on `unseen_cell`. It isn't the TxPert contrast between a raw control profile and a learned encoder of it. As written, C2 spends ~1.7 GPU-h on a nearly foregone result and would be recorded as evidence about the wrong hypothesis. | Redefine C2 as **`l_control = 0`**: both control views keep their per-gene expression embedding, `h = RMSNorm(gene_tok + expr(x))`, and lose the attention blocks. That's "the raw control profile, no learned encoder". It needs no new code path (an empty `ModuleList`, and the stochastic-depth divisor is already guarded). Acceptance test: the control stream has zero blocks, and the prediction still changes when `x_ctl` changes. If you also want the "no control" arm, it's a separate candidate with its own prior. |
| 2 | MINOR | code-vs-intent | **C4's weighting isn't defined in the packet.** "α = stopgrad((L_all+L_DE)/L) as in their eq." leaves `L` undefined, doesn't say how `α_all` differs from `α_DE`, and doesn't say whether the α are per batch or per row. A worker can't implement that exactly, and a reviewer can't check it. | Write the two α's as explicit formulas. For example, if it is PertAdapt's: `α_all = stopgrad(L_all / (L_all + L_DE))`, `α_DE = stopgrad(L_DE / (L_all + L_DE))`, per batch. Whatever the form, write it out and cite the equation number. Acceptance test: on a fixed batch, the α's equal the formula, and `α.requires_grad` is False. |
| 3 | MINOR | stats | **C3's ListNet is sign-asymmetric, and the metric isn't.** `softmax(z(y_Δ))` over 978 genes puts its weight on the most **up**-regulated genes. Strongly down-regulated genes get almost none, although per-row Pearson rewards them equally. That may be intended, and it's ExPO's form, but it's a design choice with consequences. Also, standardising ŷ per row divides by its sd, which is unstable on a near-flat prediction. | Decide it knowingly and state it: signed ListNet (as defined), or symmetric (average the loss on `z` and `−z`), or ranking on `\|z\|`. One of them, fixed now. Add an ε to the row sd. Acceptance test: the term equals a reference implementation on a fixed batch. |
| 4 | MINOR | code-vs-intent | **Give each candidate an acceptance test, so "exact enough to check" is literal.** Beyond the default-off regression test: **C1**, the prediction is invariant to the atom inputs (dY = 0 when atom tokens are replaced), which also proves they're removed rather than zeroed-but-present; zeroed atom tokens would still enter as bias tokens (`ln_atom(w_a(0)) + type_atom`). **C6**, the sign head's parameters get no gradient from the prediction losses, and `out` is identical with the head present or absent. **C8b**, the post-perturbation activations change with the drug (Δa ≠ 0) while the pre-perturbation ones don't; `aux['pathway_activations']` still returns the **pre**-perturbation layer, since §85's P4 gate reads it, with the new layer under its own key; and state its stochastic-depth rate (the first readout uses `sd_path`). | Write these into the worker brief as asserts. They cost seconds and remove any room for reading the definitions differently. |

## Answers to the asks

**Ask 1 — yes for C1, C6 and C8b once C4's tests are added. C3 once its sign form is fixed. Not yet for C2 (C1) or C4
(C2).**

**Ask 2 — one fixed value per candidate is right, but fix it by a rule, not a number chosen for its roundness.** A grid
multiplies comparisons, and with them §85's false-acceptance rate. But "0.1" means different things for different
terms. At init the ListNet cross-entropy is ≈ log 978 ≈ 6.9, most of it a constant. BCE is ≈ 0.69. The gradients they
contribute relative to the Huber and PCC terms are unknown. A null at an effectively negligible weight would be recorded
as "ListNet doesn't help". So fix each auxiliary weight by a rule applied to the **baseline's** logged gradient norms
(outcome-free): e.g. `w` such that the auxiliary term's gradient norm is 10 % of the delta-loss gradient norm at the
start of training, computed once from P2's seed-0 setup and frozen. State any null as "at the pre-registered weight".
k = 50 and τ = 1 are fine as fixed choices.

**Ask 3 — acceptable. It gates whether the MoA study exists, not how it's read.** The MoA data (fold-0 test rows) are
disjoint from the cc1 dev cells, so nothing in the accuracy screen can leak into the MoA reading. Two conditions:
- **Define "does not fail" exactly.** Either not dropped at §85.2 rule 6's first seed, or accepted under rules 6–8.
  They are very different bars.
- **Write the MoA pre-registration now, or at least before C8b's screen is read.** Only its execution is gated. That
  way its design can't be tuned to what the screen showed.

**Ask 4 — nothing else blocks.** One arithmetic note for the ledger. With s0 = 0.00169, `2·√(s0²/3 + s_v²/3)` ≈ 0.0028
if s_v ≈ s0. So the 0.003 floor is what binds acceptance, and the advance threshold is max(2 s0, 0.003) = 0.003. Both
are as intended.

## What I checked and found sound

- **The heads and loss** (`modules_v9.py:412-417`; `model_v9.py:203-243`): the delta head reads gene tokens only, with
  `abs = x_ctl + delta`. The delta loss is Huber plus a PCC term. C4's modification applies to the Huber term exactly
  where the packet says.
- **`ControlEncoder`:** an expression embedding plus `l_control` attention blocks. So a raw-control variant is well
  defined and cheap (C1).
- **The fixed common command** (`--dp_seed_mode distinct`, `--dev_cells 6 --dev_seed 0`, 12 epochs, `d_model` 256,
  batch 48). The lockstep fix is carried into every candidate, so all dev comparisons share i.i.d. masks with P2.

## What I could not assess, and why

- **The relative gradient scale of each auxiliary term.** That's measured, not argued (ask 2).
- **PertAdapt's exact equation.** The packet cites it without reproducing it (C2).
