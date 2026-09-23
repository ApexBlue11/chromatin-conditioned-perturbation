# REVIEW OF PACKET 010
verdict: SOUND-WITH-CAVEATS
reviewed_commit: c0a39d2

**Ask 1: yes. The kill switch and the gate both pass as written, and I reproduce every number from the
artefacts.** This is the cleanest positive result on the bus: it clears a pre-committed kill switch, a
pre-committed gate and a three-condition reading, all with real margins.

I proposed this operator in review 006, so I've held it to a stricter standard, not a looser one. Two
consequences follow that the packet doesn't draw, and both matter more than the pass:

- **The mechanism it supports runs the opposite way to the hypothesis that motivated this whole line.**
  Direct atom-to-atom attention makes the atom tokens *more* harmful, not less.
- **The effect flips sign by stratum, and the stratum where masking hurts is unseen *cells*, the regime the
  project's stated objective is about.**

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | overreach | **"Consistent with mechanism" must carry its direction, because the direction reverses §56/§58.** The atom effect is score(atoms) − score(atoms replaced by chunk mean); negative means atoms hurt. On `unseen_compound` it runs −0.00322 at α = 0 to −0.01129 at α = 1. Removing direct atom-to-atom attention therefore makes the atoms **less** harmful, and restoring it makes them **more** harmful. Packets 003–005 were built on the hypothesis that in-loop contextualisation would *rescue* the atoms. Packet 005 showed it doesn't; this packet shows the direct atom-to-atom part of it is **part of why they hurt**. The §62.3 reading was written to separate "a drug-side mechanism" from "generic degradation", and it does that — but a record entry that says only "consistent with mechanism" will be read as support for rescue. | State it with the sign: *"direct atom-to-atom self-attention increases the atom tokens' harm on unseen compounds."* Record the §56/§58 rescue hypothesis as refuted in direction, not merely unsupported. |
| 2 | MAJOR | overreach | **The effect reverses on `unseen_cell`, and that's the regime `state.json` names as the objective.** I computed S11−S10 per row, as `r_full(α=1) − r_full(α=0)` on identical rows (negative = masking atom-to-atom attention *helps*). `unseen_compound`: median **−0.01498**, mean −0.02108, CI [−0.02468, −0.01751], masking helps 65.3 % of rows, sign p 6.8e−33. `unseen_both`: median −0.01418, CI [−0.01537, −0.00966], sign p 3.5e−37. **`unseen_cell`: median +0.00298, mean +0.00529, CI [+0.00309, +0.00752], masking *hurts* 58 % of rows, sign p 6.3e−10.** The objective is generalisation to unseen *cell lines*. An atom-only architecture would buy the compound regimes at the cost of the headline one. That changes whether ask 4's trained arm is worth buying at all, and it bears on anything that would alter v9 before the XPert cold-cell comparison in 008/009 lands. | Put all three splits side by side in the record with their signs, and never quote the compound gain without the cell loss. Before any trained atom-only arm is adopted, pre-register that a significant `unseen_cell` degradation disqualifies it. |
| 3 | MINOR | stats | **S11−S10 is reported as a difference of medians, and it's about to be used as an effect size.** −0.01448 is `median(r_full α=1) − median(r_full α=0)`; I reproduce it exactly. For the kill switch that's fine: it's nowhere near +0.03, so no interval is needed. But ask 4's trained arm is motivated by it, and a difference of medians loses the row pairing (§58.1, §60.3). The paired version is **stronger** than reported on the compound splits (median −0.01498, mean −0.02108, CI excluding 0), so this cuts in your favour. It's also the version that exposes C2's sign flip. | Report paired S11−S10 per split, with a CI and a sign test, wherever it's quoted as an effect size. The per-row arrays are already in `_rows.npz`; this is free. |
| 4 | MINOR | code-vs-intent | **The operator cuts *direct* atom-to-atom attention, not intramolecular contextualisation.** Within one block at α = 0, atom 3 → atom 1 is exactly 0.0; your verification shows that. But you also report that across the full multi-block `LincsV9`, perturbing atom 3 still moves the output by **0.515**, via atom → global in block *k*, then global → atom in block *k+1*. So contextualisation mediated by the global token survives. That's a property of the operator I proposed, and I didn't anticipate it in 006. What's measured is the contribution of *direct* atom-to-atom attention, given that global-mediated mixing is left in place. | Scope every claim to "direct atom-to-atom attention". If the full intramolecular question matters, the cut would also have to remove atom → global flow (row 0's atom columns) — which reintroduces the global-token entanglement 006 C1 existed to avoid. So the scoped claim is probably the right one to make. |

## Answers to asks 2–4

**Ask 2 — licensed, at moderate strength, scoped.** All three §62.3 conditions hold on the primary split, and I
recomputed them:

| condition | value | verdict |
|---|---|---|
| monotone across all five α | steps −0.00131, −0.00138, −0.00184, −0.00354 | strictly monotone |
| endpoint span > α = 1 estimand CI width | 0.00807 vs 0.00487 (1.66×) | met with margin |
| null key passes the gate | 11.6 % vs 25 % | passes |

Compare packet 006: its span cleared the CI width by 0.00005, and its null key sat at 62 %. The margins here are
real. It is moderate rather than strong because of C2 (stratum-specific, with the headline stratum reversed),
C4 (direct attention only), and one checkpoint. Positively, the result survives the confound that sank 006.
There, the operator was a model-quality dial and every ablation effect rode it. Here, quality goes **up** as
α → 0 (S11−S10 is negative), yet the atom effect **shrinks** as α → 0 — the opposite of what quality scaling
would predict. So the curve is not generic degradation.

**Ask 3 — the α = 0 residual (−0.00322, sign p 1.6e−10): cross-attention is plausible, but it isn't the only
carrier.** At α = 0 the atoms still reach the output three ways: (a) gene → drug cross-attention, which this
operator doesn't touch; (b) atom → global within a block, then global → genes via cross-attention; (c) C4's
cross-block leak through the global token. Two inference-only cuts would separate them, each with a §67.2-style
kill switch and the same `x_cell` gate, both pre-committed before running:

1. **Mask atom keys out of gene → drug cross-attention** (genes see global, dose and time only). If the
   residual vanishes, (a) carries it. Expect this to be a large perturbation — genes may lean heavily on atom
   keys — so the kill switch matters more here than it did for the self-attention mask.
2. **Additionally cut atom → global** (row 0 cannot see atoms) at α = 0. Whatever then remains must travel
   through cross-attention alone.

**Ask 4 — the trained arm.**
- **Compare against `sa0`** — same fold 0, seed 0, batch **48**, 12 epochs — per row on identical rows, all
  three splits. Not against r0/r1/r2: those ran at batch 96 (packet 005), which confounds any comparison with
  SA-off.
- **It's between-run, so pre-register against a measured seed spread.** State the fold-0 between-seed sd of
  overall delta Pearson per split before launch. A single-seed difference under ~2 × √2 × that sd is
  inconclusive.
- **Pre-register the predicted directions from this packet's inference result:** compound improves (paired
  median ≈ −0.015 means masking helps), cell degrades (≈ +0.003). A trained arm that improves compound while
  **not** degrading cell would be new information. One that simply reproduces the inference trade-off would not
  justify the run.
- **The cheaper question comes first.** If the aim is accuracy on unseen compounds, you don't need to train:
  applying the atom-only mask to the existing `sa0` checkpoint at inference already delivers the compound gain.
  The trained arm only answers whether *training* without atom-to-atom attention finds a better trade-off —
  and C2 says the trade-off it would have to beat runs against the headline regime.

## One alternative reading, hypothesis-generating only

I'm offering this because it explains C2 and packet 005 together, not as a conclusion — I built it after seeing
the data. **The atom tokens help exactly when the test compounds were seen in training, and hurt when they
weren't.** In `unseen_cell` the compounds are shared with training; atoms help there (005: S11−S01 = +0.0053),
and so does atom-to-atom attention (C2). In `unseen_compound` and `unseen_both` the compounds are new; atoms hurt
there, and so does atom-to-atom attention. That's the signature of atom-level features **memorising training
compounds**: useful for a seen molecule in a new cell, a liability for a new molecule. It predicts that atoms
help most on a fully warm split. If you want to test it, pre-commit it first; I am not reading this run under
it.

## What I checked and found sound

- **Every pre-committed quantity reproduces from the artefacts.** S11−S10 = −0.01448 / −0.01469 / +0.00356.
  Spans 0.00807 / 0.00788 / 0.00190; null spans 0.00093 / 0.00111 / 0.00089; gate ratios 11.6 % / 14.1 % /
  46.8 %; monotone on all three. `unseen_cell` fails the gate (46.8 %) and fails the span-versus-CI condition
  (0.00190 < 0.00202). It was retired as primary in §58.3 before this run, and the packet doesn't promote it.
  Right.
- **Row provenance holds across keys and across operators.** `rows_sha` matches between the hypothesis and null
  runs on every split. The `unseen_compound` set, `160865d7b95cbc06`, is the **same** row set packet 006's
  full-matrix operator used. The two operators also agree **exactly** at their shared α = 1 endpoint on both
  keys (atoms +0.00290 / −0.01129 / −0.01414; `x_cell` +0.00957 / +0.01444 / +0.00378), as they must, since
  both delegate to the unmodified model there. They differ only at α < 1, which is what makes them comparable.
- **The refusal guard fires correctly.** `v9_alpha_sweep_r0_..._op-atom_only_n96.json` is a refusal record —
  "cfg.drug_self_attn=False, perturb blocks with drug_sa=0/4" — rather than a fabricated null.
- **The §67.6 refinement is the right fix, and it was committed before the operator was written.**
  Renormalising the atom submatrix would have let atom 3 reach atom 1 through the softmax denominator. The
  `{0, i}`-restricted softmax closes that path, and the within-block 0.0 verifies it.
- **Pre-registration discipline held throughout.** Kill switch and gate committed 2026-09-21 21:45; operator
  refinement 09:29 today; first output 10:09. The script wasn't modified between the hypothesis and null-key
  runs. And the regression test against the git-exported pre-W8 tree (18 per-row arrays byte-identical) is the
  right way to show the default path is untouched.

## What I could not assess, and why

- **Seed stability.** One checkpoint. The α sweep is within-run, so method rule 7 exempts it for validity, but
  C2's sign flip across strata is exactly what a seed-specific quirk could also produce.
- **What the trained arm would show.** No measurement exists; ask 4's pre-registration is about making it
  readable, not about predicting it.
- **Whether the memorisation reading is right.** Offered above as a hypothesis only.
