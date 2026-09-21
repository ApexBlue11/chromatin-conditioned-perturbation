# REVIEW OF PACKET 005
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 7e36626

Scoped, because the packet contains one compromised measurement and one clean one:

- **The 2×2 interaction is not interpretable from this run.** The pre-committed estimand returns 4.2 σ on a
  key with no mechanistic path, and on `unseen_both` the null key beats the hypothesis key (6.1 σ against
  1.7 σ). By the pre-commitment discipline that is disqualifying, and C2 explains why without rescuing it.
- **The atom effect under full attention is clean and it answers the objective.** `S11−S01` on
  `unseen_compound` is −0.01814 [−0.02635, −0.00916]; per-row mean −0.01901, median −0.01129, ablation
  improves 66 % of rows. No `S10`, no `S00`, no interaction, no cross-run comparison. **In a model where
  drug self-attention is present and trained, ablating the atom tokens still improves accuracy on unseen
  compounds.** §57's prediction was that this would stop. It did not.

The verdict is SOUND-WITH-CAVEATS rather than NOT-SUPPORTED because the packet claims nothing it has not
measured, and because running a null key and then reporting that it came out significant is the most
valuable thing anyone has put on this bus. The design failed; the conduct did not.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | missing-null | **The pre-committed estimand fails its own negative control.** On the primary split `interaction_paired_mean` is −0.00938 (4.7 σ) for `atoms` and **+0.00616 (4.2 σ) for `x_cell`** — comparable magnitude, comparable significance, on byte-identical rows. On `unseen_both` it is worse: the null key returns **+0.00519 at 6.1 σ** while the hypothesis key returns −0.00258 at **1.7 σ**, i.e. the control is larger and more significant than the thing it controls for. Answering ask 2b directly: this does **not** license reading the atom interaction as specific. It means the estimand returns high-sigma values where no mechanism exists, so a high-sigma value on the hypothesis key carries no information about mechanism. Note the control is not null "by construction" in any case — for a nonlinear model the second-order mixed partial between any two inputs is generically non-zero unless the model is additively separable in them, so `x_cell` was never guaranteed zero. It nonetheless did its job. | Nothing from this run. Pre-commit a new estimand (C2), re-run the 2×2 and the null key together, and require the control to be null on the new estimand *before* the hypothesis key is read. |
| 2 | MAJOR | stats | **Why it failed is diagnosable, and it is my error as much as yours.** The per-row contrast is heavily skewed and the mean is not a location estimate of it. Reproduced independently: `atoms` has mean ≈ median (−0.00938 / −0.00803), 42.2 % of rows positive, sign test **1.7e−09**, skew −1.74 → a broad distributional shift. `x_cell` has median **+0.00016** against mean +0.00616, 50.5 % of rows positive, sign test **0.70**, skew +1.58 → a tail artifact that the mean reports as 4.2 σ. The discriminating statistics — **median of the per-row contrast, and the sign test on it** — separate them cleanly on every split. **Review 004 C1 pushed you to the paired mean over the difference-of-medians. That was right about which of those two is better and silent about the paired mean's non-robustness to skew; the correct recommendation was the median of the per-row contrast, which is neither of the two quantities you emit.** Note this is a *third* estimand: median of per-row contrast (−0.00803) is not the difference of four medians (−0.00116). | Pre-commit `median(per-row contrast)` plus the sign test as the estimand of record, keep the paired mean alongside, and treat **mean/median disagreement as the built-in skew diagnostic**. Do **not** re-read this run under the robust estimand — it was chosen after seeing the data, so it is hypothesis-generating for the next run and nothing more. Your own note that "the estimand was fixed in 58.4 before the spend and the sign test was not" is the correct instinct; hold it. |
| 3 | MAJOR | wrong-quantity | **Ask 2: yes, the magnitude compromises it, and the operator is my recommendation.** `S11−S10` is +0.0905 / +0.1168 / +0.0980 — five to seven times the atom effect being measured. Diagonal masking does not produce "this model without contextualisation"; it produces a model whose drug pathway is broken far outside anything it saw in training. Review 004 C2 argued for diagonal attention because it preserves parameters, FFN and residual scale while removing only cross-atom mixing. It does all of that — and I did not anticipate that removing cross-atom mixing would itself be catastrophic, which makes the ablated arm a poor stand-in for a model that never had it. | A dose-response, inference-only on the existing checkpoint: blend `α·A + (1−α)·I` for α ∈ {0, 0.25, 0.5, 0.75, 1} and plot the atom effect against α. Smooth and monotone ⇒ mechanism. Flat then collapsing ⇒ the interaction is an out-of-distribution artifact. This costs no training and would settle ask 2 properly. A sharper null key would also be **magnitude-matched** — `x_cell`'s main effect (+0.0355) is about twice the atom main effect (−0.0181), so it is not like-for-like even before C1. |
| 4 | MAJOR | overreach | **Ask 3: nothing can be concluded from a contrast whose sign is stratum-dependent at high sigma, and the sign pattern is not noise.** `unseen_cell` gives +0.01200 at 8.4 σ with sign test 4.6e−21; `unseen_compound` gives −0.00938 at 4.7 σ with sign test 1.7e−09. Both are broad, per-row-consistent, and opposite. A single mechanism does not do that. What it tracks instead is visible in the main effects: atoms *help* on `unseen_cell` (`S11−S01` = +0.00530) and *hurt* on both compound splits (−0.01814, −0.02637), so the interaction is inheriting the stratum's answer to a different question. Relevant to the hypothesis: on the pre-committed primary split the interaction is significantly **negative**, i.e. full attention makes the atom tokens slightly *more* harmful — the opposite of §58's prediction. | Covered by C1 and C3; no separate experiment. If the dose-response is run, run it on both `unseen_cell` and `unseen_compound`, since a mechanism should not reverse between them. |
| 5 | MINOR | provenance | **Ask 5 answered: the batch discrepancy does not reach the within-checkpoint quantities.** All four cells come from one set of weights, in one loop iteration, at `--batch 48`, with chunk-mean ablation semantics matched to §57 — so batch size is held constant *inside* the contrast by construction. It does, however, make the training-metrics table non-comparable: SA-off 0.5846 vs SA-on 0.5703 on `unseen_compound` confounds batch 96 vs 48, +26.9 % parameters, and the architecture change simultaneously. The packet does not claim otherwise, but the table invites the reading. | Label the cross-run table non-comparable in the record, as §D requires, rather than leaving it to the reader. |
| 6 | MINOR | provenance | **The pre-training guard printed a parameter count that is wrong by 24×.** It reports "32,868 params/block". Loaded from the checkpoint directly: `drug_sa` is **3,146,016** parameters across **40 tensors in 4 blocks = 786,504 per block** (qkv 786,432 + o 262,144 + three SwiGLU projections at 698,368 each + norms). The packet's headline delta and total are exactly right and I verified them; it is the guard line that is wrong, which matters because that guard is cited as evidence the module was wired correctly. Separately, the SA-off baseline changed between packets — 004 quoted 10,466,725 → 13,612,741 (+30.06 %), 005 quotes 11,706,043 → 14,852,059 (+26.9 %) — with an identical 3,146,016 delta and no explanation of the 1,239,318 base change. | Fix the guard's arithmetic, and record what changed in the SA-off baseline between the two packets. |

## What I checked and found sound

- **Every per-row statistic reproduces, computed independently from the npz dumps.** mean, median,
  fraction positive, sign-test p, skew and sigma match the packet to the digits shown on all six
  split × key combinations. The sigma convention (CI width / 3.92) also checks out — 8.42 / 4.74 / 1.68
  for atoms and 0.83 / 4.16 / 6.05 for `x_cell`.
- **The row-identity guarantee is real, not asserted.** I verified `rows` arrays byte-identical between
  the atoms run and the `x_cell` run on all three splits, and `r11` byte-identical as well. The control
  genuinely runs on the same rows, chunks and weights.
- **The parameter accounting verifies exactly** against the checkpoint: 14,852,059 total, 40 `drug_sa`
  tensors across 4 blocks summing to 3,146,016, `cfg['drug_self_attn'] = True`, d_model 256 / d_ff 1024.
  C6 is about the guard's print statement, not about this.
- **A detail that supports the null-key logic and that the mean hides:** the `x_cell` contrast has
  **96 / 0 / 108 exact zeros** while the atoms contrast has **0 / 0 / 0**. On a large minority of rows the
  `x_cell` ablation and the context ablation genuinely do not interact at all — visible in the median and
  the sign test, invisible in the mean.
- **The reading rule was pre-committed in §58.4 before the spend, and the packet reports against it rather
  than around it** — including the parts that go badly. It also explicitly flags that the sign test was
  *not* pre-committed. That is the discipline working.
- **Running a null key at all**, on identical rows, and disclosing that it came out significant on the
  pre-committed estimand. This is the finding of the packet, and you found it, not me.
- **Determinism rerun** from the same checkpoint reproducing every field to five decimals; the refusal
  guard that will not return `S10 == S11`; `fired_atom` / `fired_context` true with `|dY|max` 3.5–9.2 on
  every cell, so nothing here is a void rather than a null.
- **Review 004's C1 and C2 both discharged as specified**, and the packet's own §58.1 correctly records
  that my power table objection was about the wrong interval. The diagonal operator does exactly what it
  claimed at 0.0e+00.

## What I could not assess, and why

- **Whether any of this is seed-stable.** One SA-on checkpoint. The interaction is a within-run contrast so
  method rule 7 exempts it for validity, but C4's stratum-dependent sign is precisely the pattern that
  would also be produced by a seed-specific quirk, and there is no second seed to tell them apart.
- **Whether a model trained without `_DrugBlock` shows the same atom effect.** That is the between-run
  comparison the design was built to avoid, and I am not asking you to buy it.
- **Whether the dose-response in C3 would be monotone.** It is the cheapest remaining discriminator and it
  does not exist yet.
- **What 32,868 refers to.** It is not the `_DrugBlock` parameter count and I could not identify it from
  the artefacts.

## The statement I think this run actually supports

> With drug self-attention present and trained, ablating the per-atom drug tokens still **improves**
> median row Pearson on unseen compounds — `S11−S01` = −0.01814 [−0.02635, −0.00916], per-row median
> −0.01129, improving 66 % of rows. A second in-loop contextualisation pass does not rescue the atom
> tokens.

That uses no `S10`, no `S00`, no interaction, no null key and no cross-run comparison, so it survives
C1 through C4 intact. It answers the objective, and it is the negative result §58 said would justify
deleting the atom tokens with a mechanism rather than as a bare empirical fact — with the mechanism now
being "even contextualised, they do not help", which is weaker than the original story but is what the
evidence carries.
