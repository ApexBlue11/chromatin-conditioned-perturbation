# REVIEW OF PACKET 004
verdict: SOUND-WITH-CAVEATS
reviewed_commit: c1cced5

**This reverses my 003 verdict. Buy the run.**

The crux resolves in your favour, and it resolves from data already in `v9_atom_ablation_CI.json` and
from a line already in `interaction_2x2.py`. The power table in ASK 1 uses the wrong one of the two
intervals your own code emits. On the right one the design is powered for partial rescue at ~3.4 σ, not
1.5 σ — so the outcome you expect is detectable, and option (a)-with-a-caveat becomes unnecessary because
option (b) already exists in your repo.

All six 003 challenges are discharged. One of them discharged by correcting me, which I accept below.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | wrong-quantity | **ASK 1 is answered by `atom_ablation_ci.py`'s own output: the 0.0132 you propagated is the *median* interval, and the paired-mean interval next to it is ~3.9× tighter.** On `unseen_compound` seed 0: `d_median` −0.01556, CI [−0.0242, −0.01044], width **0.01376**; `d_paired_mean` −0.01726, CI [−0.01907, −0.01551], width **0.00356**. Same ratio on the others (unseen_both 5.5×, unseen_cell 5.4×). Propagating the paired-mean width instead gives an interaction CI of **≈0.0050** even assuming zero correlation between the two atom contrasts — and they will be strongly positively correlated, so that is an upper bound. Re-running your own scenario table in paired-mean units (atom effect ≈ −0.0173): full rescue **6.9 σ**, **partial rescue 3.4 σ — detectable**, quarter rescue 1.7 σ. The design is not blind to the outcome C3 makes likely. | Nothing to run. Define the interaction as `interaction_paired_mean` and report `interaction_paired_mean_ci95`, which `boot_2x2` already returns. Re-derive the power table from the paired-mean widths before committing, so the pre-commitment is made against the statistic you will actually read. |
| 2 | MAJOR | stats | **ASK 2 is answered by line 236 of the file you shipped.** You ask whether a paired/blocked bootstrap on the per-row interaction contrast would be tighter than √2 propagation. It would, and you already implemented it: `diff_interaction = diff_atom_with_ctx - diff_atom_no_ctx` then `boot_mean_inter = diff_interaction[idx].mean(axis=1)`, returned as `interaction_paired_mean_ci95`. That is the blocked bootstrap — it resamples rows once and recomputes the *per-row* double difference, so the row main effect and both single-factor row components cancel inside each draw. The `boot_inter` path immediately above it does not have that property, because a difference of four medians does not decompose per row. You have both; only one of them is the quantity your ASK 1 reasoning describes. | Report both, lead with the paired mean, and say in the record which one the decision was taken on. |
| 3 | MAJOR | stats | **A tight paired-mean CI is not a licence to read one run as decisive — and your own C4 table contains the counterexample.** `unseen_cell`: seed 0 gives `d_paired_mean` −0.00155 with CI [−0.00264, −0.00046], **excluding zero**; seed 1 gives **+0.00037** with CI [−0.00049, +0.00123], spanning zero and of **opposite sign**. Two tight row-level intervals, incompatible signs, seed variance dominating. That is the C1 failure from review 001 reappearing under a new estimator. The interaction is a genuine within-run contrast so method rule 7 exempts it from ≥3 seeds for *validity* — but its *magnitude* is not thereby seed-stable, and a marginal interaction from one run would not be interpretable. | Pre-commit, before spending, to a reading rule: interaction ≥3 σ on the paired-mean interval → report as a result from one seed; 1.5–3 σ → explicitly inconclusive, needs a second seed at another 5.7 h; <1.5 σ with `\|dY\|max` confirming both ablations fired → informative null. Write the rule into the packet, not after the number arrives. |
| 4 | MINOR | overreach | **`unseen_cell` was dropped for the wrong reason, though the decision is right.** "All three CIs span zero" is true of `d_median` but false of `d_paired_mean` at seed 0. The defensible reason to drop it is C3's sign flip across seeds, not the width of the weaker interval — and stating it the first way would leave the record saying there is no effect there, when what the data show is an effect too small and too seed-unstable to move. | Restate the exclusion as seed-instability. Costs nothing and keeps the record honest about which statistic was consulted. |
| 5 | MINOR | provenance | **ASK 4 conflates calibration with validity, and the run answers it for free.** Whether the SA-**off** atom effect transfers to an SA-**on** model matters for the *power scenarios* but not for the *design*: the interaction is computed entirely within the SA-on weights and never references −0.0146. The correct internal reference for "did contextualisation rescue atoms" is that run's own `atom_effect_without_context` (S10−S00), which comes out of the same evaluation. | Read the interaction against S10−S00 from the same run, not against the historical SA-off number. Then ASK 4's assumption is never needed. |
| 6 | MINOR | provenance | **C5 is discharged better than you allow — the 1.009× is probably close to right, not compressed.** Parameters overstate FLOPs badly here because the two sequences differ by ~29×: the drug side is ~34 tokens (703,851 atoms / 21,220 compounds ≈ 33, plus the global token) against 978 genes. Drug-side FFN is ≈3.5 % of gene-side FFN, and drug self-attention is 34²/978² ≈ **0.1 %** of gene self-attention. +30 % parameters buys almost no additional compute. So the memory-bound-compression worry, while correct in general, is unlikely to be doing much work here. | Keep the budget guard — it costs nothing and is the right instinct — but ~5.7 h is more likely a good estimate than a floor. |

## On your correction to my C4 — accepted

You are right and I was wrong. I asserted 480 was the population because `--n_eval` defaults to 1500 and
the code takes `min(n_eval, len(idx))`. It was `--n_eval 480` at the call site, unrecorded in the JSON.

The tell was in front of me: **all three splits reported exactly 480**, from strata of 47,002 / 58,796 /
15,083. Three differently-sized populations cannot coincidentally yield identical n. I read an invocation
parameter as a data property, which is the same class of error as reading a config over the executed path
— the thing I am here to catch. Recording it as mine, not as a packet defect, notwithstanding the missing
provenance. `n_eligible` / `n_eval_bound` in `interaction_2x2.py` is the right fix.

The conclusion it supported — that the motivating number needed an interval before the spend — survives,
and C4 turned out to be the most valuable item in review 003 precisely because running it halved the
effect and removed a whole split.

## What I checked and found sound

- **The 2×2 harness computes what it claims.** `evaluate_chunks_2x2` produces S11/S01/S10/S00 on identical
  chunks; `make_atom_ablated_batch` matches `interp_v9.py::ablate_to_mean` (dim-0 chunk mean, broadcast);
  `INTERACTION = (S11−S01) − (S10−S00)` is formed as stated; `|dY|max` is tracked per ablation.
- **The refusal guard is the right guard.** `check_diagonal_available` refuses on an SA-off checkpoint
  rather than returning `S10 == S11` and an interaction of exactly 0.0. A harness that silently
  manufactures a perfect null is precisely how retraction 7 happened, and the refusal record being written
  as an artefact rather than a log line is better practice than the thing it replaces.
- **C3 retracted cleanly.** The premise is withdrawn in the packet and the restated version — a second,
  in-loop re-contextualisation co-evolving with the cell embedding — is what the architecture actually
  supports.
- **C2's operator does what it should.** Delegating to `super()` unless `diagonal=True` makes the default
  path bit-identical by construction rather than by test, cross-atom flow is removed at exactly 0.00e+00,
  parameter count is unchanged, and `|dY|max` 1.1266 shows it is not a no-op. Checking the fully-masked
  padded query row for NaN was not asked for and is the right kind of unprompted check.
- **C4's numbers are internally consistent.** `frac_rows_intact_better` 0.3233 (unseen_compound) and
  0.2847 (unseen_both) sit correctly against negative paired means, and `fired: True` with `|dY|max`
  1.30–2.49 confirms the atom branch is live in every cell.
- **ASK 3, answered from your own table: `unseen_compound` is the better primary split.** Cross-seed range
  0.0069 against `unseen_both`'s 0.0138, with a slightly larger paired-mean effect (−0.01726 vs −0.01444)
  and a comparable interval. `unseen_both` has seed range ≈ its own CI width, which is the signature of
  seed variance dominating. Your choice is right for a reason the packet states only in passing.
- **GPU hours committed to date: 0.** Three design iterations and a halved effect size, bought with
  inference on checkpoints that were already on disk.

## What I could not assess, and why

- **The actual interaction CI.** It cannot exist until an SA-on model does. My 0.0050 is a propagated
  upper bound from the paired-mean widths; the realised width should be smaller by however much the two
  atom contrasts correlate, which is unmeasurable without the run.
- **Whether the SA-on model's own atom effect resembles −0.0173.** Per C5 it does not need to for
  validity, but the power scenarios scale with it, so a much smaller SA-on baseline would shrink partial
  rescue back toward the margin. This is the residual risk in the buy decision and it is not removable
  in advance.
- **Whether 1.009× holds on T4×2.** My FLOP argument says the true ratio is small; it does not say it is
  1.009. The budget guard covers this.
