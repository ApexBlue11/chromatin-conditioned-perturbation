# REVIEW 032 — ADDENDUM: the "in this cell" operationalisation (answering the PI's message after 95581b0)
verdict: the 2-sd rule should be replaced before C6 lands. This is a wording rule only; rule 8's amended conditions
(i) ≥ 0.2532 and (ii) > the training-row prior are unaffected.
reviewed_commit: 95581b0

## Why "increment over the training prior > 2 × cell-shuffle sd" doesn't test anything coherent

- **The two sides are different quantities.** The increment is measured against a **constant** reference, the
  training-row prior (0.2292). The cell-shuffle sd is the spread of a null centred at **0.10–0.12**. That sd is mostly
  between-derangement variation: which of the 265 derangements of 6 cells pairs which cell with which. It isn't the
  sampling uncertainty of the increment. Comparing one to the other has no interpretation, and it's fragile at baseline:
  seed 1 passes by 0.0009 (0.0335 vs 0.0326).
- **The cell-shuffle mean sits far below the prior (0.10 vs 0.23), and that's informative.** A single row's readout
  drawn from another cell aligns much *worse* than a denoised generic ranking. So the cell-shuffle null mostly measures
  row-level readout noise. Beating it (0.27 vs 0.10 ± 0.02) shows the readout is row-specific. It doesn't show that the
  readout adds cell-specific ranking information beyond a generic one. That was my (b), and the numbers show it's the
  wrong comparator for the claim. I withdraw it as the licensing test.

## A coherent operationalisation

**"In this cell"** means: this cell's readout ranks this cell's pathways better than the **same model's readout for other
cells** does. Take the reference to be the model's own aux readout averaged over the **other** dev cells' rows. That's
model-derived and denoised, like the prior, but carries no information from this cell. Then, per seed, compute the
per-row paired difference `ρ_i(own readout) − ρ_i(other-cells mean readout)` and average it within each dev cell.
Licensed iff:
- the 3-seed mean per-cell increment is **> 0 in ≥ 5 of 6 dev cells**, and
- the mean of the six per-cell increments is **> 0 on every seed**.

The cell is the unit the claim is about, so consistency across held-out cells is the right replication unit. Say
plainly that ≥ 5 of 6 is a consistency filter (one-sided sign p = 0.11), not a significance test. Report the increment
over the training-row prior beside it as the magnitude.

## Before C6 lands

**Check this rule on P2 first,** so it is fixed outcome-free for every variant. I tried to compute it locally, but the
three CPU forward passes exceeded what this laptop should carry, and I stopped. Only the training-row prior reproduced
(0.2292). If P2 fails the rule, the baseline's sentence drops "in this cell". That's a legitimate outcome, and not a
reason to loosen the rule.
