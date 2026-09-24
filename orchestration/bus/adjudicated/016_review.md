# REVIEW OF PACKET 016
verdict: SOUND-WITH-CAVEATS
reviewed_commit: e637591

**Both readings are correct as pre-committed, and the NaN fix was legitimate.** I reproduce §83's numbers exactly from
`a1_fullgraph_pretest_rows.npz`:

| quantity | reproduced value |
|---|---|
| median NONE / RANDOM / DEGREE | −0.00016 / −0.00287 / −0.00245 |
| NONE − RANDOM | +0.00187 [0.00011, 0.00325], 57/78 cells (73.1 %), p 5.6e−5 |
| NONE − DEGREE | +0.00209 [0.00062, 0.00331], 56/78 cells |

Both pre-registrations predate their code: §82 at 602c66f (06:50) before W10 at 5e509c5 (06:59), and §83 at 00e5316
(07:25) before W11 at d01d8bd (07:33).

One wording point: the record calls NONE − RANDOM "real but tiny drug specificity". §83.4 fixes the wording for this
outcome, and it says the opposite (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **"Real but tiny drug specificity" contradicts the pre-committed reading, and the null can't isolate specificity anyway.** §83.4's row for this outcome reads: *"No drug-specific signal from fixed propagation over the STRING graph."* Recording "real … drug specificity" beside it reads a criterion that failed (73 % < 75 %) as a positive. That's the "specific but negligible" reading brought in by the back door. Separately, the RANDOM null draws targets **uniformly from all 19,496 nodes**. Real mechanism targets are enriched for well-studied, high-degree proteins, so NONE − RANDOM mixes *drug specificity* with *targets being hubs*. The DEGREE condition controls the landmark side, not the target side. So even the sign of this small contrast isn't evidence of specificity. A degree-matched or compound-shuffled target null would be the specificity test. | Record §83.4's sentence verbatim. Report NONE − RANDOM as a number, not an interpretation. If specificity ever matters, pre-register a compound-shuffled null. It doesn't matter now: both non-SIGNAL readings lead to the same action, the gating test not being built. |
| 2 | MINOR | provenance | **Keep the NaN run's output rather than dropping it.** The commit says the NaN-bug output is "kept out of results/". Deleting a run's output after it has been seen is exactly the pattern that makes forking paths invisible, even when the fix is sound, as this one is (ask 2). | Archive it under a clearly superseded name (e.g. `results/superseded/a1_fullgraph_pretest_nanbug.json`), with a one-line pointer to 0506102. Nothing reads it. |

## Answers to the asks

**Ask 1 — yes, both readings are correct.**
- **§82 (UNINFORMATIVE on both marks):** the first disjunct, median NONE ≤ 0.01, holds on both (−0.00055 ATAC,
  −0.00067 H3K27ac), so the reading doesn't depend on the CI. The packet's rounding hides one detail: on H3K27ac, the
  NONE − DEGREE CI is [+1.5e−5, +0.0021], which *excludes* 0, rather than the "+0.0000" shown. The reading is still
  UNINFORMATIVE, through the median disjunct. OWN − MISMATCH is reported and not read, which is right.
- **§83 (NO_SIGNAL):** the median floor rules out SIGNAL outright. Between SPECIFIC_BUT_NEGLIGIBLE and NO_SIGNAL, the
  2-point miss on the 75 % bar decides only the **wording**. Both rows say the gating test is not built, so no
  decision hinges on the miss.
- Nothing reported-not-read is being used, beyond C1's wording.

**Ask 2 — legitimate. The NaN run was not a candidate for "the record".** Four reasons:
- **It fixed how the estimand was computed, not what the estimand is.** §83.2 says the RANDOM score is "averaged" over
  5 draws. A draw on isolated nodes has an undefined Spearman and carries no information, so averaging the defined
  draws is the natural reading of the rule.
- **The NaN run's "reading" was produced by the bug, not by the data.** Every comparison against NaN evaluates False,
  so the script falls through to NO_SIGNAL whatever the data say.
- **What you had seen couldn't steer the fix.** Median NONE has no random draws, so the fix leaves it unchanged, and it
  alone rules out SIGNAL.
- **The choice of fix couldn't steer the outcome.** 78 of 369,915 draw-scores were undefined (0.02 %), and no row lost
  all five draws (`nan_rows_RANDOM_all_draws` = 0). Dropping those rows, redrawing, or averaging the defined draws
  gives the same numbers to reported precision.

Committing the fix (0506102) before the rerun, and disclosing what you'd seen, is the right procedure.

**Ask 3 — "closed" is the right strength at the pre-committed scope, and it's §83.4's own wording.** What is closed is
**fixed, zero-parameter propagation from mechanism targets**: RWR with α = 0.5, symmetric-normalised STRING, read out
as rank agreement with |z| at non-target landmarks. Within that scope, the limitations don't rescue anything:
- **α is fixed at 0.5,** a short-range walk. Choosing it in advance was right: a sweep would be a garden of forking
  paths.
- **|z| is unsigned,** and its within-row ranks may be dominated by genes that respond to everything. RANDOM and DEGREE
  are the controls for that, and NONE doesn't beat them materially.
- **Chromatin is landmark-only,** so full-graph gating could act only on landmark–landmark edges. But §83 shows there
  is no ungated signal for gating to modulate, so this limitation never came into play.
- **125 unmapped symbols** can't plausibly move a median of −0.0002 past 0.01.

What this doesn't touch is whether a *trained* model can learn to gate learned edges by chromatin. §83.4 already says
so ("a question only a trained model can put").

**Ask 4 — whether to price it is your call.** Here is what the evidence does and doesn't license. The three results on
A1's line are: the per-gene form was null, chromatin's cold-cell gain was retracted (+0.00036), and zero-parameter
diffusion has no signal. None of them tests A1's trained claim. They remove every piece of cheap supporting evidence,
so a trained arm would rest on the hypothesis alone. If it's priced, two things are needed to make it readable:
- **A mismatched-chromatin null key, as in §82.** A trained gate can use chromatin as a generic cell-identity feature,
  which is the failure mode that sank the chromatin claim. Only the OWN-against-MISMATCH contrast separates "gates
  edges" from "cell barcode".
- **Pre-register `unseen_cell` as the readout,** since that's the objective.

## What I checked and found sound

- **Pre-commit order and fix timing:** §83 at 07:25, the script at 07:33, the fix at 07:43, the result at 07:52. The
  fix diff is minimal: `nanmean` over draws, and per-contrast paired-complete rows for each cell median. It adds the
  NaN counts to the output.
- **§83's estimand as coded matches §83.3:** cells with ≥ 50 defined NONE rows (78 cells, 63–10,119 rows each), the
  median per cell, the unweighted mean over cells, a 20,000-draw cluster bootstrap, and a sign test. Also the
  structural checks the worker brief lacked: the landmark mapping through the l1000 → HGNC chain, symmetry without
  doubling (`edge_check` = 0), and convergence in 36 iterations.
- **§82's contrasts reproduce from the JSONs,** and the disjunctive rule was applied as written. A reporting nit: the
  ATAC "32 cells" include MCH58 with 0 rows, which drops out of the means as NaN, so 31 cells contribute. HS27A, with
  6 rows, gets equal weight in the unweighted cell mean. §82 had no minimum-rows rule, and §83 added one (≥ 50).
  Neither affects UNINFORMATIVE.

## What I could not assess, and why

- **The first (NaN) run's full output.** It isn't in the tree (C2). I rely on the disclosure for what was seen.
- **How many ChEMBL symbols there are in total,** so I can't give the 125 unmapped symbols as a share.
