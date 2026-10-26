# REVIEW OF PACKET 100 (T-C read once, §100.14: no claim)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 628e85e

**The registered reading is correct, and it reproduces exactly from the prediction files with my own code.**
- **My implementation:** truth = X − its control, from the substrate. Contrast = the row minus the mean of the
  other-compound scored rows on its plate, with ≥ 2 mates. Then the per-row Pearson r.
- **Rows:** 21,151 scored rows (21,321 test rows minus 170 featureless) and 16,167 contrast rows.
- **Agreement:** my per-row r equals the kernels' `r_contrast` to 1e-8 in all 6 runs.
- **Identical results:** the per-cell table, Δ_c, the mean −0.00178, the CI [−0.00605, +0.00141] (`default_rng(0)`, 20,000
  draws), 3 / 7 cells favouring on, and H1975 (0.2630 against 0.2595).
- **Ledger:** all 12 sha1s (6 records, 6 predictions) match `tc_ledger.jsonl`.
- **Run selection:** every slot uses its first completed attempt.

**"No claim" is right.** The caveats concern the strata reading (C1), the collapse count (C2), and three wording or reporting
points (C3–C5).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | statistics: the DNA-damage "family-wise chance of about 0.6 %" (Ask 3) | **The 0.6 % treats the seed z as normal. It is a Welch t on ≈ 4 df.**<br>• `strata.group_stats` computes z = Δ / √(s_a²/3 + s_b²/3) from **3 + 3 slot means**. For DNA damage, Welch df = 3.99.<br>• **t = 3.43 on 4 df gives p = 0.027** unadjusted. Over 10 classes the family-wise chance is **≈ 0.24**, not 0.006 (10 × the normal p = 0.0060 is where 0.6 % comes from).<br>• **Exact relabelling** of the 6 slots: the real split is the most extreme of the 10 balanced splits (every on slot 0.2019–0.2101 > every off slot 0.1890–0.1972). So **p = 0.10 two-sided**, the floor any 3 + 3 design can reach.<br>• **The compound-cluster CI conditions on these six runs.** It covers compound heterogeneity, not run-to-run noise, so its exclusion of 0 doesn't answer the seed question.<br>**The signal is also one cell's**, recomputed from my own per-row r (Δ, rows, compounds):<br>• **MCF7 +0.0258** (218 rows, 42 compounds);<br>• HT29 −0.0012 (157 rows);<br>• THP1 −0.0106 (32);<br>• MDAMB231 −0.0066 (8);<br>• BJAB +0.004 (10);<br>• HS578T +0.006 (7).<br>The top 5 compounds carry 59 % of the class sum, and 29 of 42 compounds are positive. | **Replace the 0.6 % sentence:** *"seed t 3.4 on ≈ 4 df (p ≈ 0.03 unadjusted, ≈ 0.24 over 10 classes; the 3 + 3 relabelling floor is 0.10, and the real split is the most extreme); the compound-cluster CI excludes 0 but conditions on the runs; the class signal sits in MCF7 (+0.026, 218 rows) and is absent in HT29 (−0.001, 157 rows)."*<br>Keep "hypothesis, not a finding", but call it **weak, and MCF7-specific**. |
| 2 | MINOR | collapse asymmetry: the count and an echo in trained runs (Ask 2) | **The count leaves out T-B's width-screen on run** (`dev_on_d384_s0`, untrained by 100.9).<br>**The full tally of runs and attempts:**<br>• **on: 5 of 10 collapsed** (T-B d256 1/3 unguarded, d384 1/1, T-C 3/6);<br>• **off: 0 of 6.**<br>• **Fisher one-sided p:** ≈ 0.06 for all widths; ≈ 0.09 for d256 only (4/9 against 0/6).<br>**The asymmetry also shows in runs that trained.** At epoch 0, the mean batch-median s is **0.029, 0.039, 0.039 in the three on runs** and **0.045–0.051 in the three off runs**: all on below all off. By epoch 3 they overlap (0.058–0.136 against 0.067–0.090).<br>**So the chromatin input systematically depresses the early dose scale.** It isn't only an all-or-nothing collapse. | Report the full tally and the epoch-0 contrast with §100.14's attempt counts.<br>**Add one sentence:** the conditional no-claim can't separate *"cell-specific chromatin carries no usable signal"* from *"v10-C doesn't use it in runs where its drug path trains"*. The across-arm prediction agreement (C4) is the evidence that the trained on runs behave like off runs. |
| 3 | MINOR | wording: the heading and the sign of the mean (Ask 1) | **The heading says *"Cell-specific chromatin does not contribute on unseen cells"*.** A failed rule licenses no claim, not a demonstrated absence. "What this closes" already says *"adds nothing detectable"*, which is right.<br>**The negative mean is CD34's.**<br>• Without CD34, the mean Δ_c over 6 cells is **+0.0001**.<br>• In CD34 every off slot (0.0985–0.1041) beats every on slot (0.0825–0.0911), seed z −4.1. That is the same extremeness as DNA damage, in the other direction.<br>• The strata reading highlights the positive extreme only. | **Heading:** *"no claim: cell-specific chromatin adds nothing detectable on unseen cells (mean Δ_c −0.0018; the cell-cluster CI bounds the mean gain at ≤ +0.0014)"*.<br>**Add:** the mean's sign comes from CD34 (all other cells average +0.0001). Treat CD34's −0.013 and DNA damage's +0.012 alike, as unconfirmed extremes. |
| 4 | MINOR | wording: "the trained model barely uses which cell's chromatin it is given" (Ask 1) | **The permutations measure how sensitive the score is, not the predictions.**<br>**Per seed, against each run's own row-pooled r:**<br>• **cell-swap:** −0.00016, −0.00020, −0.00019;<br>• **gene-shuffle:** −0.00063, −0.00073, −0.00062;<br>• both are below the seed SD of row-pooled r (0.0013–0.0017).<br>The gene-shuffle cost has the same sign in every seed: a small **gene-level** use of chromatin that the off arm matches without it.<br>**Direct evidence about the predictions** (my check):<br>• the on runs' predicted contrasts agree with the off runs' at a **mean per-row r of 0.8053**;<br>• within on 0.8056, within off 0.8035.<br>So turning chromatin on changes the predictions **no more than changing the seed does.** | *"Which cell's chromatin the model is given barely matters to the score (cell-swap −0.0002 in each seed; gene-shuffle −0.0007), and the on runs' predicted contrasts agree with the off runs' as closely as runs within an arm agree (per-row r 0.805 against 0.806 / 0.804)."* |
| 5 | MINOR | a registered reported item is missing | **§100.5 registers *"the gate γ against accessibility, by mark"*** among the items reported beside the reading.<br>It isn't in `tc_reading.json` or §100.14. The records hold only `gamma_mean_last_batch`. | **Report it from the 3 on checkpoints**, or state why it's omitted. |

## Answers to the asks

**Ask 1 — the verdict and the clause are right; adjust two sentences.**
- **"No claim" is correct:** neither the rule nor its mirror holds. I reproduced it exactly.
- **The clause** *"among runs whose drug path trained (100.8)"* **is necessary and correctly placed.**
- **The heading** should say "nothing detectable", with the CI bound (C3).
- **"Barely uses"** should be the score and prediction statement of C4.

**Ask 2 — reporting it is enough for T-C's reading; it doesn't change the verdict.**
- **Report it fully** (C2): 5 of 10 on runs or attempts collapsed against 0 of 6 off, plus the epoch-0 dose-scale contrast in the
  trained runs.
- **The used on runs look normal at the end.** Checkpoint median s is 0.058–0.144 against off 0.069–0.095, and the predictions
  agree with off as closely as seeds do (C4). So the conditioning hasn't produced visibly atypical on runs.
- **What it bears on is scope, not the reading.** The statement is conditional on a trained drug path, and in v10-C the
  chromatin input makes that condition harder to meet. Say so once.
- **For v11:** its arms run with the same guard. Its collapse counts per arm (already registered) will show whether chromatin
  through FG behaves the same way.

**Ask 3 — "hypothesis, not a finding" is right, but it is a weak hypothesis (C1).**
- **The evidence:** p ≈ 0.03 unadjusted at the seed level, ≈ 0.24 family-wise, relabelling floor 0.10, concentrated in MCF7.
- **Whether to spend a test on it is your call.** If you register one, the following bear on rows and rule.

**Rows.** The untouched rows for v10-C are the clean cells' `excluded_test_cell` rows. The 6 T-C checkpoints never trained on
them and T-C didn't score them. My counts of DNA-damage rows and compounds, before the featureless drop and the mate filter:
- MCF7: 98 rows / 29 compounds;
- HT29: 123 / 33;
- MDAMB231: 147 / 26;
- THP1: 109 / 21;
- BJAB, CD34, HS578T: 0.

**The compounds are new per cell.** In each cell, every DNA-damage compound in those rows is new relative to that cell's test
rows. Across cells, 21 of the 50 appear in some other cell's test rows. So this is new (cell, compound) pairs, not replicate
wells.

**Choose the hypothesis before the read: MCF7 alone or pooled.**
- T-C's pooled signal is MCF7's alone, so a pooled test over cells where T-C showed nothing has a low prior.
- MCF7 alone gives 98 rows / 29 compounds.

**The rule:**
- **The ceiling:** with the same six checkpoints, run noise is fixed at 3 + 3, and no seed-level test can do better than
  p = 0.05 one-sided.
- **A rule that fits that ceiling:**
  - (i) all three on slot means exceed all three off slot means on the class rows (the one-sided relabelling floor); **and**
  - (ii) the compound-cluster bootstrap's one-sided 95 % bound > 0.
- Report per cell.
- **More seeds would be needed** for anything stronger.

**Fix the mates rule before the read.** Contrasts can be computed within the excluded rows only, or with test-row plate-mates
included: 1,017 of 2,947 excluded plates are shared with test rows.

**Timing.** B reads these same rows for H-E2r / H-E4 / H-E5.
- These are different statistics on observed versus predicted values, so there is no conflict if both are registered before
  either runs.
- Otherwise, run it after B.

**v11's test-set evaluation is not a replication.**
- It is a different model, and its chromatin enters through FG.
- A DNA-damage stratum there would test a related but different hypothesis.

**Ask 4 — acceptable for T-C; no need to redraw.**
- **The rows are T-C training rows:** the dev carve's training rows are a subset of `split_cold_cell_1` train.
- **One subtlety:** `V10CData(dev=True)` normalises dose and time over the dev-carve training rows, not T-C's 208 cells. This is
  immaterial:
  - **the checkpoint medians agree with each run's own end-of-training records** to within 7 % (on 0.144 / 0.066 / 0.058
    against 0.136 / 0.063 / 0.057; off 0.095 / 0.069 / 0.082 against 0.090 / 0.067 / 0.080);
  - **moving a median of ≈ 0.06 to the 1e-4 floor** needs a logit shift of ≈ 6.4, far beyond any normalisation difference from
    dropping 6 of 208 cells.
- **Optionally,** note it in the check's header.

## What I checked and found sound

- **The ledger.** sha1 of all 6 used records and 6 prediction files against `tc_ledger.jsonl` (all equal). The attempt
  structure (on: s0 attempts 0–2, s1 attempts 0–1, s2 attempt 0; off: attempt 0 each). Guard stops at step 1000 with s ≈ 1e-9
  to 1e-12.
- **`tc_read.py` in full against §100.5:**
  - the 7 clean cells, the 3-seed mean, Δ_c;
  - ≥ 5 of 7 and the CI (20,000 draws, `default_rng(0)`);
  - the mirror rule;
  - first-completed-attempt selection with the record sha1 check;
  - the reported items.
- **An independent re-scoring from predictions and the substrate.** Rows, contrast rows, per-row r (to 1e-8), the per-cell
  table, Δ_c, the mean, the CI, the counts and H1975, all identical.
- **Per-seed permutation shifts** against each run's own row-pooled r, and the seed SDs.
- **Across-arm prediction agreement:** the mean per-row r of predicted contrasts between runs (C4).
- **`strata.py` and `tc_strata.py`:**
  - the seed-z formula (C1);
  - the DNA-damage class reproduced from my own per-row r (Δ 0.01183; slot means identical);
  - its Welch df, the t and normal p-values, and the exact relabelling;
  - its per-cell and per-compound breakdown;
  - the CD34 and THP1 slot means.
- **`tb_trained_check.py` and `tc_trained_check.json`** against the runs' own dose-scale histories. The T-B trained checks
  (`dev_on_d256_s0` and `dev_on_d384_s0` untrained).
- **The untouched DNA-damage rows** in the clean cells' `excluded_test_cell` split, with compound novelty per cell and plate
  overlap with test rows.

## What I could not assess, and why

- **The per-compound list.** I read the BH result (0 of 731) and its 9 relabellings (all 0) from `tc_strata.json`. I didn't
  recompute them.
- **The gate γ against accessibility.** It isn't reported (C5).
- **Why the on arm's dose scale collapses or starts lower.** The records don't log chromatin-route magnitudes at the guard's
  first check, and the collapsed attempts have no checkpoints.
- **The gap and the head-to-head.** They are pending `gap_on_s0–2`.
