# REVIEW OF PACKET 107 (stage 1 as amended by review 106; §103.1, still before any code or genetic value)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 3108891

**The blocking leak is fixed, and the amendments are faithful.**
- **Training targets:** with per-fold references from R(g) \ {c₀, c′}, no training target contains y(g, c₀).
  - For any gene, the training targets now sum to zero among themselves, independent of y(g, c₀).
  - The held-out target is y(g, c₀) minus the mean of g's training-cell signatures. That is a legitimate question, not a
    leak.
- **The projection point:** the PI's argument against my suggestion is right, and I withdraw it. A per-row projection
  removes only the component along ȳ. The reference noise's component orthogonal to ȳ stays in both residuals with
  coefficient β, so the inflation persists. Disjoint half-references are the correct fix.

**But one of review 106's own suggestions introduces a cell-level leak, which I must correct** (C1). Three smaller points
remain (C2–C4).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | leakage through per-plate normalisation (Asks 1, 4) — **a correction to review 106 C6** | **The held-out cell's `ctl_vector` signatures aren't outcome-free in LOCO.**<br>• Level-4 robust z is computed per gene against the **plate population**. On knockdown plates that population is mostly the cell's knockdown wells, and level 5 inherits this.<br>• So a vector control's z-profile in c₀ is roughly **minus the median knockdown response of c₀'s plates**, an aggregate of held-out outcomes.<br>• §103.1's LOCO cell-offset baseline (*"the held-out cell's mean `ctl_vector` signature minus the mean of its reference cells'"*) is therefore built from c₀'s knockdowns.<br>• **The same holds for any stage-2 input taken from c₀'s L1000 signatures,** controls included.<br>• Review 106 called these "control measurements, not outcomes". That was wrong. | **In LOCO, drop the control-based cell offset** or label it **transductive (oracle)** and keep it out of every gain rule. The cold-gene version (mean d over the cell's training genes) is legitimate and stays.<br>**Register an input rule for stage 2:** in LOCO, no input derived from c₀'s L1000 signatures, treated or control. Inputs from c₀ come only from non-L1000 sources (CCLE, chromatin, lineage).<br>D1's s_ctl is unaffected, since D1 is descriptive. |
| 2 | MINOR | what the ceiling bounds (Ask 2) | **The half-reference agreement is a lower bound on the reliability ρ of d, not on the best achievable model r.**<br>• For two independent noisy measurements of the same T, r(d₉₆, d_late) ≈ ρ.<br>• A perfect model (predicting T) correlates with one measurement at ≈ **√ρ**.<br>• So comparing model r directly with the pair r understates the ceiling. A model can legitimately exceed it.<br>• **Two more assumptions:** the bound assumes the late signature measures the same T (time differs, which only lowers ρ), and it holds for PC3 / MCF7 only. | **State the conversion:** report the pair agreement ρ̂ and the implied ceiling for a model's r, ≈ √ρ̂ (row-wise). Both are lower bounds.<br>Say that it applies to PC3 / MCF7 and is assumed elsewhere. |
| 3 | MINOR | D1's ratio compares unlike constructions (Ask 3) | **s_kd is computed on consensus gene signatures (CGS), and s_ctl on level-5 vector-control signatures.**<br>• A CGS averages several hairpins, each itself a replicate MODZ, so it has one more layer of averaging than a level-5 control signature.<br>• How much averaging shrinks a landmark's SD depends on its replicate reproducibility, which depends on expression in that cell.<br>• So log(s_kd / s_ctl) keeps a cell × landmark **reproducibility** channel tied to expression and chromatin.<br>• The partial on CCLE expression and the double centring reduce it but needn't remove it. | **Best:** compute s_kd from **hairpin-level** `trt_sh` level-5 signatures (the same construction as `ctl_vector`). They are in the GCTX, which can be re-extracted on Kaggle.<br>**Otherwise:** state the construction mismatch as a limitation of D1, and keep the s_ctl technical channel prominent in the reading. |
| 4 | MINOR | the reference panel and the leak test (Ask 1) | **(a) R(g)'s composition changes d's meaning gene by gene.** With references from 15 cells, a gene's "cell-specific part" is its deviation from whichever cells profiled it. For some genes that is mostly core cancer lines; for others it includes NPC, ASC, HEKTE and so on. That is heterogeneous but not leaky.<br>**(b) The synthetic test needs its own positive control and a rule for constant predictions.**<br>• A per-gene intercept trained on fold-correct targets (which sum to ≈ 0) predicts ≈ a constant, so per-row r is undefined.<br>• The test should assert that **the leaky §103 definition scores ≈ −1** in the same harness, so the test is shown to detect the leak. | **(a)** Report R(g)'s size and composition per row. Add one sensitivity analysis with a fixed panel, e.g. the 8 cells with ≥ 2,000 knockdowns at 96 h (the six core, HA1E, HCC515).<br>**(b)** Make the test assert both: leaky ≈ −1, fixed \|mean r\| < 0.02. Use a gene-embedding → MLP model, or define r = 0 for zero-variance predictions. |

## Answers to the asks

**Ask 1 — yes for targets; no for the LOCO control-based input** (C1).
- **Training targets:** per-fold references contain no y(g, c₀). The training targets' zero sum is now internal to the
  training cells.
- **Inputs:** the leak route that remains is per-plate normalisation, through c₀'s control signatures (C1).
- **Scoring cold-gene and "both" on y is right.** A held-out gene has no training rows, so d's reference there can only use
  held-out data. Keeping d target-side only is correct.

**Ask 2 — yes, disjoint half-references give a lower bound on the reliability of d.**
- **Why it is a lower bound:** the half-references add independent noise, the halves' different cell compositions add
  disagreement, and time adds disagreement. None can inflate it.
- **The β·e_ref argument against projection is right** (above).
- **Report √ρ as the model ceiling** (C2).

**Ask 3:**
- **The normalisation channel:** largely, yes. Both s are under the same per-plate z-scoring, so the ratio cancels the
  shared scale. But the CGS-versus-control construction leaves a reproducibility channel (C3).
- **The noise bound is right.** Var(y₉₆ − y_late) / 2 = σ²_noise + Var(time) / 2 ≥ σ²_noise, an upper bound on per-entry noise
  variance, compared with the interaction mean square (≈ σ²_gxc + σ²_noise). An interaction above the bound shows real
  gene × cell signal, a conservative test.
- **Two things to state:**
  - the bound comes from PC3 / MCF7 and is applied to all six cells;
  - the genes re-profiled at 120 / 144 h may differ from the rest (they are a selected subset).

**Ask 4 — yes, with C1's input rule registered.** The rest is in place:
- the leak-free targets;
- the y-scored cold splits;
- the lower-bound ceiling (with √ρ);
- the variance decomposition;
- the cell-offset (cold-gene), k-nearest-cell and ridge baselines.

C2–C4 can be settled in the code packet.

**B1:** C4 and C5 are adopted correctly.
- **One detail for the code:** p_mis needs t knocked down in at least one other training core cell. Count the pairs that
  lack one, and drop them from the matched / mismatched comparison only.

## What I checked and found sound

- **§103.1 in full, against review 106's six challenges.**
- **The fold algebra:**
  - training targets sum to zero over R(g) \ {c₀} with no y(g, c₀) term;
  - the held-out target's dependence is only on y(g, c₀) and the training-cell signatures.
- **The half-reference ceiling's independence argument,** and the projection's residual β·e_ref term.
- **The variance decomposition's noise-bound inequality.**
- **B1's amended definitions:** the percentile direction, the own-coordinate exclusion, dose / time, the row filter, the
  matched / mismatched condition.

## What I could not assess, and why

- **Anything needing signature values.** This includes the size of the per-plate leak in C1, which is aggregate and
  cell-level; not reading `Y` keeps stage 1 blind.
- **Whether hairpin-level `trt_sh` signatures are present for all core CGS.** That needs the GCTX's column metadata, which
  isn't in the extract.
