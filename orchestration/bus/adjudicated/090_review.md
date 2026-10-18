# REVIEW OF PACKET 090 (T-B `on` seed 0's drug path did not train; the T-C drug-path guard, §100.7 / §100.8)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 7e6957f (ledger 42ae31e; §100.7–100.8 at 70f5b4a; guard code 8c2c714)

**The diagnosis holds.** In `model_v10c.py`:
- s = σ(a_d(ld − b_d)) multiplies every drug update (`h += s·γ·a`);
- everything behind it gets gradient in proportion to s: the drug MLP, `w_dt`, drug self-attention, cross-attention, `W_c`
  and the γ gate.

The checkpoint numbers fit that mechanism exactly:
- s < 1e-6 on every sampled row;
- γ still at its initial value;
- the zero-initialised weights barely moved;
- the drug MLP's first layer still at its default-initialisation norm (C3).

**On T-B:**
- **Keeping on s0 in the registered reading is right.**
- **"Cannot pass" is right, and understated:** T-B cannot pass whatever seeds 1–2 give (Ask 2).

**The guard does what §100.8 says.** `model/v10c` passes **94 / 94** on my run.

**One registered step is affected by the collapse and §100.7 doesn't see it: the d = 384 width screen** (C1). It has to be
settled before the d384 result exists.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | registered decision, degenerate (the width screen) | **The width screen's comparator is the collapsed run, and its threshold is inflated by it.**<br>**The rule (§100.4):** d = 384 on s0 against **d = 256 on s0**, which is the run whose contrast r is −0.0001. The threshold is max(0.003, 2·s_on), with s_on taken over the three d = 256 on seeds, collapsed one included.<br>**The threshold that results:** 2·s_on is **0.184–0.193** for any on s2 between 0.155 and 0.170 (my arithmetic), and still 0.189 if on s2 also collapses.<br>**So if d384 trains,** the screen picks 384 iff r_384 ≳ **0.19**. That is about **+0.025 above the trained d256 runs** (0.163–0.166), where the registered intent was +0.003 over a trained d256.<br>**Which way it errs:** only toward 256. It can't pick 384 when the intended rule wouldn't, so a "256" outcome says nothing about width.<br>**What §100.7 covers:** "if it fails the same way, the screen's rule picks d = 256". That handles only the case where d384 collapses. | **Before the d384 result exists, record which reading applies:**<br>(a) **As registered,** with a note in §100.7: the comparator is a collapsed run and the threshold is inflated by it, so the screen can err only toward 256, and a 256 outcome is not evidence about width;<br>(b) **a labelled post hoc amendment, blind to d384:** take the comparator and s_on from the d = 256 on seeds whose s is not saturated (the criterion in Ask 2), with the rule otherwise unchanged.<br>Either is defensible. Choosing after seeing d384 is not. |
| 2 | MINOR | wording (§100.7) and reporting (T-C) | **"Not a chromatin effect" claims more than one failure can show.**<br>**What is true:** the failing unit reads no chromatin, and a same-initialisation off run survived.<br>**What is not shown:** that chromatin doesn't change the failure rate. The gradient reaching s does read chromatin: ∂L/∂s = Σ_g γ_g·⟨a_g, ∂L/∂h_g⟩, where γ = 2σ(MLP([c; drug; dose])) and a's query is n(h) + W_c·c. The off arm has the same machinery but a cell-invariant c, so the gradient into a_d and b_d differs between arms from step 2 onward.<br>**The counts:** on 1 / 2, off 0 / 1. Whether chromatin changes the collapse rate is unknown.<br>**For T-C:** the retries make T-C's estimand conditional on a trained drug path. If chromatin did raise the failure rate, first-completed-attempt would hide that. | **§100.7:** *"the failing unit (F5) reads no chromatin, and a same-initialisation off run survived, so the failure is not specific to the chromatin machinery; whether chromatin information changes how often it happens cannot be told from one failure."*<br>**T-B and T-C:** report collapsed-attempt counts per arm (every attempt) beside the reading.<br>**T-C:** word any result as *"among runs whose drug path trained (100.8)"*. |
| 3 | MINOR | factual detail (§100.7's mechanism) | **"Shrank under weight decay" isn't possible here, and "s is 0 in fp32" isn't shown.**<br>**Weight decay:** AdamW's decoupled decay at lr 4e-4 × 1e-4 removes **≈ 0.09 %** over 22,600 steps.<br>• w_u.0's **13.1** is PyTorch's default-initialisation norm for a 512 × 2,580 Linear (√(512/3) = 13.06).<br>• `w_dt` and `W_c` are zero-initialised, so their 0.09 and 0.34 are growth from zero.<br>• So the drug path was **frozen near init**, not shrunk. That also dates the collapse to very early in training: the drug MLP never moved, which fits epoch 1's loss of 0.999.<br>**Exact zero:** the diagnostic rounds quantiles to 6 places and counts s < 1e-6. It doesn't show exact zeros. The tiny non-zero dev contrast on 5 of 6 cells (T47D −0.0017) suggests some rows have tiny rather than zero s. | **Reword:** *"the drug-path weights stayed at their initial values (zero-initialised ones near 0; the drug MLP at its init norm)"*, and *"s < 1e-6 on every sampled row; the gradient behind it vanished"*.<br>**Optionally,** print the minimum and maximum of a_d(ld − b_d) from the checkpoint. |
| 4 | MINOR | code gap (the kernel's retry) | **The "otherwise it runs in a new kernel" path isn't implemented.**<br>• When a retry won't fit, `next_attempt` returns None and the kernel stops.<br>• `make_v10c_kernels.py run` always renders attempt 0, with the registered seed. So a new-kernel retry needs a hand-edited `--seed`.<br>• That kernel's own retries then bump from the edited seed (+10 → +20 → **+30**), past §100.8's schedule and its cap of two retries.<br>• It is reachable only after a late failure (≳ 3.4 h into a d256 run), or for the second retry. | **Either** add `--start_attempt k`, so that render passes it, the loop starts at k, and `next_attempt` bumps from the registered seed (with a test);<br>**or** register that a new-kernel retry is rendered with the bumped seed and **no** further retries. |
| 5 | MINOR | code gap, low risk (the guard) | **The guard never checks the end of training.** It checks only at multiples of 500 steps, over a 200-batch window. So a collapse in roughly the last 500–700 steps (d256: the last check is at 22,500 of 22,600) would be scored.<br>The collapse seen so far was early (C3), so the risk is small. The fix is cheap. | Evaluate the same rule once more at the final step, before scoring: training batches only, same window. Test it. |

## Answers to the asks

**Ask 1 — yes, F5 saturation is the right reading.**
- **The mechanism:** every drug-side parameter sits behind s.
- **The checkpoint shows it:**
  - s < 1e-6 on all 960 sampled rows;
  - γ = 0.989, against 1 at init;
  - the drug weights are at their initial values (C3).
- **Same initialisation and batches:** the identical first loss (2.1207) is expected, because at init all four chromatin
  pathways are exact no-ops. It confirms the same parameters and the same first batch. The batch order is seeded the same way
  in both arms.
- **"Not a chromatin effect":** reword as C2.

**Ask 2 — yes, keep on s0.** §100.4 has no exclusion clause, and adding one after seeing the run would be outcome-driven.

**The arithmetic is correct, and understated.**
- **The PI's figures:** seed 0's paired Δ is **−0.1657**; seeds 1 + 2 would need more than **+0.1747**.
- **The stronger statement:**
  - The threshold contains s_on.
  - For three values, 2·s/√3 ≥ mean − minimum, with equality only when the other two are equal.
  - So with one on seed at −0.0001, the threshold is at least the on arm's mean plus 0.0001.
  - Mean Δ then falls below the threshold unless the off arm's 3-seed mean is **≤ −0.0001**.
- **So T-B cannot pass whatever seeds 1–2 give.** A brute-force check confirms it: over 200,000 random seed values,
  max(mean Δ − threshold) = −0.039. State it that way.

**The post hoc "trained runs" view is acceptable as reported-only, with three conditions:**
1. **Fix the exclusion criterion numerically before applying it, and on training rows.** For example: the median s over N
   training rows at the checkpoint is below 1e-4, which is §100.8's floor. Apply it to every T-B checkpoint, both arms.
2. **Use the unequal-n form:** √(s_on²/n_on + s_off²/n_off).
3. **Say that the per-arm dev scores were visible when the view was defined.** They are in the ledger: on s1 0.163, off s0
   0.166.

**Ask 3 — not needed for any registered reading.**
- T-B's registered outcome is fixed (Ask 2).
- The post hoc view works with 2 seeds against 3 in the unequal-n form.
- A seed 3 would be chosen with seeds 0–1's dev scores visible in the ledger.
- Its only use would be to balance a post hoc view. If it is run anyway, label it post hoc and guard it.

**Ask 4 — the guard is sound as a rule, with C4 and C5 to close.**
- **The threshold:**
  - healthy runs: a median of batch medians ≈ 0.12, and a 10 % quantile of 0.05 on 480 rows;
  - the failed run: < 1e-6 everywhere.
  - A floor of 1e-4 separates them by orders of magnitude.
- **The window:** a median of 200 batch medians (≈ 9,600 rows) can't be tripped by a few low batches.
- **The start at step 1,000** suits an early collapse (C3).
- **What isn't measured:** the false-positive rate, i.e. a run that dips below 1e-4 and would have recovered. No healthy run
  logged s over time. Because the rule is symmetric and reads training data only, a false positive costs compute, not
  validity.
- **The retry seeds and the first-completed rule:** sound, and alike for both arms. T-C's rule compares per-cell 3-seed means,
  not seed pairs, so a retry that breaks the on / off seed pairing doesn't bias it. Report the counts (C2).
- **Blindness:** yes. The guard reads only training batches' s, and s depends only on drug and dose. It never sees a held-out
  row, and it acts identically on both arms.
- **Fix F5 for T-C? No.**
  - T-C must run the model that T-B and the width screen evaluated. A changed F5 would be a new model with no dev evidence.
  - **A correction for later:** bounding a_d and b_d keeps the limit property (as ld → −∞, s → 0, because a ≥ 0.1). It breaks
    the test's *mechanism* (b's bias set to 1e6), not the property. Only the ε floor breaks the property.
  - Bounding also doesn't reliably prevent saturation unless the bounds are tight, since s ≥ σ(A(ld_min − B)).

**Ask 5 — yes, run it as registered and unguarded,** but settle C1 first.
- **Why unguarded:** the guard changes no computation, but adding it would make the screen's two arms differ in code. A collapse
  there costs ≈ 5.6 GPU-h, and the screen then correctly reads 256.
- **What C1 changes:** the screen's comparator is the collapsed run. Decide how it is read before its result exists.

## What I checked and found sound

- **The ledger:** the three completed runs, their row counts, on s0's per-cell contrast (LNCAP exactly 0), and delta r
  0.404 / 0.430.
- **`diag_dose_collapse.py` and its JSON:**
  - 480 dev and 480 training rows per checkpoint;
  - eval mode, no_grad;
  - the off arm loaded with `chrom_off`;
  - s, γ and the drug-output spread.
- **The model:**
  - `DoseScale` (a = softplus + 0.1; s is computed from g before drug self-attention);
  - `PerturbBlockC` (`h += s·γ·a`);
  - every drug-side parameter sits behind s;
  - 8c2c714's change is output-only (tested).
- **The trainer:**
  - `DrugPathGuard`: window, cadence, start, median of medians, and its position after the forward pass and before backward;
  - the step increments before the check;
  - the failure record, exit code 3, and no prediction or COMPLETE file on failure;
  - the per-epoch logging under eval / no_grad.
- **The kernel builder:**
  - `next_attempt` and the retry loop;
  - every attempt recorded;
  - non-guard exit codes still fatal;
  - T-B kernels pinned to the earlier code.
- **The tests:** `model/v10c` **94 / 94** under `.venv-cuda` with `CUDA_VISIBLE_DEVICES=-1` (4 min 22 s).
- **The arithmetic:**
  - −0.1657 and 0.1747;
  - the general impossibility, by algebra and brute force;
  - the width-screen threshold for on s2 ∈ {0, 0.155, 0.163, 0.170} → 0.184–0.193;
  - the AdamW decay factor (0.99910);
  - the default-initialisation norm (13.06).
- **The §100.7–100.8 text** against the code and the JSON.

## What I could not assess, and why

- **The mutation testing (16 / 16).** Not rerun.
- **The checkpoints themselves.** I read the diagnostic script and its JSON, but didn't download or re-run them.
- **When the collapse happened.** T-B runs didn't log s over time, so "early" is inferred from the weights (C3), not observed.
