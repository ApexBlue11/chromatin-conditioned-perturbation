# REVIEW OF PACKET 036
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 6500770

**Most numbers check against their records:**
- §46.2's published range (0.195–0.383);
- §27's 99.65 %;
- §42's 0.694 against 0.738–0.744;
- §46.4's ridge 0.296 against TranSiGen 0.293;
- §87's 0.386, 21,151 rows, 5/8, +0.0465 [0.0048, 0.0861] and 0.473/0.387;
- every §85.8 screen Δ;
- C6's +0.0048 / 0.262 / Δ_c +0.0002;
- §85.9's +0.029 / +0.027 / 0.81;
- §85.10's 0.273 / 0.229 / +0.044 and 6/6;
- C 4.1a's 0.560, C 4.15's 0.218 / 0.229, and §55's +0.0004 / +0.0042 / +0.00036.

§5.4's pathway bullet is exactly C 4.16's licensed form, and the withdrawal of §37 is right.

**One abstract clause states the opposite of the evidence (C1),** and two framing sentences overreach (C2, C3). The rest
are corrections of detail.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | overreach | **Abstract (iii) lists "atom-level drug tokens" among components that "add nothing or harm". The dev screen says the opposite.** Training without atom tokens (C1) cost **−0.0086, 0 of 6 dev cells** (§85.8), the draft's own §5.3 says "a model trained without them is 0.0086 worse", and §85.2 rule 9 records atoms *helping* at inference when drug self-attention is on (+0.0029). The only "harm" evidence is §37's inference ablation of one fold-0 checkpoint with self-attention off. | Remove atom tokens from the list, or say it accurately. For example: *"… add nothing or harm; atom tokens are the exception: a model trained without them is worse (−0.0086), although removing them from one trained model at inference helped."* |
| 2 | MAJOR | overreach | **The paper claims pre-registration and review for every reading, and several reported readings predate both.** The Intro says *"Every reading in this paper was pre-registered in a public log before its data existed, and each was reviewed by a blinded adversarial reviewer"*; §4 says *"Every reading was committed before its data"* and *"the test cells were touched only by pre-registered comparisons"*. But: §37 (dissection, 08-29), §21 / §38.4 (EMA, 08-15 / 08-29), §42–§43 (forensics and the warm split, 08-30) and §46 all predate the log's protocol and the reviewer, whose first packet was 09-20 (§51). More importantly, `split_cold_cell_1`'s **test cells were read in §44, §45 and §46.4**. That's why rule 10 labels P7 *"a baseline partly chosen with test-cell knowledge"*. There are also smaller exceptions: §86's reader was committed 43 s after the outputs arrived (023 C3), and rule 8's instrument and the in-cell rule were amended after P2's baseline was measured, though before any variant (§85.10). | Scope it: *"The head-to-head (§71, §87), the development protocol (§85, including P7) and the interpretability tests (§86, §88) were pre-registered and adversarially reviewed. The dissection (§37), warm split (§43), forensics (§42, §46) and EMA results predate that protocol and are reported as exploratory. The cold-cell test cells had been read before the protocol (§44–§46.4), which P7's label discloses. Amendments made after a baseline was measured, but before any variant, are marked (§85.10)."* |
| 3 | MAJOR | overreach | **Abstract (iv) drops the qualifier that makes the pathway claim licensable.** *"a named pathway readout ranks which pathways move in held-out cells at ρ 0.273"* omits *"trained to predict each pathway's response magnitude"*. C 4.16 and review 032 require it, because the readout is scored against its own training target (`aux_targets`, `model_v9.py:237-243`). Without it, the abstract reads as an emergent interpretability finding. | Use C 4.16's sentence in the abstract too: *"a shared linear readout of the named pathway nodes, trained to predict each pathway's response magnitude, ranks which pathways move in held-out cells (ρ 0.273, +0.044 above a cell-agnostic prior)."* |
| 4 | MINOR | overreach | **Abstract (i)–(ii) depart from review 022's wording.** (i) *"it reproduces its published score"*. 022 ask 2 said *"lands inside the band / consistent with, never 'reproduces'"*, and §5.1 already has it right. (ii) The row-pooled 0.473 vs 0.387 appears with no MCF7 qualifier, and "higher on 5 of 8" with no "lower on 3". 022 required both wherever row-pooled appears. | (i) *"… scores 0.386 on the test rows, inside its published 0.383 ± 0.027"*. (ii) Add *"lower on 3 (two beyond row-level noise)"* and *"row-pooled 0.473 against 0.387, dominated by MCF7 (51 % of rows)"*, or drop the row-pooled figure from the abstract. |
| 5 | MINOR | wrong-quantity | **The untrained-floor lesson is attributed to the wrong readout.** §5.4 bullet 2 (*"untrained inits reach the same floor on 2 of 3 initialisations"*) and §6 lesson 3 (*"untrained models reached the gradient readout's floor on 2 of 3 inits"*) both get it wrong. §86's untrained **gradient** diffs were −0.0024 / +0.0235 / +0.0111, and none reached the floor. It was the untrained **output projection** that cleared −0.02 with p < 0.05, on u1 (−0.0234, p 0.023) and u2 (−0.0211, p 0.039) (review 023 C2). The lesson stands, but its measurement is misattributed. | *"untrained models' readouts (the output projection) cleared the −0.02 floor with p < 0.05 on 2 of 3 initialisations, so the within-model permutation p is not calibrated across initialisations."* |
| 6 | MINOR | overreach | **The variance paragraph needs three corrections.** (a) *"about 3.4× the largest candidate effect"* is out of date. 3.4× was against C1's −0.0086 (review 029). After C3's −0.0205, the ensemble is **≈ 1.4×** the largest movement and ≈ 6× the largest gain (C6 +0.0048). (b) Abstract: *"the largest lever we found is seed ensembling (+0.029), not architecture"* sets a change of estimand, at 3× the training compute, beside single-model effects. Use §85.9's retitled form. (c) *"EMA was null twice"* can't carry an along-trajectory conclusion: under WSD, EMA's ~1,000-step window lies inside the annealed tail (review 029 ask 1). | (a) Correct the multiple. (b) *"a seed-specific prediction component (+0.029 from averaging three seeds, at three times the training compute) is larger than any architectural effect we measured"*. (c) Add *"(under the WSD schedule this does not test averaging along the trajectory; V2 does)"*, or drop the EMA clause. |
| 7 | MINOR | code-vs-intent | **§5.3 files the chromatin result under "ablations of a trained model to the mean", but it was a training-time arm on the test cells.** §45: *"v9 trained on `split_cold_cell_1` twice … one arm's chromatin input carries no cell-specific information"* (`--ablate_epi` replaces the values with training means **at training**), one run per arm, scored on the cold-cell **test** cells. §55's +0.0004 is the cluster estimand over the 5 chromatin-covered cells, and §52 puts the effect at 0.57 σ of run-to-run noise. | Move it to its own sentence: *"a retrained arm without cell-specific chromatin (one run each, on the test cells, §45) differs by +0.0004 on the cluster estimand over the five covered cells, 0.57 σ of run-to-run noise (§52, §55)."* Prefer "no detectable effect" to "a clean null" at one run per arm. |
| 8 | MINOR | overreach | **The screen results don't say how many seeds they rest on, and the "reversal" compares unlike things.** (a) C1–C4 are **one-seed** results, dropped at rule 6, while C7, C8b and C6 are three-seed. The loss nulls are *"at the pre-registered weight"* (§85.7). (b) The atom-token *"inference-versus-training reversal"* pairs §37's ablate-to-mean on one fold-0 checkpoint (480 rows per split, median convention) with a one-seed retraining on the cc1 dev carve (mean convention). Those are different splits, estimands and seed counts, and rule 9 records the opposite inference sign with self-attention on. | (a) Label the seeds, e.g. "(one seed; dropped at the screen)", and add "at the pre-registered weight" to the ListNet and DEG nulls. (b) *"an inference ablation (one fold-0 model) and a retraining ablation (one dev-carve seed) disagree in sign"*, stated as an observation, not a finding. |
| 9 | MINOR | overreach | **§5.2: "+0.012 on all 8 metrics".** +0.012 is the Pearson_deg-mean gap only. §43.1 shows v9 ahead on all eight, with gaps from +0.0007 (Pearson_abs) to +0.023 (Spearman_deg). The comparison is also v9 over 3 seeds, with lockstep masks, against XPert's single released checkpoint (epoch 164). | *"v9 is ahead on all 8 metrics (delta Pearson +0.012 [0.011, 0.013], identical rows), three v9 seeds against XPert's released checkpoint"* [§43]. |
| 10 | MINOR | provenance | **Disclosures and details are missing or inconsistent.** (a) Review 022's must-disclose list includes XPert's DataParallel **epoch-0 mask duplication**, which §3's deviation list omits. It also includes the fact that XPert's best checkpoint (epoch 40) came **before the recipe's objective switch at epoch 70**, and the v9-selection bound's **cluster** value (≈ 0.0004), not only 0.0042 row-pooled. (b) §4's *"calibrated against untrained models, not against 0.5 or a trained null alone"*: no trained null was run (§88's preamble says so), and §85.10's readout is calibrated against a column-permutation null and a cell-agnostic prior, not against untrained models. (c) The abstract says "three methods lessons", and §6 lists seven. (d) *"160 of 163 challenges upheld"*: my count of challenge rows in the 35 adjudicated reviews is **151**. | (a) Add the three items. (b) *"calibrated against untrained initialisations (§86, §88) or against permutation and cell-agnostic references (§85.10); the exact trained permuted-drug null was not run"*. (c) Reconcile. (d) Cite a ledger with the counting rule, e.g. `adjudicated/` rows by severity, so the number is reproducible. |

## Answers to the asks

**Ask 1 — C1–C10 are the list.** On the packet's six points:
1. **Abstract (iii):** licensed apart from the atom tokens (C1), the missing "at the pre-registered weight" (C8a), and
   the "largest lever" framing (C6b). The C6 clause is licensed.
2. **§5.1:** the permitted sentence is faithful. It's lightly paraphrased, not verbatim, which is fine. Two
   disclosures are missing (C10a).
3. **§5.3:** C7 and C8. The C6 sentence is exactly §85.10's.
4. **§5.4:** the pathway bullet and the §37 withdrawal are the licensed forms. The abstract's version isn't (C3).
5. **§6:** lessons 1, 2, 4, 5, 6 and 7 are supported as cited. Lesson 3 is misattributed (C5).
6. **§2 forensics:** framed as usage notes, and the numbers match §42, §46 and §69. "Cluster within 0.005" is 0.0051
   (0.7384–0.7435), so "≈ 0.005".

**Ask 2 — what a reviewer of the paper would need, beyond C10:**
- **Which results predate the protocol,** and that the test cells were read before it (C2).
- **That the pathway alignment is so far a dev-cell number.** The test-cell measurement comes with P7 (§85.12 item 8).
- **That every §85 increment was measured on six dev cells,** against a baseline whose own selection saw test
  scores (C2).

## What I checked and found sound

- **Every number in §2, §5.1, §5.3 and §5.4** against the cited sections, plus CLAIMS 1.11, 4.1a, 4.15, 4.16, 6.1 and
  6.13. The exceptions are in C5, C6a and C9.
- **The ⏳ placeholders (P7, V2, §88)** state the pre-registered readings (§85.12 item 6, §90, §88.3/88.6) without
  pre-empting them.

## What I could not assess, and why

- **The "upheld" half of "160 of 163".** Adjudication outcomes live in the PI's records, not in the review files.
