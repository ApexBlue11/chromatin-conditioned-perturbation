# v10: the architecture and target audit, and the plan to fix it

**Opened:** 2026-10-13, at the principal's request ("there must be some mistake or multiple mistakes ... identify and
confirm all of them and then make a plan to fix all of them").
**Logged as:** RESULTS §99, which registers Phase 1 (§5 here) before any Phase 1 code runs or any test row is read.
**Critic:** packet 083.

**Status of each part:**

| part | status |
|---|---|
| §2 reference models | read from their code (commits below), nothing run |
| §3 findings A0–A3 | **measured**: training rows only; no model, prediction or test row |
| §4 mistakes M1–M10 | each marked CONFIRMED, CODE FACT or TO CONFIRM |
| §5 Phase 1, tests C1–C4 | **registered**; no Phase 1 code exists yet |
| §6 Phase 2, fixes F1–F9 | **draft**; each is registered in its own packet before any of its code |

---

## 1. Summary

The principal's worry was that the model "doesn't really seem to use the drug input or the chromatin input". That
worry is right, but the largest cause is not inside the network.

- **The target we train and score on is mostly the plate, not the drug.** That target is XPert's Level-3 delta,
  y = X − X_ctl: a treated profile minus the control profile from its own plate. On 31,722 training rows:
  - a predictor that never sees the drug scores **r = 0.662** (the mean response of the *other* drugs on the same plate);
  - the drug's own measured response from other plates, added at the best weight, raises it to **0.675**.
  - So **on this target the drug is worth at most about +0.013**.
- **That explains the drug-free tie (R1, §98.11)** without any bug in the network: v9 scores 0.642 and Bai et al.'s
  control-only MLP 0.635.
  - It also undercuts the drug-blind retrain (B1/R2), PertDiT's head-to-head (D) and the XPert comparison (O9) as tests of
    drug use. All of them are read on the same target.
  - It probably also explains part of Bai et al.'s field-wide finding that seven L1000 models do not use their drug
    features (§50): they score on the same target.
- **With the plate removed** from the target (each well minus the mean of the other drugs' wells on its plate, CMap's
  Level-4 logic), drug-free information predicts almost nothing (r 0.012). The drug's own replicate on other plates
  predicts it at r 0.147, and 0.189 for the strongest third.
  - So the drug signal is weak but real, and **only the drug can supply it**. That is the target a drug-response model
    must be trained and judged on.
- **On top of the target, the network has its own faults** (§4):
  - the pathway layer reads before the drug enters;
  - chromatin has no way to interact with the drug;
  - dose is not tied to the drug;
  - nothing forces the response through the drug;
  - the drug encoder is one linear layer over 2,580 numbers, with 1,581 training compounds;
  - an input leak for unseen cells, flagged by review 001 and never fixed, is still in the code.

## 2. What the published models do, read from their code

Cloned 2026-10-13 into `external/ref_models/` (gitignored, code only, nothing executed). The first five are at the
commits Bai et al. benchmarked (`external/bai_benchmark_1d750cd/reference/PROVENANCE.md`). XPert and PertDiT were
already local (`external/xpert/code`, `external/bai_benchmark_1d750cd/models/PertDiT`).

| model | what it predicts | control / cell input | drug encoder | how the drug reaches the output | dose |
|---|---|---|---|---|---|
| DeepCE (NMI 2021, `9b0d04f`) | CMap Level-5 z-score signatures; 7,122 signatures, all 24 h | cell one-hot, no control | neural fingerprint (graph conv, trained) | gene-to-atom attention, **and the pooled drug vector is concatenated onto every gene** just before the 2-layer head | one-hot embedding |
| CIGER (Patterns 2022, `81c16f1`) | z-score ranks (listwise / pairwise ranking losses) | cell one-hot | ECFP or neural FP | drug + cell + dose **concatenated onto every gene token at the input**, then attention | one-hot |
| MultiDCP (PLoS CB 2022, `36ecdef`) | z-score signatures, multiple doses | cell **basal expression**, encoded to 50 dimensions | neural FP | as DeepCE, plus the 50-d cell vector concatenated per gene | embedding |
| PRnet (Nat Commun 2024, `f19174b`) | perturbed expression, Gaussian NLL | control profile → small latent z | FCFP4 bits **× log10(dose + 1)** → MLP adaptor | decoder(z, drug, noise) | **multiplies the drug vector** |
| TranSiGen (Nat Commun 2024, `8ec2218`) | treated profile x2 from basal x1 (VAE); loss includes MSE on (x̂2 − x1) vs (x2 − x1) | basal x1 → latent z1 | pretrained KPGT embedding → MLP | z2 = f(z1, drug) | — (not checked) |
| chemCPA (`43e830e`, HEAD) | perturbed expression, Gaussian | expression → basal latent, with an **adversary trained to remove drug identity from it** | pretrained molecular embedding → MLP | **z = z_basal + s(dose, drug) · e_drug** + cell embedding, then decoder | learned dose-response curve **scales the drug vector** |
| STATE (Arc, `9bbfe78`, HEAD) | set of perturbed cells, energy / Sinkhorn loss, residual prediction | set of control cells | perturbation embedding | **added to every control-cell token**, transformer over the set | — (not checked) |
| XPert (NMI 2026) | Level-3 delta, as ours | control profile | Uni-Mol atoms + global, drug self-attention inside every cross-encoder block (§47) | dual-branch cross-attention | dose and time inputs |
| v9 (ours) | Level-3 delta (+ L5 auxiliary) | plate control + cell-average control, each through 2 transformer blocks | Uni-Mol CLS 512 + 20 descriptors + 2048 Morgan bits → **one Linear(2580 → 256)**, RMS-normalised; Uni-Mol atoms, not contextualised | 4 cross-attention residuals after 2 drug-free blocks | **FiLM on the gene tokens before the drug is seen** |

**Three things the strongest designs do that v9 does not:**
1. **Predict a plate-normalised signature** (DeepCE, CIGER, MultiDCP: CMap Level 4/5, where each well is z-scored
   against the plate's population), or bottleneck the control so it cannot carry the answer (PRnet, TranSiGen).
2. **Tie dose to the drug** (PRnet multiplies; chemCPA learns a dose-response curve that scales the drug vector), so
   zero dose means zero drug effect.
3. **Route the effect through the drug by construction:**
   - chemCPA's adversary strips drug identity from the basal latent;
   - STATE adds the perturbation to every token;
   - DeepCE, CIGER and MultiDCP concatenate the drug onto every gene.

## 3. Findings, measured (`model/v9/audit99_target_sources.py` → `model/results/audit99/target_sources.json`)

**Rows and proxy:**
- Training rows only, for `split_cold_drug_1` (A0–A2) and `split_cold_cell_1` (A3).
- The bundle has no plate key. The **plate proxy** is the control group: rows whose `X_ctl` vectors are byte-identical.
- Per-row Pearson over the 978 genes, as every reading in this project uses.

**A0, structure** (55,385 rows, 1,581 compounds, 39 cells):
- 12,241 distinct control profiles. **85.5 % of rows share their exact control with at least one other row.**
- Every (drug, cell) has ≥ 2 rows (median 7, across doses and times). No (drug, cell, dose, time) repeats.
- In the test sets (identity only, no values read):
  - **unseen compounds:** 10,524 of 13,445 test rows sit on a plate with ≥ 2 other-drug *training* wells, and 8,878
    have ≥ 2 other-drug *test* wells;
  - **unseen cells:** 16,326 of 21,321 have ≥ 2 other-drug test wells, and **none** shares a plate with a training row.

**A1, what predicts the target y = X − X_ctl** (44,894 rows that have both P and D):

| predictor | needs the drug? | r |
|---|---|---|
| the cell's mean delta | no | 0.162 |
| F0 = −(X_ctl − cell mean control) + cell mean delta (§98.6) | no | 0.399 |
| **P: mean delta of the other drugs on the same plate** | **no** | **0.651** |
| D: same drug and cell, measured on other plates | yes | 0.238 |
| P + D − cell mean | yes | 0.571 |

The same holds in every strength tercile of ‖y‖:

| tercile | P | D |
|---|---|---|
| weak | 0.654 | 0.243 |
| mid | 0.660 | 0.218 |
| strong | 0.638 | 0.253 |

**A2, how much the drug can add, and the target without the plate** (31,722 rows; D now matches time as well):
- **The drug's increment:**
  - P alone: **0.662**;
  - P + w·(D − cell mean), best w = 0.2: **0.675**;
  - **so the increment is +0.013.**
- **Plate-contrast target** y_c = X − mean X of the other-drug wells on the plate (26,892 rows):
  - the same (drug, cell, time) on other plates predicts it at **r 0.147** (median 0.111);
  - by strength tercile: 0.121 / 0.132 / **0.189**;
  - its drug-free predictor (the cell mean of y_c) scores **0.012**.

**A3, unseen cells: the drug-free term the cell-average input enables** (`split_cold_cell_1`, 47,509 training rows):

| predictor | r |
|---|---|
| −(X_ctl − cell mean control), alone | **0.362** |
| the cell's mean delta, alone | 0.172 |
| both (F0) | 0.397 |

**Disclosure:** A0–A3 were computed before this registration, as desk diagnostics on training rows (scratch on
2026-10-13, then this script, which reproduces them). No test row, model output or prediction was read. The
test-set row counts in A0 are identity-level (plate membership only).

## 4. The mistakes

| # | mistake | evidence | status | what it explains | fix |
|---|---|---|---|---|---|
| **M1** | **The training and evaluation target is plate-dominated.** On y = X − X_ctl a drug-free predictor reaches 0.662, against 0.675 with the drug's own measured response added at the best weight | A1, A2 | **CONFIRMED** | R1's tie with the drug-free MLP; why no drug change ever moved accuracy much | F1 |
| **M2** | **Plate shortcut through the exact control.** The control vector identifies the plate, and the plate's mean response is learnable from training wells | A0: 85.5 % share, and 78 % of unseen-compound test rows have ≥ 2 training plate-mates. R1's strata (§98.11): the drug-free MLP is ahead on shared-control rows (−0.011), v9 on new controls (+0.044) | **CONFIRMED** (structure); its use by v9 is C1 / C3 | the shared- vs new-control pattern in R1 | F2 |
| **M3** | **Our tests of drug use cannot detect it.** R1, R2 (B1), O9 and D are all read on M1's target, where the most the drug can add is ≈ +0.013 | A2, plus R1's CI half-width ≈ 0.005 | **CONFIRMED** (arithmetic) | why B1 would almost certainly read "no difference" | C1 now, F1 later |
| **M4** | **Transductive cell-average input for unseen cells** (`xpert_arm.py:311`): with no training rows, `x_cell` is the mean control over the cell's *test* rows. XPert has no such input | code; review 001 C4 (§51.5) asked for a rerun that was never done; A3: the term it enables scores 0.362 | **CODE FACT**; its effect on the unseen-cell lead is **C2** | possibly part of v9's unseen-cell lead over XPert (+0.050, 6/8 cells) | F3 |
| **M5** | **The pathway layer reads before the drug enters** (`model_v9.py` forward: `self.pathway(h)` runs before the perturb blocks) | code | **CODE FACT** | §88's null (no drug-dependent mechanism); §85.14's licence is about *cell* state, not the drug | F4 |
| **M6** | **Chromatin cannot interact with the drug.** It enters as part of each gene's identity per cell (`GeneRepresentation`), and as an additive per-gene offset with no drug input (`SignedChromatinHead`: contrib = f(E)·r). With 39 training cells it can act as a cell ID | code; T4: blanking it at inference *improves* unseen dev cells by +0.006 (§91.11); the gain is gene-generic (§91.12, §93.17); nothing beyond basal expression (§44); final +0.0004 (§55) | **CODE FACT + MEASURED** | every chromatin null and the small harm | F8 |
| **M7** | **Dose is not tied to the drug.** `DoseTimeFiLM` modulates the gene tokens before the drug is seen; the drug token is RMS-normalised to a fixed size; the drug enters only through softmax attention (a convex average) | code | **CODE FACT**; its effect is **C4** | no zero-dose anchor, no magnitude channel | F5 |
| **M8** | **Nothing forces the response through the drug.** Two full transformer stacks read the controls; the drug arrives late as 4 attention residuals; no adversary, no bottleneck | code; together with M1, gradient descent can reach most of the loss without the drug | **CODE FACT** | why the drug-free and drug models converge to the same solution | F6 |
| **M9** | **Drug encoder.** 2,580 inputs (2,048 of them fingerprint bits) through one `Linear(2580 → 256)`, with 1,581 training compounds; atoms not contextualised (`drug_self_attn` False) | code; §37: removing the atoms at inference *helps* (−0.007 / −0.025 / −0.022); C1 (no atoms in training) hurt dev | **CODE FACT + MEASURED** | weak transfer to unseen chemistry | F7 |
| **M10** (minor) | **Stochastic depth is highest on the drug blocks** (rates rise 0 → 0.1 with depth, so the 4 perturb blocks get 0.04–0.10) | code | **CODE FACT** | slightly less reliance on the drug path | F9 |

**What is not a mistake:** reliability weighting, the delta anchoring of the absolute head, the separate control
encoder and the binned expression encoder. These were each measured before (§27–§34) and are not implicated above.

## 5. Phase 1: confirm on test data (registered; CPU; about 0 GPU-hours)

**Common rules:**
- **The plate proxy** is the control group, as in §3.
- **Within-plate contrast:** for a quantity v (truth or prediction) on row i,
  c_i = v_i − mean(v_j) over the *scored* rows j on the same plate with a different compound, required to number ≥ 2.
  - It removes every plate-level component, from truth and prediction alike. A model that predicts the same response for
    every drug on a plate scores no correlation by construction.
  - Per-row Pearson of c(prediction) against c(truth). Rows where c(prediction) has zero variance are scored 0 and counted.
- **Readings:**
  - **compound split:** P9's per-molecule machinery (`score_p9.molecule_reading`, `verdict`), on the clean subset (§96.7);
  - **cell split:** §71.3's cell machinery (`coldcell_h2h`).
- **v9's row score** is the mean of the three per-seed Pearsons, as of record (§96.3).
- **Code** is written after this registration, reviewed (packet 084), and run once.

**C1 (the decisive one): does v9 use the compound, judged on a target where only the compound can help?**
- **Rows:** P9's unseen-compound test rows with ≥ 2 other-compound test plate-mates (8,878 before the clean filter).
- **Primary:** v9 (P9, 3 seeds) against **M+dt**, Bai et al.'s drug-free MLP with dose and time (§98.13), on the
  within-plate contrast. M+dt is the right comparator here: it knows dose and time, so a v9 margin over it is compound
  information.
- **Reading, mechanical (`verdict`):**
  - **v9 better** → *"v9 uses the compound: on the within-plate contrast it beats a drug-free model with dose and time"*.
    M1 then stands as the reason R1 could not see it.
  - **No claim** → *"v9's compound use is not detected even where only the compound can help"*. M8 and M9 then become the
    leading explanations.
  - **M+dt better** → stated as such.
- **Reported, never read:**
  - v9 per seed;
  - v9 against 0: the per-molecule mean contrast r and its sign test;
  - M (control-only): zero by construction, stated;
  - the contrast-magnitude ratio, median over rows of ‖c(ŷ)‖ / ‖c(y)‖, for v9 and M+dt (how much each model varies its
    predictions across compounds, relative to how much the truth varies);
  - **C1b**, the oracle-plate version on all 10,650 rows with ≥ 2 other-compound plate-mates in any split
    (truth − P_meas against prediction − P_meas, where P_meas is the measured mean delta of the plate-mates);
  - **C1c**, the zero-parameter plate predictor (mean delta of the other-compound *training* wells on the plate), scored
    on the original target: what the benchmark gives away with no drug.

**C2: how much of v9's unseen-cell lead is the transductive cell average (M4)?**
- **C2a (zero-parameter, reported):** on P7's 21,151 rows, per cell:
  - F_tr = −(X_ctl − the cell's mean control over its own test rows), the term v9's input makes available;
  - F_ind = −(X_ctl − the mean training control over all training cells), inductive;
  - each beside v9 (P7) and XPert (O2).
- **C2b (inference, Kaggle CPU):**
  - **Runs:** P7's three final-snapshot checkpoints, run twice: intact, and with `x_cell` replaced by the global
    training-row mean control (review 001's requested rerun, §51.5 item 6).
  - **Reported:**
    - the per-cell Δ (intact − replaced) and its cluster mean;
    - §71.3's cell-level comparison against XPert, redone with the replaced predictions. That comparison is
      final-snapshot against final-snapshot, so the intact final snapshot is reported beside it.
  - **Reading, fixed now:** if the replaced cluster mean against XPert is **below half** of the intact final snapshot's,
    the record says *"v9's unseen-cell lead over XPert is substantially carried by the transductive cell-average input"*;
    otherwise *"it is not"*.
- **C2c (reported):** the within-plate contrast on the unseen-cell test rows (16,326 rows), v9 (P7) and XPert (O2) side by
  side. Does either model separate compounds on the same plate in an unseen cell?

**C3: does the trained v9 depend on the right compound? (inference, Kaggle CPU)**
- **Runs:** P9's three final-snapshot checkpoints. Each test row is run intact and with its drug inputs (`u_feats`,
  `atoms`, `atom_mask`) replaced by those of a different test compound measured in the same cell. The replacement is a
  derangement, seed 0; control, dose, time and cell are unchanged.
- **Reported:**
  - the per-row r on the original target, intact − swapped;
  - the same on the within-plate contrast (C1's rows);
  - the sensitivity, median ‖ŷ − ŷ_swap‖ / ‖c(y)‖.
- **Reading:**
  - contrast Δ per molecule > 0 with cluster CI excluding 0 → *"v9's predictions depend on the right compound"*;
  - otherwise → *"not shown"*.

**C4: is the predicted response tied to dose? (M7; reported only)**
- **Groups:** (compound, cell, time) in the unseen-compound test rows with ≥ 3 doses and plate-mates available.
- **Per group:**
  - ρ_true = Spearman(dose, ‖y − P_meas‖);
  - ρ_pred = Spearman(dose, ‖ŷ − P_meas‖), for v9 and M+dt.
- **Reported:** the mean of each, the fraction of groups with ρ > 0, and corr(ρ_true, ρ_pred) across groups.

**Order:** C1, C1b, C1c, C2a and C4 rescore saved predictions (minutes, local CPU, GPU hidden). C2b and C3 run as Kaggle CPU
kernels.

## 6. Phase 2: the fixes (draft; each registered before its code)

| fix | for | what changes | how it is judged |
|---|---|---|---|
| **F1** | M1, M3 | **Train and evaluate on a plate-normalised target.** Primary: the within-plate contrast of §5, applied to the training rows (A2's y_c); rows with < 2 plate-mates carry no contrast loss. Alternative to decide at registration: CMap Level-5 MODZ signatures, needing GSE92742/GSE70138 Level 5 | **primary reading:** the contrast r per molecule or cell. **Reported beside it:** the field's metric, for comparability, and the noise ceiling (A2's 0.147, by tercile) |
| **F2** | M2 | **Remove the plate identity from the input:** feed the cell's baseline rather than the exact plate control (or substitute controls within cell during training, IDEAS B3 i). Add a plate-disjoint evaluation where the data allow it (IDEAS B3 ii) | C1-style contrast; the shared- vs new-control strata of R1, which should close |
| **F3** | M4 | Compute the cell average from **training rows only**. An unseen cell gets an external baseline (CCLE, or a registered rule over its non-test controls), **given equally to every compared model** | the unseen-cell head-to-head is re-read under the fair input |
| **F4** | M5 | **Drug-conditioned pathway layer:** the named pathway readout moves after the drug blocks. Option: the drug's effect passes *through* the named nodes (drug → pathway activations → gene changes), so the mechanism readout is the model's own bottleneck | §88's mechanism test, re-run |
| **F5** | M7 | **chemCPA-style dose:** e(drug, dose) = s(log dose \| drug) · MLP(drug), where s is a learned sigmoid with drug-amortised parameters, so zero dose gives zero effect. Time modulates the drug effect, not the cell tokens | C4's ρ agreement; dev contrast |
| **F6** | M8 | **Explicit composition:** ŷ = baseline(cell) + Δ(drug, dose, time, cell), with Δ the only drug-dependent term and the contrast target predicted by Δ alone (STATE-like residual). An adversary on the basal path (chemCPA) is used only if the basal path still carries drug information | C1 / C3 on dev; the drug-free arm's contrast score must stay ≈ 0 |
| **F7** | M9 | **Drug encoder:** compact pretrained embeddings (Uni-Mol CLS; MolT5 / KPGT optional) → 2-layer MLP with dropout; fingerprints, if kept, through a regularised branch; atoms either contextualised (drug self-attention on) or removed, chosen on the dev carve | the cold-drug contrast r |
| **F8** | M6 | **Chromatin as a gate on the drug's effect:** Δ_g ← Δ_g · (1 + g(E_g, r_g)), so accessibility changes how far a gene can move under a drug. It is trained on the contrast target and kept only if it beats no-chromatin by the §85 rule on the dev carve. **Otherwise chromatin is removed from v10 and the paper says so** | §85 rules 7–8 on the contrast target |
| **F9** | M10 | No stochastic depth on the drug path | part of the v10 base |

**Order and gates:**
1. **F1 + F3 (data and target; CPU):** build the contrast target and the fair cell baseline; re-score every existing model
   on them (v9 P7/P9, M, M+dt, XPert O2, and later O9 and PertDiT), reported.
2. **F2.**
3. **The v10 base = v9 + F4 + F5 + F6 + F7 + F9,** screened on the dev carve against v9 retrained on the contrast target
   (one change at a time where the screen allows, §85's rules).
4. **F8 (chromatin)** last, on the v10 base.
5. **Final runs** on `split_cold_drug_1` and `split_cold_cell_1`. The head-to-heads are re-read on both targets, the
   contrast target primary for any claim about drug use.

**GPU:**
- About 1.7 GPU-h per dev seed. A three-seed screen of 5 components is ≈ 25 GPU-h, and the final runs ≈ 12 GPU-h.
- That is about two Kaggle weeks. Phase 1 and F1 / F3 are CPU.

## 7. What changes now

- **Saturday 17 Oct reset:**
  - **O9 session 3** and **D's timing run** go ahead (the head-to-head is part-spent; PertDiT is the field's comparison).
  - **B1 (the drug-blind v9 retrain, ≈ 6 GPU-h) is held**: its registered reading is on M1's target, where the drug is
    worth ≈ +0.013 at most, and C1 asks the same question for free. Its kernel stays committed.
- **Claims:**
  - **"v9 does not beat a drug-free model on unseen compounds"** (R1) stands as measured. It is now worded as a property of
    the target, not of the model, until C1 reads.
  - **The unseen-cell lead over XPert** (§85.14) carries M4's caveat until C2 reads.
  - **Chromatin** stays closed for accuracy under v9 (§92.14). F8 is a new mechanism on a new target, not a re-test.
- **The paper:** M1 / M3 is a finding in its own right, if C1 confirms it: *the standard L1000 benchmark target is
  dominated by plate structure, so "do models use the drug" cannot be answered on it.* Bai et al.'s conclusion needs to
  be re-examined on their own single-dose data before we say anything about it.
