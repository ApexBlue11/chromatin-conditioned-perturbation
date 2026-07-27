# LINCS Model — Mathematical Specification (implemented + verified)

Deterministic, fully-attention-attributable predictor of the 978-gene differential drug response.
**Interpretability first, accuracy second** (user). Predicts `Y` (LINCS Level-5 z-score = already a
vehicle-relative differential) **directly**; `X_base` (cross-platform CCLE) is cell *context*, not a
subtrahend, so the XPert `xpert−xbase` residual does not apply. **No VAE backbone** (avoids the papers'
over-denoising → negative-R² failure). Every prediction traces through inspectable attention + gates.

Code: `model/{config,modules,model,losses,data}.py`. All properties below are checked by
`model/tests/test_model.py` — **20/20 pass** (Kaggle CPU).

## Dimensions
G=978 genes (frozen order), d=`d_model`(256), H heads, M=96 max heavy-atom tokens, L_b/L_p base/perturb layers.

## Tensors (files on disk)
| symbol | shape | file | role |
|---|---|---|---|
| `Y` | N×978 | `Y_target_level5_978.npy` | target (predicted directly) |
| `X[c]` | 978 | `X_base_lincs.npy` | cell context (gated, not subtracted) |
| `E[c]`,`mask` | 978×3 | `E_final.npy`,`_mask` | gate + gene feature |
| `r[c]` | 978 | from `E_final_mask`×`E_reliability.tsv` | availability×reliability∈[0,1] |
| `A[p]`,off | Σn×512,+off | `drug_atom_reprs.npy`,`_offsets` | per-atom UniMol tokens |
| `u[p]` | 2964 | unimol CLS512+chemberta384+desc20+fp2048 | global drug token |
| `P_cop`,`P_ppi` | 978×978 | `A_copathway.npy`,`STRING_adj_978.npy` | gene↔gene bias |
| `δ,τ` | scalar | `signatures_usable.tsv` (parsed "10 µM"/"6 h") | dose/time FiLM |

## Forward (mirrors drug→target→pathway→chromatin→response)
1. **Gate** `EpiGate`: `s_{c,g}=1−r_{c,g}(1−σ(MLP_epi(E)))∈(0,1]`; r=0⇒s=1 (neutral fallback). Gated
   baseline `x̃=X·s`. *[inspectable "can gene g move in cell c"]*
2. **Gene tokens** `h⁰_g=GeneEmb[g]+W_x·x̃_{c,g}+W_e·[E_{c,g};avail]`; **DoseTimeFiLM** (nonlinear
   Fourier features → (γ,β), identity at init, inverted-U-capable — no one-hot dose).
3. **Drug tokens** `Ã=LN(W_a·A)+type_atom` (M×d), global `u_tok=LN(W_u·u)+type_drug`; `D=[u_tok;Ã]`
   ((1+M)×d) with key-mask from atom validity.
4a. **Base/context encoder** (L_b × `BaseLayer`): gene↔gene `BiasedSelfAttention`, drug-free →
   poised cell state. Logits `+Σ_p softplus(λ_{h,p})·log1p(P_p)` on the first `n_prior_heads` heads
   (rest free/discovery; optional hard-masked on-support head). *[α^gene = pathway propagation]*
4b. **Perturbation encoder** (L_p × `PerturbLayer`): atom→gene `MultiHeadCrossAttention`
   *[α^drug = drug-substructure→gene = learned DTI]* → gene↔gene biased self-attn → FFN. Pre-norm+residual.
5. **Head** `Ŷ_g=MLP_head(LN(h^L_g))+b_g`. (Optional output gate `Ŷ·s`, default OFF.)
6. **Loss** `Σ_i w_i·Huber(Ŷ_i,Y_i)` + weight decay; w=cell epi-reliability. Report **R²/magnitude +
   cold-cell**, vs Mean/Meancell/Meandrug baselines. **DTI never in the loss** (learn-then-validate).

## Interpretability & validation
α^drug (atoms→genes), α^gene (pathway), gate s (chromatin) are inspectable. Aggregate α^drug over atoms
→ per-drug gene attribution → **recall@k vs `drug/outputs/dti/dti_reference.tsv`** (ChEMBL-curated +
STITCH≥700). Confidence: OOD proxy (cell similarity vs `ccle_full_background` 1719×978) + epi reliability.

## Verified (test_model.py, 20/20)
shapes · masked-softmax rows=1 · gate∈(0,1] & s≡1 where r=0 · no-Y-leakage · λ→0 zero-bias & masked-head
on-support-only · FiLM identity@init · finite grads · overfit loss→0 & beats Mean · baselines.

## HEADLINE (2026-07-25, v3 checkpoint, fair evaluation)
Measured on **reproducible signatures only** (mean|Y|≥1; n=12,000; cold cells):
| | R² | Pearson (median) |
|---|---|---|
| with epigenetics | **0.3306** | **0.5288** |
| epi ablated | 0.2573 | 0.4903 |
| **Δ (epi contribution)** | **+0.0732 (+28% rel.)** | **+0.0385** |

**Epigenetics contributes materially.** Earlier ablations over ALL signatures reported −0.0107/−0.0015
— dilution by inert perturbations *inverted the sign*. **Rule: stratify by signature strength before
judging any biological feature's contribution.**
Model vs truth on the balanced 42-drug × 6-cell design: corr(overall)=**0.7455**,
corr(interaction)=**0.4494**; variance shares model 58.4/16.6/**24.9%** vs truth 42.3/9.8/**47.9%** ⇒
the model still **under-expresses drug×cell interaction** (MSE shrinkage; reliability weighting did NOT
fix it) — that is the next target, not epigenetics.

## Results log
- **v1** (7/12 epochs, T4, hit 8h budget, not converged): cold-cell R² **0.156** (val 0.208), still rising;
  beats Mean/Meancell 1.819 & Meandrug 1.800 (MSE 1.557). Positive cold-cell R² — no VAE collapse.
  **Epi-ablation ΔR² = −0.011** (removing epi HELPS) → the multiplicative gate latched onto the per-cell
  batch offset and anti-transferred. Phase gap: cold-cell P1 0.069 vs P2 0.222.
- **Epigenetics diagnosis:** the effect is SIGNED not magnitude (low H3K27ac/high H3K27me3 → gene UP;
  r≈−0.17 signed, partial r≈−0.10..−0.14 beyond X_base, i.e. non-redundant). The `s∈(0,1]` multiplicative
  gate structurally cannot express a signed additive shift → v2.
- **v2** (additive signed epi head + per-cell E centering, 7 epochs, same budget): cold-cell R² **0.160**
  (v1 0.156), MSE 1.550 (v1 1.557) — within noise. **Epi-ablation ΔR² = −0.0107, IDENTICAL to v1 ⇒ the
  fix FAILED.**
- **v3** (reliability weighting by MEASURED strength→replicate-r curve; eff. sample size 310k→71k;
  7/12 epochs, budget-capped, still improving): **cold-cell on REPRODUCIBLE sigs: R² 0.300,
  per-signature Pearson median 0.501** (noise ceiling √0.51–0.62 ≈ 0.71–0.79 ⇒ ~65–70% of achievable).
  Aggregate cold-cell R² 0.152 (≈v1 0.156) — unchanged as expected, since weighting de-emphasises the
  inert majority that dominates that number. **Epi harm largely fixed: ΔR² −0.0107 → −0.0015 (~7×),**
  i.e. now ~neutral, not yet positive. Per-cell cold R² spans 10×: HUVEC .223 / U266 .201 / HT29 .181 /
  HA1E .169 / MCF10A .146 … COV644 .034 / NPC .023 ⇒ common cell types learnable, rare ones not
  (this is the real content of the "phase gap").
- **DIAGNOSIS (supersedes the "disable epi" call below).** Measured on REPRODUCIBLE signatures
  (mean|Y|≥1.0; replicate r≈0.37+), balanced 42 drugs × 6 cells:
  - Data reliability: replicate Pearson median **0.127** overall; **~75% of signatures are
    irreproducible** (r 0.07–0.14). Only 16% have mean|Y|>1.0. All-signature R² is meaningless.
  - Truth partition: drug×gene 42.3% / cell×gene 9.8% / **drug×cell interaction 47.9%**.
  - Model: **corr(true,pred)=0.73 overall, 0.43 on the interaction** — much better than the 0.16
    all-signature R² implied. But predicted shares are 61.3/13.3/**25.4%** ⇒ it **under-expresses
    cell-specificity** (MSE shrinkage induced by training on noise).
  - Epi explains ~0.1% of the interaction (H3K27ac r=−0.032, H3K27me3 r=+0.014, correct signs, ~50 SE)
    ⇒ weak predictor, valid mechanistic signal. It was competing for a component the model was being
    trained to shrink — hence it looked worthless.
  **v3 = filter/weight training+eval by signature strength**, then re-test epi fairly.
- (superseded) Ceiling argument: only
  **5.73%** of total Y variance is explainable by ANY static per-(cell,gene) feature (94% is
  drug-specific); epi explains r²≈1.9% of that ⇒ **max ΔR² ≈ 0.107% vs measured cost 1.07%** (10× under-
  water). Disabling epi is a free ≈ +0.011 cold-cell R². METHOD LESSON: the r≈−0.17 signed effect was
  measured on CELL-AVERAGED responses; always convert an averaged correlation into its per-signature
  variance share before building on it. **Architectural implication: ~94% of variance is drug-specific —
  capacity belongs in the drug branch and drug×gene attention, not cell-static features.**

## v1 design decisions (argued both ways, then settled)
Four proposals were stress-tested against themselves; **two were rejected after argument**:
- **Output chromatin gate `Ŷ·s` — REJECTED.** `E` is *baseline* chromatin, but a major LINCS MoA class
  (HDAC/EZH2 inhibitors) works by *opening* chromatin — an output gate would systematically suppress
  exactly the genes those drugs de-repress. It also conflates up/down asymmetry (a silenced gene can't
  fall but can rise), and since only 45/83 cells have epigenetics it would rescale predictions for
  epi-covered cells with no way for the network to compensate (benign at the input, not at the output).
  **Kept at the input; the insight is measured instead** via `gate_diagnostic` (corr of learned `s` with
  observed mean|Y| per gene) — hypothesis tested, not imposed.
- **Plate/batch covariate — REJECTED.** Level-5 MODZ/COMPZ z-scores are already computed against
  plate controls (the correction is in the target). We have only `phase`, not plate. An unseen-at-test
  nuisance embedding is a *training* cheat channel that can hurt generalization, and adversarial removal
  is heavy machinery for speculative gain. **Detect instead of model:** metrics are reported
  **stratified by phase**; if a gap appears we revisit.
- **Loss weighting — uniform.** Epi reliability belongs in the *gate* (it is there); the papers' low-SNR
  warning really targets *signature quality* (`distil_cc_q75`/`is_gold`), which we haven't joined. No
  unvalidated weighting hyperparameter before we know noise is the ceiling. Top v2 candidate.
- **Dose aggregation — deferred.** Aggregation is a workaround for one-hot/discrete dose encodings; we
  use continuous log-dose through Fourier-feature FiLM (inverted-U capable). Aggregating destroys
  information and would change the verified target.

Adopted: **cold-cell k-fold** (below) and an **epi-ablation metric** (`evaluate(..., ablate_epi=True)`
zeroes `E`,`r`) to prove epigenetics is actually used and guard against multimodal collapse.

## Splits / training
**Cold-cell k-fold** (`cell_folds`): all 82 signature-bearing cells partitioned into `n_cell_folds`=5
folds greedily balanced by signature count (loads 62.0k ± 0.1k); MCF10A+NPC pinned into fold 0. v1 tests
on **fold 0 = 17 unseen cells / 62,085 sigs** (train 235,628, val 12,401) — far more reliable than the
original 2-cell test. NOTE: **NPC has no epigenetics in `E_final`** (the Checkpoint-3 "both covered" claim
came from the pre-final coverage report), so fold 0 mixes epi-covered and non-covered cells. `LincsDataset` gathers
by key (normalized, no per-row copies); `collate` pads atoms to M. No local GPU → training runs on Kaggle
GPU (checkpointed for the 12 h cap). Defaults: d=256, L_b=2, L_p=4, H=8, M=96.
