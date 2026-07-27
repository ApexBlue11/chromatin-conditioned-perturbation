# Chromatin-conditioned, attributable prediction of drug-induced transcriptional response

**Status: DRAFT.** Sections marked 🔲 are blocked on measurements that have not been run — they are
deliberately left empty rather than filled with plausible text. Every number below traces to
`model/results/CLAIMS.md` (claim id in brackets) and was produced by a script named in `METHODOLOGY.md`.

---

## Abstract

Predicting how a cell line responds transcriptionally to a small molecule is a central problem in
computational pharmacology, but the field's headline accuracy numbers are difficult to interpret: leading
models predict *absolute* post-perturbation expression while being supplied the unperturbed baseline as
input, so a large and unreported share of their correlation reflects reproduction of that supplied
baseline rather than prediction of the drug effect. We build a deterministic, fully attributable
cross-attention model that instead predicts the *differential* response directly over the 978 LINCS L1000
landmark genes, conditioned on cell-line baseline expression, three chromatin tracks (ATAC-seq, H3K27ac,
H3K27me3), pathway/PPI priors, and atom-level molecular representations. On reproducible signatures the
model attains Pearson 0.44 on unseen cell lines and **0.47 on unseen compounds** under a Bemis-Murcko
scaffold split whose residual chemical leakage we audit and report. Three findings follow. First,
**chromatin state contributes measurably** (ΔR² = +0.089 in distribution) with the biologically expected
sign — low activation and high Polycomb marking predict larger responses — and this survives stratification
by baseline expression, excluding a floor-effect artefact; however its benefit **declines monotonically
with cell familiarity** and vanishes for unseen cells, so it is an in-distribution refinement rather than a
generalisation mechanism. Second, **aggregated atom→gene attention is enriched for known drug targets**,
and the enrichment strengthens monotonically with the confidence of the reference (0.9× on all
STITCH-inclusive edges, 2.1× on ChEMBL-curated, 2.6× on both-source gold), demonstrating that a noisy
reference can entirely mask a real attribution signal. Third, we show that the same predictions score
anywhere from 0.44 to 0.99 Pearson depending only on the basal-to-effect variance ratio of the evaluation
target, so **absolute-convention correlations are not comparable across protocols** unless that ratio and
the basal source are reported. We additionally report a negative result of methodological interest: a
cell-conditional pathway-gating module appeared to contribute ΔR² = +0.103 under a naive ablation, but
contributes ≈0 once the ablation preserves the module's learned scale — a 30× artefact that we trace and
correct.

🔲 *Final sentence pending the protocol-matched comparison (see §6).*

---

## 1. Introduction

🔲 *To draft. Points to make: (i) the perturbation-prediction task and its use in MoA/repurposing;
(ii) why interpretability is the objective here, not accuracy alone; (iii) the metric-convention problem
we quantify in §5.3; (iv) chromatin as an unused conditioning signal.*

## 2. Data

| Branch | Content | Provenance |
|---|---|---|
| Target `Y` | 312,438 × 978 Level 5 MODZ differential signatures, both LINCS phases, 0 NaN; 310,114 usable after SMILES filtering | GSE92742 + GSE70138 |
| Baseline `X_base` | 83 × 978 CCLE expression + 1,719-cell background | DepMap |
| Chromatin `E` | 83 × 978 × 3 (ATAC / H3K27ac / H3K27me3), ±10 kb TSS narrowPeak, + availability mask + per-(cell,mark) reliability | ENCODE / ChIP-Atlas |
| Priors | co-pathway and STRING PPI adjacency, 978 × 978 | STRING v12 |
| Drugs | 21,220 compounds: 20 descriptors, 2048-bit ECFP4, Uni-Mol CLS (512) + **per-atom tokens** (703,851 × 512), ChemBERTa (384, later dropped) | RDKit, Uni-Mol |
| DTI reference | 19,174 drug×gene edges in evidence tiers (`stitch`, `stitch_high`, `chembl`, `chembl+stitch*`) — **validation only, never in the loss** | ChEMBL + STITCH v5 |

**Data quality (6.1).** LINCS L1000 is often described as noisy; that is a mischaracterisation. Replicate
pairs agree at median r = 0.127 across *all* signatures but at **r = 0.509 (phase 1) / 0.619 (phase 2)**
for real responses (mean |Y| ≥ 1), and the same (cell, drug) measured in the two independent productions
agrees at r = 0.561. The low aggregate reflects that **≈75 % of perturbations are biologically inert**, not
corrupted measurement. All evaluation is therefore stratified to the reproducible subset; §5 explains why
this is not cherry-picking but a precondition for correct inference.

## 3. Model

Deterministic cross-attention predictor; **no VAE or diffusion backbone**, chosen so every prediction
remains attributable. Flow: chromatin gate → gene tokens (+ dose/time/lineage FiLM) → base encoder
(gene↔gene, prior-biased) → perturbation encoder (atom→gene cross-attention + gene↔gene) → per-gene head
with an additive signed chromatin term. Predicts `Y` directly; `forward()` never sees the target.
Equations and their implementing modules: `model/MODEL_MATH.md`. 45 code-vs-math unit checks gate every
training run.

## 4. Experimental design

**Splits.** A single training run holds out both a cell fold and a Bemis-Murcko scaffold fold, yielding
three disjoint generalisation tests: unseen cell (47,002 sigs), unseen compound (58,796), unseen both
(15,083); train 179,772. Leakage verified zero.

**Scaffold splitting is not sufficient (6.4).** Auditing residual similarity, the median test compound
still has a training neighbour at **Tanimoto 0.655**, with 39.2 % ≥ 0.70 and 8.0 % ≥ 0.85. Any
unseen-compound number — ours or a published one — is inflated by analogue leakage unless audited; a random
drug split is substantially worse. We report the audit alongside the number.

## 5. Results

### 5.1 Accuracy
Reproducible signatures: unseen cell Pearson 0.440 (R² 0.273), **unseen compound 0.471 (R² 0.188)**,
unseen both 0.451 [1.2–1.4]. All beat Mean/Meancell/Meandrug baselines (MSE 1.481 vs 1.733–1.747) [1.1].
Unseen-compound exceeds unseen-cell, i.e. **new chemistry is easier for the model than a new cell line**
[1.5] — consistent across every analysis we ran.

### 5.2 Chromatin — Figure 2
ΔR² = +0.089 in distribution [2.1], with the mechanistically expected sign [2.2]. The relationship
**survives stratification by baseline-expression quartile in all four strata for all three marks** [2.2a],
excluding the floor/headroom artefact; it attenuates but does not flip at high expression [2.2b].
Contribution declines with cell familiarity: +0.089 (in-dist) → +0.035 (unseen compound) → ≈0 (unseen
cell) [2.3], so we frame it as an in-distribution refinement, **not** a cold-cell generalisation mechanism
[2.6]. Literature is consistent: Polycomb-repressed genes in moderately H3K27me3-marked chromatin remain
inducible, and enhancer priming links naive-state accessibility to stimulus response [L.1–L.2].

### 5.3 Metric convention — Figure 3
Sweeping the basal:effect variance ratio of the evaluation target moves the Pearson correlation of
**identical predictions** from 0.44 to 0.999 [6.6c]; R² roughly doubles even at our own modest ratio
[6.6a]. Absolute-convention numbers are therefore uninterpretable across protocols unless the ratio and
basal source are reported [6.6f]. We explicitly **do not** claim parity with published models on this basis
— that requires a protocol-matched re-benchmark [6.6e, 6.6g], and an earlier attempt of ours to infer a
competitor's variance ratio from its reported PCC was circular and is retracted [6.6d].

### 5.4 Attribution — Figure 1
Atom→gene attention enriches for known targets at 0.9× / 2.1× / **2.6×** across all / ChEMBL-curated /
both-source-gold edges [4.1]; predicted |Ŷ| shows the same monotone pattern [4.2]. Zeroing atom tokens is
the single largest feature ablation (Δpearson +0.163), so the attribution substrate is load-bearing [4.3].
The effect is **modest** (recall@50 ≈ 10 %), partly a biological ceiling: a drug's target gene is often not
where its transcriptional response peaks [4.4]. Correct wording is "enriched for", not "recovers" [4.5].

### 5.5 Cell-specificity — Figure 4, Figure 6
The model expresses 26.5 % of variance as drug×cell interaction against 47.9 % in truth. This is largely
**correct**: it sits at MSE-optimal dispersion (std-ratio 0.47 ≈ corr 0.42) [6.8], and interaction
expression rises monotonically with signature reliability [Fig 6]. A magnitude-invariant correlation loss
designed to "fix" it produced no improvement [6.9], confirming that interaction magnitude cannot rise
without first improving cell-specific correlation [6.10].

### 5.6 A 30× ablation artefact — Figure 5
A cell-conditional pathway-conductance module appeared to contribute ΔR² = +0.103 (unseen cell) under
ablation-to-one. Its learned conductance averages 0.60, so that ablation also rescales the pathway output
by 1.67×. Ablating instead **to the module's mean** — preserving scale, removing only structure — gives
**−0.003 / +0.006** [3.1a]. The module is 84.7 % per-cell scalar and correlates 0.74 with baseline
expression [3.1b]. We retract the original claim [3.1] and state the general lesson: **ablate a learned
multiplicative component to its mean, never to 1 or 0** [3.6]. It remains a genuine chromatin readout
(partial correlations −0.26 / −0.35 / +0.20) that buys no accuracy [3.3a].

### 5.7 Feature ablations
ECFP4 fingerprints and per-atom Uni-Mol tokens are the two pillars (ΔR² +0.089 / +0.075) [5.1]; ChemBERTa
is dead weight and was dropped (ΔR² +0.001) [5.2]; RDKit descriptors negligible [5.3]; cell lineage
marginal, unsurprising since 26/83 cells lack a DepMap lineage including 7/17 test cells [5.4].

## 6. 🔲 Protocol-matched comparison — NOT YET RUN
Requires LINCS **Level 3** (confirmed: the comparison paper states *"we used level 3 quantile-normalized GE
profiles from phase I"*) plus paired unperturbed profiles, then re-benchmarking an open competitor under a
single protocol — reproducing its published number on its own split first as a sanity check [6.7].
**Until this exists we make no comparative accuracy claim.**

## 7. 🔲 Cross-validation — NOT YET RUN
All numbers derive from cell fold 0. k-fold across the remaining folds is required before treating them as
robust.

## 8. Limitations
Single fold (§7); no protocol-matched baseline (§6); chromatin covers 45/83 cells with no imputation;
attribution is modest and evaluated in-distribution; the epigenetics benefit does not transfer to unseen
cells [2.6]; novelty of atom→gene attention relative to XPert is **unresolved** [7.3]; the chromatin-
conditioning novelty claim rests on targeted, not systematic, search [7.1].

## 9. Reproducibility
Code, environments (three, pinned), figure-generation script and the full claims ledger — including
retracted claims — are in the repository. Data is public; `METHODOLOGY.md` §9 gives the exact run order.
Unit tests (45) must pass before any training run.
