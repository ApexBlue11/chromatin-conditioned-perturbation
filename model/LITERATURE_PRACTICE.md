# Are we practising bad DL? — literature audit of our METHOD (2026-07-30)

Scope note: `results/CLAIMS.md` §6b already audits our **biology** against the literature, and §7 audits
**novelty**. Neither audits our **methodology**. This file does that: what the field's own critics say, where
we are already aligned or ahead, where we are weaker, and what to take from other papers' limitations and
future-work sections. Sources at the bottom; every "we" claim cites a claim id from `results/CLAIMS.md`.

The most useful sources are not the model papers — they are the three adversarial benchmarking papers that
exist specifically to catch what we are trying to catch in ourselves.

---

## 1. What the field's own critics say

**Ahlmann-Eltze, Huber & Anders, Nature Methods 2025** — *"Deep-learning-based gene perturbation effect
prediction does not yet outperform simple linear baselines."* Five foundation models plus two DL models,
none of which beat deliberately simple baselines:
- **Mean baseline**: always predict the mean over training perturbations. Beat none of the DL models for
  *unseen* perturbations.
- **Linear model**: solve `min ||Y_train − (G W Pᵀ + b)||²` with `G` a gene basis (PCA of training data) and
  `P` a perturbation embedding, `b` = row means. This is the baseline that matters.
- **Additive model** for combinations: `ŷ = y_A + y_B − y_∅`.

Their explicit recommendations: use **more than one metric** (L2 in expression space, not correlation
alone); rank reported gene subsets by **baseline expression, not differential expression** — the latter
"cannot be applied in real-world use cases" because it needs the ground truth; **beat simple baselines
before claiming success**; and **report prediction variance across conditions**, because underfitting shows
up as predictions that barely vary.

**scPerturBench, Nature Methods 2025** — 27 methods, 29 datasets, 6 metrics (MSE, PCC-delta, E-distance,
Wasserstein, KL, Common-DEGs), split into *cellular-context generalisation* and *perturbation
generalisation*. Headline: the field needs **cellular-context embedding** to generalise to unseen contexts.
Warnings: **Wasserstein fails in high dimensions under variance scaling**, and **E-distance can miss
disrupted gene–gene dependencies** — so metric count is not metric quality.

**"Benchmarking virtual cell models for in-the-wild perturbation response" (2026)** — critiques evaluation
on curated strong-effect perturbations, weak baselines, and absent noise modelling. Recommends **stratified
assessment across strong / moderate / minimal effect sizes**, **noise-ceiling estimation**, and
per-prediction confidence.

---

## 2. Where we are already aligned — or ahead

| Practice | Us | Field |
|---|---|---|
| **Noise ceiling computed before chasing the gap** | replicate r 0.127 all signatures vs **0.509 / 0.619** reproducible [6.1]; method rule #4 | recommended by the in-the-wild benchmark; **most papers skip it** |
| **Prediction variance reported, not just correlation** | std-ratio **0.47** vs corr 0.42 ⇒ at MSE-optimal dispersion [6.8/6.10]; under-expression is near-optimal under noise, and a correlation loss aimed at it changed nothing | Ahlmann-Eltze recommendation #6. Rarely done |
| **Delta target, R² reported alongside correlation** | we predict `Y = xpert − xbase` and report R², per-gene R², MSE [1.2–1.4] | their central metric critique; delta is standard, R²-alongside is not |
| **Three generalisation axes from one run** | unseen cell / unseen compound / **unseen both** [1.2–1.4] | the latent-diffusion paper reports unseen-cell and unseen-compound separately; unseen-both is rarer |
| **Reliability weighting for low-SNR labels** | per-signature weight from the measured strength→replicate-r curve [6.1] | StateXDiff does something comparable; most do not |
| **Adversarial self-testing** | we falsified our own headline interpretability claim [4.1a/4.1b] and a +0.103 that was a scale artefact [3.6] | XPert claims atom-level SAR interpretability with no target-recovery null test in the accessible material. Our negative result is a contribution *against* current practice [4.1c] |
| **Uni-Mol input integrity** | **verified 2026-07-30: 0/21,220 conformer failures, 0 drugs truncated at the 96-atom cap, 0 missing CLS embeddings** | Uni-Mol's documented downstream failure is exactly this (it loses to SOTA on SIDER because 3D conformers fail for natural products and peptides). **Does not affect us** |

---

## 3. Where we are weaker — ranked, each with the fix and its cost

### G1. No linear baseline. Only mean-based ones. ← the one that matters
We compare against Mean / Meancell / Meandrug (cold-cell MSE 1.733–1.747 vs our 1.481). The Nature Methods
critique is not about mean baselines — it is about a **ridge-style linear model on drug × cell features**,
which beat every deep model tested. We have never fit one. Until we do, "beats all naive baselines" is a
weak claim and a reviewer will say so.
- **Fix**: closed-form ridge from `[ECFP4 2048 | descriptors 20 | lineage 16 | dose | time]` → `Y[978]`,
  fit on the train split, scored with the identical protocol/stratum/splits.
- **Cost**: minutes of CPU. 2085 features ⇒ `XᵀX` is 2085², trivially solvable. **Do this before any
  accuracy claim.** No accelerator needed, so the queued TPU run does not block it.

### G2. Our evaluation stratum is defined by the ground truth we score against
Method rule #1 is "always stratify to mean|Y| ≥ 1", where `mean|Y|` comes from the same Level-5 measurement
we then evaluate against. Ahlmann-Eltze call this out directly: selection by differential expression
"cannot be applied in real-world use cases". Two honest qualifications: our rationale is **measurement
reliability**, not effect size (r 0.127 → 0.509/0.619), which is a different argument from cherry-picking;
and the bias direction is not obviously optimistic, since selecting on `mean|Y|` also admits rows that are
large because their *noise* was large, which depresses measured performance. The real problem is
**auditability and prospective applicability**, not inflation.
- **Fix (a)**: report **every stratum** — bins of mean|Y| (<0.5, 0.5–1, 1–2, ≥2) — instead of only ≥1. The
  in-the-wild benchmark recommends exactly this. Nearly free: we already have per-signature strength.
- **Fix (b)**: where one stratum must be chosen, define it by something that does **not** use the evaluated
  measurement — leave-one-out compound/dose mean strength, or a held-out replicate. Small script.

### G3. One fold, one seed ⇒ no error bars on effects we quote to three decimals
Every number is cell-fold 0, single run. We make Δ claims from +0.006 to +0.089 with no run-to-run variance,
and we already know this project produced a +0.103 that was an artefact [3.6]. A Δ of +0.006 is
indistinguishable from seed noise without a variance estimate.
- **Fix**: ≥3 seeds on one config, or k-fold over the 5 cell folds already defined in `cell_folds`.
- **Cost**: accelerator time — the binding constraint. Until then, **quote small deltas as "within
  unmeasured seed variance"**, which costs nothing and is simply accurate.

### G4. No unseen dose/time split
We have nonlinear dose/time FiLM conditioning [5.4-adjacent] but never test generalisation to unseen dose or
time. XPert ships a `cold_dose&time` split. Ours is a split definition, not a model change.
- **Cost**: cheap; reuses the existing eval harness.

### G5. Narrow metric set
We report R², per-gene R², per-signature Pearson, MSE. The closest metric to how a biologist would use the
output — **Common-DEGs**, the overlap of top-k differentially expressed genes between predicted and true —
is missing and is cheap. Do **not** reflexively add Wasserstein or E-distance: the same benchmark that lists
them documents their failure modes.

### G6. No per-prediction confidence
A recurring future direction across all these papers. Our reliability weighting is input-side only. Worth
noting that the latent-diffusion paper **tried** and its predicted variance "converged to a near-constant
value across conditions, precluding its direct use as a condition-aware confidence signal" — so this is
genuinely open, not merely unimplemented.

---

## 4. What to take from their limitations / future-work sections

1. **Our contribution is another paper's stated gap.** The 2026 latent-diffusion paper conditions on basal
   expression, structure, dose and time and lists what it lacks: *"additional cellular information like
   chromatin state or pathway activity that could improve condition-dependent predictions."* That is
   verbatim our design. Cite it as motivation rather than asserting novelty ourselves — and it strengthens
   the narrowed novelty claim [7.1].
2. **But pair it with our own negative result.** scPerturBench's headline recommendation is cellular-context
   embedding for unseen contexts. We built one and measured **≈0 benefit on unseen cells** [2.3]. Reporting
   that honestly is more valuable than the citation: we tested the field's recommended direction and it did
   not transfer.
3. **Level 3 is the comparison substrate.** The latent-diffusion paper uses **level-3 quantile-normalised**
   phase-I profiles and does not discuss inert perturbations or replicate reproducibility at all. We have
   Level 5 only [8]. This blocks a protocol-matched SOTA comparison and is already open item #2.
4. **Selective omics beats piling omics on.** Multi-omics DRP work finds two complementary layers give
   maximal accuracy at lower complexity — support for our baseline + 3-track chromatin design rather than
   adding modalities. Their other future directions (drug structure, pathway/network modules) we already do;
   **time-course omics** we cannot, with static CCLE/ENCODE inputs.
5. **XPert 7.3 remains unresolved.** The public README confirms Uni-Mol drug features and cross-attention
   but not whether **per-atom** tokens are the cross-attention keys/values; that detail is in the
   Zenodo/Figshare bundle. Do not claim atom→gene cross-attention as ours-novel. (Less relevant now that we
   make no interpretability claim for it [4.1c].)

---

## 5. Recommended order of work

| # | Action | Needs accelerator? | Blocks a claim? |
|---|---|---|---|
| 1 | **Ridge linear baseline** (G1) | no | yes — every accuracy claim |
| 2 | **Report all mean\|Y\| strata** (G2a) | no | yes — auditability of every number |
| 3 | Common-DEGs metric (G5) | no | no, but expected by reviewers |
| 4 | Independent stratifier (G2b) | no | strengthens G2 |
| 5 | Unseen dose/time split (G4) | no (eval only) | adds a generalisation axis |
| 6 | Seeds / k-fold (G3) | **yes** | yes — all small deltas |

1–5 are all CPU and none of them wait on the queued TPU run.

---

## Sources
- [Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/) — Ahlmann-Eltze, Huber & Anders, Nature Methods 2025
- [Benchmarking algorithms for generalizable single-cell perturbation response prediction](https://www.nature.com/articles/s41592-025-02980-0) — scPerturBench, Nature Methods 2025 ([code](https://github.com/bm2-lab/scPerturBench))
- [Benchmarking virtual cell models for in-the-wild perturbation response](https://arxiv.org/pdf/2604.27646) — 2026
- [PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis](https://arxiv.org/pdf/2408.10609)
- [Predicting condition-aware drug-induced transcriptional responses via a latent diffusion model](https://pmc.ncbi.nlm.nih.gov/articles/PMC13107963/) — the "chromatin state" future-work quote
- [Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch transformer](https://www.nature.com/articles/s42256-025-01165-w) — XPert, Nature MI ([code](https://github.com/GSanShui/XPert))
- [StateXDiff: Cell State-Contextualized Multimodal Diffusion for Single-Cell Perturbation Prediction](https://arxiv.org/pdf/2605.16104)
- [Uni-Mol: A Universal 3D Molecular Representation Learning Framework](https://openreview.net/pdf?id=IfFZr1gl0b) — conformer-failure limitation; [Uni-Mol2](https://arxiv.org/html/2406.14969v1)
- [Anticancer drug response prediction integrating multi-omics pathway-based difference features](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012905)
- [Deep learning-based enhancement of epigenomics data with AtacWorks](https://www.nature.com/articles/s41467-021-21765-5) — ATAC signal-to-noise/depth dependence, relevant to our reliability weights
