# LINCS Model — Results Log

> **For the writeup, start with [`CLAIMS.md`](CLAIMS.md)** — every claim with its evidence, strength
> (A/B/C/✗), and what would falsify it, including deliberately-kept **tempered and retracted** claims.
> This file is the detailed method/measurement log behind that ledger.

Running, honest record of what we did, implemented, found, and — critically — the **limitations** of each
finding. Rule: **no claim without the evidence next to it; flag every caveat; never overclaim.** Numbers
here are only as strong as the metric that produced them (see the operating rules in `../HANDOFF.md`).

Convention per entry: **What / How / Found / Limitation.**

---

## 1. v3 training — converged (2026-07-25)
- **What:** finish the reliability-weighted v3 model (resume to 12 epochs).
- **How:** Kaggle GPU (single T4), resumed epoch 7→11; `train.py`, OneCycle, weighted-Huber. Metrics in
  `scratchpad/v3final/metrics.json`.
- **Found:** reproducible-cold Pearson flat 0.503→0.502 across epochs 8–11 ⇒ **converged/plateaued**.
  Beats Mean/Meancell/Meandrug baselines (cold MSE 1.57 vs 1.80–1.82).
- **Limitation:** the "12h runtime" was **GPU queue wait**, not compute (script ran 6.56h, ~76min/epoch).
  "Plateau" is within `train.py`'s inline metric; the careful reproducible metric (see §2) reads higher.

## 2. Diagnose — epigenetics + variance partition (2026-07-25, `lincs-diagnose`, free CPU)
- **What:** fair reproducible epi-ablation + truth-vs-model variance partition at the converged ckpt.
- **How:** `diagnose.py`, reproducible sigs (mean|Y|≥1, n=12k), balanced 42 drugs × 6 well-covered cells.
- **Found:** with-epi **R²=0.355 / Pearson 0.548**; **epi ablation ΔR²=+0.089** (epi-ablated 0.266) ⇒ epi
  contributes (stronger than the earlier +0.073). Overall corr(true,pred)=0.77, corr(interaction)=0.49.
  Variance partition **model 59.1/14.4/26.5 vs TRUTH 42.3/9.8/47.9** ⇒ model **under-expresses the
  drug×cell interaction** (26.5% vs 47.9%).
- **Limitation:** these are **in-distribution** (well-covered cells, mostly seen in training), NOT cold-cell
  — so 0.548 is an upper-ish bound, cold is ~0.50. Epi ablation is on reproducible in-dist sigs.

## 3. DTI recall@k — interpretability (2026-07-25, `eval.py`/`lincs-eval-dti`, free CPU)
- **What:** does atom→gene attribution localize to known target genes? (the stated objective)
- **How:** rank 978 genes by 3 signals — `ca_gene_norm` (atom→gene contribution L2), `|Ŷ|`, atom-attention
  — recall@k vs `dti_reference.tsv`, reproducible sigs, split by evidence tier.
- **Found:** enrichment rises **monotonically with reference confidence** (strong evidence it's real, not
  artifact): atom→gene attention `ca_gene_norm` recall@5 = **0.9× (ALL, 19,174 noisy STITCH edges) →
  2.1× (447 ChEMBL curated) → 2.6× (156 both-source gold, 111 drugs)**; `|Ŷ|` recall@10 = 1.5×→1.5×→2.1×.
  Gold median target rank pctile: ca 0.43, |Ŷ| 0.39 (<0.50 random). ⇒ **the top-attended genes are enriched
  ~2.6× for the highest-confidence known targets — interpretability objective SUPPORTED, modestly.**
- **Limitation:** modest absolute effect (recall@50 ~10% — most targets still not in top-50); partly a
  **biological ceiling** (target *gene* ≠ where the response peaks). Small n at gold (~1–2 targets/drug ⇒
  recall@5 coarse; `atom_attn` shows 0.0 sparsity-noise there). In-distribution, not cold. Claim = "attention
  is enriched for curated/gold targets, strengthening with confidence," NOT "attention recovers targets."

## 3b. Interaction under-expression diagnostic (2026-07-25, `analyze.py`/`lincs-analyze`, free CPU)
- **What:** WHY does the model express only 26.5% drug×cell interaction vs 47.9% truth? Cause before fix.
- **How:** balanced drug×cell design within signature-strength bins; measure std(predI)/std(trueI) & corr(I)
  per bin + cross-cell liveness. `analyze.json`.
- **Found:** **noise-driven MSE shrinkage, confirmed.** Interaction expression rises monotonically with
  strength: std-ratio 0.16→0.20→0.31→0.48→**0.49**, corr(I) 0.01→0.03→0.18→0.33→**0.42** across bins
  mean|Y| 0.3→1.6+. Liveness ratio 0.49 (pathway alive, not dead). H_capacity rejected (not flat/low),
  H_dead rejected (ratio≠0).
- **Interpretation (do NOT overclaim):** MOST of the aggregate under-expression is **OPTIMAL** — on ~75%
  inert sigs there's no reproducible cell-specific signal, so shrinking to the drug-average minimises MSE;
  "fixing" it = fitting noise. The **winnable** part is the STRONG stratum (mean|Y|≥1.6, replicate r≈0.75)
  where the model still captures only ~half the magnitude (0.49) and corr 0.42 ⇒ real headroom.
- **Fix direction (a retrain experiment):** a **correlation/rank loss on REPRODUCIBLE sigs** (immune to
  magnitude shrinkage) + keep reliability weighting; instrument strong-stratum interaction-expression
  per-epoch; gate claims on the reproducible stratum + re-measured ceiling, never aggregate. 2-GPU code ready.
- **Limitation:** in-distribution balanced design; strong-stratum bins have modest n (38–40 drugs).

## 4. SOTA comparison + novelty (2026-07-25, literature)
- **What:** are we competitive, and is our contribution novel?
- **Found (SOTA):** latent-diffusion (Bioinformatics 2026) reports unseen-cell **PCC 0.743/R² 0.500**,
  unseen-compound 0.870/0.739. **BUT confirmed: they predict ABSOLUTE expression** (basal given as input),
  per-sample across all 978 genes, no reproducibility filter ⇒ **baseline-inflated** (perturbed≈basal for
  most genes). We predict the **differential** (baseline removed), reproducible-filtered — a strictly harder
  quantity. **The numbers are not comparable.**
- **Found (novelty):** SOTA leaders are VAE/diffusion black boxes (sacrifice interpretability). **Epigenetic
  chromatin conditioning of a perturbation predictor appears novel** (related work goes expression→chromatin
  or predicts baseline expression, not chromatin→drug-response). Atom→gene attribution novelty vs **XPert is
  UNCONFIRMED** (XPert also uses UniMol; README lacks detail).
- **Limitation — DO NOT OVERCLAIM:** (a) we have **no L1000 control** in-data, so a same-metric head-to-head
  is not yet possible — needs L1000 controls or running an open SOTA (PRnet) on our differential metric;
  (b) novelty rests on a handful of searches, not an exhaustive review — XPert's methods (paywalled) must be
  read before claiming atom→gene or pathway-attention as ours-novel.

## 5. v4 — correlation-loss finetune (2026-07-26, `lincs-train-v4`, GPU — RUNNING)
- **What:** un-shrink the drug×cell interaction (the winnable strong-stratum gap from §3b) via noise-shielding.
- **How:** `correlation_loss` (1−Pearson per sig, reliability-weighted) added to Huber with `LAMBDA_CORR=0.5`;
  **FINETUNE** mode loads v3 weights only, fresh low-LR (1e-4) schedule, 4 epochs; a **per-epoch interaction
  probe** logs std(predI)/std(trueI) on a fixed strong-stratum design (watch the fix work in-between). Corr
  loss is magnitude-invariant (unit-tested 37/37). New kernel id (v3 ckpt kept intact as a source).
- **Found: NEGATIVE — the correlation loss did NOT un-shrink the interaction.** Over 4 finetune epochs the
  interaction probe std-ratio stayed 0.47–0.48 (v3 baseline 0.49) and corr stayed ~0.38 (v3 0.42); cold-rep
  pearson 0.502→0.509 (~+0.01, noise). (First attempt ERRORed on a Kaggle-assigned **P100**; fixed via
  `--accelerator NvidiaTeslaT4` + fail-fast probe.)
- **KEY INSIGHT (reframes the whole interaction problem):** the model is **already at MSE-optimal
  dispersion**. For an MSE-optimal predictor std(pred)/std(true) ≈ corr(pred,true); our numbers **std-ratio
  0.47–0.49 ≈ corr 0.42** sit on that line. ⇒ the "under-expression" is NOT wrongly-suppressed magnitude —
  it's the CORRECT dispersion for the model's predictive accuracy. You **cannot inflate interaction magnitude
  without first raising the correlation** (more magnitude on a 42%-correct pattern only adds error). The
  correlation loss (magnitude-invariant) couldn't lift corr in a gentle finetune.
- **Corrected conclusion:** the interaction gap is a **cell-PREDICTION-ACCURACY problem, not a loss/shrinkage
  problem.** Noise-shielding was a reasonable hypothesis but is REJECTED by this test. The real lever is
  better cell modeling (Strategy D: CCLE mutations/CNV features, or the fundamental limit of only 83 training
  cells), and/or a fair SOTA accuracy comparison to know how much headroom actually exists.
- **Limitation:** a gentle 4-epoch finetune at LR 1e-4, λ_corr=0.5 — a from-scratch run or higher λ MIGHT
  move corr, but the std-ratio≈corr relationship says magnitude gains require accuracy gains regardless.

## 6. Drug-feature ablations (2026-07-26, `lincs-drug-ablation`, CPU — RUNNING)
- **What/How:** zero each drug-feature block at inference on reproducible cold-cell sigs; ΔR²/Δpearson vs base.
- **Found (base R²0.298/P0.502):** pillars = **fingerprint** (ΔR²+0.089/ΔP+0.147) and **atoms**
  (ΔR²+0.075/**ΔP+0.163**, the biggest Pearson drop); unimol_cls moderate (+0.035/+0.033); **descriptors
  (+0.003) and ChemBERTa (+0.001) ~dead weight.** ⇒ (a) DROP ChemBERTa (384 dims, redundant); (b) atom
  tokens are load-bearing ⇒ supports the interpretability story (atom→gene substrate isn't decorative).
- **Limitation:** deltas overlap (features partly redundant); reproducible cold-cell only.

## 7. Pathway-flow MoA (2026-07-26, `moa.py`/`lincs-moa`, free CPU)
- **What/How:** does the model lean on the co-pathway/STRING priors that shape gene↔gene flow? Report learned
  λ=softplus(log_lambda) per layer + on-support attention mass (prior vs free heads).
- **Found:** λ **non-zero everywhere** (~0.24–0.94, mostly 0.4–0.7; higher in base/early layers 0.5–0.9,
  decaying to 0.24–0.55 in late perturb) ⇒ priors ARE used, not discarded. BUT on-support attention (last
  perturb layer): PRIOR heads 0.148 (**1.1× random**) vs free 0.141 (1.0×) vs density 0.138 ⇒ the bias only
  **weakly** steers where attention lands (soft λ≈0.5 washed out by content attention).
- **Verdict:** pathway→drug MoA is the **WEAKEST interpretability leg** (epi +0.089 R² > atom→gene DTI 2.6× >
  pathway flow 1.1×). Priors help as a soft inductive bias, not a strong attributable channel. Do not
  overclaim the MoA story.
- **Limitation:** measured only the LAST perturb layer (lowest λ); early/base layers (λ up to 0.9) likely
  steer more — cheap re-run to complete. n=16 sigs (direction reliable, magnitude approximate).

## 8. Cell-gap diagnostic (2026-07-26, LOCAL numpy — free, instant)
- **What/How:** is the interaction/cold-cell gap a REPRESENTATION problem (features help) or a COVERAGE
  problem (only 83 cells)? Correlate cold cell's expression-similarity-to-nearest-training-cell vs per-cell
  cold R² (17 cold cells).
- **Found: INCONCLUSIVE, leans "features might help."** corr(similarity, R²)=+0.36 (rank +0.30, NOT sig at
  n=17); high-sim cells mean R² 0.12 vs low-sim 0.09. Outliers break a pure-coverage story: NPC sim=0.99 but
  R²=0.01 (worst); HA1E sim=0.30 but R²=0.17. ⇒ per-cell perf is MULTIFACTORIAL, NOT a clean coverage wall ⇒
  added cell features (tissue/lineage/mutations/CNV) have PLAUSIBLE but UNCERTAIN headroom.
- **Data availability:** tissue/lineage present (`baseline/CCLE/Model.csv`); mutations/CNV would need a DepMap
  download. **Limitation:** n=17 too small to conclude; expression-similarity may not be the right metric.

## 9. Strategy D — cell lineage feature (2026-07-26, premise measured LOCALLY, retrain pending)
- **Premise test (before spending GPU):** does lineage add info about drug response BEYOND `X_base`?
  1,412 cell pairs (180 same-lineage), reproducible sigs: response-similarity **same-lineage +0.284 vs
  diff +0.246**; **partial corr(lineage, response | X_base) = +0.092, permutation p<0.0005** ⇒ small but
  REAL and significant (same shape as the epigenetics result).
- **Implemented:** `cell_lineage.npy` [83,16] one-hot (col0=UNKNOWN) → FiLM cell-context conditioning
  (`DoseTimeFiLM(d_extra=16)`), zero-init so it starts as a no-op and must EARN its use.
- **Fairness:** cell features are legitimate for cold-cell eval — lineage/expression/chromatin are observable
  WITHOUT any perturbation experiment. (It would only be cheating to leak the cell's perturbed response.)
- **Limitation — expect a SMALL effect:** only 57/83 cells have a lineage; the 26 UNKNOWN are the
  non-cancer/primary/stem lines (DepMap catalogs cancer models only). **7 of 17 cold cells are UNKNOWN —
  and they include the BEST performers** (HUVEC 0.225, U266 0.214, HA1E 0.169 has an unseen lineage), so the
  feature supplies nothing exactly where the model already does well.

## 10. ChemBERTa dropped (2026-07-26)
- Per §6 ablation (ΔR²=+0.001). `DataConfig.use_chemberta=False`; `d_global` 2964→2580 (−384 dims).
  Requires a from-scratch retrain (input dim change ⇒ not checkpoint-compatible).

## 11. Scaffold split for the unseen-COMPOUND benchmark (2026-07-26, LOCAL, `build_scaffold_split.py`)
- **What/How:** Bemis-Murcko scaffold split (5 balanced folds, 21,220 drugs / **6,035 scaffolds**, 4,244 per
  fold) so a whole chemical series lands on one side + an ECFP4/Tanimoto **leakage audit**.
- **KEY HONEST FINDING — scaffold splitting alone does NOT make compounds "unseen":** max Tanimoto from each
  test drug to its nearest TRAIN drug is **median 0.655**, p90 0.836, **39.2% ≥0.70**, **8.0% ≥0.85**,
  max=1.000 (some near-duplicates share a fingerprint across different Murcko scaffolds). ⇒ any
  "unseen-compound" number (ours OR a paper's) is inflated by analogue leakage unless audited. A RANDOM drug
  split (what most papers use) would be far worse.
- **Consequence:** report our unseen-compound number **with this audit**, and consider a stricter
  Tanimoto-capped variant (drop test drugs with a train neighbour above a threshold) as the honest headline.
- **Limitation:** max=1.000 suggests true duplicate structures under different `pert_id`s — worth a check.

## 12. v5 — combined experiment (2026-07-26, `lincs-train-v5`, T4x2 — RUNNING)
One from-scratch run testing everything at once (quota-efficient; each part stays independently ablatable
at inference so we can still attribute post-hoc):
- **Lineage FiLM cell-context** (§9), **cell-conditional pathway conductance** (below), **ChemBERTa dropped**
  (§10), **combined cold-CELL x cold-COMPOUND split** (§11) → yields THREE generalization numbers from one
  run: unseen-cell (47,002 sigs) / **unseen-compound (58,796)** / unseen-both (15,083); train 179,772.
  Leakage verified locally = 0 (no test cell and no test-scaffold drug appears in train).
- **Pathway conductance (the user's idea, refined by measurement):** `c_{cell,g} = 1+tanh(MLP([E_g;x_g;avail]))`
  scales how much gene g *listens to its pathway neighbours*. Rationale: the prior bias was STATIC (same λ
  for every cell) so the model could not say "this pathway is open here, closed there". **Per-EDGE cell
  conditioning is infeasible** ([B,8,978,978] ≈ 10GB/batch) — gating conductance per (cell,gene) is the
  memory-safe equivalent. Zero-init ⇒ exact no-op until earned; `pathway_cond` exposed in aux for maps.
  **Watch:** module activity correlates 0.85 with a gene's own baseline, so it can only add value via the
  EPIGENETIC input, not by re-encoding X_base.
- **Config:** from scratch (d_global 2964→2580 ⇒ v3 ckpt NOT loadable, so no `kernel_sources`), 10 epochs,
  BATCH 64 (DataParallel 32+32), LR 4e-4, budget 10.5h. λ_corr=0 (v4 showed the corr term didn't help).
- **Validated locally before launch: 45/45 unit checks + 3 padding/masking checks.**
- **RESULT (completed, 10 epochs, reproducible sigs):**
  | metric | v5 | v3 |
  |---|---|---|
  | unseen-CELL pearson | **0.440** (R² 0.273) | 0.502 (R² 0.298) |
  | **unseen-COMPOUND pearson** | **0.471** (R² 0.188) | — (new) |
  | unseen-BOTH pearson | **0.451** (R² 0.173) | — (new) |
  | interaction probe std-ratio / corr | 0.356 / 0.322 | ~0.47 / 0.42 |
  Cold baselines MSE: Mean/Meancell 1.747, Meandrug 1.733 (model 1.481 — still beats them).
- **v5 REGRESSED on cold-cell vs v3 — but the comparison is CONFOUNDED:** v5 trained on **179,712 sigs vs
  v3's 235,628 (−24%)** because the scaffold drug fold is now also held out, and ran 10 epochs from scratch
  vs v3's 12. **Do NOT conclude the new components hurt (or helped) from this number** — attribution needs
  the inference ablation (`ablate_v5.py`, run separately).
- **NEW + notable: unseen-COMPOUND (0.471) > unseen-CELL (0.440)** ⇒ the model generalizes BETTER to new
  drugs than to new cell lines, consistent with every earlier finding that cell-specificity is the hard part.
- The inline `EPI ABLATION delta_r2=+0.0009` is again the **all-signature artifact** — ignore it; the fair
  reproducible test is in `ablate_v5.py` / `diagnose.py`.

## 13. Padding & attention-masking safety (2026-07-26, LOCAL) — prerequisite for TPU/XLA
- **What/How:** XLA needs STATIC shapes, so `collate(fixed_pad=True)` always pads atoms to `max_atoms`.
  Risk: attention leaking onto padded slots. Three tests: Yhat invariance to padding amount; attention mass
  on padded keys; fixed vs dynamic collate equality.
- **Found: all three exactly 0.00e+00.** Padded slots are False in `atom_mask` → `key_mask=~valid` → −inf
  pre-softmax; the global drug token is always valid so no query row is fully masked (no NaN).
- Also hardened `collate` to force float32 (python-float scalars silently became float64 → Linear dtype crash).

## 14. Tooling correction (2026-07-26): **local torch EXISTS** (2.11.0+cpu, system Python 3.14)
The long-standing "no local torch, test on Kaggle" belief was STALE and cost real time (Kaggle's
5-concurrent-CPU-session cap blocked pushes for ~1h). Full suite runs locally in ~2-3 min. Kaggle is now
needed only for GPU/TPU and the big bundle. (`drug/.venv-drug` has rdkit but no torch; system Python vice versa.)

## 15. TPU feasibility probe (2026-07-26, `lincs-tpu-probe`, free TPU quota)
- **What/How:** does `torch_xla` work on Kaggle, how many cores, and how fast is OUR bottleneck shape
  (978x978 biased attention, B=16,H=8)? Benchmarked on TPU; T4 number derived from v3's measured 1.24 s/step.
- **Found:** torch_xla **2.8.0**, **8 cores**, TPU **39.4 ms/iter on ONE core**. Derived T4 ≈ 30 ms for the
  same (explicit-logits) code. ⇒ **per core the TPU is ~0.76x — SLOWER than a T4.** The advantage is purely
  **core count**: 8 cores vs 2 T4s ≈ **3.1x aggregate, ONLY IF all 8 cores scale**.
- **Correction to an earlier claim:** a previous "~3x plausible" estimate cited per-chip FLOPS — that
  reasoning was WRONG. The number is roughly right but for a different reason (core count, not speed).
  **A single-core torch_xla port would be a DOWNGRADE.**
- **Recommendation: do NOT port now.** Requires full 8-core data parallelism (xmp.spawn/SPMD + sharding +
  ckpt handling), TPU queue alone was >1h (eats much of a 3x gain at our cadence of a few retrains), and
  remote XLA debugging is slow. Training speed is not the bottleneck; the SOTA comparison is.
- **Limitation:** the T4 figure is DERIVED, not measured (GPU session cap=2 was full with v5 running).
  Measure it (`lincs-gpu-bench`, staged) before treating 3.1x as confirmed.
- **Banked either way:** XLA-safe fixed padding (§13) is done and proven leak-free — the hard correctness
  prerequisite for any future TPU port.

## 16. v5 component attribution (2026-07-26, `ablate_v5.py`, LOCAL CPU, n=1200 reproducible per split)
Each component toggled at INFERENCE on identical signatures (positive Δ = component contributes):

| ablated | unseen-CELL ΔR² | Δpearson | unseen-COMPOUND ΔR² | Δpearson |
|---|---|---|---|---|
| **pathway conductance** | **+0.1030** | +0.0279 | **+0.0726** | +0.0257 |
| epigenetics | −0.0042 | +0.0018 | **+0.0345** | **+0.0385** |
| lineage | +0.0014 | +0.0105 | −0.0057 | +0.0170 |
(full model: unseen-cell R² 0.2805/p 0.4506; unseen-compound R² 0.1946/p 0.4913)

- **Pathway conductance is by far the largest dependence** — 0.103 R² on unseen cells, ~10x lineage.
  **CRITICAL CAVEAT: an inference ablation measures DEPENDENCE, not value-add** — removing any load-bearing
  trained component hurts. This does NOT prove it beats v3. **The decisive test is a MATCHED run (same
  179,712-sig data, same 10 epochs, `pathway_conductance=False`).** What it DOES show: the mechanism is
  genuinely used, not decorative — and the ablation discriminates (epi came out ~0 on unseen-cell rather
  than automatically positive).
- **Epigenetics splits by axis (NEW):** clearly contributes on unseen-COMPOUND (+0.0345 R²/+0.0385 p) but
  is **neutral/slightly negative on unseen-CELL** (−0.0042). Differs from v3's +0.089 in-distribution ⇒
  chromatin appears to aid generalization to new CHEMISTRY more than to new CELLS. Worth understanding,
  not glossing.
- **Lineage is marginal** (+0.001…+0.017), as predicted from 7/17 cold cells having UNKNOWN lineage; it
  nudges pattern (pearson) not magnitude. Keep (cheap) but do not feature it in any claim.
- **Limitation:** n=1200/split, single fold, and all deltas are dependence-not-improvement (see above).

## 17. Dual-metric / metric-convention analysis (2026-07-26, `dual_metric.py`, LOCAL CPU)
- **What/How:** report the SAME v5 predictions under BOTH conventions — ours (differential: corr(Ŷ,Y)) and
  the published one (absolute: corr(basal+Ŷ, basal+Y)) — and sweep the basal anchor scale α to expose how
  much of a published number is metric convention.
- **Measured (n=1200 reproducible):** unseen-cell PCC 0.451→0.490, **R² 0.281→0.497**; unseen-compound
  PCC 0.491→0.497, **R² 0.195→0.388**. ⇒ **R² nearly DOUBLES on identical predictions.**
- **A prediction of mine was WRONG:** I expected a large PCC inflation; measured only +0.040/+0.006, because
  our normalized `X_base` anchor has basal:delta variance ratio only **0.4–0.6** — far below a real
  raw-expression setup. Our anchor is the wrong scale ⇒ that single number is inconclusive as a protocol proxy.
- **Fixed by sweeping α (the rigorous version), unseen-compound n=900:**
  | basal:delta var ratio | 0 (ours) | 0.4 | 1.6 | 3.6 | 10 | 40 | 1000 |
  |---|---|---|---|---|---|---|---|
  | PCC | **0.485** | 0.485 | 0.682 | **0.814** | 0.918 | 0.978 | 0.999 |
- **KEY RESULT: published SOTA PCCs (0.743 unseen-cell / 0.870 unseen-compound) fall INSIDE this curve**
  (ratio ≈2–4), where our own unchanged predictions score **0.68–0.81**. ⇒ **a published 0.87 is NOT evidence
  of better drug-effect prediction than our 0.471** — the convention alone spans ~0.49→1.0.
- **Limitation (do not overclaim):** this does NOT show we match SOTA. Their exact variance ratio is unknown
  and our anchor is CCLE (no L1000 controls exist in our data). Only running their code on our protocol
  settles it. What it DOES establish is **non-comparability**, rigorously.

---

## 18. Linear + mean baselines, protocol-matched (2026-07-30, `baseline_linear.py`, LOCAL CPU)

**Why**: the methodology audit found we had never fitted a **linear** baseline (M.1) — only Mean/Meancell/
Meandrug, and those only ever as MSE on *all* cold-cell signatures, which is not comparable to any number we
quote. Ahlmann-Eltze/Huber/Anders (Nature Methods 2025) showed no deep model beat a ridge-style linear model.

**Protocol**: closed-form ridge, λ chosen on the same `val` split the neural models use (λ=1e4 for both
variants, interior to the 1e2…1e6 grid). Scored with the **identical** protocol as v5/v6 — same fold-0
splits, same reproducible stratum, same metric functions, same v5 subsampling caps. Mean baselines rescored
with those same metrics. All numbers below use the **pre-fix dose parse**, matching how v5 was trained, so
the comparison is apples-to-apples (see §19).

| unseen **CELL** (n=7296) | pearson | R² | MSE | cDEG@100 |
|---|---|---|---|---|
| v5 neural | 0.440 | +0.273 | 4.348 | — |
| ridge `full_linear` (D=3576) | **0.4470** | +0.2755 | 4.334 | 0.250 |
| ridge `ecfp_cell` (D=2086) | 0.4235 | +0.2623 | — | 0.240 |
| **Meandrug** (predict the drug's training-set mean) | **0.4475** | **+0.2846** | **4.279** | 0.260 |
| Mean / Meancell | 0.3004 | +0.0412 | 5.735 | 0.170 |

| unseen **COMPOUND** (n=6000) | pearson | R² | MSE |
|---|---|---|---|
| **v5 neural** | **0.471** | **+0.188** | **5.126** |
| ridge `full_linear` | 0.3824 | +0.1072 | 5.636 |
| best mean (Meancell) | 0.3649 | +0.0671 | 5.889 |

| unseen **BOTH** (n=2998) | pearson | R² | MSE |
|---|---|---|---|
| **v5 neural** | **0.451** | **+0.173** | **6.035** |
| ridge `full_linear` | 0.3315 | +0.0817 | 6.699 |
| best mean (Mean) | 0.3162 | +0.0391 | 7.010 |

- 🔴 **KEY NEGATIVE RESULT: on unseen CELL, v5 is beaten by predicting the drug's average response** —
  Meandrug wins on **all three** metrics (pearson 0.4475 vs 0.440, R² +0.2846 vs +0.273, MSE 4.279 vs
  4.348), and ridge with the same global inputs also edges it on pearson (0.4470). Not a metric artefact.
  ⇒ **the model adds no cell-specificity.** This reproduces the Nature Methods finding inside our own project.
- ✅ **On unseen COMPOUND and unseen BOTH the neural model is far ahead** — +0.089 and +0.120 pearson over the
  best linear, +0.106 and +0.135 over the best mean. Chemical generalisation is real and large.
- ⇒ **The model's value is chemical generalisation, not cellular.** This sharpens [1.5] (unseen-compound >
  unseen-cell) from "cell-specificity is harder" to "cell-specificity is **not being achieved at all**", and
  it is exactly consistent with chromatin's ≈0 benefit on unseen cells [2.3] and with the model sitting at
  MSE-optimal dispersion [6.8/6.10] — i.e. hedging toward the drug mean is the optimal thing to do here.
- **`full_linear` is the fair competitor**: all of `u_feats` (UniMol CLS + descriptors + ECFP4) + `X_base` +
  lineage + dose + time — the same global information the neural model gets, minus per-atom tokens and
  chromatin. It costs seconds to fit.

**All-strata report** (M.2 — equal n per bin, `full_linear`, so the ≥1 headline is auditable):

| mean\|Y\| bin | unseen cell | unseen compound | unseen both |
|---|---|---|---|
| 0.0–0.5 | 0.112 (R² −0.140) | 0.086 (−0.251) | 0.065 (−0.279) |
| 0.5–1.0 | 0.171 (+0.008) | 0.137 (−0.046) | 0.100 (−0.087) |
| 1.0–2.0 | 0.398 (+0.153) | 0.327 (+0.113) | 0.276 (+0.087) |
| 2.0+ | 0.682 (+0.355) | 0.450 (+0.103) | 0.371 (+0.079) |

- **R² is NEGATIVE in both weak strata** — below mean|Y| = 1 the global mean beats the model. This is signal
  dilution quantified, and it is why rule #1 exists; but reporting only the ≥1 bin hid it. Report all bins.

## 19. Dose unit parsing bug (2026-07-30, LOCAL, found while designing an unseen-dose/time split)

The raw `dose` field has **110 distinct strings with mixed units** ('10 µM', '500 nM', '100 nM'). `_num()`
took the leading number and discarded the unit, so **500 nM was parsed as 500 µM**.

- **13,910 rows = 4.49 %** affected (nM only; no mM present). Error is exactly **1000×**.
- Because it is multiplicative on a log axis it **inverted the ordering**: median z(log-dose) for nM rows was
  **+1.710**, *above* the µM rows (+0.501). After the fix: **−1.769**, correctly below (+0.669).
  Mis-scaling put the **lowest** doses at the **top** of the dose axis — worse than dropping them.
- Fixed by `data.py::_dose_um` (unit-aware, normalises to µM). **Every checkpoint before 2026-07-30,
  including v5, was trained with the bug**, so `V6DataConfig.legacy_dose_parsing` restores it: evaluating an
  old checkpoint with the corrected parse would feed those rows a log-dose the model never saw and quietly
  understate it. Match the flag to the checkpoint.
- Impact bound: dose is a *conditioning* feature, not a target, and 95.5 % of rows were always correct. It
  does not invalidate the pathway-bottleneck measurement, but it does invalidate any dose-response analysis
  on a pre-fix checkpoint, and dose demonstrably moves the v6 pathway readout (Δ1.59, claim 4.12).

---

## 20. v6 FULL EVALUATION (2026-08-15, `eval_v6.py`, T4x2-trained ckpt, LOCAL CPU eval)

Trained on T4 x2, 10 epochs, 2.58 h, clean inputs (`input check OK`, compound holdout present, splits
identical to local). Protocol-matched to v5 throughout; split-identity check passed on all three
(7296 / ≥6000 / 2998).

| split | v6 pearson | v6 R² | v5 pearson | Δ | best baseline |
|---|---|---|---|---|---|
| unseen CELL | 0.4466 | +0.2866 | 0.440 | **+0.0066** | **Meandrug 0.4475** |
| unseen COMPOUND | 0.4686 | +0.1775 | 0.471 | −0.0024 | ridge 0.3824 |
| unseen BOTH | 0.4663 | +0.1452 | 0.451 | **+0.0153** | ridge 0.3315 |

**v6 ≈ v5.** The complete rebuild — hard Reactome mask replacing the soft prior, late fusion replacing
early — moved the headline by −0.002 to +0.015 depending on the split, all **single fold, single seed**, so
none of it is separable from run-to-run variance (M.3). On unseen-both v6 gains the most pearson (+0.0153)
while *losing* R² (+0.1452 vs +0.173) — better pattern, worse magnitude.

**Ablate-to-mean, Δpearson / |dY|max, all three splits** (|dY|max ≫ 0 everywhere ⇒ every null is a TRUE null):

| component | unseen cell | unseen compound | unseen both |
|---|---|---|---|
| **drug global** (UniMol+desc+ECFP4) | **+0.2502** / 10.7 | **+0.2971** / 10.3 | **+0.2423** / 9.8 |
| baseline expression | +0.0257 / 10.7 | +0.0445 / 8.2 | +0.0365 / 9.4 |
| atom tokens | +0.0022 / 8.5 | +0.0240 / 4.2 | +0.0141 / 4.5 |
| lineage | +0.0024 / 8.4 | +0.0215 / 7.4 | +0.0080 / 5.5 |
| chromatin | −0.0001 / 2.1 | **+0.0061** / 5.9 | −0.0001 / 1.1 |
| **pathway layer** | +0.0003 / 0.7 | −0.0002 / 0.5 | −0.0002 / 0.5 |
| pathway chromatin gate | +0.0001 / 0.2 | +0.0002 / 0.4 | −0.0001 / 0.2 |

- 🔴 **The pathway layer is null on ALL THREE splits** (+0.0003 / −0.0002 / −0.0002). Not a placement
  artefact of one split. The readout is alive and healthy — **0/360 dead nodes**, activation spread evenly
  (top-10 share 0.04) — it simply carries nothing the prediction uses. v6's central innovation is
  accuracy-neutral, exactly as ARCHITECTURE.md §6 warned.
- 🔴 **The drug features ARE the model**: +0.25 to +0.30 of a 0.45–0.47 total. Everything else combined is
  worth ~0.03–0.08.
- ✅ **Chromatin's cell-familiarity pattern reproduces in v6**: +0.0061 on unseen COMPOUND (cells seen) vs
  ≈0 on both unseen-CELL splits. Same shape as [2.3] measured on v5 (+0.089 in-dist → +0.035 unseen
  compound → ≈0 unseen cell), now with the late-fusion architecture that was supposed to fix it.
- **Lineage (+0.0215 on unseen compound) beats chromatin (+0.0061)** — a 16-dim one-hot outperforms three
  genome-wide chromatin tracks. [5.4] called lineage "marginal"; it is marginal but it is *more* than what
  the entire epigenetics branch delivers.
- Late-fusion chromatin gate ended at **sigmoid = 0.4798**, i.e. slightly *below* its 0.5 init: chromatin
  did not earn its way in.

---

## 21. v7 seed 0 — WORSE than v6 on all three splits, and diagnosable (2026-08-15)

v7 bundled: supervised pathway + chromatin aux heads with uncertainty weighting, STRING PPI message
passing, 765 finer Reactome nodes, stochastic depth, EMA, WSD schedule, RMSNorm, SwiGLU, QK-norm.

| split | v7 s0 (raw) | v7 s0 (EMA) | v6 | v7 − v6 |
|---|---|---|---|---|
| unseen CELL | 0.4322 | 0.4317 | 0.4466 | **−0.0144** |
| unseen COMPOUND | 0.4448 | 0.4451 | 0.4686 | **−0.0238** |
| unseen BOTH | 0.4237 | 0.4238 | 0.4663 | **−0.0426** |

**ALL THREE SEEDS — and this is the most consequential table in the project** (see M.10):

| split | seed 0 | seed 1 | seed 2 | mean | **range** | **sd** | v6 | v5 |
|---|---|---|---|---|---|---|---|---|
| unseen CELL | 0.4322 | 0.4374 | 0.4425 | 0.4374 | 0.0103 | 0.0052 | 0.4466 | 0.4400 |
| unseen COMPOUND | 0.4448 | 0.4624 | 0.4746 | 0.4606 | **0.0298** | 0.0150 | 0.4686 | 0.4710 |
| unseen BOTH | 0.4237 | 0.4532 | 0.4694 | 0.4488 | **0.0457** | 0.0232 | 0.4663 | 0.4510 |

- 🔴 **v5, v6 and v7 are statistically indistinguishable on two of the three splits.** v6 *and* v5 both fall
  **inside** the v7 three-seed range on unseen-compound and unseen-both. Only on unseen-cell is v6 (0.4466)
  outside the v7 range [0.4322, 0.4425], i.e. v7 is consistently ~0.009 worse there.
- 🔴 **Seed sd reaches 0.0232** ⇒ a 2-sd interval of ±0.046. **Every architecture difference this project
  has reported is smaller than that.** v6−v5 was +0.0066 / −0.0024 / +0.0153.
- ⇒ **With one seed per configuration, none of v5 → v6 → v7 could ever have been distinguished.** Three
  full rebuilds were compared on differences the measurement cannot resolve.
- The variance is **split-dependent**: unseen-cell is fairly stable (sd 0.005) while unseen-both is wild
  (sd 0.023) — smaller n (2998) and the hardest task compound each other.
- Correction to two earlier statements of mine: "seed spread ≈0.004" (from the n=3000 training proxy,
  understated by ~5×) and "v7 is clearly worse than v6, outside noise" (not supported — seed 2 alone lands
  at 0.4694 on unseen-both vs v6's 0.4663).

**What is NOT affected:** the ablation results, which are within-run comparisons on identical signatures and
therefore carry no seed variance at all. The pathway-layer null (+0.0003 / −0.0002 / −0.0002 with
\|dY\|max ≫ 0), drug-features-dominate (+0.24…+0.30), and Meandrug tying v5 on unseen cells all stand.
**Within-run ablation is trustworthy here; between-run comparison is not.**
- **EMA contributed nothing**: −0.0005 / +0.0003 / +0.0001. The one item in the modern recipe chosen because
  its documented benefit (robustness to noisy labels) matched our regime did not show up at all.
- 🔴 **PRIME SUSPECT — the uncertainty weighting inverted the objective.** Final learned weights:
  **main 1.59, pathway-aux 10.75, chromatin-aux 3.78** ⇒ the main task received about
  **1.59 / 16.1 ≈ 10 %** of the weighted loss. Kendall-style weighting sets weight by *inverse task noise*,
  so an auxiliary task that is merely EASY collects a huge weight. Predicting per-pathway mean\|Y\| from
  pathway activations is easy; that is not the same as being useful. **The model spent ~90 % of its
  gradient on auxiliary tasks.**
- **Method lesson (generalise this):** uncertainty weighting is for tasks you care about *equally*. For
  auxiliary tasks it must be capped, scheduled, or replaced by a small fixed λ — otherwise "easy" is
  rewarded as "important".
- ⚠️ **Attribution is impossible from this run**: ~8 changes at once, which is exactly the mistake v5 made
  (4 bundled changes, cause unattributable [1.7]). The fix is one-factor-at-a-time, and the cheapest
  decisive test is `--no_aux` (everything else identical).
- Cost note: epoch time rose **915 s → 1382 s** despite RAM-caching the inputs. The added compute (765-node
  pathway layer, a dense 978×978 PPI einsum ≈ 12 GFLOP/call, aux heads) more than offset the I/O saving —
  the earlier claim that dataloading was the bottleneck was wrong and was not costed before being asserted.

## 22. `--no_aux` one-factor ablation — the auxiliary losses were the problem (2026-08-15)

Identical to v7 seed 0 in every respect except the auxiliary heads are off. Everything else — STRING PPI,
765 Reactome nodes, stochastic depth, EMA, WSD, RMSNorm/SwiGLU/QK-norm — unchanged.

| config | unseen cell | unseen compound | unseen both |
|---|---|---|---|
| v5 (1 seed) | 0.4400 / +0.2730 | 0.4710 / +0.1880 | 0.4510 / +0.1730 |
| v6 (1 seed) | 0.4466 / +0.2866 | 0.4686 / +0.1775 | 0.4663 / +0.1452 |
| v7 + aux, seed 0 | 0.4322 / +0.2656 | 0.4448 / +0.1851 | 0.4237 / +0.1705 |
| v7 + aux, seed 1 | 0.4374 / +0.2541 | 0.4624 / +0.2438 | 0.4532 / +0.2411 |
| v7 + aux, seed 2 | 0.4425 / +0.2612 | 0.4746 / +0.2628 | 0.4694 / +0.2226 |
| **v7 NO-AUX (1 seed)** | **0.4452 / +0.2765** | **0.4900 / +0.3135** | **0.4772 / +0.2878** |

- ✅ **no-aux is above ALL THREE aux seeds on ALL THREE splits, in both pearson and R².** Not inside the
  range — above its maximum, 6/6 times.
- ✅ **It is the best model measured to date**: best pearson on unseen-compound (0.4900) and unseen-both
  (0.4772) of anything tested, and the R² gains are large — **+0.3135 vs v6's +0.1775** on unseen-compound,
  **+0.2878 vs +0.1452** on unseen-both, i.e. roughly double the variance explained.
- ⇒ **Confirms the diagnosis in §21**: uncertainty weighting handed ~90 % of the gradient to the auxiliary
  tasks (learned weights main 1.59 / pathway 10.75 / chromatin 3.78, reproducible to 3 decimal places across
  all three seeds), and that actively degraded the model. **Kendall-style weighting is for tasks you care
  about equally; an auxiliary task that is merely EASY collects a huge weight.**
### THREE SEEDS OF no-aux — the first improvement in this project that survives its own variance test

| split | no-aux mean | no-aux range (3 seeds) | aux mean | v6 | v5 | best baseline |
|---|---|---|---|---|---|---|
| unseen CELL | **0.4549** | [0.4452, 0.4612] | 0.4374 | 0.4466 | 0.4400 | Meandrug 0.4475 |
| unseen COMPOUND | **0.4985** | [0.4900, 0.5033] | 0.4606 | 0.4686 | 0.4710 | ridge 0.3824 |
| unseen BOTH | **0.4825** | [0.4772, 0.4863] | 0.4488 | 0.4663 | 0.4510 | ridge 0.3315 |

- ✅ **Complete separation from the aux runs on all three splits** — no-aux *minimum* exceeds aux *maximum*
  every time (0.4452>0.4425, 0.4900>0.4746, 0.4772>0.4694). Two sets of three seeds, zero overlap.
- ✅ **On unseen-compound and unseen-both, ALL THREE no-aux seeds beat v6 AND v5** — the entire range sits
  above both. Given M.10, this is the **first result here that cannot be explained by seed variance**.
  Mean gains: **+0.0299 / +0.0162 vs v6**, **+0.0275 / +0.0315 vs v5**.
- ✅ **Two of three seeds beat `Meandrug` on unseen cell** (0.4582, 0.4612 vs 0.4475) — the first time
  anything in this project has cleared the drug-mean baseline on the cold-cell axis [1.8].
- **unseen-compound crosses 0.50** on two seeds (0.5021, 0.5033).
- ⚠️ **Attribution is still open.** no-aux contains STRING PPI + 765 nodes + stochastic depth + EMA + WSD +
  RMSNorm/SwiGLU/QK-norm, all at once. We know the *bundle minus aux* helps; we do not know which part. That
  is the next one-factor sweep, and it must be run with 3 seeds per arm.
- **What made the difference was removing something, not adding it.** The single largest measured
  improvement in the project came from deleting a component I had added two turns earlier.

---

## 23. XPert head-to-head, part 1: their own released predictions (2026-08-18)

Downloaded XPert's Zenodo release (code, trained weights, and the released prediction arrays for their
HDAC-inhibitor figure: `y_true` / `y_pred` / `ctl_true`, 3,439 signatures x 978 genes).

**Their code already computes BOTH conventions** (`get_evaluation_metrics.get_metrics_new`), which is better
practice than we assumed:
```
metrics['Pearson']     = pearson(y, f)              # ABSOLUTE
metrics['Pearson_deg'] = pearson(y - ctl, f - ctl)  # DELTA  <- identical in form to our metric
```

### Finding 1 — the absolute convention IS inflated, and by roughly what we predicted

| on XPert's own released predictions | Pearson (mean) |
|---|---|
| XPert, ABSOLUTE (`y` vs `f`) | **0.9804** |
| XPert, DELTA (`y-ctl` vs `f-ctl`) | **0.8440** |
| **"predict no change" — just copy the control** | **0.9200** |

- Convention inflation = **+0.136**.
- The absolute number beats *doing nothing at all* by only **+0.060**. A reported ~0.98 absolute correlation
  is overwhelmingly the supplied baseline being copied. This confirms the mechanism quantitatively, on the
  authors' own data, without reinterpreting their metric.

### Finding 2 — but their DELTA number is still far above ours, and that does not flatter us

Comparing gains over the trivial baseline **inside each model's own frame** (the only currently fair contrast):

| frame | model | best trivial baseline | **gain** |
|---|---|---|---|
| XPert (Level-3 log-expression delta) | 0.8440 | 0.3682 (same delta for every signature) | **+0.4758** |
| ours (Level-5 MODZ z-score, reproducible) | 0.4985 | 0.3824 (ridge) | **+0.1161** |

XPert's gain over trivial is **~4x ours**. Three confounds are known and **all of them favour XPert**:
1. **HDAC inhibitors only** — their easiest subset. We measured epi-drugs at +0.20 easier than average in
   our own data [2.6], so this is not a headline benchmark.
2. **Split unknown** — this figure may be in-distribution, while our 0.4985 is unseen-compound.
3. **Different target** — Level-3 log-expression differences vs Level-5 MODZ z-scores.

⇒ **Do not conclude "XPert is 4x better" from this.** Do not conclude we are fine either.

### The hypothesis this raises, and why Level 3 is now justified

Confound 3 is the one that matters strategically. Our Level-5 target has a **measured noise ceiling of
0.71-0.79** (replicate r 0.509-0.619 [6.1]) and we sit at 0.4985 = ~63-70 % of it. If Level-3 deltas have a
**higher replicate reliability**, then every model trained on Level 3 is solving an intrinsically easier
problem, and no architecture change on our side can close that gap.

**That is a measurable question, and it decides whether to keep tuning models or change the data.** It needs
Level 3 + `inst_info` (to pair treated wells with same-plate DMSO controls). This is the first
evidence-based reason in this project to acquire it — not to match anyone's convention, but to measure
whether the target itself is the ceiling.


---

## 24. XPert head-to-head, part 2: THE TWO TARGETS ARE NOT THE SAME QUANTITY (2026-08-18)

`model/compare_targets_xpert.py`. XPert's released h5ad is **replicate-collapsed Level-3 log-expression with
a plate-matched control** (`X`, `obsm['X_ctl']`, 336,852 x 978, `n_replicates` 1-6+). Ours is **Level-5 MODZ
z-scores**. **153,478 (cell, pert, dose, time) conditions exist in BOTH datasets**, so the two targets can be
compared directly — no model involved. Aligned on the 959 shared gene symbols.

### The two targets agree only moderately

| our mean\|Y\| stratum | n | agreement r |
|---|---|---|
| 0.34–0.57 | 1000 | 0.4381 |
| 0.57–0.68 | 1000 | 0.4695 |
| 0.68–0.90 | 1000 | 0.4940 |
| 0.90–4.78 | 1000 | 0.6159 |
| **reproducible (≥1)** | 802 | **0.6419** |

Overall mean **0.5044**. For the *same drug, cell, dose and time*, the Level-3 delta and the Level-5 z-score
correlate at only ~0.5–0.64. **They are different quantities**, and no protocol-matching makes a score on one
comparable to a score on the other.

### …and the disagreement is almost entirely OUR measurement noise

Our replicate reliability ρ was measured per strength bin [6.1]. If their target were relatively clean and
ours is signal+noise, then `corr(theirs, ours) ≤ sqrt(ρ_ours)`:

| our mean\|Y\| | n | observed r | ρ | ceiling √ρ | % of ceiling |
|---|---|---|---|---|---|
| 0.5–0.8 | 2289 | 0.4690 | 0.086 | 0.2936 | 160 % |
| 0.8–1.0 | 500 | 0.5116 | 0.195 | 0.4419 | 116 % |
| 1.0–1.5 | 438 | 0.5737 | 0.398 | 0.6306 | **91 %** |
| 1.5–2.5 | 218 | 0.6775 | 0.668 | 0.8175 | **83 %** |
| 2.5+ | 146 | 0.7934 | 0.751 | 0.8666 | **92 %** |

- ✅ **In every stratum we actually evaluate on, the observed agreement is 83–92 % of the maximum the noise
  in our own target permits.** ⇒ **the two targets measure essentially the SAME biology, and ours is simply
  the noisier measurement of it.**
- ⚠️ In the weak strata the agreement *exceeds* the naive ceiling (160 %, 116 %). Either the ρ estimates from
  [6.1] are conservative there, or the two targets share structure that is not biological signal. Does not
  affect the strong strata, which is where every number we report lives.

### What this means — and it is the answer to "how do we compare fairly?"

1. **Cross-paper score comparison on LINCS is invalid unless the DATA LEVEL matches**, not merely the
   convention. Level-5 z-scoring divides by plate-population variability and demonstrably destroys signal:
   a perfect model on our target would score ~0.71–0.87 depending on stratum, never 1.0.
2. **Part of the XPert gap [§23] is a cleaner target, not a better model.** How much, we cannot yet say.
3. ⇒ **Acquiring Level 3 is now justified on evidence.** Not to match anyone's convention — to stop paying a
   noise penalty that Level-3 models never incur. This is the first data-vs-model question in the project
   where the data side has direct measured support.


---

## 25. Level 3 MEASURED — and it REVERSES §24's conclusion (2026-08-18)

Level 3 arrived (GSE92742, 65.1 GB, 1,319,138 wells x 12,328 genes; 978/978 landmarks located; 672,128
trt_cp wells; 2,023 plates carrying DMSO controls; **183,485 conditions with >=2 wells on different
plates**). `model/level3/replicate_reliability.py`.

Delta construction is the honest one: a well's delta = its expression minus the **median of ctl_vehicle
wells on the SAME plate**; replicate pairs are restricted to **different plates**, since same-plate pairs
share a control vector and would inflate agreement.

| measurement | reliability r |
|---|---|
| Level-3, **single well** (1,546 pairs) | 0.0893 |
| Level-3, **split-half, 3+3 replicate-averaged** (400 conditions) | **0.1441** (median 0.0990) |
| — top strength quartile of that | **0.2429** |
| **Level-5 MODZ, all signatures** [6.1] | 0.127 |
| **Level-5 MODZ, reproducible stratum (top ~15 %)** [6.1] | **0.509–0.619** |

The single-well number is not a fair contrast — a MODZ signature *is* a replicate average, so it was
compared against 3+3 averaging to match.

- 🔴 **LEVEL 5 IS THE BETTER-DENOISED TARGET, NOT THE WORSE ONE.** At roughly matched percentile the
  Level-3 top quartile reaches 0.243 while the Level-5 reproducible stratum reaches 0.509–0.619. MODZ is a
  *correlation-weighted* replicate average with plate-population z-scoring, explicitly engineered for
  reproducibility; a plain mean of plate-matched deltas is cruder and measurably so.
- 🔴 **§24's conclusion is WRONG and is retracted.** That section inferred "our target is the noisy one"
  from the fact that cross-target agreement sat at 83–92 % of our reliability ceiling. The inference was
  consistent with the evidence available then, but direct measurement beats inference: **migrating to
  Level 3 would raise our noise penalty, not remove it.**
- ⇒ **Do NOT migrate to Level 3 for denoising reasons.** (It remains useful for one thing only: supplying
  *true within-plate controls* if we ever need to report in the absolute convention.)

### The contradiction this exposes, and the most likely resolution

XPert reports **Pearson_deg 0.844** on a Level-3-derived delta, yet the measured reliability of such a
target is **0.144–0.243**. **A model cannot predict a target more accurately than that target agrees with
itself** — so their evaluation subset cannot be a random sample of conditions. The consistent explanation is
that the released set is **HDAC inhibitors**: strong, stereotyped, heavily-replicated perturbations. We
measured exactly this effect in our own data — epigenetic drugs are **+0.20** easier than average [2.6].

**The fairest comparison currently available, like-for-like on drug class:**

| | drug class | split | Pearson (delta) |
|---|---|---|---|
| XPert | HDAC inhibitors | unknown, possibly in-distribution | 0.844 |
| **ours (v7 no-aux)** | epigenetic drugs [2.6] | **unseen COMPOUND** (the hard axis) | **0.648** |

0.844 vs 0.648 — a real gap, but a quarter the size of the naive 0.844-vs-0.4985 headline, and still
favouring them on split difficulty. Closing the remaining confound requires running one model on the other's
conditions, which the h5ad now makes possible.


---

## 26. Level 3 resolved: the TARGET gate FAILS, the INPUT A/B PASSES decisively (2026-08-18)

Two separate questions, two separate answers. `model/level3/`.

### Extraction
189,482 conditions x 978 genes from GSE92742 (672,128 trt_cp wells, 2,023 plates with >=3 DMSO controls,
700,060 wells read in one sequential pass). Each condition carries a **plate-matched DMSO median** as its
control and a MODZ-weighted replicate delta. Median 3 wells/condition. 138,954 of our Level-5 signatures
(44.8 %) join to a matched control; the remainder are GSE70138 phase-2, not yet extracted.

### GATE on using Level 3 as the TARGET — **FAILED**
Split-half over disjoint plate groups, 1,500 conditions, within-condition so both methods see identical wells:

| aggregation | mean r | top-quartile |
|---|---|---|
| plain mean of replicate deltas | 0.1642 | 0.2332 |
| **MODZ-weighted (the proposed mitigation)** | **0.1569** | 0.2275 |
| *Level-5 MODZ reference* [6.1] | *0.127 all* | ***0.509–0.619 reproducible*** |

MODZ weighting is **worse** than a flat mean here (−0.0072). The likely reason: Level 5's reliability comes
mostly from the **Level-4 robust z-scoring against the plate population** (divide by plate MAD, per gene),
not from the replicate weighting — and our Level-3 delta has no such per-gene scaling. ⇒ **Keep the Level-5
z-score as the target.** The gate did its job: it stopped us replacing a good target with a worse one.

### A/B on using Level 3 for the INPUT — **PASSED, and by more than any architecture change to date**
Controlled: identical signatures, identical drug/dose/time features, identical protocol. **The only thing
that varies is which baseline vector the model sees.** Ridge, so the fit is deterministic.

| split | CCLE `X_base` (today) | **plate-matched control** | both | gain |
|---|---|---|---|---|
| unseen CELL | 0.3856 | **0.4171** | 0.4119 | **+0.0314** |
| unseen COMPOUND | 0.3910 | **0.4345** | 0.4369 | **+0.0435** |
| unseen BOTH | 0.3140 | **0.3512** | 0.3476 | **+0.0372** |

- ✅ **The matched control beats the CCLE proxy on every split, by +0.031 to +0.044.**
- ✅ **"Both" is no better than "matched" alone** ⇒ the matched control **subsumes** CCLE; once you have it,
  the CCLE baseline adds nothing.
- ✅ **This number carries no seed variance.** Ridge is a closed-form fit, so unlike every architecture
  comparison in this project [M.10] there is no run-to-run noise to argue about. For scale: the gain is
  larger than the entire v5 → v6 → v7 progression, all of which sat inside seed noise.

### The resolution
**Use Level 3 for the INPUT. Keep Level 5 for the TARGET.** That is exactly what V8_PLAN §1 argued the
migration was for, and the gate correctly prevented the target swap that was never justified.

**Caveats:** ridge, not the full model — the gain may differ once atom tokens and attention are present;
44.8 % coverage until GSE70138 is extracted; and the smaller matched subset makes these splits smaller than
the headline ones, so these numbers are not comparable to §22's, only to each other.


---

## 27. v9 substrate rebuilt: coverage 44.8 % -> 99.65 %, and three handoff claims corrected (2026-08-26)

`model/level3/extract_level3_distil.py`, `test_extract_l3.py`, `noise_ceiling_l3.py`,
`network/scripts/build_priors_v9.py`, `pretrain_gene_vectors.py`, `model/level3/ab_matched_control.py`,
`split_dmso_control.py`.

### 27.1 The key join was the problem, not the missing phase
V9_HANDOFF §D step 1 says extract GSE70138 to lift coverage from 44.8 % to >=90 %. Decomposed first:
**P1 66.3 % (with GSE92742 already extracted), P2 4.1 %** -- and those 4.1 % are FALSE matches, because the
old condition table contains GSE92742 wells only, so a P2 signature keyed to it received a "plate-matched"
control from a different experiment. Extracting GSE70138 under the same key logic projects to ~66 %:
**the handoff's own gate would have failed after the work was done.**

`sig_info.distil_id` is the exact Level-5 -> Level-3 well mapping. Rebuilt on it: **P1 100.00 %,
P2 98.98 %, total 99.65 % (309,020/310,114)**, control drawn from the same wells' plates as the target,
cross-phase pairing impossible by construction. 26/26 design tests pass, including recomputing sampled
signatures directly from the GCTX (max abs diff 1.7e-06).

Things that would have been silent bugs: `GSE70138_Level3.gctx` is a **gzip stream, not HDF5**; the plate
key is **rna_plate for P1** (the inst_id prefix disagrees 20000/20000) and **det_plate for P2**;
`signatures_usable.tsv`'s `row` column indexes the **312,438-row** Level-5 target, not its own 310,114 line
positions.

### 27.2 Noise ceiling: §25's stratum comparison was not like-for-like
56,811 signatures with >=4 wells on >=2 plates, halves drawn from **disjoint plates**:

| stratum | delta | absolute |
|---|---|---|
| all | 0.1376 | 0.9300 |
| 3+3 averaged (>=6 wells) [matches §25] | 0.1700 | 0.9384 |
| top strength quartile, **L3-defined** | 0.3263 | 0.9022 |
| **Level-5 reproducible stratum (strength >= 1.0)** | **0.5283** | 0.9183 |

- 🔴 §25 compared an **L3-defined** top quartile (0.2429) with the **L5-defined** reproducible stratum
  (0.509-0.619). On the stratum this project actually evaluates on, the Level-3 delta self-agrees at
  **0.5283**. The reported gap was largely a **stratum-definition artefact**. Level 5 may still lead at the
  very top; "migrating to Level 3 raises our noise penalty" is **not supported**.
- 🔴 The **absolute convention self-agrees at 0.90-0.95 regardless of perturbation strength.** That is the
  local proof of why absolute numbers look high, next to "copy the control" = 0.9200 [§23].

### 27.3 Priors: one handoff diagnosis right, one wrong, one defect nobody had noticed
- **34/978 landmark symbols are stale 2012 L1000 names** (AARS->AARS1, IKBKAP->ELP1, KIAA0196->WASHC5...).
  All resolve through the Entrez id L1000 itself carries, against the local HGNC set, no collisions. This
  alone moves STRING-absent **28 -> 8** and Reactome orphans **231 -> 213**.
- **STRING truncation: the handoff is right.** Full-proteome at combined_score >= 400 gives
  **19,496 nodes / 929,472 edges**, against XPert's released **19,392 / 901,260** -- so 400 is the field's
  threshold, derived rather than assumed. Isolated landmarks **66 -> 8**. GATE MET.
- 🔴 **Reactome truncation: the handoff is wrong.** Nothing was truncated. `ReactomePathways.gmt` annotates
  only **11,963 genes**, so 231 landmarks are in no pathway at **any** filter (verified at min_size=1 with
  the umbrella exclusion off; coverage caps at 747/978). **"231 -> ~0" is unreachable from Reactome.**
  Adding **GO:BP** as a second NAMED source gives 800 nodes (367 Reactome + 433 GO:BP) and **50 orphans**
  against a two-source floor of 45.
- Gene vectors pretrained by link prediction on the full graph: held-out AUC **0.9198** (degree-matched
  null) / 0.8934 (uniform). Expectation that uniform would be the inflated null was **wrong**; both are
  recorded. Co-membership recovery, never in the objective: **AUC 0.6388**, cos 0.066 same-pathway vs
  -0.002 different.

### 27.4 The baseline A/B, redone clean -- and what it says about CCLE
Closed-form ridge, identical rows across arms, reproducible stratum. `matched - ccle` on the Level-5
target: **+0.0268 / +0.0401 / +0.0354**, reproducing §26's +0.0314 / +0.0435 / +0.0372 on the clean join.

**CCLE adds nothing on top of the matched control** on any target or split (`both - matched` between
-0.0104 and +0.0003). It is **redundant, not harmful** -- and it is **dominated**, not merely matched, by a
per-cell mean of L1000 DMSO controls: identical to 4 dp on seen cells, **+0.037 / +0.027 better on unseen
cells**.

### 27.5 The delta convention lets a model cancel noise instead of predicting biology
v9 trains on `delta = trt - ctl` and is handed `ctl`. The control's measurement noise therefore enters the
target negatively and the input positively. Measured with **two independent half-plate DMSO medians**
(`split_dmso_control.py`; mean |ctlA - ctlB| = 0.1852 per gene, against mean|delta| ~ 0.377):
target built from half B, `ctlB` the coupled input, `ctlA` the uncoupled one.

| split | ccle | cellmean | ctlA (independent) | ctlB (coupled) | ctlB-ctlA | ctlA-cellmean |
|---|---|---|---|---|---|---|
| unseen_cell | 0.2517 | 0.2890 | 0.2055 | 0.2826 | **+0.0771** | **-0.0835** |
| unseen_compound | 0.3434 | 0.3434 | 0.4467 | 0.4547 | +0.0080 | **+0.1033** |
| unseen_both | 0.2224 | 0.2495 | 0.1698 | 0.2453 | **+0.0755** | **-0.0797** |

- 🟢 **On unseen compounds the plate-matched control is genuinely worth +0.103** over a per-cell mean, and
  only +0.008 of that is noise cancellation.
- 🔴 **On unseen CELLS it inverts.** An independent plate control is **worse than a per-cell mean**
  (-0.084 / -0.080), and essentially all of what the coupled control appears to buy (+0.077 / +0.076) is
  **noise cancellation, not signal**. Plate state helps only where the model has seen the cell.
- ⇒ v9 should feed the control encoder **both** the matched control and a per-cell aggregate, and every
  delta number must be reported alongside the independent-control version.
- A first attempt at this control (`l3delta_ind`: target = trt - cellmean) **FAILED and is recorded as a
  negative** -- it swaps the coupling for a worse one, since the target then contains the plate offset
  `ctl - cellmean`, readable straight off the matched input (+0.3771 on unseen compounds: the size of a
  leak, not an effect).

## 28. The SOTA gap is mostly the SPLIT, and the handoff's description of their splits is wrong (2026-08-26)

`model/v9/sota_split_audit.py`, `model/v9/sota_matched_difficulty.py`.

### 28.1 XPert's splits are NOT tissue holdouts
V9_HANDOFF §C states their splits are "tissue-holdouts: `split_lung_1..5`, `split_breast_1..5`,
`split_haematopoietic_and_lymphoid_tissue_1..5`". Read from their own released h5ad, all **15 splits**
restrict to ONE tissue and then divide it ~90/10, so train and test share the tissue, the cell lines, and
nearly all compounds:

| | test rows whose CELL was in training | COMPOUND | (cell, compound) PAIR | exact (cell, cmpd, dose, time) |
|---|---|---|---|---|
| mean over 15 splits | **100.0 %** | **98.6 %** | **89.4 %** | 0.0 % |

So the conditions themselves are genuinely held out, but the task is overwhelmingly **interpolation to a
different dose or time of a (cell, compound) pair already in training**. Our benchmarks hold out entire cell
lines and entire Bemis-Murcko scaffold families. 🔴 **The handoff's characterisation is retracted.**

### 28.2 What the split alone is worth — one ridge, one feature set, only the split changes

| regime | cell seen | cmpd seen | pair seen | **delta** | **abs** | copy_ctl | mean_drug |
|---|---|---|---|---|---|---|---|
| xpert_pair (matched to theirs) | 100.0 % | 93.4 % | 60.2 % | **0.5459** | 0.9496 | 0.9287 | 0.4057 |
| xpert_style (random 90/10) | 100.0 % | 90.8 % | 49.4 % | 0.5294 | 0.9474 | 0.9273 | 0.3681 |
| xpert_tissue (within one lineage) | 100.0 % | 85.1 % | 73.0 % | 0.5219 | 0.9575 | 0.9404 | 0.3365 |
| cold_compound (ours) | 100.0 % | 0.0 % | 0.0 % | 0.4744 | 0.9319 | 0.9185 | 0.3156 |
| cold_cell (ours) | 0.0 % | 96.7 % | 0.0 % | 0.4193 | 0.9184 | 0.9257 | 0.3574 |
| cold_both (ours, hardest) | 0.0 % | 0.0 % | 0.0 % | 0.3764 | 0.8966 | 0.9080 | 0.3215 |

- 🟢 **Changing only the split moves the delta metric by +0.17** (0.3764 → 0.5459) for an identical model.
  That is roughly four times the 2-sd seed band (±0.046) and larger than every architectural effect this
  project has ever measured, combined.
- 🔴 **In the absolute convention our ridge is WORSE than doing nothing on unseen cells**: 0.9184 against
  copy-the-control 0.9257 (−0.007), and 0.8966 against 0.9080 on cold_both (−0.011). It only clears the
  do-nothing baseline in the easy regimes (+0.020 / +0.017 / +0.021).
- The convention-free way to read any absolute number is **value added over copying the control**. XPert's
  released predictions: 0.9804 against 0.9200 = **+0.060** [§23]. Our ridge on the comparable regime:
  0.9496 against 0.9287 = **+0.021**. A published transformer adds three times what a ridge does — a real
  difference, and a far smaller one than "0.98 versus 0.95" suggests.
- `mean_drug` on the delta target is 0.32–0.41 in EVERY regime. The drug-mean null barely notices the split;
  the model's margin over it swells from +0.05 (cold_both) to +0.14 (xpert_pair). Most of what the easy
  split buys is cell- and pair-specific memorisation, not better drug modelling.

⇒ **No comparison of our headline numbers with published LINCS numbers is admissible without stating the
split structure.** Reporting 0.4985 against 0.844 as a deficit is measuring the benchmark, not the model.

## 29. GATE 5: the binned expression encoder is NOT measurably better (2026-08-26)

`model/v9/ab_encoder.py` on Kaggle T4 x2. The REAL v9 model at reduced width (d_model 128, 3 epochs,
50,000 training rows), changing exactly one thing — `expr_encoder` — with identical data, split, schedule
and seed sequence, and identical data parallelism in both arms. 3 seeds each.

| target | split | raw | binned | binned − raw | |
|---|---|---|---|---|---|
| delta | unseen_cell | 0.4180 [0.4146, 0.4223] | 0.4261 [0.4227, 0.4281] | **+0.0081** | ranges do not overlap |
| delta | unseen_compound | 0.4416 [0.4379, 0.4471] | 0.4447 [0.4440, 0.4456] | +0.0031 | inside seed range |
| delta | unseen_both | 0.4270 [0.4221, 0.4356] | 0.4301 [0.4222, 0.4420] | +0.0032 | inside seed range |
| l5 | unseen_cell | 0.3740 [0.3712, 0.3780] | 0.3821 [0.3752, 0.3905] | +0.0082 | inside seed range |
| l5 | unseen_compound | 0.3619 [0.3492, 0.3777] | 0.3543 [0.3441, 0.3738] | −0.0075 | inside seed range |
| l5 | unseen_both | 0.3733 [0.3646, 0.3858] | 0.3757 [0.3590, 0.3980] | +0.0024 | inside seed range |

- 🔴 **NO DIFFERENCE on 5 of 6 comparisons.** The sixth (delta, unseen_cell) has non-overlapping seed
  ranges and favours binned by +0.0081 — one comparison out of six, at n=3, with an effect a fifth of the
  2-sd band. On this project's record that is not a result yet.
- ⇒ **v9 uses the binned encoder for FIELD-COMPARABILITY, not for accuracy**, and says so. It is what
  XPert's config specifies (`n_bins: 128`) and it makes our expression encoding the same object as theirs;
  the A/B says it costs nothing and buys nothing measurable. `--expr_encoder raw` remains one flag away.
- The A/B is also this project's seventh architecture comparison to come back indistinguishable from seed
  noise. The two effects that HAVE cleared the noise band both remain data effects, not architecture ones:
  the plate-matched control (+0.027…+0.040) and the split itself (+0.17, §28).

### 29.1 A capacity finding that shaped the seed runs — CORRECTED 2026-08-27
**The estimate was wrong and cost the first round a fair schedule.** The 12-epoch probe actually ran
**12 epochs in 5.62 h** (1,636 s/epoch), comfortably inside the 7.5 h budget — its accuracy numbers are
void (NaN quantiser, §32) but its clock is not. The projection below came from scaling an I/O-bound
measurement as if it were compute-bound, and it was pessimistic by roughly 3x. The re-runs therefore use
**12 epochs**, which is also v7's schedule, so the v7 comparison is no longer budget-limited.

*Original reasoning, kept because it is why the first round was shaped the way it was:*
The A/B ran at 0.197 s/step (d=128, 4 blocks, batch 48) — and that run was I/O bound, since it used
`cache_in_ram=False`. Scaled to d=256 at full depth, a 12-epoch run over 179,772 rows projects **past
Kaggle's 9 h session limit**. A run truncated by the budget guard stops at whatever epoch it reached, and
three seeds stopping at *different* epochs are not comparable — which would defeat the reason for running
three. The seed runs are therefore shaped to FIT (6 epochs, batch 96, `l_control` 1, budget 7.5 h) rather
than shaped to be cut off.

## 30. SOTA arm: v9 on XPert's own data, split and metric (2026-08-26)

`model/v9/xpert_arm.py`, 3 seeds per split, on their released rows with our drug features and our
chromatin where the cell line is one of ours.

| their split | train | test | **Pearson (abs)** | **Pearson_deg (delta)** | copy-the-control | mean-drug |
|---|---|---|---|---|---|---|
| breast_1 | 47,696 | 5,310 | **0.9838** [0.9838, 0.9839] | **0.7070** [0.7064, 0.7079] | 0.9673 | 0.2245 |
| haematopoietic_1 | 17,362 | 1,943 | **0.9858** [0.9857, 0.9858] | **0.7067** [0.7053, 0.7081] | 0.9718 | 0.2999 |
| lung_1 | 37,479 | 4,185 | **0.9801** [0.9800, 0.9801] | **0.6993** [0.6989, 0.6995] | 0.9615 | 0.1757 |
| *their reported* | | | *0.9804* | *0.8440* | | |

- 🟢 **On the absolute convention v9 matches or beats the published number** (0.9801–0.9858 vs 0.9804) —
  on lung_1 to four decimal places. That is the headline number of a published transformer, reproduced by
  this project's model on this project's features.
- 🔴 **And it means almost nothing**, which is the point. On these same rows **copy-the-control scores
  0.9615–0.9718**, so the entire distance between doing nothing and a state-of-the-art model in this
  convention is about **+0.02**, and that is what both models deliver (+0.0140…+0.0186 for v9).
- 🔴 **On the delta convention v9 reaches 0.699–0.707 against their 0.844** — a real ~0.14 gap on the
  metric that actually measures perturbation response. Stated limits: 5.4 % of their split rows use
  compounds we cannot featurise and were dropped; 6–22 % of rows use cell lines with no chromatin or
  lineage of ours; 8 epochs; and their heterogeneous graph carries DTI and drug–drug edges we exclude
  deliberately, because `dti_reference.tsv` is our held-out interpretability validation set.
- 🟢 **Seed variance collapses on their split structure**: range 0.0001–0.0015 here, against 0.005–0.045 on
  our cold splits. With 100 % of test cells and ~89 % of test (cell, compound) pairs already in training
  [§28], the task is nearly deterministic. The "≥3 seeds" rule is far more load-bearing on our benchmark
  than on theirs — which is also why a single-seed number on a split like theirs looks so stable.

## 31. v9, 3 seeds — and a negative result that must not be buried (2026-08-26)

`model/v9/train_v9_gpu.py`, fold 0, 6 epochs, batch 96, d_model 256, `l_control` 1, binned encoder.
**Two of three seeds complete at the time of writing; seed 2 and a 12-epoch probe are running.**

| split | delta | abs | copy-the-control | value added | l5 | v7 `--no_aux` on l5 |
|---|---|---|---|---|---|---|
| unseen_cell | 0.3961 [0.3943, 0.3979] | 0.9350 | 0.9235 | +0.0115 | 0.3946 [0.3890, 0.4002] | **0.4549** |
| unseen_compound | 0.4783 [0.4652, 0.4913] | 0.9358 | 0.9191 | +0.0167 | 0.4354 [0.4129, 0.4579] | **0.4985** |
| unseen_both | 0.3533 [0.3513, 0.3553] | 0.9174 | 0.9080 | +0.0094 | 0.3403 [0.3379, 0.3427] | **0.4825** |

- 🔴 **v9 is BELOW v7 `--no_aux` on the target they share** (−0.060 / −0.063 / −0.142), and below the §28
  ridge on its own delta target on two of three splits (0.3961 vs 0.4193 unseen_cell; 0.3533 vs 0.3764
  unseen_both). It clears the ridge only on unseen_compound (0.4783 vs 0.4744).
- Three confounds, all stated rather than used as excuses: (1) **6 epochs against v7's 12**, with the
  training loss still falling (0.8773 → 0.7024) and unseen-compound l5 still climbing (0.335 → 0.458) at
  the last epoch — the run used **2.91 h of a 7.5 h budget**, so the shape was chosen from a bad estimate
  and left capacity unused; (2) **v9's l5 head is a 0.3-weighted auxiliary** while it was v7's entire
  objective, so this compares a side task with a main task; (3) reduced depth (`l_control` 1, not 2).
- ⇒ The 12-epoch probe decides which of these it is. Until it returns, the defensible statement is: **at
  this training budget the v9 substrate and priors do not beat v7, and do not beat a ridge on the two
  unseen-cell splits.** The eighth architecture comparison in a row that fails to clear its baseline.

## 32. 🔴 RETRACTION: the binned quantiser was NaN-poisoned; §29–31's binned numbers are void (2026-08-26)

`model/v9/modules_v9.py`. Found by the interpretability probe, not by inspection.

The probe reported `ablate matched_control -> |dY|max 0.0000` and `ablate cell_control -> |dY|max 0.0000`
on all three splits — **exactly** zero, which is far too clean for "the model learned to ignore it". The
weights say why: **all 127 bin edges in `ckpt_v9_fold0_seed0` are NaN.**

**Cause.** The Level-3 substrate stores the 1,094 uncovered signatures as **NaN on purpose**, so that using
them without the mask fails loudly [§27.1]. The quantiser's fitting sample was selected with
`ds_to_l3 >= 0` — which means the row EXISTS in the arrays, not that it is COVERED. 153 NaN rows landed in
a 40,000-row sample, and `np.percentile` propagates NaN to **every** output. Every value then bucketed to
0, so the expression embedding was a constant.

The model trained to convergence, produced falling losses, **matched a published absolute number on
XPert's own benchmark**, and reported sensible metrics throughout — with no expression value reaching its
trunk at all.

**The guard was checking the wrong property.** `test_v9.py` asserted `fitted == 1.0`, i.e. that `fit()` had
been *called*. What matters is whether the bins **discriminate**. That is the same class of error this
project keeps making: a valid computation of the wrong quantity [method rule 5].

| voided | still valid |
|---|---|
| §29 encoder A/B — the **binned** arm | §29's **raw** arm |
| §31 seeds 0–2, and the 12-epoch probe | §27 substrate, noise ceiling, priors, gene vectors |
| §30 the whole XPert arm | §27.4/27.5 baseline A/B and the split-DMSO test (ridge, no quantiser) |
| the untrained null of `v9_probe_untrained` | §28 split audit and matched difficulty (ridge, no quantiser) |

**What §29 actually measured, in hindsight.** Its "binned" arm had *no expression input whatsoever*, and it
still matched the raw arm on 5 of 6 comparisons. So the honest reading of that experiment is not
"binning does not help" but **"the control profile contributes ~nothing to the delta prediction through
this trunk"** — a stronger and more interesting negative, and now a hypothesis to re-test rather than a
result.

**Fix.** `fit()` refuses non-finite input (`drop_nan=True` to override deliberately) and verifies the edges
are finite and non-degenerate before installing them; a new `discriminates()` backs both the forward guard
and `model.bins_fitted`; every call site filters to covered rows and asserts finiteness. Five regression
tests added, including *one NaN row is refused* and *a NaN-edged quantiser reports itself unusable even
though `fitted == 1`*. 55/55 pass. After the fix: edges 127/127 finite over [3.822, 14.261], 128 distinct
bins on a real batch, and the matched-control ablation moves the output.

**The readout's NULL survives the bug, re-measured on a working quantiser (2026-08-27).** An untrained v9
scores pathway alignment **−0.0271 / −0.0124 / −0.0214** against permutation nulls of
**+0.0002 ± 0.0095 / −0.0014 ± 0.0113 / −0.0006 ± 0.0114** — within noise of the NaN-poisoned measurement
(−0.0264 / −0.0128 / −0.0173), as expected: an untrained model's pathway activations are random whether or
not the expression input is live. So **chance for this readout is ≈ 0.000 ± 0.010** and that figure stands.
The TRAINED alignment below still has to be re-measured, because the checkpoint it came from did not see
its inputs.

🟢 **One readout did survive, and it is the project's deliverable.** The named pathway alignment was
measured against its own permutation null on the *broken* checkpoint and beat it on all three splits:
**+0.0737 / +0.1516 / +0.0872 against nulls of +0.0002 / −0.0003 / −0.0011 (sd ≈ 0.009–0.010), p = 0.005**
— 8–15 sd above chance. It is the first interpretability readout in this project to clear its own measured
chance level, and it did so while the expression input was dead, i.e. from drug, gene identity and
chromatin alone. It must be re-measured on a correct checkpoint before it is claimed.

## 33. v9 with a WORKING quantiser: the first architecture here to clear its linear baseline (2026-08-29)

12 epochs (v7's schedule), batch 96, d_model 256, `l_control` 1, binned encoder, fold 0.
**All 3 seeds complete (updated 2026-08-29).** All three logged
`mounted code verified: NaN-quantiser fix present and bins discriminate` and
`quantiser fitted on 40000 TRAINING rows, 128 bins`, so §32's failure cannot be present.

### On v9's own target (the Level-3 delta)

| split | v9 delta, mean [min, max] | ridge [§28] | 6-epoch **broken** [§31] | v9 − ridge |
|---|---|---|---|---|
| unseen_cell | **0.5168** [0.5152, 0.5192] | 0.4193 | 0.3961 | **+0.0975** |
| unseen_compound | **0.5792** [0.5639, 0.5890] | 0.4744 | 0.4783 | **+0.1048** |
| unseen_both | **0.4646** [0.4569, 0.4690] | 0.3764 | 0.3533 | **+0.0882** |

- 🟢 **v9 beats the ridge on every split by +0.086…+0.100**, against a 2-sd seed band of ±0.046 and
  observed seed ranges of 0.0007–0.0207. **This is the first architecture in this project to clear a linear
  baseline on the cold splits at all** — v5 was *beaten* by the drug mean on unseen cells [1.8], and the
  6-epoch broken round sat below the ridge on two of three.
- The comparison against §31 is not a like-for-like ablation — it changes two things at once (the quantiser
  fix and 6→12 epochs) — but §31's model had **no expression input whatsoever**, so the honest reading is
  that the substrate plus a real schedule is worth ~+0.11 over what §31 measured.
- 🟢 On unseen cells, delta **0.5155** sits just under the target's own half-vs-half agreement of **0.5283**
  on the same stratum [§27.2]: **the model predicts the Level-3 delta about as well as the target predicts
  itself from half its replicates.**

### On the Level-5 target, the only one comparable with v3–v7

| split | v9 l5, mean [min, max] | v7 `--no_aux` | verdict |
|---|---|---|---|
| unseen_cell | **0.5058** [0.5019, 0.5100] | 0.4549 | **+0.0509, v9 ahead** (just outside the ±0.046 band) |
| unseen_compound | 0.4930 [0.4627, 0.5140] | 0.4985 | −0.0055, **NO DIFFERENCE** |
| unseen_both | 0.4009 [0.3946, 0.4049] | **0.4825** | −0.0816, **v7 ahead** |

- v9 wins unseen_cell, ties unseen_compound, loses unseen_both — **and it does so with the l5 head as a
  0.3-weighted AUXILIARY**, where it was v7's entire objective. The comparison is stacked against v9 and it
  still takes the split this project has never been able to move (§1.8: on unseen cells v5 lost to the drug
  mean).
- 🔴 **unseen_both is a real regression** and is not explained away by the weighting: −0.083 is well outside
  the band. Whatever v9 buys on unseen cells, it does not carry to unseen cells × unseen compounds.

### The absolute convention, with its null attached

| split | v9 abs | copy-the-control | value added |
|---|---|---|---|
| unseen_cell | 0.9434 | 0.9238 | **+0.0196** |
| unseen_compound | 0.9422 | 0.9189 | **+0.0233** |
| unseen_both | 0.9248 | 0.9080 | **+0.0168** |

XPert's released predictions add **+0.060** over the same null on their own (far easier) rows [§23]. Ours
add +0.016…+0.023 on cold splits. Neither number means anything without the null beside it.

## 34. GATE 5, redone with a quantiser that works: binning still buys nothing (2026-08-29)

`model/v9/ab_encoder.py`, 3 seeds per arm, T4 x2. §29's binned arm was measuring a constant embedding
[§32]; this one logged `mounted code verified: NaN-quantiser fix present and bins discriminate` before
training, so the comparison is real this time.

| target | split | raw | binned | binned − raw | |
|---|---|---|---|---|---|
| delta | unseen_cell | 0.4158 [0.4102, 0.4189] | 0.4260 [0.4202, 0.4345] | +0.0102 | inside seed range |
| delta | unseen_compound | 0.4429 [0.4418, 0.4451] | 0.4422 [0.4411, 0.4430] | −0.0007 | inside seed range |
| delta | unseen_both | 0.4245 [0.4199, 0.4306] | 0.4249 [0.4174, 0.4368] | +0.0005 | inside seed range |
| l5 | unseen_cell | 0.3743 [0.3700, 0.3783] | 0.3785 [0.3687, 0.3906] | +0.0042 | inside seed range |
| l5 | unseen_compound | 0.3680 [0.3627, 0.3781] | 0.3516 [0.3461, 0.3604] | **−0.0164** | **raw ahead** |
| l5 | unseen_both | 0.3783 [0.3706, 0.3853] | 0.3709 [0.3538, 0.3851] | −0.0074 | inside seed range |

- 🔴 **The field's 128-bin embedded encoding buys nothing over a linear layer on the raw scalar.** Five of
  six comparisons sit inside the seed range, and the one that clears it **favours raw** (l5, unseen
  compound, −0.0164). Measured now with a quantiser that demonstrably discriminates, so §29's confound is
  gone and the conclusion is about the encoding rather than about a dead branch.
- The v9 seed runs [§33] use `binned`. On this evidence they could equally have used `raw`; the choice is
  recorded as field-comparability, and it is not doing any work.
- 🔴 This is the **eighth** architecture comparison in this project to land inside seed noise. Everything
  that has ever cleared the band is data: the plate-matched control (+0.027…+0.040 [§27.4]), the split
  itself (+0.17 [§28]), and now the substrate-plus-schedule as a whole (+0.086…+0.100 over the ridge [§33]).
- Note the scale gap: this A/B runs d_model 128 / 3 epochs / 50k rows and reaches delta ≈ 0.42–0.43, while
  the full 12-epoch d_model-256 model reaches 0.5155 [§33]. A/B verdicts at reduced scale bound the
  *encoding* question, not the model's ceiling.

## 35. Is Phase 2 a cleaner substrate? Partly — but P2-only training is not the win (2026-08-29)

`model/v9/phase_ab.py`. Ridge on the Level-3 delta; **only the training rows change**, and every arm is
size-matched to N2 = 55,123 so the phase effect is separated from the data-volume effect.

| arm | n_train | cell/P1 | cell/P2 | cmpd/P1 | cmpd/P2 | both/P1 | both/P2 |
|---|---|---|---|---|---|---|---|
| p2_only | 55,123 | 0.2180 | **0.5304** | 0.2522 | 0.5214 | 0.2275 | 0.4197 |
| p1_only | 55,123 | 0.3511 | 0.3056 | 0.4683 | 0.2460 | 0.3119 | 0.2680 |
| both_eq | 55,123 | 0.3676 | 0.5011 | 0.4596 | 0.5133 | 0.3385 | **0.4510** |
| both_full | 60,000 | 0.3574 | 0.4948 | 0.4619 | 0.5231 | 0.3301 | 0.4563 |

- 🟢 **P2 rows ARE substantially easier**: with the same model and the same mixed training set, P2 test rows
  score **+0.11 to +0.13** above P1 test rows (0.5011 vs 0.3676 on unseen cells). The "less noise" intuition
  about the later, more standardised production run is real — **as a property of the evaluation rows.**
- 🔴 **But training on P2 alone is NOT better, even when you only care about P2 rows.** Phase effect at
  fixed size (`p2_only − both_eq`): **+0.029 / +0.008 / −0.031** on P2 — at or inside the noise, and
  *negative* on the hardest split — while costing **−0.111 to −0.207** on P1. Keeping both is strictly
  better overall and no worse on P2.
- 🔴 **There is a real domain shift between the phases.** Each trains best on its own kind: `p1_only`
  scores 0.3511 on P1 but 0.3056 on P2; `p2_only` scores 0.5304 on P2 but 0.2180 on P1. Pooling them is a
  modelling choice with consequences, not a free concatenation.
- 🔴 **The ridge saturates around 55k rows**: `both_full − both_eq` is −0.010…+0.010 everywhere. Beyond
  ~55,000 training rows the linear model gains nothing, so "more LINCS" is not the lever.
- ⇒ **Action: do not switch to P2-only. DO report P1 and P2 separately.** A pooled headline is dominated by
  the harder P1 rows and hides that the model does markedly better on the cleaner phase — the same
  stratify-and-report-all-strata rule as method rule 1, applied to a new axis.
- Caveat: this is a ridge, not v9. That is the point of running it as a gate — it costs one CPU run instead
  of three GPU sessions, and it says the GPU sessions are not worth spending on P2-only.

## 36. Can we run the published SOTA models themselves? Read from their release (2026-08-29)

The gate says a SOTA comparison needs the same data, convention and split. The strongest version of that is
running THEIR model, so this is an audit of whether their release permits it.

### 36.1 What their release contains, and what it is missing
`external/xpert/` ships their code, the 336,852-condition h5ad, trained weights
(`l1000_mdmt_warm_split.pth`, 67 MB) and `HG_data/` (901,260 PPI edges, 12,890 DTI edges, 287,834 drug-drug
edges, node features, and the pretrained drug HG embedding). **`processed_data/` is empty except a
gitkeep**, and `configs/config_l1000.yaml` needs nine files from it, including:

| required by the config | what it is | shipped? |
|---|---|---|
| `PPI_gene_vector_128d.npy` | the pretrained 128-d gene vector the model reads | ✗ |
| `all_drugs_unimol_arr.npy` | per-drug UniMol atom features | ✗ |
| `all_drugs_idx2smi_8981.npy` | the SMILES map keyed by THEIR `pert_idx` | ✗ |
| `l1000_mdmt_full_336852.h5ad` | the dataset | ✓ (at the release root) |

🔴 **Their release is not runnable as published.** Two of the three missing inputs can be regenerated
(`pretrain_hg.py` rebuilds the gene vector from `HG_data/`; UniMol features can be recomputed from SMILES),
but the third — the SMILES map keyed by their internal `pert_idx` — would have to be reconstructed from our
own drug table plus an index alignment. **Anything produced that way is OUR pipeline, not theirs**, and any
number from it must be labelled as a reimplementation rather than a reproduction.

Also: their model imports `flash_attn` at module level and uses it in exactly two places. FlashAttention is
exact rather than approximate, so a `scaled_dot_product_attention` shim is a faithful substitute and the
dependency is not a real barrier — nor is the environment, since a Kaggle kernel with internet enabled can
install `torch_geometric` and `flash-attn` directly. **The barrier is the missing preprocessed inputs, not
the compute.**

### 36.2 Their released predictions are the HDAC-inhibitor figure, not a benchmark split
`reproducing/fig4/hdaci_predict/y_pred.npy` is what §23 scored (0.9804 absolute / 0.8440 delta /
0.9200 copy-the-control). §23 said so; **later summaries of mine, including §30's table, presented those as
"their reported" benchmark numbers, which overstates what they are.** Corrected here.

### 36.3 Their split names, from their own artifacts
Their checkpoint is named **`l1000_mdmt_warm_split.pth`**, `train_xpert.py` exposes
`--nfold split, split_cold_drug, split_cold_cell`, and the paper's three regimes are **warm-start,
cold-cell and cold-drug**. So §28's finding is confirmed from their side: what they RELEASED is the
warm-start split. The cold-split definitions and the baseline prediction files their own metric script
reads (`deepce`, `prnet`, `transigen`, `xpert`) are **not** in the release.

From the paper (text, not figures): XPert's PCC beats the next-best model by **8.85 % (cold-drug)** and
**30.54 % (cold-cell)**, and **in the cold-cell scenario only XPert and DeepCE avoid negative R²** — i.e.
PRnet, TranSiGen and CIGER score *worse than predicting the mean* on unseen cell lines. ~~The per-model
absolute values live inside figure panels and are not extractable, so they are not quoted here.~~
🔴 **RETRACTED 2026-09-20 — see §46.1. They are in Supplementary Table R8**, a plain table in the public
Supplementary Information. This sentence sent the project on a six-week detour reverse-engineering numbers
that were published. XPert cold-cell PCC = **0.383 ± 0.027**; TranSiGen = **0.293 ± 0.017**.

### 36.4 A concrete difference in objective, worth testing
Their `loss_weight: [0.2, 0.003, 0.2, 1]` = absolute 0.2, control-reconstruction 0.003, delta 0.2,
**PCC 1.0**. Their objective is dominated by the correlation term — the metric they report. Ours is
`task_w = (1.0 abs, 1.0 delta, 0.3 l5, 0.5 pcc)` with abs and delta being the same objective [§fca65f1], so
ours is MSE-dominated at an effective 2.0 against PCC 0.5. **Directly optimising the reported metric is a
cheap, testable lever we have not tried.**

## 37. v9 interpretability report: what is load-bearing, what is a true null (2026-08-29)

`model/v9/probe_v9.py` on the fixed seed-0 checkpoint, 480 rows per split, ablate-to-the-MEAN, dPearson =
median row Pearson lost when the component is removed. Within-run on identical signatures, so seed variance
does not apply [method rule 7].

| ablated | unseen_cell | unseen_compound | unseen_both | \|dY\|max |
|---|---|---|---|---|
| gene representation† | **+0.325** | **+0.410** | **+0.282** | 7.5 |
| drug global features | **+0.151** | **+0.223** | **+0.148** | 7.8 |
| **matched control** | **+0.105** | **+0.236** | **+0.136** | 6.3 |
| **chromatin** | −0.003 | **+0.030** | −0.006 | 4.5 |
| per-cell control | +0.027 | +0.028 | +0.018 | 5.5 |
| lineage | −0.004 | +0.021 | −0.007 | 3.4 |
| atom tokens | **−0.007** | **−0.025** | **−0.022** | 2.1 |
| STRING message passing | −0.0004 | −0.0005 | −0.0003 | 0.50 |
| named pathway readout | +0.0009 | +0.0007 | −0.0018 | 0.88 |

† **This row does NOT isolate the pretrained gene vectors.** `gene_repr` emits learned embedding + STRING
vector + chromatin summed; ablating the module to its gene-axis mean removes *all gene identity*, which is
expected to be catastrophic. Attributing +0.33 to the pretrained vectors specifically would need a
`--no_gene_vectors` training run, which has not been done.

- 🟢 **The matched control is a top-three contributor** (+0.105 / +0.236 / +0.136). The data-work thesis
  holds inside the trained model, not just in the ridge A/B.
- 🟢 **CHROMATIN NOW CONTRIBUTES ON UNSEEN COMPOUNDS: +0.030**, against v6/v7's +0.0061 on the same split —
  roughly five times larger — while staying ~0 on both unseen-CELL splits (−0.003 / −0.006). The pattern is
  identical in shape to [2.5] and larger in size, which is what moving chromatin from a parallel branch to a
  per-gene embedding summed into the gene token [handoff §D.4] was supposed to do. **It tracks cell
  familiarity exactly as before: chromatin helps when the cell is known and the compound is not.**
- 🔴 **ATOM TOKENS ARE ACTIVELY HARMFUL**: removing them *improves* accuracy on all three splits
  (−0.007 / −0.025 / −0.022). The one measured improvement in this project's history came from deleting a
  component (`v7 --no_aux`); this is the next deletion candidate, and it is cheap to test.
- 🔴 **STRING message passing is a true null**: −0.0003…−0.0005 with \|dY\|max ≈ 0.5, so it fires and
  contributes nothing. Unchanged from v7 despite the graph now being full-proteome.
- 🔴 **The named pathway readout is a true null FOR ACCURACY** (+0.0009 / +0.0007 / −0.0018,
  \|dY\|max 0.6–0.9), replicating v6/v7 exactly.
- 🟢 **…and yet its alignment beats its own permutation null by 8–12 sd**: **+0.0806 / +0.0767 / +0.1231**
  against nulls of −0.0002 ± 0.0085, −0.0002 ± 0.0100, −0.0008 ± 0.0099, all **p = 0.005**. Chance for this
  readout was measured at ≈ 0.000 ± 0.010 on an untrained model [§32].
- ⇒ **This is the cleanest statement of the project's position.** The named pathway layer buys **nothing**
  in accuracy and carries **real, measurable mechanistic signal**. Those are not in tension; they are the
  reason the interpretability claim has to be made on its own null rather than on the accuracy number.

**Correction to the probe itself.** The first version of this table reported `y.abs().mean()` — the change
in prediction MAGNITUDE, which is what \|dY\|max already answers — as though it were a contribution. Worse,
the accuracy lambda that existed alongside it indexed a global target array per chunk, so it would only
have lined up for the first chunk. Both fixed; every number above is a change in accuracy against that
chunk's own targets.

## 38. The three regimes the field reports, ours for the first time — plus phase, epi-drugs and EMA (2026-08-29)

`model/v9/regimes_v9.py`, seed-0 checkpoint. Everything here is on the reproducible stratum.

### 38.1 Warm-start, measured for the first time in this project
The field reports **warm-start / cold-cell / cold-drug**; we had only ever reported the two cold regimes,
which meant there was nothing of ours to set beside their headline. Our `val` split IS warm-start by
construction — it excludes the held-out cells and the held-out scaffold family, so its rows share both cell
lines and compounds with training and only the exact condition is unseen.

| regime | delta | abs | copy-the-control | value added | l5 |
|---|---|---|---|---|---|
| **warm_start** | **0.6607** | 0.9647 | 0.9316 | **+0.0331** | 0.6168 |
| cold_drug | 0.5855 | 0.9437 | 0.9219 | +0.0218 | 0.5212 |
| cold_cell | 0.5144 | 0.9441 | 0.9253 | +0.0188 | 0.5023 |
| cold_both | 0.4642 | 0.9254 | 0.9086 | +0.0168 | 0.4012 |

- **Warm-start is worth +0.15 to +0.20 in delta over the cold regimes** for the same model — the same
  ordering §28 measured with a ridge, now inside the trained model.

### 38.2 By phase: the split the ridge predicted, larger in the model
| regime | P1 delta | P2 delta | P2 − P1 | P1 l5 | P2 l5 | P2 − P1 |
|---|---|---|---|---|---|---|
| cold_cell | 0.4216 | **0.5951** | **+0.1735** | 0.3841 | 0.5926 | **+0.2085** |
| cold_drug | 0.5426 | **0.6201** | +0.0775 | 0.4438 | 0.5793 | +0.1355 |
| cold_both | 0.3780 | **0.5561** | **+0.1781** | 0.2997 | 0.5546 | **+0.2549** |

The ridge put the P2 advantage at +0.11…+0.13 [§35]; the trained model puts it at **+0.08…+0.18 on delta
and +0.14…+0.25 on Level-5**. GSE70138 is a materially cleaner benchmark, and a pooled headline hides it.
**Report both phases, always.**

### 38.3 Epi-drugs: the like-for-like read of their HDACi figure
| regime | epi-drugs | all others | epi advantage |
|---|---|---|---|
| cold_cell | **0.6472** (n=205) | 0.5085 | **+0.1387** |
| cold_drug | **0.7106** (n=961) | 0.5656 | **+0.1450** |

Epi-drugs are **+0.14 easier**, close to the +0.20 this project measured earlier [2.6]. XPert's released
0.8440 is an HDAC-inhibitor figure [§36.2], so the honest comparison for it is our epi-drug row, not our
headline — and on cold-drug we reach **0.7106** on that class.

### 38.4 EMA does nothing in v9 either
| regime | raw | EMA | EMA − raw |
|---|---|---|---|
| warm_start | 0.6607 | 0.6601 | −0.0006 |
| cold_cell | 0.5144 | 0.5144 | +0.0000 |
| cold_drug | 0.5855 | 0.5824 | −0.0031 |
| cold_both | 0.4642 | 0.4654 | +0.0012 |

v7 measured −0.0005 / +0.0003 / +0.0001 [§21]; v9 reproduces that on a different architecture, a different
target and a different data substrate. **EMA is confirmed dead here, not merely unmeasured.** The plausible
reason — the WSD schedule already anneals the learning rate to ~0, so the endpoint is effectively an average
over a low-LR phase, and reliability weighting removes some inert-row noise upstream — remains an
explanation, not a measurement.

## 39. The closest like-for-like comparison we can build against a published number (2026-08-29)

XPert's released predictions are an HDAC-inhibitor figure in a warm regime [§36.2], and their headline
delta on it is **0.8440**. Every previous comparison in this project set that against our *cold-split,
all-compound* number (0.4985), which differs from theirs in drug class, split regime and stratum at once.

`model/v9/regimes_v9.py` + bootstrap. Median row Pearson on the Level-3 delta, 95 % CI over 4,000
bootstrap resamples of the rows:

| our set | n | median delta [95 % CI] |
|---|---|---|
| **warm-start × epi-drugs** | **35** | **0.7963 [0.7372, 0.8430]** |
| warm-start, all other compounds | 600 | 0.6561 [0.6286, 0.6788] |
| cold-drug × epi-drugs | 961 | 0.7106 [0.6973, 0.7189] |
| cold-drug, all other compounds | 600 | 0.5135 [0.4835, 0.5415] |
| cold-cell × epi-drugs | 205 | 0.6472 [0.6249, 0.6688] |
| cold-cell, all other compounds | 600 | 0.4548 [0.4370, 0.4687] |
| *XPert, released HDACi figure* | *3,439* | *0.8440* |

- ⚠️ **SUPERSEDED BY §40 — read that first.** This section compares our model on OUR rows with their model
  on THEIR rows. §40 runs both on IDENTICAL rows and the conclusion reverses: XPert leads 0.857 vs 0.523 on
  the delta. What follows is still true as written (matching drug class and regime closes most of the
  *nominal* gap between two separately-measured numbers) but it must not be read as a model comparison.
- On the closest matched construction — warm regime, epigenetic compounds, Level-3 delta — v9 reaches
  0.7963, and their 0.8440 sits at the top edge of our 95 % CI.
- 🔴 **n = 35, CI width 0.106.** Our Bemis-Murcko scaffold holdout puts most of the 30 epi-drugs in the
  cold-drug fold, so only 35 warm epi-drug rows survive the reproducible-stratum filter. **This is
  underpowered and is reported as an estimate with its interval, not as a result.** Widening it needs a
  split built for the purpose.
- Remaining unmatched: their subset is HDAC inhibitors specifically while ours is 30 epi-drugs
  (HDAC + DNMT + others), their exact split for that figure is not stated, and ours is one seed.
- The epi-drug advantage is consistent across every regime — **+0.13 to +0.20** over other compounds —
  which is the same effect [2.6] measured at +0.20. **Any published number computed on an
  epigenetic-compound subset should be read against an epigenetic-compound baseline, not a general one.**

## 40. 🔴 DIRECT HEAD-TO-HEAD ON IDENTICAL ROWS: XPert beats v9 by a wide margin (2026-08-29)

`model/v9/head_to_head_hdaci.py`. Their `reproducing/fig4/l1000_mdmt_HDACi.h5ad` ships 3,439 conditions
with `X`, `obsm['X_ctl']` and full metadata, alongside their `y_pred.npy` for exactly those rows. So both
models can be scored on **the same rows, against the same targets, from the same controls**. Their published
numbers reproduce exactly first (0.9804 / 0.8440 / 0.9200), which validates the artefact.

Median row Pearson, 95 % CI over 4,000 bootstrap resamples:

| subset | n | **XPert** | **v9 (ours)** | copy-the-control |
|---|---|---|---|---|
| all runnable rows — **delta** | 3,423 | **0.8569** [0.8533, 0.8596] | **0.5228** [0.5111, 0.5341] | 0 |
| all runnable rows — absolute | 3,423 | 0.9841 [0.9834, 0.9846] | 0.9580 [0.9567, 0.9593] | 0.9355 |
| pair NOT in our training set — delta | 2,419 | 0.8598 | **0.4860** [0.4724, 0.5002] | 0 |
| pair in our training set — delta | 1,004 | 0.8481 | **0.6186** [0.5860, 0.6415] | 0 |

- 🔴 **XPert wins decisively on its own benchmark: 0.857 against our 0.523 on the delta**, with
  non-overlapping confidence intervals by a wide margin. On the absolute convention the gap is much smaller
  (0.984 vs 0.958) because copy-the-control alone scores 0.9355 there — value added +0.049 for them,
  +0.023 for us.
- 🔴 **This reverses the optimistic reading of §39.** That section compared our-model-on-our-rows against
  their-model-on-their-rows and found 0.7963 vs 0.8440 — close. Run on identical rows the gap is 0.33.
  **Two separately-measured numbers being similar is not a model comparison**, and this project has now made
  that mistake in both directions.
- The asymmetry is real and is stated rather than used as an excuse: **their model is in-distribution here
  and ours is transferring.** These rows come from the corpus XPert trained on, and this is their own
  figure; v9 was trained on our Level-3 substrate and has never seen their aggregation. The training-overlap
  split shows exactly that effect — where the (cell, compound) pair WAS in our training set we score
  **0.6186**, where it was not we score **0.4860**, a difference of +0.133.
- Still unmatched: 16 rows use a compound we cannot featurise, 1,159 use a cell line we have no chromatin or
  lineage for, and 15.6 % of their rows pool multiple doses into one condition (median 1, max 8) where ours
  never do.
- ⇒ **The defensible claim is narrow: on their benchmark, under transfer, v9 reaches 0.49–0.62 delta where
  XPert reaches 0.86.** Closing that would require training v9 on their corpus with their preprocessing —
  which §36.1 shows their release does not permit without reconstructing inputs they did not ship.

## 41. Running XPert's own code and weights: what their release does not ship, and four traps inside it (2026-08-30)

§36.1 recorded that their release ships a `processed_data/` directory containing one `gitkeep.txt`, so
their code cannot be run as released. That closed the door on the only comparison with no confound left in
it — **their model and ours, on the same rows**. This section reopens it, and reports what had to be true
for the comparison to mean anything.

### 41.1 Retrieving the missing assets without downloading 1.6 GB, on a disk with 28 GB free

`model/v9/fetch_xpert_assets.py`, `model/v9/fetch_xpert_unimol.py`. The assets are in Zenodo record
`10.5281/zenodo.17182939` (the DOI in the paper, `15357711`, redirects there). Zenodo honours HTTP range
requests, so nothing is downloaded whole:

- the zip's **central directory** (~1 KB at the end of a 1.6 GB archive) is read first, and only the needed
  members are range-fetched and inflated — **714 MB instead of 1.6 GB**, skipping `l1000_sdst_78453.h5ad`
  (759 MB) and the KPGT/morgan drug features their unimol-trained checkpoint never reads;
- `all_drugs_unimol_arr.npy` is `(8981, 122, 514)` float64 = **4.5 GB**, but a `.npy` is a short header
  followed by one contiguous C-order block, so the 1,970 drug rows this benchmark touches are fetched by
  **computed byte offset** — 1.1 GB instead of 4.5 GB.

Integrity is verified rather than assumed. Every zip member is CRC32-checked against the value in the
central directory before it is put in place. The array row offsets are checked three ways: a row re-fetched
through a differently-aligned range must be byte-identical; column 0 is a padding mask, so every row must be
a run of 1s then 0s; and `mask_len - AddHs(mol).GetNumAtoms()` must be the **same integer for every
molecule** (it is +2, Uni-Mol's two special tokens). The first version of that third check compared against
the *heavy*-atom count and fired on correct data — the offsets were right and the assumption was wrong.
**Constancy is the invariant; the value is not.**

### 41.2 What the release does contain: their MAIN benchmark, which is not the file §28 audited

`l1000_mdmt_68830_subset.h5ad` — 68,830 conditions x 978 genes, **40 cell lines, 1,977 compounds** — carries
the split families the field quotes:

| family | folds | sizes |
|---|---|---|
| `split_1..5` (warm) | 5 | 55,064 train / 13,766 test, an exact 80/20 |
| `split_cold_cell_1..5` | 5 | 47,509-58,737 train |
| `split_cold_drug_1..5` | 5 | 54,493-55,561 train |

The five warm folds **partition** the corpus — every row is test in exactly one fold, verified in
`test_xpert_compare.py`. This is the corpus `l1000_mdmt_warm_split.pth` is named for. The fifteen **tissue**
splits §28 audited come from a different file (`l1000_mdmt_full_336852.h5ad`) and are not this benchmark.

### 41.3 Four things that would each have produced a plausible, wrong number

1. **Gene axis.** Verified identical to ours *entry by entry* against their own `l1000_gene_info_978.csv` —
   978 genes, same order, no remapping. This one passed, but it was checked, not assumed.
2. **A non-default architecture flag.** Their released weights contain `cls_token` and `class_fc`, which
   `XPertNet` only builds under `--include_cell_idx True`. That is **not** the argparse default. Building
   the model from defaults gives a different forward pass; `load_state_dict(..., strict=True)` is now
   mandatory in our driver so a mismatch is an error rather than a number.
3. **Flash attention is their DEFAULT path, not an optional speedup.** Their `model_utils.py` branches
   `if output_attention: <dense> else: <flash_attn_func>`, so the ordinary forward takes the flash branch —
   and the dense branch additionally **adds an attention mask that the flash branch never receives**. The
   two are therefore *not* interchangeable, and running on CPU by flipping `output_attention=True` would
   have silently changed the drug branch. FlashAttention is an *exact* algorithm, so the fix is a dense
   re-implementation of the same function (`model/v9/_shims/flash_attn/`), checked against a hand-written
   reference to 5e-7 and checked to *differ* from the masked branch.
4. **Their metric is the MEAN of per-row Pearson; ours has always been the median.** On these data the
   median flatters by ~0.02-0.05. `xpert_arm.py` now reports their convention first and ours alongside.

A fifth, in the data rather than the code: **18.9 % of their benchmark rows pool 2-8 distinct doses into one
condition** (`pert_dose` = `'0.12;0.04;0.01'`). Their model never sees that — `MyDataset` reads
`pert_dose_idx`, which is single-valued — so v9 is given a per-bin representative dose and no finer, a
bijection from the bin index that carries no extra information. The pooled rows stay flagged so any result
can be stratified on them.

### 41.4 🔴 Their published HDACi figure cannot be reproduced from any released checkpoint

`model/v9/xpert_native_eval.py`. Their `reproducing/fig4/hdaci_predict/y_pred.npy` is the artefact §39 and
§40 were read against, and its rows are in the same order as the h5ad, so their checkpoint can be run on
exactly those rows. **None of the three released mdmt checkpoints reproduces them.** Mean of per-row Pearson,
their convention, all 3,439 rows:

| predictor on the 3,439 HDACi rows | Pearson (abs) | Pearson_deg |
|---|---|---|
| their released `y_pred.npy` | 0.9804 | **0.8440** |
| `l1000_mdmt_warm_split.pth` (their released warm checkpoint) | 0.9589 | **0.6444** |
| `pretrain_mdmt_full_200_epoch.pth` | 0.9711 | **0.7610** |
| `pretrain_mdmt_new.pth` | 0.9678 | **0.7297** |
| copy-the-control | 0.9200 | 0 |

The reason shows up when the rows are split by whether they are in the 68,830-row benchmark corpus at all
(1,136 are, 2,303 are not):

| HDACi rows | n | warm checkpoint | their released `y_pred` |
|---|---|---|---|
| **in** the benchmark corpus | 1,136 | 0.7973 | 0.8496 |
| **not** in the benchmark corpus | 2,303 | **0.5689** | **0.8413** |

**Their released predictions are as accurate outside the benchmark corpus as inside it; the warm checkpoint
loses 0.23 crossing that boundary.** That is the signature of a model whose training corpus contained those
rows — the full 336,852-condition file, which does contain them — not of a model generalising to them.

This is not our driver mis-scoring. On corpus rows using the same 30 HDACi compounds, the warm checkpoint
scores **0.7974 on `split_1` TRAIN rows and 0.7968 on `split_1` TEST rows** — no memorisation gap at all,
so the driver is measuring generalisation, not fit, and it reports ~0.797 for their model on held-out rows
of their own benchmark.

- ⇒ **§40's 0.8440 is a number of unknown training provenance.** §40 is not retracted — v9 really does
  score 0.5228 on those rows, and copy-the-control really is 0.9355 on the absolute convention — but the
  XPert column there should not be read as a held-out result, and the sentence "their published numbers
  reproduce exactly, which validates the artefact" validates only the *arrays*, not the *evaluation*.
- ⇒ The comparison that survives is their released checkpoint and ours on the held-out rows of their
  own published split — but §42 shows that split must be `split_2`, the only fold their
  checkpoint did not train on.

## 42. 🔴 Which fold their released checkpoint was trained on — and what it scores when that is respected (2026-08-30)

`model/v9/xpert_native_eval.py --diagnose`, `model/v9/xpert_mdmt_baselines.py`. Their checkpoint is named
`l1000_mdmt_warm_split.pth` and their benchmark has **five** warm folds. Which one it was trained on is not
recorded anywhere in the release — and it decides whether any number measured on a given fold is a
held-out result or a partly in-sample one, because each fold's train set is 80 % of the corpus, so a fold's
test rows are ~80 % *inside* every other fold's training set.

Scoring the one checkpoint on all five folds answers it. Mean of per-row Pearson, their convention,
1,500 sampled test rows per fold:

| fold scored | Pearson | **Pearson_deg** |
|---|---|---|
| `split_2` | 0.9801 | **0.6939** |
| `split_5` | 0.9832 | 0.7384 |
| `split_4` | 0.9825 | 0.7423 |
| `split_3` | 0.9824 | 0.7434 |
| `split_1` | 0.9825 | 0.7435 |

**`split_2` sits 0.045–0.050 below the other four, which cluster within 0.005 of each other.** That is the
shape contamination makes: one genuinely held-out fold, four whose test rows the model largely trained on.

The sweep was run twice, on **independent random samples and different hardware** -- n=1,500 on the RTX
3050 (above) and n=1,200 on CPU. The CPU pass gives `split_2` 0.6959 against 0.7396 / 0.7415 / 0.7436 /
0.7450, and the CPU `split_1`-train control gives 0.7267 against the GPU's 0.7292. The ordering and the
size of the gap replicate.

### 42.1 The control that makes the inference safe

A lower score could in principle mean `split_2` is simply a harder fold. It is not, and this is measured
rather than argued: a **closed-form ridge refitted from scratch on each fold** — which by construction has
seen no test row of either — scores essentially the same on both.

| | fitted per fold | `split_1` delta | `split_2` delta |
|---|---|---|---|
| ridge | yes | 0.6054 | 0.6062 |
| **XPert released checkpoint** | **no, one fixed checkpoint** | **0.7403** | **0.6932** |

A model refitted per fold sees no difference between the folds (+0.0008). The fixed checkpoint sees
+0.047. ⇒ **the released warm checkpoint was trained on `split_2`**, and `split_2` is the only fold on
which it can be scored honestly.

### 42.1a The contamination model, confirmed quantitatively

If the checkpoint trained on `split_2`'s train rows, three predictions follow, and all three hold:

| rows scored | seen by the checkpoint? | predicted | **measured Pearson_deg** |
|---|---|---|---|
| `split_2` **train** | yes, all | high | **0.7421** |
| `split_2` **test** | no, none | low | **0.6932** |
| `split_1` **train** | ~80 % (it is the corpus minus `split_1` test, and one fifth of that is `split_2` test) | 0.8 x 0.7421 + 0.2 x 0.6932 = **0.732** | **0.7292** |

The memorisation gap is **+0.049**, and the mixed fold lands within 0.003 of the value the mixture
predicts. This also explains why the earlier train-vs-test check in §41.4 showed nothing: it compared
`split_1` train against `split_1` test, and **both** are inside `split_2`'s training set, because the five
folds partition the corpus — so `split_1` test is entirely contained in `split_2` train. That check was
sound but blind by construction; only `split_2` test is outside.

### 42.2 Their model's honest number on their own benchmark, with the nulls attached

Full `split_2` test set, n = 13,766, their metric:

| predictor | Pearson (abs) | **Pearson_deg** | delta over copy-the-control (abs) |
|---|---|---|---|
| copy the control | 0.9592 | 0 by construction | — |
| mean drug delta | 0.9621 | 0.2203 | +0.003 |
| ridge | 0.9747 | 0.6062 | +0.016 |
| **XPert, released checkpoint** | **0.9797** | **0.6932** | **+0.021** |

- **XPert's held-out delta on its own benchmark is 0.693, and a ridge on [control, ECFP4, descriptors,
  log dose, time] reaches 0.606.** The published absolute Pearson of ~0.98 is dominated by the control:
  copying it unchanged scores 0.959, so the model's value-added on that convention is **+0.021**.
- The same three baselines on the other two regimes, for reference (`split_cold_cell_1` n=21,321,
  `split_cold_drug_1` n=13,445):

| split | copy-ctl (abs) | mean-drug (delta) | ridge (delta) |
|---|---|---|---|
| `split_1` warm | 0.9597 | 0.2211 | 0.6054 |
| `split_cold_cell_1` | 0.9571 | 0.1105 | **0.2951** |
| `split_cold_drug_1` | 0.9588 | 0.0778 | **0.5295** |

  Cold-**cell** is far harder than cold-**drug** for a linear model (0.295 vs 0.530) — **the same asymmetry
  this project reported from its own data [CLAIMS 1.5, "cell-specificity, not chemistry, is the hard
  part"], now reproduced on an external benchmark with someone else's splits.**

### 42.3 How warm their warm split actually is

On `split_1`, of the 13,766 held-out rows: **100.0 % use a cell line seen in training, 99.9 % a compound
seen in training, and 99.8 % have the exact (cell, compound) PAIR in the training set.** Only the specific
dose/time condition is unseen — and 18.9 % of rows pool doses in the first place [§41.3]. This is the
regime their headline is quoted in, and §28 found the same for their tissue splits (89.4 % of pairs seen).

### 42.4a Two per-cell facts that constrain how our own numbers may be read

Scoring XPert's predictions per cell line on `split_2` (25 cells with >= 30 test rows):

- **Accuracy is flat in per-cell training volume**: Spearman(training rows for that cell, per-cell
  Pearson_deg) = **0.094**. A cell with 159 training rows (HEC108, 0.6037) is not systematically worse than
  one with 8,776 (MCF7, 0.6855). That is what a 99.8 %-pair-seen warm split should look like — per-cell data
  volume barely matters when the exact (cell, compound) pair is already in training.
- 🔴 **The cells we have chromatin for are EASIER than the ones we do not** — on XPert's own predictions,
  which use no chromatin at all: **0.7050 over the 18 covered cells vs 0.6744 over the 7 uncovered**. This
  is a property of which cells our panel happens to cover, not evidence about chromatin. It means any v9
  number computed on the chromatin-covered subset is flattered by ~0.03 relative to the full set, and a
  chromatin ablation must be run **within** the covered cells, never by comparing covered against uncovered.

### 42.5 A large part of the residual error is target noise, not model error

`model/v9/replicate_noise_mdmt.py`. Their benchmark ships no noise ceiling and their metric reports none,
but every row carries `n_replicates` -- the number of wells averaged into that condition. Averaging k wells
cuts the target's noise by ~sqrt(k), so a noise-limited score must rise with k. It does:

| replicate wells | rows | XPert Pearson_deg | mean abs delta |
|---|---|---|---|
| <= 2 | 4,428 | 0.6771 | 0.477 |
| 3 | 3,640 | 0.6925 | 0.360 |
| 4-5 | 2,455 | 0.6882 | 0.334 |
| >= 6 | 3,243 | **0.7198** | 0.291 |

**That raw column understates the effect**, because the two variables are confounded in opposing
directions: more-replicated conditions have systematically WEAKER measured effects (mean abs delta 0.477 ->
0.291), and §42's quartile table shows weak effects are harder. Controlling for effect size roughly
doubles it -- Spearman(replicates, Pearson) goes from **0.1438 raw to 0.3017 within-quartile**:

| effect-size quartile | n<=2 | n=3 | n=4-5 | n>=6 |
|---|---|---|---|---|
| Q1 (weakest) | 0.6145 | 0.6586 | 0.6551 | 0.6802 |
| Q2 | 0.6599 | 0.6758 | 0.6680 | 0.7269 |
| Q3 | 0.6642 | 0.6915 | 0.7032 | 0.7675 |
| Q4 (strongest) | 0.6983 | 0.7585 | 0.7784 | **0.8372** |

- **On the best-measured stratum -- strongest effect quartile, at least 6 replicate wells (n=330) -- XPert
  reaches 0.8372**, against 0.6932 over the whole test set. The headline number is depressed by noise in
  the LABEL, not only by model error.
- This is the same argument this project has made from its own data (Level-3 delta self-agreement 0.5283
  [§27.3]), now demonstrated on an external benchmark with an external model, using only a column that
  benchmark already ships.
**The warm regime is noise-limited; the cold-cell regime is not.** Repeating the same analysis on the
ridge's `split_cold_cell_1` predictions inverts the pattern: Spearman(replicates, Pearson) is **-0.2338
raw and only +0.0509 within effect-size quartiles**, against +0.1438 / +0.3017 on the warm split. On unseen
cell lines, accuracy barely tracks how well the label was measured, because the dominant error is no longer
in the label — it is the model failing to generalise to a cell it has never seen. Its best-measured
stratum reaches only **0.5265** (against 0.8372 warm).

⇒ **This is where modelling effort actually has room to work.** On the warm split a better model can only
chase a shrinking noise-limited margin; on cold-cell there is real, unclaimed signal. It is also why §44
tests the chromatin claim there rather than on the warm split.

- ⇒ **Any delta Pearson on this benchmark, ours or theirs, should be read as noise-limited.** A model
  comparison is still valid -- both models face the same labels -- but "0.69 vs 0.61" understates how much
  of the gap to 1.0 is unreachable.

### 42.4 Two checks on the apparatus itself

- **Device independence.** The 13,766-row `split_1` evaluation was run on CPU and again on the RTX 3050:
  max |CPU − GPU| over 13,766 × 978 predictions is **1.1e-05**, mean 1.3e-07, and Pearson_deg agrees to six
  decimals (0.740281 both). The flash-attention stand-in and the rest of the driver are numerically
  device-independent. GPU is ~12× faster (0.0125 vs 0.15 s/row), which is what makes the five-fold sweep
  affordable.
- **Input coverage on this benchmark is good, but the two kinds of cell input are not the same number and
  must not be quoted as one.** Of their 40 cell lines, **33 are in our cell index** (so carry a lineage
  vector) but only **23 have any real chromatin track**. By rows of the `split_2` test set: **97.3 % have a
  known cell line, 80.7 % have actual chromatin.** Drug features cover **98.81 %** of rows. The 164 rows we
  cannot featurise are dropped from both sides, and the paired comparison refuses to run below 95 % overlap.
  This is still far better than the tissue splits (§28: 143 of 217 cells had neither), and 80.7 % is the
  figure any claim about the chromatin branch on this benchmark has to be read against.

## 43. v9 vs XPert on the fold NEITHER model has seen — v9 ahead, and not because of chromatin (2026-08-30)

`model/v9/xpert_arm.py`, `model/v9/head_to_head_mdmt.py`. v9 trained on `split_2`'s training rows — the
only fold their released checkpoint did not train on [§42] — and both models scored on the same 13,615
held-out rows, with their metric and their prediction convention.

**Three seeds: 0.7051 / 0.7055 / 0.7052, spread 0.0004.** A 36-epoch run of the same arm reaches
**0.7135**, so the 12-epoch figure below is a floor, not a peak.

| `split_2` test, n = 13,615 | absolute Pearson | **delta Pearson** |
|---|---|---|
| copy the control | 0.9591 | 0 by construction |
| mean drug delta | 0.9621 | 0.2203 |
| ridge | 0.9747 | 0.6062 |
| **XPert, their released checkpoint** | 0.9796 | **0.6933** [0.6915, 0.6951] |
| **v9 (ours, 3 seeds, 12 epochs)** | 0.9803 | **0.7053** [0.7035, 0.7070] |
| v9, same arm at 36 epochs (1 seed) | 0.9807 | **0.7135** |

Paired on identical rows: **+0.0120 [0.0113, 0.0127]** in v9's favour, ahead on **67.1 %** of rows,
Wilcoxon p ~ 0, confidence intervals disjoint.

**The training budget runs strongly in their favour and the gap still holds.** Their checkpoint is epoch
164; v9's headline is **12** epochs, a complete run at that budget because the schedule anneals to zero.
Tripling it to 36 adds +0.008 (0.7053 -> 0.7135), so v9 is still improving where their model has long
since stopped. Whatever separates the two models, it is not that v9 was given more optimisation.

### 43.1 The verdict does not depend on the metric

A lead that exists only under the metric the other paper happens to report is not a result. v9 is ahead on
all eight:

| metric | XPert | v9 | better |
|---|---|---|---|
| Pearson_deg, mean (their metric) | 0.6933 | 0.7051 | v9 |
| Pearson_deg, median (our old convention) | 0.6981 | 0.7082 | v9 |
| Pearson_abs, mean | 0.9796 | 0.9803 | v9 |
| Spearman_deg, mean | 0.6088 | 0.6316 | v9 |
| MSE_deg (lower better) | 0.2001 | 0.1921 | v9 |
| MAE_deg (lower better) | 0.2761 | 0.2651 | v9 |
| Precision@100 up-genes | 0.5003 | 0.5162 | v9 |
| Precision@100 down-genes | 0.5202 | 0.5355 | v9 |

v9 also leads in every effect-size quartile (Q1 0.6801 vs 0.6661 … Q4 0.7427 vs 0.7321) and in both dose
strata. `MSE_deg` and `MSE_abs` come out identical for both models — the expected consequence of both
anchoring the absolute prediction as control + delta, and a free consistency check on the comparison.

### 43.2 🔴 NEGATIVE: the lead is NOT attributable to the chromatin branch

The obvious hypothesis is that v9 wins because it sees chromatin and XPert does not. **It does not survive
its own test.** XPert uses no chromatin at all, so it serves as a per-cell difficulty control: if chromatin
were driving the margin, the margin would be larger on the cells we have chromatin for.

| cells | rows | XPert | v9 | **v9 margin** |
|---|---|---|---|---|
| with a chromatin track (18 of 40) | 10,980 | 0.6939 | 0.7057 | **+0.0118** |
| without | 2,635 | 0.6906 | 0.7036 | **+0.0130** |

**Difference in margin: −0.0011, 95 % CI [−0.0029, +0.0007]** (and see §44: this warm-split test is
weak by construction, because 99.8 % of these rows have their (cell, compound) pair in training; §44 runs
the test on unseen cell lines instead, and agrees) — spans zero, and bounds any chromatin-driven
difference at under 0.003. This is consistent with the project's own repeated finding that the epigenetics
benefit does not transfer across cells [CLAIMS 2.5, 2.6].

Stated limits: cells "without chromatin" still receive a lineage vector and a zeroed chromatin input with
its mask, so this is an observational split rather than an ablation; and the arm does not save a checkpoint,
so the project's standard ablate-to-the-mean test could not be run on these weights. A definitive answer
needs a retrain with the branch removed, or a saved checkpoint to ablate. **What can be said now is that
v9's advantage over XPert on their benchmark is not explained by the input XPert lacks.**

### 43.3 What is and is not claimed

- **Claimed:** on XPert's own published benchmark, on the only fold their released checkpoint did not train
  on, evaluated with their code, their metric and their prediction convention on identical rows, v9 scores
  higher than their released checkpoint by +0.0118 delta Pearson, on every metric tried.
- **Not claimed:** that v9 is a better model in general. This is one benchmark, one warm split, and their
  split is extremely warm (99.8 % of test rows share a (cell, compound) pair with training [§42.3]).
- **The budget asymmetry runs in THEIR favour, not ours:** their checkpoint is epoch 164 (their config
  allows 2,500 with patience 50); v9 gets **12**. v9's schedule anneals to zero, so 12 epochs is a complete
  run at that budget rather than a truncated one, and a 36-epoch probe is running to bound the effect.
- Both models use inputs the other lacks — v9 has chromatin (80.7 % of rows) and lineage, XPert has a
  heterogeneous-graph drug embedding and a cell-identity auxiliary loss. Each is the model as its authors
  designed it.
- 151 test rows (1.1 %) use a compound we cannot featurise and are dropped from **both** sides; the
  comparison refuses to run below 95 % row overlap.

## 44. 🔴 Does chromatin add anything beyond the cell's own baseline expression? Measured: no (2026-08-30)

`model/v9/xpert_mdmt_baselines.py --with_chromatin --lam_sweep`. Chromatin conditioning [CLAIMS 7.1] is the
last surviving novelty claim of this project — 7.3 was falsified the same day [§41]. §43.2 reported no
chromatin effect on the warm split, but **that test was close to worthless**: 99.8 % of warm test rows share
a (cell, compound) pair with training [§42.3], so the cell's own response *to that very drug* is already in
the training data and a chromatin prior is redundant by construction. The claim was tested in the one
regime where it cannot be true.

The regime where it must earn its keep is **unseen cell lines**, where a model has no response history for
the cell and must characterise it some other way. `split_cold_cell_1` holds out 8 cell lines with **zero**
overlap with the 32 training cells; we have chromatin for 5 of the 8, covering **94.3 % of its test rows**.

### 44.1 The question in its sharpest form

A ridge already receives **the cell's own baseline expression profile** (`x_ctl`, 978 genes) as an input.
That profile is itself a rich cell descriptor. So the question is not "does chromatin describe a cell" —
obviously it does — but **does it add anything the baseline transcriptome does not already carry?**

A closed-form ridge answers exactly that, with no seed noise and no training-budget confound. Two design
points make the answer trustworthy rather than an artefact of encoding or regularisation:

- **The chromatin block is encoded EXACTLY, not summarised.** A per-cell feature block over 40 cell lines
  has rank <= 40, so the 3,912 raw chromatin dimensions (978 genes x 3 tracks + a 978-gene availability
  mask) were reduced to their **24 exact SVD components**, reconstruction verified to 1e-3. Nothing the
  ridge could have used was discarded; the reduction only makes the design matrix tractable.
- **The penalty is swept**, because a null for an added feature block can always be L2 shrinking it away.
  The Gram matrix does not depend on lambda, so each extra value costs one solve.

### 44.2 The result: nothing, in all three regimes

Best lambda taken **per arm**, which favours the chromatin arm:

| split | chromatin OFF | chromatin ON | gain |
|---|---|---|---|
| `split_2` (warm) | 0.6062 | 0.6064 | **+0.0002** |
| `split_cold_drug_1` (unseen compounds) | 0.5295 | 0.5298 | **+0.0003** |
| `split_cold_cell_1` (**unseen cell lines**) | 0.2951 | 0.2980 | **+0.0029** |

The lambda sweep on the cold-cell split, where the effect should be largest:

| lambda | OFF | ON |
|---|---|---|
| 1e2 | 0.2559 | 0.2576 |
| 1e3 | 0.2942 | **0.2980** |
| 1e4 | **0.2951** | 0.2969 |
| 1e5 | 0.2354 | 0.2367 |

The null holds at every penalty. For scale: the gap between a ridge and XPert's released checkpoint on the
warm split is **0.087** [§42.2], thirty times the largest chromatin gain measured here.

- ⇒ **Given the cell's baseline expression profile, cell-line chromatin adds at most ~0.003 delta Pearson
  to a linear model — including on cell lines never seen in training.** Three regimes, an exact encoding,
  and a swept penalty all agree.
- ⇒ Together with §43.2 (deep model, warm split: chromatin-covered vs uncovered margin differs by
  −0.0011 [−0.0029, +0.0007]) this is **two independent lines of evidence that the epigenetic input is not
  carrying accuracy** in this project's current form.

### 44.3 What this does NOT settle, and the test that would

A ridge can only use chromatin linearly and additively. v9 uses it as a **per-gene embedding summed into
the gene representation** [handoff §D.4] plus a signed additive head, which is a different functional form
and could in principle extract something a linear model cannot. That test — v9 trained on
`split_cold_cell_1` with and without `--ablate_epi`, 3 seeds each — is written, guarded and queued, and is
**blocked on Kaggle's weekly 30 h GPU quota being exhausted**. `--ablate_epi` mean-ablates the chromatin
values *and* the track-availability mask (E std across rows 0.5722 -> 1.4e-04, verified) while leaving
architecture, parameter count and the lineage input untouched, so it isolates chromatin rather than cell
identity.

Until that runs, the defensible statement is the narrow one: **the chromatin branch has not been shown to
contribute accuracy, and two measurements say it does not.** The claim must not be made in a paper on the
strength of the architecture containing the branch.

## 45. The chromatin branch, ablated in the regime where it must matter: real, significant, and small (2026-09-01)

`model/v9/xpert_arm.py --ablate_epi`. §44 answered the chromatin question for a ridge and found nothing.
This is the deep-model version, on the same split, and it is the test CLAIMS 7.1 has needed since July:
**v9 trained on `split_cold_cell_1` twice, identical in every respect except that one arm's chromatin input
carries no cell-specific information.** Eight held-out cell lines, zero overlap with the 32 training cells,
5 of the 8 with chromatin covering 94.3 % of test rows.

`--ablate_epi` replaces the chromatin values **and** the track-availability mask with their training means,
so architecture, parameter count, and the lineage input are untouched — the ablation removes the
information, not the machinery. Verified before the run: E's standard deviation across rows falls
0.5722 -> 1.4e-04 while lineage still varies and the targets are byte-identical.

| `split_cold_cell_1`, n = 21,151 paired rows, 12 epochs, seed 0 | delta Pearson |
|---|---|
| copy the control | 0 by construction |
| mean drug delta | 0.1101 |
| ridge (no chromatin) | 0.2951 |
| ridge (with chromatin) | 0.2980 |
| **v9, chromatin ABLATED** | **0.4692** |
| **v9, chromatin ON** | **0.4734** |

**Paired difference +0.0042, 95 % CI [+0.0036, +0.0049], chromatin better on 55.8 % of rows, Wilcoxon
p ~ 0.**

### 45.0 The number that dwarfs it: v9 on unseen cell lines

Before reading the chromatin effect, the scale it sits inside. Paired against the ridge on the same 21,151
rows, `model/v9/head_to_head_mdmt.py`:

| `split_cold_cell_1` | absolute Pearson | delta Pearson |
|---|---|---|
| ridge | 0.9584 | 0.2959 [0.2941, 0.2978] |
| **v9 (chromatin on, seed 0)** | 0.9665 | **0.4734** [0.4716, 0.4754] |

**Paired +0.1775 [0.1760, 0.1791], v9 better on 93.3 % of rows, Wilcoxon p ~ 0**, and ahead on all eight
metrics and in every effect-size quartile (Q1 0.4127 vs 0.2124 ... Q4 0.5287 vs 0.3903).

Set against v9's **+0.012** over XPert on the warm split [§43], this is fifteen times larger. That is what
§42.5 predicted: the warm regime is noise-limited and bunches every model near the ceiling, while cold-cell
is model-limited and leaves real signal on the table. **Unseen cell lines are where this architecture
distinguishes itself, and they are the regime the field cares about.**

### 45.1 What this means, stated at the size it actually is

- **The effect is real.** The interval excludes zero by a wide margin on 21,151 paired rows, and it is
  ~10x the seed-to-seed spread measured for this arm elsewhere (0.0004 across three seeds, §43).
- **The effect is small.** v9 beats a ridge on this split by **+0.178**. Chromatin accounts for
  **+0.0042 of that — about 2.4 %.** Ablate it completely and v9 still reaches 0.4692, keeping 97.6 % of
  its advantage. As a fraction of the score itself it is 0.9 %.
- **Two model classes agree on the magnitude.** The ridge, given the same chromatin encoded exactly
  (rank-preserving SVD, penalty swept), gained **+0.0029** [§44]. The deep model, with a per-gene
  chromatin embedding and a signed additive head, gains **+0.0042**. The richer functional form buys
  almost nothing over the linear one — about a thousandth of a Pearson.
- ⇒ **The defensible claim is: cell-line chromatin conditioning produces a statistically detectable but
  practically negligible accuracy gain on unseen cell lines.** It is not the reason this model works, and
  it must not be the headline of a paper. What carries v9 on this split is everything else.

### 45.2 The regime matters, and this is the one that does

§43.2 looked for a chromatin effect on the warm split and found none. That was near-uninformative: 99.8 %
of warm test rows already have their (cell, compound) pair in training [§42.3], so the cell's own response
to that very drug is in the data and a chromatin prior is redundant by construction. Cold-cell is where a
model has no response history for the cell at all — and it is also the regime that is **model-limited
rather than noise-limited** [§42.5], so there was real room for chromatin to show itself. It showed
itself, faintly.

### 45.3 Limits, stated plainly

- **One seed pair.** This project's rule is three seeds before a difference is reported. The comparison is
  paired across 21,151 rows, which controls row-level variance but not training-run variance; the measured
  seed spread of 0.0004 makes a +0.0042 artefact unlikely, but two more seed pairs are required before this
  number is quoted as final. They are written and queued, blocked on Kaggle GPU quota.
- **Batch 8, not 48**, because 4 GB of VRAM will not hold batch 48 (§ commit log). Both arms share it, so
  the contrast is unaffected, but the absolute 0.4734 is not comparable to the batch-48 Kaggle runs.
- The ablated arm still receives the **lineage** vector. This isolates chromatin specifically; it is not a
  test of cell identity as an input.

## 46. 🔴 THEIR NUMBERS WERE PUBLISHED ALL ALONG — and reading them corrects three of our claims (2026-09-20)

`external/xpert/supplementary/` (gitignored). Retrieved from the public Springer static-content endpoint,
no authentication, ~30 seconds:
`https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-025-01165-w/MediaObjects/42256_2025_1165_MOESM{1,2}_ESM.pdf`

### 46.1 The claim that sent this project down a six-week detour

§36.3 states: *"The per-model absolute values live inside figure panels and are not extractable, so they are
not quoted here."* 🔴 **That is false.** They are in **Supplementary Table R8**, a plain text table in the
63-page Supplementary Information. Every number this project reverse-engineered from released prediction
arrays, checkpoint forensics and fold sweeps (§23, §30, §39–42) was published with the paper.

**METHOD RULE 8 (new): before reverse-engineering any published number, download the supplementary.**
This cost more time than the NaN quantiser [§32] and it was pure omission rather than a bug.

### 46.2 Their actual benchmark numbers — L1000_mdmt, fivefold CV mean ± sd, PCC on xdeg

Table R8. Our benchmark, our metric convention (mean of per-row Pearson), their splits.

| scenario | Mean | CIGER | PRnet | TranSiGen | **XPert** |
|---|---|---|---|---|---|
| warm-start | 0.236 | 0.525 | 0.352 | 0.635 | **0.688 ± 0.011** |
| cold-drug | 0.236 | 0.397 | 0.344 | 0.609 | **0.645 ± 0.008** |
| **cold-cell** | 0.224 | 0.236 | 0.195 | 0.293 ± 0.017 | **0.383 ± 0.027** |

### 46.3 🟢 §42 was a SUCCESSFUL REPRODUCTION, not an exposé

| | published | ours |
|---|---|---|
| XPert warm-start, L1000_mdmt | **0.688 ± 0.011** | their checkpoint on `split_2`: **0.6933** |

**We recovered their published warm-start PCC to within half a standard deviation**, by identifying which
fold their single released checkpoint belongs to and scoring it only there. That independently validates
the whole apparatus: the Zenodo range-fetch, the `strict=True` load under `--include_cell_idx`, the dense
flash-attention stand-in, and the mean-vs-median convention.

🔴 **Two framings in §42 are retracted as loaded.** Their Methods state plainly: *"all datasets are
strictly split using fivefold cross-validation."* They trained five models and released one. That the
released one is fold 2's is **expected and innocent**, not "the shape contamination makes". The surviving
content is a **usage note**: scoring their released checkpoint on `split_1/3/4/5` yields 0.738–0.745, which
is inflated, because those folds' test rows are inside `split_2`'s training set. Useful for anyone running
a head-to-head; not a criticism of the paper.

🔴 **§23/§39/§40's 0.8440 is retracted as a benchmark reference.** It is the Fig. 4 vorinostat/HDACi
**case study**, confirmed from the paper text. It was never their benchmark number. §36.2 caught half of
this; the rest of the project kept citing it for six weeks.

### 46.4 🟢 The comparison that now matters — cold-cell

| `split_cold_cell_1`, their metric, their split, xdeg | PCC |
|---|---|
| Mean baseline (published) | 0.224 |
| TranSiGen (published, next-best) | 0.293 ± 0.017 |
| **our ridge** (measured, §44) | **0.2959** |
| **XPert (published)** | **0.383 ± 0.027** |
| **v9 (measured, §45, 1 fold, 1 seed, batch 8)** | **0.4734** |

- 🟢 **v9 sits +0.090 above XPert's published cold-cell PCC — 3.3 of their own fold-to-fold sd.**
- 🟢 **Free calibration check:** our ridge scores 0.2959 where their published TranSiGen scores 0.293. Our
  apparatus is anchored to their scale independently of any claim we make.
- 🔴 **This is NOT yet a head-to-head.** It is our measured number against their published number — the
  exact form §39 got wrong. It is safer than §39 (their released fold, their metric, their task, their
  convention) but it is one fold of five, one seed, batch 8 on a 4 GB laptop GPU against a fivefold CV mean.
  **Do not quote it as a model comparison until §46.5 is done.**

### 46.5 The test that converts it into a head-to-head

They released **no cold-cell checkpoint** — only the warm one, which trained on those cell lines and cannot
be scored here. But §41 recovered every missing input and the shim is verified to 5e-7, so **XPert can now
be trained by us** on `split_cold_cell_1..5`. That yields both models on identical rows in the regime that
is model-limited rather than noise-limited [§42.5].

Required before any claim: **5 cold-cell folds × 3 seeds** for v9, and XPert trained on the same folds.
Same machinery admits TranSiGen / PRnet / DeepCE / CIGER, which have never been run here at all.

### 46.6 A fourth split we have never tested
Their Methods list **four** strategies for L1000_mdmt; the fourth is `cold-dose&time` (partitioning each
drug–cell pair by dose/time). We have nonlinear dose/time FiLM and have never tested it [M.4].

## 47. 🔴 THE HANDOFF'S "top-k sparse vs dense" IS DEAD CODE — and the real difference is drug self-attention (2026-09-20)

Read from the executed code path in `external/xpert/code/XPert/models/model_utils.py` and
`models/model_XPert.py`, not from the config or the constructor signatures.

### 47.1 The claimed difference does not exist

V9_HANDOFF §C's architecture table asserts: *their* `width / heads` = "256 / 8, **top-k sparse attention
(128 cell, 32 drug)**" against *ours* = "256 / 8, **dense**". 🔴 **False, three times over:**

| check | finding |
|---|---|
| is `topk` stored? | `SelfAttention.__init__` and `CrossAttention.__init__` both **accept** `topk` and **never assign it to `self`**. `grep "self.topk"` in `model_utils.py` returns nothing. |
| is `sparse_flag` read? | It is threaded through **every** forward signature (`SelfAttention`, `CrossAttention`, `Encoder`, `crossEncoder`) and **never referenced in any attention body**. It is passed down the chain and dies. |
| what does their config say? | `configs/config_l1000.yaml`: `topk_cell: 128`, `topk_drug: 32`, and **`sparse_flag: False`**. |

Both attention branches are dense: a manual dense softmax when `output_attention=True`, and
`flash_attn_func` — which is exact, not approximate — otherwise. **XPert's attention is dense. Ours is
dense. There is no sparsity difference between the two models.**

This is the **ninth** instance of this project's characteristic failure: a claim derived from reading a
config and a constructor signature rather than the code that runs [method rule 5]. It came within one step
of shaping an architecture change — "add top-k sparse atom attention to match their config" — that would
have implemented a feature the reference model does not use, against a baseline that never had it.

### 47.2 🟢 The real difference: they contextualise the drug tokens, we do not

`crossEncoder.forward` (`model_utils.py:361`) runs, **in every cross-encoder block**:

```
drug_SA_embed, _ = self.drug_SA(drug, drug_attention_mask, ...)   # 1. drug tokens SELF-ATTEND
cell_attention_out_0, _ = self.attention(cell, ...)               # 2. gene tokens self-attend
cell_attention_output_1, _ = self.attention_CA(cell_embed, drug_SA_embed, ...)   # 3. genes attend over
                                                                  #    the CONTEXTUALISED drug
```

Ours (`model/v9/model_v9.py:125-127`) builds the drug side **once**, outside the block loop:

```
D = torch.cat([(ln_u(w_u(u)) + type_drug).unsqueeze(1),
               ln_atom(w_a(atoms)) + type_atom], dim=1)
```

— a global token concatenated with **independently linearly-projected per-atom Uni-Mol vectors** — and
hands that same `D` unchanged to every perturb block. **There is no drug self-attention anywhere in v9.
Our atoms never see each other.**

So our 978 gene queries cross-attend over a **bag of uncontextualised atoms** carrying no intramolecular
structure, while theirs attend over a molecule whose atoms have been mutually contextualised first.

### 47.3 Why this is the leading explanation for §37's atom-token result

§37 measured that **removing atom tokens IMPROVES accuracy** on all three splits
(−0.007 / −0.025 / −0.022) — flagged there as "the next deletion candidate". §47.2 supplies a mechanism:
uncontextualised per-atom vectors are noise injected into the gene stream, so deleting them helps.

⇒ **Do NOT delete the atom tokens. Test the contextualisation first.**
Single-factor A/B: add a drug self-attention encoder over the ~33 drug tokens before cross-attention,
change nothing else. Cheap — the drug sequence is two orders of magnitude shorter than the gene sequence.
Predictions, both falsifiable: (a) the atom-token ablation flips sign, from −0.025 to positive; (b) if it
does not, atom-level attribution in this architecture is dead and deletion is then justified **with a
mechanism attached** rather than as a bare empirical result.

### 47.4 A second difference in the same place, worth a separate arm
Their drug sequence is `[dose, time, HG_embed, atom_1..atom_n]` (`unimol_Embeddings`, `model_utils.py:133`)
— **dose and time are tokens INSIDE the drug stream**, so cross-attention can re-weight individual atoms
by exposure. Ours applies dose/time as a FiLM scale/shift on the gene tokens, far from the atoms. Same
information, structurally unable to express the same interaction. Test separately; do not bundle with 47.3.

## 48. How the 2026 SOTA actually uses graphs — and why our STRING null was predictable (2026-09-20)

Delegated to an `agy` worker (`research/W2_mechanisms_REPORT.md`, brief in the commit). **Its TxPert claims
were then verified by me directly against the arXiv full text** before any of this was written down; the
quoted strings below are from that verification pass, not from the worker's summary. State and PertAdapt
claims are recorded as REPORTED-NOT-VERIFIED and must be checked before they are cited.

### 48.1 TxPert (Nat Biotech 2026), verified against arXiv:2505.14919

| question | finding | quote |
|---|---|---|
| combining graphs | union of edges, with a **multi-hot edge feature** encoding which source graph each edge came from | *"Exphormer-MG, an extension of the Graph Transformer architecture adapted for multi-graph learning via a union graph methodology"* |
| which graphs | **STRINGdb, GO, PxMap, TxMap** — and all four together is best | *"STRINGdb, GO, PxMap and TxMap…when all four graphs were combined"* |
| graph conditioning | **static prior** over learnable node embeddings, independent of basal state | *"each perturbation p is associated with a randomly initialized input node embedding h_p⁰ … treated as model parameters learned via backpropagation"* |
| unseen cell lines | **no basal-state encoder**; predict a delta onto the raw control | *"no basal state encoder is by far the most effective option…the model predicts delta instead: ŷ = x + gφ(Σ z_p)"* |

### 48.2 🔴 This CUTS AGAINST my own "additive injection is the problem" hypothesis

TxPert's graph is **static**, operates on learnable node embeddings **independent of the cell**, and its
output is **summed** into the basal representation — and it is SOTA for OOD transfer to unseen cell lines.
So "static + additive" is not disqualifying on its own, and [§47]'s framing must not be over-generalised.

**The difference that does survive is WHAT the graph is used for:**

- **TxPert uses the graph to represent the PERTURBATION.** In genetic perturbation the perturbation *is a
  gene*, so a gene–gene graph directly answers "what does perturbing gene X do" through X's neighbourhood.
  The graph is doing causal work.
- **We use the graph to SMOOTH THE TARGET.** Our STRING step message-passes over gene representations,
  drug-invariantly. It answers no question about the perturbation at all.

⇒ That is a better explanation of our true null (−0.0003, \|dY\|max 0.5 [§37]) than "it is additive", and it
survives the fact that a static additive graph works elsewhere. **For a chemical perturbation the analogue
of TxPert's move is to propagate from the DRUG'S TARGETS outward through the graph**, not to smooth the
978 landmark genes. We hold `dti_reference.tsv` (19,174 edges, 1,718 drugs) as a held-out validation set,
so doing this costs us that validation set — an explicit trade to decide, not to make silently.

### 48.3 Two directly actionable findings

1. **Union multiple graphs with multi-hot provenance edge features.** We have STRING, Reactome, GO:BP and
   DTI and have only ever used them separately or not at all. TxPert's ablation says combining helps
   monotonically and all four beats the best three (p < 0.027). Exact per-graph numbers are
   **UNKNOWN — figure panel only** (worker checked the LaTeX source; values are not in text or tables).
2. **Test removing the control ENCODER on cold-cell.** TxPert found *"no basal state encoder is by far the
   most effective option"* for cross-cell-line transfer. v9 runs **two** control encoders (`ctl_enc`,
   `cell_enc`) mixed into the gene tokens. We already anchor absolute as control + delta [§43.1], so we are
   half-way there; the untested half is whether encoding the basal state at all hurts on unseen cells.
   Cheap ablation, directly on the split that matters.

### 48.4 🔴 REJECTED: the worker's novelty claim, as an example of the failure mode to watch for

The worker concluded that supplying auxiliary biology as *its own attended token set* is
**"a completely novel comparison against the current literature."** **I am not accepting that, and it is
recorded here as a caution rather than a finding.**

- It generalises from **three papers** — all genetic and all single-cell — to "the current literature".
- It is contradicted inside our own reference set: XPert carries `HG_embed` as a **token** in the drug
  sequence, and **our own named pathway layer is already a token set**.
- It is the flattering answer. The brief asked whether the literature addresses the PI's hypothesis, and
  the answer came back "your hypothesis is novel". Agent output that confirms the requester's prior is
  exactly where verification effort belongs.

**What IS defensible:** these three papers use auxiliary biology as an additive embedding (TxPert, State)
or a static attention mask (PertAdapt), and **none of them ablates tokens-vs-additive**. That is a gap in
the evidence, not an established novelty. Settling it needs W3-style adversarial search.

### 48.5 Provenance
`research/W2_mechanisms_REPORT.md`. TxPert: verified by me against arXiv:2505.14919 full text.
State (bioRxiv 2025.06.26.661135) and PertAdapt (PMC13341120): **REPORTED, NOT VERIFIED.** The worker read
a preprint for State and an XML extraction for PertAdapt; preprint and published versions can differ. Do
not cite either without a verification pass [method rule 8 family].

## 49. Adversarial novelty sweep on the chromatin claim: SURVIVES, but flanked — and the field has named it (2026-09-20)

`research/W3_novelty_REPORT.md`. An `agy` worker was briefed **to destroy** claim 7.1, not to defend it,
with the four known near-misses supplied up front so it could not waste effort re-finding them.
**I then verified every checkable assertion against primary sources.** The worker was accurate on all of
them — including the one that decides the verdict.

### 49.1 Verdict: CLAIM WEAKENED, not killed. No paper meets all four criteria.

Criteria a killer must meet: (1) **predicts**, not analyses; (2) output is a **continuous multi-gene
response profile**; (3) input is **measured prior chromatin state**; (4) perturbation is a **small molecule**.

| paper | status | criterion it fails | verified by me? |
|---|---|---|---|
| **Agrawal et al.**, *F1000Research* 2023/2025 (PMC12103705) | 🔴 **closest flank** | **(2)** — binary up-vs-down classification restricted to the **top 1000 up + top 1000 down** genes, not a continuous profile over unselected genes | ✅ **VERIFIED** — *"we first identified the top 1000 up-regulated and 1000 down-regulated genes"*, task is *"to distinguish up versus downregulated genes after HDACi-treatment"* |
| **BaiZe**, bioRxiv 2026 (10.64898/2026.07.15.738608) | flank | **(3×4)** — architecture accepts *optional* ATAC, but the chemical arm never receives it | ✅ **VERIFIED** — for Sci-Plex, *"BaiZe received control-state RNA, Morgan fingerprints derived from SMILES strings and treatment dose as inputs"*; ATAC assessed only *"in K562/RPE1 and human embryo tasks"* |
| **MultiFlow**, bioRxiv 2026 (10.64898/2026.08.20.746112) | adjacent | **(4)** — and it *predicts* ATAC as an output rather than conditioning drug response on prior chromatin | ⚠️ **PARTIAL** — preprint, title, authors, abstract verified; the "CRISPR-only" detail is **NOT verified** (abstract does not name datasets) |
| ExPO, GRIP-Lung, DEPICT, PrePR-CT | adjacent | **(3)** — no chromatin input at all | ✅ ExPO verified |

**Agrawal et al. is the paper a hostile reviewer will cite.** Pre-treatment H3K27ac in 21 genic bins →
direction of change under HDAC inhibitors, ROC AUC 0.71–0.89, cross-cell-line and cross-drug. Our claim
survives on four distinctions — continuous magnitude vs binary direction; 978 unselected genes vs the 2,000
most-changed; 40+ cell lines vs 2; three marks vs one — **but it must be cited and distinguished
pre-emptively, not discovered at review.**

### 49.2 🟢 Three independent 2026 groups have now named this direction as the open frontier

| source | what they say |
|---|---|
| **ExPO**, J Cheminform 2026 | ✅ **verbatim, verified:** *"further gains may require richer cell context (e.g., chromatin marks)"* — in their leave-cell-line-out discussion, i.e. **our exact regime** |
| latent-diffusion L1000, Bioinformatics 2026 | lists its own gap as *"additional cellular information like chromatin state or pathway activity"* [CLAIMS M.8] |
| **BaiZe**, bioRxiv 2026 | built the optional-ATAC socket into the architecture and **did not plug it in for drugs** |

This cuts **both** ways and both must be stated. It validates the direction — the field independently
converged on chromatin conditioning as the next thing to try. It also means **we are not alone here and the
window is closing**: BaiZe has the socket already built.

### 49.3 🟢 The reframe this licenses — and it is stronger than the original claim

Our chromatin effect is +0.0042 on unseen cells [§45]. Set beside §49.2 that stops being a disappointing
number and becomes a **result the field has asked for**:

> Three 2026 papers name chromatin conditioning as the missing cellular context for drug-response
> prediction. **We are the first to actually build it for chemical perturbation at scale — 40 cell lines,
> 978 genes, three marks, a proper cold-cell ablation — and we measure that it contributes
> +0.0042 [+0.0036, +0.0049], about 2.4 % of our margin over a ridge.**

That is a *useful* negative: it tells three groups the socket is not worth plugging in **in this form**,
with the measurement to back it. Far more defensible than the original framing, which claimed chromatin as
the contribution while the ledger said +0.004.

### 49.4 🟢 Agrawal supplies the lead for the one chromatin form we have NOT tried

They reach **AUC 0.71–0.89** — not nothing — by predicting **direction of change**, on the **extreme
responders only**, from H3K27ac. We predict **continuous magnitude for every gene** and get +0.004.

Those two facts are consistent, and together they suggest we have been asking chromatin the wrong question:

- chromatin may carry **which genes move and which way**, not **how much**;
- the signal may live in the **tails** (the genes that actually respond), which our
  reproducible-stratum filter keeps but our per-gene regression loss dilutes across all 978;
- our own [2.2/2.2a] measured exactly this — signed correlations of the right sign surviving all four
  baseline-expression quartiles — a **directional** finding we then fed into a magnitude head.

⇒ **New arm, cheap, and independent of the token-vs-additive question:** an auxiliary
**direction/sign head** conditioned on chromatin, scored as AUC on the top-k up vs top-k down genes —
Agrawal's task on our data and our scale. Its null is the same head with chromatin mean-ablated. If
chromatin carries direction but not magnitude, this is where it shows, and it is the one form of the
question this project has never put.

### 49.5 Worker reliability, recorded
W3 was accurate on **every** claim I could check, including the decisive BaiZe detail on which the verdict
turned, and it correctly placed Agrawal in *Partial* rather than inflating it to a kill. Contrast [§48.4],
where W2's errors were concentrated in a conclusion that flattered the requester's hypothesis. **Adversarial
briefs verify better than confirmatory ones** — worth carrying into every future delegation.
Open item: MultiFlow's perturbation-type detail still needs the full text.

## 50. 🔴 A May-2026 benchmark says SEVEN L1000 MODELS DO NOT USE THEIR DRUG FEATURES (2026-09-20)

Surfaced by an `agy` worker (`research/W1_lincs_competitors_REPORT.md`). **Verified by me against the
preprint full text**, which also caught one worker error (§50.5).

### 50.1 The paper

**"Deep learning models for chemical perturbation prediction do not yet utilise drug molecular features"**
— Jinming Bai, Sharon Prince, Geoff S. Nitschke. bioRxiv, 15 May 2026, doi `10.64898/2026.05.13.724458`.
Code: `github.com/baijinming97/drug-perturbation-benchmark` (✅ exists, last push 2026-05-13, verified via
`gh repo view`). Harness vendors all seven upstream repos at pinned commits.

Abstract, verbatim:
> *"We retrained seven such models from scratch with zeroed or shuffled drug inputs, and compared them with
> a multilayer perceptron that uses only cell-line basal expression. Under drug-blind evaluation, ablation
> caused negligible performance changes and the drug-free baseline matched all models. Current
> architectures do not yet utilise drug molecular features for generalisation to unseen compounds."*

| ✅ verified from full text | value |
|---|---|
| drug-free MLP (cell basal expression only), PCC_DEG | **0.637** |
| XPert, strongest full model, PCC_DEG | **0.633** |
| max ΔPCC_DEG from **zeroing** drug features, across all 7 models | **0.012** |
| max ΔPCC_DEG from **shuffling** drug features | **0.027** |
| Mean (global) null | 0.099 |
| Mean (cell-line) null | **0.243** |
| ablation procedure | *"All models were retrained from scratch under both ablation conditions"* |
| split | drug-blind only — *"test drugs were entirely absent from the training set"*; **cell-line generalisation was not evaluated as a separate condition** |

### 50.2 🔴 Which of our claims this threatens

| ours | status |
|---|---|
| [1.9] *"chemical generalisation is real and large"* — unseen-compound, v5 beats best linear by +0.089 | **THREATENED** |
| [5.1] *"ECFP4 and per-atom UniMol tokens are the two pillars"* | **THREATENED** |
| [§37] drug global features +0.151 / **+0.223** / +0.148 (ablate-to-mean) | **see §50.3** |
| V9_HANDOFF §B *"Drug features are the model"* (+0.25…+0.30) | **OVERREACH — must be rewritten** |

### 50.3 🟢 The distinction that decides this — and both measurements are valid

They are **not** measuring the same thing, and the difference is exactly this project's method rule 5:

- **Ours (§37) is an INFERENCE-TIME ablate-to-mean on a trained model.** It answers *"does the trained
  model USE this input?"* Our answer: heavily, +0.15…+0.22.
- **Theirs is a RETRAIN-FROM-SCRATCH ablation.** It answers *"does the TASK REQUIRE this input?"* Their
  answer: no — a from-scratch model compensates through cell-line expression pathways.

**Both can be true at once.** A trained model can lean on drug features while a drug-free model retrained
from scratch matches it, if the drug information is redundant with what basal expression plus the training
distribution already supply. ⇒ **§37 is not refuted, and it does not refute them.** But
*"drug features are the model"* claims the second thing on the strength of the first, and **that is an
overreach we have to correct before a reviewer does it for us.**

### 50.4 🟢 What this does NOT threaten — and the gap it opens for us

Their evaluation is **drug-blind only**. By their own methods, *"cell-line generalisation was not evaluated
as a separate condition."* So:

- Our cold-cell result [§45.0] — v9 **0.4734** vs ridge **0.2959**, +0.178 on 21,151 paired rows — is
  **outside their scope**. And our ridge baseline already carries ECFP4 + descriptors, so that margin is
  not a drug-feature artefact in the first place.
- Their `Mean (cell-line)` null of 0.243 **cannot even be computed in our headline regime**: on an unseen
  cell line there is no per-cell training mean to take. The baseline that dominates their benchmark is
  undefined on ours.

⇒ **The open question nobody has asked: do drug features matter for unseen-CELL generalisation?** Bai et al.
answered it for unseen compounds and explicitly did not test unseen cells. That is a clean, novel, cheap
experiment and it sits directly on our headline regime.

### 50.5 Required experiments, in priority order

1. **Bai's ablation on v9, their regime** — retrain from scratch on `split_cold_drug_1` with drug features
   zeroed. If v9 matches its full self, we inherit the field's problem and must say so. If it does not,
   v9 is the exception and that is a result worth the paper.
2. **Bai's ablation on v9, OUR regime** — the same retrain on `split_cold_cell_1`. Untested by anyone.
3. **Their drug-free MLP as a baseline on our splits.** It beat seven published models; it belongs beside
   our ridge in every table. Their harness is runnable, so this is cheap.
4. Rewrite V9_HANDOFF §B's *"Drug features are the model"* to state the inference-vs-retrain distinction.

### 50.6 Worker error caught by verification
W1 reported `Mean (cell) = 0.395`. The full text says **0.243**. The rest of its extraction
(0.637 / 0.633 / 0.012, repo status, the DEPICT 404) verified correct. Logged to
`orchestration/AGENT_REGISTRY.json`. **This is the second delegated batch in which a worker's numbers were
right and one figure was not** — the verification pass is not optional, and the error is never where the
worker sounds least confident.

## Open program (gated on: accuracy must be comparable for the interpretability story to carry weight)
1. **Diagnose interaction under-expression BEFORE any retrain** (`analyze.py`, running): is it noise-driven
   MSE shrinkage (→ correlation/rank loss) or dead cell-conditioning (→ architecture)? Test = does interaction
   expression improve on stronger/reproducible sigs?
2. **Noise-shielding** (arch and/or training) — likely a correlation/rank loss term + reliability-conditional
   weighting; decide after (1).
3. **Fair SOTA comparison** — get L1000 controls OR run PRnet on our differential split. Report both metrics.
4. **Unseen-COMPOUND number** — needs a **scaffold/Tanimoto-aware drug-holdout split + retrain** (current
   model saw all drugs; only cells were held out). Must match SOTA's split rigor (verify they enforce
   chemical dissimilarity; if not, note it; we should enforce it regardless — no leaky similar test drugs).
5. **Pathway→drug (crude MoA)** — add gene↔gene (pathway-prior) attention attribution to the interpretability
   eval; compose atom→gene + gene→pathway + epi→gene into an MoA readout.
6. **Drug-feature ablations** (`Strategy C`, cheap CPU) — which of descriptors/fingerprint/UniMol/ChemBERTa carry signal.

Sources: latent-diffusion PMC13107963 · XPert nature s42256-025-01165-w + github GSanShui/XPert · PERD
PMC11139989 · H3K27ac PLoS Comp Biol pcbi.1012272.
