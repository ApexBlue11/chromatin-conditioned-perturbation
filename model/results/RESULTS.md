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

### 47.5 IMPLEMENTED 2026-09-20 — built, tested, default OFF, not yet trained

`model/v9/modules_v9.py::_DrugBlock`, wired into `model_v9.py::PerturbBlock` behind
`config_v9.drug_self_attn` (**default False**, so the A/B changes exactly one thing and every existing v9
checkpoint still loads). The updated `D` is **returned** from each block, so contextualisation compounds
across blocks as it does in XPert, where `crossEncoder.forward` emits `drug_SA_embed` alongside the cell
output.

**Guards, written against [§32]'s lesson.** §32's quantiser test asserted `fitted == 1.0` — that `fit()`
had been *called* — while the bins were NaN and the expression input was dead. So `test_drugsa_v9.py`
does not check that the module exists; it checks that it **discriminates**:

| check | result |
|---|---|
| perturbing atom 3 moves atom 1 (**contextualisation**) | ✅ \|d\|max **0.2754** |
| with the arm off, atom 1 is bit-identical when atom 3 changes (**it really is a bag**) | ✅ exact |
| changing a **padded** atom does not move a real one | ✅ \|d\|max **0.00e+00** |
| shape preserved, output finite | ✅ |
| flag defaults False, switchable | ✅ |

`test_v9.py` still passes **55/55** with the arm off, so the change is backward-compatible. Both arms of
the full model build, run and produce finite, *different* predictions.

Incidentally, the integration attempt hit `BinnedExpression`'s §32 guard — it refused to run with
unfitted bins rather than silently bucketing everything to zero. **The fix from §32 is doing its job.**

### 47.6 🔴 A confound in this A/B that must be controlled BEFORE it is run

**The arm adds parameters: +65,736 at the reduced test width, +11.38 %.** 🔴 **CORRECTED 2026-09-20 — at FULL width the confound is ~3x larger: 10,466,725 → 13,612,741 params, +3,146,016 = +30.06 %** (`_DrugBlock` is 786,504 params x `l_perturb` 4). Measured by construction, not extrapolated. The reduced-width figure understated the confound threefold [packet 003]. If the contextualised arm wins,
"was it the contextualisation or the extra capacity?" is unanswered, and this project has already been
burned once by a comparison that changed two things at once [§33].

Required control, one of:
1. a **matched-capacity** arm that adds the same parameter budget to the gene side (where §37 says the
   representation is already load-bearing), so capacity is held constant and only *where* it sits varies; or
2. a **width-matched** arm: shrink `d_model` in the drug-SA arm until total parameter counts agree.

(1) is the more informative control, because it tests *placement* rather than *amount*.

⇒ **Do not launch the A/B until one of these is in place.** The prediction being tested is specific: with
contextualisation on, the atom-token ablation from [§37] should flip from **−0.025** to positive. That is
a within-run ablation, so it is immune to seed variance [method rule 7] and does not need 3 seeds — which
makes it a far cheaper decisive test than a headline accuracy comparison.

**Cost, measured rather than guessed (2026-09-20).** §29.1 gives Kaggle T4 x2, d_model 256, full depth, 12 epochs = **5.62 h** over 179,772 rows (1,636 s/epoch = 0.0091 s per row-epoch). So fold0 ≈ **5.6 h/run**, `split_cold_drug_1` ≈ 1.7 h, `split_cold_cell_1` ≈ 1.5 h, against a **30 h weekly quota**. A three-arm headline comparison at fold0 is ~16.8 h (56 % of quota) and would STILL need ≥3 seeds to clear the ±0.046 band — unaffordable and not decisive. **Proposed instead: ONE run with TWO within-run ablations** (the atom tokens, and the `drug_sa` module itself), measuring their interaction inside a single set of weights. **That also removes the need for the capacity control, since capacity is identical across both ablation arms** — capacity only confounds a headline accuracy comparison, which this design does not make. Sent to adversarial review as **packet 003 BEFORE any spend**.

**C5 discharged, with its limit stated (2026-09-20).** Review 003 C5 noted the 5.6 h figure scaled a rate
measured with the module OFF, so it was a floor. Measured: 50 optimiser steps at full width, both arms,
local RTX 3050, batch 8, 40 atoms.

| arm | s/step |
|---|---|
| `drug_self_attn=False` | 2.4636 |
| `drug_self_attn=True` | 2.4858 |
| **ratio** | **1.009x** |

Structurally this is expected: the drug sequence is ~40 tokens against 978 gene tokens, so attention added
on the drug side is a rounding error even at +30 % parameters.

🔴 **But the measurement is NOT a clean transfer and must not be quoted as the projected cost.** The
local step is **2.464 s** where Kaggle T4 x2 at batch 48 is **0.437 s/step** (1,636 s/epoch over 179,772
rows, §29.1) — **5.6x slower per step and 33.8x slower per row.** That is a memory-bound regime on 4 GB,
where a large fixed cost dominates and therefore **compresses** any ratio. The true ratio in the Kaggle
regime is plausibly larger.

⇒ Defensible statement: **the module's wall-time cost is structurally small and measured at 1.009x in an
unrepresentative regime; the definitive figure is the first epoch of the real run.** Budget the run at
~5.7 h with a stated risk that it is higher, and abort on the budget guard rather than on this estimate.

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

## 51. 🔴 ADVERSARIAL REVIEW 001: the chromatin claim is RETRACTED. 12 challenges, 12 upheld (2026-09-20)

`orchestration/bus/to_pi/001_review.md`. A blinded independent reviewer — given objective, function,
results and code, and **not** the PI's reasoning — returned `NOT-SUPPORTED` on packet 001.
**I verified its four most consequential challenges against the code and artefacts myself. All four hold.**
Nothing was rejected.

### 51.1 Adjudication

| # | sev | challenge | verdict |
|---|---|---|---|
| 1 | BLOCKING | row-bootstrap CI is the wrong uncertainty; 8 test cells, 78.8 % of rows from two; two runs at one seed each | **UPHELD** — §51.2 |
| 2 | BLOCKING | the central number has no generating script; the ON-arm artefact was overwritten | **UPHELD, verified** — §51.3 |
| 3 | MAJOR | ridge λ selected on test rows; at matched λ the ridge gains **more** than reported | **UPHELD, verified, changes the conclusion** — §51.4 |
| 4 | MAJOR | `x_cell` for cold cells is computed over that cell's **test** rows; the ridge has no such input | **UPHELD, verified** — §51.5 |
| 5 | MAJOR | 3 of 8 test cells have no chromatin track ⇒ a free placebo stratum estimating the noise floor | **UPHELD — best proposed experiment in the review** |
| 6 | MAJOR | ridge 0.2951 is n=21,321 while every v9 number is n=21,151; correctly paired value is **0.2959** | **UPHELD** — §45/§46.4 mixed both in one table |
| 7 | MAJOR | `v9_vs_ridge_cold_cell_1.json` stores the **ridge** under the key `XPert_released_ckpt` | **UPHELD** — I saw this and called it "a label quirk". It is a landmine. |
| 8 | MINOR | fold 1 only against a five-fold mean | **UPHELD, already flagged** [§46.4] |
| 9 | MINOR | chromatin SVD basis fitted over all 40 cells; ridge L2 is not rotation-invariant | **UPHELD** |
| 10 | MINOR | arm JSON records no batch/epochs/lr/seed | **UPHELD, verified absent** |
| 11 | MINOR | `known_cell_frac` counts train+test but is printed as a test-row property | **UPHELD** |
| 12 | MINOR | our `nanmean` vs their `mean` over per-row correlations | **UPHELD** |

### 51.2 🔴 My error, stated plainly

§45.1 defended +0.0042 as "~10x the seed-to-seed spread measured for this arm elsewhere (0.0004 across
three seeds, §43)". **That spread was measured on the WARM split** — and §30 records, in this project's own
words, that seed variance *collapses* on their warm split structure: "range 0.0001–0.0015 here, against
**0.005–0.045 on our cold splits**". So the cold-split seed spread plausibly **exceeds the entire effect**.
Importing a warm-split variance estimate to defend a cold-split effect is invalid, and it is the same
class as every other retraction here: a valid number computed on the wrong thing [method rule 5].

The reviewer also caught the deeper point: rows within a cell line **share the chromatin vector exactly**,
so for a claim that generalises over cell lines the denominator is 8 clusters dominated by two, not 21,151
rows. Four decimals are defensible for the point estimate; the interval is not.

### 51.3 🔴 The number is unreproducible — verified
- `grep -rn "chromatin_ablation" --include=*.py` over the whole repo returns **nothing**. No script writes
  `v9_chromatin_ablation_cold_cell_1.json`. The packet's claim that `head_to_head_mdmt.py` produced it is
  wrong, so that script's row-alignment guards (`row_index` intersection, `y_true`/`ctl_true` assertions)
  **never ran** on this comparison.
- `v9_xpert_arm_split_cold_cell_1_seed0.json` carries **`ablate_epi: true`**. The un-suffixed filename
  holds the *ablated* arm, so **the chromatin-ON arm has no artefact at all**. The overwrite the script's
  own comment warns about happened again, and the `_noepi` fix landed in the same commit as the runs.
- That file records **no** `batch`, `epochs`, `lr` or `seed` (verified absent), so "12 epochs, seed 0,
  batch 8" cannot be checked against anything. The script default is `--batch 48`.

### 51.4 🔴 The ridge gains MORE than we reported, and that decides it

`xpert_mdmt_baselines.py:174-187` scores each λ with `per_row_pearson(pr - C[te], (X - C)[te])` — on the
**test** rows — and keeps the argmax. No validation split. The arms therefore chose different λ
(no-chrom 1e4, chrom 1e3), and §44's +0.0029 is `max_λ(chrom) − max_λ(no chrom)`, a cross-λ difference.

Matched λ, read straight from the two sweeps:

| λ | no chromatin | with chromatin | **gain** |
|---|---|---|---|
| 1e2 | 0.2559 | 0.2576 | +0.0017 |
| **1e3** | 0.2942 | 0.2980 | **+0.0038** |
| 1e4 | 0.2951 | 0.2969 | +0.0018 |
| 1e5 | 0.2354 | 0.2367 | +0.0013 |

🔴 **§44 assumed per-arm-best λ FLATTERED the chromatin arm. It deflated it.** At λ=1e3 a
**closed-form ridge with no seed noise whatsoever** gains **+0.0038** against v9's +0.0042.

⇒ **The evidence does not separate "v9 extracts chromatin signal" from "chromatin is a weak linear cell
covariate that any model picks up equally."** And the cleanest chromatin measurement in this project is
the **ridge's**, precisely because it carries no training noise.

### 51.5 🔴 A transductive input on the v9 side only

`xpert_arm.py:176`:
```
self.cell_mean[c] = self.C[sel].mean(0) if sel.any() else self.C[m].mean(0)
```
`m` is every row of cell `c`; `sel` is its training rows. For a **cold** cell `sel.any()` is always False,
so `x_cell` for all of that cell's test rows is the mean control **over its own test rows** — while the
comment three lines above reads *"per-cell aggregate control, from TRAINING rows only"*. **The comment
asserts the opposite of what the code does.**

It is the mild form: an unsupervised aggregate of an *input* (control expression), never of the target. But
it is computed over the evaluation set, in exactly the regime under study, and `xpert_mdmt_baselines.py`'s
`feats()` uses only per-row `C[idx]` ⇒ **v9 has a cell-level input the comparator does not.** It applies
equally to both v9 arms, so it does not explain +0.0042 — but **+0.1775 and the XPert comparison are not
protected.**

### 51.6 What SURVIVED, per the reviewer

- **The split is genuinely cold** — loaded from the npz rather than trusted by name: 32 train vs 8 test
  cell lines, intersection **empty**, all 21,151 rows in the "pair NOT in train" stratum.
- **v9 > ridge, +0.1775** — *"I could not break this."* Holds on all 8 metrics, both dose strata, and all
  four effect-size quartiles, with the **largest** margin in the weakest-signal quartile (0.4127 vs
  0.2124), which addresses the inert-signature retraction. Subject to C4 and C6.
- **The published numbers are cited correctly** — independently re-read from Table R8.
- **Ablate-to-mean is correctly applied** on both channels, training rows only, lineage untouched.
- **§32's quantiser guard is genuinely repaired** — though `xpert_arm.py` calls `discriminates()` with
  `x=None`, which skips the strongest check; passing a real batch would close it.

### 51.7 Consequences — CLAIM 7.1 RETRACTED pending re-measurement

**Chromatin has now failed three times**: +0.0029 (ridge, §44), +0.0038 (ridge at matched λ, §51.4), and
+0.0042 (v9, §45 — now unreproducible with an invalid interval). A model with **no seed noise** gains what
the deep model gains.

> **The defensible statement is now: cell-line chromatin is a weak linear cell covariate. Nothing in this
> project demonstrates that a deep model extracts anything from it that a ridge does not.**

That is a **stronger** negative than §45 claimed and it costs nothing to say, because [§49] establishes the
field has explicitly asked for this measurement.

### 51.8 Required before chromatin is mentioned again, cheapest first
1. **C5, the placebo stratum** — free, no GPU. 3 of 8 test cells have no chromatin; both arms see zeroes
   there, so the paired delta on those 1,212 rows **is** the training-noise floor. If it is also ~+0.004
   the question is settled negatively. Blocked only by C2 (needs saved predictions).
2. **Re-run both arms with `--save_pred`, distinct filenames, args recorded in the JSON**, and route the
   comparison through `head_to_head_mdmt.py` so its guards execute.
3. **≥3 seeds per arm**, reporting the ON-arm seed spread on *this* split beside the effect.
4. **Cluster-bootstrap over the 8 cell lines**, plus the per-cell-line paired delta.
5. **Fix λ selection** on a held-out fold of training cells; re-state the ridge gain at matched λ.
6. **Re-run v9 cold-cell with `x_cell` = global training control mean**, and report the change in 0.4734.
7. **Key the JSON off `--theirs_label`**, then grep every results file for `XPert_released_ckpt`.

### 51.9 Reviewer calibration, recorded
12 challenges, **12 upheld, 0 rejected**, severities not inflated — it separated the +0.1775 result (which
survived) from the +0.0042 (which did not) and said so in the first paragraph. It loaded the split from the
npz rather than trusting its name, re-read Table R8 independently, and traced three numbers to the code.

It also **self-disclosed a blinding leak**: tracing C2 put `git log` on screen and it saw commit
`5e80c5d`'s subject line, which states a conclusion. It flagged this unprompted, noted its C1/C3
assessment was already written, and invited discounting. **That is the behaviour the blinding design
exists to produce.** The leak is real and worth fixing: commit subjects in this project state conclusions,
so a reviewer tracing provenance will see them. Future packets should include the artefact paths and
`git show --stat` output the reviewer needs, so it never has to run `git log`.

## 52. The chromatin effect is 0.57 sigma of the run-to-run difference — and that saves 41 GPU-hours (2026-09-20)

`orchestration/bus/to_pi/001_adjudication_ack.md`. The reviewer raised no new challenges but made an
arithmetic point about §51.8's action item 3. **It had one figure wrong. Correcting it makes the
conclusion stronger.**

### 52.1 Their argument, and the misreading in it
They took §30's "0.005–0.045 on our cold splits" as a **standard deviation** and derived
`sd_diff = sd x sqrt(2) ≈ 0.007–0.064`.

🔴 **Those are RANGES, not sds.** §21's three-seed table is explicit: unseen-cell range **0.0103** /
**sd 0.0052**; unseen-both range **0.0457** / **sd 0.0232**. So "0.005–0.045" is the span of *ranges*
across splits, and using it as an sd inflates the noise floor by roughly 2x.

### 52.2 Done properly, with the directly relevant number
Our regime is **unseen cell**, and this project has measured its seed sd directly — v7, three seeds,
protocol-matched [§21]: **sd = 0.0052**.

Two independent single-seed runs differ with
`sd_diff = 0.0052 x sqrt(2) =` **0.0074**.

The observed chromatin effect is **+0.0042**.

⇒ **The effect is 0.57 sigma of the run-to-run difference distribution.** Not "unproven against an
unmeasured floor" — **inside a floor this project measured in August**, by a factor of two.

### 52.3 What it would cost to detect an effect that size
One cold-cell arm is **37,290 s = 10.36 h** (recorded in the arm JSON).

| plan | runs | GPU-hours | verdict |
|---|---|---|---|
| §51.8 item 3 as written (3 seeds x 2 arms) | 6 | **62 h** | **> 2 weeks of a 30 h/week quota, and it only measures the sd we already have** |
| powering +0.0042 to significance at sd_diff 0.0074 | ~24 | **~250 h** | infeasible |
| **§51.8 items 1+2 combined** (2 arms, `--save_pred`) | **2** | **~21 h** | affordable, fixes provenance, **and yields the placebo stratum** |

🔴 **ACTION ITEM 3 IS DROPPED as a means of deciding the question.** Buying 62 GPU-hours to re-measure
a spread [§21] already recorded, in order to test an effect at 0.57 sigma, is not a defensible spend. If
the this-split spread is ever wanted **for the record**, that is a separate and much later purchase, and
it must be labelled as such rather than as a decision procedure.

### 52.4 The reviewer's second point, which is simply correct
§51.8 listed items 1 and 2 as separate. They are **one experiment**: the re-run (item 2) produces exactly
the saved predictions the placebo stratum (item 1) needs. Revised plan:

> **Two runs, ~21 GPU-hours, `--save_pred`, distinct filenames, args written into the JSON, comparison
> routed through `head_to_head_mdmt.py` so its row-alignment guards execute.** That simultaneously fixes
> the C2 provenance failure and delivers the C5 placebo stratum — the paired delta on the 1,212 rows from
> the 3 test cell lines with no chromatin track, where **both arms see identical zeros**, which is a
> direct within-comparison estimate of the training-noise floor requiring no extra seeds at all.

### 52.5 METHOD RULE 10, from the reviewer's own generalisation
> **Compute the direction of a selection bias. Never assume it.**

§44 wrote "best lambda taken **per arm, which favours the chromatin arm**" — an assumed direction. Matched
lambda showed the opposite: the per-arm argmax *deflated* the chromatin gain (+0.0029 reported against
**+0.0038** at matched lambda = 1e3) [§51.4]. The reviewer reports it had the same prior and computed it
anyway, which is why it found the inversion. **Whenever a per-arm argmax, best-of-N, or tuned
hyperparameter appears in a comparison, recompute at matched setting. It is one solve per value.**

## 53. The chromatin retraction is robust to the noise floor being wrong (2026-09-20)

`orchestration/bus/to_pi/001_note_reply.md`. The reviewer accepted §52's correction to its own arithmetic
("keep the correction, not my arithmetic") and then **corrected mine**, and added the robustness check
§52 should have run. **All three points verified numerically before acceptance.**

### 53.1 🔴 §52's cost table quoted the 51 % power point as though it were the decision point

§52.3 gave "~24 runs / ~250 h" to power +0.0042. That is where the effect merely **equals** the critical
value — about 51 % power, i.e. a coin flip. Recomputed at sd 0.0052, α = 0.05 two-sided, 10.36 h/run:

| power | n/arm | runs | GPU-h |
|---|---|---|---|
| 51 % (§52's figure) | 11.8 | 23.6 | 244 |
| **80 % (the decision-grade number)** | **24.1** | **48.1** | **499** |
| 90 % | 32.2 | 64.4 | 667 |

⇒ **The experiment §52 declined costs twice what §52 said it did.** Conclusion unchanged, margin doubled.

### 53.2 🟢 The retraction does not rest on the floor being exactly 0.0052

The obvious objection to resting a retraction on one small measurement: sd = 0.0052 is an **n = 3**
estimate. On 2 df its 95 % CI is **[0.00271, 0.03268]**. So does the effect survive *anywhere* in that
interval?

| σ | sd_diff | +0.0042 in σ |
|---|---|---|
| 0.00271 (CI low — most favourable) | 0.0038 | **1.10** |
| 0.0052 (point) | 0.0074 | **0.57** |
| 0.03268 (CI high) | 0.0462 | **0.09** |

⇒ **There is no value of the noise floor consistent with the August measurement at which +0.0042 becomes
detectable.** At the most favourable end of the floor's own confidence interval the effect is still 1.10 σ.
The retraction is robust to the floor estimate being wrong in the direction that would rescue it.

### 53.3 🟢 A check the reviewer ran and then declined to report as a finding

Both sds are almost exactly half their range (0.0103/0.0052 = 1.98; 0.0457/0.0232 = 1.97). That is the
signature of an `sd = range/2` shortcut, which is biased low and would have inflated every σ argument here.

It is **not** that. For n = 3 the sample range/sd ratio is algebraically confined to **[√3, 2]**, reaching
2.000 only for a perfectly evenly-spaced triple. Both ratios sit inside the legal band, so they are real
sample sds. Verified independently.

The reviewer wrote: *"Raising this as a finding would have been manufacturing one, so I am recording it as
a check that passed."* **That is the calibration behaviour the review contract asks for**, and it is worth
recording as prominently as the challenges it did raise.

### 53.4 METHOD RULE 11, amended by the reviewer
The original: *size an experiment against a measured noise floor before buying it.* The amendment, which
is what this whole exchange actually turned on:

> **… and check WHAT KIND of statistic the measurement is.**
> Ranges, sds and sems all render as "the spread" in prose. The failure here was not using an *unmeasured*
> floor — it was two parties in a row using a *measured* number whose definition neither had read.
> The reviewer read a range as an sd [§52.1]; §52 then read a 51 % power point as a decision point [§53.1].

### 53.5 Why the 21-hour re-run is STILL worth buying
The argument above rests on an **n = 3 sd measured on v7, on a different target, in August**. The C5
placebo stratum gives a noise-floor estimate from **these two runs, on these rows, in this architecture** —
1,212 rows from the 3 test cell lines with no chromatin track, where both arms see identical zeros. That is
strictly better evidence than a transferred variance estimate, it is a within-comparison quantity needing
no extra seeds, and it arrives free inside a re-run that has to happen anyway because [§51.3] the current
number has no generating script.

## 54. 🔴 CHROMATIN, SETTLED: the correct denominator gives a NEGATIVE point estimate. Cost: 0 GPU-hours (2026-09-20)

`model/v9/chromatin_ablation_analysis.py` → `model/results/v9_chromatin_ablation_cold_cell_1_RESCORED.json`.

### 54.1 The 21 GPU-hours were never needed

§51.8/§52.4 planned a ~21 GPU-hour re-run to obtain saved predictions. **They already existed** — both arms,
since 2026-08-31, in `external/v9_mdmt_preds/`:
`v9_cc1_epi_seed0.npz` and `v9_cc1_noepi_seed0.npz`, each carrying `deg_pred`, `y_true`, `ctl_true` and
`row_index` for all 21,151 rows, plus `.pt` checkpoints.

Review 001's C2(c) — *"No `--save_pred` `.npz` exists for either cold-cell arm, so nothing can be re-scored
without retraining"* — is therefore **the one challenge that does not hold**, and the fault is **ours**:
`external/` is gitignored, so a reviewer reading the repository could not see them.
**Packet defect. Every future packet lists artefact paths explicitly.**

Adjudication updated: review 001 stands at **11 of 12 upheld**, with the single miss caused by a
deficiency in the packet rather than by the reviewer.

### 54.2 🟢 C5, the placebo stratum — the reviewer's own test, and it is decisive

3 of the 8 test cell lines have **no chromatin track**, so **both arms see identical zeros** there. The
paired delta on those rows is a direct, within-comparison estimate of the training-noise floor — no extra
seeds, no variance transferred from another split or another model version.

| stratum | n | epi ON | epi ABLATED | paired Δ |
|---|---|---|---|---|
| chromatin present (MCF7, HT29, MDAMB231, CD34, THP1) | 19,950 | 0.4717 | 0.4659 | **+0.00586** |
| **NO chromatin — PLACEBO** (HS578T, BJAB, H1975) | 1,201 | 0.5019 | 0.5245 | **−0.02263** |
| pooled — the §45 number | 21,151 | 0.4734 | 0.4692 | +0.00424 |

🔴 **The training-noise floor, measured from these two runs on these rows, is ≈0.023 in magnitude — five
times the claimed effect of +0.0042.** On rows where neither arm can possibly use chromatin, the arms
differ by more than the entire effect attributed to chromatin.

### 54.3 🔴 C1, the correct denominator — and the point estimate goes NEGATIVE

Rows within a cell line **share the chromatin vector exactly**, so for a claim that generalises over cell
lines the unit is the cell line, not the row.

| cell line | n rows | tracks | paired Δ |
|---|---|---|---|
| HT29 | 5,837 | 2 | **+0.00946** |
| MCF7 | 10,815 | 3 | **+0.00702** |
| THP1 | 815 | 2 | −0.00219 |
| MDAMB231 | 2,188 | 3 | −0.00466 |
| CD34 | 295 | 3 | −0.00783 |
| BJAB | 73 | 0 | −0.00769 |
| HS578T | 1,074 | 0 | −0.02043 |
| H1975 | 54 | 0 | **−0.08652** |

**Cluster bootstrap over the 8 cell lines (20,000 resamples):**

| | |
|---|---|
| mean of per-cell-line means | **−0.01411** |
| 95 % CI | **[−0.03666, +0.00102]** |
| cell lines with a positive effect | **2 of 8 (25 %)** |
| row-bootstrap CI quoted in §45 | **[+0.0036, +0.0049]** |

🔴 **The two intervals do not overlap.** The row bootstrap was answering a different question, exactly as
[§51.1 C1] said. Under the correct denominator the point estimate is **negative**, only a quarter of cell
lines show a positive effect, and **the pooled +0.0042 is an artefact of MCF7 and HT29 holding 78.7 % of
the rows** and happening to be the two positives.

### 54.4 The settled statement

> **Cell-line chromatin conditioning does not contribute measurably to unseen-cell-line generalisation in
> this architecture.** The effect is indistinguishable from zero on the correct unit of analysis, its
> point estimate there is negative (−0.014, 95 % CI [−0.037, +0.001]), and the training-noise floor
> measured within the same comparison is ≈ 5x larger than the effect once claimed for it.

Stated with equal care in the other direction: **this is not evidence that chromatin HURTS.** The cluster
CI spans zero. The correct reading is *no measurable effect*, with the earlier positive number explained.

Chromatin's tally is now: +0.0029 ridge [§44] → +0.0038 ridge at matched λ [§51.4] → +0.0042 pooled rows,
retracted [§51] → **−0.014 on the correct denominator** [§54]. **The question is closed.** The novelty
claim is untouched [§49]; what is gone is the benefit.

### 54.5 Guards that executed here and never executed on the §45 number
`row_index` identical across arms (n=21,151) · `y_true` and `ctl_true` byte-identical (max\|diff\| 0.0e+00)
· every scored row confirmed a **test** row of `split_cold_cell_1` · 8 test cells vs 32 train cells,
intersection **empty** · 0 degenerate rows, counted rather than absorbed [C12].

### 54.6 Method note: the cheapest experiment was the one already paid for
§52 was about to buy 21 GPU-hours — and §53.1 showed the decision-grade version of the same question would
have cost **499**. The answer was on disk, and it is **better** evidence than the retrain would have been,
because the placebo stratum is a within-comparison floor rather than a variance estimate imported from v7
in August.

> **METHOD RULE 12: before buying compute, enumerate what is already on disk — including gitignored
> artefact directories.** This project has now twice spent effort on something already in hand
> (§46.1 supplementary tables; §54.1 saved predictions).

## 55. ✅ CHROMATIN, FINAL: +0.0004 on the right estimand — a clean null, not a negative (2026-09-20)

Review 002 (`orchestration/bus/to_pi/002_review.md`) returned `SOUND-WITH-CAVEATS`, reproduced every number
in packet 002 independently from the raw `.npz`, **and caught §54 committing the mirror image of the error
§51 had just retracted.** All three load-bearing challenges verified numerically before acceptance.

### 55.1 🔴 §54's −0.0141 is as wrong as §45's +0.0042, with the sign reversed

The three no-track cell lines (HS578T, BJAB, H1975) contribute **102 % of the 8-cell unweighted mean.**
Those are cells where **the treatment does not exist** — they cannot carry information about whether
chromatin helps, and including them as 3 of 8 equally-weighted clusters is what manufactured the negative.

Restricted to the **5 cells where chromatin is actually present** — which is the estimand the objective
asks for:

| | |
|---|---|
| unweighted per-cell mean | **+0.000360** |
| cluster bootstrap CI95 | **[−0.005433, +0.006153]** |
| cells positive | 2 of 5 |
| sign test | p = 1.000 |

✅ **A tight, well-centred null.** Better than either headline, and it is the number that belongs in the
paper. §45 was row-weighting; §54 was three cell lines with no chromatin at all.

### 55.2 🔴 "Both arms see identical zeros" is FALSE — my error, in the script and in §54.2

Verified in the executed path:
- `E_final[HS578T]` is **all-zero** (absmax 0.0000, 0/2934 non-zero); BJAB and H1975 are **absent** from
  the cell index. So the **ON** arm feeds zeros on those rows.
- `xpert_arm.py:157-158` does `self.E[:] = self.E[m_tr].mean(0)` and `self.r[:] = self.r[m_tr].mean(0)`
  for **every row**. So the **ABLATED** arm feeds a non-zero training-mean constant on those same rows.

**The inputs differ, and the direction handicaps the ON arm**: zeros are out-of-distribution for a model
trained mostly on real chromatin, while the training mean is in-distribution.

⇒ The −0.022625 placebo figure is **a noise floor PLUS a handicap**, not a clean floor. It **overstates**
the floor by an unknown amount. §54.2's "the training-noise floor … is ≈0.023" is withdrawn; the honest
reading is "how much these two models differ where chromatin cannot inform them", which is weaker.
Clean version available with no retraining: re-score an ON-arm variant feeding the training mean on
no-track rows, from the saved checkpoint.

### 55.3 🔴 The sign of the headline was an artefact of estimator choice — five estimators, two signs

Same 8 per-cell numbers:

| estimator | value |
|---|---|
| size-weighted (= pooled, §45) | **+0.004237** |
| unweighted, 8 cells (§54) | **−0.014105** |
| trimmed (drop min + max) | −0.005964 |
| drop H1975 | −0.003761 |
| **treated cells only (§55.1, adopted)** | **+0.000360** |

🔴 **With 8 clusters, no directional statement survives the choice of estimator.**
⇒ **METHOD RULE 13: fix the estimand and its estimator BEFORE computing the number, not after seeing it.**
Report the rest as a sensitivity table, which is what they are.

### 55.4 🔴 The per-cell pattern tracks cell SIZE, not chromatin coverage

- **Spearman(n_rows, paired Δ) = 0.762, p = 0.028.** The two positive cells are the two largest.
- **No dose-response in track count**: 3 tracks **−0.001825** (MCF7, MDAMB231, CD34) vs 2 tracks
  **+0.003636** (HT29, THP1) vs 0 tracks −0.038213.

If chromatin were doing the work, 3-track cells should lead 2-track cells. **They do not.** So the pattern
is not "an effect present only in well-covered cells" (ask 3, answered negatively) — the leading
alternative is a size-dependent artefact. At 5 cells nothing cheap distinguishes these. **That is a reason
to stop, not to run another experiment.**

### 55.5 The final statement on claim 7.1

> **On the five unseen cell lines where chromatin data exists, ablating it changes accuracy by +0.0004
> (cluster CI [−0.0054, +0.0062], 2 of 5 positive, sign test p = 1.000). The earlier +0.0042 was
> row-weighting; the −0.0141 was the three cell lines that have no chromatin at all.**

Chromatin's full arc: +0.0029 [§44] → +0.0038 matched λ [§51.4] → +0.0042 retracted [§51] → −0.0141
wrong estimand [§54] → **+0.0004, a clean null on the correct estimand [§55]**. **Closed.** The novelty
claim is untouched [§49]; there is no benefit to claim. This is a *null*, not a negative — stating it as
"chromatin hurts" would repeat §54's error a third time.

### 55.6 Reviewer calibration — two things it did that raise its credibility
- **It cut against its own C2.** HT29 and THP1 have 2 of 3 tracks, so the C2 handicap should apply
  partially — yet HT29 is the *most positive* cell. It reported this as evidence its own challenge does
  not explain the whole pattern.
- **It hunted a bug, failed to find one, and said so.** HS578T is in the cell index with an empty mask, so
  `xpert_arm.py`'s normalisation loop skips it — raw values would have reached the model un-z-scored. They
  are all-zero, so it does not occur. Reported as a check that passed, for the second review running.

Running tally: **23 of 24 challenges upheld across two reviews**, the single miss caused by a packet
defect [§54.1].

## 56. 🔴 PRE-SPEND REVIEW KILLS THE DESIGN — and §47's mechanism premise is FALSE (2026-09-20)

Review 003 (`orchestration/bus/to_pi/003_review.md`) returned `NOT-SUPPORTED` on the **design**, before any
GPU was committed. **6 challenges, 6 upheld.** The two load-bearing ones I verified directly against the
code. **The gate paid for itself on its first use.**

### 56.1 🔴 C3: "a bag of atoms with no intramolecular structure" is FALSE — §47.2's premise collapses

`drug_atom_reprs.npy` holds Uni-Mol **`atomic_reprs`** — confirmed from
`drug/scripts/step8_integrate_atom_tokens.py` and `drug_atom_meta.json` (`remove_hs: True`,
`mean_tokens_per_mol` 33.2, dim 512). Those are the **per-atom outputs of Uni-Mol's transformer encoder**,
which attends over every atom with a 3D distance bias.

⇒ **Atom i's 512-d vector already encodes its molecular environment.** `linear(atoms)` is a linear map of
**already-contextualised** representations. §47.2's claim that v9's genes attend over "a BAG of
independently projected per-atom Uni-Mol vectors with no intramolecular structure" — repeated in the
`_DrugBlock` docstring and in packet 003 — is **wrong**.

What v9 actually lacks relative to XPert is a **second, in-loop re-contextualisation that co-evolves with
the cell embedding across blocks**. Real, but a far weaker deficit. And critically: **§47.3's falsifiable
prediction (−0.025 flips positive) was calibrated against a baseline that does not exist.** A null would
have been over-read as "atom-level attribution here is dead" when it would only license "a second round of
contextualisation adds little on top of Uni-Mol's".

Free test of the strong premise, if wanted: are a molecule's `atomic_reprs` predictable from its own CLS
token plus atom identity? If largely not, the structure is already there.

### 56.2 🔴 C4: the effect being explained has NO error bar, and n = 480

Verified: `grep -c "bootstrap\|ci95\|percentile" model/v9/probe_v9.py` → **0**. The −0.00671 / −0.02549 /
−0.02187 figures come from **one checkpoint, one seed**, on **480 signatures per regime** — and 480 is the
whole split, not a cap (`--n_eval` defaults to 1500 and the code takes `min(n_eval, len(idx))`).

⇒ **§47 was about to spend 5.6 GPU-h explaining a number whose uncertainty has never been computed.**

And the check is free: **three fold0 seeds are already on disk** — `r0_ckpt_v9_fold0_seed0.pt`,
`r1_ckpt_v9_fold0_seed1.pt`, `r2_ckpt_v9_fold0_seed2.pt` in `external/v9_checkpoints/`. Inference only.
**If −0.025's interval spans zero, or moves across seeds by more than its own width, there is no
phenomenon and the run must not be bought.** Same lesson as §54.1 and §46.1: *the answer was already on
disk.* **Third time.**

### 56.3 🔴 C1: the 2x2 is missing its fourth cell, so I measured two main effects and called it an interaction

Packet 003 proposed S11 (intact), S01 (atoms ablated), S10 (`drug_sa` ablated). The quantity actually
wanted is **`(S11−S01) − (S10−S00)`**, and **S00 — both ablated — was not in the design.** Without it,
"does the atom contribution revert toward −0.025?" can only be answered against §37's SA-**off** run —
which is precisely the between-run comparison the design existed to avoid.

**Direct answer to my own ask 1: I moved the problem, I did not remove it.** Fix costs one extra eval
pass, zero GPU.

### 56.4 🔴 C2: mean-ablating `drug_sa` re-admits the +30 % capacity confound INSIDE the run

`_DrugBlock.forward` is two residual branches: `D + attn(n1(D))` then `D + ff(n2(D))`. Neutralising the
module's output kills **the SwiGLU as well as the cross-atom mixing**, so the ablated arm is not "the same
model without contextualisation" — it is a **smaller-capacity model**. That is exactly the confound
§47.6 claimed to escape by staying within one run.

✅ **Better operator, adopted: ablate to DIAGONAL ATTENTION.** Mask the attention matrix to the identity so
each atom attends only to itself. Parameters, FFN, residual scale and the per-atom pathway all survive;
only cross-atom information flow is removed — which is the hypothesis, exactly. **Validation is already
built**: under diagonal attention, perturbing atom 3 must move atom 1 by **exactly 0.00e+00**, matching
the arm-off case in `test_drugsa_v9.py`. Report `|dY|max` so a true null stays distinguishable from a
module that never fired.

### 56.5 C5/C6: the cost figure is a floor, and the cheap-split option is not free
§47.6's ~5.6 h scales §29.1's 0.0091 s/row-epoch — **measured with the module OFF**. `_DrugBlock` adds
O(A²)-in-atoms attention per block plus a SwiGLU on top of +30 % parameters, so the direction is known and
the magnitude is not. Fix: time 50 steps at full width with the flag on. Two minutes.

And `probe_v9.py` takes `--ckpt` but no `--bundle`/`--split` — it is wired to the fold0 pipeline. The
"cheap 1.7 h split" costs porting work that was not in the cost table.

### 56.6 🟢 The one useful asymmetry, which the reviewer stated better than I did
> A **null** interaction needs no capacity control and is decisive on its own. Only a **positive** result
> obliges buying the matched-capacity arm.

So the design is a sound **one-sided screen** — once C1 and C2 are fixed. That is worth keeping.

### 56.7 DECISION: GPU spend DEFERRED. Free work first, in the reviewer's order.
1. **C4 — free.** Add a paired row bootstrap to `probe_v9.py` and run it across the three fold0 seeds on
   disk. Inference only. **If −0.025 is not solidly non-zero and stable, stop.**
2. **C3 — restate the mechanism** in §47, the `_DrugBlock` docstring and the config comment, and
   re-derive what effect size would count as confirmation.
3. **C1 + C2 — add S00; switch the operator to diagonal attention.** Code only.
4. **C5 — measure real step time**, then choose the split.
5. Only then spend.

**GPU hours committed so far this session: 0.**

### 56.8 Reviewer calibration
6 of 6 upheld. It also **nearly made a retraction-class-8 error and caught itself**: its first grep for
`drug_SA` covered only `external/xpert/code/XPert/*.py`, returned nothing, and looked like a config-vs-code
failure on our side — it then found it in `models/`, confirmed `crossEncoder.__init__:359`,
`forward:364`, the cross-attention at `:371`, and `model_XPert.py:44/65` reassigning `drug_embed` per
layer, and reported the claim as **sound**. Read from the executed path, not from the first grep.

Running tally: **29 of 30 challenges upheld across three reviews**, the single miss caused by a packet
defect [§54.1].

## 57. C4 GATE RESULT: the atom-token effect is REAL and seed-stable — at HALF the size, on 2 of 3 splits (2026-09-20)

`model/v9/atom_ablation_ci.py` → `model/results/v9_atom_ablation_CI.json`. Review 003's C4 demanded an
interval and a cross-seed check before any GPU was bought. **Cost: 0 GPU-hours** — three fold0 checkpoints
already on disk [§56.2], local-GPU inference only, `pearson_rows` copied verbatim from `probe_v9.py` so
the intervals attach to *that* number.

Bootstrap is over **rows**, on the **difference of medians** — the statistic §37 actually quotes — with the
paired mean reported beside it. Chunking held at `batch 48` because `ablate_to_mean` uses the **chunk**
mean, so a different batch is a different ablation.

### 57.1 The result, 3 seeds x 3 splits, n = 1500 each

| split | per-seed d_median | all 3 CIs exclude 0? | verdict |
|---|---|---|---|
| **unseen_cell** | −0.0032, −0.0039, −0.0015 | 🔴 **NO — all three SPAN ZERO** | **NULL. §37's −0.00671 was noise.** |
| **unseen_compound** | −0.0156, −0.0176, −0.0107 | ✅ **yes, all three** | **REAL** |
| **unseen_both** | −0.0245, −0.0109, −0.0107 | ✅ **yes, all three** | **REAL** (range 0.0138 sits just inside its 0.0149 CI width) |

### 57.2 ✅ The cross-seed check PASSES on all three
| split | range across seeds | mean CI width | stable? |
|---|---|---|---|
| unseen_cell | 0.00239 | 0.00975 | ✅ |
| unseen_compound | 0.00690 | 0.01321 | ✅ |
| unseen_both | 0.01376 | 0.01489 | ✅ (narrowly) |

Seed-to-seed movement is **smaller than the within-seed interval in every case**, so where the effect is
non-zero it is **not a training-run artefact**. That is the question `probe_v9.py` never asked.

### 57.3 🔴 But the headline was inflated ~1.7x, and one third of it was noise

§37 reports **−0.00671 / −0.02549 / −0.02187** from one checkpoint at n = 480.
At n = 1500 over three seeds the medians are **−0.0029 / −0.0146 / −0.0142** (mean of per-seed).

- **unseen_cell: retract.** −0.00671 becomes a null whose interval spans zero on every seed.
- **unseen_compound: halve.** −0.02549 → **−0.0146** (range −0.0107…−0.0176).
- **unseen_both: shrink.** −0.02187 → −0.0142.

Consistent with n = 480 being an underpowered sample — see §57.4.

### 57.4 🔴 CORRECTION TO REVIEW 003 C4: 480 was a CAP, not the split size

C4 states 480 is *"the whole split, not a cap: `--n_eval` defaults to 1500 and the code takes
`min(n_eval, len(idx))`"*. **That cannot be right.** All three splits report **exactly 480** — for strata
drawn from 47,002 / 58,796 / 15,083 signatures. Three different populations cannot each yield exactly 480
eligible rows. And this run finds **≥1500** eligible rows in `unseen_cell` alone under identical filters.

⇒ `probe_v9.py` was invoked with `--n_eval 480`, presumably for a CPU budget. **The cap is undocumented
in the output JSON**, which is why the reviewer read it as the population size.
**This is the first factual error in 30 challenges** — and it cuts *for* C4's conclusion, not against it:
the sample was arbitrarily small for compute reasons, which is exactly why the effect needed an interval.
Review 003 stands at **5 of 6 upheld**, running tally **34 of 36**.

⇒ **Fix: `probe_v9.py` must record `n_eval` and whether it bound.** A sample size that is silently a
budget cap and reads as a population is the same class as §56.2's missing interval.

### 57.5 A disagreement between estimators, recorded rather than resolved
On `unseen_cell` seed 1 the two estimators differ in **sign**: d_median **−0.00389** vs paired_mean
**+0.00037**. Both intervals are near zero, so this is what a null looks like under two estimators rather
than a contradiction — but it is worth noting that the paired mean is the tighter estimator throughout
(CI widths ~0.002 vs ~0.010) and would have been the better choice for §37. Per method rule 13 the
estimator is now fixed in advance: **difference of medians**, because that is what §37 quoted and what the
comparison must attach to.

### 57.6 ✅ What this licenses — the GPU spend is now justified, at a smaller predicted effect
The phenomenon is real on unseen compounds and unseen both, stable across three seeds, with `dY_max`
1.3–2.5 so the component fires. **There is something to explain.**

But §56.1 still stands: the atom vectors are Uni-Mol **`atomic_reprs`**, already contextualised by a
transformer with a 3D distance bias, so the mechanism under test is a **second, in-loop
re-contextualisation** — not "adding structure where there was none". **The prediction must therefore be
calibrated to −0.0146, not −0.025**, and a confirmatory result means that figure moving toward zero or
positive on `unseen_compound`, which is now the primary split (unseen_cell is out — there is no effect
there to move).

**Remaining before spend, in review 003's order:** C1 (add S00) and C2 (diagonal-attention operator,
delegated as W4) are code-only; C5 (measure real step time with the module on) is two minutes. Then packet
004 carries the revised design and the revised prediction.

**GPU hours committed this session: 0.**

## 58. ✅ SPEND APPROVED — the design IS powered; I propagated the wrong interval (2026-09-20)

Review 004 (`orchestration/bus/to_pi/004_review.md`) **reversed its own 003 verdict**: `SOUND-WITH-CAVEATS`,
buy the run. The crux in packet 004's ASK 1 resolved **from data already in our own JSON and a line already
in our own code.** Both claims verified before acceptance.

### 58.1 🔴 My power table used the wrong one of two intervals our code emits

`atom_ablation_ci.py` emits **two** intervals per cell. I propagated the **difference-of-medians** width.
The **paired-mean** width sitting beside it is 3.9–6.6x tighter:

| split | median CI width | paired-mean CI width | ratio |
|---|---|---|---|
| unseen_cell | 0.0082–0.0118 | 0.0015–0.0022 | **5.4–5.6x** |
| unseen_compound | 0.0114–0.0145 | 0.0025–0.0036 | **3.9–5.5x** |
| unseen_both | 0.0135–0.0170 | 0.0022–0.0031 | **5.5–6.6x** |

Redone in paired-mean units on `unseen_compound`, mean over three seeds (atom effect **−0.01310**,
paired CI width **0.00291**, propagated interaction width **0.00411**):

| scenario | interaction | §004 ASK 1 claimed | **actual** |
|---|---|---|---|
| full rescue | +0.01310 | 3.06 σ | **12.5 σ** |
| **partial (half)** | +0.00655 | **1.5 σ — "not detectable"** | **6.2 σ** |
| quarter rescue | +0.00327 | — | **3.1 σ** |

⇒ **The design is powered for partial rescue, and for a quarter rescue.** Packet 004's headline objection
was my own arithmetic error, not a property of the experiment. And **0.00411 is an upper bound** — it
assumes zero correlation between the two atom contrasts, which share rows *and* weights.

### 58.2 ✅ ASK 2 was already implemented — `interaction_2x2.py:236`

I asked whether a blocked bootstrap on the per-row interaction contrast would beat √2 propagation. It
would, and the delegated harness already does it:
```python
diff_interaction = (a11 - a01) - (a10 - a00)        # PER-ROW double difference
boot_mean_inter  = diff_interaction[idx].mean(axis=1)   # line 236
```
Resamples rows once and recomputes the per-row double difference, so the row main effect and both
single-factor row components **cancel inside each draw**. The reviewer's point that `boot_inter` (a
difference of four medians) lacks this property is correct — a difference of medians does not decompose
per row. **Both are emitted; only one is the quantity ASK 1's reasoning described.**

### 58.3 🔴 C3: a tight interval is NOT a licence to read one run as decisive
The counterexample is in our own C4 table. `unseen_cell`:

| seed | d_paired_mean | CI95 | |
|---|---|---|---|
| 0 | **−0.00155** | [−0.00264, −0.00046] | **excludes zero** |
| 1 | **+0.00037** | [−0.00049, +0.00123] | spans zero, **opposite sign** |

Two tight row-level intervals, incompatible signs, seed variance dominating. **This is review 001's C1
reappearing under a new estimator.** The interaction is a genuine within-run contrast, so method rule 7
exempts it from ≥3 seeds for *validity* — but its **magnitude is not thereby seed-stable**.

### 58.4 ✅ PRE-COMMITMENT, recorded BEFORE the spend (method rule 13)
The estimand is **`interaction_paired_mean`** = mean over rows of `(S11−S01) − (S10−S00)`, with the
blocked-bootstrap interval `interaction_paired_mean_ci95`. Primary split **`unseen_compound`**. Reading
rule, fixed now:

| result | reading |
|---|---|
| interaction **≥ 3 σ** on the paired-mean interval | **report as a result from one seed** |
| **1.5–3 σ** | **explicitly INCONCLUSIVE** — needs a second seed at another ~5.7 h, a separate decision |
| **< 1.5 σ**, with `\|dY\|max` confirming **both** ablations fired | **informative null** — contextualisation does not change what atoms are worth |
| either ablation shows `\|dY\|max ≈ 0` | **void** — a branch never fired; not a null [method rule 2] |

Read against **`S10−S00` from the same run**, never against the historical −0.0146 [C5] — so the question
of whether the SA-off effect transfers never arises.

### 58.5 Two corrections to our own record
- 🔴 **`unseen_cell` was excluded for the wrong stated reason.** "All three CIs span zero" is true of
  `d_median` but **false of `d_paired_mean` at seed 0** (which excludes zero). The defensible reason is
  **seed-instability with a sign flip** [§58.3]. §57.1 said there is *no effect*; the data say the effect is
  too small and too seed-unstable to move. Restated.
- ✅ **C5 is discharged better than I allowed.** Parameters overstate FLOPs here because the sequences
  differ ~29x: drug side ≈ 34 tokens (703,851 atoms / 21,220 compounds ≈ 33, plus the global token) against
  978 genes. Drug-side FFN ≈ 3.5 % of gene-side; **drug self-attention is 34²/978² ≈ 0.1 %** of gene
  self-attention. So +30 % parameters buys almost no compute, and **1.009x is probably close to right
  rather than a compressed floor.** Budget guard kept regardless.

### 58.6 The reviewer conceded my correction, and classified its own error
It accepted that 480 was an `--n_eval` cap and recorded the error **as its own**, not as our packet defect:
*"I read an invocation parameter as a data property, which is the same class of error as reading a config
over the executed path — the thing I am here to catch."* It also noted C4 was the most valuable item in
review 003 **because** running it halved the effect and removed a split.

### 58.7 ✅ DECISION
**Buy the run.** One SA-on training run, fold0, 12 epochs, `drug_self_attn=True`, budget ~5.7 h with the
guard armed. Then `interaction_2x2.py` on its checkpoint, read by §58.4's pre-committed rule.

Three design iterations, an effect halved, a split removed, a false premise retracted and a power error
caught — **all before the first GPU hour.** Running tally: **39 of 42 challenges upheld**, and the two
reversals in this project's favour both came from the reviewer reading our own artefacts more carefully
than we did.

## 59. The SA-on run landed: 12/12 epochs, both guards fired, and a batch confound I created (2026-09-21)

`apexblue/lincs-v9-drugsa-s0`, T4x2, seed 0, fold 0, `drug_self_attn=True`. Status COMPLETE, epochs 0-11
all present, **6.232 h elapsed** against a 7.5 h guard that never tripped. Artefacts retrieved:
`ckpt_v9_fold0_seed0.pt` (216.98 MB, now `external/v9_checkpoints/sa0_ckpt_v9_fold0_seed0.pt`) and
`metrics_v9_fold0_seed0.json`. **GPU hours spent this session: 6.23.**

### 59.1 ✅ Both guards fired, and GUARD 2 reproduced the dry run exactly
From the kernel log, line 4, before a single training step:
```
mounted code verified: quantiser fix present; _DrugBlock contextualises (|d|=0.4526)
and diagonal is exact (|d|=0.0e+00); 32,868 params/block
```
`0.4526` and `0.0e+00` are the same values the local dry run produced, so the mounted dataset version was
the one intended and the diagonal operator is exact **on Kaggle's torch**, not only on ours. Line 5 confirms
the arm was actually enabled: `drug_sa=True`. The checkpoint carries `cfg['drug_self_attn'] = True` and
**40 `perturb.*.drug_sa.*` tensors**, so `interaction_2x2.py` will not hit its refusal path.

### 59.2 🔴 The cross-run main effect is CONFOUNDED BY BATCH SIZE. My error, in the invocation.
`V9TrainConfig.batch = 48`. The three SA-off reference runs (`r0/r1/r2`) were launched with `--batch 96`.
**I did not pass `--batch` to this kernel**, so it trained at 48. Comparing the two runs therefore varies
batch size *and* drug self-attention *and* +26.9 % parameters (11,706,043 -> 14,852,059) at once:

**🔴 NON-COMPARABLE TABLE — do not quote any cell of it.** Three things vary at once.

| split | SA-off (batch 96) | SA-on (batch 48) | naive difference |
|---|---|---|---|
| unseen_cell | 0.5152 | 0.5165 | +0.0013 |
| unseen_compound | 0.5846 | 0.5703 | **−0.0143** |
| unseen_both | 0.4679 | 0.4669 | −0.0010 |

**None of these three numbers may be read as the effect of drug self-attention** and none will be quoted as
such. This is exactly the failure class review 003's C1 forced out of the design: the pre-committed estimand
[§58.4] is the **within-checkpoint** 2x2 interaction, computed on one set of weights where batch size,
parameter count and training trajectory are all held fixed by construction. The design survives my
invocation error; a main-effect design would have been destroyed by it.

If a clean main effect is ever wanted it costs another ~5.8 h at `--batch 96`, and that is a separate
decision, not something to be inferred from this table.

### 59.3 Step cost, measured rather than projected
1816.9 s/epoch against the SA-off run's 1684.5 s at half the batch -> **1.079x per epoch**, i.e. the drug
side is cheap, as §58.5 argued from the 34-vs-978 token asymmetry. The local `1.009x` full-width estimate
was a floor, as flagged; the true figure is larger but still small, and per *row* the SA-on run did twice
the optimiser steps.

### 59.4 Method rule 14
**An A/B invocation must pin every knob the reference run pinned, not rely on defaults matching.** The
defaults were never the baseline's settings; `--batch 96` was an explicit choice on the reference runs and
silence reverted it. Before any future arm is launched, diff the full `tcfg` of the intended baseline
checkpoint against the argv being sent, and log the diff in the kernel.

## 60. 🔴 The pre-committed reading REFUTES the hypothesis on the primary split (2026-09-21)

`interaction_2x2.py` on `sa0_ckpt_v9_fold0_seed0.pt`, n_eval 1500/split, batch 48, n_boot 20000, seed 0.
Artefacts: `model/results/v9_interaction_2x2_sa0_ckpt_v9_fold0_seed0.json`, run log
`v9_interaction_2x2_sa0_run1.log`. **Both branches fired on every split** (`|dY|max` atom 3.5–5.7,
context 5.3–9.2), so nothing here is void under method rule 2.

| split | S11 | S01 | S10 | S00 |
|---|---|---|---|---|
| unseen_cell | 0.51687 | 0.51156 | 0.42636 | 0.43648 |
| **unseen_compound** | 0.56877 | **0.58690** | 0.45198 | 0.46895 |
| unseen_both | 0.47750 | **0.50387** | 0.37954 | 0.39797 |

| split | atom w/ context | atom w/o context | interaction (median) | **interaction (paired mean)** | σ |
|---|---|---|---|---|---|
| unseen_cell | **+0.00530** [+0.00023,+0.01383] | −0.01012 [−0.01616,−0.00190] | +0.01542 [+0.00616,+0.02516] | **+0.01200** [+0.00920,+0.01480] | **8.4** |
| **unseen_compound** | **−0.01814** [−0.02635,−0.00916] | −0.01697 [−0.02353,−0.01093] | −0.00116 [−0.01080,+0.00998] **spans 0** | **−0.00938** [−0.01326,−0.00547] | **4.7** |
| unseen_both | −0.02637 [−0.03618,−0.01628] | −0.01843 [−0.02381,−0.00883] | −0.00794 [−0.02182,+0.00161] **spans 0** | −0.00258 [−0.00567,+0.00039] | 1.7 |

### 60.1 Applying §58.4 exactly as written
Estimand `interaction_paired_mean`, primary split `unseen_compound`, both ablations confirmed fired.
**−0.00938 at 4.7 σ** clears the ≥3 σ bar, so by the rule fixed before the spend this **is a result from
one seed** — and it is **negative**. The hypothesis of §58.7 was that in-loop re-contextualisation would
make the atom tokens stop hurting. On the primary split the atoms hurt **slightly more** with
contextualisation live than without it.

The plainer statement needs no interaction at all: **in the SA-on model the atom tokens still hurt.**
−0.01814 with contextualisation on `unseen_compound`, CI excluding zero; −0.02637 on `unseen_both`. Every
split's best configuration is `S01` or `S11` with atoms ablated (`S01 > S11` on both compound splits).
**Drug self-attention did not rescue the atoms.** That is the finding, and it cost 6.23 GPU-hours to get.

### 60.2 The splits disagree in SIGN, both at high σ — §58.3's warning, realised
`unseen_cell` gives **+0.01200 at 8.4 σ**; `unseen_compound` gives **−0.00938 at 4.7 σ**. Tight intervals,
incompatible signs, across strata this time rather than across seeds. `unseen_cell` is also the one split
where the atom effect turns **positive** under contextualisation (+0.00530), and the one split §57/§58.3
already removed as primary for being tiny and seed-unstable. So the single split that appears to support
the hypothesis is the split we pre-committed to *not* reading. Had the primary split not been fixed in
advance, §58.4 would have let me pick the +8.4 σ one.

### 60.3 🔴 The two estimators disagree by 8x on the primary split — and §60.6 shows I diagnosed it wrong
Median-based interaction −0.00116 (**spans zero**, width 0.0208); paired-mean −0.00938 (**excludes zero**,
width 0.0078). These are not two estimates of one number — mean-of-per-row-differences and
difference-of-medians are different functionals, and a 8x gap means the per-row double differences are
skewed. §58.2 chose the paired mean on a **variance** argument (it cancels row main effects inside each
draw), which says nothing about which functional is the honest summary. §58.4 pre-committed to it, so it
is what I read — but **a reader told only "−0.0094, excludes zero" would not learn that the median of the
same rows is indistinguishable from zero.** Both are in the JSON and both are quoted here.
The four per-row vectors are now dumped to `*_rows.npz` so the shape of that distribution is answerable
without another inference pass.

> **Superseded by §60.6.** I attributed the gap to skew in the per-row double differences. The npz says
> the gap is **loss of pairing**, not skew, and the paired estimand is the defensible one. Kept as written
> so the wrong diagnosis is on the record next to the measurement that replaced it.

### 60.4 🔴 The interpretive limit I have to raise against myself: S10/S00 are a badly damaged model
The context effect `S11−S10` is **+0.0905 / +0.1168 / +0.0980** — masking drug attention to the identity
costs about **0.10 median Pearson**, roughly 7x the entire atom effect. S10 and S00 are therefore not
"the model without in-loop contextualisation"; they are *this* model with a module it was trained on
crippled, operating far off its training manifold. A model that degraded may use **every** input worse,
which would produce a non-zero interaction with no drug-specific mechanism behind it whatsoever.

This is not a hypothetical. It is testable for free, and the test is running: the same 2x2 with `--key`
pointed at **`x_cell`**, the cell control expression — an input with **no mechanistic path to drug
self-attention at all**. If ablating `x_cell` also shows an interaction of order ±0.01, then the
interaction operator measures generic damage and **§60.1's 4.7 σ is uninterpretable in either direction**
— which would not rescue the hypothesis, because §60.1's plain statement (atoms still hurt at −0.018 with
context on) does not depend on the interaction at all.

### 60.5 Method rule 15
**A within-model ablation contrast needs a null-key control before its interaction is read as mechanism.**
Run the identical 2x2 on an input the ablated module cannot reach. Without it, "X matters more when module
M is live" is indistinguishable from "M is load-bearing and breaking it degrades everything". This should
have been in the design at §56, not added after the number arrived — logged as a design miss, and the
reason it is being run before the packet goes out rather than after.

### 60.6 ✅ The estimator question is settled from data already on disk, and my §60.3 diagnosis was wrong
The npz dump answers ASK 1 without another inference pass. Per-row double difference
`(r11−r01) − (r10−r00)`, n=1500 per split:

| split | mean | **median** | sd | skew | frac > 0 | Wilcoxon signed-rank p |
|---|---|---|---|---|---|---|
| unseen_cell | +0.01200 | **+0.00849** | 0.0552 | +0.06 | 0.621 | **1.3e−30** |
| **unseen_compound** | −0.00938 | **−0.00803** | 0.0766 | −1.75 | 0.422 | **1.4e−07** |
| unseen_both | −0.00258 | −0.00225 | 0.0595 | −1.14 | 0.460 | 0.072 |

**The mean and the median of the per-row contrast agree on every split.** So the 8x gap on
`unseen_compound` is **not** a skew artefact as §60.3 claimed. It is the difference between two things I
was calling "the median": the pre-committed statistic is the **median of the paired per-row contrast**
(−0.00803), while `interaction` is a **difference of four independently-taken medians** (−0.00116) — a
functional that discards the pairing entirely and is therefore much noisier. A nonparametric paired test
that assumes nothing about shape gives **p = 1.4e−07**, and 57.8 % of rows are negative.

⇒ **ASK 1 answers itself: the paired estimand is the defensible one**, and §58.2 picked it for the right
reason even though §58.1's justification was only about variance. The wide-spanning −0.00116 is not a
competing estimate of the same quantity; it is a worse estimator of it.
⇒ `unseen_both` is a **clean null** on this test too (p = 0.072), consistent with its CI spanning zero.

### 60.7 ✅ ASK 4 is independent, and it is the robust part
The atom effect under full attention, per row, needs no interaction and no diagonal arm at all:

| split | mean | median | frac of rows where atoms HELP |
|---|---|---|---|
| unseen_cell | +0.00562 | +0.00290 | 57.2 % |
| **unseen_compound** | **−0.01901** | **−0.01129** | **34.0 %** |
| **unseen_both** | **−0.01434** | **−0.01414** | **33.7 %** |

On both compound splits the atom tokens hurt on **two rows in three**, and mean, median and
difference-of-medians all agree in sign and rough magnitude. This survives every objection in §60.2–60.4,
because it never touches S10 or S00. **It is the finding this run bought:** giving the drug tokens in-loop
self-attention did not make v9's per-atom features worth their place.

### 60.8 ✅ THE NULL-KEY CONTROL — it first looked like a failure, and the per-row dump says otherwise
`--key x_cell`: the identical 2x2 with the ablation pointed at the **cell control expression**, an input
the drug self-attention block has no path to. Identical rows and chunks and weights — verified, not
assumed: `rows` arrays byte-identical between the two runs and the `r11` per-row vectors byte-identical on
all three splits. Artefacts `..._key-x_cell.json` / `..._key-x_cell_rows.npz`. Cost: 0 GPU-hours (local
inference).

**First read — apparent failure.** On the pre-committed estimand the control is *not* null:

| split | control interaction (paired mean) | atoms, for comparison |
|---|---|---|
| unseen_cell | −0.00074 [−0.00247, +0.00098] **spans zero** | +0.01200, 8.4 σ |
| **unseen_compound** | **+0.00616** [+0.00327, +0.00910] — **4.1 σ** | −0.00938, 4.7 σ |
| unseen_both | **+0.00519** [+0.00355, +0.00689] — **6.2 σ** | −0.00258, spans zero |

At that point the honest conclusion was that the operator has a non-specific component of ≈0.006 on the
compound splits and the atom interaction, at 1.5x that, is not cleanly above it. **That is what I believed
for about ten minutes.**

**Second read — the per-row vectors say the contamination is entirely in the MEAN's tail.**

| split | key | median | frac of rows > 0 | **sign test p** | skew |
|---|---|---|---|---|---|
| unseen_cell | atoms | +0.00849 | 0.6213 | **4.6e−21** | +0.06 |
| unseen_cell | x_cell | **+0.00000** | **0.5057** | **0.69** | −0.62 |
| **unseen_compound** | **atoms** | **−0.00803** | **0.4220** | **1.7e−09** | −1.75 |
| **unseen_compound** | **x_cell** | **+0.00016** | **0.5053** | **0.70** | +1.58 |
| unseen_both | atoms | −0.00225 | 0.4600 | 2.1e−03 | −1.14 |
| unseen_both | x_cell | +0.00000 | 0.5316 | 2.0e−02 | +1.84 |

The control's +0.00616 comes from a **right tail with skew +1.58 sitting on a distribution whose median is
+0.00016 and whose rows split 50.5/49.5** — i.e. no location shift at all. The atoms' −0.00938 comes from a
distribution whose **median is −0.00803 and whose rows split 42/58**. Under a sign test, which ignores
magnitude entirely, **the control is a textbook null on the primary split (p = 0.70) and the atom effect is
p = 1.7e−09.**

⇒ **The control passes, read as a location shift** — but see §61.1: that reading uses a statistic chosen
after the data were seen, and the **estimand of record failed**. The operator's non-specific component
lives in the mean's tail sensitivity; that diagnosis stands. What does not stand is treating it as a
rescue. §60.4's worry was real and is **not** bounded by this run.
⇒ And the contamination on `unseen_compound` is **positive** while the atom interaction is **negative**, so
it would mask a negative effect rather than manufacture one.

### 60.9 What this round establishes, stated at the strength the evidence supports
1. ✅ **Atoms still hurt under full attention.** −0.01814 [−0.02635, −0.00916] on `unseen_compound`,
   −0.02637 [−0.03618, −0.01628] on `unseen_both`; per row they hurt on **two rows in three**; mean, median
   and difference-of-medians agree. Touches neither S10 nor S00, so it inherits nothing above. **This is
   the finding the 6.23 GPU-hours bought, and §58.7's mechanism is dead.**
2. 🔴 **RETRACTED by review 005 C1 — see §61.1.** ~~The interaction on the primary split is negative and
   IS specific to the atom tokens.~~ The pre-committed estimand fails its own negative control, and the
   statistic under which the control passes was chosen **after** seeing the data. The reading is not
   admissible from this run. What the numbers are is unchanged; what they license is not.
3. 🔴 **The interaction's sign is stratum-dependent and that is not explained.** +0.01200 at 8.4 σ on
   `unseen_cell` against −0.00938 at 4.7 σ on `unseen_compound`, both surviving the control on
   `unseen_cell` and the sign test on both. One seed. **Not a result; an open question.**
4. ✅ One seed is sufficient for (1) and (2) by method rule 7 — within-run contrasts on identical rows and
   identical weights — but §58.3 stands: **the magnitudes are not established as seed-stable**, and (3)
   would need seeds to become anything.

### 60.10 Method rule 16
**A paired mean over per-row correlation differences is tail-sensitive; never report it alone.** Report the
median and a **sign test** on the same rows beside it, and put the null-key control through **all three**.
Here the mean said the control was contaminated at 4.1 σ and the sign test said it was null at p = 0.70;
the same pair of statistics separated a genuine location shift from a tail artefact in §60.6 as well. Two
reversals from one npz dump in one afternoon — the dump is now unconditional in `interaction_2x2.py`.

## 61. Review 005: SOUND-WITH-CAVEATS, one BLOCKING, and the sharper form of my own objection (2026-09-21)

`orchestration/bus/to_pi/005_review.md`, reviewed at commit `7e36626`. Six challenges. **All six upheld** —
two of them after I checked the arithmetic myself, which is the only reason to accept a factual claim.
Running tally: **45 of 48 challenges upheld.**

The verdict is scoped rather than global: *"The design failed; the conduct did not."*

### 61.1 🔴 C1 BLOCKING — the objection I made, in the form I failed to make it
I wrote in §60.8 that the control "passes, read as a location shift." The reviewer makes the point I
stopped one step short of:

> On `unseen_both` the null key returns **+0.00519 at 6.1 σ** while the hypothesis key returns −0.00258 at
> **1.7 σ** — the control is **larger and more significant than the thing it controls for.**

I had both numbers in my own table and did not put them next to each other. An estimand that returns 6.1 σ
where no mechanism exists and 1.7 σ where one is hypothesised is not measuring the mechanism. And the
rescue — reading the median and the sign test instead — uses a statistic **chosen after seeing the data**,
which makes it hypothesis-generating and nothing more. My own packet said as much ("the estimand was fixed
in §58.4 before the spend and the sign test was not"); the reviewer's instruction is to hold that line
rather than argue past it, and it is right.

⇒ **§60.9 item 2 is retracted.** The 2x2 interaction **is not interpretable from this run**, in either
direction. §60.8's diagnosis of *why* the estimand failed stands; its conclusion does not.

One thing the reviewer adds that I had wrong in the other direction: I called `x_cell` an input with "no
mechanistic path" and therefore null by construction. It is not. For any model that is not additively
separable in two inputs, the second-order mixed partial between them is **generically non-zero**, so a null
key was never guaranteed to return zero. It did its job anyway — as a control, not as a proof.

### 61.2 ✅ C2 — the correct estimand is a THIRD quantity, and we never emitted it
The reviewer accepts responsibility for review 004 C1 having pushed me from difference-of-medians to the
paired mean: *"That was right about which of those two is better and silent about the paired mean's
non-robustness to skew."* The estimand that separates the two keys cleanly on every split is **median of
the per-row contrast** — −0.00803 on the primary split — which is **neither** of the two statistics
`interaction_2x2.py` emits. I computed it ad hoc from the npz; it was never in the JSON.

**Pre-commitment for the next round, fixed now and before any spend:**

| | |
|---|---|
| estimand of record | **`median(per-row contrast)` plus a two-sided sign test on the same rows** |
| reported alongside | paired mean and difference-of-medians, always |
| built-in diagnostic | **mean/median disagreement** flags skew; if they disagree, the mean is not a location estimate |
| **gate** | **the null key must be null on the estimand of record BEFORE the hypothesis key is read** |
| forbidden | re-reading *this* run under the robust estimand |

### 61.3 ✅ C3 — ask 2 answered against me, with a free experiment attached
Yes, `S11−S10` at +0.09 to +0.12 (5–7x the effect measured) compromises the interaction, and the reviewer
classes the diagonal operator as **its own** recommendation from 004 C2: it preserves parameters, FFN and
residual scale exactly as claimed, and *"I did not anticipate that removing cross-atom mixing would itself
be catastrophic, which makes the ablated arm a poor stand-in for a model that never had it."*

The proposed discriminator is inference-only on the checkpoint we already have: blend the post-softmax
attention as **alpha·A + (1−alpha)·I** for alpha in {0, 0.25, 0.5, 0.75, 1} and plot the atom effect
against alpha. **Smooth and monotone means mechanism. Flat then collapsing means an out-of-distribution
artefact.** Zero GPU hours. Delegated as task W6 (`research/W6_alpha_doseresponse.md`) the same hour the
review landed; alpha=1 must be bit-identical to the default path and alpha=0 bit-identical to the existing
diagonal path, which is a test rather than a hope. Also noted: a sharper null key would be
**magnitude-matched** — `x_cell`'s main effect (+0.0355) is about twice the atom main effect (−0.0181), so
it was never like-for-like.

### 61.4 ✅ C4 — the stratum-dependent sign tracks the main effect, not a mechanism
Atoms *help* on `unseen_cell` (S11−S01 = +0.00530) and *hurt* on both compound splits (−0.01814, −0.02637),
and the interaction's sign follows. *"A single mechanism does not do that."* So the interaction is
inheriting the stratum's answer to a different question. §60.9 item 3 already called it an open question
rather than a result; it is now closed as uninterpretable rather than open.

### 61.5 ✅ C5 — the batch discrepancy does not reach the within-checkpoint quantities
Confirmed independently: all four cells come from one set of weights, one loop iteration, `--batch 48`,
chunk-mean semantics matched to §57, so batch size is held constant *inside* the contrast by construction.
The cross-run training-metrics table is the part that is non-comparable, and the reviewer is right that
flagging it in prose while leaving the table clean invites the wrong reading. **§59.2's table is now
labelled NON-COMPARABLE in place.**

### 61.6 ✅ C6 — both halves verified against the checkpoint, and both resolve
**(a) "32,868 params/block" — the reviewer is right and I can say what it is.** It is the parameter count of
the **throwaway probe block the guard itself constructs**, at `d_model=64, d_ff=128`, not of the trained
module. Reproduced exactly: 32,868. The block at training shape (`d_model=256, d_ff=1024`) is **786,504**,
and 786,504 x 4 blocks = **3,146,016**, which is the reviewer's figure to the digit. Nothing was mis-wired —
the other two numbers on that line (0.4526 and 0.0e+00) also describe the probe, and the probe is what the
guard is testing. But the line is cited as evidence about the trained model, so printing a 24x-smaller
count in it is a provenance defect. The guard will print the training-shape count too.

**(b) 🔴 The §47.5/§58 parameter pair 10,466,725 to 13,612,741 (+30.06 %) is RETRACTED.** It cannot be
reproduced from either checkpoint, and I found what it was instead. Reconciling everything:

| counting | SA-off | SA-on | increase |
|---|---|---|---|
| `parameters()` on the checkpoint's own cfg | 9,117,717 | 12,263,733 | **+34.5 %** |
| `state_dict` total (adds the `ppi.A`, `pathway.M_norm`, `gene_repr.vec` buffers, 1,864,068) | 11,706,043 | 14,852,059 | **+26.9 %** |
| **§47.5's pair** | 10,466,725 | 13,612,741 | +30.06 % |

§47.5's pair is `V9Config()` **defaults** (`l_control=2`, not the `l_control=1` that was trained) with
**both `use_gene_vectors` and `use_ppi` disabled** — 10,690,725 − 158,208 − 65,792 = **10,466,725**, exact,
and the same three deltas reproduce the SA-on figure. So it described a model differing from the trained one
in **three** ways, and the trainer's own log (`params 12.26M`, `ppi=True gene_vec=True`) contradicted it all
along. The delta 3,146,016 was right in every version, which is why the error survived.

⇒ The real capacity confound is **+34.5 %, larger than the +30.06 % quoted** — which strengthens rather than
weakens the case for the within-checkpoint design, and means any future capacity-matched arm must match
3.15 M parameters against a **9.12 M** base, not a 10.47 M one.

### 61.7 What survives, and it is the reviewer's own sentence
> With drug self-attention present and trained, ablating the per-atom drug tokens still **improves** median
> row Pearson on unseen compounds — S11−S01 = −0.01814 [−0.02635, −0.00916], per-row median −0.01129,
> improving 66 % of rows. A second in-loop contextualisation pass does not rescue the atom tokens.

No S10, no S00, no interaction, no null key, no cross-run comparison. It survives C1 through C6 intact, it
answers the objective, and it is what 6.23 GPU-hours bought. §58.7's mechanism is dead and the empirical
fact that motivated it is now stronger, not weaker.

### 61.8 Method rule 17
**A negative control must be pre-committed as a gate, not run as a reassurance.** State before the spend:
*this is the null key, this is the estimand, and if the null key is not null on that estimand the hypothesis
key will not be read at all.* Running the control afterwards and then choosing the statistic under which it
passes is how a failed design becomes a published claim. The control here was run before the packet went
out, which is the only reason this is a retraction inside one afternoon rather than a retraction after
review.

## 62. PRE-COMMITTED reading of the alpha dose-response, written before the numbers exist (2026-09-21)

Review 005 C3 proposed the dose-response and stated the qualitative reading: *"Smooth and monotone means
mechanism. Flat then collapsing means the interaction is an out-of-distribution artifact."* That is not yet
a rule — "smooth" and "flat" need thresholds, and method rule 17 says the null key must be a **gate** fixed
in advance rather than a reassurance run afterwards. So this section is written while
`alpha_sweep.py --n_eval 1500` is still executing and **before any output of it has been read.** The
operator was verified first (§W6 commit `02d8aa4`: alpha=1 bit-identical to the default path, alpha=0
bit-identical to the existing diagonal arm, 55/55 + 11/11 + 4/4).

### 62.1 The estimand, fixed
Per §61.2, the estimand of record is **`atom_effect_median_per_row`** — the median over rows of the
per-row difference between the full and atom-ablated predictions — with a **two-sided sign test** on the
same rows. The paired mean and the difference-of-medians are reported alongside and are **not** the
estimand. Primary split **`unseen_compound`**, as in §58.4. All five alphas score identical rows in
identical chunks, and `rows_sha` in the output must equal `160865d7b95cbc06` on that split, which is the
row set `interaction_2x2.py` used — so the curve's alpha=0 and alpha=1 endpoints are directly comparable
to §60's S10-S00 and S11-S01.

### 62.2 The gate, fixed BEFORE the result is read
`alpha_sweep.py --key x_cell` runs the identical sweep on the cell control expression. Method rule 17:

> **If the null key shows a monotone trend in alpha on the primary split, the hypothesis key's curve is
> not read at all.** No statistic chosen afterwards rescues it, and no partial reading is admissible.

This is the exact discipline whose absence cost §60.9 item 2. The null key is not expected to be flat at
zero — it has a large main effect (+0.0355) and §61.1 notes a mixed partial between any two inputs of a
non-separable model is generically non-zero. What is required is **no monotone alpha dependence**: the
mechanism claim is that the atom tokens' worth tracks how much cross-atom information flows, and that
claim is only distinguishable from generic degradation if degradation does not produce the same shape on
an input the module cannot reach.

### 62.3 The reading rule, fixed
Let `m(alpha)` be `atom_effect_median_per_row` on `unseen_compound`, and let the alphas be
0, 0.25, 0.5, 0.75, 1.

| pattern | reading |
|---|---|
| `m` **monotone across all five alphas** (no sign reversal of successive differences), endpoints differing by more than the alpha=1 bootstrap CI width, **and** the null key non-monotone | **consistent with mechanism** — the atom tokens' contribution tracks cross-atom information flow. One seed; magnitude not established. |
| `m` **flat for alpha in [0.25, 1]** (spread across those four within one CI width) with a **jump at alpha=0** | **the binary diagonal contrast was an out-of-distribution artefact.** §60.9 item 2 stays retracted and §60.4's worry is confirmed as the explanation. |
| `m` **non-monotone / reversing** | **uninterpretable.** No mechanism claim, and the operator is retired rather than re-specified. |
| null key **monotone on the primary split** | **the hypothesis curve is not read**, whatever it looks like. |
| any alpha with `dY_max` near 0 | **void at that alpha**, not a null [method rule 2]. |

### 62.4 What cannot be rescued by any outcome
Nothing here revives §58.7's mechanism. §61.7 stands independently and is unaffected by every branch
above: with drug self-attention present and trained, ablating the atom tokens **still improves** accuracy
on unseen compounds (−0.01814 [−0.02635, −0.00916], per-row median −0.01129, 66 % of rows). The
dose-response decides whether the *interaction* was ever measuring anything — not whether the atoms earn
their place. They do not, on either reading.

Cost of this round: **0 GPU-hours.** Local inference on a checkpoint already paid for.

## 63. The alpha curve, and a defect in §62's own threshold — written before the gate returns (2026-09-21)

`alpha_sweep.py --n_eval 1500 --n_boot 20000 --seed 0 --alphas 0,0.25,0.5,0.75,1.0`, SA-on checkpoint.
Artefact `model/results/v9_alpha_sweep_sa0_ckpt_v9_fold0_seed0.json`, log `alpha_full.log`. **0 GPU-hours.**

**The null-key gate (§62.2) has not returned. Under §62.2 the curve is therefore not read in this section.** — 🔴 **RESOLVED IN §65: the gate FAILED and this curve is never read.**
What follows is the arithmetic, and one defect in my own pre-commitment, both recorded before the gate
result exists so neither can be said to have been shaped by it.

### 63.1 ✅ The operator is anchored to §60 — row-identical and endpoint-identical
`rows_sha` in the output matches `interaction_2x2.py`'s row set on **all three splits**
(`434418d7677d3f9c`, `160865d7b95cbc06`, `02fb5b09d78a8d7e`), as §62.1 required. And the endpoints
reproduce §60's 2x2 cells exactly:

| | alpha=0 vs §60's S10−S00 | alpha=1 vs §60's S11−S01 |
|---|---|---|
| unseen_cell | −0.01012 = −0.01012 ✓ | +0.00530 = +0.00530 ✓ |
| unseen_compound | −0.01697 = −0.01697 ✓ | −0.01814 = −0.01814 ✓ |
| unseen_both | −0.01843 = −0.01843 ✓ | −0.02637 = −0.02637 ✓ |

Bootstrap intervals match too. A new code path reproducing a previously measured number **and its CI** on
byte-identical rows is the strongest available evidence that the continuous operator is the same
measurement as the binary one, extended.

### 63.2 The curve, in the estimand of record
`atom_effect_median_per_row` — median over rows of the per-row full-minus-ablated difference (§61.2):

| alpha | unseen_cell | **unseen_compound** | unseen_both |
|---|---|---|---|
| 0.00 | −0.00468 | **−0.00637** | −0.01122 |
| 0.25 | −0.00093 | **−0.00753** | −0.01195 |
| 0.50 | +0.00105 | **−0.00831** | −0.01267 |
| 0.75 | +0.00208 | **−0.00894** | −0.01266 |
| 1.00 | +0.00290 | **−0.01129** | −0.01414 |

Sign-test p at every alpha on every split is between 2.8e−02 and 6.9e−69; `dY_max` rises with alpha on
every split (1.32–1.72 at alpha=0 up to 3.53–5.95 at alpha=1), so no cell is void.

### 63.3 🔴 §62.3's magnitude threshold names a CI the script does not emit for the estimand
§62.3 required "endpoints differing by more than the **alpha=1 bootstrap CI width**". But §62.1 fixed the
estimand as `atom_effect_median_per_row`, and the only interval `alpha_sweep.py` emits is for
`atom_effect` — the **difference of medians**. Those are different statistics, which is exactly the
distinction §58.1 already caught me getting wrong once and §61.2 identified as a **third** quantity.

The estimand's own interval is computable from the 2x2 npz at zero cost for the two endpoints, and it is
**3.5x tighter** than the one §62.3 accidentally named (0.00487 against 0.01719 on the primary split).
Both numbers are given so a reader can apply either rule:

| split | strictly monotone? | endpoint span | estimand CI width (alpha=1) | difference-of-medians CI width |
|---|---|---|---|---|
| unseen_cell | **yes**, +0.00375 +0.00198 +0.00102 +0.00082 | 0.00758 | 0.00202 → **span exceeds** | 0.01360 → span does not |
| **unseen_compound** | **yes**, −0.00116 −0.00078 −0.00063 −0.00235 | **0.00492** | **0.00487 → span exceeds, by 0.00005** | 0.01719 → span does not |
| unseen_both | **no** — diffs −0.00073 −0.00072 **+0.00001** −0.00148 | 0.00292 | 0.00355 → span does **not** exceed | 0.00990 → does not |

Reading the threshold against the estimand's own CI is the faithful reading of §62 — the rule fixed the
estimand in 62.1 and the natural referent of "the alpha=1 bootstrap CI" is that estimand's. Using the
difference-of-medians width instead would repeat §58.1's error. But the honest summary of the primary
split is: **monotonicity is clean and strict; the magnitude criterion is met by 1.01x, a margin smaller
than the last digit worth quoting.**

And `unseen_both` **fails both halves** — its one positive successive difference is +0.00001, a hundredth
of its neighbours and plainly noise, but §62.3 was written with no tolerance and by its letter that is a
reversal. Recorded as a fail rather than waved through.

### 63.4 Method rule 18
**A pre-committed threshold must name the statistic its interval belongs to, and the script must emit
that interval.** Three pre-commitments this project has written have now referred to the wrong one of
several available intervals (§58.1 the power table, §61.2 the estimand, §63.3 the threshold). The pattern
is always the same: several plausible summaries exist, the code emits one, and the rule silently assumes
it is the relevant one. Fix in the code, not in the discipline — `alpha_sweep.py` will bootstrap the
**estimand of record** and dump per-row vectors, as `interaction_2x2.py` now does. That change is deferred
until the gate run finishes, so the gate is computed by **identical code** to the hypothesis run.

### 63.5 What is still true independently of all of this
§61.7 is untouched by every line above, because it uses only alpha=1: **with drug self-attention present
and trained, ablating the atom tokens still improves accuracy on unseen compounds**, −0.01814
[−0.02635, −0.00916], per-row median −0.01129 [−0.01395, −0.00908], 66 % of rows. The curve decides
whether the *interaction* ever measured anything. It does not decide whether the atoms earn their place.

## 64. The union graph exists — and my reason for term-size filtering was factually wrong (2026-09-21)

**A2 is built.** `network/outputs/v9/union_graph_v9.npz` + `..._provenance.json`, from
`network/scripts/build_union_graph.py` (delegated W7). Tests **7/7, run by me**; I read the test bodies as
well as their output, and they can fail — check 2 asserts STRING presence **and absence**
(`is_in_string == in_union`), check 3 walks every co-member pair of 50 sampled pathways, check 5 asserts
`raw > surviving`, check 7 asserts an edge unique to each source. Cost: **0 GPU-hours, nothing downloaded.**

### 64.1 The artefact
81,846 undirected edges over the 978 landmarks, each with a multi-hot `[STRING, Reactome, GO]` provenance
bit and the three sources' own unnormalised weights.

| source | raw | surviving size filter | edges contributed | density |
|---|---|---|---|---|
| STRING v12 | 13,001 edges | — | 13,001 | 2.72 % |
| Reactome | 800 terms | **784** | 59,860 | 12.53 % |
| GO:BP | 5,407 terms | **1,087** | 61,295 | 12.83 % |

| provenance | edges | | provenance | edges |
|---|---|---|---|---|
| STRING only | 3,465 | | GO only | 17,902 |
| Reactome only | 14,922 | | STRING+GO | 619 |
| STRING+Reactome | 2,164 | | Reactome+GO | **36,021** |
| | | | all three | 6,753 |

Jaccard: STRING∩Reactome 0.139, STRING∩GO 0.110, **Reactome∩GO 0.546**.

### 64.2 🔴 My justification for the filter was wrong on the facts
The W7 brief asserted: *"A 400-gene pathway contributes a 79,800-edge clique encoding almost no
specificity, and `A_copathway.npy` at 12.83 % density is the symptom of exactly this."* Checked against
`M_pathway_v9.npy`:

- Reactome's landmark-restricted terms have **median 14 members and a maximum of 79**.
- **Zero** terms exceed 200 members. **Zero** exceed 100.
- So `max_term=200` was **entirely non-binding** — it removed nothing, and the 16 terms dropped went for
  being *too small*.
- The largest single term contributes 3,081 edges, **0.64 %** of all possible pairs. There is no giant
  clique. The 12.53 % density is the **aggregate of 784 medium terms**, not the symptom of a few large ones.

The filter only starts to bite far lower, and it costs terms to do it:

| max_term | terms kept | unique edges | density |
|---|---|---|---|
| 200 / 100 | 784 | 59,860 | 12.53 % |
| 50 | 774 | 52,669 | 11.02 % |
| 30 | 706 | 37,489 | 7.85 % |
| 20 | 590 | 26,097 | 5.46 % |

The worker implemented the brief correctly. **The wrong premise was mine**, and it was the kind that is
cheap to check and was not checked before being written into a delegation.

### 64.3 ✅ This makes A1 a stronger idea, not a weaker one
The union is **17.13 %** dense (81,846 of 477,753 possible pairs). Message passing or an attention bias
over a graph that connects one pair in six is close to uniform smoothing — which is a concrete mechanism
for why §48 called our STRING null *"predictable"*, and the density is **intrinsic to co-membership over a
978-gene landmark panel**, not a fixable artefact of filtering.

So the union graph is **not useful as a static prior**, and that is exactly the argument for A1 rather than
against it: if a generic biological graph carries little information at this density, then **the thing that
would make it informative is a per-cell mechanism selecting which of its edges conduct.** Chromatin gating
turns a 17 %-dense generic graph into a sparse, cell-specific one — and the sparsification *is* the
hypothesis, not a preprocessing step.

⇒ **Consequence for A1's design, recorded before it is built:** the consumer must **not** be "message
passing over the union". It must be a **chromatin-gated edge weight**, where the gate does the
sparsification and the null is the same gate with chromatin mean-ablated. An arm that message-passes over
the raw union first would be testing the dense prior, which §64.3 predicts is worth little, and would
confound that with the gating question.

⇒ **Also recorded:** `Reactome∩GO = 0.546` means the two co-annotation sources are half-redundant, so the
"three-graph union" is closer to **STRING plus one co-annotation blob**. TxPert's monotone-improvement
result (§48.1, p < 0.027) was obtained on graphs with different provenance than ours, and should not be
assumed to transfer to two sources this correlated.

## 65. 🔴 THE GATE FAILS. The attention-masking probe is retired. (2026-09-21)

`alpha_sweep.py --key x_cell`, identical rows, identical chunks, identical weights, same code as the
hypothesis run — the script was deliberately not modified between the two (§63.4). Artefact
`model/results/v9_alpha_sweep_sa0_ckpt_v9_fold0_seed0_key-x_cell.json`. **0 GPU-hours.**

### 65.1 The null key on the primary split, in the estimand of record

| alpha | **null key `x_cell`** | hypothesis key `atoms` |
|---|---|---|
| 0.00 | **+0.01140** | −0.00637 |
| 0.25 | **+0.01303** | −0.00753 |
| 0.50 | **+0.01404** | −0.00831 |
| 0.75 | **+0.01502** | −0.00894 |
| 1.00 | **+0.01444** | −0.01129 |
| successive diffs | **+0.00163 +0.00101 +0.00098 −0.00058** | −0.00116 −0.00078 −0.00063 −0.00235 |
| endpoint span | **0.00304** | 0.00492 |

Sign-test p on the null key runs 1.0e−30 to 2.8e−64 at every alpha. `dY_max` 6.32–6.54, so nothing is void.
`rows_sha` on the control run matches the 2x2's row set on **all three splits**, so hypothesis and null key
ran on byte-identical rows.

### 65.1a The other two splits show the same pattern, so the failure is not primary-split-specific

| split | null-key span | hypothesis-key span | null as % of hypothesis | null key monotone? |
|---|---|---|---|---|
| unseen_cell | 0.00160 (+0.00797→+0.00957, peak at 0.25) | 0.00758 | **21 %** | no — rises then falls |
| **unseen_compound** | **0.00304** | 0.00492 | **62 %** | over four of five points |
| unseen_both | 0.00192 (+0.00186→+0.00378, peak at 0.75) | 0.00292 | **66 %** | over four of five points |

`unseen_cell` is the one split where the null key is comparatively quiet (21 %) — and it is the split
§57/§58.3 removed as primary for seed-instability, so it cannot be promoted now to carry the claim. On both
compound splits the null key reproduces **roughly two thirds** of the hypothesis key's alpha dependence.

### 65.2 The gate fails under BOTH readings of §62.2, so the wording's ambiguity does not matter
§62.2 said: *"If the null key shows a monotone trend in alpha on the primary split, the hypothesis key's
curve is not read at all."* "Monotone trend" is not as sharp as it should have been, so both readings:

- **As monotonicity.** The null key rises monotonically across the first **four** alphas and then dips by
  −0.00058. Strictly monotone? No. A monotone trend? Plainly yes — three substantial positive steps.
- **As magnitude.** Ignoring monotonicity entirely: the null key's alpha dependence is **0.00304 against
  the hypothesis key's 0.00492 — a specificity ratio of 1.6 : 1** on an input the drug-attention block
  **cannot reach.**

Neither reading supports attributing the hypothesis curve to a drug-side mechanism. — ⚠️ **AMENDED by
review 006 [§66.1]: drop the magnitude ratio from the justification.** It was not part of the
pre-committed rule, and leaning on it implies the trend alone would not have been disqualifying. It would.
The gate fails **on the monotone trend**, full stop; 1.6:1 is a description, not a criterion. **The verdict
is robust to how I read my own wording**, which is the only reason I can state it without the charge of
choosing the convenient reading — a charge review 005 C1 correctly brought against my last attempt.

⇒ **Under §62.2 the alpha curve is NOT READ.** §63's monotone, strictly-ordered, razor-thin-but-passing
primary-split curve is **not interpretable as mechanism.** Recording that plainly: it is the prettiest
curve this project has produced and it does not survive its own control.

### 65.3 The probe is retired, not re-specified
§62.3's last-resort row said a failure means *"the operator is retired rather than re-specified."* That
now applies, and for a reason bigger than one estimand:

| probe | null key | outcome |
|---|---|---|
| binary diagonal masking, 2x2 interaction | `x_cell`, paired mean | **failed** — null key 4.1 σ on the primary split, 6.1 σ on `unseen_both` where the hypothesis key gave 1.7 σ [§61.1] |
| continuous alpha masking, dose-response | `x_cell`, median of per-row contrast | **failed** — null key trend 62 % of the hypothesis key's, monotone over four of five points |

**Two different estimands, two different operators, one architecture, the same failure.** The conclusion
is not "find a third statistic" — it is that **masking a trained attention module and reading the change
in some other input's marginal value does not isolate mechanism in this model.** The masked arm is far off
its training manifold (`S11−S10` = +0.09 to +0.12, five to seven times any effect being measured, §61.3)
and a model in that state redistributes what it leans on across **every** input, including ones the masked
module never touches. That is a property of the architecture and the probe, not of a choice of summary.

⇒ ⚠️ **AMENDED by review 006 C1 [§66.2]: "the whole family is confounded" was too strong.** The
operator masked the **global token's** access to the atoms as well as atom–atom flow, because the global
token is position 0 of the drug sequence and the blend covered the whole matrix. The hypothesis is about
atom–atom only. So *this* operator is retired, and the family has an **untried member that matches the
hypothesis**: blend only `A[..., 1:, 1:]`. Free.

⇒ **No mechanism claim will be built on the FULL-MATRIX masking operator.** A mechanism claim needs an arm that
is *trained* in the configuration being compared, which is the between-run comparison this design existed
to avoid — so it costs GPU hours and must be priced as such, not smuggled in as a free inference-time
probe.

### 65.4 Method rule 19
**Write the gate as an inequality, not as an adjective.** §62.2 said "monotone trend" and §62.3 said
"strictly monotone"; had the null key landed between those two readings the verdict would have been mine
to choose, which is exactly the position a pre-commitment exists to prevent. State a number:
*the null key's endpoint span must be below X % of the hypothesis key's*, fixed before the run.

### 65.5 What survives, for the third time unchanged
§61.7, which uses only alpha=1 and touches no masked arm:

> With drug self-attention present and trained, ablating the per-atom drug tokens still **improves**
> median row Pearson on unseen compounds — −0.01814 [−0.02635, −0.00916], per-row median −0.01129
> [−0.01395, −0.00908], improving **66 %** of rows.

Three probes have now been aimed at *why* the atom tokens fail and all three have been retracted or
retired. The fact that they fail has survived every one of them, on three seeds SA-off (§57) and one seed
SA-on (§61.7), and it is the finding this round delivers.

**Round cost: 6.23 GPU-hours total**, all of it the single training run. Every probe, control, gate and
correction after it was free.

## 66. Review 006: the gate call stands, and the operator was masking the wrong thing all along (2026-09-21)

`orchestration/bus/to_pi/006_review.md`, reviewed at `5076f9c`. Six challenges, **all six upheld** — three
of them verified by me against the code or the arithmetic before acceptance. Running tally:
**51 of 54 challenges upheld.**

Verdict: *"Part 1: your gate call is correct... Part 2: I would not buy this run."*

### 66.1 ✅ Ask 1 answered independently: the gate is a fail, and my justification was half wrong
The reviewer reaches FAIL on its own reading, and corrects how I got there:

> *"Resolving a pre-committed ambiguity in the direction that lets you read your hypothesis, after seeing
> which reading does that, is the degree of freedom pre-commitment exists to remove."*

It also tells me to **drop the 1.6:1 magnitude ratio from the justification** — that ratio was not part of
the pre-committed rule, and leaning on it implies the trend alone would not have been disqualifying. It
would. §65.2 is amended: the gate fails **on the monotone trend**, full stop. The magnitude ratio is a
description, not a criterion.

**Ask 3 answered, and against the obvious move:** `unseen_cell` cannot be read. Removed as primary in
§58.3 *before* this run, null key not flat (21 %), and its hypothesis effect at alpha=1 is +0.00290 — there
is almost nothing there to rescue. *"Promoting the split where the result looks best, after the
pre-committed split failed its gate, is the textbook case the pre-commitment forbids."*

### 66.2 🔴 C1, verified in the code: THE OPERATOR HAS BEEN MASKING MORE THAN THE HYPOTHESIS IS ABOUT
This is the most consequential finding in six rounds on this question, and neither of us caught it until
now. Verified directly:

`model/v9/model_v9.py:139-140`
```python
D = torch.cat([(self.ln_u(self.w_u(u)) + self.type_drug).unsqueeze(1),
               self.ln_atom(self.w_a(atoms)) + self.type_atom], dim=1)
```
**The global token is position 0 of the drug sequence.** It carries `u` — ECFP4 (2048) + RDKit
descriptors (20) + Uni-Mol CLS (512).

`model/v9/modules_v9.py:244`
```python
A_blend = alpha * A + (1.0 - alpha) * I
```
applied to the **whole** L×L matrix, L = 1 + n_atoms.

⇒ At `alpha=0`, **row 0 attends only to itself**: the global token can no longer read the atoms, and no
atom can read the global token. So the operator removes **global↔atom flow together with atom↔atom flow**,
while the hypothesis (§56.1, as restated) is about **atom↔atom only** — a second in-loop
re-contextualisation of the atom tokens.

That is very likely where most of the 0.09–0.12 `S11−S10` comes from, and therefore why every estimand
built on this operator drowned: the global token holds essentially all the drug information that §50's
benchmark says these models actually use, and cutting its contextualisation is a far larger intervention
than removing atom mixing.

⇒ **§65.3 is amended.** *This* operator is retired, correctly. But "the whole family is confounded" was
too strong: the family has an **untried member that actually matches the hypothesis** — blend only the
atom–atom submatrix, `A[..., 1:, 1:]`, leaving row 0 and column 0 at their learned values. One line,
inference-only, reuses the entire harness, and should collapse `S11−S10` toward the size of the effect
being measured. **Zero GPU-hours.**

### 66.3 ✅ C2, arithmetic verified: it is not a three-source union
| | edges | share |
|---|---|---|
| Reactome ∪ GO | **78,381** | **95.77 %** |
| STRING-only | **3,465** | **4.23 %** |

With Reactome∩GO Jaccard at 0.546, the two co-annotation sources are largely one source counted twice.
⇒ **The "unioned biological graph" framing is dropped.** The artefact is a **co-annotation graph plus a
4.2 % STRING increment**, and §64/IDEAS A2 are corrected to say so. TxPert's monotone-improvement result
was obtained over sources that are not 55 % overlapping and is not assumed to transfer.
⇒ One free consequence worth keeping: the provenance bit makes it costless to report any future gate's
behaviour separately on the 3,465 STRING-only edges — the only subset that is not co-annotation.

### 66.4 🔴 C3: I re-imported a row-level estimand into a cell-level question
I proposed "median of the per-row contrast plus a sign test" as the estimand for a **chromatin claim on
unseen cell lines**. That is the estimand for the *drug* question, where the unit is the signature. For a
cell-level claim the unit of generalisation is the **cell line** — and this project has already been
burned by exactly this substitution:

| | row bootstrap | cluster over cell lines |
|---|---|---|
| the retracted chromatin effect | +0.0042 [+0.0036, +0.0049] | **+0.00036 [−0.0054, +0.0062]** |

⇒ Pre-commitment for any chromatin arm, replacing my proposal: **per-cell-line paired delta, reported
individually and summarised over cells; restricted to the five cells that HAVE chromatin tracks** (the
no-track cells cannot carry the treatment — this is §54's error); **cluster bootstrap plus per-cell
signs**; and the informative-width threshold stated **before** the run, accepting in advance that ~5
clusters give a wide interval.

### 66.5 🔴 C4: my own argument for the design, turned against it
I argued a 17 %-dense graph is near-uniform smoothing, so a per-cell mechanism selecting edges is what
would make it informative. The reviewer completes the thought: **if that is true, then *any* learned gate
helps and chromatin is incidental.** The gate's benefit may be **sparsification per se**.

So the free inference-time null (mean-ablate `E` inside the trained gate) answers *"does this trained gate
use cell-specific chromatin?"* — not the claim, which is *"chromatin selects which edges conduct"*, whose
null is *"a static learned edge weight does just as well"*. Only a **second training run** with `E` held
at its training mean separates them.
⇒ **The edge-gate arm costs ~11.6 GPU-h, not 5.8.** Budget both runs or neither. Shuffled-`E` is
second-line and only worth buying if the primary separates. The inference-time ablation may be reported
but **must not be presented as the null for the claim.**

### 66.6 ✅ C6: the margin is a stable tie, and it does not matter
The reviewer calls the 0.00005 margin a tie rather than a pass and asks whether its sign is
bootstrap-stable. Checked over 12 seeds: **12/12 positive**, margins +0.00001 to +0.00006, CI width
0.00486–0.00491. So it is a *consistent* tie, not a coin flip — and it is 0.2–1.2 % of the width, so it
bears no weight either way. Recorded as **"met to within bootstrap resolution"**. Moot regardless: the
gate failed.

### 66.7 The reviewer built a rescue and refused to use it
It noticed that `score_full` rises 0.4520 → 0.5688 (+26 %) across alpha, so the operator is partly a
model-quality dial, and that dividing by it makes the null key's trend nearly vanish (0.02522 → 0.02540,
non-monotone) while the hypothesis key's survives (34 % span). Then:

> *"I am not offering this as a reason to read the hypothesis curve. I constructed this normalisation
> after seeing that the pre-committed gate failed... That is precisely the move I objected to in review
> 005 C2, and I will not make it while criticising it."*

It also declined to compute the alternative normalisations (`1−score`, `score−chance`) **specifically to
avoid selecting among them.** Recorded as hypothesis-generating for a future pre-commitment, beside
§66.2's atom-only mask — and recorded as the standard to hold.

### 66.8 ✅ DECISION: the next 5.8 GPU-hours go to §46.5, not the edge gate
C5, accepted. `state.json`'s objective is *"establish whether v9 generalises to unseen cell lines better
than published SOTA"*, and review 001 C8 left that claim **inadmissible**: v9's 0.4734 is fold 1, one
seed, 21,151 rows; XPert's 0.383 ± 0.027 is a five-fold mean on 21,321 rows; **no XPert run exists on any
cold-cell fold.** §46.5 converts the project's headline claim from inadmissible to admissible on identical
rows, through `head_to_head_mdmt.py`, which exists and is already guarded.

The decisive argument is one I should have made myself: **§46.5 has no null-key problem.** It is a direct
paired comparison, not an ablation contrast — and the last two probes both died on their null keys.

Ranking adopted: **§46.5 first**, §50.5 (drug-blind retrain) second, **edge gate third** — *"a new
hypothesis stacked on a measured null (+0.000360), on a graph that is 96 % co-annotation."* The reviewer
explicitly does not defend the §50.5-versus-edge-gate ordering strongly; it does defend §46.5 first.

Free work queued ahead of any of it, both zero GPU-hours: **§66.2's atom-only mask** (pre-committed
first, per method rule 17) and the STRING-only-subset reporting from §66.3.

## 67. PRE-COMMITMENT for the atom-only mask, written before it is implemented (2026-09-21)

Review 006 C1 [§66.2] identified that every masking operator used so far also cut the **global token's**
access to the atoms, because the global token is position 0 and the blend covered the whole matrix. The
untried operator that matches the hypothesis blends only the atom–atom submatrix. Method rule 17 says the
gate is fixed **before** the run, and method rule 19 says as an **inequality, not an adjective**. Both are
applied here, before a line of it exists.

### 67.1 The operator — ⚠️ **definition superseded by §67.6 before implementation**
`alpha_atoms`: `A[..., 1:, 1:] <- alpha * A[..., 1:, 1:] + (1 - alpha) * I`, with **row 0 and column 0
left at their learned values**. Rows must still sum to 1, so the atom rows are renormalised over their
(atom-submatrix + preserved column 0) support rather than blended and left unnormalised — the
implementation must state which it does and a test must confirm rows sum to 1 to float tolerance.

Acceptance tests, all of which must be able to fail:
1. `alpha_atoms=1.0` **bit-identical** to the default path (0.00e+00).
2. `alpha_atoms=0.0` removes atom→atom flow **exactly**: perturbing atom 3 moves atom 1 by 0.00e+00.
3. `alpha_atoms=0.0` **preserves** global→atom and atom→global flow: perturbing the global token must
   still move atom 1 by a **non-zero** amount, and perturbing atom 3 must still move the global token by
   a **non-zero** amount. **This is the check that distinguishes the new operator from the retired one**
   — under the old full-matrix blend both are exactly zero.
4. Attention rows sum to 1 at every alpha.
5. Parameter count identical at every alpha; padding holds; no NaN.

### 67.2 The diagnostic that decides whether this operator is usable at all
The retired operator's disqualifying property was `S11−S10` = **+0.09 to +0.12**, five to seven times any
effect being measured. The prediction under §66.2 is that most of that came from the global↔atom cut.

> **Pre-committed: if `S11−S10` under the atom-only mask is not below +0.03 on `unseen_compound`, the
> operator is abandoned without computing any interaction.** A masked arm that still costs more than
> 3x the measured effect is not a usable counterfactual, and no estimand fixes that.

That number is chosen now, from the ratio that killed the last one, not after seeing the result.

### 67.3 The gate, as an inequality
Null key `x_cell`, identical rows and chunks and weights, same code, run **before** the hypothesis key's
curve is read.

> **The null key's endpoint span must be below 25 % of the hypothesis key's endpoint span** on the primary
> split, in the estimand of record. At or above 25 %, the hypothesis curve is **not read at all.**

25 % is chosen because the two failed gates came in at 62 % and 66 % on the compound splits, and because
`unseen_cell` — the split whose null key I was tempted to call quiet — sat at 21 %. Setting the bar at 25 %
means the quietest null key seen so far would only just have passed, which is the level of specificity
this operator has to reach before anything is attributed to it.

### 67.4 Estimand and primary split
Unchanged from §61.2, and this is a **drug-side** question so the row-level estimand is correct here — the
unit is the signature, not the cell line [§66.4 applies only to chromatin claims]:
`atom_effect_median_per_row` plus a two-sided sign test; primary split `unseen_compound`; paired mean and
difference-of-medians reported alongside; per-row vectors dumped; `rows_sha` required to equal
`160865d7b95cbc06`.

### 67.5 What no outcome of this can rescue
§61.7 is unaffected either way, and §65.5 stands: three probes have been aimed at *why* the atom tokens
fail, and all three are retracted or retired. This is a fourth attempt at the *why*. **The fact that they
fail is not under test and does not depend on it.**

Cost: **0 GPU-hours.** If it fails its own diagnostic in §67.2 or its gate in §67.3, it is abandoned and
recorded as abandoned, and the GPU decision in §66.8 (§46.5 next) is untouched either way.

### 67.6 🔴 REFINEMENT of §67.1's operator, before any line of it exists or any data — the first definition was wrong
§67.1 defined the atom-only mask as: blend `A[..., 1:, 1:]` toward the identity and renormalise the atom rows
over their atom-submatrix-plus-column-0 support. Working the algebra before writing the contract showed that
**this operator does not remove atom→atom influence**, and §67.1's own acceptance test 2 — *"perturbing atom 3
moves atom 1 by exactly 0.00e+00"* — would fail by construction:

Row `i` of `A` is **one** softmax over **all** keys, `A[i, ·] = softmax(q_i · k_· / √d)`. Changing atom 3 changes
`k_3`, which changes that row's normaliser and therefore **how atom 1 splits its mass between the global token
and itself**, even with every `A[1, j≥2]` blended to zero. Atom 3 would still reach atom 1 through the softmax
denominator.

**The operator that matches the hypothesis**, fixed now:

- **Row 0 (the global token): unchanged.** The global token keeps reading every atom, as learned.
- **Atom rows `i ≥ 1`:** `A'[i, ·] = α · A[i, ·] + (1 − α) · B[i, ·]`, where `B[i, ·]` is the softmax of the
  **same logits** restricted to the key set `{0, i}` — the global token and the atom itself — and zero elsewhere.
- Both `A` and `B` are row-stochastic, so every convex combination is; **no renormalisation**.
- `α = 1` delegates to `super().forward()`, so the default path is bit-identical by construction.

At `α = 0` each atom attends only to the global token and itself, with weights that depend on no other atom's
key. So **within one block**, perturbing atom 3 moves atom 1 by exactly zero.

**What this operator deliberately does NOT remove, stated before measurement:** across stacked blocks, atom 3 can
still influence atom 1 **through the global token** — atom 3 → global (row 0 intact) in block `b`, global → atom 1
in block `b+1`. That is the global↔atom pathway the hypothesis is *not* about [§66.2], so it is kept, and the test
suite **measures and prints** it on the full model rather than treating it as a defect.

§67.2's kill switch (`S11 − S10 < +0.03` on `unseen_compound`, else abandoned without computing any interaction),
§67.3's gate (null-key span below 25 % of the hypothesis-key span), and §67.4's estimand are **unchanged**.

## 68. §46.5 feasibility audit: three blockers, and XPert's training path does not apply its attention mask (2026-09-22)

§66.8 committed the next 5.8 GPU-hours to §46.5 — training XPert ourselves on `split_cold_cell_1..5` for a
paired head-to-head. Before buying an hour I audited whether their code can actually run here. It can, but
not for 5.8 h and not without two declared deviations. **Everything below cost 0 GPU-hours.**

### 68.1 ✅ Their trainer takes our folds natively
`external/xpert/code/XPert/train_xpert.py:27` — `--nfold` with help `'split, split_cold_drug,
split_cold_cell'`, and `:423` splits it on commas, so it accepts a fold list. And
`processed_data/l1000_mdmt_68830_subset.h5ad` carries **`split_cold_cell_1..5`, `split_cold_drug_1..5` and
`split_1..5` as native obs columns** — the same splits v9 was evaluated on. So the comparison is their
code, their data, their split definition, and `head_to_head_mdmt.py` already refuses to pair unless
`row_index` matches exactly.

Note for later: `l1000_mdmt_full_336852.h5ad` carries **tissue-holdout** splits instead
(`split_breast_*`, `split_lung_*`, `split_haematopoietic_*`) — a harder cold-cell variant we have never
run and which is theirs. Logged in IDEAS, not in scope here.

### 68.2 🔴 Blocker 1 — `num_epochs: 2500`, `patience: 50`. The run is not 5.8 h; it is unbounded.
`configs/config_l1000.yaml`: `num_epochs: 2500`, `init_epoch: 70`, `patience: 50`, `batch_size: 128`,
`train_lr: 0.004`. So runtime is early-stopping driven and could be anywhere from ~120 epochs to 2500.
**The §66.8 figure of 5.8 h was my estimate of a v9-shaped run and does not transfer.** Any purchase here
needs a wall-clock budget guard and a max-epoch cap, and the cap must be declared as a deviation because
a run stopped by our cap is not their training recipe.

### 68.3 🔴 Blocker 2 — `flash_attn` is a hard training dependency, and I first got this wrong
`models/model_utils.py:8` imports `flash_attn_func` unconditionally; it is called at `:226` and `:281`.

My first reading was that this is dead code under `sparse_flag: False`, by analogy with §47.6. **That was
wrong and I checked it before acting on it.** The branch at `:200` is `if output_attention:` — *not*
`sparse_flag`. `train_xpert.py:52` defaults `--output_attention` to `False`, so **training takes the
flash branch.** flash_attn is required to train XPert as released.

§47.6 is nonetheless confirmed from a second angle: `sparse_flag` is threaded through four signatures
(`:186, :251, :342, :361`) and **never branched on anywhere**. Two different switches; only one is dead.

### 68.4 🔴 Blocker 3 — the flash branch is called WITHOUT an attention mask — ⚠️ **NOT A NEW FINDING, see §69.2**
`SelfAttention.forward(hidden_states, attention_mask=None, sparse_flag=False, output_attention=False)`
uses `attention_mask` **only inside the `if output_attention:` branch**. The else branch is:
```python
context = flash_attn_func(query, key, value, dropout_p=self.dropout_p if self.training else 0.0)
```
No mask argument. `CrossAttention` at `:281` is the same. And the mask **is** constructed and passed:
`model_XPert.py:204` calls `get_unimol_drug_feat`, which builds the standard additive mask
`atom_musk = (1.0 - atom_musk_raw) * -10000.0`, and `:225` passes it as `drug_attention_mask`.

⇒ In the training configuration, **the drug attention mask is computed, passed, and then not applied.**

Measured on their own data, `processed_data/unimol_mdmt_1970.npz`, (1970, 122, 514):

| | |
|---|---|
| valid atoms per drug | min **5**, max 122, **mean 53.9** of 122 slots |
| padded slots | **134,172 = 55.8 %** of all slots |
| padded atom features | **exactly zero** (`|max| = 0`) |
| padded symbols | exactly zero |

Padded slots are not inert, because `self.key` and `self.value` are `nn.Linear(hidden, hidden)` with
default `bias=True` (`:179-181`, `:244-246`): a zero feature vector maps to the **learned bias**, so each
padded slot emits ~~one identical constant key and value~~ a key and value **distinct per position**, because position embeddings are added before attention [§70.3(a)]. At equal scores the share of softmax mass taken by
padding is:

| valid atoms | 5 | 27 | **53.9 (mean)** | 80 | 122 |
|---|---|---|---|---|---|
| padded mass | **95.9 %** | 77.9 % | **55.8 %** | 34.4 % | 0 % |

The model can learn to push the padded key's score down, but ~~all ~68 padded slots **share one key**~~ (false — they differ by position embedding, §70.3(a)), so
suppressing them requires a margin large enough to beat a count of 68 — and 117 for the smallest molecules.

### 68.5 What I am NOT claiming, and why this goes to review rather than into a result
I have been wrong three times criticising this paper (§36.3, §47.6, and the retractions in §46), and the
pattern each time was asserting from code-reading without exhausting alternatives. So, explicitly:

- I **cannot** verify what the **released checkpoint** was trained with. This is what *this code* does with
  *these defaults*, not necessarily what produced their published numbers.
- `flash_attn_func`'s signature takes no padding mask (variable-length masking is
  `flash_attn_varlen_func`), so the omission is not a mask passed by another name — but I have not run
  flash_attn to confirm its behaviour directly, because it is not installed.
- A padded slot's **value** is also the learned bias, so the model could in principle absorb a constant
  additive term harmlessly; that is an argument I can construct but not test without training.
- The cross-attention `cell_attention_mask` is passed as `None` at `:225` anyway, so for the cell side
  there is nothing to drop.

### 68.6 What this does to the §46.5 decision
Training "XPert as published" now requires choosing between **two different models**, and the choice
changes the benchmark:

| option | what it is | deviation to declare |
|---|---|---|
| **A** substitute `F.scaled_dot_product_attention`, **no mask** | faithful to their training path as coded | kernel swap only; same math |
| **B** substitute SDPA **with** their additive mask applied | faithful to their evident intent, and to how v9 does it | kernel swap **and** a behaviour change |
| **C** install `flash_attn` on Kaggle | no deviation | build risk, long install, may fail on the image |

Option A is honest to the code, B is honest to the intent, and **they are not the same experiment.** v9
masks padding properly — our tests assert padding moves real tokens by exactly 0.00e+00 — so B compares
two masked models and A compares a masked model against an unmasked one.

🔴 **RETRACTED in §70.3(b)** — measurement shows the released model scores *better* with its padding attended. ~~This also bears on §50 and on our own §61.7.~~ If XPert's atom attention is ~56 % diluted by padding during
training, then "XPert uses its atom features" is itself in question, which is **consistent** with §50's
finding that seven L1000 models barely use their drug features — and it is a second, independent mechanism
for it beyond the one Bai et al. propose.

⇒ **Not spending yet.** Packet 007 puts the option choice and the epoch cap to review first. The free
prerequisite work (building the `pert_idx`-indexed unimol array their loader expects, which we hold as an
`idx`/`feat` npz rather than `all_drugs_unimol_arr.npy`) proceeds regardless.

## 69. 🔴 Review 007 NOT-SUPPORTED: their recipe selects on the test fold, and I rediscovered our own shim (2026-09-22)

`orchestration/bus/to_pi/007_review.md`, reviewed at `726a5a2`. Seven challenges, **all seven upheld** —
four verified by me against the code and the released h5ad before acceptance. Running tally:
**58 of 61 challenges upheld.** Verdict **NOT-SUPPORTED**: the audit was right, the plan was not.

### 69.1 🔴 C1 BLOCKING, verified twice over: XPert's released recipe early-stops on the TEST fold
`external/xpert/code/XPert/utils.py:127-133`:
```python
tr_data   = data[data.obs[nfold] == 'train']
val_data  = data[data.obs[nfold] == 'valid']
test_data = data[data.obs[nfold] == 'test']

# for five-fold cross-validation
if val_data.n_obs == 0:
    val_data = test_data
```
Whether that fallback fires is an empirical question about their data, so I measured it on the released
`l1000_mdmt_68830_subset.h5ad`. **No split column has a `valid` level:**

| column | levels |
|---|---|
| `split_1`, `split_2` | train 55,064 / test 13,766 |
| **`split_cold_cell_1`** | train 47,509 / **test 21,321** |
| `split_cold_cell_2` | train 57,337 / test 11,493 |
| `split_cold_cell_3` | train 56,567 / test 12,263 |
| `split_cold_drug_1` | train 55,385 / test 13,445 |

`valid` level present: **False on every one.** So the fallback fires on **every fold**, and then
`train_xpert.py:537-542`:
```python
val_loss, ... = validate(model, val_dataloader, ...)   # val_dataloader IS the test set
early_stop = stopper.step(val_loss4, model, epoch, optimizer)
...
best_model = stopper.load_checkpoint(model, optimizer)  # best-by-test-loss checkpoint
```
⇒ **Training XPert to its own early-stopping criterion gives it test-guided model selection that v9 does
not get.** It also explains `:574`'s `test_metrics = val_metrics` as internally consistent rather than a
bug — under the fallback they are the same rows.

Note `split_cold_cell_1`'s test size is **21,321**, which is exactly the row count XPert's published
cold-cell number is quoted on. So the published figure is on these rows.

**⇒ DECISION, taken explicitly and recorded before any spend: option (a), as published, disclosed.** We
run their recipe including the fallback, and state the asymmetry in the paper. The reasoning is the
reviewer's and I accept it: the objective in `state.json` names *published* SOTA, so the comparison should
be against what was published. The consequence is asymmetric and must be written down in advance:

| outcome | reading |
|---|---|
| **v9 wins** | **conservative** — v9 beat a model that selected its checkpoint on the evaluation rows |
| **XPert wins** | **uninterpretable** — the win may be the test-guided selection |

Option (b), carving a real validation split out of `train`, is fairer but is no longer XPert-as-published
and could not be set against 0.383 ± 0.027. Running both doubles the cost to answer a question the
disclosure already answers.

**What I am NOT putting in the record:** any claim that XPert's *published* 0.383 inherits this. The
reviewer notes `--mode train` and `--mode test` differ in which metric survives, and neither of us can
tell from the code which produced Table R8. C1 is solid about **what we would be running**, and that is
all it is used for.

### 69.2 🔴 C2: my §68.4 "finding" was already in this repo, written by us, tested, and validated
`model/v9/_shims/flash_attn/flash_attn_interface.py` exists, dated 2026-09-20, and its docstring states
§68.4's finding **verbatim**:

> *"its attention branches the wrong way round from what the name suggests… `else:` # DEFAULT path:
> `flash_attn_func(q, k, v, dropout_p)`, NO mask argument… Reproducing their numbers therefore requires
> reproducing the flash path's semantics — including the fact that it ignores the mask."*

It is installed only when the real package is absent (`xpert_native_eval.py:62-73`), checked by
`test_xpert_compare.py`, and **it is the path that produced our 0.6933 on warm `split_2` against their
published 0.688 ± 0.011** (packet 001). So packet 007's ask 1 was not a live choice: **option A is already
built and already validated against a published number to within 0.005.**

⇒ 🔴 **This is a process failure and the reviewer says it is the third: §002 asked for saved predictions
already on disk, §005 asked for a paired-mean CI the harness already emitted, §007 asked for a decision
the repo had already made and tested.** Method rule 20 below.

What in §68.4 *was* new: the **quantification** — 55.8 % of slots padded, mean 53.9 valid atoms of 122,
padded features exactly zero, `bias=True` on the projections, and the resulting size-dependent
attenuation. The shim's docstring records that the mask is ignored; it does not record how much that costs.
That part stands.

### 69.3 ✅ C3, verified: the dense path is broken too, so "faithful to their intent" is not definable
`models/model_utils.py:212-217`:
```python
if attention_mask is not None:
    if attention_scores.size(-2) != attention_mask.size(-2):
        attention_mask_pad = torch.ones((attention_scores.size(0), 2), ...)
        attention_mask = torch.cat([attention_mask_pad, attention_mask], dim=-1)
    else:
        attention_scores = attention_scores + attention_mask
```
In the shape-mismatch branch — the case the branch exists to handle — the mask is padded, **reassigned,
and never added.** Only the `else` adds it. And the pad value is `torch.ones(...)`, i.e. **+1 additive**,
where their own convention from `get_unimol_drug_feat` is `0 = attend, −10000 = block`, so even when added
it biases those positions *upward*.

⇒ **Option B was never "their intent"; it would be our correction of their code — a third model.** If it
is ever run it must be labelled that way. It changes nothing about the decision: A is the published path.

### 69.4 ✅ C4: "absorbed harmlessly" does not hold as I stated it, and the counter-argument is testable
Padded keys are `b_k` and padded values `b_v` — constant but **not zero**, so the output is
`Σ a_i·v_i + m_pad·b_v` with `m_pad` running ~0 % to ~96 % across the library. Real-atom signal is
attenuated by `(1 − m_pad)`, which varies **systematically with molecule size**. A constant term would be
absorbable; this one is not constant.

The reviewer's counter-argument, which I think is strong: all padded keys are **identical**, so the model
can learn to suppress them by driving `q·b_k` low, and 2,500 epochs is ample. ⇒ Settled empirically by the
free experiment in §69.6.

### 69.5 ✅ C5, C6, C7 — three of my asks were overreach or wrong
- **C5 / ask 4:** one fold is admissible. Review 001 C8's defect was that **no** XPert run existed on
  **any** cold-cell fold; one fold scored by us on identical rows repairs exactly that. And **v9 itself is
  fold 1 at one seed**, so five XPert folds against one v9 fold reintroduces an asymmetry. **Buy fold 1.**
- **C6 / ask 3:** "unbounded runtime" was wrong. `patience: 50` with `init_epoch: 70` terminates in
  practice; `num_epochs: 2500` is a ceiling. The real problem is C1 — the monitored metric is test loss.
  If a wall-clock guard is imposed, record **per fold** whether early stopping or the guard fired, and
  exclude guard-tripped folds from any cross-fold statistic.
- **C7:** `train_xpert.py:573-574` computes `test_metrics` then overwrites it with `val_metrics`. Harmless
  under C1's fallback, but it becomes silent misreporting under option (b). ⇒ **Stated rule: never read
  their reported metrics.** Take predictions and score them with `head_to_head_mdmt.py`.

### 69.6 The free experiment that settles A-versus-B by measurement
Run the **released checkpoint** on warm `split_2` twice — once through the shim (unmasked, as coded) and
once with their additive drug mask applied — and see which reproduces our **0.6933**. Inference only,
every asset already on disk, **0 GPU-hours**. Reading pre-committed by the reviewer, adopted verbatim:

> *"If both give ≈0.693 the distinction is moot; if only the unmasked one does, A is confirmed as the
> published path by measurement rather than by reading."*

This also tests C4's counter-argument directly: if masking barely moves the released checkpoint, the model
already learned to ignore padding and A ≈ B in practice.

### 69.7 Method rule 20
**Grep the repo before asking a question, and before calling anything a finding.** Three packets have now
asked for something this project already had. The cost is not just the round trip — §68.4 was written up as
a discovery when a file in `model/v9/_shims/` had stated it two days earlier, which is a provenance error
in our own record. Before any "we should build X" or "I have found Y": `grep -ril` the concept across
`model/`, `research/` and `orchestration/`, and read `IDEAS.md`. The reviewer should not be the mechanism
by which we learn what we have already done.

## 70. PRE-COMMITTED: the masked-versus-unmasked test on XPert's released checkpoint (2026-09-23)

Written while `xpert_native_eval.py --mask_drug_keys` is executing and **before its output has been read.**
This is the experiment review 007 C2/C4 pre-committed in words; method rule 19 requires it as
inequalities. Released checkpoint, warm `split_2` test, `--max_rows 3000`, seed 0, identical rows across
every arm (verified by `row_index` equality, not assumed). The unmasked `asis` arm is already measured:
**0.6989** mean per-row delta Pearson.

**Masked arm definition.** Key-padding mask applied inside the shim to exactly the drug-keyed attention
calls (seqlen_k = 124 = dose, time, HG, 121 atom slots), with dose, time and slot 0 always attendable. A
counter refuses the run if the mask never applied.

### 70.0.1 Gate first: is the mask complete?
Masked+`noise` against masked+`asis` must give **`|dY|max` < 1e-4**. If padded content can still move the
output under the mask, the mask leaks (pooling, residual paths, anything) and **the masked arm is void** —
no comparison is read.

### 70.0.2 Then the reading, on the paired per-row difference masked − unmasked
| result | reading |
|---|---|
| **\|mean diff\| < 0.005 and CI spans 0** | **A ≈ B.** The released model is insensitive to whether its padding is masked; the distinction is moot for the benchmark |
| masked **lower** by > 0.005, CI excluding 0 | the released checkpoint **depends on** the unmasked path it was trained on. A is confirmed as the published behaviour by measurement; B is a different model and must never be substituted |
| masked **higher** by > 0.005, CI excluding 0 | padding **hurts the released checkpoint even at inference**, i.e. its predictions would improve if its own mask were applied |

Whatever the outcome, the §69.1 decision stands: the head-to-head runs option A, as published.

### 70.1 ✅ RESULT: the gate passes exactly, and the pre-committed reading is "masked lower"
Released checkpoint, warm `split_2` test, n = 3000, **rows and targets byte-identical across all five arms**
(verified by `row_index` and `deg_true` equality). Artefacts `model/results/xpert_native_split_2_test_*n3000.json`
and `external/xpert/padtest/*_profile.npy`. **0 GPU-hours** — local inference.

**Gate §70.0.1: PASS.** `|dY|max`(masked+`asis` vs masked+`noise`) = **0.000e+00**, against a threshold of
1e-4. Under the mask, padded content has **exactly zero** effect on the output — so the mask is complete,
there is no pooling or residual path by which the drug padding leaks, and the masked arm is valid. The mask
applied in 188 attention calls, counted.

**Reading §70.0.2**, paired per row, mean per-row delta Pearson (their `metrics.py` convention):

| arm | delta Pearson |
|---|---|
| unmasked, as released | **0.6989** |
| **masked** (drug keys, correctly) | **0.6844** |
| paired masked − unmasked | **−0.0144 [−0.0162, −0.0127]**, median −0.0051 |
| rows where masking helps | **35.4 %**, sign p = 2.8e−57 |

⇒ **Pre-committed row 2 applies: masking lowers the released checkpoint by more than 0.005 with the CI
excluding zero.** ⚠️ *Review 008 C3:* the **magnitude** is skew-inflated — mean −0.0144 against a median of
**−0.0051**, which clears the 0.005 threshold by only 0.0001. The robust statement is the **direction**:
masking hurts **64.6 %** of rows, sign p = 2.8e−57. "Masking costs 0.0144" is a mean pulled by a tail and is not
quoted without the median beside it. The released model *depends on* the unmasked path it was trained on. **Option A is
confirmed as XPert's published behaviour by measurement rather than by reading**, and option B is a
different model that must never be substituted for it. §69.1's decision stands and now has a measurement
under it.

### 70.2 The perturbation arms, for completeness — and why they were not the pre-committed test
Unmasked, the same 3000 rows, only the 512 feature channels of padded slots changed:

| perturbation | paired drop vs `asis` | rows hurt | `|dY|max` |
|---|---|---|---|
| `noise` (valid-atom scale) | +0.0998 [+0.0950, +0.1045], median +0.0594 | 87.2 % | 7.68 |
| `ones` | +0.0683 [+0.0643, +0.0723], median +0.0357 | 82.0 % | 7.59 |

In a correctly masked model these are **exactly zero** (v9's tests assert 0.00e+00; §70.1's masked arm gives
0.000e+00). So the released XPert model does not treat its padding as inert.

These were run first and are **not** the test review 007 pre-committed, for a reason I only saw after
reading `unimol_Embeddings.forward`: perturbing padded *features* changes padded **keys** as well as values,
so a model that had learned to suppress the specific padded keys would still be disrupted. The perturbation
could not separate "padding is attended" from "a learned suppression that does not survive key changes".
The masked arm can, and it is the one that answered.

### 70.3 🔴 Two corrections, one of them to both of us
**(a) Padded keys are not identical.** `models/model_utils.py:133-165`:
```python
input_embeddings = self.linear(input_embed)          # padded slot -> bias
input_embeddings[:,0,:] = HG_embed                   # slot 0 is overwritten by the HG token
... torch.cat([pert_dose_embed, pert_time_embed, input_embeddings], dim=1)   # -> length 124
embeddings = input_embeddings + position_embeddings  # every slot, padded ones included
embeddings = self.LayerNorm(embeddings)
```
Every padded slot is `LayerNorm(bias + pos_emb[i])` — **distinct per position.** §68.4 said *"each padded slot
emits one identical constant key"* and review 007 C4 built its counter-argument on *"all padded keys are
identical, so the model can learn to suppress them."* **Both statements are false**, for the same reason.
Recorded against both of us; it does not change any verdict, because §70.1 measured the thing directly.

It also explains review 007 C3 concretely: the sequence inside attention is **124** long (dose, time, then
122 slots with slot 0 carrying HG), while `get_unimol_drug_feat` builds a **122**-long mask. That is exactly
the `size(-2) != size(-2)` mismatch the dense branch then pads with `+1` and discards.

**(b) 🔴 RETRACTED: my §68.6 suggestion that padding dilution is "a second, independent mechanism" for §50.**
I proposed that XPert's atom attention being ~56 % diluted by padding might explain why these models
under-use drug features. The measurement says the opposite of dilution: the released model scores **better
with its padding attended than with it masked**, by 0.0144. A model trained unmasked has adapted to use
what it attends to. So padding is not demonstrably *hurting* the released XPert at inference, and §68.6's
"consistent with §50" line is withdrawn.

What the padding might be carrying is a hypothesis, not a finding: the padded slots are the only positions
whose count varies by molecule (122 − n_atoms), so an unmasked attention output can encode **molecule
size**. Untested. Whether an XPert *trained* masked would do better or worse is a separate question that
needs a training run and is not being bought.

### 70.4 What this closes
Review 007 **C2 and C4 are closed by measurement.** The remaining pre-spend items for §46.5 are
engineering, not decisions: the kernel with the shim on the path, the in-kernel unimol builder, a wall-clock
guard that records whether early stopping or the guard fired, and their `pip` dependencies installed into
Kaggle's disposable image rather than into `.venv-cuda` — whose dry-run would have pulled ~50 packages
including a numpy change into the environment every v9 result depends on.

## 71. PRE-COMMITTED: how the v9-versus-XPert cold-cell head-to-head will be read (2026-09-23)

Written before the XPert training run is launched, so before any number from it exists. Method rules 13,
16, 17, 19 apply, and review 006 C3 applies with full force: **for a claim about unseen cell lines the unit
of generalisation is the CELL LINE, not the row.**

### 71.1 What the fold looks like — measured, and why it forces a per-cell estimand
`split_cold_cell_1` holds out **8 cell lines** (32 in train, **zero overlap**). Test rows per held-out line:

| MCF7 | HT29 | MDAMB231 | HS578T | THP1 | CD34 | BJAB | H1975 |
|---|---|---|---|---|---|---|---|
| **10,969** | 5,837 | 2,188 | 1,074 | 820 | 295 | 84 | 54 |

**MCF7 alone is 51.4 % of the test rows; MCF7 + HT29 are 78.8 %.** A row-pooled mean over this fold is
therefore mostly a statement about MCF7, and a row bootstrap would report a tight interval around it. That is
exactly the substitution that produced the retracted chromatin +0.0042 (row CI [+0.0036, +0.0049]) against a
cluster estimate of +0.00036 [−0.0054, +0.0062]. It will not be made twice.

**Row coverage.** v9's saved predictions (`external/v9_mdmt_preds/v9_cc1_epi_seed0.npz`) cover **21,151 of the
21,321** test rows; the **170** missing are MCF7 154, BJAB 11, THP1 5; none extra. Pairing is on the
intersection (99.2 %), via `head_to_head_mdmt.py`'s `row_index` guard. BJAB loses 13 % of its 84 rows, so its
per-cell estimate is the noisiest and is flagged as such.

### 71.2 Estimand of record
For each held-out cell line `c`: **`d_c` = median over that cell's rows of `r_v9 − r_XPert`**, where `r` is
per-row delta Pearson against `y − ctl` (their `metrics.py` convention for the row score). All 8 `d_c` are
reported individually, with each cell's row count.

Summary: the **unweighted mean of the 8 `d_c`**, with a **cluster bootstrap over cell lines** (resample the 8
cells with replacement, 20,000 draws, seed 0), plus the **count of cells favouring v9** with a sign test. For
reference, with 8 cells: 8/8 gives p = 0.0078, 7/8 gives p = 0.070, 6/8 gives p = 0.29.

Reported alongside, **labelled as MCF7-dominated and not as a generalisation claim**: the row-pooled paired
mean from `head_to_head_mdmt.py`, which is the convention the field quotes.

### 71.3 The reading, as inequalities, with the asymmetry of §69.1 built in
| result | reading |
|---|---|
| cluster mean **> 0**, cluster CI **excluding 0**, and **≥ 7 of 8** cells favour v9 | **v9 generalises to these unseen cell lines better than XPert as published.** Conservative, because XPert selected its checkpoint on these test rows. One training run each. |
| cluster mean **< 0**, CI excluding 0, ≥ 7 of 8 favour XPert | **Uninterpretable as a model comparison** — the win may be the test-guided checkpoint selection. Reported plainly, not explained away. |
| anything else | **No cell-level claim.** The per-cell table and the row-pooled number are reported, the latter explicitly as MCF7-dominated. |

**Informative width, fixed in advance:** a cluster CI wider than **0.10** is uninformative whatever its
sign. (For scale: v9's fold-1 number is 0.4734 against XPert's published cold-cell 0.383, a gap of about
0.09, so an interval narrower than 0.10 is the smallest that could separate a gap of that size from zero.)

### 71.4 Reproduction check on XPert, separate from the comparison
Our XPert run's row-pooled fold-1 score is compared against their published cold-cell **0.383 ± 0.027**
(a five-fold mean). **Outside [0.302, 0.464] — three published SDs — is flagged as a possible reproduction
failure** before anything is concluded from the head-to-head. One fold against a five-fold mean is a weak
check; it can catch a broken run, not validate a good one.

### 71.5 What no outcome licenses
- **Seed stability.** One seed of v9 (0) and one of XPert (their default, 2024). §58.3's lesson stands: a
  tight interval within one run says nothing about variance across runs.
- **Any statement about XPert's published numbers.** §69.1: C1 is solid about what *we* run, and neither of us
  can tell from the code which mode produced their Table R8.
- **Anything about the other four folds.** Fold 1 only, by decision [review 007 C5].

### 71.6 ✅ v9's 170 missing rows are exactly the unfeaturisable compounds
Measured after packet 008 went out. All **170** rows absent from v9's cold-cell predictions have a compound
missing from `drug/outputs/drug_feature_index.json` (21,220 compounds), and **0 of 21,151** kept rows do. 28
distinct compounds. The exclusion is the documented, deterministic compound-featurisation drop in
`xpert_arm.py:94-110`, so pairing on the intersection means "the rows both models can score". It is not a
random subset of the fold: by cell it removes MCF7 154, BJAB 11, THP1 5.

### 71.7 🔴 AMENDMENT, required by review 008 as the condition of GO — committed before launch, no data seen
**What a guard-stopped run licenses.** Their early stopping monitors test `loss4` [§69.1], which normally
*favours* XPert — the reason a v9 win is "conservative". But if the 8.3 h guard fires while test loss is still
improving, the kernel loads XPert's best-*so-far* checkpoint. XPert is then **under-trained**, and a v9 win is
**inflated**, not conservative. The asymmetry reverses exactly in that case, and §71 never said so.

Pre-committed, as an inequality:

| training ended by | `counter_at_end` | reading |
|---|---|---|
| their early stopping (`finished`) | 50 by construction | §71.3 applies unchanged |
| **the wall-clock guard** | **≥ 45** of patience 50 | effectively converged; §71.3 applies |
| **the wall-clock guard** | **< 45** | **supports NO v9-win claim**, whatever the numbers — XPert was still improving when cut off |
| crash | — | void; the kernel fatals |

The **45** is my resolution of the review's *"a watchdog stop with the counter near patience is effectively
converged"*: 90 % of patience without improvement. The review's own inequality, taken literally, would forbid
any guard-stopped claim, since a guard stop always has counter < 50; I have taken its stated intent instead and
say so here so the choice is visible. The kernel writes `admissible_for_v9_win` from exactly this rule.

*Multi-session scope: see §78.5, committed before any multi-session launch. The ≥ 45 threshold is unchanged.*

**How convergence is measured** (review 008 C2). `counter_at_end = last_epoch_index − best_epoch`, where
`last_epoch_index` is parsed from their `Epoch {n}, Valid Total Loss` line and `best_epoch` is read from the
checkpoint their stopper wrote. This is exact: every epoch after the best is by definition non-improving. The
kernel's first draft recorded `early_stop_counter_hits` = the count of `EarlyStopping counter` lines, which the
reviewer showed is the **total** number of non-improving epochs over the run, because an improving epoch
resets the counter **silently** (`utils.py` `step()`: `self.counter = 0`, no log line). Verified in the code.
That field is renamed `nonimproving_epochs_total` and marked descriptive-only; the last logged counter is kept
as a cross-check against `counter_at_end`.

**Disclosed, not vetoed:** if `best_epoch < 70`, XPert's checkpoint was selected on the "accelerated"
`batch_weighted_loss` objective before the switch at `init_epoch`, and never benefited from the full one. The
kernel records `best_selected_before_init_epoch_70`.

**Per-cell uncertainty** (review 008 C4). Each of the 8 `d_c` is reported with its own bootstrap CI over that
cell's rows and its scored-row n, so that BJAB (73 scored of 84) and H1975 (54) are visibly noisy rather than
eight equal-looking numbers. The reviewer's point about the conjunction is recorded as the reason the
unweighted mean is kept: requiring **both** the cluster CI **and** ≥ 7/8 cells means a noisy small cell can
only cause a false **no-claim**, never a false v9 win.

**On 71.6.** The reviewer noted `n_dropped_test_unfeaturisable = 170` was already on disk in
`model/results/v9_xpert_arm_split_cold_cell_1_seed0.json`. It then corrected its own tally — 008a went out
before its review, so the ask was answered by us, not by it. Both are true: I answered it before review, and I
did so by recomputing a number the repo already held. A small method-rule-20 miss on my side, not a packet
defect. The reviewer's independent check agrees in full, and my converse check (0 of 21,151 kept rows
unfeaturisable) is the half that shows the exclusion is *exactly* the compound rule.

### 71.8 ✅ The analysis script, validated on a result we already have — before the run it will read
`model/v9/coldcell_h2h.py` implements §71 / §71.7 mechanically, including the verdict. Written while the XPert
run is executing, so it is fixed before its input exists. Validated on warm `split_2`, v9 against XPert's
**released** checkpoint, where the answers are already known:

| check | this script | the committed record |
|---|---|---|
| row-pooled paired mean | **+0.01182** [+0.01108, +0.01255] | **+0.0118** [+0.0113, +0.0127] (§43; its JSON rounds to 0.012) |
| paired rows | 13,615 | 13,615 |
| XPert on all its rows | **0.69320** | **0.6932** (`xpert_native_split_2_test.json`) |
| targets agree on shared rows | passed (`y_true`, `ctl_true` within 1e−4) | — |

The point estimates match exactly; the interval differs only in bootstrap draws. So the pairing, the row score
and the target guard are right. Artefact `model/results/v9_vs_xpert_split_2_percell_VALIDATION.json`.

**A byproduct, labelled as what it is.** The same run gives §43 a per-cell view it never had: v9 ahead of the
released checkpoint in **37 of 40 cells**, cluster mean of per-cell medians **+0.0115 [+0.0073, +0.0157]**, sign
p = 2e−8, with MCF7 only 15.9 % of warm-split rows. That says §43's win is not carried by one cell line. It is a
**post-hoc robustness check of an existing result**, not a pre-committed claim, and warm `split_2` shares its
cell lines with training, so it says nothing about unseen cells.

### 71.9 🔴 Launch 1 failed on an import, after every guard had passed
`apexblue/lincs-xpert-cc1` v1 ran ~3 min on one T4 and stopped with `KernelWorkerStatus.ERROR`. Its
`run_record.json` shows **all four guards passed** — dependencies importable with torch unchanged
(`2.10.0+cu128`), the split counts exact, the 2.24 GB unimol array built and round-tripped, `flash_attn`
resolving to our shim unmasked. Then the trainer died on its first import:
```
File "/kaggle/working/XPert/utils.py", line 15, in <module>
    from datasets.MyDataset import MyDataset
ModuleNotFoundError: No module named 'datasets.MyDataset'
```
**Cause, reproduced locally before fixing.** Their `datasets/` and `models/` directories have **no
`__init__.py`**, so they are namespace packages, and Python lets a *regular* package anywhere later on
`sys.path` win over a namespace one. The Kaggle image ships HuggingFace's `datasets`; `import datasets`
resolved to it. It never showed locally because `.venv-cuda` has no HF `datasets`. A stand-in regular package
placed later on the path reproduces the exact error, and the fix defeats it.

**Fix.** Two empty `__init__.py` files in the **staged copy** of `datasets/` and `models/`. No executed line of
their code changes; it is recorded as the third declared deviation in the run record. **GUARD D** added: in a
subprocess with the trainer's exact `PYTHONPATH` and cwd, `datasets.MyDataset`, `models.model_XPert`,
`models.model_utils`, `metrics` and `utils` must each resolve inside the staged copy, or the run refuses. It
would have caught v1 with an explicit message instead of a traceback.

**Cost** ~0.05 GPU-hours. **My error in reporting it:** I told the principal training was under way, on the
strength of a `RUNNING` status one minute before the failure. A status is not evidence that any guard passed,
let alone training — the log is. Method rule 21: **do not report a remote run's progress from its status;
report it from its log, or say it is unknown.**

### 71.10 🔴 Launch 2 failed on the config path — and exposed that my command was not their recipe
v2 passed all five guards, printed its arguments, and died:
`FileNotFoundError: configs/config.yaml`. Their `--config` defaults to `config`; the release has
`config_l1000.yaml` and siblings and **no `config.yaml`**. Their published training script,
`scripts/train.sh:15`, for `l1000_mdmt`:
```
python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol
       --nfold split_cold_drug_1,split_cold_cell_1,split_1 --dataset l1000_mdmt
       --use_gradscaler True --include_cell_idx True
```
The README gives the same `--config` and `--use_gradscaler`. **My kernel differed in three flags, not one**,
because I built the command from argparse defaults instead of from their scripts.

The crash was the lucky outcome. `--include_cell_idx` adds `cls_token` and `class_fc`; trained without it, the
checkpoint could not have been loaded by our harness under `strict=True`, so the run would have failed at
**prediction, after eight hours of training**. And **our own harness already said so**:
`xpert_native_eval.py:21` — *"`--include_cell_idx True` — a NON-DEFAULT flag. Running with argparse defaults
would have built a…"* — learned in packet 001, sitting in the very file the kernel reuses, and not read.
Method rule 20 again, and the most expensive near-miss of it so far.

**Fix.** The command is now their `train.sh:15` with only the fold list reduced. That is a strict subset:
`train_xpert.py:425-455` builds a fresh `XPertNet`, `init_weights()` and optimizer **per fold**. Their seed is set
once at `:402`, before the fold loop, so our fold starts from a fresh seed-2024 state rather than where their
`split_cold_drug_1` run would leave the RNG — a seed difference, disclosed, not a recipe difference.

**GUARD E** reads the trainer's own printed argument block and kills it on any mismatch with the published
recipe, before an epoch is paid for. It checks the **executed** namespace, not our argv, because their booleans
are `type=bool` and `bool("False")` is `True`. Replayed on v2's real log it flags exactly `config`,
`use_gradscaler` and `include_cell_idx`. **GUARD F** runs their `arg_parse()`, their `load_dataloader()` and
`XPertNet` on one real batch with one forward pass in the trainer's environment, and asserts the validation set
*is* the test set — so the disclosed fallback is verified in-kernel, not assumed. Each launch so far has failed
one stage later than the last; F is aimed at the next stage.

Cost of v2 ~0.13 GPU-h. **Method rule 22: "as published" is defined by the authors' scripts and README, never
by argparse defaults, and the executed arguments are checked against them.**

## 72. The atom-only operator passes its kill switch — and confirms review 006 C1 by measurement (2026-09-23)

`alpha_sweep.py --operator atom_only`, SA-on checkpoint, n_eval 1500, seed 0, rows identical to the 2x2 on all
three splits (`rows_sha` verified). Operator per §67.6, verified in W8 including a byte-identical regression of
the default path. **0 GPU-hours.**

### 72.1 §67.2's kill switch, applied exactly as committed
`S11 − S10` = `score_full(α=1) − score_full(α=0)`, the cost to the model of removing the thing under test.
Pre-committed: *abandon without computing any interaction unless below +0.03 on `unseen_compound`.*

| split | atom-only operator | retired full-matrix operator [§60] |
|---|---|---|
| unseen_cell | **+0.00356** | +0.09050 |
| **unseen_compound** | **−0.01448** | **+0.11679** |
| unseen_both | **−0.01469** | +0.09796 |

**PASSES**, by a wide margin. The operator is usable, and the gate (§67.3) is now running.

### 72.2 What the kill-switch quantity itself says — stated at the strength it supports
Review 006 C1 predicted that most of the retired operator's 0.09–0.12 damage came from cutting the **global
token** off from the atoms, not from removing atom–atom mixing. **Confirmed:** masking atom→atom alone costs
−0.014 to +0.004, against +0.09 to +0.12 for the old operator, on byte-identical rows and weights. Every
estimand that "drowned" in §60–§65 was drowning in the global-token cut.

And on both compound splits the masked model scores **higher**: 0.58325 against 0.56877 on `unseen_compound`,
0.49220 against 0.47750 on `unseen_both`. That is a measurement on a model **trained with** atom–atom attention,
evaluated without it at inference; it is **not** evidence that training without atom–atom attention would be
better, and it is not read as mechanism. It is recorded because it is the pre-committed kill-switch number and
because it is the opposite of what one would expect from removing a trained component.

**The atom-effect curve is NOT read here.** §67.3 requires the null key's span to be below 25 % of the
hypothesis key's first, and that run is executing.

## 73. Launch 3: GUARD F caught an out-of-memory kill — their loader needs ~24.7 GB on this fold (2026-09-23)

v3 passed GUARDs A–E (deps, split, unimol array, attention, module resolution) and **GUARD F refused** before a
single epoch, as it was built to. ~0.2 GPU-h. Session total now **6.61 GPU-h**.

### 73.1 What happened, and two defects in my own guard
The probe's stderr ends in their `tqdm` over the 21,321-row test set: ~2,000 rows/s to 86 %, then 29 → 19 rows/s,
a stall of over a minute at row 18,528, and death **with no Python traceback**. That is the signature of the OS
killing a process for memory, not of an exception.

Two defects in GUARD F made that harder to establish than it should have been: it recorded **that** the probe
failed but **not its return code** (−9 would have said "killed" outright) and **not its stderr** in
`run_record.json`; and the kernel log came back **empty** because the Kaggle CLI on Windows could not encode
tqdm's block characters (`'charmap' codec can't encode`) — recovered with `PYTHONUTF8=1`. Both fixed: GUARD F now
writes its return code, a `killed_by_signal` flag and the stderr tail stripped of progress bars into the record.

### 73.2 The mechanism, measured rather than inferred
`datasets/MyDataset.py`, `load_data`, once per row:
```python
drug_feat = tensor(drug_feat, dtype=torch.float32) if self.args.drug_feat != 'smi' else drug_feat
```
`torch.tensor` **copies**. Measured on 400 real test rows of `split_cold_cell_1`: **268 KB per row, 245 KB of it
the drug block**, and **all 58** drugs present stored as separate per-row copies. Their loader builds train,
val and test — and under the `val = test` fallback [§69.1] the test rows are built **twice** — so
47,509 + 21,321 + 21,321 = 90,151 rows × 268 KB = **24.7 GB**, before the 2.24 GB dense array, the h5ad and the
CUDA context, on an image with ~29 GB. Their published recipe does not fit Kaggle's memory as written.

### 73.3 The proposed fix, and its proof — put to review before any GPU hour
Replace that one line with: compute the **same** `tensor(drug_feat, dtype=torch.float32)` **once per drug** and
reuse it for every row of that drug. `model/v9/prove_mydataset_patch.py` loads their `MyDataset.py` verbatim and
a patched copy, builds both on the same 600 rows, and requires every field of every item to match:

| | |
|---|---|
| tensors compared | **6,000** |
| `torch.equal`, identical dtype and shape | **all** |
| unique storage on the sample | 164.6 MB → 33.4 MB (4.9×) |
| projected dataset RAM, train + val + test | **24.7 GB → at most 5.5 GB** (upper bound; double-counts shared drug storage) |

Safety of sharing, checked in their code: `__getitem__` returns the stored tuple as-is, collation is the default
stack (a copy into the batch), and **no in-place operation on `drug_feat` exists anywhere** in their source. So
no row can observe another row's use of the shared tensor.

The kernel applies exactly the proven patch — `_ORIG` and `_PATCH` compared as AST literals between the kernel
and the proof, identical — refuses if the target line does not occur exactly once, and records sha1 before and
after. **This is the first deviation that changes a line of their executed code**, so it goes to review
(packet 009) rather than straight to launch.

## 74. ✅ THE GATE PASSES: the atom tokens' harm on unseen compounds runs through atom-to-atom attention — ⚠️ **read with §75: the sign refutes the §58.7 rescue hypothesis, and the effect REVERSES on unseen cells** (2026-09-23)

`alpha_sweep.py --operator atom_only`, hypothesis key `atoms` and null key `x_cell`, SA-on checkpoint, n_eval 1500,
seed 0, `rows_sha` identical across both keys and equal to the 2x2's on every split. Operator per §67.6, verified in
W8 (T5: the retired operator gives exactly 0 where this one must not; default path byte-identical under
regression). Artefacts `..._op-atom_only.json`, `..._key-x_cell_op-atom_only.json`, both `_rows.npz`. **0 GPU-hours.**

**This is the first mechanism reading in this line of work — §56 through §73 — to survive its pre-committed
null-key gate.**

### 74.1 The gate (§67.3), as committed: null-key span below 25 % of hypothesis-key span, primary split
Estimand of record `atom_effect_median_per_row` [§61.2]:

| split | hypothesis key `atoms` | null key `x_cell` | null / hypothesis |
|---|---|---|---|
| **unseen_compound** | −0.00322 −0.00453 −0.00591 −0.00775 −0.01129 (span 0.00807) | +0.01538 +0.01509 +0.01523 +0.01502 +0.01444 (span 0.00093) | **11.6 % → PASS** |
| unseen_both | −0.00626 −0.00738 −0.00883 −0.01002 −0.01414 (span 0.00788) | +0.00489 +0.00454 +0.00462 +0.00440 +0.00378 (span 0.00111) | 14.1 % |
| unseen_cell | +0.00100 +0.00160 +0.00186 +0.00255 +0.00290 (span 0.00190) | +0.00868 +0.00866 +0.00878 +0.00900 +0.00957 (span 0.00089) | 46.8 % |

For comparison, under the **retired** full-matrix operator the null key moved **62 %** and **66 %** as much on the
two compound splits [§65.1a]. Under the atom-only operator it is essentially flat, and non-monotone. That is the
difference between an operator that measures generic damage and one that measures the pathway it names.

### 74.2 The reading (§62.3 row 1, inherited by §67.4), every criterion checked
On `unseen_compound`: **strictly monotone** (steps −0.00131, −0.00138, −0.00184, −0.00354); endpoint span
**0.00807 against an α=1 estimand CI width of 0.00487, i.e. 1.66×** (the §63.3 tie was 1.01×); and the α=0 and
α=1 intervals **do not overlap** — [−0.00433, −0.00226] against [−0.01395, −0.00908]. Sign-test p between 1.6e−10
and 1.0e−35 at every α. `unseen_both` agrees on every criterion (2.22×).

⇒ **Pre-committed reading: consistent with mechanism** — *with its sign* [§75.1]: **direct atom-to-atom self-attention increases the atom tokens' harm on unseen compounds**, which refutes the §58.7 rescue hypothesis in direction. In this trained model, **the harm the atom tokens do on
unseen compounds grows with atom-to-atom attention.** Cutting that one pathway — leaving the global token's
pathway intact — takes the per-row median atom effect from **−0.01129 to −0.00322**: about **71 %** of the harm
is carried by atom-to-atom mixing. The remaining −0.00322 persists without it and is still significant.

It agrees with the kill switch from the same run [§72]: removing atom-to-atom attention **raises** the trained
model's own score on both compound splits (+0.0145, +0.0147).

### 74.3 What this reading does NOT license — stated before anyone else has to
- **One seed.** One SA-on checkpoint. §58.3 stands: within-run contrasts are valid on one seed, magnitudes are
  not established as seed-stable.
- **Inference-time masking of a model trained WITH the pathway.** It says what the atom-to-atom pathway *does in
  this trained model*. It does **not** say that a model *trained without* atom-to-atom attention would be better.
  That is a trained arm — GPU hours — and it has to be priced and pre-registered on its own.
- **`unseen_cell` does not agree and is not claimed.** There the atoms *help*, the gate fails (46.8 %) and the
  span misses its CI (0.94×). It was removed as primary in §58.3 before any of this and is not promoted now.
- **Not a claim about atoms in general** — about these Uni-Mol atom tokens, in this architecture, on these
  splits.

### 74.4 Why it matters for the principal's mechanism line
§61.7 established that the atom tokens *don't earn their place*. §74 says *where the damage is*: mostly in
atom-to-atom mixing, not in the atom features themselves — with atom-to-atom attention cut, the atoms are close
to neutral. That turns "delete the atom tokens" into a testable constructive alternative: **keep the atoms, keep
the global↔atom pathway, drop atom-to-atom self-attention**, as a trained arm against the SA-on and SA-off
references. Logged as IDEAS A8, to be priced through review; nothing bought here.

### 73.4 Review 009: SOUND, GO — and why the patch is correct without needing the proof
`orchestration/bus/adjudicated/009_review.md`. Five challenges, all minor, all upheld. **Tally 68 of 71.**

**Correct by construction (C2).** In `load_data` the lookup is
`drug_feat = self.drug_feat[pert_id] if transigen_sdst else self.drug_feat[pert_idx]`, and the cache key is
`_k = pert_id if transigen_sdst else pert_idx` — **the same expression**. Nothing touches `drug_feat` between
the lookup and the patched line; `pert_idx` is still the raw scalar there. So `drug_feat` is a pure function of
the key and a per-key cache cannot return a wrong value for any row. The 6,000-tensor proof confirms the
*implementation*; correctness needs no sample. Recorded so the proof is not read as the basis for correctness.

**Declared, not result-bearing (C1).** It changes an executed line, which the `__init__.py` files did not, so it
is the fourth entry in `RECORD['deviations']` with sha1 before and after — and the run is still reported as XPert
as published.

**The reviewer's own error, conceded.** Review 008 checked the training command for a **forbidden** flag and never
for the **required** ones, and approved a kernel whose command would have built a different architecture; our
harness said so in capitals at `xpert_native_eval.py:18-24`. Its fourth error on the bus.

**Why that command would have run silently (C4).** `train_xpert.py:84-85` enters `autocast` only when a GradScaler
exists, i.e. only under `--use_gradscaler True`. Without it the model runs in **fp32**. The real `flash_attn`
accepts only fp16/bf16 and would have crashed on the first attention call; **our shim accepts fp32**, so it would
have trained to completion at the wrong precision with the wrong architecture. The shim removed the failure that
would have caught the error, which is why GUARD E is load-bearing. The shim's docstring now says so; asserting fp16
there would break the CPU harness. *(The shim copy in the uploaded dataset predates that docstring; behaviour is
identical, so it was not re-uploaded.)*

**Framing (C3).** The seed is set once at `train_xpert.py:402`, before the fold loop, and `split_cold_cell_1` is the
**second** fold of `train.sh:15`. So this run is **"XPert trained to its published recipe on `split_cold_cell_1`"** —
an independent draw — and **never** "their cold-cell run reproduced". Written into the run record.

**Memory (C5).** Post-patch dataset ≈ 3.5 GB; the larger new term is 20 DataLoader worker processes (train and
val at `num_workers=10` each), estimated at 4–6 GB; total ≈ 14 GB of 29 — comfortable but estimated. v4 therefore
samples system MemAvailable every 60 s for the whole run and writes the trace into the run record, and a trainer
killed by a signal is now recorded as `killed_by_signal`, not as a generic crash.

## 75. Review 010: the §74 mechanism, stated with its sign — and it reverses in the regime the objective is about (2026-09-23)

`orchestration/bus/adjudicated/010_review.md`, reviewed at `c0a39d2`. **SOUND-WITH-CAVEATS.** The reviewer reproduces
every number from the artefacts and calls §74 the cleanest positive result on the bus. Four challenges, all upheld,
the two major ones verified by me from the per-row dump before acceptance. **Tally 72 of 75.**

### 75.1 🔴 C1 — "consistent with mechanism" must carry its direction, and the direction refutes §56/§58
The atom effect is score(atoms) − score(atoms → chunk mean); negative means the atoms hurt. On `unseen_compound` it
runs **−0.00322 at α = 0 to −0.01129 at α = 1**. So removing direct atom-to-atom attention makes the atoms **less**
harmful, and restoring it makes them **more** harmful.

Packets 003–005 were built on the hypothesis that in-loop contextualisation would **rescue** the atom tokens [§58.7].
§61.7 showed it does not. §74 shows more: **the direct atom-to-atom part of that contextualisation is part of why
they hurt.** ⇒ **The §58.7 rescue hypothesis is REFUTED IN DIRECTION, not merely unsupported.** A record entry that
said only "consistent with mechanism" would have been read as support for rescue; it is not. §74.2 is amended in
place to carry the sign.

The statement of record, in the reviewer's words: ***direct atom-to-atom self-attention increases the atom tokens'
harm on unseen compounds.***

### 75.2 🔴 C2 — it REVERSES on `unseen_cell`, the regime `state.json` names as the objective
Paired `S11 − S10` = `r_full(α=1) − r_full(α=0)` per row, identical rows, from the npz; **negative means masking
atom-to-atom attention helps.** Reviewer's computation, reproduced by me to the digit:

| split | median | mean [CI95] | rows where masking helps | sign p |
|---|---|---|---|---|
| **unseen_cell** | **+0.00298** | **+0.00529 [+0.00307, +0.00752]** | **42.0 %** | 6.3e−10 |
| unseen_compound | −0.01498 | −0.02108 [−0.02474, −0.01749] | 65.3 % | 6.8e−33 |
| unseen_both | −0.01418 | −0.01255 [−0.01546, −0.00969] | 66.3 % | 3.5e−37 |

On unseen **compounds** masking helps; on unseen **cells** it **significantly hurts**. The project's stated objective
is generalisation to unseen cell lines, so an architecture without direct atom-to-atom attention would **buy the
compound regimes at the cost of the headline one.**

⇒ **Rule, from now on:** the compound gain is **never** quoted without the cell loss beside it.
⇒ **Pre-registered for IDEAS A8** before any trained arm: *a significant `unseen_cell` degradation disqualifies the
architecture, whatever it does on the compound splits.*

### 75.3 C3 — the paired version is the effect size
§72 reported `S11 − S10` as a difference of medians (−0.01448 on `unseen_compound`). That was adequate for the kill
switch — nowhere near +0.03 — but it loses the row pairing [§58.1, §60.3], and the paired statistic above is both
**stronger** on the compound splits and the one that exposes C2's sign flip. Wherever S11 − S10 is quoted as an
effect size, it is the paired version with its CI and sign test.

### 75.4 C4 — the operator removes DIRECT atom-to-atom attention, not intramolecular contextualisation
Within one block at α = 0, atom 3 → atom 1 is exactly 0.0. But across the full multi-block model, perturbing atom 3
still moves the output by **0.515** (W8 T11): atom → global in block *k*, global → atom in block *k+1*.
Global-mediated contextualisation survives. **Every claim in §74 and here is scoped to "direct atom-to-atom
attention".** Removing the rest would require cutting atom → global too, which reintroduces the global-token
entanglement review 006 C1 existed to avoid — so the scoped claim is the right one to make.

### 75.5 Why the result is not generic degradation, answered by the reviewer
The failure of §60–§65 was that the operator was a model-quality dial and every ablation effect rode it. Here model
quality goes **up** as α → 0 on the compound splits (S11 − S10 < 0), yet the atom effect **shrinks** as α → 0 —
the opposite of what quality scaling would predict. The curve does not ride the dial.

### 75.6 Queued, not run — each to be pre-committed before execution
- **A9, where the α = 0 residual lives** (−0.00322, sign p 1.6e−10). Three possible carriers: (a) gene → drug
  cross-attention, untouched by this operator; (b) atom → global within a block, then global → genes; (c) the
  cross-block leak of C4. Two inference-only cuts separate them — mask atom keys out of gene → drug
  cross-attention; then additionally cut atom → global at α = 0 — each with a §67.2-style kill switch and the
  `x_cell` gate, committed first. The reviewer expects the cross-attention cut to be a large perturbation, so the
  kill switch matters more there.
- **A8, the trained arm** — revised per ask 4: compare against **`sa0`** (fold 0, seed 0, batch **48**, 12 epochs)
  per row on identical rows on all three splits, **not** against r0/r1/r2 (batch 96). It is between-run, so it is
  pre-registered against the measured fold-0 between-seed sd, with a single-seed difference under ~2√2 × sd
  declared inconclusive; its predicted directions (compound up, cell down) are pre-registered; and **the cheaper
  question comes first** — if the aim is only unseen-compound accuracy, the inference-time mask on `sa0` already
  delivers it without training.
- **A10, the reviewer's memorisation hypothesis — hypothesis-generating only, built after the data.** *Atom tokens
  help exactly when the test compounds were seen in training, and hurt when they were not*: in `unseen_cell` the
  compounds are shared with training and atoms help; in both compound splits they are new and atoms hurt. That is
  the signature of atom-level features **memorising training compounds**. It **predicts atoms help most on a fully
  warm split.** Not read from this run. A test needs its prediction committed first.

## 76. PRE-COMMITTED: two tests of the memorisation hypothesis, written before either is computed (2026-09-23)

Review 010 offered, after seeing the data, that **atom tokens memorise training compounds**: they help where the test
compounds were seen in training (`unseen_cell`) and hurt where they were not (both compound splits). The split-level
contrast that suggested it cannot also test it. These two tests use data **nobody has examined**, and their rules are
fixed here first. Checked in code, not data: `build_splits` (`model/data.py:225`) makes `val` a random slice of rows
whose cell **and** compound are both in training — a fully warm split — and `test_coldcell` contains only compounds
seen in training.

### 76.1 T2 (primary) — dose-response on training exposure, WITHIN `unseen_cell`
Every compound in `test_coldcell` was seen in training, but some far more often than others. Memorisation predicts
the atom tokens help **more** for compounds seen **more often**. Nothing about this within-split variation has been
looked at.

- **Rows:** the 1,500 `unseen_cell` rows already scored by `interaction_2x2.py` on `sa0` (`rows_sha 434418d7677d3f9c`).
- **Per-row atom effect** `e_i = r11_i − r01_i`: per-row delta Pearson with atoms present minus atoms replaced by the
  chunk mean, **full attention** (the trained model as it is). Positive = the atoms help.
- **Exposure** `n_c` = the number of **training** rows whose compound is `c`, from `build_splits` on the same config.
- **Unit = the compound, not the row** — the hypothesis is about compounds, and rows of one compound are not
  independent. Per compound `c`: `E_c` = median of `e_i` over its rows. Statistic: **Spearman ρ(E_c, log n_c)**
  across compounds, with a **one-sided permutation test** (20,000 permutations of `n_c` over compounds, seed 0).
- **Null-key gate**, same construction with `x_cell`'s per-row effect from the `x_cell` 2x2 on the same rows. If a
  generic effect makes *every* ablation scale with exposure — e.g. well-trained compounds are simply predicted better
  — the null key will show it too.

| result | reading |
|---|---|
| atoms ρ > 0, permutation p < 0.01, **and** the null key not significant (p ≥ 0.05) with \|ρ_null\| < 0.5 × ρ_atoms | **consistent with memorisation** |
| null key fails its gate | the exposure correlation is **generic**, not attributed to atoms |
| atoms ρ ≤ 0, or p ≥ 0.05 | **not supported** |
| otherwise | inconclusive |

Known limitation, stated now: exposure correlates with *which* compounds are well studied, so a positive result is
consistent with memorisation, not proof of it.

### 76.2 T1 (secondary) — the reviewer's own prediction, on the warm split
Atom effect, median of the per-row contrast, full attention, on `val`, scored exactly as the other splits. The
reviewer's prediction: atoms help **most** on a fully warm split.

| result | reading |
|---|---|
| warm median-per-row > 0 **and** its CI lower bound above `unseen_cell`'s CI upper bound (**0.00378**) | **supported** |
| warm median-per-row ≤ 0, or its CI entirely below 0 | **refuted** |
| otherwise | inconclusive |

T1 needs the harness to score `val`, which it does not yet; it runs after T2.

### 76.3 T2 RESULT: **GRADED** memorisation is NOT SUPPORTED — the sign is reversed, specifically for atoms — ⚠️ *scope corrected by the reviewer, §76.4*
`model/v9/memorisation_t2.py`, run after the rule was committed at `55a3098`; artefact
`model/results/v9_memorisation_T2_unseen_cell.json`. 1,500 `unseen_cell` rows, **858 compounds** (3 rows whose compound
has no training rows are excluded); training exposure per compound from 1 to 1,616 rows (median 29). 0 GPU-hours.

| key | Spearman ρ(E_c, log n_c) | one-sided p (ρ > 0) | two-sided p |
|---|---|---|---|
| **atoms** | **−0.1189** | 0.9997 | **0.0006** |
| `x_cell` (null key) | +0.0330 | 0.167 | 0.333 |

**Pre-committed row applies: atoms ρ ≤ 0 → NOT SUPPORTED.** Memorisation predicted that the atom tokens would help
**more** for compounds seen **more often** in training. Within the split where every compound was seen, they help
**less** — and the null key shows no exposure dependence, so this is not the generic "well-trained compounds are
predicted better" effect the gate was there to catch.

**What is and is not read from the sign.** ρ = −0.119 at two-sided p = 0.0006 is a real, atom-specific *negative*
dose-response. But the committed table had no row for "significantly negative", so it is recorded as a **descriptive,
hypothesis-generating observation**, not a finding: *on seen compounds, the atom tokens' contribution falls as the
compound's training exposure rises.* One reading — constructed after seeing this and not tested — is that for heavily
trained compounds the global drug token and the training signal already carry what the atoms would add, so the atoms
contribute noise; that would make atoms matter most where a compound is least known, which is the opposite of
memorisation. It does not by itself explain why atoms hurt on compounds never seen at all, and it is not relied on.

⇒ ~~IDEAS A10's memorisation hypothesis is not supported by the within-split test. T1 ... can no longer rescue
memorisation as an explanation, since T2 contradicts the mechanism T1 would be evidence for.~~ **Overstated — see
§76.4.**

### 76.4 Review note 010b: T2 refutes the GRADED form only; the BINARY form is open and is what T1 tests
The reviewer checked the ordering (`55a3098` committed 13:53:42, the result written 13:55:14) and the design, and made
two corrections, both upheld:

- **Scope.** T2 tests a *graded* prediction — more exposure, more help. The hypothesis actually offered in review 010
  was *binary*: atoms help when the test compound was **seen** in training and hurt when it was **not**. "Seen once is
  enough" predicts ρ ≈ 0 inside `unseen_cell`, and T2 cannot tell that from no memorisation at all. So the record
  says **"graded memorisation not supported"**, and the binary form stays open until T1 reports. My §76.3 claim that T1
  could no longer rescue it was wrong.
- **A confound on the descriptive ρ = −0.1189**, so it is not read even descriptively as anti-memorisation. Exposure
  runs from 1 to 1,616 rows and is not random: heavily profiled compounds are largely **reference compounds**, which
  plausibly give stronger and more stereotyped signatures. If the global drug token already captures those, atom detail
  adds nothing for them — a negative ρ with no memorisation story in either direction. Descriptive, with an obvious
  uncontrolled covariate.



## 77. Launch 4: host memory fixed, then CUDA out of memory — their recipe needs ~15 GiB of activations on a 14.6 GiB card (2026-09-23)

v4 passed **all seven guards** — dependencies, split, unimol array, attention, module resolution, the one-batch probe,
and the executed-arguments check — and the memory patch worked: host MemAvailable never fell below **18.5 GB**
(sampled every 60 s, in the run record). The trainer then died on its first training forward:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 124.00 MiB. GPU 0 has a total capacity of 14.56 GiB
of which 82.81 MiB is free. ... 14.17 GiB is allocated by PyTorch
```
~0.13 GPU-h. Session total **6.74 GPU-h.**

### 77.1 Where the memory goes, measured
`xpert_gpu_mem.py`, their model and data, one training step (autocast fp16 forward + backward, the recipe's precision
under `--use_gradscaler True`) at batch 2, 4, 8, extrapolated. The flash SDPA backend is disabled to match a T4 (sm_75;
PyTorch's flash backend and the real `flash_attn` both need sm_80+):

| SDPA backend | per sample | projected at batch 128 |
|---|---|---|
| T4-like default (PyTorch picks memory-efficient) | 0.114 GiB | **14.6 GiB** |
| math only (materialises the attention matrix) | 0.691 GiB | 88.5 GiB |
| memory-efficient only | 0.117 GiB | 15.0 GiB |

PyTorch already uses the memory-efficient kernel on this hardware, so **attention matrices are not the problem;
stored activations are.** 14.6 GiB projected against 14.48 GiB in use at the moment of death: the measurement
matches the failure.

**A gap in my GUARD F:** its one-batch probe ran the forward pass **under `no_grad`**, so it never held training
activations and could not have caught this. It is now a real training step at batch 128.

### 77.2 Why neither obvious fix is acceptable
- **Gradient accumulation (2 × 64) is NOT exact for this recipe.** `train_xpert.py:104-106`: for epochs below
  `init_epoch = 70` they train on `batch_weighted_loss = sqrt(loss1/num_samples)·a + ... + sqrt(loss3/num_samples)·c`
  — **square roots of whole-batch means.** `sqrt(mean over 128)` is not the average of two `sqrt(mean over 64)`, so
  splitting the batch changes the loss itself.
- **DataParallel over both T4s** would keep the batch whole (outputs gathered to GPU 0, loss computed on all 128), but
  their forward hard-codes `drug_feat.to(self.device)` (`model_XPert.py:193`) and holds `drug_HG_embed` as a plain
  tensor attribute rather than a buffer, so replicas would mix devices. It needs edits to their model code.

### 77.3 The fix: activation checkpointing, applied at runtime, proven exact
Store activations only at layer boundaries and recompute the rest in the backward pass. Same batch 128, same loss,
same gradients — more compute, not different maths. `model/v9/xpert_ckpt_patch.py` wraps `Encoder.forward` and
`crossEncoder.forward` (every layer of both their encoders) with `torch.utils.checkpoint`, `use_reentrant=False`
(preserves the RNG state for dropout replay and the autocast state), active only in training with grad enabled.
Applied by a wrapper that then calls their `train_xpert.main()`, so **their files stay verbatim**.

`model/v9/prove_checkpoint_exact.py` imports and applies **that module**, and compares one training step with and
without it, same weights, inputs and seed, under autocast fp16:

| | |
|---|---|
| loss, unpatched twice and checkpointed | **7.56921387**, all three |
| max \|grad\| difference, unpatched vs unpatched (the GPU's own noise floor) | 4.63e−05 |
| max \|grad\| difference, unpatched vs checkpointed | **4.35e−05 — inside the floor** |
| projected memory at batch 128 | **14.95 GiB → 3.73 GiB** |

The comparison is against the noise floor because GPU backward kernels are not bitwise deterministic; checkpointing
cannot be distinguished from running the same step twice.

The kernel embeds the patch source **generated from the repo file** — verified byte-identical — and declares it as the
fifth deviation. GUARD F now runs a batch-128 **training** step with the patch and refuses if peak GPU memory exceeds
13.0 GiB. Sent to review as packet 011 before relaunch, because it changes how their model's forward executes.

### 77.4 Review 011: GO on checkpointing — and it may make the run fit but not count
`orchestration/bus/adjudicated/011_review.md`. **SOUND-WITH-CAVEATS**, three challenges, all upheld. **Tally 75 of 78.**

The reviewer confirms checkpointing is exact and that the proof tested the hard case: all four dropout rates are 0.1 and
the proof ran in training mode, so a within-noise gradient match means the dropout masks were replayed; peak memory
falling 3.8× shows the wrapper actually executed. The proxy loss is sufficient (ask 2): exactness is a property of the
recomputed layers, and exact recomputation gives exact vector–Jacobian products for any upstream gradient. It is the only
exact option of the three (ask 4). Declared as the fifth deviation; it does not bear on "as published".

- **C1 (MAJOR).** Checkpointing adds roughly one extra forward through the encoders per step, so fewer epochs fit inside
  the 8.3 h guard — and §71.7 says a guard stop while test loss is still improving supports **no** v9-win claim. The fix
  that makes the run **fit** may make it **inadmissible**, and XPert's convergence epoch is unknown. ⇒ **v5 runs in
  MEASURE_ONLY mode**: everything through GUARD F, then five timed steady-state training steps and five validation
  forwards on the real T4, an epochs-within-budget projection into `run_record.json`, and a clean stop. The full run is
  decided from that number, before any 8-hour commitment, rather than discovered through `stopped_by == 'watchdog'`.
- **C2.** GUARD F ran forward + backward with no `optimizer.step()`; Adam allocates its state lazily on the first step,
  so the probe measured a lighter step than the one it guards — the §32 class. It now runs a real Adam step under a
  GradScaler before measuring. Its CUDA context is released because it is a subprocess.
- **C3.** The patch docstring cited the earlier inline-copy proof run (5.9e−05 within 8.7e−05); it now cites the run of
  the committed module (4.353e−05 within 4.630e−05), and the kernel's embedded copy was regenerated and re-verified
  byte-identical.

## 78. v5 measured the real step time: one 8.3 h session is ~31 epochs, and an ADMISSIBLE XPert run needs ~210 (2026-09-23)

v5 ran `MEASURE_ONLY` [§77.4]: every guard, then GUARD F's batch-128 training step **with a real Adam step under a
GradScaler** (review 011 C2), five timed steady-state training steps, five validation forwards, and a clean stop.
`external/kaggle_out/cc1_v5/run_record.json`, `stopped_by: measure_only`. **~0.06 GPU-h. Session total 6.80 GPU-h.**

| quantity | value |
|---|---|
| training step, batch 128, checkpointed, optimizer included | **2.265 s** |
| validation step | 0.411 s |
| peak GPU memory in the training step | **3.74 GiB** (projected 3.73 in §77.3 — the proof's number held on the T4) |
| batches per epoch, train / val | 372 / 167 |
| **projected epoch** | **911 s = 15.2 min** (92.5 % training) |
| setup before training | 208 s |
| **epochs that fit in the 8.3 h guard** | **31.6** |
| host MemAvailable minimum | 20.0 GB |

### 78.1 What that number does to the run — it cannot be admissible in one session
Their recipe [§68.2]: `num_epochs 2500`, `patience 50`, `init_epoch 70` (the loss switches from
`batch_weighted_loss` to the full objective at epoch 70), and the LR halves at epoch 40.

- **31.6 epochs never reaches epoch 70**, so the checkpoint would be selected on the "accelerated" objective only.
- **§71.7 needs `counter_at_end ≥ 45` for a guard stop to count.** In 31 epochs that requires the best epoch to be
  **epoch 0 or earlier** — i.e. it is **impossible**. A single-session run is inadmissible **by construction**, not
  by bad luck.
- **A capped run is uninformative in BOTH directions** — *corrected by review 012 C3; I first wrote that an
  XPert win would be the informative outcome.* A v9 win is inadmissible under §71.7; an XPert win is
  **uninterpretable** under §71.3, committed before any of this, because XPert's checkpoint is selected on test
  loss — and that holds at epoch 31 exactly as at epoch 214, since `save_checkpoint` fires on every test-loss
  improvement. **Not bought.**

### 78.2 What an admissible run costs, anchored on the only convergence number available
Their **released** warm-split checkpoint stores `epoch: 164` (`l1000_mdmt_warm_split.pth`; the two pretraining
checkpoints store 198 and 244). With patience 50, that run lasted **≥ 214 epochs**. The cold-cell fold may converge
earlier or later; 164 is the only anchor there is, and it is stated as an anchor, not a prediction.

| epochs | GPU-h, one T4, checkpointed | 7.95 h training sessions |
|---|---|---|
| 70 (reach the loss switch) | 17.7 | 2.2 |
| 120 (earliest possible admissible stop: best at 70, +50) | 30.4 | 3.8 |
| **~210 (best at 164, +45)** | **~53** | **~7** |

Against a **30 h/week** quota. **The §66.8 estimate of 5.8 GPU-h — already withdrawn in §68.2 as "a v9-shaped
run" — was low by about 9×.** Every earlier cost statement for §46.5 is superseded by this table.

### 78.3 Two further facts any multi-session design has to face
- **Their own resume path is lossy.** `train_xpert.py:480-490` (`--resume_from`) reloads model weights only: the
  optimizer load is **commented out**, the LambdaLR schedule restarts from epoch 0 (so the LR would jump back to
  0.004), and the `EarlyStopping` best score and counter are not restored. Chaining sessions through it is **not**
  their recipe. A faithful continuation needs our own full-state checkpoint (model, Adam, GradScaler, scheduler,
  stopper, RNG) — a sixth declared deviation, and one that is *closer* to the recipe than their resume.
- **The second T4 is idle.** Kaggle charges T4×2 by session-hour. DataParallel keeps the batch at 128 and the loss
  on the gathered batch (LayerNorm only, no BatchNorm, so per-replica statistics cannot change the maths), and at 64
  per GPU (≈ 7.5 GiB of activations) checkpointing may not be needed at all. Estimated, **not measured**: ~7 min per
  epoch, **~25 GPU-h** for ~210 epochs. It needs a runtime patch: `forward` moves inputs to `self.device`
  (`model_XPert.py:188-198`), `drug_HG_embed` is a plain tensor fixed on `cuda:0` (`:135`), and a DataParallel
  `state_dict()` gains a `module.` prefix that their strict test-time load would reject.

⇒ **Packet 012 takes this to review before any further GPU-hour is committed to §46.5.** The options it puts are:
the full recipe on one T4 (~53 h); DataParallel measured first (~0.15 h) and then ~25 h if the measurement holds; or
no XPert training at all, with the cold-cell comparison made against the published five-fold mean plus the warm
split head-to-head we already have on their released weights — which loses §71's per-cell paired estimand.

### 78.4 Review 012: measure DataParallel first; never resume through their path; §71.7 needs scoping, not rewriting
`orchestration/bus/adjudicated/012_review.md`, at `b85f835`. **SOUND-WITH-CAVEATS**, five challenges, all upheld.
**Tally 80 of 83.** The reviewer reaches point 1 independently (in 31 epochs `counter_at_end` is at most 30, and early
stopping cannot fire either) and endorses the recommendation: **buy the ~0.15 GPU-h DataParallel measurement, then
choose between O2 and O4 on the measured number; do not buy O3.**

- **C1 (MAJOR) — their resume can restart from random initialisation while logging success.** `train_xpert.py:484-486`
  keeps only checkpoint keys already present in the model. A DataParallel checkpoint prefixes every key with
  `module.`, so **nothing matches**, `load_state_dict` succeeds on the fresh weights, and `:489` logs *"Load
  previous-trained parameters sucessfully!"*. Verified in the code. ⇒ **No session boundary goes through
  `--resume_from`.** Our full-state checkpoint loads **strictly** and asserts the loaded key set equals the model's;
  tested before launch by a save-and-reload that must give bit-equal parameters.
- **C2 (MAJOR) — §71.7 was written for one session.** §78.5 below, committed now, before any multi-session run.
- **C3 — a capped run supports no claim in either direction.** §78.1 amended in place.
- **C4 — the DataParallel proof zeroes all four dropout rates in `.train()`, not via `.eval()`**, because `eval()`
  also switches off the checkpoint wrapper [§77.3] and any other train-only branch: the proof must exercise the
  path training runs.
- **C5 — the full-state checkpoint must carry the on-disk best checkpoint** (`/kaggle/working` does not persist; if
  the best epoch fell in session 1, session 2's stopper would point at a file that no longer exists) **and resume
  only at epoch boundaries**, because the shuffle order is drawn at the start of each epoch. Resume equivalence is
  proved before launch: N epochs straight through against N/2, save, restore, N/2.

**Answers that change the record.** Ask 2: a full-state checkpoint is *more* faithful than their resume, so it does
not bear on "as published"; it is declared. Ask 3: equal one-step gradients with dropout off is the right proof;
per-replica dropout masks are i.i.d. Bernoulli(0.9) draws, a different realisation of the same distribution, as a
different seed is. Ask 4: **O4 is honest only if labelled as not a head-to-head** — unpaired, different rows, one
fold against a five-fold mean with their checkpoint selection — and an admissible run is *probably informative*: a
gap of ~0.09 against a cold-cell seed sd of 0.0052. Ask 5: **TranSiGen cannot substitute** — weaker (0.293 ± 0.017),
and it trains through the same `train_xpert.py`, so it is test-selected too. A secondary at most, not first.

### 78.5 §71.7 in a multi-session run — a scoping clarification committed BEFORE launch, threshold unchanged
Drafted by the reviewer (012 C2) and adopted in substance:

> **In a multi-session run, §71.7's `stopped_by` and `counter_at_end` refer to the FINAL termination of training.
> Per-session wall-clock guard fires are checkpoint-and-resume events, logged separately in `run_record.json`
> (session index, epoch reached, wall time) so the distinction is auditable. The ≥ 45 threshold is unchanged.**

What this is not: lowering 45 to admit a capped run would be post-hoc and is refused. What it is: §71.7 was written
when the guard *was* the end of training; in a chained design the guard is a resume point, and saying so before any
data exist is a design statement. A **final** stop by the guard is still read by §71.7's table exactly as written.
*Amended by review 015 C1 before session 1 [§84]: "e.g. the quota runs out" is withdrawn — a quota kill is
**incomplete**, not final. The only final guard stop is reaching the committed epoch horizon.*

## 79. PRE-COMMITTED: where the α = 0 residual lives — two inference-only cuts in a 2×2, written before the code exists (IDEAS A9)

§74 left a residual: with direct atom-to-atom attention cut (`atom_alpha = 0`), the atoms still cost
**−0.00322** per row on `unseen_compound` [CI −0.00433, −0.00226], sign p 1.6e−10. Review 010 ask 3 named the
carriers. This section fixes how they will be separated, before the operator is written or any number exists.
Checkpoint `sa0_ckpt_v9_fold0_seed0.pt`, batch 48, n_eval 1500, seed 0 — every knob as in §74 [rule 14].

### 79.1 The route inventory
At `atom_alpha = 0`, atom content reaches the output **only** through the gene → drug cross-attention of each
perturb block (`model_v9.py` `PerturbBlock`), by one of two kinds of key: the **atom keys** themselves, or the
**global key** (position 0), which reads the atoms inside drug self-attention. Two cuts, each leaving every
parameter, norm, SwiGLU branch and residual live:
- **X** — in every perturb block's cross-attention, the genes attend to the global key only.
- **G** — in every drug self-attention block, the global row attends to itself only: the softmax restricted to
  {0, i} that §67.6 already applies to atom rows, extended to row 0, where it is {0}.

### 79.2 The four cells, all at `atom_alpha = 0`
Atom effect `E` = median over rows of `r(atoms) − r(atoms → chunk mean)`, the estimand of record [§61.2].

| X | G | the atoms reach the genes through | effect |
|---|---|---|---|
| on | on | everything | `E0` (= §74 at α = 0) |
| on | **cut** | the atom keys only — atoms contextualised by nothing but the global token's `u` features | `Ea` |
| **cut** | on | the global token only | `Eb` |
| **cut** | **cut** | **nothing** | **must be exactly 0** |

The cross-block chain of §75.4 (atom → global in block *k*, global → atom in *k*+1, atom → genes) needs both X
and G on, so it lives only in `E0`. Per row, `e0 = ea + eb + i`, with `i` defined as the remainder: the chain plus
any non-additivity. **Paired means are additive, medians are not**, so the shares are reported as paired means
with bootstrap CIs, and each cell's own effect in the estimand of record.

### 79.3 Structural checks — stop on failure, nothing is read
1. **(on, on) reproduces §74's α = 0 per-row arrays byte for byte**, both keys, all three splits, `rows_sha`
   `434418d7677d3f9c` / `160865d7b95cbc06` / `02fb5b09d78a8d7e`. Otherwise the harness has drifted.
2. **(cut, cut) gives `dY_max == 0.0` exactly** for key `atoms` on every split. Anything else means a route is
   missing from 79.1 or the implementation is wrong; either way the decomposition is void.

### 79.4 Kill switch, per single cut, on `unseen_compound`
`cost` = median over rows of `r_full(no cut) − r_full(cut)`, both at α = 0, identical rows [paired, per §75.3].
§67.2's principle: *a masked arm that costs more than 3× the measured effect is not a usable counterfactual.*
The number it produced (+0.03) was 3× the effect then under test; the effect under test now is |E0| = 0.00322.

> **Pre-committed: a cell is read only if |cost| < 0.00966** (3 × |E0|). **Two-sided**, unlike §67.2: a cut that
> moves the model's score by three times the effect in either direction is equally far from the trained model.

Stricter than inheriting +0.03, deliberately. Review 010 expected X to be a large perturbation; if it fails here,
that is the expected outcome, not a defect.

### 79.5 Null-key gate, per single cut, `x_cell` run first
> **To attribute `E(cut) − E0` to the cut route: |N(cut) − N0| < 0.25 × |E(cut) − E0|**, in the estimand of
> record, on `unseen_compound`, where `N` is the same effect with key `x_cell`.

The §67.3 bar, applied to a two-point change instead of a five-point span. If the median of the paired per-row
difference `e(cut) − e0` has a CI95 including 0, the reading is *"this cut removes no detectable atom effect"*;
the gate is reported, and there is no attribution for it to protect.

### 79.6 Readings — primary split, cells that pass 79.4 and 79.5; "significant" = CI95 of the median per row excludes 0
| `Ea` | `Eb` | reading |
|---|---|---|
| significant < 0 | not significant | the residual harm is carried by the genes reading **atom tokens directly** |
| not significant | significant < 0 | it is carried **through the global token** |
| significant < 0 | significant < 0 | both routes carry harm; shares from the paired means |
| not significant | not significant | neither route alone carries it: the remainder `i` (chain + interaction) |
| significant > 0 in either | — | the atoms **help** through that route; stated with its sign |

- X fails 79.4 ⇒ `Eb` is not identifiable by this cut; `Ea` is read alone if G passes, and `E0 − Ea` is only
  "everything that does not go through the atom keys".
- G fails ⇒ symmetric: `Eb` alone, if X passes.
- Both fail ⇒ **A9 is not answerable by inference-time cuts.** Recorded as such; the question passes to a trained
  arm, which is not priced here.

### 79.7 Reported beside it, always
`unseen_cell` and `unseen_both`, every table, every cell [§75.2]. On `unseen_cell` the atoms *help* at α = 0
(+0.00100): a route reading there is reported and **not claimed** — `unseen_cell` stopped being primary in §58.3.

### 79.8 What no outcome licenses
One seed, one checkpoint. Inference-time cuts of a model trained with every route open: it says where the harm
*flows in this trained model*, not what a model trained without a route would do [§74.3]. "Direct" is scoped as in
§75.4. And §61.7 — that the atom tokens do not earn their place — is not under test.

**Cost: 0 GPU-hours.** Seven local inference passes at α = 0 only (four cells with key `atoms`, three with
`x_cell`). Operator and harness flag delegated as W9, verified by me before any run.

### 79.9 RESULT: A9 is NOT answerable by inference-time cuts — both kill switches fail, as pre-committed (2026-09-24)
Seven local passes, `alpha_sweep.py --operator atom_only --alphas 0 --cut {none,xattn,global,both}`, analysed by
`model/v9/a9_decompose.py` **as committed at `e20147d`, before it was run** (verified unchanged against HEAD). Output
`model/results/v9_a9_residual_routes_sa0.json`. **0 GPU-hours.**

**79.3 structural checks — both pass.**
- The uncut cell reproduces §74's α = 0 per-row arrays **byte for byte**, both keys, all three splits; `rows_sha`
  as required.
- With both cuts, **`dY_max = 0.0` exactly on every split.** The route inventory of 79.1 is complete: at
  `atom_alpha = 0` there is no path from the atoms to the output other than the two cut.

**79.4 kill switch, `unseen_compound` — both single cuts FAIL.** Bar `|cost| < 0.00966`:

| cut | cost (median per row, paired) | CI95 | verdict |
|---|---|---|---|
| X — genes read the global key only | **+0.04209** | [+0.03696, +0.04682] | **FAIL**, 4.4× the bar |
| G — the global token reads only itself | **+0.04609** | [+0.03997, +0.05184] | **FAIL**, 4.8× the bar |

Both also exceed the +0.03 that §67.2 used, so the stricter choice made in 79.4 did not decide the outcome.

⇒ **Pre-committed reading, 79.6 last row: A9 is not answerable by inference-time cuts.** *Wording corrected by review
013 C4:* A9 as posed — where the harm flows **in this trained model** — is **closed** at inference. A trained arm would
answer a **different** question (what a model trained without a route does, §79.8), and is priced as that question
under A8, not as A9 deferred.

**Reported, NOT read** (the kill switch forbids attribution; the numbers are recorded so nobody has to rerun them):

| `unseen_compound` | effect, median per row [CI95] | paired mean |
|---|---|---|
| E0, nothing cut | −0.00322 [−0.00433, −0.00226] | −0.00682 |
| G cut — atoms reach genes by their own keys only | −0.01201 [−0.01329, −0.01046] | −0.01780 |
| X cut — atoms reach genes through the global token only | +0.00023 [+0.00007, +0.00038] | +0.00085 |
| remainder `i` = e0 − ea − eb | — | **+0.01013** [+0.00779, +0.01253] |

The remainder is reported and **not read**. *(Review 013 C3: the first version of this paragraph, and commit
`9a667c6`'s message, interpreted it as "what a counterfactual 4–5 points from the trained model looks like" — an
interpretation of a quantity 79.6 says is not read, with no reference distribution behind it. Withdrawn. The kill
switch decides the outcome alone.)* The route signs above are **not** evidence about the trained model's residual.
`unseen_both` and `unseen_cell` are in the artefact [§75.2]; on `unseen_both` cut X costs +0.00930, just under the bar,
but `unseen_both` is not primary and its gate fails (0.34), so nothing is read there either.

**What IS established, at the strength it supports.** (1) A **structural** fact, proved by `dY_max = 0.0`: at
`atom_alpha = 0`, atom information reaches the output only through the gene → drug cross-attention, by the atom keys
or by the global key. (2) The kill-switch quantity itself, as §72.2 treated its own, *worded per review 013 C2 and verified by me from
the per-row arrays:* **cutting either route costs 0.042–0.046 whether the atom tokens carry their own molecule or
the batch mean (0.041–0.046: X +0.0455, G +0.0409, both +0.0602 with batch-mean atoms): the trained model depends on
the attention pathways, not on molecule-specific atom content through them.** At α = 0 that content is net
harmful (E0 < 0). On `unseen_both` the one near-pass (X, +0.0093, CI [0.0064, 0.0123] straddling the bar) costs
+0.0187 in the ablated arm — even that depends on which arm is measured.

## 80. PRE-COMMITTED: what the DataParallel measurement (v6) must show before O2 can be priced (2026-09-23)

Review 012 approved the ~0.15 GPU-h measurement. This fixes, before launch, what counts as DataParallel being the
same computation, and what the timing is then used for. No training run is bought by this section.

### 80.1 The patch, and why it is not `nn.DataParallel(model)`
`model/v9/xpert_dp_patch.py` replaces `XPertNet.forward` at runtime: at the top level, in training with grad enabled
and at least one row per GPU, it calls `torch.nn.parallel.data_parallel` on itself; on each replica it localises
`self.device` and every plain tensor attribute, then runs their forward unchanged. **The object their code holds stays
an `XPertNet`**, so `state_dict()` keys carry no `module.` prefix and review 012 C1's silent-restart trap cannot be
reached through the model itself. Validation and prediction are untouched (single GPU).

### 80.2 The proof, through THEIR `train()`
`model/v9/xpert_dp_probe.py`, one process per configuration: `single_ckpt` (the v5 recipe), `single_ckpt_repeat`,
`dp`, `dp_ckpt`. Identical initial weights and one identical recipe batch, loaded strictly from disk. Every gradient
is taken by calling `train_xpert.train()` itself with an lr-0 SGD step, so it is their `batch_weighted_loss` (epoch 0)
and their `weighted_loss` (epoch 70), under their GradScaler, in `.train()`.

**Dropout is zeroed on every module, not through the config** (review 012 C4 named the four config rates;
`model_XPert.py:144, 158, 164, 170` hard-code four more `nn.Dropout(p=0.1)` in the heads, and the attention layers
carry their own `dropout_p`). The probe asserts none is left.

### 80.3 Pass criteria, as inequalities
`rel_L2(a, b) = ||g_a − g_b|| / ||g_b||` over all parameter gradients.

| check | criterion |
|---|---|
| **same function** — fp32 (their autocast replaced by a null context), first 32 rows, epochs 0 and 70 | `rel_L2(dp, single) < 1e-5` **and** `rel_L2(dp_ckpt, single) < 1e-5`; loss relative difference `< 1e-6` |
| **within the recipe's own precision noise** — fp16 autocast, same 32 rows, epochs 0 and 70 | `rel_L2(dp, single)` and `rel_L2(dp_ckpt, single)` each **≤** `rel_L2(single fp16, single fp32)` |
| structure | state-dict keys equal to the unpatched model's, no `module.` prefix; plain tensor attributes exactly `['drug_HG_embed']`; every gradient finite |
| memory | the variant chosen peaks below **13.0 GiB on each GPU** (the GUARD F ceiling) |

Why these: fp32 isolates the mathematics — the only thing DataParallel changes is reduction order, and a real defect
(a loss computed per replica, a mis-scattered input) would show at 1e-2 or worse, three orders above the bar. fp16
is the recipe's precision, where rounding alone moves gradients; the fair scale for "no worse than rounding" is the
recipe's own fp16-versus-fp32 difference on the same rows and weights. At 128 rows the fp16 comparisons are
**reported, not gated**, against the `single_ckpt` / `single_ckpt_repeat` floor.

**If any check fails, O2 is not priced** and the choice is O1 or O4 [§78.2].

### 80.4 What the timing is used for
Each passing variant's steady-state step (Adam, their GradScaler, loader included, five batches after one warm-up)
gives `epoch_s = 372 × s_train + 167 × s_val`, with `s_val` from GUARD F. The faster variant that passes 80.3 is
the O2 price: `~210 × epoch_s` GPU-seconds for an admissible run at the §78.2 anchor, and the session count at
7.95 h of training per session. That number, and nothing else from v6, goes to the O2-versus-O4 decision.

### 80.5 v6 crashed in its own comparison; review 013; amendments committed BEFORE any v6 gradient is compared
**v6** (`external/kaggle_out/cc1_v6/`, ~0.15 GPU-h, session total **~6.95 GPU-h**): every guard passed, all four GUARD G
probe processes ran, and the kernel's comparison then stopped on its own assertion — `different parameters received
gradients`. Timing and memory were logged before it (reported in §80.6). The per-probe JSON (losses, scale) was lost
with the crash; the four gradient dumps survived as output and were downloaded.

**What the key sets show — structural, inspected before any value comparison.** On every tag, the single-GPU
reference has gradients for **174** parameters and DP for **184**. The ten extra are exactly the parameters their
forward and loss **never use**: `cell_emb.linear` (weight, bias), `ctl_fc.0` and `ctl_fc.3` (weight, bias), and the
`LayerNorm` gamma/beta of `attnEncoder_trt.crossEncoders.0` and `.1`. On one GPU they get `grad = None`; under DP they
get **exactly zero** — `Broadcast.backward` materialises zeros for replica copies that received no gradient.

**Why that matters beyond the assertion — found by me, missed by both of us before launch.** `torch.optim.Adam` applies
`weight_decay` to every parameter **whose grad is not None**. On one GPU these ten are skipped for the whole run and
stay at their initial values. Under DP each would receive `0 + 1e-5 · p`, which Adam's normalisation turns into a
step of order `lr` = 0.004 **per step**: they would drift, in DP only. They do not affect training outputs (they are
unused), but it is not the recipe, and `ctl_fc` produces `ctl_output`, which their code returns.

> **Amendment A (exact).** The parameters with `grad = None` in the single-GPU reference, on **both** loss branches, are
> set `requires_grad = False` in **every** variant before the optimizer is built. On one GPU this changes nothing
> (Adam already skips them). Under DP, `Broadcast` marks outputs of inputs that need no grad as non-differentiable,
> so they get no gradient at all. The set is asserted at runtime to equal the ten names above; if it ever differs,
> the kernel refuses. Declared as a deviation. For the v6 comparison, gradients are compared on the reference key
> set, and the extras are required to be exactly zero (they are).

**Review 013** (`orchestration/bus/adjudicated/013_review.md`, at `9a667c6`), **SOUND-WITH-CAVEATS**, five challenges,
all upheld. **Tally 85 of 88.** Part A's reading is confirmed and every number reproduced; C2–C4 amended 79.9 in place
above. Part B: the patch is correct and the criteria test the right thing, with these amendments, adopted **before
reading**:

> **Amendment B (C1).** A tag on which the **single-GPU reference's own** gradients are non-finite is **void** for
> criteria 2 and 3 in every variant, not failed — at the default scale of 65,536 the reviewer estimates the epoch-70
> `trt_fc` bias gradient at ~7×10⁶ against fp16's 65,504, because `loss1` is a **sum** over rows of un-centred
> targets (mean 8.3). The finiteness gate becomes: **each variant's finiteness equals the reference's, per tag**; DP
> non-finite where the reference is finite stays a failure. (`scale` was lost with the crash; a scaler backs off
> exactly when gradients are non-finite, so the reference's finiteness is the same test.) fp32 at epochs 0 and 70
> still proves the semantics in both loss regimes.
>
> **Amendment C (ask 3, the floor).** If fp32 `rel_L2(single_repeat, single)` ≥ **3e−6**, a third of the bar,
> criterion 1 does not discriminate as written, and the fp32 tags are re-run for all four variants with the
> deterministic math SDPA backend before criterion 1 is read.
>
> **Amendment D (C5).** Criterion 2 cannot detect autocast failing to reach the replica threads (DP would then compute
> in fp32 and land near the bar itself). The next run asserts `torch.is_autocast_enabled()` inside the replica
> forward, records `torch.__version__`, and keeps the gradient dumps as output.
>
> **Amendment E (ask 5).** The projection is from five batches inside the loader's prefetch buffer. **If the first real
> epoch's wall time exceeds the projection by more than 25 %, O2 is re-priced before session 2.**
>
> **Amendment F (ask 5).** `_localise` moves only `drug_HG_embed`, asserts every other tensor attribute is already on
> the replica's device, and checks its cache entry's source by identity (`is`), so a reused `id()` can never serve a
> stale copy.

**Criterion 1's loss clause cannot be evaluated on v6** — the losses were in the lost JSON. It is evaluated on the
next run, which is needed anyway for the full-state checkpoint (review 012 C1, C5) and Amendments A and D.

### 80.6 🔴 v6 VERDICT: criterion 1 FAILS — DataParallel is not shown to be the same computation; O2 is not priced
`model/v9/dp_v6_offline.py`, committed at `0425dd8` before it ran; output `model/results/xpert_dp_v6_offline.json`.

| check (80.3 as amended by 80.5) | result |
|---|---|
| structure: variant keys ⊇ reference keys; the extras are exactly the ten unused parameters, all exactly zero | **PASS** |
| Amendment C floor: fp32 `rel_L2(single_repeat, single)` | **2.8e−7** (< 3e−6: criterion 1 discriminates) |
| **criterion 1, fp32 gradients, `dp` vs single** | epoch 0 **3.53e−4**, epoch 70 **3.77e−4** against **< 1e−5** → **FAIL** |
| criterion 1, `dp_ckpt` vs single | 3.53e−4, 3.77e−4 → **FAIL** |
| criterion 1, loss clause | not evaluable (losses lost with the crash) |
| criterion 2, fp16 | **all four fp16 tags VOID**: the single-GPU reference's own gradients are non-finite on every one, **including epoch 0** (review 013 estimated epoch 0 at 5× under the fp16 limit; it overflowed) |
| memory | `dp` 7.39 / 7.36 GiB, `dp_ckpt` 2.00 / 1.96 GiB per GPU: PASS |

⇒ **By the rule committed in 80.3: any failure and O2 is not priced.** It is not priced on this measurement.

*A defect in my offline script, not affecting the verdict:* it applied Amendment B's finiteness-equality check to the
void fp16 tags too, which Amendment B excludes, and so also printed `finiteness_equals_reference: false`. On the
non-void (fp32) tags every variant is finite, as the reference is.

**Timing — recorded, and explicitly NOT a price** (the proof it was conditional on failed):

| variant | s / train step | epoch (372 × train + 167 × 0.400 s val) | ~210 epochs |
|---|---|---|---|
| single_ckpt (the v5 recipe) | 2.229 | 896 s | 52.3 GPU-h |
| `dp` | **0.925** | **411 s** | **24.0 GPU-h, 3.0 sessions** |
| `dp_ckpt` | 1.169 | 502 s | 29.3 GPU-h |

**Where the difference sits — diagnosis after the verdict, which it does not change.** Per parameter, fp32 epoch 0:
**84 of 174** parameters differ by more than 1e−4 relative (range ~2.7e−4 to 7.7e−4), against a repeat floor of
~2–6e−7 on the same parameters. The largest shares are in the first cross block and the drug path
(`crossEncoders.0` query/key of the gene self-attention, `drug_SA`, `attention_CA`; `drug_emb.linear`), but
feed-forward weights differ by ~3e−4 too. Nothing in their forward couples samples: no reduction over the batch
dimension, no per-batch trimming of the fixed-length drug sequence, and the attention shim is per-sample (checked
against `model_utils.py:232-300, 351-420`, `model_XPert.py:10-19, 186-260`).

**Hypothesis, not a finding:** a changed per-GPU batch shape changes GEMM tiling and so fp32 rounding by ~1e−7, and
at initialisation the loss is ill-conditioned — predictions are nearly constant across genes, and `pcc_loss_sum`'s and
the `sqrt(loss/N)` terms' gradients scale with the inverse of small spreads — amplifying that rounding ~10³×. A
repeat uses identical shapes and cannot see it, which is why the 80.3 floor did not. **If true, the fp32 test as
designed cannot separate semantics from numerics at initial weights,** and the design assumption both the reviewer
and I made — reduction-order differences of 1e−7 to 1e−6 — was wrong for this loss at this point in weight space.
It is tested next, locally and at zero GPU cost, as a post-hoc diagnostic; any redesigned proof goes to review as a
**new** pre-registration, labelled as designed after this failure.

### 80.7 Post-hoc diagnostic (local, 0 GPU-h): the DP computation is the recipe's in exact arithmetic; the fp32 bar was set on a false premise
`model/v9/diag_split_local.py`, one laptop GPU. **Not pre-registered; it decides only what the redesigned proof is.**
No DataParallel involved: on ONE GPU, their `train()` gradient on a batch is compared with the same rows run as two
half-batch forwards concatenated before their loss — the computation DataParallel performs, minus the second device.
Their `scanpy` / `unimol_tools` imports are stubbed as empty modules (unused by `train()`; installing them would
downgrade numpy and pandas in the CUDA venv).

| precision | rows | SDPA backend | weights | repeat vs full | **split vs full** |
|---|---|---|---|---|---|
| fp32 | 32 | mem-efficient | init / released | 1.3e−7 / 1.3e−8 | **2.4e−4 / 2.2e−4** (epoch 0) |
| fp32 | 16 | mem-efficient | init / released | 7.6e−8 / 6.5e−9 | 3.5e−4 / 9.6e−7 |
| fp32 | 16 | math | init / released | 0 / 0 | 5.8e−4 / 1.3e−4 |
| **float64** | 16 | math (only) | released | 0 | **1.1e−16 (epoch 0), 1.1e−17 (epoch 70)** |

Readings, at the strength they support:
1. **In float64 a split batch reproduces the full batch to machine precision.** Nothing in the forward or the loss
   couples samples; the computation DataParallel performs is the recipe's function. (For float64 only, their
   `get_unimol_drug_feat`'s hard `.float()` cast on atom features, `model_XPert.py:14`, is bypassed.)
2. **In fp32, an exactly equivalent reorganisation on one GPU moves gradients by 1e−6 to 6e−4** depending on rows,
   weights and attention backend — the range DP's 3.5e−4 on the T4 falls in. It is not specific to initial weights (my
   80.6 sub-hypothesis is wrong: 2.2e−4 at the released weights, 32 rows) and not specific to the memory-efficient
   kernel (the math backend is no better). Its exact source is not identified.
3. ⇒ **80.3's fp32 bar of 1e−5 assumed batch-shape rounding of 1e−7 to 1e−6** — an assumption review 013 and I both
   made, and this model's fp32 numerics break. *Scoped by review 014 C3:* **v6's test could not discriminate — no exact
   reorganisation of this model meets a 1e−5 bar against the full batch in fp32. DP equivalence remains UNSHOWN until
   81.1 runs.** The split tests batch reshaping, not the second device, `Broadcast`, `ReduceAddCoalesced` or the
   localised `drug_HG_embed`. *And review 014 C2:* the diagnostic cast gradients with `.float()` before comparing, so
   its "1.1e−16 in float64" is **agreement to fp32 resolution of a float64 computation** — still enough to settle the
   absence of batch coupling, not a float64-precision measurement.

## 81. PRE-REGISTRATION (final form in 81.5, after review 014): the redesigned DataParallel proof, and full-state resume — designed AFTER 80.6's failure
**Labelled as designed after a failure.** 80.6 stands as a failure of the proof as committed. What follows is a new
test, and review 014 decides whether it is admissible before anything is run.

### 81.1 The proof, v7, measure-only
Four processes as before, plus `split_same_gpu` (the 80.7 construction inside their `train()`), weights both fresh
(seed 0) and the released checkpoint, with Amendment A (the ten unused parameters frozen in every variant):
- **81.1a Semantics, float64**, 32 rows (16 per GPU), epochs 0 and 70: `rel_L2(dp, single) < 1e−10` and
  `rel_L2(dp_ckpt, single) < 1e−10`; loss relative difference `< 1e−12`. float64 epsilon is 1.1e−16 and the
  same-GPU split measured 1e−16; a semantic defect (a mis-scattered or stale tensor, a loss on a shard) would show at
  1e−6 or worse.
- **81.1b fp32, DP against the RIGHT reference**, 32 rows: `rel_L2(dp, split_same_gpu) < 1e−5`. A replica runs the same
  kernels on the same shapes as a same-GPU half, so DP should add only the cross-device gradient sum;
  `rel_L2(split_same_gpu, single)` is reported as the recipe's own batch-shape sensitivity.
- **81.1c fp16**, reported only: one fixed `GradScaler(init_scale=2**6)` shared by every variant (review 013 C1).
- Structure: equal key sets once Amendment A freezes the ten; in-replica `torch.is_autocast_enabled()` asserted in
  fp16 (Amendment D); `torch.__version__` recorded; gradient dumps kept as output.

### 81.2 Full-state checkpoint and resume (review 012 C1, C5; 78.5)
Captured by runtime hooks, with their files verbatim: `XPertNet`, `Adam`, `LambdaLR`, `GradScaler`, `EarlyStopping`
instances are registered at construction. **Save** happens after `lr_scheduler.step()` at the end of each epoch
(`train_xpert.py:545`, after `stopper.step` at `:538`), atomically: model, Adam, GradScaler and LambdaLR state dicts;
the stopper's `best_score`, `counter`, `early_stop`; the bytes of the on-disk best checkpoint (`folder` is time-stamped
per session, so it is restored under the new session's path); torch CPU, CUDA (both devices), numpy and python RNG
states; the finished epoch index. Resume only at epoch boundaries. Carried between sessions as a Kaggle dataset
version (~150 MB), attached to the next session.

**Restore**, the one design choice put to review (ask 3): their `--resume_from` is used **only** to set `start_epoch`
(`:489`). The full restore happens in the `EarlyStopping.__init__` hook (`:524`, the last construction before the
loop, when every object exists): a **strict** model load with the loaded key set asserted equal to the model's
(review 012 C1), then optimizer, scaler, scheduler, stopper and RNG states. Their filtered load at `:484-486` runs
first and is overwritten.

### 81.3 Resume equivalence, before any multi-session run (review 012 C5)
`dp` recipe, published dropout: 2 epochs straight, twice (the floor), against 1 epoch + save + fresh process + restore
+ 1 epoch. Pass: every per-epoch train and validation loss of the resumed run within the straight-vs-straight spread,
and final parameters no further from a straight run than the two straight runs are from each other. ~6 epochs at
~7 min, ~0.7 GPU-h.

### 81.4 What it buys
If 81.1a–b and 81.3 pass: O2 is priced from the `dp` timing (0.925 s/step in v6: ~24 GPU-h, ~3 sessions at the
anchor), subject to Amendment E's 25 % epoch-1 re-price. If 81.1a fails, DataParallel is not the recipe's computation
and the choice is O1 or O4.

### 81.5 Review 014, and the FINAL form of this pre-registration — committed before any v7 code enters the kernel
`orchestration/bus/adjudicated/014_review.md`, at `5b45f9e`. **SOUND-WITH-CAVEATS**, four challenges, all upheld.
**Tally 89 of 92.** The redesign is **admissible as a new measurement for a new run**, on the reviewer's conditions,
adopted verbatim: v6's dumps are never re-read under 81 and called a pass; 81.1a is not dropped or loosened after
v7; no bar is set after v7; and **`rel_L2(split_same_gpu, single)` and fp32 `rel_L2(dp, single)` stay REPORTED ONLY**
— never promoted to "within the recipe's batch-shape sensitivity, therefore a pass". The reviewer records its own
fifth error on the bus: it endorsed 013's 1e−5 bar by asserting a numerical property of the network without
measuring it; and its epoch-0 fp16 estimate was wrong.

**81.1, as run:**
- **81.1a float64 semantics:** 32 rows, epochs 0 and 70, `dp` against `single_ckpt`, the ten frozen. **Gradients kept
  in float64 for these tags** (C2: casting to fp32 would put the floor at ~√(6e−8·δ) and could fail 1e−10 on
  quantisation alone). Bars unchanged: `rel_L2 < 1e−10`, loss relative difference `< 1e−12`.
- **81.1b fp32 against a same-GPU split, on the MATH SDPA backend** (the reviewer's strengthening, free): `dp`
  paired with `split_same_gpu` at the same checkpointing setting (none), 32 rows, epochs 0 and 70. Bar unchanged at
  `< 1e−5`; the expected value is ~0 (the diagnostic's math-backend repeat was exactly 0), so anything above 1e−9
  is recorded as a finding, though only 1e−5 decides.
- **81.1c fp16, reported only:** the single-GPU reference halves the GradScaler scale from 2⁶ until its gradients are
  finite; that scale is then used for every variant. In-replica `torch.is_autocast_enabled()` is asserted.
- **Structure:** with the ten frozen, the set of parameters with `grad is None` must equal exactly the frozen set in
  every variant — no other unused parameter exists.

**81.3, replaced (C1):** the straight-vs-straight floor would fail a perfect resume about half the time, because a
correct resume and the straight runs are exchangeable draws. Two tests, neither needing a floor, on their real
`main()` in the `dp` configuration, with each epoch truncated to 10 training and 10 validation batches (test mode
only: state completeness does not depend on epoch length):
- **81.3a state round-trip, exact:** the process that stops after epoch 0 saves the full state; the resuming process
  dumps its live state immediately after restore. **Required bitwise equal:** every model tensor; Adam `m`, `v`,
  `step`; GradScaler scale and growth tracker; LambdaLR `last_epoch` and `_last_lr`; stopper `best_score`, `counter`,
  `early_stop`; torch CPU and both CUDA RNG states, numpy and python RNG; the best file's sha1; and `start_epoch`.
- **81.3b continuation:** three straight runs of 2 truncated epochs and one resumed run (1, stop, fresh process,
  restore, 1), all with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. **If the
  three straight runs are bitwise identical, the resumed run must be bitwise identical to them.** If they are not —
  determinism was not achieved — the resumed run's parameter distance to each straight run must not exceed the
  largest of the three pairwise straight distances (the reviewer's fallback). If deterministic mode raises for an op,
  that op is recorded and the fallback applies.

**C4, in the code:** if `--resume_from` is set and the strict restore has not run by the first `train()` call, the
trainer **fails hard**. The full-state file is loaded with `map_location='cpu'`, independently of their load at
`:483`. `--resume_from` sets `start_epoch` only (ask 3: acceptable with this assert).

**Carried forward:** the `dp` variant (7.39 GiB, 0.925 s/step) is the production choice, which keeps `dp_ckpt`'s
dropout replay across replica threads (013) off the critical path. State is written atomically at every boundary,
and each session leaves time after its guard to write output before Kaggle's hard limit.

### 81.6 Pre-launch amendment: 16 rows, not 32, for 81.1a and 81.1b — memory, fixed before any v7 data
On the math SDPA backend each of the 8 gene self-attention layers (trt 2 cross + 2 self, ctl 4) stores its
979×979×8-head probability tensor for backward: ~30.7 MB per sample per layer in fp32, ~245 MB per sample in all,
on top of ~0.23 GB of other activations — ~0.48 GB per sample, twice that in float64. So the uncheckpointed 32-row
same-GPU split of 81.1b (~15.4 GB) and the float64 `dp` at 16 per GPU of 81.1a (~15 GB per GPU) would not fit a
14.56 GiB T4. **Both run at 16 rows (8 per GPU).** Review 014 ask 3 already established that every way DP could change
the function shows at any per-GPU batch of 2 or more, so the semantic coverage is unchanged. **No bar changes.**
The fp16 tags (reported only) stay at 128. The same-GPU split runs without checkpointing, paired with `dp`.

### 81.7 ✅ v7 RESULT: every pre-registered check passes — DataParallel is the recipe's computation, and full-state resume is exact (2026-09-24)
`external/kaggle_out/cc1_v7/run_record.json`, kernel v7 at `da3c5ac`, `stopped_by: measure_only`, 15.3 min wall,
**~0.27 GPU-h; session total ~7.2 GPU-h.** Bars as fixed in 81.5 and 81.6, before launch.

**81.1 — the proof.**

| tag | repeat vs single | **dp vs single** | dp vs split | split vs single | dp loss rel. diff. |
|---|---|---|---|---|---|
| float64, epoch 0 | 0 | **5.4e−16** (bar 1e−10) | — | — | 0 (bar 1e−12) |
| float64, epoch 70 | 0 | **4.8e−16** | — | — | 0 |
| fp32 math, epoch 0 | 0 | 5.4e−4 *(reported only)* | **2.4e−9** (bar 1e−5) | 5.4e−4 *(reported only)* | 2.8e−7 |
| fp32 math, epoch 70 | 0 | 4.9e−4 *(reported only)* | **2.7e−9** | 4.9e−4 *(reported only)* | 2.8e−7 |
| fp16, epoch 0 (scale 64) | 1.0e−5 | 3.0e−4 *(reported)* | — | — | 1.8e−7 |
| fp16, epoch 70 (scale 16) | 1.0e−4 | 2.9e−4 *(reported)* | — | — | 1.8e−7 |

Structure: in every mode and tag the set of parameters with `grad is None` is **exactly the ten frozen**; the only plain
tensor attribute is `drug_HG_embed`; every fp32/float64 gradient is finite; **autocast was on in every DP replica call**
on both fp16 tags; `dp` peaks at **7.37 / 7.33 GiB** per GPU.

⇒ **81.1 PASSES.** In float64, DataParallel reproduces the single-GPU gradient to **machine precision** (5e−16) and
the loss exactly: it computes the recipe's function. In fp32 it matches a same-GPU split to 2.4e−9; the ~5e−4
against the unsplit batch is the recipe's own batch-shape sensitivity [80.7], and stays **reported only** [81.5].

**Finding, as 81.5 required for anything above 1e−9:** fp32 `dp` vs `split` is 2.4e−9 and 2.7e−9, not the 0 of a
same-GPU repeat. *Hypothesis, not tested:* the two replica gradients are summed by `ReduceAddCoalesced` after the
backward, while the same-GPU split accumulates the two halves' contributions inside one autograd pass — a different
order of the same two-term sums, i.e. a few fp32 ulps. It is five orders under the bar and the float64 test excludes a
semantic cause.

**81.3 — full-state resume, on their real `main()`, DataParallel, the ten frozen.**
- **81.3a exact round-trip: PASS.** The live state right after restore equals the saved state in **every** field
  (model, Adam `m`/`v`/`step`, GradScaler, LambdaLR, stopper, both CUDA RNGs and the CPU/numpy/python RNGs, the best
  file's sha1); `start_epoch` is the saved epoch + 1.
- **81.3b continuation: PASS, under the stricter branch.** Deterministic mode ran without raising; the three straight
  runs are **bitwise identical**, and the resumed run is **bitwise identical** to them.

**Timing (O2's price, 81.4):** `dp` 1.113 s / step in v7 (0.925 in v6 — 20 % slower, unexplained; Amendment E's
first-epoch check will show which holds) → epoch 482 s → **~28.1 GPU-h for ~210 epochs, ~3.5 sessions** at the
anchor [78.2]. Against a 30 h/week quota with ~7.2 used, an admissible run spans two quota weeks.

⇒ **O2 is priced and every precondition in 78.4, 80.5 and 81 is met.** The launch design goes to review as packet
015 before any training GPU-hour.

## 82. PRE-REGISTERED: a zero-parameter pre-test of A1 — does the cell's OWN chromatin make the union graph conduct a drug's effect better? (2026-09-24)

IDEAS A1 is the principal's idea: chromatin decides **which edges conduct in this cell**, a different job from the
per-gene gating that measured null [§55]. Its trained arm costs a v9-shaped run (~5.8 GPU-h) and competes with O2 for
the week's quota. The registry rule is that GPU spend is gated on a free CPU result first. This is that result,
fixed before any code exists. **0 GPU-hours; local CPU, minutes.**

### 82.1 The predictor — no fitted parameters at all
For signature (compound *d*, cell *c*): `t_d` = indicator of *d*'s DTI targets among the 978 landmarks
(`drug/outputs/dti/dti_reference.tsv`, already landmark-indexed), normalised to sum 1. Graph `A` = the **binary union**
of the three sources in `network/outputs/v9/union_graph_v9.npz` (an edge present in any), symmetric. Gated weights
`W_c[i,j] = A[i,j] · sqrt(a_ci · a_cj)`, with `a_c` = the ATAC track of `E_final` (channel 0, in [0, 1]) for cell *c*.
Propagation, random walk with restart: `s = (I − α P)^{-1} t_d`, `P = D^{-1/2} W D^{-1/2}`, **α = 0.5 fixed**. Nothing is
fitted, so no split is needed and nothing can be tuned toward the answer.

### 82.2 Rows, target, score
Rows: every usable signature (`signatures_usable.tsv`) whose cell has the ATAC track observed for ≥ 90 % of landmarks
and whose compound has ≥ 1 DTI target. Target: `|z|` from `Y_target_level5_978.npy` (Level 5). Score per signature:
**Spearman(s, |z|) over the non-target landmarks** — the drug's own targets are excluded, so only propagation can
score. Gene order is asserted identical, by symbol, across Y, the graph, `E_final` and the DTI table.

### 82.3 Conditions, all on identical rows
| condition | edge weights | what it isolates |
|---|---|---|
| **OWN** | the cell's own ATAC | the hypothesis |
| **MISMATCH** (null key) | the ATAC of another ATAC cell, 5 deterministic draws (seed 0), score averaged | chromatin, but not *this cell's* |
| **MEAN** | mean ATAC over the ATAC cells | a gene-level accessibility prior, not cell-specific |
| **NONE** | `A` ungated | the graph alone |
| DEGREE (sanity) | `s` = gene degree in `A` | hubs respond more; no drug information |

### 82.4 Estimand — the unit is the cell line [review 006 C3]
Primary: per cell *c*, `d_c` = median over *c*'s signatures of `score(OWN) − score(MISMATCH)`. Summary: the
unweighted mean of `d_c` over cells, a cluster bootstrap over cells (20,000 draws, seed 0), and the count of cells with
`d_c > 0` with a sign test. Secondary, same form: `OWN − MEAN`, `OWN − NONE`, `NONE − DEGREE`.

### 82.5 Readings, as inequalities
| result | reading |
|---|---|
| NONE's median score over rows **≤ 0.01**, or NONE − DEGREE's cluster CI includes 0 | **Uninformative.** Zero-parameter diffusion carries no drug-specific signal here, so it cannot test gating. A1 is neither supported nor refuted. |
| OWN − MISMATCH: mean > 0, cluster CI excluding 0, **and ≥ 75 % of cells** favour OWN | **Supports A1's mechanism**: this cell's chromatin makes the graph conduct the drug's effect better than another cell's. A1's trained arm is then priced and pre-registered. |
| OWN − NONE > 0 (CI excl. 0) but OWN − MEAN's CI includes 0 | A **gene-level accessibility prior**, not cell-specific conduction. Not A1's mechanism; recorded as such. |
| anything else | **No support from a zero-parameter model.** Not a refutation: a trained model can express gating that fixed diffusion cannot. A1's trained arm is not motivated by this test. |

### 82.6 What no outcome licenses
Unsigned targets and `|z|` only: nothing about the *direction* of response. One chromatin track (ATAC) as primary;
H3K27ac is reported as a secondary with the same estimand and not read unless ATAC's reading is "supports". Diffusion
is a model of propagation, not of mechanism; a positive result says the gated graph *predicts better*, not that
edges physically conduct. Delegated as W10; verified by me before any number is read.

### 82.7 RESULT: UNINFORMATIVE on both tracks — zero-parameter diffusion of landmark targets carries no drug-specific signal, so it cannot test gating (2026-09-24)
`model/v9/a1_diffusion_pretest.py` as committed at `5e509c5` (W10, verified: code read in full, 9/9 tests), run
locally on CPU. Artefacts `model/results/a1_diffusion_pretest_ch{0,1}.json` and `_rows.npz`. **0 GPU-hours.**

| | ATAC (primary) | H3K27ac (secondary) |
|---|---|---|
| eligible cells / rows | 32 / 52,496 | 35 / 60,406 |
| median Spearman(s, \|z\|): OWN, MISMATCH, MEAN, **NONE**, DEGREE | +0.0002, −0.0003, −0.0001, **−0.0006**, −0.0018 | −0.0002, −0.0005, −0.0006, **−0.0007**, −0.0021 |
| NONE − DEGREE, cell-level mean [CI95] | +0.0010 [−0.0001, +0.0023] | +0.0010 [+0.0000, +0.0021] |
| **OWN − MISMATCH** (primary) | +0.0002 [−0.0004, +0.0009], 16 / 32 cells | +0.0001 [−0.0004, +0.0006], 12 / 35 cells |
| OWN − MEAN | +0.0002 [−0.0004, +0.0010], 12 / 32 | +0.0002 [−0.0003, +0.0007], 13 / 35 |
| OWN − NONE | +0.0005 [−0.0007, +0.0018], 16 / 32 | +0.0006 [−0.0005, +0.0019], 15 / 35 |
| NaN rows | 0 | — |

⇒ **Pre-committed reading, 82.5 row 1: UNINFORMATIVE**, on both tracks — NONE's median (−0.0006) is below the 0.01
floor. A random walk from a compound's landmark targets over the union graph does not predict which landmarks
respond at all, so there is no signal for chromatin gating to improve or degrade. **A1 is neither supported nor
refuted by this test.** The contrasts are reported and not read; none is distinguishable from zero.

**What it does establish, at the strength it supports:** in this dataset, **proximity to a compound's annotated
landmark targets on the STRING + Reactome + GO union graph carries no information about response magnitude** — not
better than gene degree, whose own median is −0.002. That is a statement about a fixed propagation model on these
inputs, not about the graph's usefulness inside a trained model.

**A limitation of the design, recorded, not used to explain the result away:** `dti_reference.tsv` lists only targets
that are **themselves landmark genes** (`gene_idx` indexes the 978), so propagation starts from a small and biased
subset of each compound's real targets. A version that starts from **all** annotated targets on the full STRING graph
(`network/outputs/v9/string_graph_v9.npz`, 19,496 nodes) and reads out at the landmarks would test the same question
without that restriction. Logged in IDEAS A1. *(Corrected the same day, method rule 20: I first wrote that it
"needs a non-landmark target table first". One exists — `drug/outputs/dti/chembl_dti_edges.tsv`, 6,020
mechanism-level edges for 1,363 compounds over 546 targets, with an `is_landmark` flag, and
`stitch_dti_edges.tsv`, 28,387 edges. The real constraint is different: chromatin covers only the 978 landmarks,
so on the full graph gating could act on landmark–landmark edges only. The first question for that form is
therefore whether ungated propagation from ALL targets carries any signal — NONE against DEGREE — before any
gating is worth testing.)*

**For the GPU decision:** A1's trained arm (~5.8 GPU-h) is **not motivated** by this pre-test, and O2 keeps priority
for the quota.

## 83. PRE-REGISTERED: does propagation from ALL of a compound's mechanism targets, over the full STRING graph, carry drug-specific signal at the landmarks? (A1's prerequisite, 2026-09-24)

§82 was uninformative because ungated diffusion from **landmark** targets had no signal. Before any gating on the full
graph is worth building, the prerequisite is whether propagation **from all annotated targets** carries signal at all.
Written before any code exists. **0 GPU-hours; local CPU.**

### 83.1 Predictor
Graph: `network/outputs/v9/string_graph_v9.npz` — 19,496 nodes, `edge_index` (2, 929,472), `weight` (STRING
confidence), `landmark_idx` (978, landmark order → node). Weighted and symmetric (`W[i,j] = W[j,i] = weight`); the
worker verifies whether edges are listed once or twice and symmetrises without double counting. Targets:
`drug/outputs/dti/chembl_dti_edges.tsv` (mechanism-level, `direct_interaction = 1`), `gene_symbol` mapped to nodes;
unmapped symbols counted and dropped. `t_d` = 1/|T| on the compound's mapped targets. Random walk with restart,
**α = 0.5 fixed**, `s = (I − αP)^{-1} t`, `P = D^{-1/2} W D^{-1/2}`, computed by the sparse fixed-point iteration
`s ← α P s + t` to an L1 change below 1e−10. Nothing is fitted.

### 83.2 Rows, score, conditions
Rows: every usable signature whose compound has ≥ 1 mapped target, all cells. Score: Spearman(s at the landmarks,
|z|) over the landmarks that are **not** targets of the compound (Level 5 `Y_target_level5_978.npy`).
- **NONE** — the compound's own targets.
- **RANDOM** (primary null) — the same number of targets drawn uniformly from all graph nodes, 5 draws per compound
  (`default_rng(seed + compound_index)`), score averaged. Tests *drug specificity*: hub structure alone cannot pass it.
- **DEGREE** — `s` = the landmarks' weighted degree in the full graph.

### 83.3 Estimand — unit is the cell line
Per cell *c* with ≥ 50 rows: `d_c` = median over *c*'s rows of `score(NONE) − score(RANDOM)` (primary) and of
`score(NONE) − score(DEGREE)` (secondary). Mean of `d_c` over cells, cluster bootstrap over cells (20,000, seed 0),
count of cells with `d_c > 0`, sign test. Also the median of `score(NONE)` over rows.

### 83.4 Readings
| result | reading |
|---|---|
| median(NONE) > 0.01 **and** NONE − RANDOM: mean > 0, CI excl. 0, ≥ 75 % of cells | **Propagation from mechanism targets carries drug-specific signal.** A full-graph gating test for A1 (landmark–landmark edges gated by chromatin) is then worth pre-registering. |
| NONE − RANDOM passes but median(NONE) ≤ 0.01 | Specific but negligible; recorded; the gating test is not built. |
| anything else | **No drug-specific signal from fixed propagation over the STRING graph.** The diffusion route to A1 is closed; A1 remains a question only a trained model can put. |

Delegated as W11; verified by me before any number is read.

## 84. 🔒 COMMITTED BEFORE SESSION 1: how the O2 run can end, and what may inform decisions between sessions (2026-09-24)

Review 015 (`orchestration/bus/adjudicated/015_review.md`, at `4f6ccfc`): **SOUND-WITH-CAVEATS, GO on O2 once C1 is
committed**; four challenges, all upheld. **Tally 93 of 96.** The reviewer reproduces every v7 number and corrects its
own 014: the 1e−9 "finding" line assumed a two-term commutative sum, but `cell_emb` feeds both encoders, so the split
accumulates several contributions per leaf in engine order while DP sums within, then across, replicas — one-ulp
differences on a small fraction of elements, rel_L2 ~1e−9. *The finding is a mis-set line, not a property of DP.*

### 84.1 Terminations (C1) — their stopper monitors TEST loss, logged every epoch, and I push each session after seeing it
If a mid-run stop could be declared final, I would be choosing where the comparison is read. So:
1. **The only two FINAL terminations:** (a) their early stopping fires; (b) the **cumulative horizon of 297 completed
   epochs** is reached (5 × 59.4 epochs per session at the v7 rate) — a horizon in *epochs*, so a crashed or re-run
   session cannot consume it and slower epochs cost quota, not horizon.
2. **Everything else is INCOMPLETE** — a quota kill, a crash, a platform failure, abandonment: resumed if possible,
   otherwise **reported as no result and never read through §71.7.**
3. **Between sessions, the only inputs to any decision are timing (Amendment E), state integrity, and quota.** Never
   the logged loss.
4. **At early stopping:** `counter_at_end == 50 == marker.counter` is asserted, with `last_epoch_index` taken from the
   marker (the state dir holds one epoch fewer: no LambdaLR step follows an early stop). *Verified in their code:*
   `utils.py` sets `early_stop` when `counter >= patience`, so 50 is exact.
5. At the horizon, §71.7 decides admissibility with the 45 unchanged: a v9-win claim is admissible if XPert's best
   epoch is at or before 252 (the warm-split anchor's best was 164).

### 84.2 Hardening, all before session 1 (C2–C4)
- **C2:** a guard kill between the three per-epoch writes could leave `best.pth` from an earlier best beside a later
  `best_score`. The session now **stops only at epoch boundaries**: after each save, if elapsed + the slowest epoch so
  far would pass the deadline, it exits cleanly (the kernel's watchdog stays as a backstop). And `restore_state()`
  asserts `sha1(best.pth) == best_sha1` and `resume_from.epoch == full_state.epoch`.
- **C3 (chain of custody through git):** session k−1's three sha1s and final epoch are pasted as **literals into
  session k's kernel and committed before the push**. At restore, 81.3a is re-run **in process** (`collect_state()` vs
  the loaded state, every field) and any difference is fatal. `torch.__version__` and the CUDA version must equal
  session 1's; a changed stack is a stop-and-return.
- **C4:** before the real training in session 1, the kernel exercises the one new path end to end in test mode
  (truncated epochs, a test-only patience of 1): marker → termination → `counter_at_end == patience` → prediction
  of the 21,321 test rows. In production, **a trainer that exits without a marker and without a boundary stop is fatal.**
- **Ask 1:** `RECORD['deviations']` gains DataParallel (runtime patch inside `forward`), the ten frozen parameters, and
  the full-state resume hooks.
- **Ask 2:** terminating at the marker, before their post-loop test pass (never read), is accepted.

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
