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
