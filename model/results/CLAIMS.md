# LINCS — Claims Ledger (for the writeup)

Every claim we can make, with its **evidence**, **strength**, and **what would falsify or is not yet shown**.
Rule: nothing moves to "SUPPORTED" without a measurement next to it. **Tempered/retracted claims are kept
deliberately** — they are part of the scientific record and stop us re-making old errors.
Detailed method/numbers: `RESULTS.md`. Model spec: `../MODEL_MATH.md`.

Strength key: **A** = measured, stratified correctly, reproducible · **B** = measured but confounded /
single-fold / small-n · **C** = suggestive only · **✗** = tested and NOT supported.

---

## 1. Model & accuracy

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 1.1 | ~~An interpretable (non-VAE, attention-attributable) model predicts LINCS L1000 differential response well above trivial baselines~~ (v5 cold-cell MSE 1.481 vs Mean/Meancell 1.747, Meandrug 1.733) | — **NARROWED 2026-07-30.** That comparison was MSE on **all** cold-cell signatures; on the **reproducible stratum with the metrics we actually report**, Meandrug **beats** v5 on unseen-cell (see 1.8). The claim survives only for unseen-COMPOUND / unseen-BOTH | **✗ as stated** |
| 1.8 | 🔴 **NEGATIVE RESULT: on unseen CELL, v5 is BEATEN by predicting the drug's training-set mean.** Meandrug wins on all three metrics — pearson **0.4475 vs 0.440**, R² **+0.2846 vs +0.273**, MSE **4.279 vs 4.348** — and a ridge on the same global inputs also edges it (0.4470). ⇒ **the model adds no cell-specificity.** Reproduces Ahlmann-Eltze/Huber/Anders (Nat Methods 2025) inside our own project | `baseline_linear.py`, RESULTS §18, protocol-matched | **A (negative)** |
| 1.9 | ✅ **…but chemical generalisation is real and large**: on unseen COMPOUND v5 beats the best linear by **+0.089** pearson (0.471 vs 0.382) and the best mean by **+0.106**; on unseen BOTH by **+0.120 / +0.135**. ⇒ **the model's value is chemical, not cellular** | same run | **A** |
| 1.10 | Below mean\|Y\| = 1 the model's **R² is negative** (−0.14 to −0.28) — the global mean is better there. Signal dilution quantified; it justifies rule #1 but means the ≥1 headline must be reported **beside all other bins**, not alone | RESULTS §18 all-strata table | **A** |
| 1.2 | Unseen-CELL (reproducible) Pearson **0.440**, R² 0.273 (v5) / **0.502**, R² 0.298 (v3) | `v5_metrics.json`, v3 `metrics.json` | **A** |
| 1.3 | **Unseen-COMPOUND (reproducible) Pearson 0.471**, R² 0.188 — on a leakage-audited Bemis-Murcko scaffold split | `v5_metrics.json` §12 | **A** |
| 1.4 | Unseen-BOTH Pearson 0.451 | `v5_metrics.json` | **A** |
| 1.5 | **Our model generalizes BETTER to new compounds than to new cell lines** (0.471 vs 0.440) — cell-specificity, not chemistry, is the hard part. **Sharpened 2026-07-30 by 1.8**: it is not merely harder, cell-specificity is **not being achieved at all** — unseen-cell performance is at baseline level | v5 three-way split, one run; `baseline_linear.py` | **A** |
| 1.6 | v3 reached ~70% of the replicate-reliability ceiling on reproducible cold-cell signatures | Pearson 0.53 vs ceiling √0.51–0.62 ≈ 0.71–0.79 | **B** (ceiling est.) |
| 1.7 | v5 < v3 on cold-cell because of the new components | — **CONFOUNDED**: v5 trained on 24% less data (179,712 vs 235,628) + 10 vs 12 epochs | **✗ do not claim** |

## 2. Epigenetics (our lead novelty candidate)

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 2.6 | ✅ **Epigenetic drugs are DRAMATICALLY more predictable than other drugs** — unseen-compound **0.6477 vs 0.4477** (+0.200), unseen-both **0.6300 vs 0.4275** (+0.203). Large, consistent across both splits, and on the *hardest* generalisation axes. 30-drug class: curated-ChEMBL chromatin-modifier targets ∪ canonical HDAC/DNMT names (STITCH-only edges excluded — they call glycerol an HDAC binder) | `probe_epidrug_v6.py`, n=992/186 per group | **A** |
| 2.7 | ⚠️ **…but NOT because the model uses chromatin for them.** Ablating chromatin to its mean changes epi-drug accuracy by **−0.0068** and other-drug accuracy by **+0.0070** (epi minus other = **−0.0138**, i.e. the *opposite* sign to "chromatin matters more for chromatin drugs"); on unseen-both the difference is **+0.0001**. All magnitudes ≤0.014 with no error bars ⇒ report as **no effect detected in either direction**, not as a reversal. ⇒ epi-drugs are predictable because their *transcriptional response is strong and stereotyped across cells*, i.e. it is drug-determined — the same theme as [1.8]: the drug sets the response, the cell contributes almost nothing | same run | **A (negative)** |
| 2.1 | Chromatin state (ATAC/H3K27ac/H3K27me3) contributes to drug-response prediction | Fair ablation, reproducible sigs, v3: **ΔR² +0.089** (0.355→0.266) | **A** |
| 2.2 | The effect is correctly SIGNED and mechanistic: low activation / high Polycomb ⇒ gene goes UP | r(H3K27ac,Y)=−0.174, r(ATAC,Y)=−0.143, r(H3K27me3,Y)=+0.067; partial vs X_base −0.137/−0.103/+0.055 | **A** |
| 2.2a | **CONFOUND TESTED AND REJECTED — not a floor/headroom artifact.** Stratifying by baseline-expression quartile, the sign **persists in ALL 4 strata for ALL 3 marks**: ATAC −0.196/−0.145/−0.188/−0.052, H3K27ac −0.296/−0.154/−0.216/−0.074, H3K27me3 +0.174/+0.046/+0.066/+0.074 (Q1=lowest expr → Q4=highest). Repressed genes having "more headroom" + noisier low-expression z-scores does NOT explain the effect | 28 cells, reproducible sigs, 2026-07-27 | **A** |
| 2.2b | Honest nuance: the effect **attenuates at high baseline expression** (Q4: −0.052/−0.074/+0.074) — strongest for lowly-expressed genes, but never sign-flips | 2.2a | **A** |
| 2.3 | **TEMPERED — the epi benefit tracks CELL FAMILIARITY and largely does NOT transfer to unseen cells**: in-dist **+0.089** → unseen-compound **+0.035** → unseen-cell **≈0 (−0.004)** | `ablate_v5.json` + v3 diagnose | **A** |
| 2.4 | The unseen-cell null is caused by missing epi coverage on test cells | — **REJECTED**: coverage equal (test 53% vs train 55%, both 1.12 marks) | **✗** |
| 2.5 | The unseen-cell null is caused by OOD chromatin profiles | — **WEAK**: 0.552 vs 0.593 mean sim, medians ~equal (0.543/0.535), n=9 | **C** |
| 2.6 | ⇒ Honest framing: epigenetics is an **in-distribution refinement**, not a demonstrated cold-cell generalization mechanism | 2.3–2.5 | **A** |
| 2.7 | (superseded, WRONG ×2) "epigenetics does not contribute / drop the branch" | Both calls were artifacts of evaluating on inert perturbations; all-signature dilution **inverted the sign** (−0.011 vs +0.073) | **✗ retracted** |

## 3. Pathway conductance (strongest accuracy component)

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 3.1 | ~~Pathway conductance is the component the model depends on most (ΔR² +0.103)~~ | — **RETRACTED 2026-07-26 (scale confound).** Learned conductance averages **0.60**; ablating `c→1` rescales pathway output by **1.67×**, so the +0.103 measured BREAKING THE LEARNED SCALE, not the mechanism | **✗ retracted** |
| 3.1a | **Controlled ablation (`c→mean`, scale preserved, structure removed): the cell/gene-specific structure contributes ≈ NOTHING** — ΔR² **−0.0033** unseen-cell, **+0.0058** unseen-compound (naive `c→1`: +0.1031 / +0.0890) | `finalized_analyses.json`, n=900/split, **independently reproduced 2026-07-27** | **A** |
| 3.1b | What the module actually learned = a **global damping factor + a per-cell scalar**: variance split **gene 2.2% / cell 84.7% / cell×gene 13.1%**, and **corr(c, X_base) = +0.742** | `pathway_maps.json` | **A** |
| 3.2 | It improves the model over static priors | — **NOT SHOWN**, and 3.1a makes it unlikely via the structural route. A matched run would now test only whether *having the capacity* changes what is learned | **✗ not shown** |
| 3.3 | It is interpretable: per-(cell,gene) scalar c∈(0,2) exposed as `aux["pathway_cond"]`; maps produced (`pathway_conductance.npy`) | maps + unit tests (zero-init ⇒ exact no-op at start) | **A** |
| 3.3a | The conductance is a **real chromatin readout** even though it does not buy accuracy: partial correlations vs baseline — ATAC **−0.26**, H3K27ac **−0.35**, H3K27me3 **+0.20** — the SAME direction as this dataset's measured epi→response relation (repressed genes respond more) | `pathway_maps.py` | **B** |
| 3.6 | **METHOD LESSON (generalize this):** an ablation that zeroes/neutralizes a learned multiplicative component also destroys its learned SCALE. Always ablate to the component's **mean**, not to 1/0, to separate *structure* from *scale* | 3.1 vs 3.1a | **A** |
| 3.4 | Per-EDGE cell-conditional priors are infeasible | [B,8,978,978] ≈ 10 GB/batch | **A** |
| 3.5 | Feeding "pathway activity" as a plain feature would add little | Module activity correlates **0.85** with a gene's own baseline (already in X_base) | **B** |

## 4. Interpretability / MoA

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 4.1 | ~~Atom→gene attention localizes to known drug targets (0.9× → 2.1× → 2.6× recall@5 by evidence tier)~~ | — **RETRACTED 2026-07-27.** recall@5 = 2.6 × means ~1.4 % vs 0.5 % random over ~111 gold drugs ≈ **1–2 drugs** with a target in the top 5 vs 0.5 expected — consistent with noise. The robust statistic over **149 gold pairs** is a median rank percentile of **0.560, WORSE than chance** | **✗ retracted** |
| 4.1a | **NEGATIVE RESULT (robust): atom→gene attention does NOT recover drug targets.** Median target rank percentile 0.560 over 149 gold pairs (0.5 = chance); per-drug case studies median 0.611 (11 textbook drugs) | `case_study.json` + gold-pair sweep | **A (negative)** |
| 4.1b | **The "targets don't move" confound is REJECTED** — attribution is not better when the target IS transcriptionally responsive: median percentile **0.645** when the target is among the top-10 % most-moved genes vs **0.493** when it barely moves; corr(attribution rank, true-response rank) = **−0.045** ⇒ no relationship. The failure is not an artefact of untestable targets | 149 gold pairs, stratified | **A** |
| 4.1c | ⇒ **Reportable as a negative result of value to the field:** we built an architecture designed for attribution, tested whether its attention recovers known drug targets under a curated reference, and it **does not**. Many papers assert attention ⇒ interpretability without such a test | 4.1a/4.1b | **A** |
| 4.2 | Predicted \|Ŷ\| also enriches for targets (2.3×@50 ChEMBL; median target rank pctile 0.36 vs 0.50 random) | `dti_eval.json` | **A** |
| 4.3 | The atom-token substrate is load-bearing, not decorative | Drug ablation: zeroing atoms costs **Δpearson +0.163** (largest of any feature) | **A** |
| 4.4 | Effect is **modest** — recall@50 ≈ 10%; partly a biological ceiling (a drug's target GENE is often not where its transcriptional response peaks) | `dti_eval.json` | **A** |
| 4.5 | Claim wording must be "attention is ENRICHED for curated/gold targets", **NOT** "attention recovers targets" | 4.1/4.4 | — |
| 4.9 | ⚠️ **PER-DRUG LOCALISATION FAILS — case study, 2026-07-27.** For 11 textbook drug/target pairs from the gold tier, the target's median rank percentile under atom→gene attribution is **0.611 — WORSE than chance (0.5)**; \|Ŷ\| gives 0.272. Wildly inconsistent: propranolol→ADRB2 0.054, troglitazone→PPARG 0.063 (excellent) vs salbutamol→**ADRB2 0.990**, danusertib→AURKA 0.941, colchicine→TUBB6 0.812 — *the same target gene* scores 0.054 and 0.990 for two different drugs | `case_study.json` | **A (negative)** |
| 4.10 | **Reconciliation (not a contradiction):** recall@5 = 2.6× means the target reaches the top 5 of 978 in ~1.4 % of cases vs 0.5 % by chance — a weak enrichment in the **extreme tail**, fully compatible with a median rank ~0.5. **Top-k enrichment does NOT imply per-drug localisation**; reading it that way was an error | 4.1 vs 4.9 | **A** |
| 4.11 | ⇒ **Do NOT publish a case-study figure**, and never describe the model as identifying a drug's target. Defensible statement: a **population-level, tail-concentrated enrichment that scales with reference confidence** — evidence the attribution is non-random, but useless as a per-drug target predictor. Underpowered (n = 1–8 signatures/drug, 11 drugs) but the direction is not encouraging | 4.9/4.10 | **A** |
| 4.6 | Pathway-prior ATTENTION flow is weak: prior heads only **1.1×** random on-support (λ non-zero, 0.24–0.94) | `moa.json` | **A** |
| 4.7 | ⇒ Interpretability ranking: **epigenetics > atom→gene DTI > pathway attention**. (Note: pathway *conductance* matters far more for ACCURACY than pathway *attention* does for interpretability) | 2.1, 4.1, 4.6, 3.1 | **A** |
| 4.8 | Measured only on the LAST perturb layer (lowest λ); early/base layers (λ up to 0.9) may steer more | `moa.py` scope | limitation |
| 4.15 | ⚠️ **METHOD TRAP (generalise this): for pathway-level target alignment, 0.5 is NOT the chance level.** Pathways containing a drug's target are systematically large and central, so they rank high under *any* scoring. Run the MoA probe on an **untrained** model and the median target-pathway rank percentile is **0.218** — which against "chance = 0.5" reads as a striking discovery — while the label-permutation null sits at **0.229** (p = 0.13, i.e. nothing). **Judge only against the permutation null**; comparing to 0.5 lets you "find" mechanism in a randomly-initialised network. Same family as [3.6] (ablate to the mean) and [6.3] (a tail statistic is not localisation) | `probe_moa_v6.py`, untrained-model control | **A** |
| 4.14 | **A drug-specific pathway readout IS mechanically available from v6 without a retrain — via GRADIENTS, not activations.** P-NET reads node *importance*, not raw activation; we built the activation readout. Measured on a live model: activation `a_p` changes by **exactly 0.0** under a different drug (4.12), but `∂Ŷ/∂a_p` changes by **0.076** — the drug enters below the pathway layer, so the gradient carries it. ⚠️ **But on an untrained model the per-pathway importance `\|a·∂Ŷ/∂a\|` RANKING is drug-invariant** (corr **+1.0000** between two unrelated drugs, top-5 identical 5/5): the drug-dependence enters as a near-uniform scale, not a reordering. Whether a *trained* model reorders is open and is the first thing to test on the checkpoint. **Do not assume it will** — the last attention-based MoA readout was falsified [4.1a] | live-model gradient probe | **A** |
| 4.13 | ⚠️ **v6's "PathwayBottleneck" is NOT a bottleneck — it is a masked SIDE BRANCH** (found 2026-08-14 reviewing the design against P-NET). `model_v6` adds its output residually (`h = h + delta`), so the main residual stream can route around it; at init, with `w_out` zero-init, it does exactly that. P-NET's interpretability guarantee comes from its pathway layers being the **only** route to the outcome. ⇒ **"pathway p's activation is a faithful masked aggregation of pathway p" HOLDS** (verified by test_v6: 243 no-pathway genes get exactly 0; perturbing a gene moves exactly its 91 pathways); **"the prediction had to pass through pathway p" does NOT.** Three of our own docs overstated this as "information is forced through named nodes" — corrected. How much it actually contributes is exactly what `eval_v6`'s `mean_pathway` ablation measures, and is still **unmeasured** | code + test_v6; ARCHITECTURE §5/§7 | **A** |
| 4.12 | ⚠️ **v6's pathway readout is DRUG-INVARIANT BY CONSTRUCTION — verified 2026-07-30.** `pathway_activations` is produced at forward step (4), *before* the drug is introduced at step (5). Measured on a live model: substituting a completely different molecule changes the readout by **exactly 0.0**, while baseline expression (2.54), chromatin (1.28) and dose/time (1.59) all move it. ⇒ it is a **cell × dose/time** quantity: "which named pathways are chromatin-permitted in this cell at this exposure". **v6 therefore does NOT repair 4.1a/4.1c** — those are DRUG-level claims, and scoring this readout against drug→target annotation is structurally guaranteed to return a null that means nothing. The valid test uses the measured LINCS response as the independent annotation | live-model probe; `model/v6/probe_pathways_v6.py` | **A** |

## 5. Drug features

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 5.1 | ECFP4 fingerprint and per-atom UniMol tokens are the two pillars | ΔR² +0.089 / +0.075; Δpearson +0.147 / **+0.163** | **A** |
| 5.2 | ChemBERTa is dead weight (dropped, −384 dims) | ΔR² **+0.001**, Δpearson +0.002 | **A** |
| 5.3 | RDKit descriptors contribute negligibly (+0.003) | drug ablation | **A** |
| 5.4 | Cell lineage is marginal | Δ +0.001…+0.017; 26/83 cells UNKNOWN, incl. 7/17 cold cells (the best performers) | **B** |

## 6. Data & methodology (defensible methodological contributions)

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 6.1 | **LINCS L1000 is GOOD data; ~75% of perturbations are biologically INERT** — never judge a model or feature on all signatures | Replicate r 0.127 overall vs **0.509/0.619** for mean\|Y\|≥1; cross-phase r=0.561; matches CMap's ~40% gold rate | **A** |
| 6.2 | Evaluating on inert-dominated sets can **INVERT** a real effect's sign | Epi ΔR²: −0.011 (all sigs) vs **+0.073** (reproducible) | **A** |
| 6.3 | Same failure mode appears in DTI validation: a noisy reference (98% STITCH) **masks a real signal** | 0.9× (all) vs 2.6× (gold) | **A** |
| 6.4 | **Bemis-Murcko scaffold splitting alone does NOT make compounds "unseen"** — median max-Tanimoto to nearest train drug **0.655**, 39.2% ≥0.70, 8.0% ≥0.85 | `scaffold_split.json` leakage audit, 21,220 drugs / 6,035 scaffolds | **A** |
| 6.5 | ⇒ Any unseen-compound number (ours or published) must be reported **with a leakage audit**; a random drug split is far worse | 6.4 | **A** |
| 6.6 | Published SOTA numbers are **not comparable to ours**: they predict ABSOLUTE expression with basal supplied; we predict the DIFFERENTIAL, reproducible-filtered | Paper methods confirmed (latent diffusion: unseen-cell 0.743 / unseen-compound 0.870) | **A** (protocols differ) |
| 6.6a | Under the absolute convention **R² nearly doubles on identical predictions**: unseen-cell 0.281→0.497, unseen-compound 0.195→0.388 | `dual_metric.json` | **A** |
| 6.6b | The **PCC** inflation is large | — **NOT SHOWN in our data**: measured only +0.040 / +0.006, because our normalized `X_base` anchor has basal:delta variance ratio **0.4–0.6**, far below a real raw-expression setup. Our anchor is the wrong scale ⇒ inconclusive as a protocol proxy | **✗ as measured** |
| 6.6c | PCC under the absolute convention is a **function of the basal:delta variance ratio**, on UNCHANGED v5 predictions — unseen-compound **0.470→0.680→0.809→0.918→0.999** and unseen-cell **0.438→0.687→0.796→0.908→0.999** at ratios ~0 / 1.6–2.4 / 3.5–5.4 / ~10–15 / ~1000. **Non-monotonic**: dips to 0.381–0.397 at ratio ~0.1–0.2 (anchor adds uncorrelated variance before dominating) | `finalized_analyses.json`, both splits n=900 | **A** |
| 6.6d | ~~Published SOTA PCCs fall at ratio ≈2–4 on our curve, where we score 0.68–0.81, so we would match them~~ | — **RETRACTED 2026-07-26 (circular reasoning).** Locating their PCC on OUR curve silently ASSUMES their model is as good as ours and back-solves the ratio. A better model reaches 0.87 at a LOWER ratio. Their ratio CANNOT be inferred from their PCC without independently knowing their model quality | **✗ retracted** |
| 6.6e | ⇒ We match/beat SOTA | — **NOT SHOWN**, and 6.6d cannot be used to argue it | **✗ do not claim** |
| 6.6f | The DEFENSIBLE version: absolute-convention PCC is **highly sensitive to the basal:delta variance ratio** (0.485→0.999 across ratios 0→1000 on fixed predictions), therefore absolute-convention numbers are **uninterpretable across protocols unless that ratio (and the basal source) is reported** | 6.6c | **A** (measurement) |
| 6.6h | ⚠️ **This is NOT a novel contribution — retracted as such 2026-07-27.** The effect is already established in the perturbation-prediction literature as **"control bias"** (systematic control-vs-perturbed differences inflating control-referenced correlations) and **"signal dilution"**; **delta-based metrics (PearsonΔ, PDCorr) are already the field standard**. See Systema (Nat Biotech 2025), Nat Methods 2025 benchmark, PerturBench, "DL does not yet outperform simple linear baselines" (PMC12328236), "Evaluating Single-Cell Perturbation Response Models Is Far from Straightforward". **Cite this literature; do not claim discovery.** Our sweep is a useful *illustration* on our data, nothing more | literature search 2026-07-27 | **✗ novelty retracted** |
| 6.6i | **Useful consequence:** since delta metrics are the standard, our differential Pearson **is** on the field-standard axis ⇒ papers reporting PearsonΔ/PDCorr give directly comparable numbers. Comparison is more tractable than the absolute-convention papers suggested | 6.6h | **A** |
| 6.5a | Scaffold-split leakage being insufficient is **likely also not novel** (well known in cheminformatics). Our contribution is at most the specific quantification for the LINCS compound library | — | **C — verify before claiming** |
| 6.6g | ⇒ The only sound comparison is a **protocol-matched re-benchmark**: same data level, same split, same basal source, same metric — ideally reproducing their published number on their own split first | 6.7 | — |
| 6.7 | Cross-model comparison requires re-benchmarking under a common protocol (run their code on our metric), incl. first reproducing their published number on their own split | standard practice | — |
| 6.8 | The drug×cell interaction "under-expression" (26.5% vs 47.9%) is largely **OPTIMAL noise-hedging**, not a fixable defect: the model sits at MSE-optimal dispersion (std-ratio 0.47 ≈ corr 0.42) | `analyze.json`, v4 negative | **A** |
| 6.9 | A correlation/rank loss fixes the interaction shrinkage | — **TESTED, NEGATIVE**: probe flat 0.47→0.47, corr 0.38, pearson +0.01 (noise) | **✗** |
| 6.10 | ⇒ Interaction magnitude cannot rise without first raising cell-specific CORRELATION (an accuracy problem, not a loss problem) | 6.8/6.9 | **A** |

## 6b. Literature corroboration (2026-07-27, searches — see sources at end)

| # | Our finding | Literature status |
|---|---|---|
| L.1 | Genes in **repressed/less-active** chromatin show LARGER signed responses (H3K27ac −0.174, ATAC −0.143, H3K27me3 +0.067) | **Consistent**: Polycomb-repressed genes in *moderately* H3K27me3-marked chromatin **remain inducible** and show dynamic transcriptional responses (vs fully silenced chromatin). Our direction matches this "poised/inducible" biology |
| L.2 | Chromatin state predicts *which* genes can respond | **Supported**: enhancer-priming work — chromatin accessibility in naive cells shapes stimulus-specific response (~60% of stimulus-specific eQTLs with chromatin effects alter accessibility in the naive state); transcriptional-memory priming via accessibility landscape |
| L.3 | Chromatin + transcriptome jointly inform drug MoA/sensitivity | **Supported** (eLife: integrated transcriptome + chromatin state decodes MoA and sensitivity) — but that is *analysis*, not a chromatin-conditioned predictive model |
| L.4 | **Confound NOT excluded by literature — floor/noise effect** | Repressed genes have low baseline expression ⇒ more headroom to rise, and L1000 z-scores are noisier for low-expressed genes. Our partial correlation vs `X_base` (H3K27ac −0.137) controls for this *linearly* but does not fully exclude it. **Test: stratify the epi↔response correlation by baseline-expression bin** |

## 6c. Methodology audit vs the field (2026-07-30) — full detail in `../LITERATURE_PRACTICE.md`

§6b audits our biology, §7 our novelty; this audits our **method** against the three adversarial
benchmarking papers that exist to catch what we try to catch in ourselves.

| # | Finding | Status |
|---|---|---|
| M.1 | ~~We have NO linear baseline~~ → **DONE 2026-07-30, and it changed a headline claim.** `baseline_linear.py` fits it; on unseen-cell the model is **beaten** by Meandrug and by ridge ⇒ see 1.8/1.9. The old "beats all naive baselines" (1.1) is retracted as stated | **CLOSED — result in RESULTS §18** |
| M.2 | **Our stratum is defined by the ground truth we score against** (mean\|Y\| ≥ 1). Ahlmann-Eltze: selection by differential expression "cannot be applied in real-world use cases". Mitigating: our rationale is measurement *reliability* (r 0.127 → 0.509/0.619 [6.1]), and the bias direction is not clearly optimistic (large-\|Y\| rows include large-*noise* rows, which depress scores). Real problem = auditability + prospective applicability | **PARTLY CLOSED** — all-strata reporting now in `eval_v6.py` + `baseline_linear.py` (see 1.10); an *independent* stratifier (M.2b) is still open |
| M.9 | 🔴 **DATA BUG, found 2026-07-30 while designing the unseen-dose/time split**: the `dose` field mixes units across 110 distinct strings and `_num()` discarded the unit, so **500 nM parsed as 500 µM** — 1000× too large on **13,910 rows (4.49 %)**. Multiplicative on a log axis ⇒ it **inverted the ordering**, putting the lowest doses at the top (median z +1.710 → −1.769 after fix). Fixed in `data.py::_dose_um`; `legacy_dose_parsing` reproduces the old behaviour, which **every checkpoint including v5 was trained with** | **A — fixed; RESULTS §19** |
| M.10 | 🔴 **SEED VARIANCE EXCEEDS EVERY ARCHITECTURE EFFECT THIS PROJECT HAS EVER REPORTED** (2026-08-15, first time ≥2 seeds of one identical config were run). Three v7 seeds, protocol-matched: unseen-cell 0.4322/0.4374/0.4425 (sd 0.0052), unseen-compound 0.4448/0.4624/**0.4746** (sd 0.0150), unseen-both 0.4237/0.4532/**0.4694** (sd **0.0232** ⇒ 2-sd = ±0.046). **v5 AND v6 both fall INSIDE the v7 seed range on unseen-compound and unseen-both** — the three architectures are statistically indistinguishable there; only on unseen-cell is v6 reliably ~0.009 above v7. The v6−v5 differences we interpreted were +0.0066/−0.0024/+0.0153, all inside the same-split seed range. ⇒ **three full rebuilds were compared on differences the measurement cannot resolve.** **Report mean ± range over ≥3 seeds or report no difference.** Supersedes M.3. **Does NOT affect ablations**, which are within-run on identical signatures and carry no seed variance — that is why ablate-to-mean kept yielding clean answers while headline deltas were noise | `eval_v7.py`, 3 seeds, protocol-matched | **A** |
| M.3 | **One fold, one seed** ⇒ no error bars on deltas quoted to 3 dp (+0.006 … +0.089), in a project that already produced an artefactual +0.103 [3.6]. Small deltas must be quoted as "within unmeasured seed variance" until measured | **GAP — needs accelerator** |
| M.4 | No **unseen dose/time** split, though we have nonlinear dose/time FiLM. XPert ships one | **GAP — cheap, eval-only** |
| M.5 | Metric set lacks **Common-DEGs** (top-k DE overlap), the metric closest to biological use. Do NOT add Wasserstein/E-distance reflexively — scPerturBench documents their failure modes (variance scaling; missed gene–gene dependencies) | **GAP — cheap** |
| M.6 | **Ahead of the field**: we compute a noise ceiling [6.1], report prediction *variance* not just correlation [6.8/6.10], report R² alongside correlation, test three generalisation axes incl. unseen-both, reliability-weight low-SNR labels, and falsified our own interpretability claim [4.1a-c] — the in-the-wild benchmark asks for most of this and notes it is usually absent | **A** |
| M.7 | **Uni-Mol input integrity VERIFIED**: 0/21,220 conformer failures, 0 drugs truncated at the 96-atom cap, 0 missing CLS embeddings. Uni-Mol's documented downstream failure (loses on SIDER because 3D conformers fail for natural products/peptides) **does not affect us** | **A** |
| M.8 | **Our contribution is another paper's stated future work**: the 2026 latent-diffusion L1000 paper lists what it lacks as *"additional cellular information like chromatin state or pathway activity"*. Cite as motivation — it supports the narrowed [7.1]. Pair with our own negative: scPerturBench's headline ask is cellular-context embedding for unseen contexts, and ours gives **≈0 on unseen cells** [2.3] | **A** |

## 6d. v9 substrate, priors and benchmarking (2026-08-26) — detail in `RESULTS.md` §27–28

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 6d.1 | ✅ **Level-3 coverage 44.8 % → 99.65 %** by joining Level 5 to Level 3 on `sig_info.distil_id`, the exact well mapping MODZ used, instead of a (cell, pert, dose, time) key. P1 100.00 %, P2 98.98 % | `extract_level3_distil.py`; 26/26 design tests, incl. recomputing sampled signatures from the GCTX to 1.7e-06 | **A** |
| 6d.2 | 🔴 **The old key join was not merely lossy, it was WRONG for P2**: its condition table holds GSE92742 wells only, so 4,419 P2 signatures were paired with a "plate-matched" control from a different experiment (3.2 % of the joined rows) | RESULTS §27.1 | **A (negative)** |
| 6d.3 | ✅ **The plate-matched control beats CCLE, reproduced on the clean join**: +0.0268 / +0.0401 / +0.0354 against the original +0.0314 / +0.0435 / +0.0372 | `ab_matched_control.py`, closed-form ridge, identical rows | **A** |
| 6d.4 | ✅ **CCLE is redundant, not harmful, and is DOMINATED**: `both − matched` ∈ [−0.0104, +0.0003] on every target and split; a per-cell mean of L1000 DMSO controls matches CCLE to 4 dp on seen cells and beats it by +0.037 / +0.027 on unseen cells | RESULTS §27.4 | **A** |
| 6d.5 | 🔴 **The delta convention lets a model cancel noise instead of predicting biology.** With two independent half-plate DMSO medians, the coupled control beats the independent one by +0.077 / +0.008 / +0.076; on unseen CELLS the independent plate control is 0.08 WORSE than a per-cell mean, so nearly all of the apparent gain there is noise cancellation | `split_dmso_control.py`; mean \|ctlA − ctlB\| = 0.185 against mean\|delta\| 0.377 | **A (negative)** |
| 6d.6 | 🔴 **RESULTS §25's "Level 5 is markedly better denoised" is a stratum artefact.** It compared an L3-defined top quartile (0.2429) with the L5-defined reproducible stratum (0.509–0.619). On the stratum we evaluate on, the L3 delta self-agrees at **0.5283** | `noise_ceiling_l3.py`, 56,811 signatures, halves from disjoint plates | **A** |
| 6d.7 | ✅ **The absolute convention self-agrees at 0.90–0.95 regardless of perturbation strength** — the local proof of why absolute numbers look high, beside "copy the control" = 0.9200 | same run | **A** |
| 6d.8 | ✅ **34/978 landmark symbols are stale 2012 L1000 names**; resolving them through Entrez against the local HGNC set moves STRING-absent 28 → 8 and Reactome orphans 231 → 213, with no collisions | `build_priors_v9.py` | **A** |
| 6d.9 | ✅ **STRING truncation was real and is fixed**: full proteome at score ≥ 400 gives 19,496 nodes / 929,472 edges (XPert's released graph: 19,392 / 901,260, so 400 is the field's threshold, derived not assumed). Isolated landmarks 66 → 8 | same | **A** |
| 6d.10 | 🔴 **The handoff's Reactome diagnosis is wrong.** Nothing was truncated: `ReactomePathways.gmt` annotates 11,963 genes, so 213 landmarks are in no pathway at ANY filter (verified at min_size=1 with the umbrella exclusion off). "231 → ~0" is unreachable from Reactome; GO:BP as a second NAMED source gives 50 orphans against a two-source floor of 45 | same | **A (negative)** |
| 6d.11 | ✅ **Pretrained gene vectors carry biology, not just topology**: held-out link prediction AUC 0.9198 (degree-matched null), and pathway CO-MEMBERSHIP — never in the objective — recovers at AUC 0.6388 | `pretrain_gene_vectors.py` | **B** (one seed) |
| 6d.12 | 🔴 **XPert's 15 released splits are NOT tissue holdouts.** They restrict to one tissue and split it ~90/10: mean over the 15, 100.0 % of test rows use a cell line seen in training, 98.6 % a compound seen in training, and **89.4 % the exact (cell, compound) PAIR** | `sota_split_audit.py`, read from their own h5ad | **A** |
| 6d.13 | ✅ **The split alone is worth +0.17 on the delta metric** (cold_both 0.3764 → xpert_pair 0.5459) for one ridge with one feature set — about 4× the 2-sd seed band and larger than every architectural effect this project has measured, combined | `sota_matched_difficulty.py` | **A** |
| 6d.14 | 🔴 **In the absolute convention our ridge is WORSE than copying the control on unseen cells** (0.9184 vs 0.9257) and on cold_both (0.8966 vs 0.9080). Value added over doing nothing: XPert's released predictions +0.060, our ridge on the comparable regime +0.021 | same | **A (negative)** |
| 6d.15 | 🔴 **An unfitted quantiser makes the expression input a silent no-op** — the model trains and reports numbers while ignoring expression entirely (\|dY\|max exactly 0.0000). Found by a design test, not by inspection; `BinnedExpression` now refuses to run before `fit()` | `test_v9.py`, 49/49 | **A (negative)** |

## 6e. Running XPert's own code and weights (2026-08-30) — detail in `RESULTS.md` §41

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| 6e.1 | **Their release cannot be run as shipped**: `processed_data/` contains one `gitkeep.txt`. The missing assets are on Zenodo (`10.5281/zenodo.17182939`) and were retrieved by HTTP range request — the zip central directory, then only the needed members (714 MB of 1.6 GB), and only the 1,970 rows of their 4.5 GB UniMol array this benchmark touches (1.1 GB). Every member CRC32-verified; array offsets checked three ways | `fetch_xpert_assets.py`, `fetch_xpert_unimol.py`, 26/26 checks in `test_xpert_compare.py` | **A** |
| 6e.2 | 🔴 **Their published HDACi figure (Pearson_deg 0.8440) is not reproducible from any of their three released mdmt checkpoints** — they give 0.6444, 0.7610, 0.7297 on the same 3,439 rows with their own code and metric | `xpert_native_eval.py` | **A** |
| 6e.3 | 🔴 **Their released HDACi predictions score the same off their benchmark as on it** (0.8413 vs 0.8496), while the warm checkpoint drops 0.7973 → 0.5689 across the same boundary. That is the signature of a model that had those rows in training, not of generalisation | 1,136 in-corpus vs 2,303 out-of-corpus rows | **A** for the measurement; **B** for the inference about their training set |
| 6e.4 | Our driver measures generalisation, not fit: on benchmark rows using the same 30 compounds the warm checkpoint scores **0.7974 on `split_1` train rows and 0.7968 on test rows** — no memorisation gap | `xpert_native_eval.py --row_file` | **A** |
| 6e.5 | **Their metric is the MEAN of per-row Pearson; ours has always been the median**, which flatters by ~0.02–0.05 on these data. Every earlier cross-paper number in this project mixed the two conventions | their `metrics.py::pearson`; equivalence checked row by row | **A** |
| 6e.6 | **FlashAttention is their DEFAULT forward path, not an optional speedup** (`if output_attention: dense else: flash`), and the dense branch applies an attention mask the flash branch never receives — so the branches are not interchangeable. Answers "why don't we use flash attention": it is an exact kernel, so it changes speed and not results, but in *their* code the choice also silently changes masking | `models/model_utils.py:200-231`; shim verified to 5e-7 and verified to differ from the masked branch | **A** |
| 6e.7 | **A closed-form ridge reaches delta Pearson 0.6054 on their own held-out rows** (`split_1`, n=13,766), where copy-the-control is 0 by construction and mean-drug-delta is 0.2211. Any delta number on this benchmark must be read against 0.605 | `xpert_mdmt_baselines.py` | **A** |
| 6e.8 | **18.9 % of their benchmark rows pool 2–8 distinct doses into one condition**; their model only ever sees the dose BIN (`pert_dose_idx`), so v9 is given the same resolution and no finer | `xpert_mdmt_extract.py`, flag carried per row | **A** |

## 6f. Their published numbers, read from the Supplementary Information (2026-09-20) — detail in `RESULTS.md` §46

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 6f.1 | 🔴 **§36.3 was false and cost six weeks.** "The per-model absolute values live inside figure panels and are not extractable" — they are in **Supplementary Table R8**, fetched unauthenticated from Springer static-content in ~30 s. Every number this project reverse-engineered (§23, §30, §39–42) was published | `external/xpert/supplementary/` | **A (negative, ours)** |
| 6f.2 | ✅ **XPert L1000_mdmt, fivefold CV, PCC on xdeg: warm 0.688 ± 0.011, cold-drug 0.645 ± 0.008, cold-cell 0.383 ± 0.027.** TranSiGen cold-cell 0.293 ± 0.017; Mean 0.224 | Supplementary Table R8 | **A** |
| 6f.3 | 🟢 **§42 is a SUCCESSFUL INDEPENDENT REPRODUCTION.** Our recovered honest number for their released checkpoint on `split_2` is **0.6933** against their published **0.688 ± 0.011** — within half an sd. Validates the Zenodo range-fetch, the `strict=True` load under `--include_cell_idx`, the flash-attention shim and the metric convention | §42.2 vs Table R8 | **A** |
| 6f.4 | 🔴 **§42's "contamination" framing is retracted as loaded.** Their Methods: *"all datasets are strictly split using fivefold cross-validation."* They trained five models and released one; that it is fold 2's is expected. Surviving content is a **usage note** — scoring it on `split_1/3/4/5` gives an inflated 0.738–0.745 | paper Methods | **A (retraction, ours)** |
| 6f.5 | 🔴 **0.8440 is retracted as a benchmark reference.** It is the Fig. 4 vorinostat/HDACi **case study**, not their benchmark number. Cited wrongly by this project for six weeks | paper text; §36.2 | **A (retraction, ours)** |
| 6f.6 | 🟢 **v9 sits +0.090 above XPert's published cold-cell PCC** (0.4734 vs 0.383 ± 0.027) — 3.3 of their own fold sd. **Calibration anchor:** our ridge scores 0.2959 where their published TranSiGen scores 0.293 | §44, §45, Table R8 | **B — our-number-vs-their-number, 1 fold of 5, 1 seed, batch 8. NOT a head-to-head** |
| 6f.7 | ✅ **XPert can now be trained by us** — §41 recovered every missing input, shim verified to 5e-7. They released no cold-cell checkpoint, so a direct cold-cell head-to-head requires training theirs. Same machinery admits TranSiGen / PRnet / DeepCE / CIGER, never run here | §41, §46.5 | **A (feasibility)** |
| 6f.8 | Their Methods list **four** split strategies for L1000_mdmt; the fourth, `cold-dose&time`, has never been tested here despite our nonlinear dose/time FiLM | paper Methods; [M.4] | **A (gap)** |

## 7. Novelty (to verify before asserting)

| # | Claim | Status |
|---|---|---|
| 7.1 | Conditioning transcriptional-RESPONSE-PROFILE prediction on cell-line chromatin state | **NOVELTY still B.** **BENEFIT NOW MEASURED, §45: real but small.** Deep-model ablation on unseen cell lines (`split_cold_cell_1`, 8 held-out lines, 21,151 paired rows): chromatin ON 0.4734 vs mean-ABLATED 0.4692, **+0.0042 [+0.0036, +0.0049], p~0**. Significant, and **2.4 % of v9's +0.178 margin over a ridge on that split** — ablate it entirely and 97.6 % of the advantage remains. A ridge given the same chromatin encoded exactly gains +0.0029 [§44], so the per-gene embedding buys ~0.001 over a linear form. **Claim it as a measured minor contribution, never as the reason the model works.** One seed pair so far; two more queued | **novelty B / benefit A but MINOR** |
| 7.2 | Cell-conditional pathway conductance appears novel | **C** — not systematically searched, AND per 3.1a it contributes no accuracy, so it is not worth claiming as a contribution |
| 7.3 | ~~Atom→gene attention is ours-novel~~ → **✗ RESOLVED NEGATIVE 2026-08-30. DO NOT CLAIM.** Settled by reading their source, which §41 obtained. `models/model_utils.py::CrossAttention.forward(cell, drug)` takes the **query from the 978 gene tokens and the key/value from the drug's ATOM tokens** (`unimol_Embeddings` builds the sequence `[dose, time, HG_embed, atom_1..atom_n]`), so XPert computes a per-gene × per-atom attention matrix — the same mechanism, on the same UniMol features. They also ablate it explicitly (`--wo_atom`, `--wo_atom_HG`) and visualise it for interpretability in `reproducing/fig3/*_attn_molecule.ipynb`. The open item "read XPert's methods for 7.3" is closed. | **✗ FALSIFIED** — the architectural detail was in the released source all along; the paywalled methods were never the blocker |
| 7.4 | SOTA accuracy leaders are VAE/diffusion black boxes; interpretability is our differentiator | **B** — true of the models found (latent diffusion, PertDiT, PRnet); XPert is also interpretable |

## 8. Infrastructure facts (save time later)

- Local **torch 2.11.0+cpu** exists (system Python 3.14) — run unit tests & analyses locally; the old
  "no local torch" belief was stale. `drug/.venv-drug` has rdkit but no torch.
- Kaggle: **never P100** (sm_60 unusable) → push with `--accelerator NvidiaTeslaT4`; `train.py` has a
  fail-fast GPU probe. Caps: **5 concurrent CPU sessions, 2 GPU**. A **cancelled** kernel's `/kaggle/working`
  is DISCARDED — never hard-cancel a long run.
- TPU: torch_xla 2.8, **8 cores**, but **39.4 ms/core vs T4 ~30 ms ⇒ slower per core**; only ~3.1× via
  full 8-core parallelism. XLA-safe fixed padding is implemented and **proven leak-free** (0.00e+00).
- We have **Level 5 only** — no L1000 controls/absolute expression (the Level-3 GCTX in the provenance
  manifest is on a machine we no longer have).

---

## Open items blocking stronger claims
0. **Ridge linear baseline (M.1)** — CPU, minutes, blocks every accuracy claim. Then **report all mean|Y|
   strata (M.2)**, **Common-DEGs (M.5)**, **unseen dose/time split (M.4)**. None need an accelerator.
1. **Matched run** (same data/epochs, pathway off) → converts 3.1 (dependence) into 3.2 (value-add). ~3.2h GPU.
2. **Fair SOTA comparison** — train PRnet on our split/metric (~2–4h GPU); reproduce their published number
   on their split first.
3. Pathway-conductance **maps** (per-cell/per-gene) — CPU, cheap, turns 3.3 into a shown deliverable.
4. Systematic novelty search for 7.1/7.2. **7.3 is closed — falsified**: XPert has the same gene×atom cross-attention, read directly from their released source (§41).
5. Why epi helps unseen-compound but not unseen-cell (2.5 unresolved).
