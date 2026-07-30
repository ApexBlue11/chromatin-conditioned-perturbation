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
| M.3 | **One fold, one seed** ⇒ no error bars on deltas quoted to 3 dp (+0.006 … +0.089), in a project that already produced an artefactual +0.103 [3.6]. Small deltas must be quoted as "within unmeasured seed variance" until measured | **GAP — needs accelerator** |
| M.4 | No **unseen dose/time** split, though we have nonlinear dose/time FiLM. XPert ships one | **GAP — cheap, eval-only** |
| M.5 | Metric set lacks **Common-DEGs** (top-k DE overlap), the metric closest to biological use. Do NOT add Wasserstein/E-distance reflexively — scPerturBench documents their failure modes (variance scaling; missed gene–gene dependencies) | **GAP — cheap** |
| M.6 | **Ahead of the field**: we compute a noise ceiling [6.1], report prediction *variance* not just correlation [6.8/6.10], report R² alongside correlation, test three generalisation axes incl. unseen-both, reliability-weight low-SNR labels, and falsified our own interpretability claim [4.1a-c] — the in-the-wild benchmark asks for most of this and notes it is usually absent | **A** |
| M.7 | **Uni-Mol input integrity VERIFIED**: 0/21,220 conformer failures, 0 drugs truncated at the 96-atom cap, 0 missing CLS embeddings. Uni-Mol's documented downstream failure (loses on SIDER because 3D conformers fail for natural products/peptides) **does not affect us** | **A** |
| M.8 | **Our contribution is another paper's stated future work**: the 2026 latent-diffusion L1000 paper lists what it lacks as *"additional cellular information like chromatin state or pathway activity"*. Cite as motivation — it supports the narrowed [7.1]. Pair with our own negative: scPerturBench's headline ask is cellular-context embedding for unseen contexts, and ours gives **≈0 on unseen cells** [2.3] | **A** |

## 7. Novelty (to verify before asserting)

| # | Claim | Status |
|---|---|---|
| 7.1 | ~~Conditioning a drug-perturbation predictor on cell-line chromatin state is novel~~ → **NARROWED 2026-07-27 after systematic search.** Chromatin IS already used for **drug-SENSITIVITY** prediction (GraOmicDRP/GraphDRP combine chromatin accessibility + PPI + mutations; eLife 78012 integrates RNA+ATAC for sensitivity signatures). **The precise surviving claim: conditioning transcriptional-RESPONSE-PROFILE prediction on cell-line chromatin state.** Response-profile models (XPert, PRnet, PertDiT, latent-diffusion, TransPro, Biolord) use expression/mutation/structure but **not chromatin**; chromatin models predict sensitivity, not the profile. No direct precedent found | **B** — 4 targeted searches across two framings; state it in the narrow form, and cite GraOmicDRP/eLife as adjacent |
| 7.2 | Cell-conditional pathway conductance appears novel | **C** — not systematically searched, AND per 3.1a it contributes no accuracy, so it is not worth claiming as a contribution |
| 7.3 | Atom→gene attention is ours-novel | **✗ UNCONFIRMED — do not claim.** XPert (Nature MI 2025) uses the same UniMol features; its Zenodo record, GitHub README and paywalled methods all lack the architectural detail needed to settle this. Resolve by reading the paper's methods or `XPert.zip` source |
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
4. Systematic novelty search for 7.1/7.2; read XPert's methods for 7.3.
5. Why epi helps unseen-compound but not unseen-cell (2.5 unresolved).
