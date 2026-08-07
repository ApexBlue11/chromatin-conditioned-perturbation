# Chromatin-Conditioned Perturbation Prediction

**Predicting drug-induced transcriptional response from chromatin state, molecular structure, and network priors.**

A deterministic, fully-attributable cross-attention model that predicts the *differential* transcriptional
response (978 LINCS L1000 landmark genes) to small molecule perturbations — conditioned on cell-line
chromatin accessibility, histone modifications, pathway topology, and atom-level molecular representations.

---

## Why This Project

Computational pharmacology wants to predict how a cell responds to a drug. The leading models in this
space — including state-of-the-art diffusion-based architectures — predict *absolute* post-perturbation
expression while being supplied the unperturbed baseline as input. A large and typically unreported share
of their correlation simply reflects reproduction of that supplied baseline (*control bias* / *signal
dilution*), not prediction of the drug effect.

We build a model that:
1. **Predicts the differential** (LINCS Level 5 MODZ z-scores) directly — not absolute post-perturbation
   expression — consistent with the delta-metric standard established in recent benchmarks.
2. **Conditions on chromatin state** (ATAC-seq, H3K27ac, H3K27me3) as an additional axis of cell
   specificity, with the biological hypothesis that chromatin accessibility and Polycomb repression
   determine which genes a cell is *poised* to respond to a perturbation.
3. **Reports negatives as prominently as positives**, including retracted claims, a 30× ablation artefact
   we caught and corrected, and a falsified drug-target attribution claim.

The scientific framing is interpretability-first: every prediction should trace to an attributable
substrate. The core contribution is **chromatin conditioning and what it reveals — including its limits**.

---

## Results Summary (v5 — measured, single fold 0)

| Split | Pearson | R² | MSE |
|---|---|---|---|
| Unseen cell | 0.440 | 0.273 | 4.348 |
| **Unseen compound** | **0.471** | **0.188** | — |
| Unseen both | 0.451 | — | — |

> All numbers on the **reproducible stratum** (mean|Y| ≥ 1): ~75% of LINCS perturbations are biologically
> inert (replicate r ≈ 0.07–0.14); the real-response subset has replicate r ≈ 0.51–0.62. Evaluating on
> all signatures once **inverted the sign** of the chromatin effect. Every metric here is on the
> reproducible subset. Not cherry-picking — a precondition for correct inference.

**Key findings:**

- **Chromatin contributes (ΔR² +0.089 in-distribution)** with the mechanistically expected sign — low
  activation and high Polycomb marking predict larger responses — and this *survives stratification by
  baseline-expression quartile in all four strata for all three marks*, ruling out a floor/headroom
  artefact. The benefit declines monotonically with cell familiarity (+0.089 in-dist → +0.035 unseen
  compound → ≈0 unseen cell) and is therefore an in-distribution refinement, not a cold-cell
  generalisation mechanism.

- **Chemical generalisation is large; cell generalisation is at baseline level.** On unseen compound the
  model beats the best linear baseline by **+0.089 Pearson** and the best drug-mean by **+0.106**. On
  unseen cell it is beaten by predicting each drug's training-set mean (Meandrug: 0.4475 vs 0.440
  Pearson). The model's value is chemical, not cellular.

- **A 30× ablation artefact, identified and corrected.** A cell-conditional pathway-conductance module
  appeared to contribute ΔR² = +0.103 under ablation-to-1. Its learned conductance averages 0.60, so
  that ablation also rescales the pathway output by 1.67×. Ablating to the module's mean (preserving
  scale, removing only structure) gives **−0.003 / +0.006** — a 30× artefact. General lesson: ablate
  a learned multiplicative component to its mean, never to 1 or 0.

- **Atom→gene attention does not recover drug targets** (falsified, v5). Median target rank percentile
  0.560 over 149 gold drug–target pairs — worse than chance. The "targets don't move" confound was
  tested and rejected (corr = −0.045). Reported as a negative result of methodological interest:
  many papers assert attention ⇒ interpretability without such a test.

---

## Architecture

### v5 (trained and fully analysed)

```
Inputs
  X_base [B, 978]          — CCLE baseline expression (83 cell lines × 978 landmark genes)
  E      [B, 978, 3]       — Chromatin: ATAC / H3K27ac / H3K27me3 at ±10 kb TSS
  r      [B, 978]          — per-(cell,mark) reliability weights
  atoms  [B, M, 512]       — per-atom Uni-Mol tokens (ragged; 703,851 atoms total)
  u_feats[B, 2580]         — global drug features: ECFP4 (2048) + RDKit descriptors (20) + Uni-Mol CLS (512)
  dose, time               — scalar conditioning

Forward pass
  (1) EpiGate:     s = sigmoid(MLP_epi([ATAC, H3K27ac, H3K27me3, avail]))   [B, 978] ∈ (0,1]
  (2) Gene tokens: h = gene_emb + w_x(X_base · s) + w_e([E; avail])
                   h = DoseTimeFiLM(h, dose, time, lineage)
  (3) Drug tokens: D = [global_tok; atom_tok_1..M]                          [B, 1+M, d]
  (4) Base encoder:   L_base × (BiasedSelfAttn(gene↔gene, prior-biased) + FFN)
  (4b) Pathway cond:  c = PathwayConductance(E, X_base·s, r)               [B, 978]
  (5) Perturb encoder: L_perturb × (MultiHeadCrossAttn(gene←atom) + BiasedSelfAttn + FFN)
  (6) Head:    Ŷ = MLP(LayerNorm(h)) + gene_bias + MLP_epi_out(E)·r        [B, 978]

Priors
  A_copathway  [978, 978]  — Reactome co-pathway membership (additive, learned-strength λ bias)
  STRING_adj   [978, 978]  — STRING v12 PPI (same; λ = softplus(log_λ))
  Model can override; never a hard mask (v5 mistake — fixed in v6)
```

**Design decisions (interpretability over accuracy):**
- Predict **Y directly** (differential), not absolute post-perturbation expression
- **No VAE backbone** — avoids over-denoising, preserves attributability
- Epigenetics enters as both a **multiplicative gate** and a **signed additive head**
- `forward()` never sees Y — no target leakage

### v6 (built, unit-tested, TPU-launched — unmeasured)

v6 is a complete rebuild correcting two structural mistakes in v5:

**Mistake 1 — soft prior instead of hard mask.** v5 biased gene↔gene attention with `λ·log1p(A_copathway)`.
Measured: prior heads land on-support only **1.1× random** — signal routes around it. The co-pathway
matrix [978,978] is also a collapsed version of the Reactome membership matrix [360,978], discarding
named pathway identities.

**Fix:** A **PathwayBottleneck** with a hard connectivity mask — genes → 360 named Reactome nodes →
member genes only. Chromatin gates at the **pathway level**. The pathway readout `[B, 360, dp]` is a
genuine biological entity (a named Reactome pathway), interpretable by construction.

**Mistake 2 — early fusion of modalities.** v5 summed projected X_base and E into one gene token.
Consequence: learned pathway conductance correlated 0.74 with X_base and contributed ≈0 once scale was
controlled.

**Fix:** Separate encoders per modality (BaselineEncoder + ChromatinEncoder), late fusion — consistent
with MOLI / DeepCDR architectures for multi-omic integration.

```
v6 flow
  X_base ──► BaselineEncoder  ─┐
  E      ──► ChromatinEncoder  ├─► ModalityFusion (LATE) ─► gene tokens [B,G,d]
                                │                            + FiLM(dose,time,lineage)
                                │
                         PathwayBottleneck (hard Reactome mask)
                         chromatin gating @ pathway level
                         pathway_activations [B, 360, dp]  ◄── cell-level readout
                                │
  atoms  ──────────────────────►└─► perturbation encoder (atom→gene cross-attn)
                                          │
                                    per-gene head + signed chromatin term
                                          ▼
                                        Ŷ [B, 978]
```

> **Note on the v6 pathway readout:** `pathway_activations` is produced *before* the drug enters the
> forward pass. Substituting a completely different molecule changes it by exactly 0.0 (measured). It is a
> *cell × dose/time* quantity — "which named Reactome pathways are chromatin-permitted in this cell at
> this exposure." v6 does not repair the falsified drug-target claim; it sidesteps it.

---

## Data

All sources are public. Nothing is redistributed in this repository.

| Branch | Content | Shape | Source |
|---|---|---|---|
| `level5 data/` | LINCS L1000 Level 5 MODZ z-scores (Phase 1 + Phase 2) — the prediction target | — | GSE92742 + GSE70138 |
| `baseline/` | CCLE baseline expression (83 cell lines × 978 landmark genes) | 83 × 978 | DepMap `OmicsExpressionTPMLogp1` |
| `epigenetics/` | ATAC-seq, H3K27ac, H3K27me3 narrowPeak at ±10 kb TSS | 83 × 978 × 3 | ENCODE + ChIP-Atlas |
| `network/` | Reactome co-pathway + STRING v12 PPI adjacency matrices | 978 × 978 each | Reactome / STRING |
| `drug/` | SMILES → RDKit descriptors + ECFP4 + Uni-Mol CLS + per-atom tokens | 21,220 compounds | RDKit, Uni-Mol v1 |
| `drug/dti_reference.tsv` | 19,174 drug×gene pairs (ChEMBL + STITCH v5, evidence-tiered) — **validation only, never in the loss** | 1,718 drugs / 729 genes | ChEMBL + STITCH |
| `Data Info/` | LINCS Phase I/II metadata (signature/instrument/gene/cell/perturbagen tables) | — | reference |

**Target Y:** 312,438 × 978, 0 NaN. After SMILES filtering (79 drugs with no SMILES dropped):
**310,114 usable signatures**.

**Signature quality:** ~75% of LINCS perturbations are biologically inert. Replicate pairs agree at
r = 0.127 overall but **r = 0.509 (Phase 1) / 0.619 (Phase 2)** for real responses (mean|Y| ≥ 1). The
same (cell, drug) measured in two independent productions (Phase 1 vs Phase 2) agrees at r = 0.561.
All evaluation is stratified to the reproducible subset.

**Chromatin coverage:** 45 of 83 cell lines have ≥ 1 chromatin mark. Cells with no coverage receive a
neutral gate (= 1). No imputation was performed. A substitute-fill sweep over 121 unresolved (cell,
mark) pairs gained only +4 slots — documented as a negative result; the missing data simply does not
exist in any public assay or database.

**Scaffold split:** Bemis-Murcko scaffold split with a Tanimoto leakage audit. Median max-Tanimoto of
each test compound to its nearest training neighbour is **0.655** (39.2% ≥ 0.70, 8.0% ≥ 0.85). Any
unseen-compound number is inflated by analogue leakage unless audited.

---

## Repository Layout

```
chromatin-conditioned-perturbation/
│
├── model/                          Main model code
│   ├── model.py                    LincsCrossAttn — deterministic cross-attention predictor
│   ├── modules.py                  EpiGate, DoseTimeFiLM, PathwayConductance,
│   │                               MultiHeadCrossAttention, BiasedSelfAttention, FeedForward
│   ├── train.py                    Training loop (GPU / DataParallel), AMP, diagnostics
│   ├── train_tpu.py                torch_xla 8-core TPU port (static shapes)
│   ├── config.py                   ModelConfig / DataConfig (every hyperparameter justified)
│   ├── data.py                     Dataset, gather-by-key (no per-row denormalization), collate
│   ├── losses.py                   weighted_huber, correlation_loss, R²/Pearson, naive baselines
│   ├── eval.py                     DTI recall@k — atom→gene attribution vs drug-target reference
│   ├── diagnose.py                 Fair epi ablation on reproducible sigs + variance partition
│   ├── analyze.py                  Drug×cell interaction: noise-shrinkage vs capacity?
│   ├── drug_ablation.py            Feature-block ablations (atom/ECFP4/descriptor/ChemBERTa)
│   ├── ablate_v5.py                Per-component ablation (lineage / pathway / epi)
│   ├── dual_metric.py              Differential metric vs absolute convention sweep
│   ├── finalize_analyses.py        Consolidated re-run (cached predictions) of corrected ablation
│   ├── baseline_linear.py          Ridge + naive mean baselines (protocol-matched to v5)
│   ├── moa.py                      Learned prior strength λ + on-support attention flow
│   ├── pathway_maps.py             Per-(cell,gene) pathway-conductance maps
│   ├── case_study.py               Per-drug target-localisation case studies
│   ├── assemble_bundle.py          Pack all inputs into Kaggle training bundle
│   ├── MODEL_MATH.md               v5 equations, each mapped to its implementing module
│   ├── ARCHITECTURE_LESSONS.md     Why v5's interpretability failed; P-NET/DCell/MOLI lessons
│   ├── LITERATURE_PRACTICE.md      Methodology audit vs the field — 6 ranked gaps with fixes
│   ├── HANDOFF.md                  State of the project, what is true, what is retracted
│   ├── tests/
│   │   └── test_model.py           45 code-vs-math unit checks (must pass before training)
│   ├── results/
│   │   ├── CLAIMS.md               ~60 claims: evidence, A/B/C/✗ strength, retractions kept
│   │   ├── RESULTS.md              Detailed measurement log (17+ numbered experiments)
│   │   ├── v5_ckpt.pt              Trained v5 checkpoint
│   │   ├── v5_metrics.json         v5 evaluation metrics
│   │   ├── finalized_analyses.json Corrected ablation + metric sweep results
│   │   └── figures/                Figure outputs
│   └── v6/                         v6 rebuild (PathwayBottleneck + late fusion)
│       ├── model_v6.py             v6 architecture
│       ├── modules_v6.py           PathwayBottleneck, separate modality encoders
│       ├── config_v6.py            v6 hyperparameters
│       ├── train_v6_tpu.py         v6 TPU trainer
│       ├── eval_v6.py              v6 evaluation harness (29/29 checks pass)
│       ├── probe_pathways_v6.py    Pathway readout vs measured response, with nulls
│       ├── test_v6.py              16/16 unit checks + ablation positive/negative controls
│       ├── ARCHITECTURE.md         v6: every design decision with its evidence
│       └── TPU_NOTES.md            12 TPU/XLA failure modes and fixes
│
├── baseline/                       CCLE baseline expression branch
│   ├── scripts/                    Ordered pipeline scripts (step1..step2 + 12 validation checks)
│   └── outputs/                    X_base_lincs.npy, lincs_cell_index.json, ccle_resolution.tsv
│
├── epigenetics/                    Chromatin data branch
│   ├── chip_atlas/                 ChIP-Atlas download + cell-line matching
│   ├── scripts/                    step01..step08 audit, CVCL-verify, coverage classification
│   └── outputs/                    coverage_report_final.tsv
│
├── network/                        Pathway/PPI prior branch
│   ├── scripts/                    Reactome membership + STRING PPI pipeline
│   └── outputs/                    A_copathway.npy, STRING_adj_978.npy
│
├── drug/                           Drug featurisation branch
│   ├── scripts/                    SMILES extraction, RDKit descriptors, Uni-Mol integration,
│   │                               DTI reference build, Bemis-Murcko scaffold split
│   └── outputs/                    drug_descriptors.npy, drug_fingerprints.npy,
│                                   drug_unimol.npy, drug_atom_reprs.npy + offsets,
│                                   dti_reference.tsv, scaffold_split.json
│
├── phase2_assembly/                Final assembly: Y target, signature table, epigenetic tensor
│   ├── scripts/                    step6 (Y extraction), step10..step15 (epigenetic tensor)
│   └── outputs/                    Y_target_level5_978.npy, E_final.npy, E_final_mask.npy,
│                                   E_reliability.tsv, signatures_usable.tsv
│
├── MANUSCRIPT.md                   Draft paper (sections blocked on unrun measurements marked 🔲)
├── METHODOLOGY.md                  Script-by-script workflow, environments, full reproduction order
├── design_notes.md                 Architecture design directions and related-work analysis
└── Data Info/                      LINCS Phase I/II reference metadata
```

---

## Compute

Local machine has **no GPU.** All accelerated work ran on Kaggle.

| Kernel | Accelerator | Purpose |
|---|---|---|
| `lincs-drug-embeddings-unimol-molberta` | GPU T4×2 | Uni-Mol + ChemBERTa embeddings |
| `lincs-atom-tokens` | GPU T4×2 | Per-atom Uni-Mol representations (703,851 × 512) |
| `lincs-train`, `-v4`, `-v5` | GPU T4×2 | v5 training runs |
| `lincs-tpu-smoke` | TPU v3-8 | torch_xla port validation |
| `apexblue/lincs-v6-tpu` | TPU v3-8 | v6 training (10 epochs, 7.5 h budget) |
| `lincs-model-tests`, `-diagnose`, `-eval-dti`, `-analyze`, `-moa`, `-drug-ablation` | CPU | Tests + all inference-only analyses |

**Environments (three, deliberately):**
| Env | Python | Key packages | Used for |
|---|---|---|---|
| system | 3.14.3 | numpy 2.4.3, torch 2.11.0+cpu | model code, unit tests, CPU analyses |
| `drug/.venv-drug` | 3.12.13 | rdkit 2026.03.4, numpy | SMILES → descriptors/fingerprints, scaffold splits |
| `phase2_assembly/.venv-epi` | 3.12.13 | pybigtools | bigWig/peak epigenetics extraction |

rdkit and pybigtools have no wheels for Python 3.14 on Windows; splitting environments was cheaper than
downgrading everything.

---

## Reproduction Order

```
1.  baseline/scripts/step1,2                    → X_base, cell index          (system env)
2.  phase2_assembly/step6                       → Y, signatures_usable         (system env)
3.  epigenetics/step01..08 + phase2_assembly/step10..15 → E                   (.venv-epi)
4.  drug/step1,2 (+ Kaggle GPU), step3, step8  → drug features, atom tokens   (.venv-drug)
5.  drug/step4..7                               → dti_reference.tsv            (.venv-drug)
6.  drug/build_scaffold_split.py                → scaffold_split.json          (.venv-drug)
7.  model/assemble_bundle.py                    → Kaggle training bundle
8.  model/tests/test_model.py                   → 45/45 must pass BEFORE training
9.  model/train.py (GPU) or train_tpu.py (TPU) → v5_ckpt.pt
10. model/{diagnose,eval,analyze,ablate_v5,moa,pathway_maps,dual_metric}.py → results/
```

**Step 8 is not optional.** It catches code-vs-math drift before any accelerator quota is spent.
45 unit checks gate every training run, each mapped to an equation in `model/MODEL_MATH.md`.

---

## Four Methodological Rules (each learned by getting it wrong)

1. **Never evaluate on all signatures.** ~75% of LINCS perturbations are biologically inert. Diluting
   with inert signatures once *inverted the sign* of the epigenetics effect (−0.011 → +0.073). Always
   stratify to mean|Y| ≥ 1 (replicate r ≈ 0.5–0.62).

2. **Ablate a learned multiplicative component to its MEAN, never to 1 or 0.** Ablating to 1 also
   destroys the learned scale. The pathway conductance appeared to contribute ΔR² +0.103 — ablating to
   mean gave +0.006. A 30× artefact, caught and corrected.

3. **A noisy reference masks real signal.** Atom→gene attribution looked null on all 19,174 drug-target
   edges (98% STITCH co-occurrence). The effect only appeared on ChEMBL-curated subsets. Always
   stratify the reference by evidence quality *and* check the per-item distribution, not just top-k.

4. **Compute the noise ceiling before chasing a gap.** Replicate reliability bounds what is predictable.
   The model's cell-specific interaction expression (26.5%) is near-optimal under noise, not a capacity
   problem.

---

## Related Work

- **XPert** (Nat. Mach. Intell. Jan 2026, `s42256-025-01165-w.pdf` in repo): dual-branch transformer on
  L1000; UniMol + drug-MoA knowledge graph (DTI/PPI); nonlinear dose/time tokens; interpretable
  atom-level attention. Primary reference architecture.

- **StateXDiff** (preprint May 2026, `statexdiff.pdf` in repo): conditional latent diffusion; multimodal
  cell state (RNA + pseudo-protein); bidirectional cross-attention; reliability weighting. Borrow the
  weighting principles; avoid the diffusion backbone (hurts attributability).

- **Ahlmann-Eltze et al. (Nat Methods 2025):** the linear baseline that beats most deep models on
  unseen-cell perturbation prediction. We reproduce their finding inside our own project (claim 1.8):
  Meandrug beats v5 on unseen-cell on all three metrics.

- **P-NET (Nature 2021) / DCell / DrugCell (Nat Methods):** hard biological connectivity masks, where
  each layer is a named biological entity. The architectural inspiration for v6's PathwayBottleneck.

- **MOLI / DeepCDR:** late modality integration for multi-omic drug response. The inspiration for v6's
  separate per-modality encoders.

---

## Status

| Component | Status |
|---|---|
| LINCS Level 5 target matrix (Y) | ✅ Done — 310,114 usable signatures |
| CCLE baseline expression (X_base) | ✅ Done — 83 × 978 |
| Chromatin tensor (E) | ✅ Done — 83 × 978 × 3, 45/83 cells covered |
| Network priors (Reactome + STRING) | ✅ Done — 978 × 978 |
| Drug featurisation | ✅ Done — 21,220 compounds; atom tokens (703,851 × 512) |
| DTI validation reference | ✅ Done — 19,174 edges, evidence-tiered |
| Scaffold split + leakage audit | ✅ Done — Bemis-Murcko, median Tanimoto 0.655 |
| v5 training | ✅ Done — checkpoint in `model/results/v5_ckpt.pt` |
| v5 analysis (ablation, attribution, metric sweep) | ✅ Done — 17+ experiments, ~60 claims |
| v6 rebuild (PathwayBottleneck + late fusion) | ✅ Built + unit-tested (16/16) |
| v6 training (Kaggle TPU v3-8) | ⏳ Launched — checkpoint pending |
| v6 evaluation | ⏳ Harness written and tested (29/29 checks); awaiting checkpoint |
| Ridge linear baseline | ✅ Done — baseline_linear_fold0.json, beats v5 on unseen-cell |
| k-fold cross-validation | 🔲 Not yet — all current numbers are fold 0 only |
| Protocol-matched SOTA comparison | 🔲 Blocked — needs LINCS Level 3 |

---

## Honest Limitations

- All numbers are single fold (fold 0). k-fold required before treating them as robust.
- No protocol-matched comparison to published SOTA. The comparison paper uses Level 3
  quantile-normalised profiles + paired unperturbed controls; we have Level 5 only.
- Chromatin covers 45/83 cells; no imputation; the benefit does not transfer to unseen cells.
- Atom→gene attribution does not recover drug targets per-drug (falsified, v5). Population-level
  enrichment is non-random but not useful as a per-drug target predictor.
- The novelty of atom→gene attention relative to XPert is unresolved.
- Chromatin conditioning of **response-profile** prediction (not drug sensitivity) is the surviving
  novelty claim, from a targeted rather than systematic literature search.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

If you use this codebase, please cite the repository and reference the companion documents:
- `MANUSCRIPT.md` — the draft paper with all numbered claims
- `METHODOLOGY.md` — the full reproduction workflow
- `model/results/CLAIMS.md` — the evidence ledger including retractions
