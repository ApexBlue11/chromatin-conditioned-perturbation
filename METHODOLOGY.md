# METHODOLOGY — script-by-script workflow

Which script produced what, in what order, with which environment. Written so a third party can rerun the
pipeline end to end. Companion documents:
- `model/results/CLAIMS.md` — every claim with evidence, strength (A/B/C/✗) and falsifiers
- `model/results/RESULTS.md` — the detailed measurement log (17+ numbered experiments)
- `model/MODEL_MATH.md` — model equations, each mapped to the module that implements it
- per-branch `report.md` — the working notes for that data branch

---

## 0. Environments (three, deliberately)

| Env | Python | Key packages | Used for |
|---|---|---|---|
| system | 3.14.3 | numpy 2.4.3, torch 2.11.0+cpu | model code, unit tests, all CPU analyses |
| `drug/.venv-drug` | 3.12.13 | rdkit 2026.03.4, numpy | SMILES → descriptors/fingerprints, scaffold splits |
| `phase2_assembly/.venv-epi` | 3.12.13 | pybigtools | bigWig/peak epigenetics extraction |

Why three: rdkit and pybigtools have no wheels for Python 3.14 on Windows; torch does. Splitting the envs
was cheaper than downgrading everything. **Accelerated work (training, embeddings) ran on Kaggle** — see §7.

---

## 1. Target: LINCS L1000 response matrix `Y`

**Input:** LINCS Phase 1 (GSE92742) + Phase 2 (GSE70138) Level 5 (MODZ replicate-consensus z-scores).

| Script | Purpose | Output |
|---|---|---|
| `baseline/scripts/step1_analyze_phases.py` | inventory both phases, cell/drug overlap | phase report |
| `phase2_assembly/scripts/step6_extract_level5_target.py` | extract the 978 landmark genes for all usable signatures, both phases | `Y_target_level5_978.npy` (312,438 × 978, 0 NaN) + `signatures_usable.tsv` |

**Filtering:** signatures are kept only if the cell resolves to the 83-cell index **and** the drug resolves
to a SMILES-bearing `pert_id` (79 no-SMILES drugs dropped, which also removes the 2 proprietary compounds)
⇒ **310,114 usable**. Verified: `Y` ↔ GCTX agreement `max|diff| = 0.00e+00, corr = 1.000000` on 8 rows
spanning both phases including the P1/P2 boundary.

**`sig_strength.npy`** = mean|Y| per signature. Drives (a) reliability weighting in training and (b) the
"reproducible" evaluation stratum (mean|Y| ≥ 1). **This is load-bearing** — see §8.

## 2. Baseline expression `X_base` (cell context)

| Script | Purpose |
|---|---|
| `baseline/scripts/step2_load_ccle.py` | load CCLE `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` |
| `baseline/scripts/checks/*.py` | 12 validation scripts (ID resolution, DMSO counts, phase comparison, ambiguity inspection) |

**Output:** `X_base_lincs.npy` (83 × 978) + `ccle_full_background.npy` (1719-cell background) +
`lincs_cell_index.json` (the canonical cell order used by every other branch) +
`ccle_resolution_lincs.tsv` (LINCS cell_id → DepMap `ACH-` id, with the match rule recorded per cell).

## 3. Epigenetics `E` (chromatin state)

Three near-orthogonal regulatory axes: **ATAC** (accessibility), **H3K27ac** (activation), **H3K27me3**
(Polycomb repression). One track cannot distinguish active / poised / stably-repressed / bivalent.

| Script | Purpose |
|---|---|
| `epigenetics/chip_atlas/download_chipatlas.py`, `chipatlas_match.py` | source peak files, match samples to cell lines |
| `epigenetics/scripts/step01..step08_*.py` | audit coverage, CVCL-verify identities, classify matches, resolve gaps, build the coverage table |
| `phase2_assembly/scripts/step10_extract_peak_tensor.py` | narrowPeak → per-gene signal at ±10 kb TSS |
| `step11_repression_revamp.py`, `step12_h3k27me3_coverage.py`, `step14_finalize_h3k27me3.py` | the H3K27me3 track (see note) |
| `step4_substitute_recover.py`, `step5_substitute_sweep.py`, `step7_apply_substitute_fills.py` | property-matched substitute assays |
| `step13_finalize_tensor.py`, `step15_rebuild_reliability.py` | final tensor + per-(cell,mark) reliability |

**Output:** `E_final.npy` (83 × 978 × 3), `E_final_mask.npy` (per-mark availability),
`E_reliability.tsv`. **45/83 cells have ≥1 mark**; cells with none get gate = 1 (neutral fallback) — we
deliberately did **not** impute.

**Substitute rule:** a missing track may be filled only by an assay measuring the *same property with the
same sign* (waterfall: direct → same-property substitute → tissue-type → gap). This keeps the right cell
line with a proxy readout. Applied over 121 unresolved (cell,mark) pairs it gained only **+4 slots** —
documented as a negative result: the missing data does not exist in any assay/DB.

**H3K27me3 note:** broadPeak does not exist for H3K27me3 in ENCODE or ChIP-Atlas (all narrowPeak), and
bigWig coverage was bandwidth-intractable. Final choice = narrowPeak at ±10 kb TSS. Sparse signal at
landmark genes is **expected biology** (landmark genes are mostly active, not Polycomb targets), not poor
quality; only raw peaks < 10 (failed ChIP) are down-weighted. Sanity: corr(ATAC, H3K27ac) = +0.37,
corr(H3K27ac, H3K27me3) = −0.185 — correct signs.

## 4. Network priors

**Output:** `A_copathway.npy`, `STRING_adj_978.npy` (both 978 × 978, non-negative). Used as an **additive,
learned-strength bias** on a subset of attention heads (`λ = softplus(log_lambda)`), never as a hard mask —
so the model can override the prior. See `network/report.md`.

## 5. Drug branch

| Script | Purpose | Output |
|---|---|---|
| `drug/scripts/step1_extract_drug_smiles.py` | pert_id → canonical SMILES | `drug_list.tsv` (21,220 drugs) |
| `step2_compute_descriptors.py` | RDKit: 20 physicochemical descriptors + 2048-bit ECFP4 | `drug_descriptors.npy`, `drug_fingerprints.npy` |
| *(Kaggle GPU)* | Uni-Mol v1 3D CLS + ChemBERTa-77M | `drug_unimol.npy` (512), `drug_chemberta.npy` (384) |
| `step3_integrate_kaggle_embeddings.py` | verify alignment, NN sanity checks | `drug_feature_index.json` |
| `step8_integrate_atom_tokens.py` | per-atom Uni-Mol tokens (`remove_hs=True`) | `drug_atom_reprs.npy` (703,851 × 512 ragged) + `_offsets` |
| `build_scaffold_split.py` | Bemis-Murcko scaffold split **+ Tanimoto leakage audit** | `splits/scaffold_split.json` |

**Atom tokens, not pooled vectors** — these are the substrate for atom→gene attention, so pooling would
destroy the attribution we want. ~+1 token/molecule is Uni-Mol's virtual global node (token 0; strip for
pure SAR).

**ChemBERTa was later DROPPED** (ablation: ΔR² = +0.001, dead weight). See §8.

### 5b. DTI validation reference (never in the loss)

| Script | Purpose |
|---|---|
| `step4_dti_foundations.py` | InChIKey resolution |
| `step5_dti_chembl.py` | ChEMBL curated mechanisms (parent-keyed) |
| `step6_dti_stitch.py` | STITCH v5 human via LINCS `pubchem_cid` |
| `step7_dti_merge.py` | merge + evidence tiers |

**Output:** `dti_reference.tsv` — 19,174 drug×gene pairs / 1,718 drugs / 729 genes, with an `evidence`
column: `stitch` (15,510), `stitch_high` (3,217), `chembl` (291), `chembl+stitch_high` (128),
`chembl+stitch` (28). **Evidence tiers matter enormously** — see §8.

## 6. Model

| File | Contents |
|---|---|
| `model/config.py` | `ModelConfig` / `DataConfig` — every hyperparameter, each with the measurement that justifies it |
| `model/modules.py` | `EpiGate`, `DoseTimeFiLM`, `PathwayConductance`, `MultiHeadCrossAttention`, `BiasedSelfAttention`, `FeedForward` |
| `model/model.py` | `LincsCrossAttn` — the full forward |
| `model/losses.py` | `weighted_huber`, `correlation_loss`, R²/Pearson metrics, naive baselines |
| `model/data.py` | dataset (gather-by-key, no per-row copies), `collate`, cold-cell × cold-compound splits |
| `model/train.py` | training loop (GPU/DataParallel), per-epoch interaction probe, diagnostics |
| `model/train_tpu.py` | torch_xla 8-core port (static shapes, on-device loss accumulation) |
| `model/tests/test_model.py` | **45 code-vs-math checks**, each mapped to an equation |
| `model/assemble_bundle.py` | pack all inputs into the Kaggle training bundle |

**Flow:** chromatin gate → gene tokens (+dose/time/lineage FiLM) → base encoder (gene↔gene, prior-biased)
→ perturbation encoder (atom→gene cross-attention + gene↔gene) → per-gene head + additive signed epi term.
Predicts **Y directly** (the differential); `X_base` is context, never a target. `forward()` never sees Y.

**Three deliberate design decisions** (interpretability over accuracy): predict Y directly (not
x_pert − x_base, which is cross-platform); **no VAE backbone** (avoids over-denoising and preserves
attributability); epigenetics enters as a gate **and** a signed additive head.

## 7. Compute

Local machine has **no GPU**; all accelerated work ran on Kaggle.

| Kernel | Accelerator | Purpose |
|---|---|---|
| `lincs-drug-embeddings-unimol-molberta` | GPU T4×2 | Uni-Mol + ChemBERTa embeddings |
| `lincs-atom-tokens` | GPU T4×2 | per-atom Uni-Mol representations |
| `lincs-train`, `-v4`, `-v5` | GPU T4×2 | training |
| `lincs-tpu-smoke` | TPU v3-8 | torch_xla port validation |
| `lincs-model-tests`, `-diagnose`, `-eval-dti`, `-analyze`, `-moa`, `-drug-ablation` | CPU (free) | tests + all inference-only analyses |

**Hard-won operational facts:** never request P100 (sm_60 — Kaggle's torch has no kernels for it; use
`--accelerator NvidiaTeslaT4`); private datasets mount at `/kaggle/input/datasets/<user>/<slug>/` so use a
recursive glob; **a cancelled kernel's `/kaggle/working` is discarded**, so never hard-cancel a long run;
caps are 5 concurrent CPU sessions and 2 GPU.

## 8. Evaluation & analysis scripts

| Script | Question |
|---|---|
| `model/diagnose.py` | fair epi ablation on reproducible signatures + truth-vs-model variance partition |
| `model/eval.py` | DTI recall@k for atom→gene attribution, split by evidence tier |
| `model/analyze.py` | is the drug×cell interaction under-expression noise-shrinkage or capacity? |
| `model/drug_ablation.py` | which drug-feature block carries signal |
| `model/ablate_v5.py` | per-component attribution (lineage / pathway / epi) |
| `model/moa.py` | learned prior strength λ + on-support attention flow |
| `model/pathway_maps.py` | per-(cell,gene) pathway-conductance maps |
| `model/dual_metric.py` | our differential metric vs the published absolute convention |
| `model/finalize_analyses.py` | consolidated re-run (cached predictions) of the corrected ablation + metric sweep |

### Four methodological rules these scripts enforce

1. **Never evaluate on all signatures.** ~75% of LINCS perturbations are biologically inert (replicate
   r ≈ 0.07–0.14). Always stratify to mean|Y| ≥ 1 (replicate r ≈ 0.5–0.62). Diluting with inert signatures
   once **inverted the sign** of the epigenetics effect (−0.011 → +0.073).
2. **Compute the noise ceiling before chasing a gap.** Replicate reliability bounds what is predictable.
3. **Ablate a learned multiplicative component to its MEAN, not to 1 or 0** — otherwise you destroy its
   learned scale and measure that instead of the mechanism. This error inflated a pathway result by 30×
   (+0.103 vs the true +0.006) before it was caught.
4. **A noisy reference masks real signal.** DTI attribution looked null on all 19,174 edges (98% STITCH
   co-occurrence) and only appeared on curated subsets. Always stratify the reference by evidence quality.

## 9. Reproduction order

```
1. baseline/scripts/step1,2         -> X_base, cell index          (system env)
2. phase2_assembly/step6            -> Y, signatures_usable        (system env)
3. epigenetics/step01..08 + phase2_assembly/step10..15 -> E        (.venv-epi)
4. drug/step1,2 (+ Kaggle GPU) ,3,8 -> drug features, atom tokens  (.venv-drug)
5. drug/step4..7                    -> dti_reference.tsv           (.venv-drug)
6. drug/build_scaffold_split.py     -> scaffold_split.json         (.venv-drug)
7. model/assemble_bundle.py         -> Kaggle training bundle
8. model/tests/test_model.py        -> 45/45 must pass BEFORE training
9. model/train.py (GPU) or train_tpu.py (TPU)
10. model/{diagnose,eval,analyze,ablate_v5,moa,pathway_maps,dual_metric}.py -> results/
```

Step 8 is not optional: it is what catches code-vs-math drift before any accelerator quota is spent.
