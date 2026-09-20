# V9 HANDOFF — read this first, read it adversarially

**Written 2026-08-18.** This supersedes `model/HANDOFF.md` and `model/V8_PLAN.md` for anything about
direction. Those remain accurate for v6/v7 history and the method rules.

> This project has repeatedly produced confident-but-wrong conclusions that were only caught by insisting on
> a proper control — **including twice in the session that produced this file** (§B). Treat every number
> below as a hypothesis with evidence attached, and check the evidence before building on it.

---

## A0. AMENDMENT, 2026-08-26 — four of this document's own claims did not survive measurement

This file was written before the v9 work ran. Its direction held; four specific claims in it did not, and
the pattern §B warns about repeated itself. Full detail and numbers: `model/results/RESULTS.md` §27–28,
`model/results/CLAIMS.md` §6d.

| this document says | measured |
|---|---|
| §D.1 "extract GSE70138 so coverage goes 44.8 % → ≥90 %" | The **join** was the problem, not the missing phase. P1 was already at 66.3 % with GSE92742 extracted, and P2's 4.1 % were FALSE matches to P1 wells. Under the same key logic, extracting GSE70138 projects to **~66 %** — the gate would have failed after the work was done. `sig_info.distil_id` is the exact well mapping; on it, coverage is **99.65 %**. |
| §C "we truncated Reactome to the 978 landmarks… that is why 231 landmarks sit in no pathway" | **Wrong diagnosis.** Nothing was truncated. `ReactomePathways.gmt` annotates only **11,963 genes**, so those landmarks are absent at ANY filter (verified at `min_size=1` with the umbrella exclusion off). The gate **231 → ~0 is unreachable from Reactome.** Adding GO:BP as a second NAMED source reaches 50, against a two-source floor of 45. The **STRING** half of the same claim IS right: 66 → 8. |
| §C "Splits are tissue-holdouts: `split_lung_1..5`…" | **Not holdouts.** All 15 restrict to one tissue and split it ~90/10. Mean over the 15: **100.0 %** of test rows use a cell line seen in training, **98.6 %** a compound seen in training, **89.4 %** the exact (cell, compound) PAIR. The task is mostly a new dose or time of something already in training. |
| §B "Level 5 is the better-denoised target" (and so v9's target move costs accuracy) | **A stratum artefact.** §25 compared an L3-defined top quartile (0.2429) with the L5-defined reproducible stratum (0.509–0.619). On the stratum this project actually evaluates on, the Level-3 delta self-agrees at **0.5283**. Level 5 may still lead at the very top; "migrating to Level 3 raises our noise penalty" is not supported. |

**One new hazard, of the same kind the handoff catalogues.** v9 trains on `delta = trt − ctl` and is handed
`ctl`, so the control's measurement noise enters target and input with opposite signs and a model can
improve its score by cancelling it. Measured with two independent half-plate DMSO medians: on unseen CELLS
essentially all of the matched control's apparent advantage is noise cancellation, and an independent plate
control is 0.08 **worse** than a per-cell mean. Every delta number v9 reports must ship with its
independent-control version.

**What this does not change.** The direction — rebuild the data substrate, keep the interpretability that
is ours, gate every step on a measurement — held up. The two measured wins the handoff identifies (deleting
a component; changing an input) are still the only two, and §28 adds a third lever that is bigger than
either: **the split itself is worth +0.17**, four times the seed band.

---

## A0b. AMENDMENT, 2026-08-30 — their code now RUNS here, and §C below is superseded where they conflict

§C was written by reading their source. Their model has now been **executed**, on their data, with their
weights. Where this section and §C disagree, this one wins. Full detail in `model/results/RESULTS.md`
§41–42; the operational facts a new session needs are here.

**Their release cannot be run as shipped.** `external/xpert/code/XPert/processed_data/` contains one
`gitkeep.txt`. The missing assets are on Zenodo `10.5281/zenodo.17182939` and are fetched by HTTP range
request, not whole-file download:
```
python model/v9/fetch_xpert_assets.py --list
python model/v9/fetch_xpert_assets.py --fetch l1000_mdmt_68830_subset.h5ad PPI_gene_vector_128d.npy \
                                              l1000_gene_info_978.csv all_drugs_idx2smi_8981.npy
python model/v9/fetch_xpert_unimol.py --idx_file <pert_idx.json> --out .../unimol_mdmt_1970.npz
```
CRC32-verified, resumable, ~1.9 GB total instead of ~6 GB. Everything lands in gitignored `external/`.

**Their MAIN benchmark is `l1000_mdmt_68830_subset.h5ad`**, not the 336,852-row file §C describes. 68,830
conditions, 40 cell lines, 1,977 compounds, carrying `split_1..5` (warm), `split_cold_cell_1..5` and
`split_cold_drug_1..5`. Bundled for our kernels by `model/v9/xpert_mdmt_extract.py` →
`external/xpert_split_bundle/xpert_mdmt_splits.npz` (568 MB), on Kaggle as `apexblue/xpert-mdmt-benchmark`.

**Four things that silently change the numbers**, all now handled in code and covered by
`model/v9/test_xpert_compare.py` (26 checks):
1. their released checkpoint was trained with **`--include_cell_idx True`**, a NON-default flag — load
   `strict=True` or you build a different model;
2. their attention branches `if output_attention: <dense, masked> else: <flash, unmasked>`, so **the flash
   path is the default** and the two are not interchangeable. `model/v9/_shims/flash_attn/` is an exact
   dense stand-in, verified to 5e-7 and verified to *differ* from the masked branch;
3. **their metric is the MEAN of per-row Pearson**, ours has always been the median (~0.02–0.05 apart);
4. **18.9 % of their rows pool 2–8 doses** into one condition; their model only sees the dose *bin*.

**🔴 Their released checkpoint was trained on `split_2`.** Scored on all five warm folds it gives 0.6939
there and 0.7384–0.7450 on the other four; a ridge refitted per fold scores 0.6054/0.6062, so the folds are
equally hard and the gap is contamination. **`split_2` is the only fold on which their checkpoint can be
honestly scored, and therefore the only fold on which a head-to-head is fair.** Because the five folds
partition the corpus, every other fold's test set lies inside `split_2`'s training data.

**Their honest numbers on their own benchmark** (`split_2` test, n=13,766, their metric):

| | Pearson (abs) | Pearson_deg |
|---|---|---|
| copy the control | 0.9592 | 0 |
| ridge | 0.9747 | 0.6062 |
| XPert released checkpoint | 0.9797 | **0.6932** |

Their published absolute ~0.98 is mostly the control. Their fig4 HDACi numbers (0.9804 / 0.8440) come from
**no released checkpoint** — the three shipped mdmt checkpoints give 0.6444 / 0.7610 / 0.7297 on those rows,
and the released predictions are as accurate off their benchmark as on it. Do not cite 0.8440 as held-out.

**CLAIMS 7.3 is falsified.** Atom→gene attention is not ours: their `CrossAttention.forward(cell, drug)`
queries from the 978 gene tokens and keys/values from the atom tokens, and they ablate and visualise it.

**Local GPU.** `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` has torch+cu126 for the RTX 3050 (4 GB,
use `--batch 8`). ~12× faster than CPU for their model, and CPU/GPU agree to 1.1e-05 over 13.5 M
predictions. The main interpreter stays CPU-only so nothing else is disturbed.

---

## A. State in one paragraph

We predict drug-induced transcriptional response on LINCS L1000. Six architectures (v3→v7) produced
**no accuracy difference distinguishable from seed noise**. The one measured improvement in the entire
project came from **deleting** a component (`v7 --no_aux`), and the *second* came from **changing an input**
(Level-3 plate-matched control, +0.031…+0.044 with no seed variance at all). The architecture work has not
paid; the data-and-inputs work has. **v9 rebuilds the data substrate to match how published LINCS models
actually work — verified by reading their released code and data, not by inference — and only then revisits
architecture.**

---

## B. What is TRUE, and what is RETRACTED

### TRUE (survived a proper control)

| # | Finding |
|---|---|
| **Seed variance dominates** | 3 seeds of one identical config: sd **0.0052 / 0.0150 / 0.0232** (unseen cell / compound / both) ⇒ 2-sd ≈ **±0.046**. v5, v6, v7 are **statistically indistinguishable** on 2 of 3 splits. **Report mean ± range over ≥3 seeds or report no difference.** Ablations are exempt (within-run, identical signatures). |
| **Best config to date** | `v7 --no_aux`, 3 seeds: **0.4549 / 0.4985 / 0.4825** (cell/compound/both). Beats v5 and v6 on compound+both with the *whole range* above both. |
| **Plate-matched control beats CCLE** | Ridge A/B, identical signatures, only the baseline vector changes: **+0.0314 / +0.0435 / +0.0372**. "Both" ≈ "matched" alone ⇒ **matched control SUBSUMES CCLE; drop CCLE.** Closed-form fit ⇒ **no seed noise.** |
| **Drug features are the model** | Ablate-to-mean: drug global **+0.25…+0.30**; baseline expression +0.026…+0.045; atom tokens +0.002…+0.024; lineage +0.002…+0.022. |
| **Pathway layer contributes nothing** | +0.0003 / −0.0002 / −0.0002 on all three splits, with `|dY|max` 0.69 ≫ 0 so it is a **true null**, not a dead ablation. 0/360 nodes dead; activation evenly spread. |
| **Chromatin ≈ 0, and it tracks cell familiarity** | −0.0001 (unseen cell) / **+0.0061** (unseen compound, cells seen) / −0.0001 (both). A **16-dim lineage one-hot beats it** (+0.0215). Fusion gate ended at 0.4798, *below* its 0.5 init. |
| **Epi-drugs are far easier but not via chromatin** | HDAC/DNMT drugs: 0.6477 vs 0.4477 (unseen compound). But chromatin ablation differs by only −0.0138 ⇒ they are predictable because their response is **strong and stereotyped across cells**, i.e. drug-determined. |
| **Meandrug ties v5 on unseen cells** | 0.4475 vs 0.440 — the drug-mean baseline beat v5 on all three metrics. `v7 no_aux` clears it in 2/3 seeds; first thing in the project that ever has. |
| **Absolute convention is inflated** | On XPert's own released predictions with their own metric code: absolute **0.9804**, delta **0.8440**, **"copy the control" 0.9200**. Their absolute beats *doing nothing* by only **+0.060**. |
| **XPert uses Level 3 for BOTH input and target** | **Verified empirically**: their `obsm['X_ctl']` vs our Level-3 plate DMSO median → **r = 0.968**, identical [0,15] ranges. Their loss trains on absolute Level-3 expression *and* the delta. |

### RETRACTED — do not resurrect

| # | Retracted claim |
|---|---|
| ~~atom→gene attention localises to drug targets~~ | Median target rank percentile **0.560 over 149 gold pairs — worse than chance**. "Target doesn't move" confound tested and **rejected** (corr −0.045). Top-k enrichment ≠ localisation. |
| ~~v5 beats all naive baselines~~ | That was MSE on *all* cold-cell signatures. On the reproducible stratum with our reported metrics, **Meandrug beats v5**. |
| ~~pathway conductance is the biggest contributor (+0.103)~~ | Scale artefact from ablating to 1. True effect −0.003/+0.006. **Always ablate to the MEAN.** |
| ~~our Level-5 target is the noisy one; migrate to Level 3 to fix it~~ | **Measured and false.** Level-3 replicate-averaged split-half **0.144 / 0.243 top-quartile** vs Level-5 MODZ **0.127 all / 0.509–0.619 reproducible**. **Level 5 is the better-denoised target.** MODZ weighting of Level-3 deltas is *worse* than a flat mean (−0.0072). |
| ~~v7 is clearly worse than v6, outside noise~~ | Not supported once 3 seeds existed. |
| ~~seed spread ≈0.004~~ | Came from the n=3000 training proxy; understated the truth by up to 7×. |

**The pattern to internalise:** four of these six retractions were *my own conclusions from one turn earlier*,
overturned by a measurement I ran the next turn. Measure before concluding.

---

## C. How published LINCS models actually work — READ FROM THEIR CODE

Source: XPert's Zenodo release, downloaded to `external/xpert/` (gitignored, 14 GB).
`code/XPert/` = full source; `l1000_mdmt_full_336852.h5ad` = their dataset; `saved_model.zip` = weights.

### Data (verified empirically, not assumed)
- **Input AND target are both Level 3.** `X` = replicate-collapsed Level-3 log-expression [0, 15];
  `obsm['X_ctl']` = the plate-matched DMSO control (**r = 0.968 with our own Level-3 reconstruction**).
- 336,852 conditions × 978 genes, `n_replicates` 1–6+, `cell_pert_dose_time` unique per row.
- **Splits are tissue-holdouts**: `split_lung_1..5`, `split_breast_1..5`,
  `split_haematopoietic_and_lymphoid_tissue_1..5` — plus a `cold_dose&time` split we have never tested.

### Loss (`train_xpert.py`, 4 weighted terms)
```
loss1 = MSE(trt_output, trt_raw)                  # ABSOLUTE Level-3 expression
loss2 = MSE(ctl_output, ctl_raw)                  # reconstruct the control
loss3 = MSE(deg_output, trt_raw - ctl_raw)        # the DELTA
loss4 = PCC_loss(deg_output, trt_raw - ctl_raw)   # correlation on the delta
```
They report **both** conventions: `metrics['Pearson']` (absolute) and `metrics['Pearson_deg']` (delta).

### Architecture (`models/model_XPert.py`, `configs/config_l1000.yaml`)
| element | what they do | what we do today |
|---|---|---|
| **expression input** | **binned into 128 levels and EMBEDDED as tokens** (`n_bins: 128`, `exp_vocab_size`) | raw continuous scalar through an MLP |
| **gene identity** | **pretrained PPI gene vector, 128-d** (`PPI_gene_vector_128d.npy`) + expression embedding | learned-from-scratch `gene_emb` |
| **PPI graph** | **901,260 weighted edges over 19,392 genes** | 12,665 edges **restricted to the 978 landmarks** |
| **drug–target** | 12,890 edges, **in the model** | validation only |
| **drug–drug similarity** | 287,834 weighted edges | none |
| **knowledge use** | heterogeneous graph **pretrained by link prediction** → drug embeddings + **drug-specific gene embeddings injected additively between layers** | one graph-conv step on a raw adjacency |
| **branches** | treated `CA+SA+SA+CA`, control `SA+SA+SA+SA` (a **separate control encoder**) | single stream |
| **drug features** | UniMol molecule + atom + dose + time embeddings | same (UniMol CLS + atoms + ECFP4 + descriptors) |
| width / heads | 256 / 8, ~~top-k sparse attention (128 cell, 32 drug)~~ 🔴 **FALSE, see RESULTS §47.1 — `topk` is never stored, `sparse_flag` is never read, and their config sets it False. Their attention is DENSE.** | 256 / 8, dense — **identical to theirs** |

**The specific defect that is ours:** we truncated STRING and Reactome **to the 978 landmarks**, deleting
every path that routes *through* a non-landmark gene. That is why **231 of our landmarks sit in no Reactome
pathway** and **66 have no STRING edge at all**. Both sources are full-proteome; we discarded most of them.

### Epigenetics — there is NO template
**No SOTA LINCS model uses chromatin.** So "do it properly like the others" has no referent — this is the
genuinely novel part and it must be made to *work*, not made to conform. The natural home given the above
architecture: chromatin becomes a **per-gene embedding summed into the gene representation, exactly as the
PPI gene vector is** — not a parallel encoder the model can (and did) ignore.

### Still to verify (do NOT cite until read)
**XPert is no longer in this list — its code has been run, see §A0b.** PRnet, chemCPA, TranSiGen, the
Bioinformatics-2026 latent-diffusion model: none ships predictions or code in XPert's release, and running
each faithfully means a separate legacy stack (TranSiGen pins python 3.6 / torch 1.5). Their reported numbers are
**not** comparable to ours without matching data level, convention, and split.

---

## D. The v9 design

**Principle: change the data substrate to match the field, keep the interpretability that is ours, and
gate every step on a measurement.**

1. **Data — Level 3 both sides, matched control as input**
   - Target: **absolute Level-3 expression** + **delta** (multi-task, as XPert does). Keep the Level-5 MODZ
     z-score as a *secondary* reported metric so all our history stays comparable.
   - Input: **plate-matched DMSO control**. **Drop CCLE `X_base`** — measured to be subsumed.
   - Extract **GSE70138** too (13.5 GB, present) — currently only GSE92742 is done, giving 44.8 % coverage.
2. **Expression encoding** — bin into 128 levels and embed, rather than an MLP on a raw scalar.
3. **Priors, full-proteome**
   - Rebuild STRING **over ~19k genes**, pretrain gene vectors, read off the 978 rows.
   - Rebuild Reactome membership without landmark truncation, so the 231 orphans become reachable.
   - Add DTI + drug-drug edges → heterogeneous graph, pretrained by link prediction.
4. **Chromatin as a per-gene embedding** summed into the gene representation alongside the PPI vector.
   Keep the **signed additive chromatin head** — the one chromatin mechanism that ever survived a test.
5. **Interpretability, non-negotiable**
   - Named Reactome nodes stay, with the row→name mapping **verified at load** (`pathway_info.tsv`).
   - No purely-latent bottleneck; every readout terminates in named units.
   - **Every readout ships with its null.** On pathway-level target alignment **0.5 is NOT chance** — an
     untrained model scores **0.218** against a permutation null of **0.229**.
6. **Separate control encoder** (their `ctl_structure`), since the control is now a real input.

### Gates (do not skip; each has a number)
| gate | requirement |
|---|---|
| GSE70138 extraction | coverage ≥ 90 % of our signatures |
| full-proteome graph | Reactome orphan landmarks 231 → ~0; STRING isolated 66 → ~0 |
| binned-expression encoding | ridge/simple A/B before committing to a full train |
| any architecture claim | **≥3 seeds, mean ± range** |
| any SOTA comparison | same data level + convention + split, else report non-comparability |

---

## E. Data inventory (exact paths, all present locally)

| what | path | size |
|---|---|---|
| **Level 3 GSE92742** | `level3 files/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx` | 65.1 GB |
| **Level 3 GSE70138** | `level3 files/GSE70138_Level3.gctx` | 13.5 GB — **not yet extracted** |
| inst_info (92742) | `Data Info/GSE92742_Broad_LINCS_inst_info.txt/GSE92742_Broad_LINCS_inst_info.txt` | 148 MB |
| inst_info (70138) | `Data Info/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt/GSE70138_Broad_LINCS_inst_info.txt` | 46 MB |
| gene_info (landmark flags) | `Data Info/GSE92742_Broad_LINCS_gene_info.txt/GSE92742_Broad_LINCS_gene_info.txt` | — |
| Level 5 (current target) | `phase2_assembly/outputs/Y_target_level5_978.npy` + `signatures_usable.tsv` | 310,114 sigs |
| **Level-3 extraction (done)** | `phase2_assembly/outputs/level3/` — `Y_delta_l3.npy`, `X_ctl_l3.npy`, `conditions_l3.tsv`, `genes_l3.txt`, `join_l5row_to_l3row.npy`, `wells_cache.dat` | 189,482 conditions |
| chromatin | `phase2_assembly/outputs/E_final.npy`, `E_final_mask.npy`, `E_reliability.tsv` | 83×978×3 |
| Reactome | `network/outputs/M_reactome.npy` (360) + `M_reactome_ms5.npy` (765) + matching `pathway_info*.tsv` | — |
| STRING | `network/outputs/STRING_adj_978.npy` (12,665 edges — **truncated, rebuild**) | — |
| raw STRING/Reactome | `network/data/` (`9606.protein.links.v12.0.txt`, `ReactomePathways.gmt`) | full-proteome sources |
| drug features | `drug/outputs/` — UniMol CLS + atom tokens, ECFP4, descriptors, scaffold split | 21,220 drugs |
| DTI reference | `drug/outputs/dti/dti_reference.tsv` (+ `epi_drug_pert_ids.json`, 30 drugs) | validation only |
| **XPert release** | `external/xpert/` — code, weights, h5ad | 14 GB, **gitignored** |

---

## F. Method rules (each learned by getting it wrong)

1. **Never evaluate on all signatures** — ~75 % of LINCS is inert; dilution once *inverted* the sign of a real effect. Stratify to mean|Y| ≥ 1, **and report all strata** so the headline is auditable.
2. **Ablate to the MEAN**, never 0 or 1 — that is the 30× artefact. Always report `|dY|max` so a true null is distinguishable from an ablation that never fired.
3. **≥3 seeds or no difference.** Seed sd reaches 0.0232.
4. **Compute the noise ceiling before chasing a gap.**
5. **Verify a measurement tests what it claims** — most errors here were valid computations of the wrong quantity.
6. **A readout's chance level must be measured, not assumed** (0.218 vs 0.229).
7. **Within-run ablation is trustworthy; between-run comparison is not.**
8. **Before reverse-engineering any published number, DOWNLOAD THE SUPPLEMENTARY.** Their L1000_mdmt
   numbers were in Supplementary Table R8 the whole time, reachable unauthenticated in 30 s. This omission
   cost more than the NaN quantiser [RESULTS §46.1]. Springer pattern:
   `https://static-content.springer.com/esm/art%3A10.1038%2F<doi-suffix>/MediaObjects/<doi_underscored>_MOESM<n>_ESM.pdf`
9. **Our-number-vs-their-number is NOT a model comparison**, however well the split, metric and convention
   match. §39 got this wrong, §40 reversed it, and §46.4 is currently in the same form. Only identical rows
   scored by both models count.

## G. Compute
- Kaggle **T4 x2** via `machine_shape: "NvidiaTeslaT4"` + `"enable_gpu": true`; non-TPU docker image. Copy metadata from `apexblue/lincs-train-v5`.
- **Never P100** (no sm_60 kernels). **2 concurrent GPU sessions max.** CLI cannot show quota — check the web UI.
- **TPU is abandoned** — v5e multi-process init fails (`Expected 8 worker addresses, got 1`), two queue cycles for zero training, and it was slower per core than a T4 anyway.
- `train_v7_gpu.py` has the guard set worth copying: no-CPU-fallback, P100 probe, both-GPUs-active assertion, and `check_inputs()` which refuses to train when a missing file has silently degraded the data (this caught a rebuilt Kaggle dataset that had lost the compound holdout **and** reliability weighting).
- torch 2.11 is installed locally — run all CPU analysis locally; Kaggle CPU kernels are free.

## H. Repo
GitHub `ApexBlue11/chromatin-conditioned-perturbation` (private, MIT). `external/` is gitignored — a
`git add -A` once swept 14 GB into two commits; they were rewound before pushing.
Key docs: `model/results/CLAIMS.md` (~80 claims with strength + falsifiers, retractions kept deliberately),
`model/results/RESULTS.md` (26 numbered experiments), `model/LITERATURE_PRACTICE.md`,
`model/v6/ARCHITECTURE.md`, `model/v6/TPU_NOTES.md`.
