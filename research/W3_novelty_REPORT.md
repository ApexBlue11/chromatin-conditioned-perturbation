# Adversarial Novelty Evaluation Report

## VERDICT
**CLAIM WEAKENED**

The claim is **not killed**, because no single prior publication meets all four criteria simultaneously. However, the claim is **substantially weakened and vulnerable to hostile review** because prior work already flanks it from two sides:
1. **From the epigenetic-drug response flank:** [Agrawal et al. (2023/2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12103705/) demonstrated that measured pre-treatment chromatin state (locus-specific H3K27ac in 21 genic bins) predicts whether genes are up- or down-regulated following small-molecule HDAC inhibitor treatment (Largazole, Entinostat). Your claim survives against Agrawal et al. *only* on the narrow distinction between binary direction classification of extreme gene sets versus continuous response profile regression across the whole transcriptome.
2. **From the generative perturbation modeling flank:** Contemporary multi-modal frameworks such as [BaiZe (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.07.15.738608v1) and [MultiFlow (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.08.20.746112v1) formulate perturbation response prediction conditioned on baseline ATAC-seq chromatin accessibility. While their implementations benchmark ATAC conditioning exclusively on genetic perturbations (CRISPR) and developmental transitions (leaving chemical drug response prediction RNA-conditioned), their architectures explicitly describe integrating chromatin context into perturbation response decoders.

If submitted without qualification, a hostile reviewer will cite Agrawal et al. to dispute novelty. The claim must be narrowed to survive.

---

## Killers
*Papers meeting ALL of criteria 1–4 (predictive model + continuous multi-gene expression response profile output + measured prior chromatin state input + small molecule / drug perturbation).*

None found.

---

## Partial
*Papers that demonstrate chromatin-conditioned perturbation prediction or epigenetic drug response modeling, but fail exactly one criterion.*

### 1. Agrawal et al. (2023 / 2025)
- **Citation:** Piyush Agrawal, Vishaka Gopalan, Monjura Afrin Rumi, and Sridhar Hannenhalli. "Predicting gene expression changes upon epigenomic drug treatment." *F1000Research* 2023, 12:1111 (version 3 published May 2025; preprint on bioRxiv July 2023). 
- **Identifiers:** PMID: [40417623](https://pubmed.ncbi.nlm.nih.gov/40417623/); PMCID: [PMC12103705](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12103705/); DOI: [10.12688/f1000research.140273.3](https://doi.org/10.12688/f1000research.140273.3); bioRxiv DOI: [10.1101/2023.07.21.550106](https://doi.org/10.1101/2023.07.21.550106).
- **URL:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12103705/
- **What it does:** Uses measured pre-treatment H3K27ac ChIP-seq signal distributed across 21 genic bins (10 promoter, 1 TSS, 10 gene body) to predict locus-specific transcriptional response to small-molecule HDAC inhibitors (Largazole at 8 doses in HCT116 cells; Entinostat at 1 µM in RH4 cells) using SVM, Random Forest, and Gradient Boosting. Demonstrates cross-cell-line and cross-drug generalizability (ROC AUC 0.71–0.89).
- **Verbatim quote showing input:**
  > *"We used the histone mark distribution in 21 genic bins as features to develop machine learning models to distinguish up versus downregulated genes after HDACi-treatment separately for both HCT116 and RH4 cell lines... For every gene, pre-treatment H3K27Ac read count in 21 regions relative to the gene (Methods) were used as features and three machine learning models – Support Vector Machine (SVM), Random Forest (RF), and Gradient Boosting (GB), were benchmarked based on five-fold cross-validation and accuracy was quantified as area under the ROC curve (AUC)."*
- **Criterion failed:** **Fails Criterion 2 (Output is a binary label, not a continuous multi-gene response profile).** The model predicts a discrete binary classification label (+1 for top 1000 upregulated vs. -1 for top 1000 downregulated genes), rather than predicting a continuous quantitative expression change profile ("which genes change, and by how much") across the unselected transcriptome.

### 2. BaiZe (Zeng et al., bioRxiv 2026)
- **Citation:** Qingwu Zeng, Wenxiang Cai, Renchong Tian, Qi Wang, Dongzhan Zhou, Minting Pan, Hai Yang, Zhe Liu, Guan Ning Lin, and Zhe Wang. "BaiZe: A Multi-View Dynamic Framework for Simulating and Interpreting Cellular Responses Across Perturbation Contexts." *bioRxiv* (July 20, 2026).
- **Identifiers:** DOI: [10.64898/2026.07.15.738608](https://doi.org/10.64898/2026.07.15.738608).
- **URL:** https://www.biorxiv.org/content/10.64898/2026.07.15.738608v1
- **What it does:** Generative conditional state-transition model with a diffusion decoder that predicts continuous transcriptional change vectors ($\Delta x = x_{\text{pert}} - x_{\text{ctrl}}$). The architecture conceptually allows conditioning on control RNA, chemical structures (Morgan fingerprints) + dose, genetic targets, and optional matched ATAC-seq embeddings.
- **Verbatim quote showing conditional capability:**
  > *"Here we present BaiZe, a multi-view conditional state-transition framework that predicts the post-perturbation transcriptome from a control-state transcriptome together with genetic, chemical, temporal and optional chromatin-accessibility information... Incorporating matched ATAC-seq context further improves selected state-transition predictions and enables model-based attribution of chromatin regions to response-associated genes and pathways."*
- **Criterion failed:** **Fails the conjunction of Criteria 3 and 4.** In the experimental benchmarks, BaiZe evaluates ATAC-seq conditioning *exclusively* on genetic perturbations (SMARCA4 and BAZ1B knockouts from MultiPerturb-seq) and developmental lineage state transitions (human embryo and HSPC time-course). For small-molecule drug perturbations (Sci-Plex 188 compounds), BaiZe's model input is strictly: control RNA + 2,048-bit Morgan fingerprint + dose (no ATAC-seq features were provided to the drug model).

### 3. MultiFlow (Wang et al., bioRxiv 2026)
- **Citation:** Haochen Wang, Charming Zhang, Mengran Zhang, Xiaoming Nie, and Qiao Liu. "MultiFlow: coupled flow matching for predicting single-cell multiomic perturbation responses in unseen cellular contexts." *bioRxiv* (August 20, 2026).
- **Identifiers:** DOI: [10.64898/2026.08.20.746112](https://doi.org/10.64898/2026.08.20.746112).
- **URL:** https://www.biorxiv.org/content/10.64898/2026.08.20.746112v1
- **What it does:** Couples flow-matching models for paired single-cell RNA-seq and ATAC-seq. Given baseline cellular multiomic state (control RNA + control ATAC), it predicts the full post-perturbation paired gene expression profile and chromatin accessibility profile across unseen cell contexts.
- **Verbatim quote showing input/output:**
  > *"By learning coupled RNA-ATAC flows conditioned on perturbation and control-derived cellular-state representation, MultiFlow enables prediction of coordinated multiomic responses in unseen cellular contexts... MultiFlow achieved the strongest overall performance in predicting both gene-expression and chromatin-accessibility responses, outperforming competing modality-specific perturbation-prediction methods."*
- **Criterion failed:** **Fails Criterion 4 (Perturbation is genetic, not small-molecule).** MultiFlow was evaluated exclusively on CRISPR single- and multi-gene knockouts (MultiPerturb-seq, Replogle datasets); it contains no small-molecule or chemical perturbation benchmarks.

---

## Adjacent but not killers
*Models and frameworks that operate in this domain but fail essential criteria.*

1. **ExPO (Ngo et al., J Cheminform 2026 / PMC13371340)**
   - **URL:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13371340/
   - **Fails Criterion 3 (No chromatin input).** ExPO predicts full L1000 978-gene response z-scores as a continuous function of dose and time using ChemBERTa-2 molecular embeddings and Fourier exposure operators. Cell line identity is handled without chromatin features. Crucially, the authors explicitly identify chromatin conditioning as the unaddressed frontier: *"narrowing that gap is a meaningful step, while further gains may require richer cell context (e.g., chromatin marks)"*.

2. **GRIP-Lung (IJMS 2026 / PMC13072768)**
   - **URL:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13072768/
   - **Fails Criterion 3 (No chromatin input).** Uses a Generative Adversarial Network (GAN) to predict post-treatment gene expression profiles in lung cancer cell lines. Cell line identity is parameterized via learnable latent embedding vectors added element-wise into normalized baseline gene expression, not measured chromatin features.

3. **DEPICT (bioRxiv 2024 / DOI: 10.1101/2024.04.12.589178)**
   - **URL:** https://www.biorxiv.org/content/10.1101/2024.04.12.589178v1
   - **Fails Criterion 3 (No chromatin input).** Predicts condition-matched, drug-induced transcriptional responses across LINCS L1000 using transformers, but conditions only on baseline transcriptomic expression and chemical structural embeddings.

4. **PrePR-CT (bioRxiv 2024 / DOI: 10.1101/2024.07.24.604816)**
   - **URL:** https://www.biorxiv.org/content/10.1101/2024.07.24.604816v1
   - **Fails Criterion 3 (No chromatin input).** Predicts drug-induced single-cell transcriptional responses in unseen cell types using cell-type-specific graphs as an inductive bias. However, the graphs are derived from basal co-expression and protein-protein interactions (PPI), not measured chromatin state.

5. **chemCPA (Hetzel et al., Nat Biotechnol 2024 / DOI: 10.1038/s41587-023-02058-4)**
   - **URL:** https://doi.org/10.1038/s41587-023-02058-4
   - **Fails Criterion 3 (No chromatin input).** Predicts single-cell drug responses (Sci-Plex) for unseen drugs across cell lines, but condition embeddings use only basal transcriptomic profiles and chemical SMILES/fingerprints.

6. **PRnet (Bioinformatics 2022 / DOI: 10.1093/bioinformatics/btac585)**
   - **URL:** https://academic.oup.com/bioinformatics/article/38/20/4765/6673809
   - **Fails Criterion 3 (No chromatin input).** Encoder-decoder deep learning model predicting compound-induced perturbational gene expression changes from baseline gene expression and chemical structure.

7. **Chromoformer (Genome Biol 2023 / DOI: 10.1186/s13059-023-03008-0)**
   - **URL:** https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03008-0
   - **Fails Criteria 2 and 4 (No perturbation; steady-state expression endpoint).** Predicts basal steady-state mRNA expression levels from histone modification ChIP-seq and 3D Hi-C chromatin conformation. Does not predict response to any perturbation or small molecule.

8. **scDrugMap / scDrug / scDrug+ (2024–2026)**
   - **URL:** https://pubmed.ncbi.nlm.nih.gov/38816426/
   - **Fails Criterion 2 (Output is drug sensitivity/viability, not an expression response profile).** Evaluates foundation model embeddings (scFoundation, scGPT) to predict drug sensitivity (IC50, AUC, binary resistance), rather than a multi-gene transcriptional profile.

9. **Re-verification of User-Listed Near-Misses:**
   - **PERD (PMC11139989 / PMID: 38816426):** Inverse direction. Predicts chromatin accessibility changes *from* drug-perturbed expression profiles. Fails Criterion 1 & 3.
   - **eLife 78012 (PMID: 36043458):** Descriptive multi-omic integration to decode MoA and sensitivity. Correlation/association analysis, not an in silico predictive model. Fails Criterion 1.
   - **Epiregulon (Nat Commun 2025 / DOI: 10.1038/s41467-025-62252-5):** Infers transcription factor activity from scATAC + scRNA to predict drug sensitivity / target vulnerability. Fails Criterion 2 (endpoint is TF activity/sensitivity, not a transcriptional response profile).
   - **CellForge / CondDiffTrans-ATAC:** Predicts CRISPR effects on chromatin accessibility. Fails Criteria 3 & 4 (inverse modality and genetic perturbation).

---

## Search log
*Total sources searched across bioRxiv, PubMed, PMC, Europe PMC, CrossRef, OpenAlex, and Semantic Scholar.*

| # | Search Engine / Database | Exact Query / API Call | Results Inspected | Relevant Findings / Outcome |
|---|---|---|---|---|
| 1 | Web / Scholar | `"ATAC-seq" OR "chromatin accessibility" predict "drug" "transcriptional response" OR "gene expression" profile` | 15 | Surfaced general landscape of ATAC-RNA integration; identified sensitivity tools (DeepDrugCancer, ATSDP-NET). |
| 2 | Web / Scholar | `"predicting" ("transcriptional response" OR "gene expression") "drug" ("ATAC-seq" OR "chromatin accessibility" OR "histone") -sensitivity -IC50` | 10 | **Key hit:** Surfaced BaiZe, MultiFlow, and machine learning models predicting expression changes upon HDAC inhibitors. |
| 3 | Web / bioRxiv | `"BaiZe" "perturbation" "chromatin accessibility" OR "ATAC"` | 8 | Identified BaiZe preprint on bioRxiv (July 2026). |
| 4 | Web / bioRxiv | `"BaiZe" biorxiv "conditional state-transition"` | 6 | Identified BaiZe's conditional state-transition framework. |
| 5 | CrossRef API | `api.crossref.org/works/10.64898/2026.07.15.738608` | 1 | Verified BaiZe metadata, authors (Zeng et al.), and official bioRxiv DOI. |
| 6 | bioRxiv Full Text | `read_url_content(https://www.biorxiv.org/content/10.64898/2026.07.15.738608v1.full)` | Full text (202k chars) | **Deep inspection:** Confirmed BaiZe supports ATAC condition embeddings, but ATAC was tested only on CRISPR (MultiPerturb-seq) and embryo/HSPC; Sci-Plex drug benchmark used only RNA + SMILES. |
| 7 | Web / PubMed | `"predict" "gene expression" "HDAC inhibitor" ("ATAC" OR "chromatin" OR "histone") ("pre-treatment" OR "baseline")` | 10 | Found studies predicting locus-specific expression changes from baseline epigenomes (ROC 0.89). |
| 8 | Web / PubMed | `"ROC" "0.89" "HDAC" "pre-treatment" "epigenom" OR "transcriptome" "locus-specific"` | 5 | Identified HCT116 Largazole / RH4 Entinostat study by Gopalan et al. |
| 9 | Web / PubMed | `"Gopalan" "Largazole" "Entinostat"` | 8 | Identified paper title: "Predicting gene expression changes upon epigenomic drug treatment" (Agrawal, Gopalan, et al.). |
| 10 | PubMed / PMC API | `esearch.fcgi?db=pmc&term=Predicting+gene+expression+changes+upon+epigenomic+drug+treatment+Agrawal` | 3 | Resolved PMID 40417623, PMCID PMC12103705, bioRxiv PMID 37781626. |
| 11 | PMC Full Text XML | `efetch.fcgi?db=pmc&id=PMC12103705&retmode=xml` | Full text (122k chars) | **Deep inspection:** Parsed ML models (SVM, RF, GB), 21 genic bins of H3K27ac features, and proved output is strictly binary classification (top 1000 up vs top 1000 down), not regression. |
| 12 | Web / bioRxiv | `"MultiFlow" "flow-matching" "perturbation" biorxiv OR arxiv OR nature OR science` | 11 | Identified MultiFlow (Wang et al., bioRxiv August 2026). |
| 13 | CrossRef API | `api.crossref.org/works/10.64898/2026.08.20.746112` | 1 | Resolved MultiFlow DOI and author metadata (Liu Lab). |
| 14 | bioRxiv Full Text | `read_url_content(https://www.biorxiv.org/content/10.64898/2026.08.20.746112v1.full)` | Full text (140k chars) | **Deep inspection:** Verified coupled RNA-ATAC flow matching from baseline multiomics; regex check confirmed 0 small-molecule/drug benchmarks (CRISPR-only). |
| 15 | Web / Scholar | `"L1000" ("ATAC-seq" OR "chromatin" OR "histone" OR "epigenom*") "predict*" ("gene expression" OR "transcriptional") "drug" OR "compound"` | 15 | Identified DEPICT, DIGERA, and PERD. |
| 16 | CrossRef API | `works?query.title=Predicting+single-cell+perturbation+responses+for+unseen+drugs` | 3 | Identified PrePR-CT (bioRxiv 2024). |
| 17 | bioRxiv Full Text | `read_url_content(https://www.biorxiv.org/content/10.1101/2024.07.24.604816v1.full.txt)` | Full text (246k chars) | **Deep inspection:** Verified PrePR-CT uses co-expression/PPI network inductive bias, not measured chromatin. |
| 18 | Europe PMC Citations | Forward citation search for PERD (`MED:38816426` / `10.1038/s41540-024-00388-8`) | 3 citations | Checked all forward citations: reviews on multi-omics drug discovery and virtual cells. |
| 19 | Europe PMC Citations | Forward citation search for eLife 78012 (`MED:36043458` / `10.7554/eLife.78012`) | 8 citations | **Key hit:** Surfaced ExPO (PMC13371340), which predicts L1000 signatures and cites eLife 78012. |
| 20 | PMC Full Text XML | `efetch.fcgi?db=pmc&id=PMC13371340&retmode=xml` (ExPO full text) | Full text (166k chars) | Confirmed ExPO does not use chromatin input and states chromatin is a future opportunity. |
| 21 | OpenAlex Citations | Forward citation search for Epiregulon (`W4412862280` / `10.1038/s41467-025-62252-5`) | 10 citations | **Key hit:** Surfaced GRIP-Lung (IJMS 2026). |
| 22 | PMC Full Text XML | `efetch.fcgi?db=pmc&id=13072768&retmode=xml` (GRIP-Lung full text) | Full text (127k chars) | Confirmed GRIP-Lung uses learnable cell embeddings + baseline RNA, not measured chromatin. |
| 23 | Europe PMC Batch | `(ATAC OR chromatin OR histone) AND (predict OR prediction) AND ("transcriptional response" OR "expression profile") AND (drug OR compound)` | 10 of 14,925 hits | Inspected top recent hits (reprogramming, ER binding, deep learning prioritization). |
| 24 | Europe PMC Batch | `TITLE:(chromatin OR ATAC OR histone) AND TITLE:(predict OR predicting) AND (drug OR perturbation)` | 10 of 49 hits | Inspected hits: confirmed Chromoformer, cfDNA profiling, and PERD. |
| 25 | Europe PMC Batch | `("prior chromatin" OR "baseline chromatin" OR "pre-treatment chromatin") AND (predict OR prediction)` | 10 of 83 hits | Inspected hits: confirmed CLASTER, enhancer-gene linking with prior Hi-C. |
| 26 | Web / Scholar | `("scGPT" OR "scFoundation" OR "Geneformer" OR "CellPLM") ("drug" OR "compound") ("ATAC" OR "chromatin") ("perturbation" OR "response")` | 23 | Verified that foundation models either evaluate genetic perturbations (scGPT on Perturb-seq) or predict sensitivity (scDrugMap), not chromatin-conditioned drug response profiles. |
| 27 | Web / Scholar | `("H3K27ac" OR "H3K27me3" OR "chromatin accessibility") "predict" "genes" ("drug" OR "small molecule") ("expression change" OR "transcriptional response")` | 13 | Confirmed PRnet vs Chromoformer distinctions and absence of unified drug+chromatin models. |

---

## How I would attack this claim next
*If acting as a hostile peer reviewer with unlimited time and access to private repositories, here is where I would dig to attempt to kill the claim:*

1. **Hostile Citation of Agrawal et al. (PMC12103705):**
   - As a reviewer, I would write: *"The authors claim that using measured prior chromatin state to predict drug-induced transcriptional response is entirely novel. However, Agrawal et al. (F1000Research 2023 / bioRxiv 2023) already trained machine learning models (SVM, Random Forest, Gradient Boosting) on pre-treatment H3K27ac signal in 21 genic bins to predict whether genes up- or down-regulate in response to HDAC inhibitors (Largazole, Entinostat) across cell lines with AUC up to 0.89. The current paper merely replaces binary classification of extreme genes with regression across the transcriptome."*
   - **How to immunize your manuscript:** Explicitly cite Agrawal et al. in your Introduction. Contrast their formulation (binary classification of a pre-selected subset of top 1000 up/down genes under an epigenetic drug) with your framework (quantitative, continuous profile regression across thousands of genes / genome-wide under diverse small molecules).

2. **Benchmarking Against BaiZe (bioRxiv July 2026):**
   - A reviewer familiar with recent 2026 generative models will point out that BaiZe's paper abstract explicitly promises: *"predicts the post-perturbation transcriptome from a control-state transcriptome together with genetic, chemical, temporal and optional chromatin-accessibility information"*. Even though BaiZe's authors omitted ATAC from their Sci-Plex chemical benchmark, a reviewer could argue that the concept of conditioning perturbation response on ATAC was disclosed first.
   - **How to immunize your manuscript:** Explicitly discuss BaiZe. Note that while BaiZe conceptually proposes multi-view conditioning, it restricted ATAC conditioning to CRISPR knockouts and developmental trajectories, while keeping chemical drug response purely RNA-conditioned. Position your work as the first to empirically demonstrate and evaluate chromatin-conditioned drug response modeling.

3. **Mining MLCB / MLSB / NeurIPS Workshop Repositories:**
   - Conference proceedings for NeurIPS / ICML workshops (Machine Learning in Computational Biology, Machine Learning for Structural Biology, Generative Biology) often feature extended abstracts that are not indexed in PubMed. I would search workshop archives from 2023–2025 for titles combining "L1000", "ENCODE chromatin", and "transfer learning across cell lines".

4. **Investigating Multi-Omic Single-Cell Drug Screening Repositories (10x Multiome / SHARE-seq):**
   - Several recent studies have generated paired single-cell RNA+ATAC data under small-molecule exposure (e.g., BET inhibitors, epigenetic probes). I would check whether any computational lab hosting these datasets on GEO or Zenodo included a baseline predictor script (e.g., predicting single-cell RNA response using baseline scATAC clusters) in their accompanying code repository.

5. **Recommended Revision of the Novelty Claim:**
   - *Current phrasing (vulnerable):* "Conditioning the prediction of a drug-induced transcriptional RESPONSE PROFILE on the cell line's PRIOR chromatin state... is novel — no existing model uses measured chromatin state as an input feature to predict which genes change, and by how much, when a small molecule is applied."
   - *Bulletproof phrasing:* **"While prior studies have used baseline epigenetic features to classify the binary direction of extreme gene subsets under chromatin-targeting drugs (e.g., Agrawal et al., 2023), and recent multi-omic perturbation models have incorporated ATAC-seq solely for genetic knockouts (e.g., BaiZe, MultiFlow), our model is the first to condition genome-wide, continuous transcriptional response profiles on baseline measured chromatin state across diverse small-molecule perturbations."**
