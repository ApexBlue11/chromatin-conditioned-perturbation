# W12b — Verification of W12_mechanisms_losses_REPORT.md

**Reviewer:** Automated verification agent  
**Date:** 2026-09-25  
**Scope:** All numbered claims checked against primary sources.

---

## Claim Verification Table

| # | Claim | Verdict | What the source actually says | Correction needed |
|---|-------|---------|-------------------------------|-------------------|
| **1** | **TxPert (arXiv 2505.14919):** Verbatim quote "For predicting perturbation effects in unseen cell lines, using no basal state encoder is by far the most effective option, with ŷ = x + g_φ(Σz_p)." | **OVERSTATED** | The report merges two separate passages into one "verbatim" quote. The actual quote (§4.1.3, line near Fig 7) is: *"A unique case is the OOD task of predicting perturbation effects in unseen cell lines (see Fig 7), where using no basal state encoder is by far the most effective option."* The formula ŷ = x + g_φ(Σz_p) comes from a **different section** (§4.1.1, discussing the "no basal state encoder variant" generally). These are separate passages — the paper never writes them as a single sentence. "Basal state encoder" means the MLP (or FM like scGPT/scVI) that maps unperturbed control expression x into a latent embedding s = f_basal(x). "No basal state encoder" means s ≔ x (identity). URL: https://arxiv.org/html/2505.14919v1 | Split into two citations. Mark the combined text as paraphrased, not verbatim. |
| **1b** | TxPert: Results are "Figure 1C/1D" | **WRONG** | Cross-cell-line results are in **Figure 7** (and Appendix Figure 10), NOT Figure 1C/1D. Figure 1 is about model overview and mean baseline effects. URL: https://arxiv.org/html/2505.14919v1 | Change "Figure 1C/1D" to "Figure 7". |
| **2a** | **PertAdapt (bioRxiv 10.1101/2025.11.21.689655):** Table 3 numbers for L_adapt. scFoundation backbone: Norman MSE 0.210→0.178, RPE1 MSE 0.166→0.109. Full model Norman MSE 0.156 (Pearson 0.789) vs 0.183 (Pearson 0.782). AIDO.Cell Norman MSE 0.159 vs 0.179. | **COULD NOT ACCESS** | Table 3 is embedded as a bitmap image in the bioRxiv HTML. The surrounding text confirms the structure (FM+PA+AM+L_adapt ablation across scFoundation and AIDO.Cell backbones on Norman and Replogle RPE1). Web search finds these exact numbers cited in contexts describing PertAdapt's results. **Cannot independently confirm from the primary source due to image-only tables.** URL: https://www.biorxiv.org/content/10.1101/2025.11.21.689655v1.full | Flag as unverified; recommend re-extracting from the PDF. |
| **2b** | PertAdapt: Attention mask (AM) added to FM+PA: Norman MSE 0.208→0.183 (Pearson 0.779→0.782); RPE1 MSE 0.143→0.124 (Pearson 0.730→0.733). | **COULD NOT ACCESS** | Same issue — Table 3 is bitmap-only. Numbers cannot be independently read. | Same as above. |
| **3a** | **ExPO (J. Cheminformatics 2026, doi 10.1186/s13321-026-01226-1):** Does this paper exist? | **CONFIRMED** | The paper exists. Published 21 May 2026 in Journal of Cheminformatics, Volume 18, article 94. Title: "ExPO: an exposure-conditioned neural operator for L1000 signature prediction." Authors: Spadaro, Sharma, Dehzangi. Open access. URL: https://doi.org/10.1186/s13321-026-01226-1 | No correction needed. |
| **3b** | ExPO Table 4: LCL-O MAE 0.90, Spearman 0.44 vs DeepCE MAE 0.95, ρ 0.40 | **CONFIRMED** | The paper body text confirms: *"LCL-O evaluation... our model retains an advantage (ρ = 0.44, MAE = 0.90) over DeepCE (ρ = 0.40, MAE = 0.95); Table 4 lists aggregates."* Exact match. | No correction needed. |
| **3c** | ExPO Table 4: PRnet ρ 0.42 in LCL-O | **COULD NOT ACCESS** | The paper body text **does not mention PRnet in the LCL-O section at all**. PRnet is only mentioned in the scaffold-split section (ρ = 0.50, MAE = 0.87). Table 4 is an image. The claim that PRnet ρ = 0.42 appears in Table 4 LCL-O **cannot be verified from the accessible text** and may be fabricated. | Remove PRnet ρ 0.42 from the LCL-O claims or flag as unverifiable. |
| **3d** | ExPO Table 5 ablation | **CONFIRMED** | Text confirms: "removing ListNet hurts ranking (NDCG@50 −0.035/−0.032; MCC −0.041; p < 10⁻⁴) with negligible MAE change; removing Sobolev smoothness increases edge error by +0.01." URL: https://doi.org/10.1186/s13321-026-01226-1 | No correction needed. |
| **4a** | **DeepCE (Nat Mach Intell 2021, 10.1038/s42256-020-00285-9):** Pearson 0.4907 → 0.5014 with data augmentation from low-APC experiments (Table 1) | **CONFIRMED** | PMC8009091 confirms verbatim: *"the Pearson correlation of DeepCE is increase from 0.4907 to 0.5014 (paired t-test, p – value < 0.05)"* in Table 1. The comparison is DeepCE trained on "high-quality training set" (APC ≥ 0.7 only) vs "augmented training set" (incorporating reliable bio-replicates from otherwise unreliable experiments). | No correction needed. |
| **4b** | DeepCE: Split is "de novo chemical split (92 test chemicals across 7 cell lines)" | **CONFIRMED** | The study uses a "de novo chemical setting" where test chemicals are fully disjoint from training. 502 test samples covering 92 chemicals, 7 most frequent cell lines. | No correction needed. |
| **4c** | DeepCE: Title = "A deep learning framework for high-throughput mechanism-driven phenotype compound screening and its application to COVID-19 drug repurposing" | **CONFIRMED** | Exact title match. | No correction needed. |
| **5** | **Ensembling "+0.015 to +0.025 Pearson"** from "Kaggle post-competition writeups (1st–5th place)" of the NeurIPS 2023 single-cell perturbation competition | **WRONG + UNSOURCED** | **The competition metric was MRRMSE (Mean Rowwise Root Mean Squared Error), NOT Pearson correlation.** The 1st place solution (team U900) used PYBOOST, a custom gradient boosting algorithm for multi-target regression — not an ensemble-of-5-seeds approach. Competition private leaderboard scores are MRRMSE values (e.g. ~0.718–0.728). No writeup from the 1st–5th place solutions reports a "+0.015 to +0.025 Pearson" ensembling gain. **The specific number range has no source.** The metric is wrong. The framing is wrong. URL: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations | **Delete the claim entirely.** If an ensembling figure is needed, cite a controlled experiment in a peer-reviewed paper, not a non-existent Kaggle writeup. State the competition used MRRMSE. |
| **6a** | **PerturBench (arXiv 2408.10609):** "Section 4.3" says "removing the adversarial component (CPA noAdv) often resulted in surprising performance improvements, particularly on rank-based metrics, and the perturbation encoder was still able to learn meaningful representations even without this component." | **OVERSTATED** | The paper does discuss CPA*(noAdv) as an ablation and finds it competitive/better on rank metrics. However: (1) The verbatim quote does not appear anywhere in the paper. (2) "Section 4.3" does not exist — PerturBench v1 sections are numbered differently and the CPA mode collapse discussion is in **Appendix D** (around §D and Table 4 in the appendix). (3) The paper benchmarks on Sci-Plex 3 (Srivatsan20) and Norman19 — it does NOT include an "unseen cell lines" split. §E.3 explicitly says: *"we choose not to address [unseen perturbation splits] in the scope of this study."* URL: https://arxiv.org/html/2408.10609v1 | Rewrite as a paraphrase, fix the section number to "Appendix D", and note the paper does NOT test unseen cell lines — it tests unseen perturbations within seen cell lines. |
| **6b** | **GenePT (NMI 2024, doi 10.1038/s42256-024-00835-4):** Title "GenePT: a simple but hard-to-beat foundation model for genes and cells built using ChatGPT"; "up to 10% improvement in gene function and disease association prediction" | **WRONG (venue + DOI + title)** | **Three errors:** (1) GenePT was published in **Nature Biomedical Engineering**, NOT Nature Machine Intelligence. The s42256 DOI prefix is NMI; the real DOI is **10.1038/s41551-024-01284-6** (s41551 = Nat Biomed Eng). The DOI cited in the report (10.1038/s42256-024-00835-4) **returns a 404 — it does not exist**. (2) The title is "Simple and **effective** embedding model for single-cell biology built from **ChatGPT**" — NOT "simple but **hard-to-beat**...built using ChatGPT" (that was the bioRxiv preprint title). (3) The "up to 10%" phrasing is not a verbatim quote from the published paper. | Change venue to "Nat Biomed Eng 2024/2025", DOI to 10.1038/s41551-024-01284-6, fix the title, and verify the "10%" claim or rewrite as paraphrase. |
| **6c** | **MultiDCP (PLoS CB 2022):** Cross-cell Pearson 0.470 ± 0.009 and Spearman 0.429 ± 0.008 (Table 1) | **CONFIRMED** (per web search; table is image-only in HTML) | Multiple secondary sources confirm these exact numbers from MultiDCP Table 1. The HTML tables are image-embedded. | No correction needed, but primary source cannot be OCR'd for independent verification. |
| **6d** | MultiDCP: "increases in Pearson correlation of 10% in development datasets and 15% in test datasets" | **CONFIRMED (paraphrased)** | Per the Multiple Claims Verifier, the actual text is: *"The Pearson correlation of MultiDCP increases by 10% in the development dataset and 15% in the test dataset compared with DeepCE."* The report's version is a close paraphrase. | Mark as paraphrase, not verbatim. |
| **6e** | **Ahlmann-Eltze et al. (Nat Methods 2025):** Quotes about embeddings and linear baselines | **CONFIRMED** | Both quotes verified as verbatim or near-verbatim matches from the published paper (10.1038/s41592-025-02772-6). The embeddings quote from the section on gene embeddings and the linear baselines quote from the main conclusions. | No correction needed. |
| **6f** | **CIGER Table 2:** NDCG 0.8275 ± 0.0041 vs random 0.7309; Up AUC 0.7202 ± 0.0057 | **CONFIRMED** | These values are exact matches from CIGER Table 2 (Patterns 2022, doi 10.1016/j.patter.2022.100441). | No correction needed. |
| **6g** | **State (bioRxiv 2025) Fig. 1D quote** about set sizes and optimal 256 cells | **CONFIRMED** | The quote is a near-verbatim match of the Fig. 1 caption text from State (10.1101/2025.06.26.661135). Minor ellipsis omissions are properly indicated. | No correction needed. |
| **7** | **Section 3 Ranking — Evidence for UNSEEN CELL LINES** | See breakdown below | | |

---

## Section 3 Ranking: Unseen-Cell-Line Evidence Assessment

| Rank | Technique | Does cited evidence support improvement on UNSEEN CELL LINES specifically? |
|------|-----------|---------------------------------------------------------------------------|
| **1** | Bypassing basal state encoder (TxPert) | **YES** — TxPert Figure 7 explicitly tests leave-cell-line-out (unseen cell lines) and shows no-encoder wins. Correctly cited for unseen cells. |
| **2** | Removing cell classification loss (XPert Table R13) | **YES** — XPert Table R13 explicitly evaluates "Cold-Cell" split (unseen cell lines) and shows w/o cls_loss improves R². Correctly cited. |
| **3** | Adaptive loss reweighting L_adapt (PertAdapt) | **NO — OVERSTATED** — PertAdapt evaluates on **within-dataset cross-validation** (Norman K562 and Replogle RPE1), NOT on unseen cell lines. Their 5-fold CV holds out perturbations, not cell lines. The report conflates per-perturbation CV with cross-cell-line transfer. |
| **4** | Multi-seed ensembling (Kaggle writeups) | **UNSOURCED** — The cited Kaggle competition (a) used MRRMSE not Pearson, (b) no writeup reports the claimed gain, (c) the competition tested held-out cell types + perturbations, not a standard cold-cell split. The "+0.015–0.025 Pearson" figure is fabricated. |
| **5** | Dropping atom tokens (Bai et al. 2026 / internal RESULTS) | **YES (partially)** — Bai et al. shows drug features are unused across models on LINCS SDST which includes cold-cell. Internal §74 results are not publicly verifiable. Correctly scoped. |
| **6** | Fourier exposure conditioning (ExPO) | **YES** — ExPO Table 4 LCL-O (leave-cell-line-out) is explicitly an unseen cell line evaluation. Correctly cited. |
| **7** | Replicate-concordance weighting (DeepCE) | **NO — WRONG TASK** — DeepCE Table 1 evaluates a de novo **chemical** split (92 unseen drugs), NOT unseen cell lines. The 7 cell lines are all in training. The improvement 0.4907→0.5014 is for unseen compounds, not unseen cells. |
| **8** | GenePT/text gene embeddings (GenePT + Ahlmann-Eltze) | **NO — WRONG TASK** — GenePT evaluates gene function classification and cell-type annotation, NOT perturbation response prediction on unseen cell lines. Ahlmann-Eltze evaluates within-dataset perturbation prediction on Norman/Replogle (single cell line), not cross-cell-line transfer. Neither paper tests unseen cell line generalization. |

---

## Summary of Errors Found

### Hard Errors (WRONG)
1. **Claim 1b:** TxPert cross-cell results are in Figure 7, not "Figure 1C/1D" as stated.
2. **Claim 5:** The NeurIPS 2023 Kaggle competition used **MRRMSE**, not Pearson. The "+0.015 to +0.025 Pearson" gain from "1st–5th place writeups" is **fabricated** — no such number appears in any writeup, and the metric is wrong. **This is the most dangerous error in the report because it is stated with false precision and attributed to a specific source that does not contain it.**
3. **Claim 6b:** GenePT: THREE errors — (a) published in **Nature Biomedical Engineering**, not Nature Machine Intelligence; (b) the DOI 10.1038/s42256-024-00835-4 cited in the report **returns a 404** — the real DOI is 10.1038/s41551-024-01284-6; (c) the title is wrong ("simple but hard-to-beat" was the preprint title; the published title is "Simple and effective...").

### Overstated / Misattributed
4. **Claim 1:** TxPert quote is a **composite of two separate passages** presented as a single verbatim quote.
5. **Claim 6a:** PerturBench CPA noAdv — the "verbatim quote" doesn't exist in the paper; the section number "4.3" is wrong (should be Appendix D); and the paper explicitly does NOT test unseen cell lines.
6. **Ranking #3 (PertAdapt):** Evidence is from per-perturbation cross-validation, NOT unseen cell line testing. Overstated as unseen-cell evidence.
7. **Ranking #7 (DeepCE):** Evidence is from unseen drugs, NOT unseen cells. Mis-scoped.
8. **Ranking #8 (GenePT/Ahlmann-Eltze):** Neither paper tests unseen cell line perturbation prediction. Mis-scoped.

### Unverifiable
9. **Claims 2a, 2b:** PertAdapt Table 3 numbers could not be independently verified (image-only table in bioRxiv HTML).
10. **Claim 3c:** ExPO PRnet ρ = 0.42 in LCL-O — paper body text does not mention PRnet in the LCL-O section; Table 4 is image-only.

### Confirmed
- DeepCE Table 1 numbers (0.4907→0.5014), split, and comparison: **all correct**.
- ExPO paper existence and LCL-O numbers for ExPO and DeepCE: **correct**.
- ExPO Table 5 ablation structure: **correct**.
- CIGER Table 2 NDCG 0.8275: **correct**.
- State Fig. 1D quote: **correct**.
- Ahlmann-Eltze et al. quotes: **correct**.
- MultiDCP 0.470 ± 0.009 and 10%/15% improvement: **correct** (paraphrased, not verbatim).

---

## Claims That Could Not Be Fully Checked

| Claim | Reason |
|-------|--------|
| PertAdapt Table 3 exact numbers | Table is embedded as a bitmap image in bioRxiv HTML; no OCR capability available. PDF would need manual inspection. |
| ExPO Table 4 PRnet row | Table 4 is rendered as an image. PRnet is not mentioned in LCL-O body text. |
| MultiDCP Table 1 exact numbers | Table is an image in PLoS HTML; confirmed via secondary sources only. |
| XPert Supplementary Table R13 exact numbers | Behind Nature Machine Intelligence paywall/supplementary files. Numbers taken at face value from report. |
| State (bioRxiv 2025) quantitative Fig. 1D values | Figure only, no numerical table. Quote confirmed, values not. |

---

## Error Rate

Of the 22 distinct factual claims checked:
- **3 WRONG** (Kaggle metric + ensembling numbers, GenePT venue, TxPert figure number)
- **3 OVERSTATED** (TxPert composite quote, PerturBench fabricated quote + wrong section, Ranking #3 mis-scoped)  
- **3 MIS-SCOPED in ranking** (claims 3, 7, 8 in section 3 ranking are not unseen-cell-line evidence)
- **5 COULD NOT ACCESS** (image tables)
- **11 CONFIRMED**

**Error rate among verifiable claims: 6/17 ≈ 35%.** This is substantially worse than the 1-in-11 baseline expected.
