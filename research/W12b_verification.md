# TASK W12b — Find the errors in another agent's literature report

Another agent wrote `research/W12_mechanisms_losses_REPORT.md` (read it). It is going to steer where scarce GPU hours
are spent, so every wrong number costs real money. **Your job is to break it.** You are not asked whether the report is
good; you are asked which of its claims are false, unsupported, mis-attributed, or overstated. Assume errors exist:
in a previous report of this kind, one figure in eleven was wrong and was not flagged.

## Check each of these claims against the PRIMARY source (open the paper / preprint / supplementary yourself)
1. **TxPert** (arXiv 2505.14919): the verbatim quote that for unseen cell lines "using no basal state encoder is by far
   the most effective option", and what exactly "basal state encoder" means there.
2. **PertAdapt** (bioRxiv 10.1101/2025.11.21.689655): Table 3's MSE / Pearson numbers for L_adapt and for the attention
   mask (AM), on Norman and Replogle RPE1. Are the numbers, rows and backbones as stated?
3. **ExPO** ("an exposure-conditioned neural operator for L1000 signature prediction", J. Cheminformatics 2026,
   doi 10.1186/s13321-026-01226-1): **does this paper exist?** If yes, check Table 4 (leave-cell-line-out MAE 0.90,
   Spearman 0.44 vs DeepCE 0.40, PRnet 0.42) and Table 5's ablation. If it does not exist, say so plainly.
4. **DeepCE** (Nat Mach Intell 2021, 10.1038/s42256-020-00285-9): Pearson 0.4907 -> 0.5014 with data augmentation
   from low-APC experiments (Table 1). Right numbers? Right comparison? Which split?
5. **Ensembling "+0.015 to +0.025 Pearson" from "Kaggle post-competition writeups (1st-5th place)"** of the NeurIPS
   2023 single-cell perturbation competition. Find the writeups. What metric did that competition use? Is any such
   Pearson number reported anywhere? If the figure has no source, say "unsourced".
6. **PerturBench** (arXiv 2408.10609) quote about CPA noAdv; **GenePT** (NMI 2024) "up to 10 %" quote;
   **MultiDCP** (PLoS CB 2022) cross-cell Pearson 0.470 ± 0.009 and the "10 % / 15 %" quote;
   **Ahlmann-Eltze et al.** (Nat Methods 2025) quotes about embeddings and linear baselines;
   **CIGER** Table 2 NDCG 0.8275; **State** Fig. 1D quote.
7. The report's **section 3 ranking**: for each of the 8, say whether the cited evidence actually supports an
   improvement on UNSEEN CELL LINES specifically, or only on another task / split / dataset. Be concrete.

## Output — write ONE file: `research/W12b_verification_REPORT.md`
A table: claim # | verdict (CONFIRMED / WRONG / UNSOURCED / OVERSTATED / PAPER NOT FOUND / COULD NOT ACCESS) |
what the source actually says (quote + URL) | how it should be corrected. Then a short list of the claims you could not
check and why. **Do not soften.** An unflagged confirmation of a wrong number is the worst outcome; "could not access"
is an acceptable answer. Do not modify any other file. Do not install anything.
