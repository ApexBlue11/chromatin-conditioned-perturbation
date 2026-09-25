# TASK W12 — What measurably improves drug-perturbation response prediction? Losses, mechanisms, training tricks

## Context you need
I run a transformer ("v9") that predicts drug-induced transcriptional response on bulk LINCS L1000 (978 landmark
genes). Inputs per row: the cell line's control expression, a drug (a global Uni-Mol embedding token plus per-atom
tokens with gene-to-drug cross-attention), dose and time, a cell-lineage vector, and three chromatin tracks
(ATAC, H3K27ac, H3K27me3). Each gene is a token. Loss: Huber on absolute expression + Huber on the delta
(treated minus control) + a Pearson-correlation term on the delta, plus small fixed-weight auxiliary losses.
Trained 12 epochs, warmup-stable-decay LR, EMA weights, fp16 autocast, PyTorch SDPA attention with a padding mask.

The headline task is **generalisation to UNSEEN CELL LINES** (whole cell lines held out), scored as mean per-row
Pearson on the delta over differentially-expressed genes. On XPert's (Nature Machine Intelligence 2025) cold-cell
split we score ~0.47 vs their published 0.383. We also care about unseen compounds.

Already measured to contribute NOTHING in this model (do not re-propose these in the same form): chromatin summed
into gene tokens; one STRING message-passing step over gene tokens; a Reactome/GO pathway layer; atom-level drug
tokens (they HURT on unseen compounds). A May-2026 benchmark (Bai, Prince, Nitschke, bioRxiv
10.64898/2026.05.13.724458) found seven L1000 models do not use drug features at all.

## Your job
Find techniques from the 2023–2026 literature that were **shown, by an ablation or a controlled comparison, to
improve accuracy** on transcriptional perturbation prediction (L1000, Perturb-seq / sci-Plex, or bulk
drug-response). Cover all five categories:

1. **Loss functions** — e.g. GEARS' autofocus + direction-aware loss, set-level losses (MMD / energy distance as in
   Arc Institute's STATE), Pearson / rank / listwise losses, DEG-weighted or tail-weighted losses, losses that
   handle replicate noise (L1000 consensus / MODZ weighting), contrastive objectives.
2. **Architecture mechanisms** — how the best models condition on the cell and on the drug; adversarial
   disentanglement (CPA, chemCPA); control-set conditioning; anything shown to help UNSEEN CELLS specifically.
3. **Pretrained representations** — gene embeddings (scGPT, Geneformer, GenePT / LLM-text gene embeddings, ESM
   protein embeddings), cell-line embeddings (CCLE / DepMap), drug encoders (Uni-Mol2, MolFormer, CheMeleon, others).
4. **Training tricks** — ensembling, SWA / EMA, mixup or augmentation, curriculum, test-time augmentation,
   multi-task pretraining on other datasets, noise-robust training.
5. **Efficiency only** (speed, not accuracy) — FlashAttention-2 / varlen attention with padding, bf16,
   torch.compile, sequence packing.

Also look for the critiques: papers showing a technique does NOT help, or that simple baselines match it
(e.g. Ahlmann-Eltze, Huber & Anders, Nature Methods 2025; PertEval-scFM; Systema's "systematic variation"
argument). **For every technique, look for the strongest evidence AGAINST it as hard as the evidence for it.**
A technique with a positive number and no ablation is weak; say so.

## Output — write ONE file: `research/W12_mechanisms_losses_REPORT.md`
1. A table, one row per technique: category | technique | paper (title, venue, year, DOI or arXiv id) | dataset
   and split it was tested on | the measured gain, **quoted verbatim with its metric** | is the gain isolated by an
   ablation? (yes / no / unclear) | evidence against (quote + source, or "none found") | cost to add to a model
   like mine (low / medium / high).
2. Per technique, 3–6 lines: how it works, precisely enough to reimplement.
3. A final section, **clearly separated and labelled as your judgement**: the 8 techniques most likely to improve
   UNSEEN-CELL accuracy for a model like mine, ranked, with one line of reasoning each.

## Rules
- Every number must be quoted from the source with a URL. If you cannot open the full text, say "abstract only".
  If you are unsure a number is right, say so in the row — an unflagged wrong number is the worst outcome.
- Do not invent papers. If a technique I named has no ablation evidence, say that plainly; that is a useful result.
- Do not modify any file except the report. Do not run training or install packages.
