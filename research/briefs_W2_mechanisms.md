# TASK W2 — Extract three specific MECHANISMS from 2026 perturbation papers

## Context you need
I run a model that predicts drug-induced transcriptional response on bulk LINCS L1000 (978 landmark
genes). Three components of my architecture are measured **true nulls** — they run, they fire, and they
contribute nothing:

- **A STRING PPI graph** (19,496 nodes / 929,472 edges, full-proteome) used as one message-passing step
  over gene representations. Ablating it changes accuracy by −0.0003. It is drug-invariant by
  construction: a static graph smoothing static gene features.
- **A named Reactome/GO pathway layer** (~800 named nodes). Contributes +0.0009 to accuracy — but its
  activations align with measured biology at 8–12 standard deviations above a permutation null. Real
  signal, zero accuracy.
- **A chromatin input** (83 cells × 978 genes × 3 tracks: ATAC, H3K27ac, H3K27me3), summed additively
  into each gene's token. Contributes +0.0042 on unseen cell lines.

My hypothesis is that these fail because they enter **additively** into a gene-token stream, rather than
as their own token sets that are attended over. I need to know what the 2026 literature actually does.

## Your job: extract MECHANISM, not summary
For each paper below, answer the specific questions. I do not want an abstract summary — I want to know
how the thing works, precisely enough to reimplement it.

### Paper 1 — TxPert, "using multiple knowledge graphs for prediction of transcriptomic perturbation effects", Nature Biotechnology 2026, doi 10.1038/s41587-026-03113-4
(Preprint with likely more detail: arXiv 2505.14919)
- **Q1.1** How exactly are MULTIPLE knowledge graphs combined? Name the operator — concatenation of
  per-graph embeddings? attention over graphs? a learned gate? separate GNN encoders then fusion? Quote
  or paraphrase the architecture section and give the equation if there is one.
- **Q1.2** Which graphs do they use, and what does the ablation say each one contributes **individually
  vs in combination**? Give numbers.
- **Q1.3** How do they represent a cell line the model has never seen? This is the key question — give the
  mechanism.
- **Q1.4** Is the graph used as a static prior, or is it conditioned on the perturbation? If conditioned,
  how?

### Paper 2 — State, "Predicting cellular responses to perturbation across diverse contexts", Cell 2026
(Preprint: bioRxiv 2025.06.26.661135; code: github.com/ArcInstitute/state)
- **Q2.1** The State Transition model applies self-attention over *sets of cells*. What exactly are the
  tokens — individual cells? What is the set? How is the set constructed at inference?
- **Q2.2** How is a perturbation represented and injected into that stream?
- **Q2.3** How is cell *context* (the unperturbed state) encoded, and what makes it transfer to a context
  not seen in training?

### Paper 3 — PertAdapt, "unlocking single-cell foundation models for genetic perturbation prediction via condition-sensitive adaptation" (PMC13341120)
- **Q3.1** What is "condition-sensitive adaptation" mechanically? Which parameters are adapted and by what?
- **Q3.2** What was failing in the un-adapted foundation model that this fixes?

## Cross-cutting question — answer this LAST, after the above
**Q4.** Across these three papers plus anything else you find: is there evidence, one way or the other,
that auxiliary biological information contributes more when supplied as **its own attended token set**
than when **summed into an existing representation**? Cite specific ablations. If the literature does not
address this, say so plainly — that is a valuable answer, because it would make the comparison novel.

## Evidence rules
- Every mechanism claim needs a **section/equation/figure reference** from the actual paper.
- Where the paper is paywalled, use the arXiv/bioRxiv preprint or PMC version and **say which you used**,
  because preprint and published versions can differ.
- If you cannot determine something, write `UNKNOWN` and say what you looked at. Do not fill gaps with
  plausible-sounding architecture description. An invented mechanism is worse than no answer.
- Distinguish clearly between what the paper *claims* and what its *ablation measures*.

## Deliverable
Write to `C:\Projects\LINCS\research\W2_mechanisms_REPORT.md`, organised by the question numbers above.
End with `## What I could NOT determine` and `## Which version of each paper I read`.

Do not write any other files. Do not modify any existing file.
