# TASK W3 — Try to FALSIFY a novelty claim

## Your job is adversarial
I am about to claim novelty in a paper. **Your task is to destroy that claim if it can be destroyed.**
You are not helping me defend it. Finding one paper that kills it is the single most valuable outcome of
this task, and far more useful to me than a reassuring "no prior work found". A false novelty claim that
survives to peer review is a career problem; a claim killed today is a free save.

## The claim to attack
> **Conditioning the prediction of a drug-induced transcriptional RESPONSE PROFILE on the cell line's
> PRIOR chromatin state (ATAC-seq accessibility, H3K27ac activation, H3K27me3 Polycomb repression) is
> novel — no existing model uses measured chromatin state as an input feature to predict which genes
> change, and by how much, when a small molecule is applied.**

## Decompose it — a paper kills the claim if it does ALL of:
1. **predicts** (not merely analyses or correlates post hoc), AND
2. the **output** is a transcriptional response/expression-change profile across many genes
   (NOT drug sensitivity, NOT IC50, NOT viability, NOT a single biomarker, NOT a binary label), AND
3. an **input** is measured chromatin state (ATAC / DNase / a histone mark / ChIP), measured on the cell
   BEFORE or independent of the perturbation, AND
4. the perturbation is a **small molecule / drug** (genetic-perturbation papers weaken but do not kill it —
   report them in a separate PARTIAL section).

A paper that fails any of 1–3 does NOT kill the claim. Say so explicitly rather than reporting a near-miss
as a hit. **Equally: do not dismiss a genuine hit because it is in an obscure venue or uses different
words.** Both error directions are costly.

## Known near-misses — I have already checked these, do not re-report them as new
- **PERD** (PMC11139989) — predicts chromatin accessibility change FROM perturbed expression. Inverse
  direction. Does not kill.
- **eLife 78012** — integrates transcriptome + chromatin state to decode MoA. Analysis, not a predictive
  model. Does not kill.
- **Epiregulon** (Nat Commun 2025) — TF activity from scATAC+scRNA to predict drug response/sensitivity.
  Different endpoint. Does not kill.
- **CellForge / CondDiffTrans-ATAC** — models CRISPR effects ON accessibility. Inverse. Does not kill.
Your job is to find what I have NOT already found.

## Search strategy — vary the vocabulary aggressively
Different communities name this differently. Search combinations across:
- chromatin accessibility / ATAC-seq / DNase / open chromatin / epigenomic state / histone modification /
  H3K27ac / H3K27me3 / Polycomb / bivalent / enhancer landscape / chromatin priming / poised
- drug response prediction / perturbation response / transcriptomic response / expression signature /
  L1000 / CMap / Connectivity Map / compound-induced / small-molecule-induced
- conditioned on / multi-omic input / epigenome-informed / cell-context encoding / prior-state
Also search: **biorxiv, arXiv, PMC, Semantic Scholar**, and check **papers citing** the near-misses above
(forward citation search often surfaces the exact thing a keyword search misses).
Do not stop at page 1 of results.

## Deliverable
Write to `C:\Projects\LINCS\research\W3_novelty_REPORT.md` with exactly these sections:

1. `## VERDICT` — one of: `CLAIM KILLED` / `CLAIM WEAKENED` / `CLAIM SURVIVES THIS SEARCH`.
   If KILLED, name the paper in the first line.
2. `## Killers` — papers meeting all of criteria 1–4. For each: citation, URL, and a **verbatim quote**
   showing chromatin is an INPUT and a response profile is the OUTPUT. If empty, write "None found."
3. `## Partial` — genetic-perturbation versions, or chromatin-conditioned models with a different output.
   Explain in one line which criterion each fails.
4. `## Adjacent but not killers` — with the specific criterion each fails.
5. `## Search log` — every query you ran and roughly how many results you inspected. This lets me judge
   how hard you actually looked; a short log means a weak search.
6. `## How I would attack this claim next` — if you were a hostile reviewer with more time, where would
   you look that you did not get to?

## Evidence rules
- A claim that a paper is a "killer" REQUIRES a verbatim quote from that paper showing the input/output
  structure. No quote, no killer — put it in Partial instead.
- Never assert a paper exists without a resolvable URL you actually opened.
- If you are unsure whether something qualifies, put it in Partial and explain the ambiguity. Ambiguity
  reported is useful; ambiguity hidden is not.

Do not write any other files. Do not modify any existing file.
