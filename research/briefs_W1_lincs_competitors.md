# TASK W1 — Competitor landscape for bulk LINCS L1000 chemical-perturbation models

## Who you are and why this matters
You are a research assistant to the PI of a computational-pharmacology project preparing a paper for a
top-tier venue. This project has been burned repeatedly by comparing numbers that were not comparable —
once by citing a case-study figure as a benchmark result for six weeks. **A wrong or invented number here
propagates into a manuscript.** Accuracy beats completeness. "UNKNOWN" is a correct and valued answer.

## Scope — INCLUDE only
Models that predict **drug/small-molecule-induced transcriptional response** on **bulk LINCS L1000**
(the 978 landmark genes, or the 12,328-gene space).

## Scope — EXCLUDE (do not report these)
- Genetic/CRISPR perturbation models (TxPert, PertAdapt, GEARS, scGPT-perturb)
- Single-cell-resolution models (State, Tahoe-100M models)
- Drug *sensitivity* / IC50 / viability prediction — a different endpoint
- Chromatin-accessibility prediction models
If a paper is borderline, put it in a separate "BORDERLINE" section with one line on why.

## Targets (start here, then find anything newer)
1. PertDiT — "Predicting drug-perturbed transcriptional responses using multi-conditional diffusion transformer", Quantitative Biology 2026, doi 10.1002/qub2.70016
2. ExPO — "an exposure-conditioned neural operator for L1000 signature prediction", J Cheminformatics 2026, doi 10.1186/s13321-026-01226-1
3. "Predicting condition-aware drug-induced transcriptional responses via a latent diffusion model", Bioinformatics 2026
4. TranSiGen, PRnet, DeepCE, CIGER — the four baselines used by XPert
5. Then search for anything published in 2026 that this list misses.

## For EACH model, fill this table. One row per model.
| field | what to record |
|---|---|
| `name` | model name |
| `venue_year` | journal + year |
| `doi_or_url` | resolvable link |
| `data_level` | LINCS Level 3, Level 5, or other — **quote the sentence that says so** |
| `target_quantity` | absolute post-perturbation expression, or the delta/xdeg. **Quote it.** |
| `metric` | PCC / Spearman / R2 / MSE, and whether per-row then **mean or median** — quote if stated, else UNKNOWN |
| `splits` | exact split definitions. For each: are test **cell lines** held out? test **compounds**? Quote the definition. |
| `headline_numbers` | reported values **with the split they belong to**. Include error bars if given. |
| `null_baselines_reported` | do they report a "copy the control"/"do nothing" baseline? a mean baseline? YES/NO + values |
| `code_available` | repo URL, and whether weights + preprocessed inputs are actually downloadable |
| `runnable_assessment` | could a third party run it? name the specific blocker if not |

## Hard evidence rules — not optional
1. **Every number must carry a verbatim quote or a precise location** (table name, figure number, section).
   If a number exists only inside a figure panel, write `UNKNOWN — figure panel only`. Never estimate.
2. **Check the Supplementary Information and read it.** For Springer/Nature the supplementary is usually at
   a public static URL like
   `https://static-content.springer.com/esm/art%3A10.1038%2F<suffix>/MediaObjects/<doi_underscored>_MOESM<n>_ESM.pdf`
   Benchmark tables are very often there and not in the main text. This project lost six weeks by not
   checking. Do the equivalent for other publishers (PMC full text is often open).
3. **Never average, convert or normalise numbers across papers.** Report them as printed.
4. **Never infer a split definition from its name.** Read the definition. Names like "cold-cell" have been
   found to mean a 90/10 split within one tissue where 100% of test cells also appear in training.
5. If two sources disagree, report both and flag the conflict. Do not silently pick one.

## Deliverable
Write your report to `C:\Projects\LINCS\research\W1_lincs_competitors_REPORT.md`.
End with two mandatory sections:
- `## What I could NOT determine` — every UNKNOWN field and why.
- `## Confidence` — per model: HIGH (read primary source directly) / MEDIUM (secondary source) / LOW
  (inferred). Low confidence honestly labelled is more useful than false high.

Do not write any other files. Do not modify any existing file.
