# ChIP-Atlas — investigated, then DROPPED (not in the final coverage)

ChIP-Atlas (https://chip-atlas.org) was explored as strategy **S4** to fill the
31 zero-coverage cancer lines that ENCODE / Cistrome / EpiMap could not.

## Scripts
- `download_chipatlas.py` — streams the 199 MB `experimentList.tab` (kyushu-u mirror),
  filters to **human ATAC/DNase/H3K27ac/H3K27me3** (72,962 of 439,593 rows). The 13.7 MB
  filtered subset (`chip_atlas_human_epi.tab`) was **deleted after use** — reproducible by re-running this script.
- `chipatlas_match.py` — matches gap cells to ChIP-Atlas by Cellosaurus synonym against
  the structured `Cell_type` field.

## Result → `chipatlas_candidates.tsv`
- Found name-matches for only **4 already-partial cells** (THP1, U266, U937, RMGI) — **zero**
  of the 31 truly-dataless obscure lines are in ChIP-Atlas.
- Verification **rejected** SNUC4 (its synonym "C4" matched LNCaP **C4-2**, a prostate line).
- A deeper GEO audit showed ChIP-Atlas pools **mix perturbed samples** (e.g. THP-1 H3K27me3
  contains HSV-1-infected samples), and ChIP-Atlas carries **no CVCL** (no sample-level
  contamination audit possible).

## Decision
**Dropped.** Marginal value (4 already-covered cells, 0 dataless lines) did not justify the
per-sample treatment-filtering required. Nothing from ChIP-Atlas is in `coverage_report_final.tsv`.
(Separately, the ChIP-Atlas GEO metadata *did* confirm that SKMEL28's Cistrome "MEL-745A"
contamination was a parser artifact — that finding rehabilitated SKMEL28 in the final report.)
