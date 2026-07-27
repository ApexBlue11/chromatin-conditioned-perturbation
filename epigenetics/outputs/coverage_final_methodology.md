# Epigenetics Coverage — Final Methodology (Part C, post-Checkpoint-1)

Deliverable: `coverage_report_final.tsv` (249 rows = 83 cells × 3 marks).
Built by `build_coverage_final.py` from the v5/v6 audits + operator decisions.
Audit of every removed sample: `dropped_contaminant_samples.tsv`.

## Resolution tiers (waterfall), per (cell, mark)
1. **direct_measurement** — real assay for the actual line (ENCODE exact / Cistrome name-verified / EpiMap perturb-filtered).
2. **substitute_assay** — different assay standing in for a missing one: DNase-seq or H3K4me3 ChIP-seq → ATAC-seq. `is_primary_assay=False`, `assay_used` names the real assay.
3. **tissue_type_match** — no line-specific data anywhere; a genuine same-tissue pool is used. `identity_confidence=tissue_type_only`.
4. **related_line_inheritance** — a dot-suffix line inherits its parent's cleaned data (documented, flagged).
5. **unresolved** — no defensible data. No imputation, ever.

## Columns
`status, resolution_tier, assay_used, is_primary_assay, source_used, genome_build,
identity_confidence {name_verified|tissue_type_only|related_line}, contamination_status
{clean|contaminant_samples_removed}, model_confidence_flag, sample_ids_used, sample_ids_dropped, notes`

## Contamination handling (operator decision: drop + re-aggregate)
The 12 GENUINE_CONTAMINATION rows: each pool's **dominant identity is the correct line**; a few
samples carry an independent second line (often ONE explicitly-multiplexed sample whose title names
two lines, e.g. HEK293T/ATAC sample 94772 "HEK293T, NIH/3T3, A549"). Those specific samples are
dropped; the clean remainder is kept. 11/12 retain a non-empty clean pool. Hybridoma/ancestry
co-tags are parser artifacts and are NOT dropped. Same cleaning applied to NPC's H3K4me3 substitute
and to NPC-inheriting lines.

## Per-line decisions (Checkpoint 1)
- **Tissue-appropriate, kept + flagged**: HUES3→hESC, MCH58→fibroblast, NKDBA→kidney-epithelial.
- **Wrong-tissue, rebuilt**: SKL/SKB/SKL.C — v6 matched a lymphoblastoid AT6BR-LCL pool via the word
  "skeletal muscle"; replaced with a **clean skeletal-muscle pool** (AT6BR + a DUX4-induced sample excluded).
- **Dot-suffix** (`.C`/`.TAK` = not documented perturbations; convention confirmed via cell_info, where
  `.311/.101/.CAS9`=Cas9, `.KCL`=explicit perturbation):
  - ASC.C ← ASC (ATAC only; lot variant).
  - SKL.C ← SKL clean-muscle pool (lot variant).
  - NPC.TAK ← NPC cleaned — flagged `possible_donor_difference` (maintained as a distinct line; 1283 sigs).
  - NPC.CAS9 ← NPC cleaned — pre-authorized (Cas9 chromatin-neutral).
- **MNEU.E** — NOT_DEFENSIBLE (terminally-differentiated motor neurons); stays unresolved (locked).
- **SKMEL28** — its only H3K27me3 sample and only ATAC-substitute sample are each a single record
  co-tagged SK-MEL-28 / MEL-745A; dropped → SKMEL28 fully unresolved (operator decision).
- **MULTIPLEXED discrepancy** — per-sample re-audit (live Cellosaurus): the briefing's 3 "multiplexed"
  rows are row-level GENUINE_CONTAMINATION *sourced from* single explicitly-multiplexed samples; the
  on-disk GENUINE_CONTAMINATION label stands, drop resolves them. No row is a whole-pool multiplexed experiment.

## Gap-filling round (S1-S4), post-Checkpoint-1
Operator asked to reduce the 127 unresolved. Strategies run, each with per-match verification:
- **S1 Cistrome synonym re-search (CVCL-verified)**: 0 net-new. Verification blocked 2 bad fills
  (SNUC4 synonym "C4" -> LNCaP C4-2 prostate; SKMEL28 hit = already-handled sample).
- **S2 Live ENCODE re-query**: 0 new. ENCODE exhausted for the roster (v5 cache was complete).
- **S3 EpiMap imputed (name-verified biosample, perturb-null)**: +4 rows -> HS27A(H3K27ac,H3K27me3),
  HT29(H3K27me3), HUH7(H3K27me3). tier=`imputed`, flagged (predicted, not measured).
- **S4 ChIP-Atlas**: INVESTIGATED, then DROPPED. Found 4 name-matches (THP1,U266,U937,RMGI, all
  already-partial cells) + rejected SNUC4 (C4-2 prostate collision). Deeper GEO audit revealed
  ChIP-Atlas pools MIX perturbed samples (THP-1 H3K27me3 pool contains HSV-1-infected samples) and
  ChIP-Atlas carries no CVCL -> per-sample treatment filtering required; marginal value (0 dataless
  lines helped). Operator dropped it. Scripts retained (download_chipatlas.py, chipatlas_match.py,
  chipatlas_candidates.tsv); 13.7MB raw subset removed (reproducible via download_chipatlas.py).
- **SKMEL28 REHABILITATED**: the MEL-745A tag that caused the drop was proven a **Cistrome parser
  artifact** -- Cistrome sample 43882 == GEO GSM838934, which GEO+ChIP-Atlas label cleanly "SK-MEL-28".
  SKMEL28 now H3K27me3 (direct) + ATAC (H3K4me3 substitute); H3K27ac still no data. 0/3 -> 2/3.

## Result (FINAL)
- Resolved **128 / 249** (cell,mark); unresolved **121**. No imputation of values anywhere.
- Per cell: **35 fully covered (3/3), 17 partial, 31 zero-coverage**.
- The 31 zero-coverage cells are obscure cancer lines with NO ATAC/H3K27ac/H3K27me3 in ENCODE,
  Cistrome, EpiMap, or ChIP-Atlas under any Cellosaurus synonym -> Phase-2 Checkpoint-3 exclusion candidates.
- Genome builds NOT yet lifted over (mix of GRCh38/hg38/hg19); pyliftover required before any Phase-2 join.
