# Epigenetics Branch — Chromatin-State Gate (ATAC / H3K27ac / H3K27me3)

## Purpose
Provide, per LINCS cell line, the epigenetic signals that **gate** baseline CCLE
expression before pathway projection:
`gate[cell,gene] = sigmoid(MLP_epi([ATAC, H3K27ac, H3K27me3]))`,
`x_mod[cell,gene] = x_base[cell,gene] * gate[cell,gene]`.
This branch resolves, for each of **83 cell lines × 3 marks (= 249 pairs)**, the best
available data source — with **no value imputation anywhere**. Join to the other branches
is by `cell_id` (see `outputs/epigenetics_cell_index.json`).

## Deliverable
**`outputs/coverage_report_final.tsv`** — 249 rows, one per (cell, mark), with an explicit
confidence schema: `status, resolution_tier, assay_used, is_primary_assay, source_used,
genome_build, identity_confidence, contamination_status, model_confidence_flag,
sample_ids_used, sample_ids_dropped, notes`. Companion: `outputs/dropped_contaminant_samples.tsv`
(every removed sample) and `outputs/coverage_final_methodology.md` (decisions log).

## Sources explored, and what each contributed
| source | role | how matched / verified |
|---|---|---|
| **ENCODE** | primary (direct measurement) | exact `biosample_ontology.term_name` equality; genome GRCh38 |
| **Cistrome DB** | curated ChIP/ATAC tracks | word-boundary name match; **CVCL identity + Cellosaurus ancestry** contamination audit |
| **EpiMap** | imputed tracks | biosample name match, `perturb`-null only |
| **ChIP-Atlas** | S4 gap-fill | investigated then **dropped** (see `chip_atlas/README.md`) |
| **Cellosaurus** | verification backbone | CVCL resolution + synonym lists + 5-hop ancestry walk |

## Pipeline (scripts/, in execution order)
1. **`step01_audit_v5.py`** — base pipeline. Roster = 83 union `trt_cp` cells; per (cell,mark)
   checks ENCODE→Cistrome→EpiMap. Result: 91 resolved (79 clean + 12 flagged), 158 unresolved.
2. **`step02_audit_v6.py`** — adds NPC.CAS9 pre-authorized sharing, alias re-matching, and
   substitute assays (H3K4me3/DNase→ATAC; H3K9me3→H3K27me3). Produced 24 new alias resolutions
   + 10 substitute rows — but did **not** run the contamination pipeline on the alias matches.
3. **`step03_verify_v6.py`** — independent re-verification: row counts, the 12 contaminated rows
   untouched, no v5-resolved row rewritten, and mechanism-classification of the 24 new resolutions
   (19 generic tissue-type, 3 name-based, 2 pre-authorized).
4. **`step04_classify_v6_matches.py`** — item-8 dominant-identity screen (sum-of-CVCL-value) on
   every new pool + substitute pool; exposed the tissue-type matches resolving to the **wrong**
   cell line (e.g. SKB/SKL/SKL.C → AT6BR lymphoblastoid, not muscle).
5. **`step05_retroactive_classify_v6.py`** — the contamination classification v6 skipped: flagged
   all 22 alias matches PENDING with raw records (`outputs/flagged_raw_records_v6.json`).
6. **`step06_resolve_gap_cvcls.py`** — resolves each gap cell's Cellosaurus CVCL + synonyms
   (`data/gap_cells_cellosaurus.json`) — the verification backbone for gap-fill.
7. **`step07_gapfill_s1s2s3.py`** — S1 Cistrome synonym re-search (CVCL-verified), S2 live ENCODE
   re-query, S3 EpiMap imputed. Verification blocked bad fills (SNUC4→LNCaP-C4-2). Net: +4 EpiMap.
8. **`step08_build_coverage_final.py`** — **canonical builder.** Applies all operator decisions and
   emits `coverage_report_final.tsv` + `dropped_contaminant_samples.tsv` + consistency assertions.
   (ChIP-Atlas S4 is in `chip_atlas/`, investigated then dropped.)

## Key decisions (full log in `outputs/coverage_final_methodology.md`)
- **12 contaminated rows → drop + re-aggregate.** Dominant identity was always correct; contaminants
  were a few (often a single explicitly-multiplexed) samples. 11/12 kept a clean remainder; the drop
  list (`dropped_contaminant_samples.tsv`) maps to the Cellosaurus-audited contaminants.
- **19 generic tissue-type matches:** tissue-appropriate ones kept + flagged (`HUES3→hESC, MCH58→fibroblast,
  NKDBA→kidney`); wrong-tissue ones (`SKL/SKB/SKL.C`) **rebuilt** from a clean skeletal-muscle pool
  (AT6BR lymphoblastoid + a DUX4-perturbed sample excluded).
- **Dot-suffix inheritance:** `.C` = lot variant (ASC.C←ASC, SKL.C←SKL); `NPC.TAK←NPC` flagged
  `possible_donor_difference` (TAK not a documented perturbation, but a distinct line); `NPC.CAS9←NPC`
  pre-authorized (Cas9 chromatin-neutral); **MNEU.E** stays unresolved (terminally-differentiated,
  NOT_DEFENSIBLE). NPC-family H3K27ac/ATAC inherit NPC's *cleaned* pool.
- **SKMEL28 rehabilitated:** its Cistrome "MEL-745A contamination" was proven a **parser artifact**
  (sample 43882 = GEO GSM838934 = cleanly "SK-MEL-28", per GEO + ChIP-Atlas). 0/3 → 2/3.
- **Briefing vs disk discrepancy** on MULTIPLEXED: per-sample re-audit (live Cellosaurus) confirmed the
  on-disk GENUINE_CONTAMINATION labels; no row is a whole-pool multiplexed experiment.

## Final coverage (asserted by the builder)
- **Resolved 128 / 249** (no imputation): direct 91, tissue-type 18, substitute 8, related-line 7, imputed 4.
- **Unresolved 121.** Per cell: **35 full (3/3), 17 partial, 31 zero-coverage.**
- The **31 zero-coverage** cells (obscure lung/colorectal/ovarian/endometrial lines) have **no
  ATAC/H3K27ac/H3K27me3 in any source under any synonym** → Phase-2 **Checkpoint-3** exclusion candidates.

## Known-open items carried to Phase 2
- **Genome-build harmonization / pyliftover:** builds are a mix of GRCh38/hg38/hg19. **Deferred** —
  there is no coordinate/signal data to lift yet (tensor not built; only sample IDs are resolved).
  pyliftover belongs in Phase-2 signal extraction at gene TSS. Recorded here as a prerequisite.
- The 31 zero-coverage cells (fill-vs-exclude, signature-count impact).

## Folder layout
- `scripts/` — step01–step08 pipeline (authored to run from the branch root; paths are relative to it).
- `chip_atlas/` — the dropped S4 investigation (scripts + candidates + README).
- `outputs/` — coverage reports (v5, v6, reclassified, **final**), audits, classifications, candidates,
  logs, methodology, provenance, cell index, dropped-sample audit, flagged raw records.
- `data/` — external caches: Cistrome JSON (34 MB), EpiMap TSV, ENCODE cache, Cellosaurus gap map,
  disputed records, derived 83-cell roster.
- Status: **CLOSED.** Phase-2 Checkpoint-3 resolved: the 31 zero-coverage cells are **kept** (not
  excluded) via a per-mark availability mask + reliability weights (no imputation). Phase 2 then
  patched +4 property-matched substitutes → `../phase2_assembly/outputs/coverage_report_phase2.tsv`
  (132/249) and built the numeric epigenetic tensor `E` from these decisions. See
  `../phase2_assembly/report.md`.
