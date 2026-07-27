# Task B — Epigenetic Signal-Tensor Extraction Plan (for review before running)

Turn `coverage_report_phase2.tsv` (which *source/samples* per cell,mark) into the actual numbers:
**`E` = (83 cells × 978 genes × 3 marks)** + an **availability mask** + **per-(cell,mark) reliability
weights**, cell-id-indexed, gene axis in the frozen 978 canonical order.

## Pipeline
1. **TSS coordinates (978 genes):** MANE Select TSS (primary) / Ensembl canonical (fallback), matched
   by Entrez, in **GRCh38/hg38**. Source: a GENCODE/MANE annotation (small download, one-time).
2. **Windows (decided):** ATAC ±2 kb, H3K27ac ±2 kb (promoter/enhancer-proximal); **H3K27me3 ±10 kb /
   gene-body** (Polycomb domains are broad). Statistic: **mean signal coverage** in the window.
3. **Liftover (pyliftover):** keep TSS in hg38; for hg19 tracks (EpiMap, some Cistrome) lift the **978
   TSS coordinates hg38→hg19** (cheap — 978 points, not whole tracks) and query the hg19 bigWigs there.
4. **Signal read — the key efficiency choice:** use **pyBigWig to read only the 978 TSS windows from the
   *remote* bigWig URLs** (ENCODE/Cistrome/EpiMap all serve bigWigs by accession) — no multi-GB whole-track
   downloads. Per (cell,mark): query each resolved sample's bigWig at the 978 windows, average across samples.
5. **Cross-source normalization:** ENCODE (fold-change/p-value), Cistrome (RPKM), EpiMap (imputed) are on
   different scales → **quantile-normalize per mark across the 83 cells** (recommended) so the gate sees
   comparable values.
6. **Mask + reliability:** `E_mask` (83×978×3, real vs missing); `E_reliability` per (cell,mark) from the
   coverage confidence tier (direct 1.0 > substitute_strong ~0.8 > substitute_weak ~0.6 > tissue_type ~0.4
   > imputed ~0.5). Zero-coverage cells → mask all-missing, gate falls back to `x_base` (gate=1). Feeds the
   reliability-weighting the papers advocate.

## Open design choices (need your call before running)
- **Signal type for ENCODE:** "fold-change over control" (recommended, comparable dynamic range) vs "signal p-value".
- **Sample capping:** some (cell,mark) pools are large (HELA/H3K27ac = 26 samples). Query **all** and average
  (more robust, more remote reads) vs cap to top-N by QC. Recommend: all, since remote window-reads are cheap.
- **Normalization:** per-mark quantile across cells (recommended) vs per-track z-score vs raw.

## Scope / cost
- ~132 resolved (cell,mark) slots across ~52 cells; remote window-reads (not downloads) → tractable
  (minutes–low hours), storage ≈ just the annotation + output tensors (~1 MB tensors).
- Outputs → `phase2_assembly/outputs/`: `E_epigenetics.npy` (83×978×3), `E_mask.npy`, `E_reliability.tsv`,
  `epi_cell_index.json`, `epi_extraction_provenance.json`.

## Dependencies (RESOLVED)
- **Reader:** native Windows bigWig is impossible (no WSL/conda; pyBigWig/pybigtools have no
  Python-3.14 wheel). SOLVED via **`uv venv --python 3.12 .venv-epi`** + **`pybigtools 0.3.0`**
  (remote bigWig reads confirmed working). `pyliftover 0.4.1` for hg38→hg19.
- Still needed: a MANE Select / GENCODE TSS annotation for the 978 genes (small download).

## Track ACCESS strategy (verified 2026-07)
Cistrome exposes NO per-sample bigWig URL, so tracks are sourced as:
- **ENCODE (23 slots):** direct GRCh38 bigWig `…/files/{acc}/@@download/{acc}.bigWig` (fold-change/control).
- **EpiMap (11 slots):** direct imputed hg19 bigWig → liftover.
- **Cistrome (98 slots):** map Cistrome sample → GEO **GSM** → **ChIP-Atlas SRX** →
  `https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/{SRX}.bw` (hg38, so NO liftover). Confirmed
  working via pybigtools. **Coverage: 73/98 Cistrome slots map to ChIP-Atlas.**
- **25 Cistrome slots (19% of total) have NO clean track source** (their GSMs aren't in ChIP-Atlas;
  mostly ATAC for A375/H1299/NPC-family/AGS/NEU/NOMO1). Options: (a) fetch per-GSM GEO supplementary
  bigWig/bed, or (b) mask them (treat as missing at tensor build, model handles via availability mask).
