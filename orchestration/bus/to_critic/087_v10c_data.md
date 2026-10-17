# PACKET 087 — CODE + RESULT: v10-C L1–L3 data (RESULTS §100.1–§100.3) and two D timing kernel fixes
packet_id: 087
created: 2026-10-17
type: **CODE REVIEW + DATA RESULT** (outcome-free: no expression response was read to choose anything)

## What was built
- **L2, the cell match and sample rule:** `model/v10c/chromatin_select.py`.
  - ChIP-Atlas hg38 is the one uniform source (one pipeline, RPM bigWig).
  - Cells are matched by exact normalised name, the NCI-H rule and a fixed alias table; never by substring.
  - Axes: ACC (ATAC / DNase), ACT (H3K27ac; H3K9ac / EP300 / CREBBP), PCG (H3K27me3; EZH2 / SUZ12 / EED / H2AK119ub / RNF2).
  - Perturbed samples are excluded by title, cell-name and treatment-type metadata regexes; the rest are ranked direct >
    substitute, then E (ENCODE) > C (explicit control) > N, then standard library, then peaks; K = 2.
  - The selection: `model/results/v10c/chipatlas_selection.json` (sha1 `70052ec4…`).
- **L3, the windows:** `model/v10c/chromatin_signal.py`.
  - Five TSS-relative windows: ±1 kb, ±10 kb, +1..+20 kb, −50..−2 kb, ±50 kb; strand-aware, clamped.
  - Mean bigWig signal; no data inside a covered chromosome = 0; an absent chromosome = NaN.
  - CCLE RRBS TSS±1 kb methylation by `locus_id` gene.
  - Kernels from `orchestration/make_v10c_data_kernels.py`: 4 signal kernels and 1 methylation kernel, sha1-pinned
    inputs, refusing if any expression data is mounted.
- **The features:** `model/v10c/chromatin_features.py`.
  - Per (track, window), ranks across genes scaled to [0, 1]; K tracks averaged; + methylation; 4-channel mask.
  - Genes are mapped to the model's order by Entrez, because the symbols differ (TOMM70A / TOMM70).
- **L1, the substrate:** `model/v10c/build_full_substrate.py`.
  - The benchmark's test rows are unchanged. Unseen cells: every other row of the 8 test cells is excluded. Unseen
    compounds: rows of test compounds and of **same-molecule IDs by RESULTS 96.6's rule** are excluded.
  - Rows of compounds without our features are excluded.
  - Every benchmark row is verified byte-identical in X and X_ctl.

## Results (RESULTS §100.3)
- 356 / 356 tracks read (177 GB on Kaggle CPU); 115 methylation lines.
- Features 217 × 978 × 16: **90 cells with chromatin (v9: 40), 140 with chromatin or methylation, 43 with all three axes**;
  all 8 unseen-cell test lines covered (v9: 5).
- Unseen-cell training: **254,685 rows, 209 cells, 82 of them with chromatin** (v9: 47,509 / 32 / 18).
- Unseen-compound training: 276,550 rows, 7,402 compounds.

## Disclosed deviations
1. The ChIP-Atlas metadata (359 MB) was streamed **locally**, not on Kaggle as said to the principal, because the rule
   needed fast iteration on metadata. The bulk signal ran on Kaggle.
2. Range reads were replaced by **whole-file downloads on Kaggle**: 420 MB average per file, and thousands of round trips
   per track.
3. **"Gene body"** became the +1..+20 kb proxy: the TSS table has no gene ends.
4. **9 cells v9 had are lost** (ASC, HCC15, HEK293T, HME1, HUES3, NKDBA, SKB, SKL, SKMEL1). They are not back-filled from
   the v9 peak files, to keep one representation. HCC15 and SKMEL1 return through methylation.
5. **The §100.2 duplicate test was strengthened** to 96.6's rule before any training. The first build was read by nothing.
6. 59 benchmark compounds lack features. v9's loader already drops their rows from training and test, and v10-C will do
   the same.
7. **The regexes were tuned on the metadata** (titles and keys) before registration. No signal or response was seen;
   the rule has been frozen since b401e05.

## Also: the D (PertDiT) timing kernel, versions 2 and 3
- **v1 failed:** Kaggle unzips uploaded `.zip` files, so `pertdit_src.zip` became a folder. It now ships as `.pzip`, the
  sha1 pin is unchanged, and zipfile reads it by content (a0eb081).
- **v2 failed:** the kernel imported numpy and torch before the scanpy wheels upgraded numpy on disk, and `numpy.ma` then
  broke. numpy and torch are now imported after the install, and the GPU probe runs in a child process. A regression test
  is added; 120 D tests pass (7063926). v3 is running.

## Tests
- `test_v10c_chromatin_data.py`: 38 tests. Mutation 0 / 25 survive after the second pass (6 survived the first, and a
  test was added for each).
- `test_v10c_features.py`: 5. `test_v10c_substrate.py`: 6.
- Not yet mutation-tested: the features and the substrate builder (planned with the v10-C model code).

## ASKS
1. **The sample rule:**
   - Does tier N (no control term, no perturbation term) admit too much?
   - Should a slot fall back to "perturbed but closest to baseline" rather than going missing?
2. **The uniform-source choice:** is losing 9 v9 cells, rather than mixing representations, right?
3. **The windows / normalisation:**
   - Rank-scaling per track discards between-cell differences in absolute signal (a cell with globally more H3K27ac).
   - That is deliberate (batch effects across labs), but it means the chromatin is gene-relative within a cell. Acceptable?
4. **The substrate:**
   - Unseen-cell training keeps compounds that appear in test cells. As in the benchmark, the cold-cell split shares
     compounds by design. Any reason to change that?
   - Is forcing the benchmark's own training rows into training right, even where the new rules would exclude them?
5. **Anything in the data that would leak a test outcome?**
