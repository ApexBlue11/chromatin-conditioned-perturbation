# PACKET 054 — CODE REVIEW: §93 Stage 1b data module (H1's features + 93.8 split halves), outcome-free
packet_id: 054
created: 2026-10-07
repo_commit: 34a1836
type: **CODE REVIEW** (the data kernel is already running on free Kaggle CPU; it reads no LINCS response, so a defect found
here means a rerun, not a contaminated read)

## OBJECTIVE
Build H1's per-(cell, landmark gene) accessibility features (F_prom, F_enh, F_reg) and per-(cell, motif) motif accessibility
from the public peak files v9's own assembly rule selects, plus 93.8's split-half reliability report. This is the data step
of 93.3, 93.5 C1/C5 and 93.8. It must reproduce step10's sample choice and freeze its outputs by sha1 before H1's test is
written.

## FUNCTION (`model/v9/chromatin_features93.py`)
- **Network:** `fetch_to_file` / `fetch` stream to `cache/<sha1(url)>`. No other function opens a URL.
- **Selection (unchanged from W31, step10 lines 36–60 and 88–130):**
  - `select_atac_samples`: Cistrome → GSM → SRX, the first 6, ChIP-Atlas bed05; ENCODE: the first 4 ENCSR, the first
    narrowPeak accession of step10's query; epimap → none;
  - only cells whose `E_peaks_log.txt` ATAC line has N > 0.
  - `parse_bed` is step10's.
- **Merging:** `merge_peaks` = `pyranges.PyRanges(df).merge()`, slack 0 (bedtools `-d 0`, verified on Kaggle).
  `merge_intervals_reference` is the same rule in pure Python; the local tests use it.
- **Windows:** `compute_windows` (F_prom = bp overlap with [TSS − 1,000, TSS + 1,000) / 2,000; F_enh = Σ (length / 1,000) ·
  exp(−d / 10,000) over merged peaks with midpoint 1–50 kb from the TSS that do not overlap the promoter window).
- **Motifs:**
  - JASPAR2024 CORE vertebrates, `all_versions=False` (879 motifs on Kaggle);
  - `MOODS.tools.log_odds(counts, bg, 0.01)` with bg = hg38 base composition (`genome_background`);
  - the RC matrix via `MOODS.tools.reverse_complement(fw, 4)`;
  - `threshold_from_p(·, bg, 1e-4)`; one `MOODS.scan.Scanner(7)` per process.
  - `motif_hits`: a peak carries motif t iff the forward or RC matrix hits.
  - **MA_raw[c, t]:** the share of c's merged peaks with midpoint within 50 kb of a landmark TSS (`near_landmark_peaks`) that
    carry t.
  - **MA_std:** per motif across kept cells with has (ddof 1).
- **F_reg[c, g]:** the mean over g's CollecTRI regulators with a JASPAR motif of sign(weight) × MA_std[c, TF].
  - JASPAR names are split on `::`, case-insensitively.
  - A TF with several motifs takes the mean of their MA_std.
  - CollecTRI comes from `dc.op.collectri(organism='human')` (decoupler 2.2.0).
- **has[c]:** False iff c has no intervals, or its merged union gives F_prom > 0 in no gene. Its features are then 0.
- **93.8 halves (`_compute_halves`):**
  - **Which cells:** cells with has and ≥ 2 usable samples (parsed, ≥ 1 interval).
  - **The split:** A = usable positions 0, 2, 4; B = 1, 3, 5.
  - **Per half:** `cell_features`; F_reg from the half's MA standardised with the **full-union** mean and sd.
- **`split_half_report`** (no gate, prints no verdict):
  - rank_normal per (cell, feature) (§91's, imported from `chromatin_funnel`);
  - dev_h[c] = F_h[c] − the mean over the other has-cells of rank_normal(F_full);
  - Pearson and Spearman over genes with a TSS;
  - the baseline is the median over c' ≠ c of corr(dev_A[c], dev_B[c']); excess = own − baseline;
  - MA likewise, over motifs.
- **Outputs:**
  - `c93_features.npz`: cells, genes, F_prom, F_enh, F_reg, has, has_tss, MA_raw, MA_std, motif_ids, motif_names;
  - `c93_halves.npz`, `c93_split_half.json`;
  - `c93_manifest.json`, recording:
    - per URL: bytes, sha1, intervals, genes with a peak, or the error;
    - n_tracks_v9 against n_tracks_now, and the mismatches;
    - per-cell counts;
    - the motif IDs and rules;
    - the genome sha1 and background;
    - CollecTRI's call, version, date, rows and sha1;
    - the package versions;
  - `C93_FEATURES_COMPLETE.json`.
- **`selftest()`** (Kaggle only; the kernel aborts on failure):
  - (a) pyranges = reference on overlapping, book-ended, 1-bp-gap and disjoint intervals, and on 2,000 random intervals.
  - (b) Real MOODS with CTCF `MA0139.2`:
    - 40 random 300-bp peaks: 10 carry the consensus, 10 only its reverse complement, 20 nothing;
    - every planted peak is hit;
    - the RC-only peaks' best RC score = `max_score(fw)` at the planted position;
    - MA = (20 + chance hits) / 40 exactly.
  - (c) `build_features` end to end on a 3-cell fixture: real MOODS + real pyranges against the reference merge, identical
    arrays; has = [T, T, F]; F_reg = ± MA_std by hand; halves for CellA only.
  - `selftest_genome`: chr1 length 248,956,422, and chr1[0:10] = N × 10.

## WHO WROTE WHAT (disclosed)
- **W31 and W31b** (gemini-3.1-pro-high): both rounds called APIs that do not exist. They were replaced. Selection, parse_bed,
  windows, near_landmark_peaks, standardise and regulon_scores were kept.
- **W31c** (agy claude-opus-4-6-thinking): written from a brief whose API facts the PI verified from the MOODS and py2bit
  sources and a Kaggle probe kernel (`lincs-c93-probe`). It was stopped by the agy quota at 19 of 21 tests.
- **The PI then wrote about 200 lines** (the AGENT_REGISTRY entry lists them):
  - `split_half_report`, `_compute_halves`, `make_parallel_scan_fn`, `bench_scan`;
  - selftest (b)'s MA check and selftest (c), with its fixture;
  - main's genome fetch;
  - three tests, and the re-specified noise-baseline test.
- **Mutation check:** six mutants were each caught by a test: the merge rule, the baseline, the TSS mask, halves without has,
  F_enh with promoter overlap, and the half positions.

## WHAT WAS CONTROLLED
- **The kernel:** `orchestration/make_chromatin93_features_kernel.py` → `lincs-chromatin93-features`.
  - It pins the module (`9da979bb`), `chromatin_funnel.py` (`c56bb6f1`) and `score_dev.py` (`a9002bce`), and the seven step10
    inputs (dataset `lincs-c93-inputs`, sha1s in the builder output).
  - It pins the package versions the probe verified and the probe's docker image, and has internet.
- **Order:** pip, then pin checks, then `selftest()`, then main. main downloads hg38.2bit from UCSC, runs `selftest_genome`
  and prints a MOODS bench before the cells.
- **No LINCS response** array is mounted or read.

## RESULTS
None yet. The kernel is running. Its outputs will be committed by sha1 before H1's test is pre-registered in code.

## PRIOR RETRACTIONS IN SCOPE
- §93.1's VCAP "imputed ATAC" (review 050 C1).
- Review 053 C1: 93.7 item 2's gap sat inside the null calibration.

## ASKS
1. Does the selection reproduce step10 exactly? In particular: the ENCODE first-file rule, the Cistrome GSM→SRX setdefault,
   and the log-based cell filter.
2. Is MA as defined (share of near-landmark merged peaks with ≥ 1 hit on either strand, p < 1e-4, genome background,
   ps 0.01) a faithful reading of 93.3 item 2 and 93.5 C5?
   - A TF with several motifs takes the mean. Is that acceptable, or should it be the max?
3. Is the 93.8 split-half implementation faithful? Including:
   - the full-union standardisation for the halves' F_reg;
   - that rank_normal is applied to every cell's full union before averaging the gene mean.
4. Any defect that would make the frozen features wrong? Some of these would only show in the data:
   - ENCODE re-releases since July, which change the "first file";
   - files that are now empty;
   - chromosome naming.
