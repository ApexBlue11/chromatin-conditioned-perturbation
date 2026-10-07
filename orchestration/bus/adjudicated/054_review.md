# REVIEW OF PACKET 054 (with the 6b7ef9e addendum)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 08f2f9f (+ addendum 6b7ef9e)

**The selection reproduces step10. The 93.8 implementation is faithful.**
- **Inputs:** the kernel mounts `coverage_report_phase2.tsv`, the file step10 read, not `_final`.
- **The log filter:** the regex keeps `…/ATAC-seq src=…: N bed track(s)` with N > 0 and drops `0 bed tracks (masked)`.
- **Cistrome:** id → GSM → SRX through `gsm2srx.setdefault` (the first SRX per GSM), the first 6 non-null, ChIP-Atlas bed05.
- **ENCODE:** the first 4 ENCSR, then step10's narrowPeak query with `[:1]`.
- **Other identities:** epimap → none; `parse_bed` is step10's verbatim; and the TSS file, ChIP-Atlas, ENCODE GRCh38 and
  hg38.2bit all use `chr`-style names.

There is one MAJOR point: as defined, MA (and so F_reg) would mostly measure **peak width** (C1). That needs a rerun of the data
kernel. Since no response has been read, the rerun is outcome-free. C2 and C3 are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | measurement | **MA_raw is the share of merged near-landmark peaks carrying ≥ 1 hit, and that share scales with peak length.** At p < 1e-4 on both strands, a motif hits a random 500-bp interval with P ≈ 1 − e^(−0.1) ≈ 0.10, and a 1,000-bp interval with P ≈ 0.18. Merged unions widen with the number and depth of samples (1 sample for some cells, 6 for others, up to 20 listed for CD34). Halves are narrower than full unions. So MA_raw differences across cells will be dominated by width, not TF accessibility. **That propagates to F_reg:** a cell-level shift c₀ in MA_std across most motifs enters F_reg as c₀ × (mean sign of g's regulators), the **same gene pattern in every cell** with a cell-specific coefficient. After rank-normal, deep and shallow cells get **opposite-signed** versions of a network-structure pattern. **The split-half for MA would also reward technical reproducibility:** both halves of a cell share its lab's peak widths. This is why chromVAR scores against GC- and width-matched backgrounds. | Scan **fixed-width windows**: [mid − 250, mid + 250) around each merged near-landmark peak's midpoint, clipped to the chromosome. MA = the share of windows with ≥ 1 hit. Use the same in the halves. Record per cell, in the manifest, the mean MA_raw across motifs (a depth diagnostic, reported). It should no longer track the number of samples. A one-line 93.5/93.8 amendment ("fixed 500-bp windows at merged-peak midpoints; chromVAR-style width control") and a selftest (identical sequences at two widths give the same MA after recentring) would close it. Rerun the data kernel. |
| 2 | MINOR | provenance | **Three per-cell quantities the read will need aren't recorded:** per cell, the number of usable samples, the median merged-peak width, and the number of genes whose F_reg is non-zero (genes with ≥ 1 CollecTRI regulator that has a JASPAR motif). Without them, coverage or depth can't be ruled out as an explanation of any H1 result, and F_reg's zeros (no regulator with a motif) can't be told from small values. | Add them to `c93_manifest.json`'s per-cell counts. Report F_reg's gene coverage beside H1's reading. |
| 3 | MINOR | scope | **The addendum's 32 cells and the earlier 19 of 25 are both right for different populations, and the population matters for standardisation.** 19 of 25 was the bundle's cells (review 050 C1). The log rule over the 83-cell index gives 32, adding 13 cells outside the split bundle (ASC, ASC.C, HS27A, HUES3, HUVEC, LOVO, MCH58, RKO, SKB, SKL, SKL.C, SW480, U266) as well as test cells. Every kept has-cell enters MA's cross-cell mean and sd and the split-half's other-cell mean. These are inputs only, so there's no outcome contamination. But the standardisation reference isn't the fitting set. | Record in 93.5 C1 / 93.8: *"features are built for all 32 cells by the log rule; MA_std and the split-half's other-cell mean use all kept has-cells (inputs only); H1's fitting cells and N1 use only the bundle's covered dev-train cells, as T1 does."* The dev cells as stated in the addendum: HEK293T, HL60, LNCAP, SKBR3 (to be re-checked by has) and U937; VCAP excluded. |

## Answers to the asks

**Ask 1 — yes, in every respect I could check.**
- **The ENCODE first-file rule:** the same query and `[:1]`.
- **The Cistrome setdefault:** the first SRX per GSM, in `chip_atlas_human_epi.tab` order.
- **The log-based filter:** tracks > 0.
- **The one rule that can't be exactly reproduced:** ENCODE's search order and releases may have changed since July, and step10
  never logged the accessions it used. Record each chosen accession (the manifest's per-URL records do), and compare
  n_tracks_v9 against n_tracks_now (it does). A per-file accession comparison isn't possible.

**Ask 2 — faithful to 93.3 and 93.5's text, but the text inherited C1's width confound. Fix it as in C1.**
- **The MOODS settings** (log-odds with ps 0.01 on counts, genome background, p < 1e-4, either strand, a peak counted once) are
  standard.
- **Mean over a TF's several motifs:** keep the mean. A max would select the noisiest motif.
- **Dimers split on `::`:** they count for both partners. Acceptable, and disclosed in the manifest.

**Ask 3 — yes, faithful.**
- **Standardising the halves' F_reg with the full union's mean and sd** is the right choice. Per-half statistics would need
  halves for every cell.
- **The gene mean:** rank-normal is applied to each other cell's full union before averaging, as 93.8 says.
- **The correlations:** TSS genes only, with the mismatched-cell baseline and excess as amended.
- **One consequence of C1:** with fixed-width scanning, the MA split-half measures motif-content reproducibility rather than
  width reproducibility.

**Ask 4 — data-side defects to watch.**
- **ENCODE re-releases:** see Ask 1.
- **Empty files:** recorded and skipped.
- **Chromosome naming:** consistent (`chr`). Alt and random contigs are harmless, because TSSs and the near-landmark filter
  restrict to chromosomes with landmark TSSs, and `dropped_chrom` counts peaks on chromosomes missing from the genome.
- **SKBR3:** v9's log had 0 genes with a peak from 3 tracks. A coordinate-build mismatch in those files is the likely cause.
  `has` will settle it. If SKBR3 has no promoter peaks now, H1 has 4 dev cells, and its cell conjunct (≥ 4 of 5) must say so
  before the test is written.

## What I checked and found sound

- **Selection:** `select_atac_samples`, `get_encode_files` and `parse_bed` against `step10_extract_peak_tensor.py:30-128`; the
  log regex and gsm2srx in `build_features`; the kernel's mounted inputs (`coverage_report_phase2.tsv`).
- **Merging:** `merge_intervals_reference` (book-ended merge, 1-bp gap kept) and `merge_peaks` (pyranges 0.1.4, slack 0).
- **Features:**
  - `compute_windows` (±1 kb bp overlap / 2,000; enhancers with midpoint 1–50 kb, not overlapping the promoter window,
    weighted exp(−d / 10 kb));
  - `near_landmark_peaks`;
  - `build_matrices`, `make_scanner` and `motif_hits` (forward or RC);
  - `standardise_across_cells` and `regulon_scores`;
  - `cell_features` (has rule, sequence clipping, upper-casing).
- **Split-half:** `split_half_report` (rank-normal, other-cell mean over has-cells, TSS mask, baseline over c' ≠ c, MA on raw
  shares).
- **Kernel:** its pins and package versions.

## What I could not assess, and why

- **The selftests and the bench.** They run on Kaggle only. I read them.
- **The running kernel's outputs.** None are committed yet.
