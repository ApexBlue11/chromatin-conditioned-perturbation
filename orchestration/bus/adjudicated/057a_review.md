# REVIEW OF PACKET 057, ADDENDUM: clearance check on d9b8e52 (93.15a, 93.16)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: d9b8e52

**Review 057's C1–C4 are implemented as amended.**
- **Tests:** `chromatin_genegeneric.py` = a862fdd5. I reran `test_chromatin_genegeneric.py` under `.venv-cuda`: **7 passed**.

**93.16 is right for ATAC and H3K27ac, and I reproduced it exactly. But it misses the most affected channel: H3K27me3 carries the
same tie-break block in about 90 % of its entries** (C1).
- **Why the run can't go as is:** the tie-corrected arms leave H3K27me3 uncorrected. The H3K27me3 column of N1 and E+N1 is then
  mostly a non-chromatin per-gene code, which bears on what 93.15 can license.
- **Not cleared until C1 is in.** C2 and C3 are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | measurement | **H3K27me3 has the same artefact, larger.**<br>**The cause:** `step12_h3k27me3_coverage.py:118–121` ranks every covered entry jointly with the same `vals.argsort().argsort()` as step10. So every tied raw value (zero coverage) gets a distinct rank. 93.16's *"H3K27me3 (bigWig coverage, step13) is not affected"* is contradicted by `E_final` itself.<br>**The evidence:**<br>(a) **Every channel is an exact rank grid.** All 25,402 covered H3K27me3 values are distinct and equal rank / (N − 1), as for ATAC and H3K27ac.<br>(b) **A tie block's signature is per-cell contiguity in value order,** because ties are broken in flattened (cell, gene) order. Below value 0.88, **22,353 entries from 26 cells form only 475 runs of cell id.** Each cell occupies its own value band, in cell-row order: A549 lowest, then A673, CD34, H1299, HCC15, HCT116, HEK293T, HELA and so on. Above the boundary, interleaving across cells jumps from 0 to **163 cell changes per 200 entries**. Real coverage can't put cells in alphabetical value bands.<br>(c) **Inside those bands, values follow gene index:** Spearman 0.999 in A549, A673, CD34, H1299, HCC15, HCT116, HEK293T, HELA, HEPG2 and HL60.<br>**The block estimate is Z ≈ 22,778 (89.7 %, b ≈ 0.897).**<br>**Per-cell share:**<br>- dev cells: SKBR3 94 %, LNCAP 93 %, HL60 80 %; HEK293T 99 % and VCAP 99 % (both on the failed list);<br>- fitting cells rank_normal **keeps**: A549 86 %, HCT116 94 %, HEPG2 75 %, MCF10A 96 %, PC3 94 % (JURKAT 2 %);<br>- the four failed tracks I checked (HEK293T, VCAP, HELA, PHH) are 99–100 %, consistent with "raw peaks < 10".<br>**What follows:**<br>- rank_normal's failed-track exclusion (§91.8) drops only the near-empty tracks. **Every other H3K27me3 track in §91, Stage 1a, N1 and E1's clean encoding is mostly tie-break code.**<br>- N1tie and E+N1tie, described as tie-corrected, correct two of three marks.<br>- For SKBR3, the T4/E2 "help" cell, v9's chromatin is ATAC 100 %, H3K27me3 94 % and H3K27ac 9 % artefact. | **Before the run:**<br>(i) **Add H3K27me3 to `TIE`.** Take Z₃ exactly from step12's raw tensor (`RAW`) or its log (`nonzero=` per cell, Σ(covered − nonzero)), if either exists. If not, use the `E_final` boundary (22,778 by my estimate, from the cell-run break), recorded as an estimate. Keep main's assertion.<br>(ii) **Correct 93.16:** H3K27me3 is affected, at ≈ 90 % of entries, with the per-cell table above. It is the largest of the three.<br>(iii) **Decide, before the run, which features carry the reading.** The question is whether the gain is *chromatin*, and §91's features are ≈ 21 / 20 / 90 % tie-break codes by mark. Either:<br>- **(a)** read on §91's features as registered, worded *"§91's encoded chromatin features (≈ 21 % / 20 % / 90 % of entries tie-break codes in ATAC / H3K27ac / H3K27me3)"*, with the fully corrected arms reported; or<br>- **(b)** make the tie-corrected features the arms of record: N1tie against FB with a perm null on the corrected means, and E+N1tie against E with E matched to the corrected means. Then keep §91's N1 − FB only as the harness check.<br>Only (b) answers "is it chromatin?". Both are outcome-free now. |
| 2 | MINOR | overreach | **93.16's closing line, *"with clean features from the same samples (93.13), the gene-local gain is null too. So the artefact is not what hid a chromatin signal in that form"*, covers accessibility only.** H1's c93 features are all ATAC-derived. The histone channels (H3K27ac ≈ 20 % and H3K27me3 ≈ 90 % artefact) have never been tested clean. | *"…So for accessibility, the artefact is not what hid a gene-local gain. The histone channels have not been tested free of it."* |
| 3 | MINOR | code | **If C1(iii)(b) is chosen, E must be matched to the arms it serves.** `variant(..., ref_enc='rank_normal')` quantile-matches E to the **uncorrected** mark means, and E+N1tie reuses that E. For the corrected arms of record, E's marginals should come from the corrected means (`ref_enc='rank_normal_tie'`), with an E_tie spec and E+perm_tie arms. Under (a) the current code is fine. | Under (b): add `E_tie` (ref_enc = `rank_normal_tie`), `E_tie+N1tie` and `E_tie+perm_tie_d`, plus `N1perm_tie_d` for the capacity null, and one test that `rank_normal_tie` ties all three marks' blocks. Regenerate the pin. |

## Answers to the asks

**Review 057's C1–C4: confirmed implemented.**
- **E:** E = [b, x1, x2, x3] with x1 = mean b, x2 = sd b and x3 = (mean b)² over `src` (h excluded). Each is quantile-matched to
  that mark's mean vector by `_quantile_match` (stable argsort) and present where the cell has the mark (the tiled `has` mask).
- **The incremental arms:** E+N1 and E+perm_d append the three mark means, so there are 6 new columns in each.
- **The permutation:** one per draw, `default_rng(9500 + d)`, applied to the [G, 3] mean matrix, shared across marks, in both
  N1perm_d and E+perm_d.
- **The reading:** HARNESS_FAULT → reading 1 (G_chr ≤ max G_perm) → 2a (Δ_chr|E ≤ max Δ_perm|E) → 2b (cells < 4) → 3, with the
  91.12 caveat in the output. This is 93.15a exactly.

**93.16, checked:**
- **Confirmed exactly for ATAC and H3K27ac:**
  - the step10 code (lines 133–136);
  - Σ(covered − nonzero_genes) from `E_peaks_log.txt` = 6,465 of 31,264 (20.7 %, b 0.2068) and 6,726 of 34,195 (19.7 %,
    b 0.1967). The count at or below b matches; H3K27ac matches at the half-step threshold, with float32 rounding at the block
    edge;
  - every dev-cell share in the table;
  - Spearman(value, gene index) of +0.999 (HEK293T ATAC), −0.997 (HL60 ATAC) and +0.999 (HL60 H3K27ac);
  - the inheritance (distinct values survive `rankdata`).
- **Where the ordering is scrambled:** SKBR3's 100 % ATAC block isn't gene-ordered (ρ 0.18), and neither is LNCAP's (0.15),
  but both are still pure tie-break.
- **Incomplete:** H3K27me3 (C1).
- **The canonical gene order** (`pathway_landmark_genes.txt`: DDR1, PAX8, RPS5, …) isn't alphabetical. Whether it correlates
  with any biology is untested. If it doesn't, the gene-ordered blocks act like one fixed permutation, which adds noise rather
  than a spurious gene-generic signal.

**Clearance: not yet.** Cleared once C1(i)–(iii) are committed, with C3 if (b) is chosen. C2 is wording.

## What I checked and found sound

- **The code:** `chromatin_genegeneric.py` a862fdd5 in full (`_quantile_match`, `variant`, `install_tie`, `run`, `reading`,
  `read`, and main's TIE assertions), and the 7 tests rerun.
- **The RESULTS text:** 93.15a and 93.16, against the code.
- **The step scripts:** step10 lines 120–138 and step12 lines 95–122 (the same joint `argsort().argsort()`); step13's
  channel map.
- **`E_final` / `E_final_mask`, all three channels:**
  - exact rank grids;
  - block sizes against the log for ATAC and H3K27ac;
  - for H3K27me3: the cell-run structure, the boundary, the per-cell shares and the within-band gene-order correlations.
- **`E_final_provenance.json`'s** failed-H3K27me3 list, against the per-cell blocks.

## What I could not assess, and why

- **H3K27me3's exact Z.** Step12's raw tensor and log aren't in the repo. My 22,778 comes from where the cell-contiguous run
  structure breaks. The boundary is sharp (0 to 163 cell changes per 200 entries), but the exact count may differ by a few
  entries.
- **Whether the canonical gene order carries biology.** Not tested. It matters only for the gene-ordered blocks.
