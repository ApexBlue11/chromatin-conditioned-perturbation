# PACKET 094 — CODE: T-E (RESULTS §101 / §101.1) before its one run
packet_id: 094
created: 2026-10-21
type: **CODE REVIEW** (T-E has not run)

**The files:** `model/v10c/te_tests.py` and `model/v10c/test_v10c_te.py` (commit 059875a).

## What it does, mapped to §101 / §101.1
- **Rows:** `split_cold_cell_1` training rows only.
- **`signed_contrast`:**
  - a row's delta minus the mean of its other-compound plate-mates whose class is not in {epigenetic, bet_inactive_control,
    sirtuin_or_natural_product};
  - it needs ≥ 2 such mates;
  - excluded-class rows are themselves contrasted, against the allowed mates.
  - Mate loss and dropout per class are reported, against the mate counts before exclusion.
- **Profiles:** `class_profiles` takes the mean contrast per (cell, gene) over a class's rows in the cell, with ≥ 5 rows.
  REF is the non-excluded classes. H-E3's I and R are the profiles of max(±contrast, 0) over all of the cell's rows.
- **Basal:** rank-normalised mean control expression of the cell's distinct training controls (T-A's basal).
- **`per_cell_r`:**
  - genes with the matched mark (mask = 1) and finite U / LCO means;
  - the partial Pearson between m and U, removing an intercept, basal (or decile dummies within the silent tercile for
    H-E3), the gene's LCO mean of U and the gene's LCO mean of m;
  - cells with < 30 eligible genes are skipped.
- **Cell eligibility:** the matched mark covers > 50 % of genes, the profile is finite, and basal exists.
- **`evaluate`:**
  - the mean of r_c with a cell-cluster bootstrap (20,000 draws, `default_rng(0)`; 98.33 % for primaries, 95 % reported);
  - Δspec paired by cell against REF (H-E1 / H-E2) or against R (H-E3);
  - 200 gene-permute and 200 cell-swap nulls;
  - the planted calibration (β = 0.02 / 0.05 / 0.1 SD of U, with a 4,000-draw bootstrap per β, which is an implementation
    choice);
  - `decide`.
- **`decide`:** the predicted sign with the CI excluding 0, **and** Δspec likewise, **and** the real mean beyond the
  cell-swap 95th percentile in the predicted direction.
- **The hypotheses:**
  - **H-E1:** DNMT inhibitors × `METH:tss_1kb`, + ;
  - **H-E2:** BET inhibitors × `ACT:tss_50kb`, − ;
  - **H-E3:** silent-gene induction × `ACC:prom_1kb`, + , with decile basal.
- **Reported:**
  - JQ1(−) × ACT broad;
  - HDAC (all, and `hdac_core`) × ACC / ACT at prom and broad;
  - silent induction × PCG / METH (−) and × ACC broad / ACT prom (+);
  - BET × ACT prom (narrow against broad);
  - `bet_with_apabetalone`.
- **Read once:** the reader refuses if the output file exists.

## Tests (13 / 13; mutation 12 / 12 killed, the script asserting the unmutated suite passes first)
- the mate-excluding contrast by hand;
- the partial r removes a confounder;
- the LCO mean by hand;
- decile dummies; the derangement;
- a planted cell-specific effect passes, and noise doesn't;
- **a gene-generic effect doesn't pass;**
- masked genes never enter, including masks that differ by cell, with a by-hand `per_cell_r`;
- negative-sign hypotheses use the lower tail;
- `class_profiles`' minimum rows;
- `decide`'s three conditions, both signs;
- the cell-swap threshold is the 95th / 5th percentile of the null draws.

## ASKS
1. Is the code faithful to §101 / §101.1?
   - Is skipping cells with < 30 eligible genes acceptable? It wasn't registered.
   - Is the 4,000-draw bootstrap inside the calibration acceptable?
2. **The gene-permute null permutes m within mask = 1 genes**, and recomputes the LCO m mean from the permuted values. Is that
   the right null for a partial correlation that includes that covariate?
3. Anything to change before the one run?
