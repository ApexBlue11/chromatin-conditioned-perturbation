# PACKET 057 — PRE-REGISTRATION + CODE REVIEW: §93.15, is the gene-generic chromatin gain chromatin-specific?
packet_id: 057
created: 2026-10-08
repo_commit: 81d90e5
type: **PRE-REGISTRATION + CODE REVIEW** (no run yet; one free Kaggle CPU kernel once cleared)

## OBJECTIVE
§5.3 of the draft says *"what chromatin does carry is gene-generic"*. That rests on T1's N1 − FB = +0.0012 (§91.12): every cell
given the fitting cells' mean chromatin. Review 056 asked for a control: a non-chromatin per-gene covariate of matched shape.
The question is whether the +0.0012 is chromatin content, or what any per-gene covariate gives T1.

## THE PRE-REGISTRATION (RESULTS 93.15, commit a16fd1b, before the code)
- **The arms**, all §91's T1 (rank_normal, 11 fitting cells, `run_t1`'s LOCO, drug-known dev rows):
  - FB and N1 (§91);
  - N1perm_d, d = 0..19: each mark's mean vector gene-permuted (rng 9500 + 10d + k);
  - N1expr: each mark's mean vector replaced by the fitting-cell mean of b, quantile-matched to it.
  - All use the same LOCO exclusion of the held-out cell.
- **The readings:**
  - NOT DISTINGUISHABLE FROM CAPACITY iff G_chr ≤ max_d G_perm(d);
  - CHROMATIN-SPECIFIC iff G_chr > max G_perm, G_chr − G_expr > 0, and > 0 in ≥ 4 of 6 cells;
  - otherwise GENE-LEVEL, NOT CHROMATIN-SPECIFIC.
- **The harness check:** G_chr reproduces 91.12's N1 − FB within 1e-6, else HARNESS_FAULT.
- **The licensed sentences:** in 93.15.

## FUNCTION (`model/v9/chromatin_genegeneric.py`, PI-written)
- `n1_variant(ctx, enc, fit_cells, kind, draw)`: `build(h)` mirrors `cf.feature_builder`'s N1. kind 'N1' is identical to it
  (tested), 'perm' permutes each mark's mean vector, and 'expr' quantile-matches the mean b to it. The b column is untouched.
- `run(ctx, y)`: one `cf.run_t1` call with 23 specs (FB, N1, N1expr, 20 N1perm), scored on the drug-known dev rows. The output
  holds scores only.
- `reading(res, n1_minus_fb_91)` and `read(DIR)`: verify the marker's sha1, take 91.12's value from
  `chromatin_funnel_91.json`, and apply the reading. This runs locally, once.
- **The kernel** (`orchestration/make_chromatin93_gg_kernel.py`) pins 8 files, among them `chromatin_genegeneric.py`
  `31a972aa`, plus the split bundle. It prints no reading.

## TESTS
4 tests (synthetic worlds). 6 of 6 mutants are caught:
- the permutation ignoring the draw;
- reversed expr order;
- the cell rule loosened to ≥ 3;
- h not excluded;
- the permutation max replaced by its mean;
- the harness check disabled.

## ASKS
1. Is the control the right one?
   - Quantile-matched gene mean of b is the "per-gene biology, not chromatin" arm.
   - The 20 permutations are the capacity null.
   - Is anything confounded? For example, b already enters every arm as its own column, so N1expr is partly redundant with b.
     Is that a defect (it handicaps N1expr), or the point (a non-chromatin covariate that is not new information)?
2. Are the readings and their sentences licensed? Is max over 20 permutations an adequate one-sided 5 % test?
3. Any leak or mismatch in the code?
