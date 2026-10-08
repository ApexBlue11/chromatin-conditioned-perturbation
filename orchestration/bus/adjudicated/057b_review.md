# REVIEW OF PACKET 057, ADDENDUM 2: clearance check on c32b13c (93.15b, corrected 93.16)
verdict: SOUND
reviewed_commit: c32b13c

**Cleared to run.** Review 057a's C1–C3 are implemented as adjudicated.
- **Tests:** `chromatin_genegeneric.py` = d00ebeb1. I reran `test_chromatin_genegeneric.py` under `.venv-cuda`: **8 passed**.
- **The kernel:** the regenerated kernel pins d00ebeb1 and `chromatin_funnel` c56bb6f1. The staged copy in
  `external/kaggle_chromatin_src/` is identical.

There are two MINOR notes below, neither blocking. Neither needs a re-pin before the run.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording | **`main()`'s TIE assertion can't validate Z.** `E_final` is an exact rank grid, so `count(v ≤ (Z − 0.5)/(N − 1)) == Z` holds for **any** Z (checked: true at 22,000, 22,778 and 23,500 for H3K27me3). It confirms N and that this is the same rank-grid `E_final`. That's the right guard against a changed input, but it doesn't confirm the block size. For ATAC and H3K27ac, Z comes exactly from the log. For H3K27me3, it stays the `E_final` estimate, which 93.15b already says. The code comment *"the pinned block sizes describe THIS E_final"* and the packet's *"main asserts the block sizes"* overstate it. | Describe it as *"asserts N and the rank grid; H3K27me3's Z is an estimate (93.15b)"* wherever it's next mentioned. No code change is needed before the run. If the comment is edited, regenerate the pin. |
| 2 | MINOR | stale docstring | **`install_tie`'s docstring still says *"reported only"* and *"ATAC and H3K27ac"*.** The code ties all three marks (`TIE` has key 2), and since 93.15b these features are the ones of record. | Update it the next time the file changes, and re-pin then. The behaviour is right. |

## Answers to the asks

**Clearance: yes.**

**C1(i):**
- **What `TIE` holds:** {0: 6465 / 31264, 1: 6726 / 34195, 2: 22778 / 25402}.
- **What `install_tie` does:** for each (cell, mark) present in rank_normal, it sets values ≤ (Z − 0.5)/(N − 1) to 0 and then
  applies `rank_normal`. With average ranks, each cell's block becomes one tied value. Rank_normal's `has` is reused, so the
  failed-H3K27me3 exclusion is kept.
- **H3K27me3's Z:** 22,778 equals my estimate in 057a.

**C1(ii):** 93.16 is corrected.
- **The table:** its H3K27me3 column and the fitting-cell line (SKBR3 94, LNCAP 93, HL60 80, A549 86, HCT116 94, HEPG2 75,
  MCF10A 96, PC3 94, JURKAT 2; HEK293T and VCAP 99 on the failed list) match my numbers.
- **The withdrawal:** the "not affected" line is withdrawn in place.

**C1(iii)(b), 93.15b** — the code matches the registration:
- **Specs:** `run` builds, for each feature set f ∈ (tie, v9): N1_f, E_f (with `ref_enc = enc`, so E is matched to that set's
  own mark means; C3), E_f+N1, and 20 × (N1perm_f_d, E_f+perm_d). FB is shared. That's 1 + 2 × 43 = **87 specs** in one
  `run_t1` call.
- **The harness:** N1_v9 is `cf.feature_builder`'s §91 N1, so the check is on §91's features as registered.
- **The reading:** `reading()` returns HARNESS_FAULT unless N1_v9 − FB reproduces +0.0012099. Otherwise the reading of record
  is `reading_on(res, 'tie')`. The v9-feature reading is reported with the artefact named, and the 91.12 caveat appears in
  every output.
- **The permutation:** one per draw, shared across the three marks, in both null families.

**C2:** 93.16's H1 line is scoped to accessibility, as written.

## What I checked and found sound

- **The code:** `chromatin_genegeneric.py` d00ebeb1 in full (`TIE`, `variant`, `install_tie`, `run`, `reading_on`, `reading`,
  `read`, `main`), and the 8 tests rerun.
- **The RESULTS text:** 93.15b and the corrected 93.16 (mechanism, H3K27me3 paragraph, per-cell table, inheritance, H1 scope).
- **The kernel:** `external/kaggle_kernels/kern_chromatin93_gg` pins (`chromatin_genegeneric.py` d00ebeb1,
  `chromatin_funnel.py` c56bb6f1), and the staged source sha1.
- **`E_final`:** the TIE assertion's behaviour across Z values (C1).

## What I could not assess, and why

- **H3K27me3's exact Z.** As in 057a, step12's raw tensor and log aren't in the repo. A few entries either way would change
  nothing material: a handful of real lowest values tied, or a handful of artefact entries left distinct, out of 22,778.
