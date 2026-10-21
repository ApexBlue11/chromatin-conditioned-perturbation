# PACKET 095 — RESULT: T-E read once (RESULTS §101.3); H-E2 passes. Please attack it.
packet_id: 095
created: 2026-10-21
type: **RESULT REVIEW** (the first registered chromatin positive in this project)

**Files:**
- `model/results/v10c/te_reading.json`, from `te_tests.py` at 42e5e96, run once;
- the RESULTS entry is §101.3 (868940d).

## The result

| | cells | mean partial r [98.33 %] | Δspec [98.33 %] | nulls (rank of real) | verdict |
|---|---|---|---|---|---|
| **H-E2** BETi × ACT tss_50kb (−) | 19 | **−0.071** [−0.093, −0.049] | −0.072 [−0.105, −0.037] | 0.00 / 0.00 | **passes** |
| H-E1 DNMTi × METH (+) | 26 | +0.022 [−0.005, +0.048] | +0.026 [−0.006, +0.054] | 1.00 / 0.99 | no pass |
| H-E3 silent induction × ACC prom (+) | 54 | +0.017 [−0.004, +0.038] | −0.007 [−0.018, +0.002] | 0.98 / 0.97 | no pass |

**H-E2's supporting evidence:**
- 18 / 19 cells are negative;
- REF gives +0.0006;
- with apabetalone, −0.068;
- the narrow window gives −0.066;
- JQ1(−) gives +0.023 [−0.003, +0.052], but over only 6 cells.

**Reported, never read (two-sided):** HDAC inhibitors × ACT prom **−0.162** [−0.194, −0.128] and × ACC prom −0.097, unchanged in
`hdac_core`.

## Alternative explanations I see (please add yours)
1. **Is chromatin just a cleaner expression measurement?**
   - BET and HDAC inhibitors are known to repress highly transcribed genes.
   - If the ACT marks measure a gene's true expression in that cell better than L1000's control profile does, then "beyond
     baseline expression" would still be an expression effect.
   - **Proposed decisive test:** with CCLE RNA-seq, an independent expression measurement for 131 of 217 lines:
     - (a) H-E2's statistic with m = CCLE expression rank;
     - (b) ACT with CCLE expression rank as an added covariate.
   - If (b) stays negative and specific, chromatin carries information beyond expression. If it vanishes, it was expression.
2. **Is the basal adjustment linear?** It is linear in rank basal. A non-linear expression dependence of BET / HDAC
   repression could leak into ACT. **Proposed:** decile basal as a sensitivity.
3. **The REF-specificity contrast** shows non-epigenetic drugs don't show this (+0.0006). Is that enough to exclude a generic
   "strong perturbation represses active genes" effect?
   - Should the comparator be other **strong** perturbagens instead, such as proteostasis, transcription or kinase classes
     matched on response magnitude?
4. **Plate / batch:** epigenetic-adjacent mates are excluded. Could the BETi rows' plates differ systematically in a way the
   contrast doesn't remove?
5. **The HDAC finding:** it was registered two-sided and reported only. It needs a confirmatory test with its direction fixed
   now, on cells or rows not yet read.
   - Every T-E row is a `cold_cell_1` training row. Is there an untouched set? The T-C test cells' rows are reserved for
     T-C's registered reading.

## ASKS
1. Is the pass sound as registered? Is anything in the code or data wrong?
2. Rank the alternative explanations. Which are decisive and must run before any sentence beyond §101.3's licensed one?
3. Design the confirmation: CCLE control, decile basal, a strong-perturbagen comparator, the HDAC direction fixed.
   - On which rows can that be done without touching T-C's test rows?
