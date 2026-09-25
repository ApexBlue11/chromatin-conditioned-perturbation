# PACKET 019 — DESIGN: a v9 development protocol that can improve accuracy without selecting on the test cells
packet_id: 019
created: 2026-09-25
repo_commit: a05dba1
type: **DESIGN, pre-registration.** No variant has been built or run. Nothing here changes §71: its comparison stays
the committed `v9_cc1_epi_seed0.npz` (the §45 model) against O2's XPert.

## Why
The principal wants better unseen-cell accuracy **with the mechanistic interpretability kept**, and wants mechanisms
explored before spending on seeds. Every design choice so far was read on test splits. Doing that now, on the cells
the XPert head-to-head uses, would repeat the flaw we disclose in XPert's recipe (test-guided selection).

## Facts this rests on (all in the repo)
- **Harness:** `model/v9/xpert_arm.py` on XPert's L1000_mdmt `split_cold_cell_1`: 32 training cells, 8 test cells,
  their metric (mean per-row delta Pearson). A 12-epoch run costs **~1.5 h on Kaggle 2×T4** (§47 cost note), not the
  10.4 h of §52.3, which was a batch-8 laptop run.
- **Seed noise on XPert's splits:** 3-seed sds of 0.0008 (`split_breast_1`) and 0.0003 (`split_lung_1`), both
  far below our own `unseen_cell` sd (0.0052, v7, §21). Unknown on a cell-level dev split, so it is measured first.
- **Interpretability that exists today (§37):** the named Reactome/GO pathway readout's alignment with measured biology
  is **+0.08 to +0.12 against a permutation null of ~0 ± 0.01 (8–12 sd), p = 0.005**. Nothing else survived
  (atom→gene target enrichment was retracted, §4.1a).
- **Literature, after an adversarial check (W12b):** evidence specific to unseen cell lines supports
  (i) using the raw control profile with no learned basal encoder (TxPert, arXiv 2505.14919, Fig. 7);
  (ii) little from any single auxiliary loss — XPert's own cold-cell ablations are all inside their fold sd, and
  everything together buys +0.020 (Table R13, checked against the supplementary by the PI);
  (iii) a listwise ranking loss mattered for ranking metrics in ExPO's leave-cell-line-out setting (J. Cheminf. 2026).

## Protocol (proposed)
**P1. Dev carve.** From the 32 training cells of `split_cold_cell_1`, hold out **K = 6** chosen by a fixed seed
(recorded with their lineages and row counts before any training). Train on the other 26; score on the 6 dev cells'
rows. **In dev mode the harness never computes test-cell predictions** (asserted in code). The 8 test cells are
touched once, by the final model.

**P2. Noise first.** Baseline (v9 as in §45 but at the harness's standard batch 48) × 3 seeds on dev → `μ0`, `s0`.

**P3. One factor at a time, one seed each.** Δ = variant − μ0 on the dev rows.
- Δ ≥ max(2·s0, 0.003) → 2 more seeds; Δ < s0 → dropped; otherwise 1 more seed, then the same rule on the mean.
- **Accepted** iff the 3-seed mean exceeds μ0 by more than 2·√(s0²/3 + s_v²/3), **and** ≥ 4 of 6 dev cells favour it
  (paired, per-row medians), **and** it passes P4.

**P4. Interpretability gate (binding).** Run `probe_v9`'s pathway-alignment test on the variant's seed-0 checkpoint on
the dev rows. **A variant that removes the named pathway layer, or whose alignment falls below 5 sd of its own
permutation null, is not accepted** whatever its accuracy (it may be reported as accuracy-only).

**P5. Candidates, in order (priors from our own measurements first):**
| # | variant | prior |
|---|---|---|
| C1 | drop atom tokens (global drug token only) | §37: ablating them at inference already helps on all three of our splits |
| C2 | raw control, no learned control encoder | TxPert Fig. 7 (unseen cell lines); IDEAS B2 |
| C3 | listwise ranking loss on the delta (ListNet) added to the current loss | ExPO LCL-O ablation (ranking metrics) |
| C4 | DEG-reweighted loss (PertAdapt's detached L_adapt over the top-50 \|Δ\| genes) | the metric is dominated by responsive genes; PertAdapt evidence is NOT unseen-cell |
| C5 | CCLE basal expression for the cell (`use_ccle`, currently off) | genome-wide basal state is the only extra input unseen cells can have |
| C6 | direction/sign auxiliary head (IDEAS A3) | never tried; near-free |
| C7 | chromatin gating graph EDGES (IDEAS A1, trained arm) | two zero-parameter pre-tests null (§82.7, §83.5); tried last |

**P6. Stack and confirm on dev.** Accepted variants combined; the combination at 3 seeds must not lose to the best
single accepted variant.

**P7. Final, once.** The stacked model trained on all 32 training cells × 3 seeds, scored once on the 8 test cells,
read as a **second, separately registered comparison** with O2's XPert under §71.3's rule (cluster CI, ≥ 7/8 cells),
alongside — never instead of — the §71 comparison of the committed model. A 3-seed prediction average is reported as
its own model, labelled as an ensemble.

**P8. Compute.** ~12–15 dev runs × ~1.2 h ≈ 15–18 GPU-h on Kaggle 2×T4 if O2 moves to Lightning (84.4); otherwise
they wait for O2. Every run's cost is logged against the 30 h/week quota.

## ASKS
1. Is the dev carve (K = 6 of 32 training cells, test never predicted in dev mode) sufficient to keep the final
   comparison free of test-guided selection? Is K = 6 right?
2. Are P3's advance/accept rules sound given that `s0` is unknown until P2 runs? Should the accept rule be fixed
   numerically now instead?
3. Is the P4 interpretability gate the right binding constraint, and is 5 sd the right floor?
4. Is registering P7 as a second comparison (with §71 untouched) the right way to use an improved v9 against XPert?
5. Anything else — including whether any candidate should be dropped or added before building starts.
