# PACKET 025 — PRE-REGISTRATION: C8b's drug-specific mechanism test (a trained post-perturbation named readout)
packet_id: 025
created: 2026-09-25
repo_commit: (the commit carrying this packet)
type: **PRE-REGISTRATION**, written before C8b exists or its accuracy screen is run (RESULTS 85.7 ordering). Its
**execution** is gated on C8b's seed-0 dev Δ ≥ −s0 (= −0.00169); its **design** is fixed here.

## Why
RESULTS 86.4: the pre-perturbation pathway layer carries no replicable drug-specific signal readable by gradient ×
activation. C8b adds a named layer after the drug enters (definition in 85.7: same 800 named nodes, residual,
unsupervised). If a trained drug-dependent named layer aligns with annotated mechanism, that is the mechanistic
interpretability claim the project lacks. Each of your review 023's seven requirements is met below.

## Design
1. **Models:** v9 + `--post_pathway`, trained on **fold 0** (our own splits — they contain unseen compounds by
   scaffold, which cc1 does not), seeds 0, 1, 2, V9Config defaults otherwise, `train_v9_gpu.py`'s fold-0 recipe **with
   distinct per-device seeding and TF32 off**, i.e. the same seeding fix as 85.5 ported to that trainer (a change I will
   make and test before training; recorded as a deviation from the r-series recipe). ~5.6 GPU-h per seed. These models
   are **separate from §85's cc1 dev work**; nothing about them enters §85's accuracy acceptance, and §85's reading
   never enters this one (req. 7).
2. **One primary readout (req. 1):** per row, `Δa[p] = ‖a_post(d)[p] − a_post(mean drug)[p]‖₂` over the node's
   `d_pathway` channels, with 020 C4's fixed mean drug (global = mean over scored compounds; atoms = `k̄` copies of the
   mean atom vector, mask exactly `k̄`); per compound the median over ≤ 4 rows. Gradient × activation is not computed.
3. **Supervision (req. 2):** the post layer is **unsupervised** — no aux loss; any alignment is emergent.
4. **Rows, positives, nulls:** exactly §86 (the same fold-0 test rows, `strength ≥ eval_min_strength`, ≤ 4 per compound;
   ChEMBL mechanism targets; full GMT gene sets; Null 1 = label permutation, 1,000; Null 2 = size-matched, 200).
5. **Calibration against untrained models (req. 3):** **5 untrained inits** (seeds 0–4, default init, quantiser fitted
   on training rows) through the identical pipeline. Their `diff` values define `m_u` and `sd_u` (sd over the 5).
6. **Readouts compared by diff only (req. 4):** the output projection and data projection of §86 reported beside it,
   each with its own nulls.
7. **Strata (req. 5, 6):** compounds unseen vs seen in fold-0 training (the unseen stratum licenses a mechanism claim;
   its Null-1 sd reported); target responsiveness split **at the median** of each compound's target responsiveness
   percentile (mean rank of its landmark targets' |y_Δ| among its rows; compounds with no landmark target form a third,
   reported group), so both sides have usable n.

## Reading, pre-committed
| result | reading |
|---|---|
| on **all three** seeds: `diff ≤ min(untrained diffs)` **and** `diff ≤ m_u − 2·sd_u` **and** Null-1 p < 0.05 **and** S below Null 2's mean; and in the **unseen-compound** stratum, on all three seeds, diff ≤ −0.02 with p < 0.05 | **SIGNAL:** *"a trained drug-dependent named pathway layer aligns with annotated mechanism, including for unseen compounds"* |
| the all-rows conditions on all three seeds, but the unseen stratum fails | **SEEN-ONLY:** alignment not separable from memory (017); no mechanism claim |
| anything else | **NULL:** reported; no mechanistic claim for C8b |

A SIGNAL is additionally worded *"beyond the data's own target alignment"* only if the data projection's diff is not
≤ −0.02 with p < 0.05 on the same rows. No per-drug case-study figure without a separate pre-registration.

## Cost
3 × ~5.6 GPU-h (Kaggle 2×T4) + untrained controls on Kaggle CPU (free). Executed only if C8b passes its gate.

## ASKS
1. Does this meet review 023's seven requirements? Is the untrained calibration (min of 5, and m_u − 2·sd_u) right?
2. Is porting the distinct-seeding fix and TF32-off into `train_v9_gpu.py` for these runs acceptable as a declared
   deviation from the r-series recipe?
3. Is the median responsiveness split, with the no-landmark-target group reported separately, right?
4. Anything else.
