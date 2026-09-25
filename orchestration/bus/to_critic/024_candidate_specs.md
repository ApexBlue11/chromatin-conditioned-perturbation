# PACKET 024 — PRE-REGISTRATION: the exact definitions of §85's first six candidates, before any is built
packet_id: 024
created: 2026-09-25
repo_commit: (the commit carrying this packet)
type: **DESIGN / pre-registration.** No candidate code exists. P2 is done (§85.6): μ0 = 0.43693, s0 = 0.00169 on the
6 dev cells, Kaggle 2×T4, distinct seeding, ~1.72 h per seed.

## Common to every candidate
`xpert_arm.py --bundle xpert_mdmt_splits.npz --split split_cold_cell_1 --dev_cells 6 --dev_seed 0 --epochs 12
--d_model 256 --dp_seed_mode distinct`, seed 0 first; **one flag changes, nothing else** (same lr, schedule, batch 48,
EMA-free as today, V9Config defaults otherwise). Scored by `score_dev.py` against the P2 baseline, §85.2 rules 6–8. Each
flag defaults to off and is byte-identical to today when off (a regression test asserts the default forward and loss equal
the committed code's on a fixed batch). Every run in the ledger.

## The six, defined
| id | flag | exact change | what it tests | prior (from our record and W12b-checked literature) |
|---|---|---|---|---|
| **C1** | `--no_atoms` | the drug token sequence is `[global]` only: no atom tokens reach cross-attention or drug self-attention; `u_feats` unchanged | does removing atom tokens help unseen cells when **trained** without them? | §37 (SA off, as here): atoms ablated at inference cost −0.007 on `unseen_cell`, i.e. removal *helped*; §74 (SA on): atoms help +0.0029. Mixed |
| **C2** | `--no_ctl_encoders` | `use_matched_ctl = use_cell_ctl = False` (existing switches): gene tokens never see the control profile or the per-cell control mean; the delta head is unchanged, and the absolute head predicts `ctl + delta` as before | TxPert's "no basal state encoder" for unseen cell lines (arXiv 2505.14919, Fig. 7) | against it: §37, the matched control is a top-3 contributor (+0.105 on `unseen_cell` when ablated) |
| **C3** | `--listnet_w 0.1` | adds `0.1 × ListNet(Δ)`: per row, cross-entropy between `softmax(z(y_Δ))` and `softmax(z(ŷ_Δ))` over the 978 genes, `z` = per-row standardisation (τ = 1), on the rows the delta loss already uses | a listwise ranking objective (ExPO's leave-cell-line-out ablation: ranking metrics) | ExPO's gain was on NDCG/MCC, MAE unchanged; ours is Pearson |
| **C4** | `--deg_adapt_k 50` | the delta Huber term becomes PertAdapt's detached reweighting: `α_all·L_all + α_DE·L_DE`, `L_DE` = Huber over each row's top-50 genes by `|y_Δ|`, `α` = stopgrad((L_all+L_DE)/L) as in their eq.; the PCC and other terms unchanged | concentrate the delta loss on the genes that dominate the metric | PertAdapt's evidence is per-perturbation CV on single-cell data, **not** unseen cells (W12b) |
| **C6** | `--sign_head_w 0.1` | an auxiliary linear head on the final gene tokens predicts `1[y_Δ > 0]` (BCE) on each row's top-50 `|y_Δ|` genes, weight 0.1; its output is never used in prediction | IDEAS A3: a direction/sign signal the project has never put | none; near-free |
| **C8b** | `--post_pathway` | a second `NamedPathwayReadout` (same 800 named nodes, same `M`) applied **after the last perturb block** and written back residually like the first, **unsupervised** (no aux loss on it); the pre-perturbation layer and its aux loss stay | accuracy effect of a drug-dependent named layer (the MoA test is separate, below) | none |

**C5 (CCLE) and C7 (chromatin edges) are not in this batch**: each needs its coverage for all 40 cells, missing-cell
handling and (C7) the mismatched-chromatin null pre-registered first (review 019 ask 5).

## Order and budget
Seed 0 of all six, two at a time (Kaggle allows two concurrent GPU sessions): ~10.3 GPU-h. Follow-up seeds per §85.2
rule 6. Stacking (P6) and the final (P7) as registered.

## C8b's MoA test — deferred to its own pre-registration, with review 023's seven requirements
One primary readout Δa = a(d) − a(mean drug); unsupervised nodes; calibration against ≥ 5 untrained inits; readouts
compared by diff; a responsiveness split with usable sizes; the unseen-compound stratum licenses a mechanism claim;
fold-0 test rows with C8b trained on fold 0, separately from the cc1 dev work, and the two readings independent. It is
priced (3 fold-0 seeds ≈ 17 GPU-h) and pre-registered only if C8b does not fail §85's accuracy screen — **that ordering
is fixed now**, so the MoA test's existence cannot depend on anything but the accuracy screen.

## ASKS
1. Are the six definitions exact enough that a worker's implementation can be checked against them?
2. Is "one fixed hyperparameter per candidate (0.1, 50, τ = 1), chosen now without tuning" right, or should any be a
   small pre-registered grid?
3. Is the C8b ordering (accuracy screen first, MoA study only if it does not fail) acceptable, or does it couple the two
   readings in a way ask 3 item 7 of review 023 forbids?
4. Anything else.
