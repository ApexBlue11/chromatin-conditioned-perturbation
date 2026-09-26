# PACKET 032 — DESIGN: which readout §85.2 rule 8 binds (the §37 channel mean is sign-unstable; the baseline fails it)
packet_id: 032
created: 2026-09-26
repo_commit: 0e83810
type: **DESIGN, decided before any variant is scored.** Only the three P2 checkpoints have been run through the instrument.
C6 seeds 1–2 land ~16:30 IST and C7 seed 0 ~12:45; neither's checkpoint has been run through `align_dev.py`.

## What was done (review 031 C1)
- P2's checkpoints `v9dev_base_dev6s0_seed{0,1,2}.pt` fetched from the P2 kernel's output (sha1 1b0ba5aa…, 7ed7ed85…, 8097c16b…);
  each reproduces its saved predictions (`mc_infer_dev.py --arm det --limit 128 --identity_check`: min per-row r 0.9999999,
  dev means equal to 6 decimals).
- `model/v9/align_dev.py` (PI-written; RESULTS 85.10): dev rows through `XPertData` (sha1 asserted); per-row Spearman between the
  named pre-drug layer's readout and the pathway-level target `|y_true − ctl_true| @ M_rownorm.T`; mean over rows
  (`interp_v9.pathway_alignment`, §37's function); null = the readout's pathway columns permuted, 200×, seed 0.

## The finding that forces a choice
| readout | P2 seed 0 / 1 / 2 | mean (sd) | null | z |
|---|---|---|---|---|
| channel mean of `pathway_activations` (what §37 scored) | −0.0301 / −0.0604 / −0.0249 | −0.0385 (0.0192) | −0.0006 ± 0.0089 | −2.7 / −6.6 / −2.7 |
| `aux['pathway_pred']` (`aux_path(pathways)`, the per-node readout the aux loss trains) | 0.2783 / 0.2627 / 0.2785 | 0.2732 (0.0091) | 0.0010 ± 0.0066 | 42.0 / 40.9 / 40.1 |

The aux loss supervises `aux_path(pathways)` (a learned linear map of the d_pathway channels), not their mean, so the channel
mean's sign is arbitrary: positive 8–12 sd in §37's fold-0 model (train_v9_gpu), inverted in the xpert_arm dev models. Under the
channel mean, the **baseline fails rule 8's own "≥ 5 sd of its permutation null"**, so no candidate could ever be accepted —
rule 8 would then decide every acceptance by an artefact of an arbitrary sign.

## Proposal (RESULTS 85.10, marked PROPOSED)
Rule 8 reads the **aux readout**: a variant passes iff its 3-seed mean ≥ 0.2732 − 0.02 = **0.2532** and ≥ its mean null + 5 × its
mean null sd. The channel mean is reported beside it, not read. Scope unchanged (CLAIMS 4.16): cell-level, partly supervised on
the target it is scored against — a non-inferiority gate for "the named nodes rank which pathways move in this cell", not a
mechanism claim.

## ASKS
1. Is binding rule 8 to the aux readout faithful to rule 8's intent ("keeps the named pathway layer's cell-level alignment …
   partly supervised by the aux loss"), or is it a change of instrument that needs to be declared as an amendment with the
   reason above? Either way, is the choice outcome-free with respect to C6 and C7 (no variant has been measured)?
2. Should the paper's interpretability sentence (CLAIMS 4.16, §37) now cite the aux readout too, since §37's positive channel-mean
   alignment does not reproduce in the dev-carve models? Does §37's claim need a caveat?
3. Anything in `align_dev.py` that differs from §37's instrument other than the readout (rows, target, null)?
