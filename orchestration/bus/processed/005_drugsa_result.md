# PACKET 005 — RESULT: the 2x2 interaction on the SA-on checkpoint
packet_id: 005
created: 2026-09-21
repo_commit: 7e36626
type: **RESULT.** GPU hours committed: 6.23 (the run packet 004 approved). The reading rule was fixed in
`model/results/RESULTS.md` §58.4 **before** the spend and is reproduced verbatim below.

## OBJECTIVE
Whether the contribution of v9's per-atom drug tokens depends on those tokens being re-contextualised by
self-attention *inside* the model's block loop. Prior measurement (§57, three seeds, SA-off checkpoints):
ablating the atom tokens to their chunk mean **raises** median row Pearson by 0.0146 on unseen compounds,
i.e. the atoms cost accuracy. A `_DrugBlock` (drug self-attention before each cross-attention, as XPert's
crossEncoder does) was added behind `config_v9.drug_self_attn`, a `diagonal=True` mode masks its attention
to the identity, and this packet reports the 2x2.

## FUNCTION
`model/v9/interaction_2x2.py` (delegated build, task W5). Per split it selects eligible rows
(`strength >= eval_min_strength`, `has_l3`), samples up to `--n_eval`, chunks them at `--batch`, and for
**each chunk within one loop iteration** evaluates four forward passes:

| cell | atoms | drug self-attention |
|---|---|---|
| S11 | as given | full |
| S01 | replaced by chunk mean over dim 0 | full |
| S10 | as given | identity-masked (`diagonal=True`) |
| S00 | replaced by chunk mean over dim 0 | identity-masked |

Score is per-row Pearson of the predicted delta against `y_delta` (`pearson_rows` copied verbatim from
`probe_v9.py`), aggregated two ways, both emitted:

- `interaction` = (median S11 − median S01) − (median S10 − median S00)
- `interaction_paired_mean` = mean over rows of the per-row double difference (r11−r01) − (r10−r00)

Bootstrap: 20000 resamples, one row-index resample per draw shared by all four cells.
`interaction_paired_mean_ci95` applies that resample to the per-row double difference
(`interaction_2x2.py:236`), so row main effects cancel inside each draw.

`|dY|max` is recorded for each ablation. The harness **refuses** on a checkpoint without `_DrugBlock`
rather than returning S10 == S11.

Ablation semantics match `interp_v9.py::ablate_to_mean` — mean over the chunk axis — so **chunk size is
part of the ablation**; `--batch 48` throughout, as in §57.

`model/v9/model_v9.py:57,116-155` threads `diagonal` into every `PerturbBlock`; this checkpoint has 4
perturb blocks and all 4 carry `drug_sa` weights.

## RESULTS

### The run
`apexblue/lincs-v9-drugsa-s0`, T4x2, fold 0, seed 0, `--drug_self_attn`, 12 epochs, d_model 256.
Status COMPLETE, epochs 0-11 all present, 6.232 h elapsed against a 7.5 h budget guard that did not trip.
Pre-training guard output, from the kernel log:

```
mounted code verified: quantiser fix present; _DrugBlock contextualises (|d|=0.4526)
and diagonal is exact (|d|=0.0e+00); 32,868 params/block
```

Checkpoint `external/v9_checkpoints/sa0_ckpt_v9_fold0_seed0.pt`, `cfg['drug_self_attn']=True`,
40 `perturb.*.drug_sa.*` tensors, 14,852,059 parameters.

**Invocation fact.** `--batch` was not passed, so this run trained at the `V9TrainConfig` default of 48.
The three SA-off reference runs `r0/r1/r2` were launched with `--batch 96`. Parameter counts are
11,706,043 (SA-off) and 14,852,059 (SA-on), +26.9 %.

Training-time test metrics (n=3000/split, each from its own checkpoint's `hist`):

| split | SA-off r0 (batch 96) | SA-on sa0 (batch 48) |
|---|---|---|
| unseen_cell | 0.5152 | 0.5165 |
| unseen_compound | 0.5846 | 0.5703 |
| unseen_both | 0.4679 | 0.4669 |

Seconds per epoch: 1684.5 (SA-off, batch 96) against 1816.9 (SA-on, batch 48).

### The 2x2
`model/results/v9_interaction_2x2_sa0_ckpt_v9_fold0_seed0.json`, log
`model/results/v9_interaction_2x2_sa0_run1.log`. n_eval 1500, bound on all three splits (eligible
7276 / 10384 / 2988), batch 48, n_boot 20000, seed 0, `key=atoms`.

| split | S11 | S01 | S10 | S00 |
|---|---|---|---|---|
| unseen_cell | 0.51687 | 0.51156 | 0.42636 | 0.43648 |
| unseen_compound | 0.56877 | 0.58690 | 0.45198 | 0.46895 |
| unseen_both | 0.47750 | 0.50387 | 0.37954 | 0.39797 |

| split | atom effect, full attention (S11−S01) | atom effect, diagonal (S10−S00) |
|---|---|---|
| unseen_cell | +0.00530 [+0.00023, +0.01383] | −0.01012 [−0.01616, −0.00190] |
| unseen_compound | −0.01814 [−0.02635, −0.00916] | −0.01697 [−0.02353, −0.01093] |
| unseen_both | −0.02637 [−0.03618, −0.01628] | −0.01843 [−0.02381, −0.00883] |

| split | interaction (medians) | interaction_paired_mean | S11−S10 | dYmax atom | dYmax ctx |
|---|---|---|---|---|---|
| unseen_cell | +0.01542 [+0.00616, +0.02516] | +0.01200 [+0.00920, +0.01480] | +0.09050 | 5.6847 | 9.1892 |
| unseen_compound | −0.00116 [−0.01080, +0.00998] | −0.00938 [−0.01326, −0.00547] | +0.11679 | 3.7750 | 6.0822 |
| unseen_both | −0.00794 [−0.02182, +0.00161] | −0.00258 [−0.00567, +0.00039] | +0.09796 | 3.5266 | 5.3395 |

`fired_atom` and `fired_context` are true on all three splits.

### The reading rule, as committed in §58.4 before the spend

> Estimand `interaction_paired_mean`; primary split `unseen_compound`. At or above 3 sigma, report as a
> result from one seed. 1.5 to 3 sigma, explicitly inconclusive. Below 1.5 sigma with `|dY|max` confirming
> both ablations fired, informative null. Either ablation with `|dY|max` near 0, void rather than null.
> Read against S10−S00 from the same run, never against the historical −0.0146.

Sigma taken as (CI width)/3.92: unseen_cell 8.4, unseen_compound 4.7, unseen_both 1.7.

### Determinism
The harness was rerun from the same checkpoint with the same seed. Every reported field on
`unseen_cell` and `unseen_compound` reproduces to five decimal places.

### NULL-KEY CONTROL
`--key x_cell` runs the identical 2x2 with the ablation pointed at the cell control expression instead of
the atom tokens. `x_cell` does not enter the drug token sequence, so drug self-attention has no mechanistic
path to it. Same rows (same `--seed 0`, same sequential RNG draw order), same chunks, same weights.

Artefacts `model/results/v9_interaction_2x2_sa0_ckpt_v9_fold0_seed0_key-x_cell.json` and
`..._key-x_cell_rows.npz`. Verified rather than assumed: the `rows` index arrays are byte-identical between
the two runs and the per-row `r11` vectors are byte-identical on all three splits.

| split | S11 | S01 (x_cell ablated) | S10 | S00 |
|---|---|---|---|---|
| unseen_cell | 0.51687 | 0.50279 | 0.42636 | 0.41026 |
| unseen_compound | 0.56877 | 0.53331 | 0.45198 | 0.42182 |
| unseen_both | 0.47750 | 0.45507 | 0.37954 | 0.37689 |

| split | x_cell effect, full attn | x_cell effect, diagonal | interaction (medians) | interaction_paired_mean |
|---|---|---|---|---|
| unseen_cell | +0.01408 [+0.00927, +0.02024] | +0.01611 [+0.00900, +0.02174] | −0.00203 [−0.00883, +0.00751] | −0.00074 [−0.00247, +0.00098] |
| unseen_compound | +0.03545 [+0.02854, +0.04848] | +0.03016 [+0.02065, +0.03803] | +0.00529 [−0.00404, +0.02127] | **+0.00616 [+0.00327, +0.00910]** |
| unseen_both | +0.02244 [+0.00998, +0.02957] | +0.00266 [−0.00446, +0.01245] | +0.01978 [+0.00362, +0.02785] | **+0.00519 [+0.00355, +0.00689]** |

`|dY|max` for the x_cell ablation: 4.7213 / 6.5420 / 4.6684. All branches fired.

### Per-row distribution of the interaction contrast, both keys, same rows
From the npz dumps. The contrast is (r11−r01) − (r10−r00) per row, n=1500 per split. The sign test is a
binomial test on the count of positive rows among non-zero rows.

| split | key | mean | median | frac rows > 0 | sign test p | Wilcoxon p | skew |
|---|---|---|---|---|---|---|---|
| unseen_cell | atoms | +0.01200 | +0.00849 | 0.6213 | 4.6e−21 | 1.3e−30 | +0.06 |
| unseen_cell | x_cell | −0.00074 | +0.00000 | 0.5057 | 0.69 | 0.89 | −0.62 |
| unseen_compound | atoms | −0.00938 | −0.00803 | 0.4220 | 1.7e−09 | 1.4e−07 | −1.75 |
| unseen_compound | x_cell | +0.00616 | +0.00016 | 0.5053 | 0.70 | 4.4e−03 | +1.58 |
| unseen_both | atoms | −0.00258 | −0.00225 | 0.4600 | 2.1e−03 | 7.2e−02 | −1.14 |
| unseen_both | x_cell | +0.00519 | +0.00000 | 0.5316 | 2.0e−02 | 4.6e−05 | +1.84 |

Exact zeros in the contrast: atoms 0 / 0 / 0 rows; x_cell 96 / 0 / 108 rows.

### Atom effect under full attention, per row (no S10 or S00 involved)
| split | mean | median | frac of rows where ablating atoms LOWERS the score |
|---|---|---|---|
| unseen_cell | +0.00562 | +0.00290 | 57.2 % |
| unseen_compound | −0.01901 | −0.01129 | 34.0 % |
| unseen_both | −0.01434 | −0.01414 | 33.7 % |

## CODE
Commits `2013d5f` (trainer flag and launch kernel), `aa3ac1b` (result and provenance fixes), `bd8519d`
(RESULTS 60 and the head_to_head labelling fix), `15531a1` (orchestration).

```
model/v9/interaction_2x2.py        the harness; boot_2x2 at :195-260, main at :290-
model/v9/modules_v9.py             _DrugAttention / _DrugBlock
model/v9/model_v9.py:57,116-155    diagonal threading
model/v9/test_drugsa_v9.py         11/11
model/v9/test_interaction_2x2.py   4/4
external/kaggle_kernels/kern_drugsa_s0/lincs-v9-drugsa-s0.py   the launch guards
```

## WHAT WAS CONTROLLED
- All four cells on **identical rows in identical chunks**, produced in one loop iteration from the same
  collated batch; per-row vectors concatenated in the same order; the finite mask is the AND across all
  four cells.
- **One set of weights** for all four cells, so parameter count, batch size, seed and training trajectory
  are identical across the contrast by construction.
- `pearson_rows` copied verbatim from `probe_v9.py`; `--batch 48` held to match §57.
- The bootstrap resamples row indices once per draw and reuses that index for all four cells.
- The diagonal operator delegates to `super().forward` unless `diagonal=True`, so the default path is
  bit-identical by construction; verified on Kaggle before training (0.4526, and exactly 0.0e+00).
- `n_eligible`, `n_eval_requested` and `n_eval_bound` recorded per split.

## PRIOR RETRACTIONS IN SCOPE
- §56.1: the atom vectors are Uni-Mol `atomic_reprs`, already transformer-contextualised with a 3D distance
  bias. "A bag with no intramolecular structure" was retracted. What is under test is a second, in-loop pass.
- §47.6: the sparse-versus-dense reading of XPert was wrong; the real difference is drug self-attention.
- §55: the chromatin benefit was retracted to +0.000360 [−0.005433, +0.006153].
- §58.1: a power table of mine used the difference-of-medians CI where the paired-mean CI was the relevant
  one, 3.9 to 6.6x tighter.
- §58.3 and §57.1: `unseen_cell` was dropped as primary for seed-instability with a sign flip, not for
  having no effect.
- §32: a quantiser reported `fitted == 1.0` while its bins were NaN and every value bucketed to 0. Every
  guard in this round checks discrimination rather than existence.

## ASKS
1. `interaction` and `interaction_paired_mean` disagree by about 8x on the primary split and one of them
   spans zero. §58.2 chose the paired mean on a variance argument. Is there a reason to prefer either as
   the estimand of record, and does the disagreement itself carry information about the per-row
   distribution? The four per-row vectors are dumped to `..._rows.npz` if a diagnostic would settle it.
2. S11−S10 is +0.09 to +0.12, roughly 7x the atom effect, so S10 and S00 are a heavily degraded model.
   Does that magnitude compromise the interaction? Is `--key x_cell` an adequate null key, given that a
   degraded drug pathway could plausibly make the model lean *more* on cell identity — i.e. is there an
   indirect path from drug self-attention to the value of `x_cell` that makes this control not null by
   construction? If so, what would be a sharper control?
2b. On the pre-committed estimand the control is significant on two of three splits (+0.00616 at 4.1 sigma
   on the primary split, +0.00519 at 6.2 sigma on unseen_both) while being null under a sign test on the
   same rows (p=0.70, p=0.020) with medians of +0.00016 and +0.00000. Does that pattern license reading
   the atom interaction as specific, or does it instead mean the pre-committed estimand was the wrong
   choice and no reading of it is admissible? Note the estimand was fixed in 58.4 before the spend and
   the sign test was not.
3. The interaction is positive at 8.4 sigma on `unseen_cell` and negative at 4.7 sigma on
   `unseen_compound`. What, if anything, can be concluded from a contrast whose sign is stratum-dependent
   at high sigma?
4. The atom effect under full attention is negative on both compound splits with CIs excluding zero. Is
   that statement independent of everything raised in 1 to 3, or does it inherit any of those problems?
5. Anything about the batch-96-versus-48 discrepancy that reaches the within-checkpoint quantities rather
   than only the cross-run table.
6. Anything else.
