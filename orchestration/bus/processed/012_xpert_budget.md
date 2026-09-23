# PACKET 012 — the XPert cold-cell run cannot be admissible in one session: what to buy, if anything
packet_id: 012
created: 2026-09-23
repo_commit: b85f835
type: **PRE-SPEND DECISION.** GPU hours this session: 6.80. Weekly Kaggle quota: 30 h. This packet commits none.

You approved activation checkpointing in 011 with C1: it slows every step, so fewer epochs fit in the 8.3 h guard,
and a guard stop may be inadmissible. v5 measured it on the real T4. The answer is worse than "fewer epochs".

## MEASUREMENT (v5, `MEASURE_ONLY`, `external/kaggle_out/cc1_v5/run_record.json`)
GUARD F now runs one batch-128 training step with `torch.optim.Adam` + `GradScaler.step` (your C2), then five timed
steady-state training steps and five validation forwards, on their `load_dataloader`, patched model, one T4.
```
s_train_step 2.265 s | s_val_step 0.411 s | peak 3.74 GiB | train batches 372 | val batches 167
epoch_s_projected 911.4 (92.5 % training) | setup 208 s | epochs_within_budget_projected 31.6 (8.3 h, 900 s reserved)
host MemAvailable min 20.0 GB
```

## THEIR RECIPE (`configs/config_l1000.yaml`, `train_xpert.py`)
`num_epochs: 2500`, `patience: 50`, `init_epoch: 70` (the loss switches from `batch_weighted_loss` to the full
objective at epoch 70), `--lr_scheduler` defaults True: `LambdaLR(lambda epoch: 1.0 if epoch < 40 else 0.5)`
(`:458-463`).

Their released checkpoints store their epoch: `l1000_mdmt_warm_split.pth` **164**; `pretrain_mdmt_full_200_epoch.pth`
198; `pretrain_mdmt_new.pth` 244. With patience 50 the warm-split run lasted at least 214 epochs. No cold-cell
checkpoint is released.

## THE ADMISSIBILITY RULE, VERBATIM (RESULTS 71.7, committed at 08b0a90 before launch 1)
> | training ended by | `counter_at_end` | reading |
> |---|---|---|
> | their early stopping (`finished`) | 50 by construction | §71.3 applies unchanged |
> | **the wall-clock guard** | **≥ 45** of patience 50 | effectively converged; §71.3 applies |
> | **the wall-clock guard** | **< 45** | **supports NO v9-win claim**, whatever the numbers — XPert was still improving when cut off |
> | crash | — | void; the kernel fatals |
>
> `counter_at_end = last_epoch_index − best_epoch`

## WHAT FOLLOWS, AS I READ IT
1. In 31 epochs, `counter_at_end ≥ 45` is impossible. **A single-session run is inadmissible by construction.** It
   also never reaches epoch 70.
2. The informative direction of a capped run is an under-trained XPert that still beats v9. v9's fold-1 score is
   0.4734; XPert's published cold-cell five-fold mean is 0.383 ± 0.027. The prior favours the uninformative outcome.
3. Cost of an admissible run, one T4, checkpointed, anchored on 164 (an anchor, not a prediction):

| epochs | GPU-h | 7.95 h sessions |
|---|---|---|
| 120 (earliest admissible: best at 70, +50) | 30.4 | 3.8 |
| ~210 (best at 164, +45) | ~53 | ~7 |

4. **Their resume is lossy.** `train_xpert.py:480-490`, `--resume_from`: reloads `model_state_dict` only;
   `# optimizer.load_state_dict(...)` is commented out; the LambdaLR is rebuilt so its epoch count restarts at 0
   (LR back to 0.004 after epoch 40); `EarlyStopping` is constructed fresh (`:524`), so its best score and counter
   reset. Multi-session needs our own full-state checkpoint: model, Adam, GradScaler, scheduler, stopper, RNG states.
5. **The second T4 is idle** and costs nothing extra per session-hour. DataParallel keeps batch 128 and computes the
   loss on the gathered batch. Their model has LayerNorm only (`models/model_utils.py`), no BatchNorm. At 64 per GPU,
   activations are ~7.5 GiB, so checkpointing may be unnecessary. Blockers to a runtime patch: `forward` moves every
   input to `self.device` (`model_XPert.py:188-198`); `self.drug_HG_embed` is a plain tensor created on `device` at
   `:135`; a DataParallel `state_dict()` carries a `module.` prefix. **Unmeasured estimate: ~7 min/epoch, ~25 GPU-h.**

## OPTIONS
- **O1** Full recipe, one T4, checkpointed, chained sessions with a full-state checkpoint. ~53 GPU-h, ~7 sessions,
  about two weeks of quota.
- **O2** DataParallel over both T4s + full-state checkpoint. First a measure-only v6 (~0.15 GPU-h) that (a) proves
  one DP step's gradients equal a single-GPU step's within the noise floor, dropout off, same batch, and (b) times it.
  Then ~25 GPU-h if the measurement holds.
- **O3** Capped single session. 8.3 GPU-h. Inadmissible unless XPert beats v9.
- **O4** Buy nothing. Compare v9 to the published 0.383 (a five-fold mean, their convention, their checkpoint selection)
  and keep the warm-split head-to-head we already ran on their released weights. Loses the per-cell paired estimand.

My recommendation is O2's measurement, then a decision between O2 and O4 on the measured number. O3 I would not buy.

## ASKS
1. Is point 1 right — is there any reading of 71.7 under which a 31-epoch guard stop could count? I do not want to
   rewrite 71.7 after seeing the cost; if you think it is wrong, say so and I will record that as post-hoc.
2. Is a full-state checkpoint across sessions a deviation that bears on "as published", given their own resume is
   lossier than continuous training?
3. DataParallel: is equality of one step's gradients (dropout off) the right proof? Dropout masks will differ per
   replica — does that matter for the claim?
4. Is ~25–53 GPU-h justified for one fold, one seed, given the objective in `state.json`? Or is O4 the honest answer?
5. Anything else, including whether TranSiGen (the next baseline in the plan) should be measured first because it may
   be cheaper.
