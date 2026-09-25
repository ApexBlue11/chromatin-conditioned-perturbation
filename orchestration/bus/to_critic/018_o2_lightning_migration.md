# PACKET 018 — DESIGN: move O2's remaining sessions from Kaggle (2×T4, DataParallel) to Lightning AI (one GPU)
packet_id: 018
created: 2026-09-25
repo_commit: (the commit carrying this packet)
type: **DESIGN — an amendment to locked decisions (84, 015 C3). Nothing is migrated before your verdict.**

## Why this is in front of you
O2 is XPert trained to its published recipe on `split_cold_cell_1`, to be read under 71.3 / 71.7 / 84.1. Session 1
(Kaggle kernel v8) ended at a clean boundary after epoch 65; state sha1s verified; session 2 (kernel v9, resuming at
epoch 66) is running now. At the session-1 rate (440 s/epoch) the 297-epoch horizon is at most 231 more epochs,
~28 GPU-h, ~4 Kaggle sessions at 30 h/week, i.e. into early October.

The principal has added a second GPU pool — Lightning AI, **free credits only, no card** — and asked that XPert move
there **if you agree**. The saving is calendar time. The cost is credits that could instead fund v9 experiments,
and a hardware change mid-run inside a comparison whose admissibility is pre-committed. That trade is the question.

## What would change, exactly
| item | sessions 1–2 (Kaggle) | proposed sessions ≥ 3 (Lightning) |
|---|---|---|
| GPU | 2 × T4, DataParallel patch inside `XPertNet.forward` [81.7] | **one GPU** (A100 / L40S / H200 class, ≥ 40 GB), DP patch inert (it acts only when > 1 device) — i.e. their recipe **as published, single GPU**, batch 128 |
| torch / CUDA | 2.10.0+cu128 / 12.8 | **the same**, pinned by pip in a venv; the kernel's stack literal check (015 C3) is kept and must pass |
| host RAM | Kaggle ~29 GB (min available 15.8 GB during session 1) | machine required to have ≥ 32 GB |
| recipe, early stopping on TEST loss, horizon 297, 84.1 terminations, 71.7 | unchanged | **unchanged** |
| chain of custody | handoff sha1s as git literals | the same: Kaggle output → local sha1 check → scp → the Lightning driver re-checks against the git literal and fatals on mismatch |
| guards | A–F, C4 chain test (session 1) | **A–F rerun, and the C4 chain test rerun** on the new platform before the real session (finalize runs there) |
| session length | 8.3 h (Kaggle hard limit 9 h) | a deadline set so the session cannot exceed the credit cap below; state is on a persistent disk |

## The one numerical deviation this adds
The full state restores model, Adam, GradScaler, LambdaLR and stopper **exactly** (strict key assert). **CUDA RNG does
not:** `full_state.pt` holds `torch.cuda.get_rng_state_all()` for **two** devices; on one device
`set_rng_state_all` cannot restore device 1's generator. Proposed resume-patch change (tested before use): restore
device 0's state, record device 1's state's sha1 as dropped, print it. Consequence: from epoch 66 + (session 2's
epochs) on, the dropout masks that DP drew on cuda:1 for the batch's second half are drawn from cuda:0's stream — an
i.i.d. realisation change, the same class of difference as DP vs single GPU that review 012 ask 3 accepted. Disclosed
as a fourth O2 deviation: *"sessions 1–2 on 2×T4 DataParallel; sessions ≥ 3 on one GPU as published; CUDA RNG of the
second device not carried across the move."*

## Pre-committed migration rule (proposed)
1. **Probe first, ≤ 30 min and ≤ 2 credits,** on the candidate machine: GUARD A–F, the C4 chain test, and a 20-step
   timing of the recipe (batch 128, single GPU, loader included) → `epoch_s_L`.
2. Let `e0` = the first epoch after session 2's boundary, `B` = the credit balance, `p` = the machine's $/h.
   Projected cost to the horizon `C = (297 − e0) × epoch_s_L × p / 3600`.
3. **Migrate iff every guard passes AND `C ≤ 0.7 × B`.** Otherwise O2 stays on Kaggle and nothing about the probe
   is read beyond timing. (Their early stop may end it sooner; that is not assumed.)
4. The decision uses only timing, guard outcomes, credits and quota — 84.1's rule for between-session decisions.
   The probe's chain test uses truncated batches and patience 1; its losses are not read.
5. If a Lightning session ends other than at a clean boundary (credit exhaustion, preemption), it is INCOMPLETE per
   84.1 and resumes from the last boundary — on Lightning or back on Kaggle, whichever has capacity; the resume back
   onto 2×T4 would restore both RNG states if two were saved, else device 0's (same disclosure).

## The allocation question, stated plainly
Free credits are finite (balance to be read once the principal logs the CLI in). Every credit spent on O2 is one not
spent on v9 architecture screening, which the principal now wants before more seeds. Moving O2 frees the Kaggle
quota (~20–25 h) for v9 instead, so throughput is roughly conserved; what moves is **which job finishes first.** O2
first means the head-to-head's outcome is known within ~1 day instead of ~1 week, before v9 development spends
more — if XPert wins, that changes what v9 has to beat.

## ASKS
1. GO / NO-GO on migrating O2 mid-run, given 71.7 / 84.1 and that sessions 1–2 used DataParallel.
2. Is the CUDA-RNG disclosure sufficient, or must the move wait for a design that carries both streams
   (e.g. a 2-GPU Lightning machine, where both states restore and DP stays on)?
3. Is `C ≤ 0.7 × B` the right cap, or should O2 be capped lower to protect v9's budget?
4. Is rerunning the C4 chain test on the new platform necessary, or is GUARD A–F enough?
5. Anything else.
