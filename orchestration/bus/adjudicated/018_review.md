# REVIEW OF PACKET 018
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 1187792

**GO on migrating, provided the conditions in C2–C4 are met.** Migrating doesn't touch anything §71.3, §71.7 or §84.1
read: the recipe, test-loss early stopping, the 297-epoch horizon and the terminations all stay fixed. What it changes
is rounding-level numerics: a different GPU architecture, different GEMM and SDPA kernels, and one device instead of
DP. The function is proven unchanged (81.1a, float64). A trajectory whose rounding changes partway through is one
realisation among the many that seeds and atomics already produce. It can't be steered toward either side, because
the decision is taken before any post-migration number exists and uses no outcome.

But first, something the session-1 state reveals about sessions 1–2, and about a claim I made in 012.

## The two halves of every DP batch shared dropout masks for all of epoch 0

`external/kaggle_out/cc1_v8/state_ds/full_state.pt`, saved at epoch 65, holds both CUDA generator states:

| device | seed | Philox offset |
|---|---|---|
| cuda:0 | 2024 | 99,602,362,035,000 |
| cuda:1 | 2024 | 99,598,169,134,992 |

- **Both offsets are exact multiples of 66,** the number of epochs completed.
- **They differ by exactly 66 × 63,528,788,** and 63,528,788 is one row's worth of attention-dropout counter
  consumption. Per-row consumption solved independently from each device's total gives 63,528,802, the same value
  within per-call rounding.
- **That difference has one explanation.** The epoch's ragged last batch (47,509 = 371 × 128 + 21) splits 11/10, so
  cuda:0 draws one extra row's worth per epoch. Nothing else consumes either generator.

Therefore **both generators started at (2024, 0) and advanced in lockstep through every full step until the end of
epoch 0.** `torch.manual_seed` (their `set_random_seed`, `utils.py:311`) seeds **every** device with the same seed. The
two replicas run identical op sequences on identically shaped tensors. So for epoch 0's 371 full batches, rows *j* and
*j* + 64 received **identical dropout masks**: attention and hidden alike. From epoch 1 the offsets differ by *k* rows'
worth, and the masks decorrelate.

**This corrects my 012 ask 3,** where I told you DP's masks were "an independent Bernoulli(0.9) draw … a different
realisation, as a different seed is". Each element's *marginal* is Bernoulli(0.9). The *joint* was not independent
across replicas in epoch 0. I asserted it without checking how the devices were seeded. **Sixth error on this bus.**
`xpert_dp_patch.py`'s docstring carries the same claim.

**What it means:** one epoch of ~100–297 trained with dropout noise correlated between paired rows, with the function
and objective unchanged. It's a small realisation-level deviation, very unlikely to move the final checkpoint in either
direction, but the record currently says the opposite (C1). **It makes migration cleaner, not murkier:** on one GPU,
every mask comes from one stream, i.i.d. as published.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | provenance | **The O2 deviation record states i.i.d. DP dropout, and epoch 0 contradicts it** (above). | Amend the DP deviation: *"Both GPUs' generators were seeded identically (`torch.manual_seed` seeds all devices) and ran in lockstep through epoch 0, so the two halves of each batch received identical dropout masks in epoch 0; the ragged final batch desynchronised the streams from epoch 1 (offsets at epoch 65: Δ = 66 × 63,528,788)."* Fix the docstring. Don't change seeding mid-run. |
| 2 | MAJOR | code-vs-intent | **Rule 5's fall-back to 2×T4 must never give both devices the same state, or it recreates the lockstep for a whole session.** After Lightning sessions, only one CUDA state exists. If the fall-back ever copied it to both devices — the tempting reading of "else device 0's" — every batch's halves would share masks for the whole session. Restoring device 0 alone leaves cuda:1 at the fresh (2024, 0), which replays session 1's epoch-0 counters. That's harmless (different step, different rows), but it should be deliberate. | In the resume patch: when restoring onto more devices than were saved, restore the saved states to the lowest indices, seed every other device from a **recorded, distinct** seed (for example 2024 + 1000·device), log it, and **assert that no two device states are equal** after restore. Include this in the tested patch change. |
| 3 | MINOR | code-vs-intent | **Pin the fp32 paths on Ampere-class hardware.** XPert sets no TF32 flags, and I searched their tree. On A100/L40S/H200, PyTorch's defaults keep fp32 matmuls at full precision, but an image or environment variable can override that (`NVIDIA_TF32_OVERRIDE`, `torch.backends.cuda.matmul.allow_tf32`, `set_float32_matmul_precision`). The recipe is fp16 autocast, but the fp32 parts (loss, reductions, and anything autocast keeps in fp32) would silently change precision class. On T4 the question couldn't arise. | Add a guard: `torch.backends.cuda.matmul.allow_tf32 is False`, `torch.get_float32_matmul_precision() == 'highest'`, and `NVIDIA_TF32_OVERRIDE` unset. Record the SDPA backend selected (flash on sm80+, not T4's memory-efficient) as part of the declared hardware change. Keep the flash_attn **shim**, and don't install the real package mid-run: identical code across sessions matters more than being closer to their original kernel. |
| 4 | MINOR | provenance | **A persistent disk on a free-credit account can strand the state.** If credits run out mid-session, the studio may not restart until more credits exist, and the only copy of the latest boundary would sit on a disk you can't reach. §84.1 makes that session INCOMPLETE and resumable, but only if the state is reachable. | Copy the state off the machine at every boundary (a sync to the laptop is enough), with the same git-literal sha1 recorded before the next session starts, even if the next session is also on Lightning. Set the session deadline so it ends with enough credit left to copy the final state out. |

## Answers to the asks

**Ask 1 — GO, conditional on C2–C4 and a declared deviation.** Add to the deviations: *"sessions 1–2 on 2×T4
DataParallel; sessions ≥ 3 on one <GPU model> as published; different GPU architecture and SDPA backend
(rounding-level); CUDA RNG of the second device not carried across."* Beyond that:
- **Configuration.** On a ≥ 40 GB card the recipe fits with no activation checkpointing (~15 GiB). Run it **unpatched
  apart from** the shim, the memory patch, the frozen ten and the resume hooks, as sessions 1–2 did (`dp`, no
  checkpointing). Don't add checkpointing by default.
- **Driver.** If a driver asserts two GPUs, replace the assertion with an explicit platform expectation. Don't just
  delete it.
- **Record.** State that the migration was decided on timing, credits and guards alone (§84.1 (iii)), without
  reference to sessions 1–2's logged losses.

**Ask 2 — the disclosure is sufficient, and waiting for a 2-GPU machine buys nothing.** Carrying cuda:1's stream only
preserves something if the computation is bitwise continuous. A change of GPU architecture breaks that anyway, and so
does every atomic on the T4s. What matters statistically is no reuse of masks within a step and shuffle-order
continuity. The CPU state, which sets shuffle order and worker seeds, restores exactly. Dropping cuda:1 while
continuing cuda:0 introduces no within-step reuse. Single-GPU masks are, if anything, *more* faithful to the published
recipe than DP's epoch 0.

**Ask 3 — the rule's structure is right. The level is an allocation call, and it's yours and the principal's.**
`C ≤ 0.7 × B` uses only non-outcome inputs and is fixed before the probe. Two refinements:
- **`C` should cover all the costs:** the probe, each session's guards and setup, the final prediction (~10 min), one
  lost epoch per session as margin, and any rerun of the chain test.
- **Re-price after the first two epochs.** Carry Amendment E over to Lightning: if the real epoch time is more than
  25 % over `epoch_s_L`, re-price before continuing. A 20-step timing is a thin basis for a multi-hour credit spend.

On the level itself, I'd only note that O2's outcome changes what v9 has to beat. That supports your sequencing
argument, but it isn't a reviewer's call.

**Ask 4 — yes, rerun the C4 chain test.** It's the only check that exercises marker → termination →
`counter_at_end == patience` → prediction end to end. On Lightning the driver and the host are both new, and the final
prediction will run there. It's minutes of truncated batches. GUARD A–F check the data and the model, not the
session-end path.

**Ask 5:** C1 and C2 are the new items. Also, restoring onto a different architecture should pass the in-process
81.3a round-trip, which is already in the production patch per 015 C3. It's the cheapest proof that every field
survived the move. The Philox state layout is device-independent, so it will, but the check costs seconds.

## What I checked and found sound

- **The state file's provenance** as far as it bears here: epoch 65, two CUDA states, both seeded 2024. The offset
  arithmetic above uses nothing else.
- **Migration touches none of §84.1's terminations.** A platform change is neither early stopping nor the horizon.
  Rule 5 keeps an interrupted Lightning session INCOMPLETE and resumable, which is §84.1's own classification.
- **The probe's losses are not read** (rule 4), and the chain test uses patience 1 on truncated batches, so nothing
  from it can leak into the comparison.
- **Hardware support:** `torch 2.10.0+cu128` supports sm80, sm89 and sm90, so the stack-literal check (015 C3) can
  pass unchanged on all three candidate GPUs.

## What I could not assess, and why

- **The exact counter mapping of the SDPA dropout kernels.** From epoch 1 the streams are offset by *k* rows'
  worth. Under either plausible mapping, the reused random bits land on different attention entries or different
  subsequence positions, so I read epochs ≥ 1 as effectively independent. Epoch 0's identity doesn't depend on the
  mapping: same seed, same offset and same shapes give the same masks.
- **Lightning's behaviour at zero credits** (C4), and the credit balance.
