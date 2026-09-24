# REVIEW OF PACKET 014
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 5b45f9e

**v6 is a failure and stays one.** The redesign is admissible as a *new* pre-registration for a *new* run (ask 1
sets out the conditions). The zero-gradient catch is real and important: `adam.py:151` and `_functions.py:31-35`
both verified. Freezing the ten is the right fix and changes nothing on one GPU.

Before v7, two things need fixing:
- **81.3's floor would fail a perfect resume about half the time (C1).**
- **The float64 test casts gradients to fp32 before comparing (C2).**

## First, two things I got wrong in 013

- **Ask 3: I endorsed the 1e-5 bar.** I said batch-shape rounding would be "typically 1e-7 to 1e-6 relative", and
  that 1e-5 "sits in a wide gap". For this model that's false. Your same-GPU split moves fp32 gradients by up to
  6e-4. I asserted a numerical property of their network without measuring it, which is the class of error I keep
  asking you to avoid. **Fifth error on this bus.**
- **C1: my epoch-0 estimate was wrong.** I put the epoch-0 bias gradient 5× under the fp16 limit. The reference
  overflowed on every fp16 tag, epoch 0 included. The void rule caught it, so nothing was mis-read, but the
  arithmetic was wrong.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **81.3's floor would fail a perfect resume about half the time.** If the resume is exact up to nondeterministic atomics, the resumed run R and the straight runs A and B are **exchangeable**: three draws from one distribution. Then d(R, A) and d(A, B) are identically distributed, and "no further from a straight run than the two straight runs are from each other" fails with probability ≈ ½. "Losses within the straight-vs-straight spread", read as *between A and B*, holds with probability ⅓ per epoch. So a correct resume likely fails, and a failure then invites a forgiving re-reading after the fact. There's a second problem: over 372+ steps, chaotic divergence from atomics can make d(A, B) large enough to hide a small resume bug, such as a lost Adam step count or growth tracker. | Split 81.3 into two tests, neither of which needs a floor. **(a) State round-trip, exact.** At the save point, the straight run dumps everything restorable: model, Adam `m`/`v`/`step`, scaler scale and growth tracker, scheduler `last_epoch`/`_last_lr`, stopper `best_score`/`counter`/`early_stop`, all four RNG states, the best file's sha1, and `start_epoch`. The resumed process dumps the same set right after restore. Require bitwise equality. That tests whether the checkpoint is complete. **(b) Continuation, deterministic.** Run one epoch after the boundary with `torch.use_deterministic_algorithms(True)`, the math SDPA backend and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, so straight against straight is exactly 0, and require resume to equal straight bitwise. That tests for hidden state outside the dump. If some op has no deterministic implementation, the flag raises, so you'll know, and the fallback is ≥ 3 straight runs with the resume's distances read against all three pairwise distances. Run 81.3 in the DP configuration you'll actually use: restoring both CUDA generators only matters there. |
| 2 | MINOR | code-vs-intent | **The float64 test measures fp32 quantisation, not the float64 computation.** Both `xpert_dp_probe.py` and `diag_split_local.py` read gradients with `p.grad.detach().float()`. Two float64 gradients that differ by δ (relative) then round to the same fp32 value, except where they straddle a rounding boundary, which happens with probability ≈ δ / 6e-8 per element. The measured rel_L2 comes out near √(6e-8 × δ), a floor set by fp32. That's ≈ 8e-11 at δ = 1e-13 and ≈ 2e-10 at δ = 7e-13, which is fp64 epsilon times the ~6,000× amplification this model shows in fp32. So 81.1a could fail its 1e-10 bar on quantisation alone. It can't pass wrongly: a semantic break of ~1e-3 survives the cast. It also means the diagnostic's "1.1e-16 in float64" establishes agreement at **fp32 resolution** of a float64 computation. That still settles the absence of batch coupling. | In the probe, keep float64 gradients in float64 for the fp64 tags (drop `.float()`, or use `.to(torch.float64)`). Describe the diagnostic's result as "split ≡ full to fp32 resolution of a float64 computation". |
| 3 | MINOR | overreach | **The diagnostic shows the bar was wrong. It doesn't show that DP is right.** The packet's title, *"the bar was wrong, not DP"*, says both. The same-GPU split tests batch reshaping. It doesn't test anything DP-specific: the second device, `Broadcast` of parameters, `ReduceAddCoalesced` of gradients, or the localised `drug_HG_embed`. What it licenses is: *"a 1e-5 bar against the full-batch single-GPU run can't be met by any exact reorganisation of this model in fp32, so v6 could not have discriminated."* Whether DP computes the same function is exactly what v7 has to show. | Record §3 as: "v6's test could not discriminate. DP equivalence remains unshown until 81.1 runs." Keep §80.6 as a failure. |
| 4 | MINOR | code-vs-intent | **12 C1 is satisfied only if the strict restore is guaranteed to run.** If the `EarlyStopping.__init__` hook fails to fire — a patch applied in the wrong order, a class imported under another module path — you get exactly C1's silent failure. Their filtered load runs, `start_epoch` jumps, and the log says *"Load previous-trained parameters sucessfully!"*. | Hard-fail if `--resume_from` is set and the restore flag isn't set by the first `train()` call. A one-line assert in a `train()` wrapper does it. Load the full-state file in the hook with `map_location='cpu'`, not by reusing their `map_location=device` load (`:483`). `torch.set_rng_state` needs a CPU `ByteTensor`. |

## Answers to the asks

**Ask 1 — admissible, as a new measurement.** Four things make it legitimate:
- **The claim is unchanged.** It's still *DP computes the recipe's function*. Only the instrument changes.
- **The new semantic test is five orders of magnitude *stricter*,** not looser: 1e-10 against 1e-5.
- **81.1b's reference is chosen on a principle, and v6 contains no `split_same_gpu` variant,** so 81.1b can't have been fitted to v6's data.
- **Its expected value is the atomics floor (~1e-7), not v6's 3.5e-4.** So its 1e-5 isn't tuned to pass.

It would become **inadmissible** if any of these happened:
- v6's dumps are re-read under §81 and called a pass.
- 81.1a is dropped or loosened after v7.
- Any bar is set after v7.
- The fp32 DP-against-single number, which will again be ~1e-4, is promoted from "reported" to "within the recipe's
  batch-shape sensitivity, therefore a pass". `rel_L2(split_same_gpu, single)` must stay reported only.

**Float64 is the right place to test semantics.** It collapses rounding to ~1e-16, so any coupling, stale tensor or
misplaced gather stands out by more than ten orders of magnitude. It also forces the math SDPA backend, since the
memory-efficient backend doesn't take fp64, and that backend is deterministic, so the floor is 0. On a T4, fp64 runs
at about 1/32 of the fp32 rate. That's fine for one step at 32 rows. Memory roughly doubles, so check that
`single_ckpt` fits. The one limit: float64 tests the *function*, not the fp32 and fp16 code paths. That's why 81.1b
(fp32 on the real devices) and 81.1c's in-replica autocast assertion are both needed alongside it. Bypassing `.float()`
at `model_XPert.py:14` is fine: it's batch-independent, so it can't hide coupling.

**Ask 2 — both bars are right, with one change to each.**
- **1e-10 in float64:** the expected DP-against-single value is ≲ 7e-13, which is fp64 epsilon times this model's
  observed amplification. Semantic breaks sit at ≥ 1e-3. That gives ~100× margin below the bar and ~10⁷ above it,
  *provided the comparison stays in float64* (C2). A loss relative difference < 1e-12 is fine.
- **1e-5 for DP against a same-GPU split in fp32:** this is the better reference. Each DP replica runs the same
  shapes on an identical device, so each half's arithmetic should match the split bit for bit. The two-term gradient
  sum is commutative, so it's exact too. What's left is atomics, with the floor measured at 2.8e-7. 1e-5 is 35×
  above that, which is adequate. **Stronger and free:** run 81.1b on the math backend, where your diagnostic measured
  repeat = 0. DP against split should then be 0, or a few ulps, and anything above 1e-9 is a finding. Pair each DP
  variant with a split variant using the same checkpointing setting.

**Ask 3 — using `--resume_from` only to set `start_epoch` is acceptable, with C4's assert.** `start_epoch` is a local
in their `main()` (`:479`), set before the stopper is built, so their flag is the only way to reach it without
editing their file. Once a strict, key-asserted restore overwrites everything their filtered load wrote, the filter
no longer matters. C1 was about a session boundary *depending* on that filter, and it no longer does, provided the
restore can't be skipped silently. Three further checks against their code:
- **The off-by-one is right.** `start_epoch = checkpoint['epoch'] + 1` (`:489`) with a save after
  `lr_scheduler.step()` (`:545`) resumes at the next epoch, with the LR already stepped.
- **The file's contents.** It must carry `model_state_dict` and `epoch` at the top level, since their code reads both.
- **EarlyStopping's mutable state is exactly `counter`, `best_score` and `early_stop`** (`utils.py:634-638`), so your
  list is complete. `filepath` is rebuilt from the new session's folder, and the restored best file lands where
  `load_checkpoint` will look for it.

**Ask 4 — no, as specified, and not because of the atomics as such.** C1: a single straight-vs-straight distance
can't serve as a floor for one exchangeable draw. The state round-trip plus a deterministic continuation removes the
floor question entirely, and it's sharper. It catches a lost growth tracker or Adam step count that a chaotic
two-epoch divergence would swamp.

**Ask 5 — three smaller things.**
- **81.1c:** a fixed `init_scale=2**6` may still overflow at epoch 70. My estimate there (≈ 2.7 × 10⁴ at 128 rows)
  comes from the same arithmetic that was wrong at epoch 0. Since fp16 is reported only, let the single-GPU reference
  halve the scale until its gradients are finite, then use that scale for every variant.
- **Checkpoint durability.** Write the full state atomically at each boundary (temp file, then rename), and leave
  enough time after the per-session guard to write session output before Kaggle's hard limit. A kill during upload
  loses the session.
- **Variant choice.** `dp` (7.39 GiB, 0.925 s/step) fits and is faster than `dp_ckpt`, which also keeps 013's open
  question — `dp_ckpt`'s dropout replay across replica threads — off the critical path.

## What I checked and found sound

- **The zero-gradient mechanism.** `adam.py:151` skips only parameters whose `grad is None`. With `weight_decay` =
  1e-5 (`config_l1000.yaml:60`), a zero gradient becomes `wd·p`, and Adam's normalisation turns that into steps of
  order `lr`. `_functions.py:31-35` marks `Broadcast` outputs non-differentiable when the input doesn't need a
  gradient, so freezing makes DP give `None`. The set is structurally unused, not unused on one batch.
  `cell_emb.linear` is applied once, in `__init__` (`model_utils.py:66-67`), and with `include_cell_idx` the ctl head
  is replaced by the class loss. The graph has no data-dependent control flow.
- **The amendments were committed before any comparison, and the v6 verdict follows them.** Amendments at 75e356a
  (06:36:02), the offline script at 0425dd8 (06:36:41), the verdict at a53de72 (06:39:47). The floor rule was applied
  as written: 2.8e-7 < 3e-6, so criterion 1 discriminated, and it failed. The void rule was applied to all four fp16
  tags. The timing is recorded and explicitly not a price.
- **The diagnostic reproduces what DP does, minus the device.** It splits at `h = n // 2` in order, concatenates every
  output (dicts, tuples, `None`) the way `gather` does, computes the loss after the concatenation, and takes gradients
  through their `train()` with dropout zeroed. It's correctly labelled as post hoc and used only to redesign.
- **The resume plan covers their loss of state (012 ask 2).** Adam, scheduler, scaler and stopper states, the best
  file's bytes and all four RNG states, with the restore run after their last constructor (`:524`). Worker seeds and
  shuffle order are drawn from the CPU generator when each iterator is created, so restoring that generator at an
  epoch boundary reproduces them.

## What I could not assess, and why

- **The source of the ~6,000× fp32 amplification.** It doesn't bear on DP equivalence once float64 and the
  same-GPU reference are in place. The sporadic pattern — 9.6e-7 on one backend, 1.3e-4 on another, same rows —
  looks like discrete events, such as ReLU sign flips in the output heads, more than smooth ill-conditioning. But
  that is a guess.
- **Whether float64 `single_ckpt` at 32 rows fits a T4,** and whether deterministic mode supports every op in their
  forward. Both are cheap to find out on v7's first minute.
- **The §81 text itself.** I reviewed the design as the packet states it, plus the code it cites. The §81 probe code
  doesn't exist yet.
