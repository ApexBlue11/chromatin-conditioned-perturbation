# REVIEW OF PACKET 011
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 50f49c7

**GO on activation checkpointing.** It's exact, and your proof tested the hard case: dropout is active at 0.1,
so a within-noise gradient match means the dropout masks were replayed. And it's the only one of the three
options that's exact at all — gradient accumulation changes their objective, and DataParallel needs edits to
their device handling (ask 4). Declare it as the fifth deviation; it doesn't bear on "as published".

**The caveat worth acting on before launch is C1.** The fix that makes the run *fit* may make it
*inadmissible*.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **Checkpointing slows every step, and under your own §71.7 that makes an inadmissible outcome more likely.** Recomputing `Encoder` and `crossEncoder` in the backward pass adds roughly one extra forward through those modules — on the order of a third more compute per step if they dominate. Fewer epochs fit inside the 8.3 h watchdog. §71.7 — which I asked for in 008 — says a watchdog stop with `counter_at_end < 50` supports **no** v9-win claim. So if unpatched XPert would have converged in, say, 6–7 h, the checkpointed run may be cut off while still improving, and the 8.5 h spend returns a result that can't carry the comparison it exists for. Nobody knows the convergence epoch (patience 50 on test `loss4`, which you've noted yourself), so this can't be ruled out by argument. | Time 3–5 steady-state training steps with the patch in GUARD F, which already runs one training step, so the cost is seconds. Record s/step, project epochs-within-budget, and write it into `run_record.json`. If the projection is marginal, decide *before* launch — longer budget, or accept the risk explicitly — rather than discovering it through `stopped_by == 'watchdog'`. |
| 2 | MINOR | stats | **Ask 3: 13.0 GiB is a sound ceiling, but GUARD F measures the wrong step.** It runs forward + backward with no `optimizer.step()`. Adam allocates its `m` and `v` states **lazily, on the first `step()`**, so GUARD F's peak omits the optimizer state that every real training step carries. For a model XPert's size that's small next to 8 GiB of headroom, so it's unlikely to matter — but a guard that measures a lighter step than the one it guards is the class of error §32 records. | Add one `optimizer.step()` inside GUARD F so its peak is the steady-state peak. Also confirm GUARD F's CUDA context is released before training begins; a subprocess exit does this, an in-process probe needs `torch.cuda.empty_cache()` and the tensors dropped. |
| 3 | MINOR | provenance | **The patch docstring cites the earlier proof run.** It quotes "gradient difference 5.9e-05 inside the 8.7e-05 run-to-run noise floor" — the inline-copy run. The run that imports the actual module gives 4.353e-05 within 4.630e-05. You disclose both in the packet, and both pass. | Update the docstring to the run of the committed module, so the file describes the code it ships with. |

## Answers to the asks

**Ask 1 — not a deviation that bears on "as published".** Checkpointing changes *when* activations exist, not
*what* is computed. Three checks, all passing:

- **The forward is identical.** The loss is `7.56921387` in the unpatched run, a second unpatched run, and the
  checkpointed run — equal to 8 decimals.
- **Gradients sit inside the GPU noise floor.** Two unpatched runs with the same seed already differ by
  4.630e-05; checkpointed differs from unpatched by 4.353e-05. The floor is atomic-reduction nondeterminism,
  most likely the memory-efficient SDPA backward, and it's present whether or not you checkpoint.
- **The proof exercised the RNG replay, not a trivial case.** `config_l1000.yaml` sets all four dropout rates
  to 0.1, and the proof runs in training mode. We know the checkpoint path actually executed because peak
  memory fell 3.8× (0.938 → 0.244 GiB at batch 8); a proof run in eval mode would have skipped the wrapper and
  proved nothing. Unreplayed dropout masks would put gradient differences orders of magnitude above 5e-5.

**Ask 2 — your reasoning is sound; the proxy loss is enough.** Checkpointing is exact if recomputation
reproduces every intermediate activation, and that's a property of the checkpointed layers, not of the loss
above them. The loss only sets the upstream gradient `dL/d(output)` fed into backward; exact recomputation
gives exact vector–Jacobian products for **any** upstream gradient. The proxy loss draws on the same three main
outputs as their loss, and 176 parameters receive gradients, so every checkpointed layer's backward is
exercised. Rerunning with `batch_weighted_loss` would cost nothing and settle the question for readers, but it
isn't needed for correctness.

**Ask 4 — checkpointing is right; both alternatives are worse, and your packet already contains why.**
- **Gradient accumulation is not exact for their loss.** For `epoch < init_epoch` (70) they train on
  `batch_weighted_loss`, whose terms are `sqrt(loss1/num_samples)` and `sqrt(loss3/num_samples)`. The square
  root is nonlinear, so `sqrt` of a batch-128 mean ≠ the mean of `sqrt` over four batch-32 micro-batches, and
  `num_samples` itself changes to 32. Accumulation would change the objective across the first 70 epochs. The
  post-70 `weighted_loss` is a sum of `*_ls_sum` terms and *would* accumulate linearly, but the pre-70 phase
  breaks it — and the best checkpoint can come from either side (`best_selected_before_init_epoch_70` records
  which). Checkpointing keeps the full batch-128 loss exactly.
- **DataParallel needs edits to their device handling.** `model_XPert.py:193` does
  `drug_feat.to(self.device)` with `self.device` fixed to one GPU, and `:135` holds `drug_HG_embed` as a plain
  attribute, not a buffer. `DataParallel` replicates parameters and buffers only, so a replica on `cuda:1` would
  still reference a `cuda:0` tensor. Fixing that means editing executed code, a larger deviation than a runtime
  wrapper.

## What I checked and found sound

- **The patch does what it says, and avoids a trap I checked for specifically.** It uses
  `cp.checkpoint(orig, self, *a, use_reentrant=False, **k)` only under `self.training and
  torch.is_grad_enabled()`. `use_reentrant=False` is the right variant: it replays both the RNG state (with
  `preserve_rng_state` at its default of True) and the autocast state, which the reentrant variant has
  historically mishandled. The `make(orig)` factory binds each class's own `forward`. A closure over the loop
  variable would have bound late and sent `Encoder` into `crossEncoder.forward` — silently, since both return
  compatible shapes. The `_lincs_checkpointed` flag makes it idempotent.
- **RNG consumption is preserved.** Non-reentrant checkpointing restores the global RNG after recomputation, so
  the sequence of dropout masks and shuffles the rest of training sees matches an unpatched run. Checkpointing
  adds no RNG divergence beyond 009 C3's independent draw.
- **Validation and early stopping are untouched.** `validate()` runs in `model.eval()`, where the wrapper takes
  the original path, so test `loss4` — the quantity the stopper monitors — comes from the unmodified forward.
- **Runtime patching keeps their files verbatim**, `run_train_ckpt.py` relies on `train_xpert.py:733`'s
  `__main__` guard to import without executing, and the AST-literal comparison of `CKPT_PATCH_SRC` against the
  repo file is the same check that made 009's proof trustworthy.
- **The diagnosis is well evidenced.** Recording host `MemAvailable` (minimum 18.5 GB) separates this GPU OOM
  from 009's host OOM. The traceback lands in `selfEncoders` on the first training forward. GUARD F having run
  under `torch.no_grad()` is exactly why it missed this. The measured 0.114–0.117 GiB/sample reproduces the
  14.6–15.0 GiB projection against the T4's 14.56.

## What I could not assess, and why

- **The actual slowdown and XPert's convergence epoch.** C1 is the risk; only the measurement it asks for
  settles it.
- **How faithfully the RTX 3050 with `enable_flash_sdp(False)` stands in for a T4.** The memory-efficient SDPA
  path is the likely T4 choice, and its 0.117 GiB/sample agrees with PyTorch's own pick (0.114). GUARD F's
  on-T4 measurement is the real check, and that's what it's there for.
- **XPert's parameter count**, so I can't size the Adam state C2 asks to include — only say that it's small
  relative to the headroom.
