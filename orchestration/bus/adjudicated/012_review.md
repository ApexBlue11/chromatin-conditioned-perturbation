# REVIEW OF PACKET 012
verdict: SOUND-WITH-CAVEATS
reviewed_commit: b85f835

**Your analysis holds, and so does your recommendation: measure O2 first (~0.15 GPU-h), then choose between O2
and O4 on the measured number. Don't buy O3.** Point 1 is correct, and I reach it independently.

Two things need settling before any multi-session run. A trap in their resume path that would silently throw
away a session's training (C1). And a scoping amendment to §71.7 that is **not** the post-hoc rewrite you're
right to refuse (C2).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **Their resume can silently restart from random initialisation while logging success — and DataParallel triggers it.** `train_xpert.py:484-486` loads only the keys already present in the model: `pretrained_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict}`, then `model_dict.update(pretrained_dict)`. A DataParallel checkpoint prefixes every key with `module.`, so **no key matches**. `pretrained_dict` is empty, the model keeps its fresh weights, `load_state_dict` succeeds, and `:489` logs *"Load previous-trained parameters sucessfully!"*. In a chained run, any session that resumed through this path would restart from scratch with a log line saying otherwise. `save_checkpoint` writes the wrapped model's `state_dict()`, so the prefix will be there. | Never route a session boundary through their `--resume_from`. Your full-state checkpoint must strip or add `module.` explicitly, and it must **assert** that the set of loaded keys equals the model's key set — a strict load, not their filter. Test it before launch: save one DP checkpoint, reload it into a fresh model, and require bit-equal parameters. |
| 2 | MAJOR | stats | **§71.7 was written for one session. In a chained run it needs re-scoping before launch — and that's a design amendment, not a post-hoc rewrite.** There, the wall-clock guard *was* the end of training. Chained, the per-session guard fires every ~8 h and is a **resume point**, not a stop. As written, a reader could classify session 1's guard as `stopped_by: watchdog` with `counter_at_end < 45` and declare the whole run inadmissible. You're right to refuse rewriting §71.7 after seeing the cost. The line: *lowering the 45 threshold to admit a capped run* would be post-hoc and wrong; *stating that in a multi-session design the admissibility row applies to how training **finally** ended, with the threshold unchanged* is a pre-launch clarification made before any data exist. | Commit it before launch: "In a multi-session run, §71.7's `stopped_by` and `counter_at_end` refer to the final termination. Per-session guard fires are checkpoint-and-resume events and are logged separately. The ≥ 45 threshold is unchanged." Record per-session guard fires in `run_record.json` so the distinction is auditable. |
| 3 | MINOR | overreach | **Point 2 overstates how informative O3 could be — it supports no claim in either direction.** You call "an under-trained XPert that still beats v9" the informative outcome. But §71.3, committed in 008, reads an XPert win as **uninterpretable**, because XPert's checkpoint is selected on test loss. That holds in a capped run too: `save_checkpoint` fires on every test-loss improvement, so the checkpoint loaded at epoch 31 is still the best on test. A v9 win is inadmissible under §71.7; an XPert win is uninterpretable under §71.3. This strengthens your recommendation not to buy O3. | Record O3 as uninformative in both directions under the pre-committed rules, not in one. |
| 4 | MINOR | stats | **Ask 3: turn dropout off by setting p = 0 in training mode, not with `model.eval()`.** `eval()` switches the checkpointing wrapper off — it fires only under `self.training` — and any other train-only branch with it. The proof would then exercise a different code path from the one training runs. If O2 drops checkpointing (plausible at 64 per GPU) the difference shrinks, but p = 0 in train mode proves the path you'll actually run either way. | Zero all four dropout rates (`attention_probs`, `hidden`, `cell_input_hidden`, `drug_input_hidden`) in the config, stay in `.train()`, then compare one DP step with one single-GPU step. |
| 5 | MINOR | provenance | **The full-state checkpoint has to carry two things that are easy to miss.** First, the **on-disk best checkpoint** that `save_checkpoint` writes on each test-loss improvement. `/kaggle/working` doesn't persist between sessions. If the best epoch fell in session 1 and session 2 never beats it, that file is gone, and the stopper's `best_score` points at a checkpoint that no longer exists. Second, **resume only at epoch boundaries**: the shuffle order comes from the RNG at the start of each epoch, and a mid-epoch resume can't reproduce it. | Persist the best-checkpoint file as session output and restore it with the rest of the state. Prove resume equivalence before launch: N epochs straight through against N/2, save, restore, N/2. Losses and parameters should agree within the GPU noise floor. |

## Answers to the asks

**Ask 1 — point 1 is right, and no reading of §71.7 rescues a 31-epoch run.** `counter_at_end = last_epoch_index −
best_epoch`, so in 31 epochs it's at most 30 (best at epoch 0), which is under 45. Early stopping can't fire either,
since the counter can't reach 50 in 31 epochs. So a single-session run can only end by the guard, and it's
inadmissible by construction. It never reaches the epoch-70 objective switch, either. Don't rewrite §71.7. C2 is the
only change that's warranted, and it leaves the threshold alone.

**Ask 2 — a full-state checkpoint is more faithful than their resume, so it doesn't bear on "as published".** Their
published runs trained continuously; their `--resume_from` would *break* continuity. I confirmed the losses you list:
Adam state not restored (`:487`, commented out); `LambdaLR` built fresh at `:460`, before the resume at `:480`, so its
internal epoch restarts at 0 and the LR returns to 0.004 for another 40 epochs after `:462`'s halving should already
have happened; and `EarlyStopping` constructed fresh, so its best score and counter reset. A full-state checkpoint
reproduces continuous training up to GPU nondeterminism; their resume would be the deviation. Declare it, and satisfy
C1 and C5.

**Ask 3 — equal gradients on one step with dropout off is the right proof, and differing dropout masks don't matter
to the claim.** DataParallel is exact where accumulation wasn't: it **gathers the outputs and computes the loss on the
full batch of 128**, so `batch_weighted_loss`'s `sqrt(loss/num_samples)` terms see all 128 samples, exactly as on one
GPU. With LayerNorm only, as you confirmed, each sample's forward is replica-independent. Dropout masks will differ per
replica, but each element is still an independent Bernoulli(0.9) draw — the same distribution with a different
realisation, which is what a different seed already is (009 C3). The deterministic computation is what needs proving,
hence dropout off (C4); the equivalence with dropout on follows from the masks being i.i.d.

**Ask 4 — your recommended path is right, but O4 isn't "the honest answer" unless it's labelled as not a head-to-head.**
This run is the one experiment that turns `state.json`'s objective from inadmissible into admissible (review 001 C8:
fold 1 against a five-fold mean, different rows, no XPert run). That is high value. O4 — setting v9 against the
published 0.383 — *is* the inadmissible comparison, so it's honest only if the paper says in so many words that it is
not paired, is on different rows, and sets one fold against a five-fold mean with their checkpoint selection. On the
prior: a fully trained XPert near 0.383 against v9's 0.4734 is a gap of ~0.09. That's far above the cold-cell seed sd
of 0.0052, and likely to satisfy §71.3's conjunction, so an admissible run is probably informative rather than
wasted. Buy the ~0.15 GPU-h measurement. If O2 measures near 25 GPU-h, it's the most objective-relevant spend
available.

**Ask 5 — TranSiGen can't substitute for XPert, and it inherits the same problem.** Its published cold-cell score is
0.293 ± 0.017, against XPert's 0.383. It's the weaker baseline: beating it says less, and a "beats SOTA" claim needs
the strongest one. It also trains through the same `train_xpert.py` (`TranSiGen` is in its model list, and
`MyDataset` has `transigen_sdst` branches), so it goes through `utils.py:133`'s val=test fallback — **also
test-selected**, with the same §71.3 caveat. If it's cheap, run it as a secondary. It doesn't replace the XPert
head-to-head, and it shouldn't be run first for being cheaper.

## What I checked and found sound

- **The measurement does what 011 asked.** GUARD F now includes `torch.optim.Adam` + `GradScaler.step`, so the Adam
  state is in the 3.74 GiB peak (011 C2), plus five timed steady-state steps (011 C1). The arithmetic holds: 2.265 s ×
  372 batches ≈ 842 s of training plus 0.411 s × 167 ≈ 69 s of validation gives ≈ 911 s per epoch, 92.5 % of it
  training. 8.3 h less 900 s reserved, less 208 s of setup, divided by 911 s gives ≈ 31.6 epochs.
- **The released-checkpoint epochs are the right anchor.** The warm-split checkpoint at epoch 164 with patience 50
  means that run lasted at least 214 epochs. You present it as an anchor, not a prediction, and that's correct: no
  cold-cell checkpoint was released, and cold-cell may converge differently.
- **The DataParallel blockers are the real ones.** `forward` moving every input to a fixed `self.device`
  (`model_XPert.py:188-198`), `drug_HG_embed` held as a plain tensor rather than a buffer (`:135`) — DataParallel
  replicates parameters and buffers only — and the `module.` prefix. C1 shows the prefix is worse than an I/O
  nuisance.
- **The 7 min/epoch estimate is plausible.** Halving each GPU's batch and dropping the checkpoint recompute takes
  training to roughly 842 × 0.5 × 0.75 ≈ 316 s. Add DataParallel scatter/gather and the GPU-0 loss overhead, plus
  validation, and you land near 7 min. But DataParallel's GPU-0 imbalance and Python overhead often cost more than
  estimated, which is why measuring first is right.
- **Refusing to rewrite §71.7 after seeing the cost.** That's the discipline this bus depends on. C2 is written so
  you can keep it.

## What I could not assess, and why

- **XPert's convergence epoch on cold-cell.** 164 is warm-split, and nothing released bounds cold-cell.
- **O2's real per-epoch time.** Only the measure-only run settles it.
- **Whether a fully trained XPert on fold 1 lands near 0.383.** The prior in ask 4 assumes it does; one fold can sit
  a between-fold sd (0.027) or more away from the five-fold mean.
