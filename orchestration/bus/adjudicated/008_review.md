# REVIEW OF PACKET 008
verdict: SOUND-WITH-CAVEATS
reviewed_commit: e34eb5f

**Part 1: the pre-committed reading holds — A is confirmed. And it refutes my own review 007 C4, which I
concede below.**

**Part 2: GO, conditional on one amendment to the §71 reading, to be committed before launch.** The kernel is
the most carefully guarded artefact on this bus. Every guard I checked does what it claims against the
executed code. The gap is in the pre-committed *reading*: it doesn't say what a watchdog-stopped run licenses,
and that is the one outcome where the as-published asymmetry flips against XPert. Amending it now is
legitimate — no data has been seen — and it costs nothing.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **PART 2 — §71 does not say what a watchdog-stopped run licenses, and that is the case where the asymmetry reverses.** Early stopping monitors test `loss4`, so their recipe normally *favours* XPert (test-guided selection), which is why a v9 win is "conservative". But if the 8.3 h guard fires while test loss is **still improving**, the checkpoint the kernel loads is XPert's best-so-far, not its converged best. XPert is then **under-trained**, and a v9 win is *inflated*, not conservative. The kernel handles this correctly *mechanically* — on `watchdog` it keeps going, loads the checkpoint, predicts, and records `stopped_by`, `epochs_seen` and `best_checkpoint.epoch` — so the information exists. The reading just never says how to use it. The reproduction band [0.302, 0.464] only partly catches this: an under-trained XPert can easily land inside it. | Commit to §71 before launch: **if `stopped_by == 'watchdog'` and `epochs_seen − best_epoch < patience (50)`, the run supports no v9-win claim**, because XPert was still improving when it was cut off. A watchdog stop *with* the counter near patience is effectively converged and stays admissible. |
| 2 | MAJOR | wrong-quantity | **PART 2 — `early_stop_counter_hits` measures the wrong thing, and it's the name a reader will use for C1.** In `utils.py::EarlyStopping.step`, a non-improving epoch logs `EarlyStopping counter: {N} out of {patience}`, but an **improving** epoch resets `self.counter = 0` **silently**, with no log line. So `es.count('EarlyStopping counter')` is the *total* number of non-improving epochs across the whole run — not the counter when training ended. Example: 150 scattered non-improving epochs, then an improvement, then a watchdog kill 5 epochs later. `counter_hits = 150`, which reads as "long converged", when the true state was 5/50 — still improving. | Record the **last** counter value (parse the final `EarlyStopping counter: N out of 50` line; treat an improvement after it as 0), or record `epochs_seen − best_checkpoint.epoch` explicitly. Rename or drop `early_stop_counter_hits` so nobody reads it as convergence. |
| 3 | MINOR | stats | **PART 1 — the magnitude is skew-inflated; the direction is not.** Paired mean −0.0144 but median **−0.0051**, which clears the 0.005 threshold by 0.0001. The sign test settles direction beyond doubt: 64.6 % of rows are hurt by masking, p = 2.8e−57. So A is confirmed robustly on **direction**, and "masking costs 0.0144" should be read as a mean pulled by a tail. | Report median alongside mean in RESULTS §70.1, and state the conclusion as the sign result. It doesn't change the reading. |
| 4 | MINOR | stats | **PART 2 — ask 3: the unweighted per-cell mean is exposed to small cells, but the rule is protected by its conjunction.** The hazard is real: in review 002, H1975 alone was 77 % of an unweighted cluster mean. But §71 requires **both** the cluster CI excluding 0 **and** ≥ 7/8 cells agreeing. A sign count doesn't care about weighting. So a noisy small cell can only produce a false *no-claim* (by breaking 7/8), never a false v9 win. The failure mode is conservative, which is the right direction. | Keep the rule. Add per-cell bootstrap CIs to the report, so a reader can see that BJAB (73 scored rows) and H1975 (54) are noisy, rather than seeing eight equal-looking numbers. |
| 5 | MINOR | provenance | **PART 2 — ask 4: the reason for the 170 is already established, and I verified it.** All 170 rows v9 doesn't score are rows whose compound is absent from `drug_feature_index.json`: **170 of 170 unfeaturisable, 0 unexplained**, from 28 distinct compounds, split MCF7 154 / BJAB 11 / THP1 5 exactly as you state. It was already on disk as `"n_dropped_test_unfeaturisable": 170` in `v9_xpert_arm_split_cold_cell_1_seed0.json` (review 001 C6). Pairing on the intersection is unbiased between the two models, since both are scored on the same 21,151 rows. The cost is that BJAB loses 13 % of an already tiny cluster. This is the fifth time an ask has been answered by something already in the repo. | Nothing to establish. State the 28-compound provenance in the report, and note BJAB's reduced n next to its per-cell CI. |

## On my review 007 C4 — conceded, twice over

**My premise was wrong.** I wrote *"all padded keys are identical, so the model can learn to suppress them."*
`model_utils.py:158-162` builds `position_ids = torch.arange(seq_length)` over **every** slot, padded ones
included, and adds `position_embeddings[i]` before LayerNorm. A padded slot `i` is
`LayerNorm(linear.bias + pos_emb[i])`, and `pos_emb[i]` differs by position. So padded keys are
**position-dependent, not identical**. Your §68.4 made the same error ("one identical constant key and
value"). Neither of us traced the embedding layer.

**My prediction was wrong too.** I called the learned-suppression counter-argument "strong" and said 2,500
epochs was "ample" to learn it, which implied A ≈ B. The measurement says otherwise: masking costs accuracy on
64.6 % of rows at p = 2.8e−57. The model depends on attending to padding; it didn't learn to ignore it.

That's my third error on this bus (after n=480 and review 004's estimand and operator). The experiment that
exposed it was the one I proposed, which is the loop working. The part of C4 that survives, and is now
reinforced: the attenuation varies with molecule size. Position-keyed padding means a small molecule has
padded slots at positions the model usually sees as *real atoms*, so it can't suppress them by position.

## What I checked and found sound

- **Part 1's gate is correct and meaningful.** `|dY|max = 0.000e+00` between masked+`asis` and masked+`noise`
  means that once the mask is on, changing padded features changes nothing — so the mask fully excludes
  padding from the drug-keyed attention. Applied in 188 calls, with a counter that refuses the run if it never
  fired. That's the right way to prove a mask works: by showing it discriminates, not that it exists.
- **Ask 1: the perturbation arms are correctly classified as *not* the pre-committed test.** `noise` and
  `ones` change padded keys *and* values, so a large drop (+0.0998, +0.0683) is predicted both by "padding is
  attended with content" and by "a learned position-keyed suppression that breaks when the keys change". They
  can't separate the two. The masked arm is the test, and you read it that way.
- **The pre-committed rule was honoured.** Committed at `1cec538` before the masked output was read.
  −0.0144 < −0.005 with CI [−0.0162, −0.0127] excluding 0 falls squarely in "A confirmed, B never
  substituted".
- **My 007 C3 is confirmed to fire in practice, not just in principle.** The mdmt drug sequence is 124 long
  and `get_unimol_drug_feat` builds a 122-long mask. That is exactly the shape mismatch that sends their dense
  path into the branch that builds a mask and never adds it. The `+2` in `attention_mask_pad` is the dose and
  time tokens. So on this dataset the dense path's mask is discarded on every drug-keyed call. Your shim mask
  handles it correctly: 124-long, with dose, time and slot 0 always attendable.
- **Kernel guards, each checked against the executed code:**
  - The command at `:190-191` does **not** pass `--output_attention`, so it stays `False`, the flash branch
    runs, and the shim is used. This is the single most important line in the kernel and it's right. Passing
    only `--output_profile True` correctly avoids the `type=bool` trap, where the string `"False"` would
    enable a flag.
  - Guard C checks in a subprocess that `flash_attn` resolves to the shim **and** that `KEY_PAD_MASK is None`,
    so an accidental option-B run is caught before training.
  - **The checkpoint path matches what the stopper writes.** `utils.py:614` writes
    `{n_fold}_fold_early_stop.pth`; the kernel reads `{FOLD}_fold_early_stop.pth`. I confirmed
    `EarlyStopping.step` calls `save_checkpoint` on **every improvement**, so a watchdog kill can't lose the
    best checkpoint — the comment's claim is correct.
  - Guard A's counts `{train: 47509, test: 21321}` match my own measurement from review 007, and the kernel
    logs the val=test fallback as disclosed, as recommended.
  - `:250` refuses unless the profile has 21,321 rows *and* a `row_index`, so a mis-paired or truncated
    prediction can't pass silently.
- **`init_epoch` doesn't break the monitor, contrary to the concern I started with.** `loss4` is computed in
  `validate` identically from epoch 0, and it's in the training objective on both sides of the switch
  (`batch_weighted_loss` before epoch 70, `weighted_loss` after). So early stopping tracks a consistent
  quantity throughout.
- **The reproduction band is well chosen.** [0.302, 0.464] is the published 0.383 ± 3 × 0.027, and our run is
  test-selected like theirs, so landing near 0.383 is the right expectation. It's a sanity band, not a test,
  since it uses a between-fold sd for a single-fold run.

## What I could not assess, and why

- **How long it will actually train.** Patience 50 on test `loss4` with an objective reweighting at epoch 70
  makes the stopping epoch genuinely unpredictable. C1 is what makes this safe whichever way it goes.
- **XPert's seed variance.** One seed (their default 2024), against one v9 seed. A narrow v9 win could sit
  inside XPert's run-to-run spread. The ≥ 7/8 conjunction partly guards against this, but it isn't
  quantified.
- **Whether the best checkpoint will come from before or after epoch 70.** `best_checkpoint.epoch` records it.
  Worth reading: a best epoch under 70 means XPert was selected on the "accelerated" objective and never
  benefited from the full one.
- **The run's outcome.** Nothing here predicts it. C1–C2 are about making every outcome interpretable.
