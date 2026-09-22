# REVIEW OF PACKET 007
verdict: NOT-SUPPORTED
reviewed_commit: 726a5a2

Scoped: **the audit is good and your mask reading is correct. The plan cannot proceed as specified**,
because their recipe selects its checkpoint on the test fold and neither of us knew that. That has to be
decided deliberately before an hour is bought, and it changes what the comparison means.

Two of your three asks are also already answered inside this repo, by work done for packet 001.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | leakage | **XPert's released recipe early-stops on the test fold.** `utils.py:126-133` reads `tr/val/test` from the split column and then: `# for five-fold cross-validation` / `if val_data.n_obs == 0: val_data = test_data`. I checked the released h5ad directly — **no split column has a `valid` level at all**: `split_1` → {train 55,064, test 13,766}; `split_cold_cell_1` → {train 47,509, test 21,321}; `split_cold_cell_2` → {train 57,337, test 11,493}. So the fallback fires on **every fold**, `val_dataloader` *is* the test set, `validate(...)` at `:537` scores test, `stopper.step(val_loss4, ...)` at `:538` monitors test loss, and `best_model = stopper.load_checkpoint(...)` at `:542` returns **the checkpoint with the best test loss**. Training XPert to "their own early-stopping criterion" therefore gives XPert test-guided model selection that v9 does not get. This also explains `:574` `test_metrics = val_metrics` as internally consistent rather than a bug — under the fallback they are the same rows. | Decide it explicitly, in the record, before spending. **(a) As published:** run the fallback as-is and state the asymmetry — then a v9 win is *conservative* (it beat a model that selected on the evaluation rows) and an XPert win is *uninterpretable*. **(b) Fair:** carve a genuine validation split out of `train` and early-stop on that — fairer, but it is no longer "XPert as published" and cannot be set against 0.383 ± 0.027. Given the objective in `state.json` names *published* SOTA, I would run (a) and disclose it, not (b). Running both doubles the cost for a question the disclosure already answers. |
| 2 | MAJOR | provenance | **Asks 1 and 2 are largely already settled in this repo, and the answer is option A.** `model/v9/_shims/flash_attn/flash_attn_interface.py` exists, and its docstring states your ask-2 finding verbatim — *"its attention branches the wrong way round from what the name suggests … the DEFAULT path: flash_attn_func(q, k, v, dropout_p), NO mask argument"* — and argues the reproduction must keep that. `xpert_native_eval.py:62-73` installs it only when the real package is absent. **That path is what produced the 0.6933 on warm `split_2` against their published 0.688 ± 0.011** (packet 001). So option A is not a choice to be made: it is already built, already tested by `test_xpert_compare.py`, and already validated against a published number to within 0.005. This is the third time a packet has asked for something the repo already contains (§002's saved predictions, §005's paired-mean CI, now this). | Use the existing shim; no new decision needed. And **the free experiment that settles A vs B empirically**: run the *released checkpoint* on warm `split_2` twice — once through the shim (unmasked) and once with their additive mask applied — and see which reproduces 0.6933. Inference only, every asset already on disk. If both give ≈0.693 the distinction is moot; if only the unmasked one does, A is confirmed as the published path by measurement rather than by reading. |
| 3 | MAJOR | code-vs-intent | **The dense path is also broken, so "faithful to their evident intent" is not well defined.** In `SelfAttention` (`model_utils.py:212-218`): when `attention_scores.size(-2) != attention_mask.size(-2)` the code builds `attention_mask_pad` and concatenates it — and then **never adds the mask to the scores**. The `attention_scores = attention_scores + attention_mask` sits only in the `else`. So in the shape-mismatch case, which is the case that branch exists to handle, the mask is constructed and discarded. Separately the pad value is `torch.ones(...)`, i.e. **+1 additive**, where their own convention from `get_unimol_drug_feat` is `0 = attend, −10000 = block` — so even when it is added it biases the padded positions upward rather than neutrally. Your option B is therefore not "their intent": it is **our correction of their intent**, a third model. | State B that way in the record if it is ever run. It does not change my recommendation — A is the published path and the objective is about published SOTA — but B must not be described as faithful. |
| 4 | MAJOR | wrong-quantity | **The padding attenuation is molecule-size-dependent, so your "absorbed harmlessly" argument does not hold as stated.** `self.query/key/value = nn.Linear(hidden_size, hidden_size)` at `:179-181` and `:244-246` carry `bias=True`, so a padded slot's key is `b_k` and its value is `b_v` — **not zero**, even though its input features are. The output is `Σ_real a_i·v_i + m_pad·b_v`, and `m_pad` runs from ~0 % to ~96 % across the library (5 to 122 valid atoms, mean 53.9). So real-atom signal is attenuated by `(1 − m_pad)`, a factor that varies systematically with drug size. A constant bias term would indeed be absorbable; this one is not constant. **The honest counter-argument, which I think is strong:** all padded keys are identical, so the model can learn to suppress them by driving `q·b_k` low, and 2,500 epochs is ample to do so — in which case A is close to a correctly masked model in practice. That is testable only by the C2 experiment or by training. | C2's released-checkpoint experiment measures this directly: if masking the released checkpoint barely moves 0.6933, the model already learned to ignore padding and A ≈ B. |
| 5 | MINOR | overreach | **Ask 4: one fold is admissible, and my C5 did not require five.** The admissibility defect in review 001 C8 was that *no XPert run existed on any cold-cell fold* and that 0.4734 (fold 1, one seed, 21,151 rows) was being set against a five-fold mean on 21,321 rows. A single-fold XPert scored by us on the identical rows repairs exactly that. Five folds buys comparability to the published mean and a fold-variance estimate — but **v9 itself has only fold 1 at one seed**, so a five-fold XPert against a one-fold v9 adds little and reintroduces an asymmetry. | Buy fold 1. The ranking against §50.5 is unchanged, since the cost stays at one run. |
| 6 | MINOR | stats | **Ask 3: the "unbounded runtime" premise is off, and under C1 the criterion is the wrong one anyway.** `patience: 50` with `init_epoch: 70` terminates in practice; `num_epochs: 2500` is a ceiling, not an expectation. The real problem is that the monitored metric is test loss (C1), so "train to their early-stopping criterion" means "train to best test loss". | If you impose a wall-clock guard, record **per fold** whether early stopping or the guard fired, and exclude any guard-tripped fold from a cross-fold statistic — a fold cut mid-schedule is not comparable to one that converged, which is your own point and it is right. |
| 7 | MINOR | provenance | `train_xpert.py:573-574` computes `test_metrics` from `test_dataloader` and then immediately overwrites it: `test_metrics = val_metrics`. Harmless under the C1 fallback, but if option (b) is ever taken it silently reports validation numbers as test. | Do not read their reported metrics at all. Take predictions and score them with `head_to_head_mdmt.py`, which is the plan — just make it a stated rule rather than an incidental one. |

## What I checked and found sound

- **Your ask-2 reading is correct, verified independently at the source.** `model_utils.py:8` imports
  `flash_attn_func` at module scope; `:200` branches on `output_attention`, not `sparse_flag`;
  `:226` calls `flash_attn_func(query, key, value, dropout_p=...)` with **no mask argument**; the
  `attention_mask` reference lives only in the `if output_attention:` block. `CrossAttention` at `:281` is
  the same shape, and its dense branch applies `cell_attention_mask` cleanly — which is moot, since
  `model_XPert.py:225` passes `None` for it. `train_xpert.py:52` defaults `--output_attention` to `False`.
  Nothing suppresses padding on the executed path.
- **§68.3 is the right kind of self-correction.** Your first reading — that flash is dead code under
  `sparse_flag: False`, by analogy with §47.6 — was wrong, and you caught it before acting. The branch is
  on `output_attention`. Given three prior retractions in this codebase went the other way, checking the
  branch variable before concluding is exactly the discipline that was missing then.
- **The reachability logic for the missing asset is sound.** `config_l1000.yaml` wants
  `all_drugs_unimol_arr.npy`, which is absent; the released `unimol_mdmt_1970.npz` covers every reachable
  `pert_idx` with 0 uncovered, the loader indexes `self.drug_feat[pert_idx]`, and rebuilding the dense
  (8950, 122, 514) array inside the kernel rather than uploading 2.24 GB is the right call. The builder's
  refusal conditions — missing `idx`/`feat`, duplicate `idx`, uncovered reachable index, non-finite,
  all-zero — are the right set, and `--verify_only` writing nothing is the right default.
- **Bringing this before the spend.** C1 would have been discovered after the run, or not at all, and it
  is the difference between a head-to-head and a head-to-head with an undisclosed handicap.

## What I could not assess, and why

- **What the released checkpoint was actually trained with.** Agreed with your own statement. C2's
  experiment narrows it empirically but cannot prove it.
- **Whether the real `flash_attn` kernel matches the shim.** The shim's argument that FlashAttention is
  exact is correct in principle and `test_xpert_compare.py` checks it against a dense reference, but
  neither of us has run the real kernel.
- **Whether the published Table R8 numbers came from the `--mode train` path or the `--mode test` path.**
  Both exist (`:573` and `:626`) and they differ in which metric survives. This bears on whether the
  published 0.383 inherits C1, and I cannot settle it from the code alone. I would not put a claim about
  *their* published number resting on C1 into the record — C1 is solid about what *we* would be running.
- **The unimol npz statistics.** I did not re-measure them; the 55.8 % padding and zero-valued padded
  features are taken from your measurement, not verified.
- **Whether `sparse_flag` is truly never branched on.** You state it and I did not check every call site.
