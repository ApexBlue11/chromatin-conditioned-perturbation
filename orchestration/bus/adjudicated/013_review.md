# REVIEW OF PACKET 013
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 9a667c6

**Part A: the reading is correct.** Both kill switches fail, far from the bar. A9 is not answerable by
inference-time cuts. I reproduce every number from the per-row arrays. Three pieces of wording go further than the
kill switch allows (C2–C4).

**Part B: settle C1 before you read v6.** As coded, GUARD G will almost certainly report **both** DataParallel
variants as failing, for a reason that has nothing to do with DataParallel. The fp16 epoch-70 gradients overflow a
fresh GradScaler in the single-GPU reference just as they do under DP. Apart from that, the criteria test the right
thing, and the patch is correct.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **The fp16 epoch-70 tags will almost certainly overflow, and GUARD G counts that as a DP failure.** Each tag builds `torch.cuda.amp.GradScaler()` at its default scale of 65,536. At epoch 70 the differentiated loss is `weighted_loss`, with weights `[0.2, 0.003, 0.2, 1]` (`config_l1000.yaml:64`). Its first term is `loss1 = Σ_rows mean_genes (pred − true)²` (`utils.py:333`), a **sum** over rows. The targets aren't centred: I sampled 256 rows of `l1000_mdmt_68830_subset.h5ad` — mean 8.32, range 0–15. At kaiming init the prediction is O(1), so ∂loss/∂pred ≈ 0.2 × 2 × (−8)/978 ≈ −0.0034 per element, ≈ −220 after scaling. `trt_fc`'s last bias sums that over 32 × 978 = 31,296 elements, all of one sign: **≈ 7 × 10⁶**, against fp16's maximum of 65,504. Under autocast that reduction writes fp16, so it becomes inf. `scaler.step` then skips the step and leaves the inf in place, and the probe reads it. In the kernel, `structure` requires `grads_finite` for **every** tag in `cmpr`, including `fp16_e70` and `fp16_e70_128`. `_cmp` on inf returns nan, so `fp16_within_precision_noise` is False as well. Both variants fail, `chosen` is None, and "O2 is not priced" — yet `single_ckpt` overflows in exactly the same way. The epoch-0 tags differentiate `batch_weighted_loss`, where the sqrt and /N terms shrink that bias gradient to ≈ 1.3 × 10⁴. That's finite, but only 5× under the limit. | Commit this **before reading**, as a pre-read fix to a design defect, applied to every variant alike: (i) a tag where `single_ckpt` records `scale` = 32768 (the reference itself backed off) is **void** for criteria 2 and 3 in every variant, not failed; (ii) the finiteness gate becomes *each variant's finiteness equals the reference's, per tag* — DP non-finite where single-GPU is finite stays a failure; (iii) with the fp16 e70 tags void, DP is still fully priced. fp32 at e0 and e70 proves the semantics in both loss regimes. fp16 e0 tests the autocast interaction, which doesn't depend on which loss is differentiated. If you want e70 in fp16 anyway, re-run only those tags with one fixed `GradScaler(init_scale=2**6)` shared by all four variants: ≈ 7 × 10³ at 32 rows, ≈ 2.7 × 10⁴ at 128. |
| 2 | MINOR | overreach | **"The trained model leans hard on both routes" needs to say what it leans on: the pathways, not the atoms' content.** I also computed the kill-switch cost in the ablated arm, with atoms replaced by the batch mean, on `unseen_compound`. X costs **+0.0455** [0.0409, 0.0501] against +0.0421 with the real atoms. G costs **+0.0409** against +0.0461. Both cuts together cost +0.0602 against +0.0561. Cutting a route costs about the same whether the atom tokens carry the molecule or the batch mean. What the model depends on is the pathway itself — genes having those keys, the global token doing its mixing — not molecule-specific atom information. At α = 0 that information is net harmful (E0 < 0). In a record about atom tokens, the unqualified sentence will read as "atom information matters to predictions", and these data say the opposite. | Record: *"cutting either route costs 0.042–0.046 whether the atom tokens carry their own molecule or the batch mean (0.041–0.046): the trained model depends on the attention pathways, not on molecule-specific atom content through them."* |
| 3 | MINOR | overreach | **The remainder is being read.** The 9a667c6 commit message says the +0.010 remainder *"is what a counterfactual 4-5 points of Pearson from the trained model looks like"*. That interprets a quantity §79.6 says is not read, and uses it to back up the kill switch. The kill switch decides the outcome alone and needs no backing. And nothing establishes what such a counterfactual "looks like": there is no reference distribution. | Record the remainder as reported and not read, with no interpretation attached. |
| 4 | MINOR | overreach | **"The question passes to a trained arm" defers a question that is actually closed.** §79.8 says it itself: a trained arm tells you what a model trained *without* a route would do, not where the harm flows *in this trained model*. A9, as posed, is closed at inference. A trained arm would answer a different question. | Record: *"A9 is not answerable at inference; a trained arm would answer a different question (§79.8)."* Price the trained arm as that question, not as A9 deferred. |
| 5 | MINOR | stats | **Criterion 2 can't detect the one fp16-specific failure it exists to catch.** Suppose autocast didn't reach the replica threads. DP would then compute in fp32, and `rel_L2(dp, single fp16)` would land at ≈ `rel_L2(single fp32, single fp16)` — the bar itself — so it passes or fails by noise. In practice `parallel_apply` does propagate autocast: in the local torch 2.13, `parallel_apply.py:77-79` captures `torch.is_autocast_enabled()` and `:109` re-enters `torch.amp.autocast(device_type, enabled=...)`. So the risk is low, but the criterion doesn't show it. And `shutil.rmtree(DPS)` deletes the saved gradients, so no gradient-level check can be done after the fact. | Free, from what is recorded: `loss_rel(dp fp16, single fp16)` should sit well below `loss_rel(single fp16, single fp32)`, and every loss for both is in `DPR`. If the two are comparable, the question stays open, and a one-line assertion inside the replica (`torch.is_autocast_enabled()`) settles it on a re-run. Log `torch.__version__` in the record. Keep `dp_shared/*.pt` as kernel output next time. |

## Answers to the asks

**Ask 1 — yes, "not answerable by inference-time cuts" is the correct reading, and it isn't borderline.** The CI
lower bounds are +0.03696 (X) and +0.03997 (G): 3.8× and 4.1× the bar of 0.00966. Nothing above uses Ea, Eb or the
remainder as evidence **for a route**, which is right. C2–C4 are where the wording goes further than the kill switch
allows. One detail supports not reading `unseen_both`. X "just under the bar" there has a CI of [0.0064, 0.0123],
which straddles the bar. In the ablated arm the same cut costs +0.0187. Even that apparent pass depends on which arm
you measure.

**Ask 2 — two-sided is right.** The principle behind §67.2 is *distance from the trained model*, and distance is
symmetric. A cut that *raised* the score by 3 × |E0| would be just as far from the model whose routes you're
dissecting. The one-sided +0.03 was an artefact of framing the question as a cost. Scaling the bar to the effect under
decomposition, rather than inheriting a fixed number, is the principled part: a cut that moves the model by several
times the effect can't apportion that effect. One limitation, for future kill switches: accuracy change is a
**necessary, not sufficient,** measure of distance. A cut can reorder predictions a great deal with little net change
in per-row r. A direct output-distance term would close that, for example the median per-row correlation between cut
and uncut predictions. It doesn't bear here, since both cuts fail on accuracy alone.

**Ask 3 — criterion 1 is the right test, 32 rows is adequate, and 1e-5 is a sound fp32 bar on a T4, with one addition.**
- **Why 16 rows per GPU is enough.** Every way DP could change the *function* shows up at any per-GPU batch of 2 or more. That means an op coupling samples, a tensor on the wrong device or stale, the loss computed on a shard, a gather out of order, or a non-batch tensor being scattered. I checked each in their forward. All ten inputs are batch-first tensors (`model_XPert.py:186-198`). Drug features are fixed-length: padded in the dataset, and `get_unimol_drug_feat` only slices them, with no per-batch trimming. No op reduces over the batch dimension. The loss is computed after the gather, in `train()` (`train_xpert.py:87-104`). `attention_dict` is a pair of empty dicts, which gather cleanly. Batch size changes only numerics and memory, and the fp16-128 comparison and the timing run cover those.
- **Why 1e-5.** Turing has no TF32, so fp32 GEMMs are true fp32. The expected DP difference is two partial gradient sums added together instead of one reduction, typically 1e-7 to 1e-6 relative. A semantic break would be of order 1e-3. So 1e-5 sits in a wide gap. The one risk is the floor. fp32 SDPA takes the memory-efficient path (flash doesn't accept fp32), and its backward is nondeterministic. The `single_ckpt_repeat` floor is recorded, not gated. **Add, before reading:** if `rel_L2(single_repeat, single)` in fp32 is ≥ 3e-6, a third of the bar, then criterion 1 doesn't discriminate as written. The fp32 tags are then re-run for all four variants with `enable_mem_efficient_sdp(False)`, the deterministic math backend.
- **Criterion 2:** see C1 (it currently fails regardless of DP) and C5 (it can't discriminate as designed).

**Ask 4 — no. Nothing in their path touches the model in a way the forward-internal data_parallel misses.**
`train()` calls `model.train()`, `model.zero_grad()`, `model(data)`, and `scaler.step(opt)` over `model.parameters()`.
Those are the original parameters, and they receive the replica-summed gradients. `validate()` calls `model.eval()`
(`:162`), so the patch falls through on `self.training` whatever the grad mode. EarlyStopping and `save_checkpoint`
call `state_dict()` on the unwrapped `XPertNet`. Submodules create tensors on their input's device
(`model_utils.py:41, 86, 93, 156, 214`), so the top-level `self.device` and `drug_HG_embed` are the only device pins.
The ragged last batch (47,509 = 371 × 128 + 21) splits 11/10 and stays exact, because the loss sees the gathered 21.
One thing the probe doesn't cover: **`dp_ckpt`'s dropout-mask replay across replica threads.** Dropout is off in the
probe (rightly, for criterion 1). This doesn't matter if `dp` is chosen, and it likely will be, since it's faster if
it passes memory. If `dp_ckpt` is chosen, it needs 011's proof repeated under DP, with dropout on: `dp` against
`dp_ckpt` with identical per-device seeds, gradients within the noise floor.

**Ask 5 — three smaller things.**
- **Timing sits inside the loader's prefetch buffer.** `num_workers=10` × `prefetch_factor` 2 means up to 20 batches queued. One warm-up step gives them time to fill, so the five timed batches measure GPU and Python time, not whether the loader keeps up in steady state. The loader is unlikely to be the bottleneck, since `__getitem__` indexes prebuilt data. But the projection prices ~25 GPU-h, and the first real epoch's wall time will show it. Commit a tolerance now: if epoch 1 exceeds the projection by more than 25 %, re-price before session 2.
- **Harden `_localise`.** On a replica, the parameter copies are plain tensors in `__dict__` (that's why `parameters()` was empty). The generic localiser skips them only because they already sit on `dev`. If the current device ever disagreed, it would `.to()` every weight and cache each copy under the `id()` of a short-lived tensor. A later step could then be served stale weights through id reuse. `parallel_apply` pins the device, so this is latent. The fix is two lines: move only names in the expected set (`{'drug_HG_embed'}`), assert every other tensor attribute is already on `dev`, and store the source in the cache value, checking it with `is`.
- **The gradients are deleted** at the end of GUARD G. Keep them, so that any comparison not pre-computed can still be done without re-spending GPU time (C5).

## What I checked and found sound

- **Pre-commitment order.** The rules were committed at a26751f (14:29:49), the operator at 79a6bcf (14:41:18), and
  the analysis script at e20147d (14:46:31). The per-row arrays were written 14:47–15:00, and the JSON at 06:07 today.
  `a9_decompose.py` is unchanged between e20147d and 9a667c6, and its bar and gate constants match §79.4–79.5.
- **Every Part A number reproduces from the npz files,** with my own bootstrap. E0 is −0.00322 [−0.00432, −0.00223],
  Ea −0.01201, and Eb +0.00023. The costs are +0.04209 and +0.04609. `dY_max` is 0.0 with both cuts on all three
  splits, and the `rows_sha` values match. The decomposition closes in paired means: −0.00682 = −0.01780 + 0.00085 +
  0.01013. Gate ratios are 0.546 (X) and 0.155 (G), as reported.
- **The cuts do what §79.1 says, and the route inventory is complete by construction.** X clones the key mask and
  sets `[:, 1:]`, leaving drug self-attention's real padding untouched. G replaces row 0 with `B_mat`'s row 0, which
  is the one-hot on key 0. D reaches the genes only through cross-attention: the heads take `h, E, r, x_ctl`, and
  dose and time enter through FiLM, not as drug tokens. So neither cut removes dose or time information, and
  `dY_max = 0.0` verifies the implementation rather than discovering a fact.
- **Part B's design.** Patching inside `forward` makes 012 C1 impossible structurally, not just guarded against. The
  dropout sweep is complete: I grepped for every `nn.Dropout` and both `dropout_p` sites (`model_utils.py:183, 248`),
  and there are no `F.dropout` calls. The exactness argument (the loss sees the gathered batch, LayerNorm only) holds
  against their code. Initial weights are loaded strictly from one file across four isolated processes. The criteria
  were committed at e20147d, before the v6 kernel. And the local smoke test caught the StopIteration bug before any
  GPU spend.

## What I could not assess, and why

- **v6's numbers.** I haven't looked at them. C1's prediction can be checked against the record: `scale` = 32768
  on `single_ckpt`'s `fp16_e70` tags means the reference overflowed.
- **C1's arithmetic exactly.** It assumes an O(1) output at init. The margin is about 100× over the fp16 limit, so the
  conclusion is robust to that assumption, but it is an estimate, not a measurement.
- **Autocast propagation in the Kaggle torch build.** I checked the local 2.13 source only.
- **The size of the fp32 floor on a T4.** It is recorded by v6, and not something I can bound from here.
