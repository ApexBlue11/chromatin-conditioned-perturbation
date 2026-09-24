# PACKET 014 — v6 failed the committed proof; a diagnostic says the bar was wrong, not DP; a redesign for review
packet_id: 014
created: 2026-09-24
repo_commit: 5b45f9e
type: **FAILURE REPORT + NEW PRE-REGISTRATION, designed after the failure.** GPU hours ~6.95. This packet commits none.

## 1. v6, and what I did with your 013 amendments
v6 ran every guard and all four probes, then stopped on its own `_cmp` assertion: *different parameters received
gradients*. The per-probe JSON (losses, scaler scale) was lost with the crash; the four gradient dumps survived.

**Key sets, inspected structurally first:** single-GPU 174 gradients, DP 184, on every tag. The ten extra are the
parameters their forward/loss never use — `cell_emb.linear` (w, b), `ctl_fc.0` and `ctl_fc.3` (w, b), and the
LayerNorm gamma/beta of `attnEncoder_trt.crossEncoders.0` and `.1`. On one GPU they get `grad = None`; under DP,
exactly zero (`Broadcast.backward` materialises zeros). Consequence neither of us caught: `torch.optim.Adam` applies
`weight_decay` to any parameter whose grad is not None (`adam.py:151`), so under DP these would take ~lr-sized steps
every iteration while the recipe leaves them at init. **Committed fix, before any value comparison:** freeze exactly
that set (`requires_grad=False`) in every variant — a no-op on one GPU; `Broadcast.forward` marks outputs of inputs
that need no grad non-differentiable (`_functions.py:31-35`), so DP then gives them no gradient. Asserted at runtime.

Your C1 void rule, the floor rule, C5 and ask 5's items were committed as amendments **before** any comparison
(RESULTS 80.5, commit 75e356a). The offline comparison script was committed (0425dd8) before it ran.

## 2. The verdict, as committed
```
structure (extras exactly zero, exactly the ten)          PASS
fp32 floor rel_L2(single_repeat, single)                  2.8e-07   (< 3e-6, so criterion 1 discriminates)
criterion 1 fp32 rel_L2(dp, single)      epoch 0 / 70     3.53e-04 / 3.77e-04   bar < 1e-5   FAIL
criterion 1 fp32 rel_L2(dp_ckpt, single) epoch 0 / 70     3.53e-04 / 3.77e-04                FAIL
fp16 tags                                                  ALL VOID: the reference itself is non-finite on all four,
                                                           epoch 0 included (your estimate had epoch 0 5x under)
memory per GPU: dp 7.39 / 7.36 GiB; dp_ckpt 2.00 / 1.96 GiB
```
**Recorded as a failure (RESULTS 80.6): O2 not priced on this measurement.** Timing, recorded and explicitly not a
price: `dp` 0.925 s/step (epoch 411 s, ~24 GPU-h at the anchor), `dp_ckpt` 1.169 s, single-GPU 2.229 s.

Per parameter (fp32, epoch 0): 84 of 174 differ by 2.7e-4 to 7.7e-4, heaviest in `crossEncoders.0` attention and the
drug path; the repeat floor on the same parameters is ~2-6e-7.

## 3. Post-hoc diagnostic (local, one GPU, no DataParallel) — not pre-registered
Their `train()`, a batch vs the same rows as two half-batch forwards concatenated before their loss (what DP does,
minus the second device). `rel_L2` of gradients vs the full batch:
```
fp32   32 rows  mem-efficient SDPA   init: repeat 1.3e-07  SPLIT 2.4e-04 | released weights: repeat 1.3e-08  SPLIT 2.2e-04
fp32   16 rows  mem-efficient SDPA   init: repeat 7.6e-08  SPLIT 3.5e-04 | released weights: repeat 6.5e-09  SPLIT 9.6e-07
fp32   16 rows  math SDPA            init: repeat 0        SPLIT 5.8e-04 | released weights: repeat 0        SPLIT 1.3e-04
float64 16 rows math SDPA            released weights: repeat 0   SPLIT 1.1e-16 (epoch 0), 1.1e-17 (epoch 70)
```
(float64 required bypassing `get_unimol_drug_feat`'s hard `.float()` on atom features, `model_XPert.py:14`, for the
diagnostic alone.) My reading: in exact arithmetic the split computation is the recipe's; in fp32 an exactly
equivalent reorganisation on one GPU moves gradients 1e-6 to 6e-4, which is the range DP's 3.5e-4 falls in. Not
init-specific, not SDPA-backend-specific; source not identified. **The 1e-5 bar assumed batch-shape rounding of
1e-7-1e-6 — your 013 ask-3 answer and my design both assumed it; this model's fp32 numerics break it.**

## 4. The redesign — RESULTS 81, a draft, labelled as designed after the failure
**81.1a Semantics in float64:** 32 rows (16/GPU), epochs 0 and 70, fresh and released weights, the ten frozen:
`rel_L2(dp, single) < 1e-10`, `rel_L2(dp_ckpt, single) < 1e-10`, loss relative difference `< 1e-12`.
**81.1b fp32 against the right reference:** add a `split_same_gpu` variant; `rel_L2(dp, split_same_gpu) < 1e-5`;
`rel_L2(split_same_gpu, single)` reported as the recipe's own batch-shape sensitivity.
**81.1c fp16 reported only**, one fixed `GradScaler(init_scale=2**6)` for every variant; in-replica autocast asserted.
**81.2 Full-state checkpoint:** runtime hooks register the `XPertNet`, `Adam`, `LambdaLR`, `GradScaler`,
`EarlyStopping` instances. Save after `lr_scheduler.step()` (`train_xpert.py:545`, after `stopper.step` at `:538`):
model/Adam/scaler/scheduler state dicts, stopper `best_score`/`counter`/`early_stop`, the best-checkpoint file's bytes
(its folder is time-stamped per session, so it is restored under the new path), torch CPU + both CUDA + numpy +
python RNG states, the epoch. Carried between sessions as a dataset version (~150 MB).
**Restore:** their `--resume_from` sets `start_epoch` only (`:489`); the full restore runs in the `EarlyStopping.__init__`
hook (`:524`, the last construction before the loop): a strict model load with the key set asserted equal, then
optimizer, scaler, scheduler, stopper, RNG. Their filtered load (`:484-486`) runs first and is overwritten.
**81.3 Resume equivalence:** 2 epochs straight, twice (floor), vs 1 + save + fresh process + restore + 1; losses per
epoch within the straight-vs-straight spread, final parameters no further from a straight run than the two straight
runs are from each other. ~0.7 GPU-h.

## ASKS
1. Is 81.1a-b an admissible replacement for 80.3's criterion 1, given it was designed after seeing the failure? What
   would make it not admissible? Is float64 on a T4 the right place to test semantics?
2. Are the bars right: `1e-10` in float64 and `1e-5` for DP against a same-GPU split in fp32?
3. `decisions_locked` says no session boundary goes through their `--resume_from` (your 012 C1). 81.2 uses it only to
   set `start_epoch`, then overwrites everything with a strict, key-asserted load. Does that satisfy C1, or should
   `start_epoch` be reached another way?
4. Is 81.3 the right resume test, and is the straight-vs-straight spread a usable floor given dropout RNG is
   restored but atomics are not deterministic?
5. Anything else before a v7 measure-only run (~0.2 GPU-h for 81.1 + ~0.7 for 81.3).
