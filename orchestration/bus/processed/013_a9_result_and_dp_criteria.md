# PACKET 013 — A9 result (not answerable at inference), and the DataParallel proof criteria, before v6 is read
packet_id: 013
created: 2026-09-24
repo_commit: 9a667c6
type: **RESULT + PRE-READ REVIEW.** GPU hours: 6.80 before v6. v6 (measure-only, ~0.2 GPU-h) is running now. Its
numbers will not be read until you have answered part B.

## PART A — A9: where the atom tokens' residual harm lives at atom_alpha = 0

### What was pre-committed (verbatim in substance, committed at a26751f before any code existed)
Background: with direct atom-to-atom attention cut (`atom_alpha = 0`, the operator you reviewed in 010), the atom
effect on `unseen_compound` is still **−0.00322** per row [CI −0.00433, −0.00226], sign p 1.6e−10. Atom effect =
median over rows of `r(atoms) − r(atoms → batch mean)`. Your 010 ask 3 named the candidate carriers.

Two inference-only cuts, all at `atom_alpha = 0`:
- **X** — in every perturb block's gene → drug cross-attention, genes attend to key 0 (the global token) only.
- **G** — in every drug self-attention block, the global token's query row attends to key 0 only.

| X | G | atoms reach genes through | cell |
|---|---|---|---|
| on | on | everything | E0 |
| on | cut | atom keys only | Ea |
| cut | on | the global token only | Eb |
| cut | cut | nothing | must be exactly 0 |

Checks, in order: (1) the uncut cell reproduces the earlier alpha=0 per-row arrays byte for byte; (2) both cuts give
`dY_max == 0.0` exactly; (3) **kill switch**, per single cut, on `unseen_compound`: `cost` = median over rows of
`r_full(no cut) − r_full(cut)`, paired; *a cell is read only if |cost| < 3 × |E0| = 0.00966*, two-sided — the earlier
+0.03 bar was 3× the effect then under test, so the principle was carried over, not the number; (4) null-key gate with
`x_cell`: `|N(cut) − N0| < 0.25 × |E(cut) − E0|`. Readings were fixed for every outcome, including: **"both fail ⇒ A9
is not answerable by inference-time cuts; the question passes to a trained arm."**

### What happened
`model/v9/a9_decompose.py`, committed at e20147d before it was run, unchanged at run time. 1,500 rows per split, same
`rows_sha` as before.
- Check 1: byte-identical on both keys × 3 splits. Check 2: `dY_max = 0.0` on all three splits.
- Check 3, `unseen_compound`: X costs **+0.04209** [+0.03696, +0.04682]; G costs **+0.04609** [+0.03997, +0.05184].
  **Both fail**, 4.4× and 4.8× the bar. (Both also exceed +0.03.)
- ⇒ recorded as **not answerable by inference-time cuts.**

Reported, not read, `unseen_compound`, median per row [CI95] / paired mean:
```
E0  (nothing cut)            -0.00322 [-0.00433, -0.00226]  mean -0.00682
Ea  (G cut: atom keys only)  -0.01201 [-0.01329, -0.01046]  mean -0.01780
Eb  (X cut: global only)     +0.00023 [+0.00007, +0.00038]  mean +0.00085
remainder i = e0 - ea - eb                                   mean +0.01013 [+0.00779, +0.01253]
gate: X ratio 0.55 (FAIL); G ratio 0.15 (PASS)
```
On `unseen_both` X costs +0.00930 (just under the bar) but that split is not primary and its gate fails (0.34).
All three splits are in `model/results/v9_a9_residual_routes_sa0.json`.

What I recorded as established: (1) structurally, at atom_alpha 0, atoms reach the output only through the two keys
cut — `dY_max = 0.0` proves it; (2) the kill-switch quantity itself: the trained model's predictions lean hard on
both routes (0.042–0.046). I recorded the remainder, larger than E0 and of opposite sign, as the signature of a
counterfactual far from the trained model, and the route signs as **not** evidence about the residual.

## PART B — v6, the DataParallel measurement you approved in 012. Criteria committed at e20147d, before launch.

### The patch (`model/v9/xpert_dp_patch.py`)
Not `nn.DataParallel(model)`. `XPertNet.forward` is replaced at runtime: at the top level, in training with grad
enabled and batch ≥ number of GPUs, it calls `torch.nn.parallel.data_parallel(self, (data,), device_ids=[0, 1],
output_device=0)`; on a replica (`_is_replica`, set by `replicate`) it sets `self.device` to the current CUDA device
and copies every plain tensor attribute to it, then runs their forward. The object their code holds stays an
`XPertNet`, so `state_dict()` keys are unprefixed. Validation and prediction stay single-GPU.

A bug the local smoke test caught before launch: `replicate()` turns a replica's parameters into plain non-leaf
attributes, so `replica.parameters()` is **empty**; my first version read the device from it and raised
StopIteration. It now uses `torch.cuda.current_device()`, which `parallel_apply` sets per worker thread. Locally (one
GPU) the replica path gives output identical to the top-level path, and the only plain tensor attribute is
`drug_HG_embed`.

### Your C4 was incomplete, and the probe goes further
You named the four config dropout rates. `model_XPert.py:144, 158, 164, 170` hard-code `nn.Dropout(p=0.1)` in the
output heads, which the config does not reach, and `model_utils.py:183` gives each attention layer its own
`dropout_p`. The probe sets `p = 0` on every `nn.Dropout` and `dropout_p = 0` on every module that has it, stays in
`.train()`, and asserts none is left. Locally it zeroed 47 sites.

### The proof (`model/v9/xpert_dp_probe.py`), four processes: single_ckpt, single_ckpt_repeat, dp, dp_ckpt
Same initial weights and one recipe batch, saved by the first process and loaded strictly by the others. Every
gradient is taken by calling `train_xpert.train()` itself with `torch.optim.SGD(lr=0.0)` and a fresh GradScaler, so the
differentiated loss is their `batch_weighted_loss` (epoch 0) or `weighted_loss` (epoch 70); `scaler.step` unscales the
gradients in place before they are read.

Pass criteria, `rel_L2(a, b) = ||g_a − g_b|| / ||g_b||` over all parameter gradients:
1. **Same function** — fp32 (their `autocast` name replaced by `contextlib.nullcontext`), first 32 rows, epochs 0 and
   70: `rel_L2(dp, single) < 1e-5` and `rel_L2(dp_ckpt, single) < 1e-5`; loss relative difference `< 1e-6`.
2. **Within the recipe's precision noise** — fp16 autocast, same 32 rows, epochs 0 and 70: `rel_L2(dp, single)` and
   `rel_L2(dp_ckpt, single)` each ≤ `rel_L2(single fp16, single fp32)`.
3. Structure: unprefixed keys equal to the unpatched model's; plain tensor attributes exactly `['drug_HG_embed']`; all
   gradients finite. 4. Memory: < 13.0 GiB per GPU.
At 128 rows, fp16 comparisons are reported against the single-vs-repeat floor, not gated. Timing: their train() with
Adam and their GradScaler, one warm-up + five batches from their loader, per variant; epoch = 372 × s_train +
167 × s_val. The faster passing variant prices O2 at ~210 epochs. Any failure ⇒ O2 is not priced.

## ASKS
1. **A9:** is "not answerable by inference-time cuts" the correct reading? Is anything in the reported-not-read table
   being used, anywhere above, beyond what the kill switch allows?
2. **A9:** is the two-sided 3×|E0| bar defensible, given it was stricter than the inherited +0.03 — or should it have
   been one-sided like the original? (It does not change this outcome: both cuts exceed either bar.)
3. **v6:** are criteria 1–2 the right tests of "the same computation"? In particular: is fp32 at 32 rows (16 per GPU)
   an adequate stand-in for the 64-per-GPU recipe, and is 1e-5 the right fp32 bar on a T4 (no TF32)?
4. **v6:** the patch leaves validation single-GPU. Does anything in their train/validate/EarlyStopping path touch the
   model in a way the forward-internal data_parallel would not cover?
5. Anything else before I read v6.
