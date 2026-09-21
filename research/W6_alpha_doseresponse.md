# TASK W6 — an alpha dose-response for the drug-attention ablation

Precisely specified. The design was set by an adversarial review; **do not redesign it.**
Repo root `C:\Projects\LINCS`. Work only in `model/v9/`.

## RUN EVERY COMMAND IN THE FOREGROUND AND BLOCK ON IT
A previous worker on this codebase launched its tests in the background, went idle, and its session
terminated them on exit — exit code 0, nothing accomplished, no report. **Do not background anything.**
Write your report last, only after you have real output in hand. `test_v9.py` takes a few minutes; run it
in the foreground and wait.

## Why this exists, in one paragraph
`model/v9` has a `_DrugBlock` (drug self-attention over `[global_token, atom_1..atom_n]`) with a
`diagonal=True` mode that masks its attention to the identity, so each token attends only to itself. That
binary ablation was used to measure whether the per-atom drug tokens are worth more when cross-atom flow
is live. A reviewer objected that masking to the identity costs about **0.10 median Pearson** — five to
seven times the effect being measured — so the ablated arm is a model operating far outside anything it saw
in training, and a binary on/off contrast cannot distinguish a real mechanism from an out-of-distribution
artifact. The fix is a **dose-response**: interpolate continuously between full attention and the identity
and look at the shape of the curve. Smooth and monotone suggests mechanism; flat and then collapsing near
the end suggests the binary result was an artifact of leaving the training manifold.

## What to implement

### 1. `_DrugAttention` gains a continuous `alpha`
Currently `_DrugAttention.forward(x, key_mask=None, diagonal=False)` (in `model/v9/modules_v9.py`)
delegates to `super().forward()` when `diagonal=False`, and when `diagonal=True` builds an additive mask
whose only unmasked entry per row is the diagonal.

Add support for a continuous blend of the **post-softmax attention matrix**:

```
A_blend = alpha * A + (1 - alpha) * I
```

where `A` is the ordinary softmax attention matrix and `I` is the identity. Because both `A` and `I` have
rows summing to 1, so does any convex combination, so no renormalisation is needed or wanted.

Requirements, all of which are testable and all of which will be tested:

- **`alpha=1.0` must be bit-identical to the current default path.** The cheapest way to guarantee this is
  to keep delegating to `super().forward()` when `alpha == 1.0` and no diagonal is requested. Do that.
- **`alpha=0.0` must be bit-identical to the existing `diagonal=True` path.** This is not an approximation
  to check loosely: pure identity attention gives `out = I @ v = v` for valid positions, and the existing
  diagonal path softmaxes over a single unmasked entry, which is also exactly 1. Assert `torch.equal` or a
  difference of exactly `0.00e+00`, and if it is not exactly zero, say so in the report rather than
  loosening the test.
- **Padded keys must not influence real tokens at any alpha.** `key_mask` is `True` at padded positions.
  Zero those columns of `A` before blending (they are already `-inf` pre-softmax), and make sure the
  identity term does not reintroduce a padded token into a real token's output.
- **Parameter count must be identical at every alpha** — this is an information ablation, not a capacity
  ablation. That is the whole reason the diagonal operator was built instead of mean-ablating the block's
  output.
- Keep `diagonal=True` working exactly as it does now; existing callers pass it and must not change
  behaviour. Treat `diagonal=True` as equivalent to `alpha=0.0`.

A full `[B, heads, L, L]` matrix is acceptable for `0 < alpha < 1` — the drug sequence is about 34 tokens,
so this is small. Do **not** make the `alpha == 1.0` path allocate one.

### 2. Threading
`PerturbBlock.forward` and `LincsV9.forward` (in `model/v9/model_v9.py`, currently lines 57 and 116-155)
already thread `diagonal` / `drug_diagonal` to every perturb block. Thread `drug_alpha` the same way, with
a default that preserves today's behaviour exactly. **Keep every existing return signature and keyword
working** — several scripts call these, including `interaction_2x2.py` and `probe_v9.py`.

### 3. A script: `model/v9/alpha_sweep.py`
Model it on `model/v9/interaction_2x2.py`, which does all the data loading, split selection, checkpoint
handling and scoring you need, and whose imports are correct (`resolve_v9` comes from `train_v9_gpu`,
`pearson_rows` is defined locally). **Copy `pearson_rows` verbatim from `interaction_2x2.py`** so the
numbers attach to that script's.

For each alpha in `--alphas` (default `0,0.25,0.5,0.75,1.0`) and each split, compute the **atom effect**:

```
atom_effect(alpha) = score( model(batch,          drug_alpha=alpha) )
                   - score( model(batch_ablated,  drug_alpha=alpha) )
```

where `batch_ablated` replaces `batch['atoms']` with its mean over dim 0 — use
`interaction_2x2.py::make_atom_ablated_batch`, do not re-derive it — and `score` is the **median** of the
per-row Pearson of the predicted delta against `y_delta`.

Emit, per split and per alpha:
- `score_full` and `score_ablated` (the two medians),
- `atom_effect` = their difference,
- **`atom_effect_median_per_row`** = median over rows of the per-row difference, and
- **`sign_test_p`** = a two-sided binomial test on how many of those per-row differences are positive,
  counting only non-zero ones (`scipy.stats.binomtest`),
- a bootstrap 95 % interval for `atom_effect` (20000 resamples, seed 0, one row-resample per draw shared
  by both cells),
- `dY_max` for the atom ablation, so a null stays distinguishable from a branch that never fired.

Why the median and the sign test as well as the mean: on this codebase a paired **mean** over per-row
correlation differences reported 4.2 sigma on an input with no mechanistic path to the ablated module,
while the median of the same rows was `+0.00016` and a sign test gave `p = 0.70`. The mean was reading a
skewed tail. Every location claim here needs all three numbers side by side.

Also emit, once per run: the checkpoint name, `n_eligible` / `n_eval_requested` / `n_eval_bound` per split,
`--batch`, `--n_eval`, `--n_boot`, `--seed`, the alpha list, and the ablated key.

CLI: `--ckpt` (required), `--alphas` (comma-separated, default `0,0.25,0.5,0.75,1.0`), `--batch` (48),
`--n_eval` (1500), `--n_boot` (20000), `--seed` (0), `--key` (default `atoms`).

**All alphas must be scored on IDENTICAL rows in IDENTICAL chunks.** Select and collate the chunks once per
split, then loop over alphas inside that. The atom ablation uses the *chunk* mean, so chunk size is part of
the ablation — that is why `--batch` exists and defaults to 48.

Output JSON to `model/results/v9_alpha_sweep_<ckpt_stem>[_key-<key>].json`. **The filename must carry
`--key` whenever it is not `atoms`**: an earlier script in this repo keyed its output on the checkpoint
alone while another argument varied, and a later run silently overwrote an earlier result. Print a readable
table as it goes, with `flush=True`.

The script must **refuse** on a checkpoint without `_DrugBlock` rather than producing numbers: without it
every alpha returns the same thing and the curve would be a flat line indistinguishable from a real null.
`interaction_2x2.py::check_diagonal_available` already does this check — reuse it.

### 4. Tests — `model/v9/test_alpha_sweep.py`
This codebase has been burned by guards that checked the wrong property: one test asserted a quantiser's
`fitted == 1.0` (that `fit()` had been *called*) while its bins were NaN and every input silently bucketed
to zero, and the model then trained to convergence reporting sensible metrics. **Test behaviour, not
existence.** On a small **untrained** model built directly with `drug_self_attn=True` (construct it; do not
load a checkpoint):

1. **`alpha=1.0` is bit-identical to the default path.** Output with no alpha argument equals output at
   `alpha=1.0` exactly.
2. **`alpha=0.0` is bit-identical to `diagonal=True`.** Difference exactly `0.00e+00`.
3. **Monotone information flow.** Perturb atom 3 and measure how much atom 1's output moves, at
   alpha = 0, 0.25, 0.5, 0.75, 1.0. It must be exactly `0.00e+00` at alpha=0 and **strictly increasing**
   thereafter. Print all five numbers.
4. **Capacity is preserved.** `sum(p.numel())` identical at every alpha.
5. **Padding holds at every alpha.** Changing a padded token moves a real token by exactly `0.00e+00`, for
   all five alphas.
6. **No NaN at any alpha**, including for a fully-masked padded query row: assert 0 non-finite elements.

## Verification you must run before reporting done, in the foreground
```
cd C:\Projects\LINCS\model\v9
python test_alpha_sweep.py
python test_drugsa_v9.py
python test_interaction_2x2.py
python test_v9.py
```
Paste the **full output of all four** into your report. `test_drugsa_v9.py` must still pass all 11 checks,
`test_interaction_2x2.py` all 4, and `test_v9.py` must still report **55/55**. If any of those regress you
have broken something that other results depend on — fix it or report the failure. **Do not report success
without real test output in hand.**

Do **not** run `alpha_sweep.py` against the real checkpoint; that is a GPU job and I will run it.
Do, however, confirm the script's `--help` works and that it refuses correctly on
`../../external/v9_checkpoints/r0_ckpt_v9_fold0_seed0.pt`, which has `drug_self_attn=False`. Paste that
refusal output.

## Constraints
- Touch only `model/v9/modules_v9.py`, `model/v9/model_v9.py`, and the two new files
  `model/v9/alpha_sweep.py`, `model/v9/test_alpha_sweep.py`.
- **Do not** change any default behaviour, any existing test, any `config_v9.py` default, or anything
  outside `model/v9/`.
- **Do not** run git commands, commit, or push. **Do not** train anything. CPU is fine for the tests.
- Match the surrounding register: docstrings in this codebase state *why* a choice was made and cite the
  measurement or review that forced it. Write comments that way.

## Report
Write `C:\Projects\LINCS\research\W6_alpha_doseresponse_REPORT.md` with: the diff you applied, the full
output of all four test runs plus the refusal check, and a section `## What I was unsure about` naming
anything you guessed at. If a requirement could not be met, say which and why — a partial implementation
honestly reported is worth more than a confident one that fails test 2.
