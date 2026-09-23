# TASK W9 — two inference-only attention cuts, and the sweep flag to use them

Precisely specified. The design is fixed by a pre-registration (RESULTS §79); **do not redesign it.** Repo root
`C:\Projects\LINCS`. Work only in `model/v9/`.

## RUN EVERY COMMAND IN THE FOREGROUND AND BLOCK ON IT
Previous workers on this codebase backgrounded their tests and went idle; the session killed them; nothing was
done. **Do not background anything.** Write the report last, with real output in hand.

## USE THIS PYTHON, AND ONLY THIS ONE
`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`. The last worker ran every test under the system Python, which is
a different environment; its results did not count until they were re-run. Never type bare `python`.

## Why this exists, in one paragraph
`model/v9` has, in each perturb block, a drug self-attention over the token sequence `[global, atom_1 .. atom_n]`
followed by a gene → drug cross-attention in which the 978 gene tokens attend over that sequence. With the existing
`atom_alpha = 0` operator, atoms no longer attend to each other directly, but atom information still reaches the
genes by two routes: (X) the genes read the **atom keys** directly in cross-attention, and (G) the **global token**
(position 0) reads the atoms in drug self-attention and the genes then read the global key. This task adds one
switch per route, so each route can be cut on its own and both can be cut together. With both cut, the output must
be **exactly** independent of the atoms. That exactness is the central acceptance test.

## 1. Cut G — `_DrugAttention` in `model/v9/modules_v9.py`
Add a keyword `global_self_only=False` to
`_DrugAttention.forward(x, key_mask=None, diagonal=False, alpha=1.0, atom_alpha=1.0)`.

Meaning: **the global token's query row (row 0) attends only to key 0.** Every other row is unchanged.

The existing `atom_alpha` branch already builds `B_mat`, the softmax restricted by `allow`, where `allow[:, 0]` and
the diagonal are True. Row 0 of `allow` is therefore `{0}` only, so **`B_mat[..., 0, :]` is already exactly the
one-hot on key 0.** Use it; do not build a second matrix. Key 0 is never padded (`model_v9.py` builds `key_mask`
with position 0 always valid).

Requirements, exactly:
- The early delegation to `super().forward()` happens only when `alpha == 1.0 and atom_alpha == 1.0 and not
  global_self_only`. The default path must stay bit-identical by construction.
- `global_self_only=True` together with `alpha < 1.0` or `diagonal=True` → `raise ValueError`. Do not invent a
  combination.
- Enter the explicit branch when `atom_alpha < 1.0 or global_self_only`. Inside it:
  - Apply the existing atom-row blend **only when `atom_alpha < 1.0`**, exactly as it is now. When
    `atom_alpha == 1.0`, atom rows must be `A` itself — **not** `1.0 * A + 0.0 * B_mat`.
  - Then, if `global_self_only`: `A_new[..., 0, :] = B_mat[..., 0, :]`.
  - Keep storing `self._last_rowsum_maxdev` over non-padded query rows, for every call that enters this branch.

## 2. Threading G — `_DrugBlock.forward` in `modules_v9.py`
Add `global_self_only=False`. Pass it to `self.attn` when the attention module's `forward` accepts it (the dispatcher
already inspects `co_varnames` for `atom_alpha`; do the same for `global_self_only`). **If `global_self_only` is True
and the module does not accept it, `raise ValueError`** — never fall back, never substitute another operator. The
file already has a comment explaining why a silent substitution is the worst failure here (RESULTS 66.2); follow it.

## 3. Cut X, and threading — `model/v9/model_v9.py`
`PerturbBlock.forward(h, D, key_mask, return_attn=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0,
drug_atom_alpha=1.0)` gains `drug_global_self_only=False` and `xattn_global_only=False`:
- `drug_global_self_only=True` while `self.drug_sa is None` → `raise ValueError` (there is no self-attention to cut).
- Pass `global_self_only=drug_global_self_only` to `self.drug_sa(...)`.
- **Cut X:** the cross-attention call `a = self.cross(self.nc(h), D, key_mask)` uses a mask in which, when
  `xattn_global_only` is True, every position except 0 is masked:
  ```python
  xmask = key_mask
  if xattn_global_only:
      xmask = key_mask.clone()
      xmask[:, 1:] = True
  a = self.cross(self.nc(h), D, xmask)
  ```
  `key_mask` itself is **not** modified — the drug self-attention must still see the real mask.
- The returned `D` and the `return_attn` behaviour are unchanged.

`LincsV9.forward` gains `drug_global_self_only=False` and `drug_xattn_global_only=False`, and reads both from the
batch dict with those exact key names, exactly the way it already reads `drug_atom_alpha` (around line 121). Pass
them to **every** perturb block, including the `return_interp` call.

Keep every existing signature and keyword working: `interaction_2x2.py`, `alpha_sweep.py` and `probe_v9.py` call
these.

## 4. The sweep — `model/v9/alpha_sweep.py`
Add `--cut {none, xattn, global, both}`, default **`none`**.
- `none` must behave **exactly** as the script does today: same code path, same output filename, **same JSON, key
  for key** — do **not** add a `"cut"` key when the cut is `none`. This is a regression requirement and will be
  checked by diffing outputs on a real checkpoint.
- Any other value requires `--operator atom_only`; otherwise `ap.error(...)`.
- In `evaluate_chunks_alpha`, set the batch-dict flags on **both** the full and the atom-ablated batch:
  `xattn` → `drug_xattn_global_only = True`; `global` → `drug_global_self_only = True`; `both` → both.
- Output filename: append `_cut-<cut>` right after `_op-atom_only` and before any `_n<n_eval>`. The JSON records
  `"cut"`. An earlier script here keyed its output on only some of its arguments and a later run silently
  overwrote a finished result, three times.
- Nothing else changes: row selection, the two RNG generators, `rows_sha`, the estimand and its CI, the per-row npz
  dump, the refusal on a checkpoint without drug self-attention.

## 5. Tests — create `model/v9/test_residual_cuts.py`
This codebase has shipped tests that could not fail. **Every check below is an assertion that can fail.**

Setup: an **untrained** `_DrugBlock` (small `d_model`, `dropout=0`, `.eval()`), `D` of shape `(2, 7, d)` with
`key_mask[:, 5:] = True`; and an untrained `LincsV9` with `drug_self_attn=True`, **at least 2 perturb blocks**,
`.eval()`, dropout 0 — build it the way `test_atom_only.py` T11 does. Use
`def moved(a, b, pos): return (a[:, pos] - b[:, pos]).abs().max().item()`.

- **T1** Default path unchanged: `_DrugBlock` with `global_self_only=False` gives `torch.equal` output to the call
  without the keyword, at `atom_alpha` 1.0 and 0.0. Same for full `LincsV9` with both new flags False vs omitted.
- **T2** Single block, `atom_alpha=0.0, global_self_only=True`: perturb atom 3 (`D2[:, 3] += 5`) →
  `moved(o, o2, 0) == 0.0` (the global token no longer reads atoms). Contrast, asserted in the same test: without
  the flag, `moved(o, o2, 0) > 1e-4`.
- **T3** Single block, `atom_alpha=1.0, global_self_only=True`: perturb atom 3 → `moved(o, o2, 0) == 0.0`, **and**
  `moved(o, o2, 1) > 1e-4` (atoms still attend to each other at atom_alpha 1).
- **T4** Single block, `atom_alpha=0.0, global_self_only=True`: perturb the **global** token (`D[:, 0] += 5`) →
  `moved(o, o3, 1) > 1e-4` (atoms still read the global token).
- **T5** `_last_rowsum_maxdev < 1e-6` with `global_self_only=True` at `atom_alpha` in {0, 0.5, 1.0}.
- **T6** Padding with `global_self_only=True`: perturb padded token 5 → real tokens 0..4 move by exactly `0.0`.
- **T7 — the acceptance test.** Full `LincsV9`, `drug_atom_alpha=0.0`, **both** `drug_global_self_only=True` and
  `drug_xattn_global_only=True`: replace **all** atom features with fresh random values (keep `atom_mask`) → the
  `delta` output moves by **exactly `0.0`**. Also with the atoms replaced by their batch mean (what the sweep does)
  → exactly `0.0`. Print both numbers.
- **T8** Full `LincsV9`, `drug_atom_alpha=0.0`: with **only** `drug_xattn_global_only=True`, replacing the atoms moves
  `delta` by `> 1e-6` (route G is still open); with **only** `drug_global_self_only=True`, also `> 1e-6` (route X is
  still open). Print both.
- **T9** No non-finite outputs under every flag combination, including a sample whose atoms are all padded.
- **T10** `ValueError` for: `global_self_only=True` with `alpha=0.5`; with `diagonal=True`; and
  `drug_global_self_only=True` on a `LincsV9` built with `drug_self_attn=False`.
- **T11** Threading: passing the two flags as `LincsV9.forward` keywords and as batch-dict keys gives `torch.equal`
  outputs.

## 6. Verification you must run, in the foreground, and paste in full
```
cd C:\Projects\LINCS\model\v9
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_residual_cuts.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_atom_only.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_alpha_sweep.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_drugsa_v9.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_interaction_2x2.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_v9.py
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe alpha_sweep.py --help
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe alpha_sweep.py --ckpt ..\..\external\v9_checkpoints\r0_ckpt_v9_fold0_seed0.pt --operator full --cut xattn --n_eval 96 --n_boot 200
C:\Projects\LINCS\.venv-cuda\Scripts\python.exe alpha_sweep.py --ckpt ..\..\external\v9_checkpoints\r0_ckpt_v9_fold0_seed0.pt --operator atom_only --cut both --alphas 0 --n_eval 96 --n_boot 200
```
The second-to-last must exit with the argparse error. The last uses a checkpoint **without** drug self-attention and
must **refuse**; paste the refusal. `test_drugsa_v9.py` must stay 11/11, `test_interaction_2x2.py` 4/4,
`test_v9.py` 55/55, `test_alpha_sweep.py` all 6, `test_atom_only.py` all passing. **Do not run `alpha_sweep.py`
against any other checkpoint** — that is mine.

## Constraints
- Touch only `modules_v9.py`, `model_v9.py`, `alpha_sweep.py`, and the new `test_residual_cuts.py`.
- No git. No training. CPU is fine. Match the codebase's comment register: say *why*, cite RESULTS §79.

## Report
`C:\Projects\LINCS\research\W9_residual_cuts_REPORT.md`: the diff, the full output of every command above, and a
section `## What I was unsure about`. If a requirement could not be met, say which and why.
