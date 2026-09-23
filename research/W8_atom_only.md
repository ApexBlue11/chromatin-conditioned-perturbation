# TASK W8 — the atom-only attention operator, and the sweep option to use it

Precisely specified. The design was set by adversarial review and one refinement (RESULTS §66.2, §67.6);
**do not redesign it.** Repo root `C:\Projects\LINCS`. Work only in `model/v9/`.

## RUN EVERY COMMAND IN THE FOREGROUND AND BLOCK ON IT
Previous workers on this codebase backgrounded their tests and went idle; the session killed them; nothing
was done. **Do not background anything.** Write the report last, with real output in hand.

## Why this exists, in one paragraph
`model/v9` has a drug self-attention block over the token sequence `[global, atom_1 .. atom_n]`. Earlier
experiments masked its attention toward the identity to test whether atom tokens are worth more when atoms
can see each other. A reviewer found that those operators **also cut the global token off from the atoms**,
because the global token is position 0 and the blend covered the whole matrix — and that is probably why the
masked model degraded so badly that no experiment built on it could be read. This task builds the operator
that removes **only atom-to-atom** attention and leaves the global token's pathway intact.

## 1. The operator — `_DrugAttention` in `model/v9/modules_v9.py`
Add a keyword `atom_alpha=1.0` to `_DrugAttention.forward(x, key_mask=None, diagonal=False, alpha=1.0)`.

Let `S` be the attention logits with padded keys set to `-inf` (follow the existing explicit branch that
implements `alpha`), and `A = softmax(S, dim=-1)` with `nan_to_num` for fully masked rows, exactly as the
existing `alpha` branch does.

Define `B`, the same logits restricted to the key set `{0, i}` for each atom query row `i >= 1`:

```python
L = S.shape[-1]
allow = torch.zeros(L, L, dtype=torch.bool, device=S.device)
allow[:, 0] = True                      # every row may attend to the global token
allow[torch.arange(L), torch.arange(L)] = True   # and to itself
S_b = S.masked_fill(~allow, float('-inf'))       # padded keys are already -inf in S
B = torch.nan_to_num(torch.softmax(S_b, dim=-1), nan=0.0)
```

Then:

```python
A_new = A.clone()
A_new[..., 1:, :] = atom_alpha * A[..., 1:, :] + (1.0 - atom_alpha) * B[..., 1:, :]
# row 0 (the global token) is left EXACTLY as A: it keeps reading every atom
out = self.drop(A_new) @ v
```

Requirements:
- **`atom_alpha == 1.0` (and `alpha == 1.0`, `diagonal == False`) must delegate to `super().forward()`** so the
  default path is bit-identical by construction.
- **Combining is undefined:** if `atom_alpha < 1.0` and either `alpha < 1.0` or `diagonal` is set, `raise
  ValueError`. Do not invent a combination.
- No renormalisation: `A` and `B` are both row-stochastic, so every convex combination is.
- Store `self._last_rowsum_maxdev` = the max over **non-padded query rows** of `|A_new.sum(-1) - 1|` whenever
  `atom_alpha < 1.0`, so a test can read it. Do not store the matrix itself.

## 2. Threading — `model/v9/model_v9.py`
`PerturbBlock.forward` and `LincsV9.forward` already thread `drug_alpha`, including reading it from the batch
dict (`model_v9.py` around line 120). Thread **`drug_atom_alpha`** the same two ways, default `1.0`. Keep
every existing signature and keyword working; `interaction_2x2.py`, `alpha_sweep.py` and `probe_v9.py` call
these.

## 3. The sweep — `model/v9/alpha_sweep.py`
Add `--operator {full, atom_only}`, default **`full`**.
- `full` must behave **exactly** as the script does today — same code path, same output filename, same JSON.
  This is a regression requirement and it will be checked by diffing outputs.
- `atom_only` sweeps `drug_atom_alpha` over `--alphas` instead of `drug_alpha`, for both the full and the
  atom-ablated batch. Everything else — row selection, the two RNG generators, `rows_sha`, the estimand and
  its CI, the per-row npz dump, the refusal on a checkpoint without drug self-attention — is **unchanged and
  reused**, not rewritten.
- The output filename must carry `_op-atom_only` when `--operator atom_only`, and the JSON must record
  `"operator"`. An earlier script in this repo keyed its output on only some of its arguments and a later run
  silently overwrote an earlier result, three times.

## 4. Tests — create `model/v9/test_atom_only.py`
This codebase has shipped tests that could not fail: one asserted a quantiser's `fitted == 1.0` while every
bin was NaN; another counted parameters once and asserted nothing. **Every check below must be written as
an assertion that can fail.** The expected form is given as code; follow it.

Build an **untrained** `_DrugBlock` directly (small `d_model`, `dropout=0`, `.eval()`), inputs `D` of shape
`(2, 7, d)` with `key_mask[:, 5:] = True` (two padded tokens), and a small untrained `LincsV9` with
`drug_self_attn=True` and **at least 2 perturb blocks** for T11.

```python
def moved(out_a, out_b, pos): return (out_a[:, pos] - out_b[:, pos]).abs().max().item()
```

- **T1** `atom_alpha=1.0` equals the default call exactly: `torch.equal(...)`.
- **T2** single block, `atom_alpha=0.0`: perturb atom 3 (`D2[:, 3] += 5`) → `moved(o, o2, 1) == 0.0`.
- **T3** single block, `atom_alpha=0.0`: perturb the **global** token (`D[:, 0]`) → `moved(o, o3, 1) > 1e-4`.
- **T4** single block, `atom_alpha=0.0`: perturb atom 3 → the **global** token moves: `moved(o, o2, 0) > 1e-4`.
- **T5** the contrast that justifies the operator: with the **retired** `alpha=0.0`, T3's and T4's quantities
  are both **exactly `0.0`**. Assert that. This proves the new operator differs from the old one exactly where
  intended and nowhere else.
- **T6** `_last_rowsum_maxdev < 1e-6` at `atom_alpha` in {0, 0.25, 0.5, 0.75}.
- **T7** parameter count asserted equal at all five `atom_alpha` values (count each time, compare).
- **T8** padding: perturb a padded token (index 5) → real tokens 0..4 move by exactly `0.0`, at every
  `atom_alpha`.
- **T9** no non-finite values at any `atom_alpha`, including a sample whose atom tokens are all padded.
- **T10** single block, influence of atom 3 on atom 1 at `atom_alpha` in {0, 0.25, 0.5, 0.75, 1}: exactly `0.0`
  at 0 and **strictly increasing** after. Print all five.
- **T11** *(documents, does not forbid)* full `LincsV9`, `drug_atom_alpha=0.0`: perturb atom 3 and measure how
  much the model's `delta` output moves. It is expected to be **non-zero**, because atom 3 reaches the global
  token in one block and the global token reaches the genes and the other atoms in the next. Assert only that
  it is finite, and **print it** with a one-line explanation.
- **T12** `atom_alpha=0.5` together with `alpha=0.5` raises `ValueError`; likewise with `diagonal=True`.

## 5. Verification you must run, in the foreground, and paste in full
```
cd C:\Projects\LINCS\model\v9
python test_atom_only.py
python test_alpha_sweep.py
python test_drugsa_v9.py
python test_interaction_2x2.py
python test_v9.py
python alpha_sweep.py --help
python alpha_sweep.py --ckpt ..\..\external\v9_checkpoints\r0_ckpt_v9_fold0_seed0.pt --operator atom_only --n_eval 96 --n_boot 200
```
The last one uses a checkpoint **without** drug self-attention and must **refuse**; paste the refusal.
`test_drugsa_v9.py` must stay 11/11, `test_interaction_2x2.py` 4/4, `test_v9.py` 55/55,
`test_alpha_sweep.py` all 6. **Do not run `alpha_sweep.py` against any other checkpoint** — that is mine.

## Constraints
- Touch only `modules_v9.py`, `model_v9.py`, `alpha_sweep.py`, and the new `test_atom_only.py`.
- No git. No training. CPU is fine. Match the codebase's comment register: say *why*, cite the RESULTS
  section that forced it.

## Report
`C:\Projects\LINCS\research\W8_atom_only_REPORT.md`: the diff, the full output of every command above, and a
section `## What I was unsure about`. If a requirement could not be met, say which and why.
