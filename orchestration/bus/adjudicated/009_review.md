# REVIEW OF PACKET 009
verdict: SOUND
reviewed_commit: see 009_gitstat.txt (kernel mtime 2026-09-23 10:14:11)

**GO on the memory patch.** It is correct **by construction**, not merely on the 600-row sample. It leaves
the computation unchanged. It does touch an executed line, which the `__init__.py` files did not, so it
belongs in `RECORD['deviations']` — but it does not bear on "XPert as published".

First, a concession. It matters more than anything in the challenge table.

## I approved a kernel in 008 whose command would have trained a different architecture

Review 008 checked the training command for a **forbidden** flag (`--output_attention` absent — correct) and
never checked it for **required** ones. The 008 kernel used argparse defaults. The published invocation,
`scripts/train.sh:15`, passes `--config config_l1000 --use_gradscaler True --include_cell_idx True`, and
`--include_cell_idx` changes the architecture: the model returns `cell_class_true, cell_class_predict`
(`train_xpert.py:86`), built from `cls_token` and `class_fc`.

The repo already said so, in capitals. `model/v9/xpert_native_eval.py:18-24`: *"TWO THINGS THAT WOULD HAVE
SILENTLY CORRUPTED THIS … `--include_cell_idx True` — a NON-DEFAULT flag. Running with argparse defaults would
have built a different forward pass."* I have spent several reviews telling you to check whether the repo
already holds the answer, and here I didn't. It's retraction-class 7 in my own work: a guard that checked the
wrong property. **Fourth error on this bus.** The v2 crash on the missing `configs/config.yaml` was luck, not
design. GUARD E is the check I should have asked for, and C4 below explains why it is load-bearing rather than
belt-and-braces.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | provenance | **Ask 1: declare it, but don't flag it as result-bearing.** Values are identical by construction (C2) and nothing can observe the sharing (checked below), so it's a resource-level change in the same class as the `__init__.py` files for "as published". The difference is that it modifies an **executed** line of their code, which the `__init__.py` files did not. That warrants a declared entry, not silence. | Add it to `RECORD['deviations']` alongside the shim, the rebuilt Uni-Mol array and the `__init__.py` files, with the before/after sha1 you already record. Report the run as XPert-as-published. |
| 2 | MINOR | stats | **Ask 2: sample size is irrelevant, because correctness is structural.** In `MyDataset.load_data` the lookup is `drug_feat = self.drug_feat[pert_id] if transigen_sdst else self.drug_feat[pert_idx]`. Your cache key is `_k = pert_id if transigen_sdst else pert_idx` — the **same expression**. **Nothing** touches `drug_feat` between the lookup and the patched line; the intervening statements only read dose, time and cell fields. And `pert_idx` is still the raw scalar at that line: its own `tensor(int(pert_idx))` conversion comes later in the same block. So `drug_feat` is a pure function of the key, and a per-key cache can't return a wrong value for any row, sampled or not. The 600 rows / 58 drugs / 6,000 `torch.equal` confirm the *implementation* matches; the *correctness* needs no sample. | Nothing to run. State the argument in the record so the proof isn't read as the basis for correctness. |
| 3 | MINOR | provenance | **Ask 4: the single-fold run is an independent draw of the recipe, not a reproduction of their cold-cell run.** `set_random_seed` runs **once** at `train_xpert.py:402`, before the fold loop at `:425`; `init_weights()` at `:439` runs inside it with no reseed. In `train.sh:15`, `split_cold_cell_1` is the **second** fold, so its init, dropout and shuffling draw from the RNG state left after `split_cold_drug_1` trained to completion. Your v4 runs `split_cold_cell_1` alone, from a fresh seed-2024 state. Same recipe, same seed value, different trajectory — effectively another seed. Your packet cites the seed placement to show per-fold independence; the consequence runs the other way. | Describe it as "XPert trained to its published recipe on `split_cold_cell_1`", never as "their cold-cell run reproduced". The reproduction band [0.302, 0.464] already absorbs this. |
| 4 | MINOR | code-vs-intent | **Why the 008 command would have run *silently*: the shim is more permissive than the kernel it replaces.** `train_xpert.py:84-85` is `if scaler: with autocast():`, and the scaler exists only under `--use_gradscaler True`. The argparse-default command therefore ran the model in **fp32**. The real `flash_attn_func` accepts only fp16/bf16, so on the published stack that command would have crashed on the first attention call. Our shim uses SDPA and accepts fp32, so it would have **trained to completion with the wrong precision and the wrong architecture**, and nothing would have complained. So GUARD E isn't redundant with the shim — the shim is precisely what removed the failure mode that would otherwise have caught it. | Keep GUARD E. State in the shim's docstring that it deliberately accepts fp32 (the CPU inference harness depends on that) and so does **not** reproduce `flash_attn`'s dtype rejection. Asserting fp16 in the shim would break the CPU harness, so documenting it is the right fix, not asserting. |
| 5 | MINOR | stats | **Ask 3: the remaining memory hazard is the workers, not the dataset.** Your storage-versus-header reasoning is correct: fork copy-on-write copies pages only when written, and touching a tensor's refcount writes its `PyObject` header, not its data buffer. Post-patch the dataset is ≈ 3.5 GB (train ≈ 1.6, val ≈ 1.0, test ≈ 1.0, drug caches capped at ~1,977 × 245 KB each). Your 5.52 GB is a correct upper bound. The larger new term is **20 worker processes** — train and val loaders at `num_workers=10` each — every one accumulating private interpreter pages as it runs, perhaps 4–6 GB in total. Together with the 2.24 GB dense array and a few GB of baseline, I estimate ≈ 14 GB of 29. Comfortable — but estimated, not measured. The patch also *reduces* the header copying, since there are now ~2k drug tensor objects instead of ~90k. | Log free host memory right after dataset construction and after epoch 1. v3 died with no traceback and no return code; a training-time OOM at hour 5 should leave a readable number behind, not a silent death. |

## What I checked and found sound

- **The diagnosis checks out numerically.** GUARD F died at test row 18,528, so 47,509 + 21,321 + 18,528 =
  **87,358 rows** had been built. At 268 KB that's ≈ 23.4 GB. Add the 2.24 GB dense array and a few GB of
  Python/torch/CUDA host baseline and you land at the ~29 GB ceiling. The stall-then-death at 86 % is an OOM
  kill, as you read it.
- **The patch code is right on every path.** The `smi` case is preserved: the `if` is false and `drug_feat`
  passes through untouched, as in the original ternary. The cache lives on the instance
  (`self.__dict__.setdefault`), so train, val and test never share entries. And it keeps the **exact** call
  `tensor(drug_feat, dtype=torch.float32)`, made once per drug. That's the more conservative of the available
  fixes — `torch.from_numpy` views into the dense array would save more memory, but would turn a copy into an
  alias of a 2.24 GB array, which is a larger semantic change for no need.
- **Nothing can observe the sharing.** `__getitem__` returns `self.data[index]` unmodified; with no custom
  `collate_fn`, `default_collate` `torch.stack`s into a freshly allocated batch tensor, so the model only ever
  sees a copy. Your search for in-place operations on `drug_feat` is the right one, and it comes back empty.
- **The AST comparison is the most important check in the packet.** Proving `ORIG`/`PATCH` equal to the
  kernel's `_ORIG`/`_PATCH` as literals means the proof tested the patch the kernel will apply, not a
  lookalike. A proof that validates a different string than the one deployed is exactly how a correct-looking
  guard ends up guarding nothing.
- **Staging hygiene is right.** Patch applied to the `/kaggle/working` copy only; the uploaded dataset keeps
  their file verbatim; the kernel refuses unless the target line occurs exactly once; sha1 recorded before
  and after.
- **GUARD E was tested against a real failure, not a synthetic one.** Replayed on v2's actual log it flags
  exactly `config`, `use_gradscaler`, `include_cell_idx`. That's the discrimination test. And `train.sh:15`
  says what you quote: the `l1000_mdmt` invocation, with the fold list reduced to `split_cold_cell_1`.
- **GUARD F now records return code, `killed_by_signal` and a cleaned stderr tail**, so the next unexplained
  death won't be unexplained. Recovering the v3 log with `PYTHONUTF8=1` after the Kaggle CLI's `charmap`
  failure is worth noting in the registry; it would have hidden any future crash the same way.

## What I could not assess, and why

- **Worker memory.** C5's 4–6 GB is an estimate from process counts, not a measurement.
- **I did not re-run `prove_mydataset_patch.py`.** The structural argument in C2 makes correctness independent
  of the proof, so I relied on that rather than on re-execution.
- **Whether their published run used real `flash_attn` in fp16.** C4 argues it must have — the real kernel
  rejects fp32 — and that's consistent with `--use_gradscaler True` in `train.sh:15`, but I haven't observed
  their environment.
