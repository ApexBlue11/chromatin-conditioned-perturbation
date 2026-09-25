# TASK W14 — A Lightning AI variant of the O2 production kernel generator

## Context
`external/kaggle_kernels/kern_xpert_cc1/generator/make_prod_kernel.py` generates the production training script for a
pre-registered experiment (XPert trained to its published recipe) from `generator/lincs-xpert-cc1_v7_base.py` and
`generator/prod_tail.py`. Its output `external/kaggle_kernels/kern_xpert_cc1/lincs-xpert-cc1.py` runs on Kaggle
(2 x T4, DataParallel). Read all three files first, plus `model/results/RESULTS.md` sections 84.1-84.4 (search for
"### 84."). Section 84.4 lists binding conditions for running later sessions on Lightning AI on ONE large GPU.

## Deliverable
Add a `--platform {kaggle,lightning}` option (default `kaggle`) to `make_prod_kernel.py`:

    python make_prod_kernel.py SESSION [PREV_JSON] [--platform lightning]

**`--platform kaggle` output must be BYTE-IDENTICAL to today's.** For `lightning`, write the result to
`external/lightning/o2/lincs-xpert-cc1_lightning_session<SESSION>.py` (do not overwrite the Kaggle kernel), with these
changes and nothing else:

1. **Paths.** `W = os.environ['LINCS_WORKDIR']` (required; fatal if unset) instead of `/kaggle/working`. Every
   `/kaggle/input` glob uses `os.environ['LINCS_INPUT_ROOT']` (required) instead. No string `/kaggle` may remain.
2. **Budget.** `BUDGET_H = float(os.environ['LINCS_BUDGET_H'])` (required) instead of the literal 8.3.
3. **Platform expectation (84.4 cond. 1)**, right after the existing GPU listing: exactly ONE visible GPU; its name
   contains one of `A100`, `L40S`, `H100`, `H200`; total memory >= 40 GiB. Otherwise `fatal(...)`. Record the GPU name
   and memory in `RECORD['platform']`.
4. **Precision guard (84.4 cond. 3):** fatal unless `torch.backends.cuda.matmul.allow_tf32 is False`,
   `torch.get_float32_matmul_precision() == 'highest'`, and `NVIDIA_TF32_OVERRIDE` not in `os.environ`. Record
   `torch.backends.cudnn.allow_tf32`, the device capability, and `torch.backends.cuda.flash_sdp_enabled()`,
   `mem_efficient_sdp_enabled()`, `math_sdp_enabled()` in `RECORD['platform']`. Also export these into the trainer
   subprocess checks: the same guard must run inside the trainer process (the simplest way: the generated script
   asserts before launching, and passes no env var that changes precision).
5. **GUARD F on one large GPU:** run the probe WITHOUT the activation-checkpoint patch (84.4: `dp` variant, no
   checkpointing), and replace the T4 assert `peak < 13.0` with `peak < 0.85 * <that GPU's total GiB>`.
6. **Chain test (84.4 cond. 6):** the C4 chain test block currently runs `if SESSION == 1`. In the Lightning variant
   it runs when `SESSION == FIRST_LIGHTNING_SESSION`, a literal the generator writes (pass it as
   `--first-lightning-session N`, required for `--platform lightning`).
7. **Declared deviation (84.4 cond. 1)** appended to the deviations list, with the GPU name filled in at runtime:
   "sessions 1-2 on 2xT4 DataParallel; sessions >= 3 on one <GPU> as published; different GPU architecture and SDPA
   backend (rounding-level); CUDA RNG of the second device not carried across".
8. **Amendment E on Lightning (84.4 cond. 5):** in the session log, when `SESSION == FIRST_LIGHTNING_SESSION` and at
   least 2 epochs ran, record `first_two_mean_s` against `LINCS_EPOCH_S_PROBE` (env, required) and a boolean
   `reprice` = mean > 1.25 x probe.
9. Keep everything else identical: the stack-literal check against PREV, the sha1 check of the attached state, the
   markers, the terminations, GUARD A-E, the frozen ten, the memory patch, the shim.

## Tests — new file `external/kaggle_kernels/kern_xpert_cc1/generator/test_make_prod_kernel.py`
- Kaggle regression: generate session 2 with `generator/prev_session2.json` and assert it is byte-identical to the
  committed `external/kaggle_kernels/kern_xpert_cc1/lincs-xpert-cc1.py` (run `git stash`-free: generate to a temp
  path; add an `--out` option if you need one, default unchanged).
- Lightning: generate session 3 with a dummy PREV JSON (same shape as `prev_session2.json`, final_epoch 131) and
  `--first-lightning-session 3`; assert it parses (`ast.parse`), contains no `/kaggle`, contains the precision guard,
  the platform check, the deviation text, and that every embedded `*_SRC` literal is byte-identical to its repo file
  (the generator already asserts this; keep it).
- Print PASS lines. Paste the full output in your final message.

## Rules
- Edit only `make_prod_kernel.py`; create the test file and `external/lightning/o2/` outputs. Do not edit
  `prod_tail.py`, the v7 base, `model/v9/*`, or the committed Kaggle kernel. If a change seems to need `prod_tail.py`,
  do it as a string replacement inside the generator (the generator already works this way) and say so.
- Interpreter: `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` or system `python`. Do not install packages.
- Another worker may be editing `model/v9/xpert_resume_patch.py` concurrently; do not touch it.
