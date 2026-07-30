# TPU / XLA engineering notes

Every issue below was hit for real on Kaggle TPU v3-8 (`torch_xla` 2.8), or is a known trap we designed
around. Kept here because TPU failure modes are silent — you get a working run that is 10× too slow, not
an error.

## The mental model
XLA **traces** your Python into a graph, **compiles** it per unique input shape, then executes. Everything
below follows from that:
- a new tensor **shape** ⇒ a new compilation (tens of seconds)
- reading a tensor value in Python (`.item()`, `float(x)`, `print(loss)`) ⇒ a **sync**: the pipeline stalls
- the accelerator is fast enough that **host-side data loading is usually the bottleneck**

## Issues and how each is handled here

| # | Issue | Symptom | Fix in `train_v6_tpu.py` |
|---|---|---|---|
| 1 | **Dynamic shapes** — v5's collate padded atoms to the batch max, so nearly every step had a new shape | permanent recompilation; looks like the model is just slow | `collate(fixed_pad=True)` — always pad to `max_atoms`. Proven not to leak into attention (3 tests, 0.00e+00) |
| 2 | **Compile time counted as step time** | "TPU is 12× slower than a T4" (our actual first measurement) | warm-up steps, then `torch_xla.sync()`, **then** start the timer |
| 3 | **Per-step host sync** — appending `float(loss)` each step | throughput collapses, TPU idles waiting on host | accumulate loss **on device**; one sync per epoch |
| 4 | **`XLA_USE_BF16` set after `import torch_xla`** — the env var is read at import | silently stays fp32; you pay for bf16 hardware and don't use it | **set before any torch_xla import** (launcher), *and* use `torch.autocast("xla", bfloat16)` which has no ordering hazard |
| 5 | **8 processes × full dataset load** — `xmp.spawn` runs `_run` per core, so `load_shared` parses a 310k-row TSV 8× and duplicates arrays | slow start, possible OOM on the VM | keys cached once to `.npz`; processes memory-map / load arrays instead of re-parsing |
| 6 | **Worker oversubscription** — 8 procs × 4 workers = 32 loaders competing | loader thrash, stalls | 2 workers/proc, `persistent_workers`, `prefetch_factor=4` |
| 7 | **`model.to("cpu")` inside the epoch loop** (for checkpoint/probe) | forces sync + can retrigger compilation | checkpoint via `xm.save` (handles device transfer); run the probe **after** training, not per-epoch |
| 8 | **Small per-core batch** — TPU MXU wants large tiles | poor utilisation | batch 32/core ⇒ effective 256 across 8 cores |
| 9 | **Eval inside the training graph** — different shapes ⇒ recompile | mid-run stalls | eval/probe deferred to a separate CPU pass on the saved checkpoint |
| 10 | **Kaggle TPU queue** can exceed an hour | looks "stuck"; easy to misread a stale poller as still-queued | check the kernel status **directly** and pull `output` — a completed run has output even if a poller says otherwise |
| 11 | **Non-clean exit loses `/kaggle/working`** | checkpoints gone | never hard-cancel; time budget triggers a clean checkpoint + exit |
| 12 | **`torch_xla` 2.8 API drift** — `xm.xrt_world_size` / `xm.xla_device` removed or deprecated | `AttributeError` at start-up | `xr.world_size()`, `xr.global_ordinal()`, `torch_xla.device()` |

## Rules of thumb
- **Never** put a Python `if` on a tensor value inside the step — it forces a sync and can change the graph.
- Keep the step function shape-stable: same batch size every step (`drop_last=True`), fixed padding.
- Measure `ms/sample`, post-warm-up, and compare against a known GPU baseline before believing any speedup.
- Prefer one long run to many short ones: every kernel start pays compilation again.

## What we measured
- Single core, our 978×978 attention: **39.4 ms/iter vs ~30 ms on a T4** ⇒ **TPU is slower per core**. The
  only reason to use it is **8 cores** (~3.1× aggregate, and only if the input pipeline keeps up).
- v5 smoke: model trains correctly (loss 1.264 → 0.482, finite) — correctness was never the problem.
