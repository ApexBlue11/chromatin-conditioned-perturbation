# REVIEW OF PACKET 030
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 684e6de

**The code implements §90.3 and §90.4 as committed, and nothing here should hold the V1 kernels.**
- **V2 at N = 1** is exactly today's run: the same schedule over every trained step, the same evaluation point, and
  byte-identical saved arrays.
- **The mid-run evaluations under N > 1** don't touch any RNG, so V2 sees the same data order as P2 (ask 2).
- **V1's checks:** two of its guards can fail open, silently passing or yielding a vacuous arm. Each costs one line to
  close (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | code-vs-intent | **Two V1 guards fail open.** (a) The identity check's per-row test is `min_pearson = r.min(); if min_pearson < 0.9999`. If any row's Pearson is NaN (a constant row on either side), `r.min()` is NaN, `NaN < 0.9999` is False, and the whole per-row check passes whatever the other 4,042 rows say. (b) Nothing asserts that the MC arms switched anything. If `isinstance` matched no module, `drop`/`full` would equal `det`, Δ would be exactly 0, and the arm would be recorded as "not accepted" after 2 × ~1 h of CPU. It works today: `modules_v7` is imported once, flat in the upload, and the PI's tests check the switching. But nothing enforces it on the Kaggle import layout. | (a) `if not np.isfinite(r).all() or r.min() < 0.9999: SystemExit`. (b) In the kernel, after each MC arm, assert the sidecar's `switched_module_counts`: `drop` has Dropout > 0 and StochasticDepth == 0, `full` has both > 0. Also assert the arm's `deg_pred` differs from `det`'s (max \|Δ\| > 0). |

## Answers to the asks

**Ask 1 — yes.**
- **The switch:** `model.eval()`, then `.train()` on `nn.Dropout` only (`drop`), or on `nn.Dropout` + `StochasticDepth`
  (`full`). No other module changes mode, and nothing else in `model_v9` / `modules_v9` / `modules_v7` branches on
  `self.training` (029).
- **The seeding:** `torch.manual_seed(1000·seed + k)` before each of the K = 8 passes, with `seed = ck['seed']`. Batches
  of 64 in a fixed order make the masks reproducible.
- **The averaging:** the mean of the passes' `deg_pred` (and `y_pred`), under `no_grad`. The eval loop is xpert_arm's,
  without autocast, and the saved predictions weren't autocast either (autocast wraps only the training step).
- **The identity check** matches §90.3: per-row Pearson ≥ 0.9999, and the dev per-row mean within 5e-5, i.e. 4
  decimals. It runs before `drop`/`full`, and its SystemExit stops the kernel.
- **The model** is rebuilt from `ck['cfg']` with `strict=True`, so the fitted quantiser edges (registered buffers) come
  from the checkpoint.
- **The data** is rebuilt through xpert_arm's own `XPertData` and `batch()`, the same inputs as at training. The rows
  are sha1-asserted on full runs.
- **The paired baseline:** `score_dev.py --centred --preds v1_{arm}_seed{0,1,2} --baseline v1_det_seed{0,1,2}`
  produces every quantity §90.3 reads: the 3-seed mean paired Δ, the per-seed paired Δs, the Δ of cell means, the
  per-cell count, and Δ_centred. `load()` enforces identical `row_index` order across the arms.
- **The pinned checkpoints:** the three sha1s (`bd88a0cd…`, `481f1cc9…`, `3c46faf9…`) match the local files.

**Ask 2 — yes, N = 1 is today's run, and the RNG concern doesn't arise.**
- **Schedule:** `L = steps` when N = 1, so `wsd_factor(s % L, L) = wsd_factor(s, steps)` for every trained step
  s ∈ [0, steps). `wsd_factor` is the old lambda verbatim. The only point where `s % L` wraps is `s = steps`, after the
  last optimiser step, and both forms give 0 there anyway.
- **Evaluation point:** one evaluation at `ep = 11`, after the last training step. The trailing `model.train()` touches
  nothing afterwards.
- **Saved arrays:** a one-element `np.mean` is float64, cast back to float32 on save, which is exact. So the npz files
  are identical. Only the JSON metrics are computed on the float64 copy, which differs by around 1e-7.
- **The numpy RNG under N > 1:** `XPertData.batch` makes **no RNG calls** (`xpert_arm.py:247-266`; atoms are truncated
  deterministically at 96). The only numpy draw in training is `np.random.permutation(D.tr)` per epoch
  (`xpert_arm.py:446`). Evaluation in eval mode draws on neither the torch CPU nor the CUDA generators. So a V2 seed sees
  **exactly P2's data order** for that seed, which makes V2 − P2 purely the schedule plus the averaging.
- **Per-cycle schedule:** with N = 3, `L = 4 · (len(tr) // batch)` divides the total steps exactly. Each cycle warms up
  over 3 % of L and decays over its last 20 % to about 0 at the cycle end. The snapshots are taken at epochs 3, 7 and 11,
  each at an annealed point, as §90.4 defines. `_snap{k}`, `_last` and the main file (the mean) are what §90.4 reads.
- **One untested line:** the tests check `wsd_factor` itself, not the one-line wiring
  `lambda s: wsd_factor(s % L, L)` inside `main()`. I verified that line by reading it. It's too simple to need more.

**Ask 3 — nothing beyond C1(b) would waste the CPU time.** The identity check runs first and stops the kernel on
failure. The checkpoints are sha1-pinned. The mounted `mc_infer_dev.py` is guarded on four strings. The staged
`mc_infer_dev.py` and `xpert_arm.py` match 684e6de byte for byte. Memory is small (8 passes × 4,043 × 978 float32 ≈
250 MB for `y_pred` and `deg_pred` together).

## What I checked and found sound

- **The StochasticDepth class identity.** `modules_v9` imports `StochasticDepth` from `modules_v7`, and so does
  `mc_infer_dev.py`. The upload is flat, with one `modules_v7.py`, so `isinstance` sees one class object.
- **The kernels are identical apart from the seed and its pinned sha1** (s0 vs s1 diffed). Each writes
  `v1_{det,drop,full}_dev6s0_seed{S}.npz` plus a sidecar with the arm, K, seed, switched counts and checkpoint sha1.
- **V2's output naming.** `_snap{N}` enters the JSON tag. The npz names come from `--save_pred`, so the V2 kernel must
  pass its own `--save_pred` name, as every candidate kernel does.

## What I could not assess, and why

- **Wall time on Kaggle CPU.** 17 passes × 64 batches of a 14.9 M-parameter model over 978 gene tokens. The ~2.5 h
  estimate is plausible, but not something I can check locally without loading the laptop.
