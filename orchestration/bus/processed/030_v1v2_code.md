# PACKET 030 — CODE: §90's V1 inference script and V2 flag (W21 + PI fix), before any V1 kernel runs
packet_id: 030
created: 2026-09-26
repo_commit: 684e6de
type: **CODE REVIEW.** Nothing has run on real dev rows beyond a 128-row identity check. V1 costs 0 GPU-h (Kaggle CPU); V2 will
cost GPU next week. The three V1 kernels (`external/kaggle_kernels/kern_v1mc_s{0,1,2}`) are pushed after this review.

## What changed (684e6de)
- `model/v9/mc_infer_dev.py` (new): rebuilds the dev carve with xpert_arm's `XPertData` (asserting dev-row sha1 `51e7e4ab…` on
  full runs), the model from `ck['cfg']` with `strict=True`; `set_mc_mode(model, arm)` (`eval()`, then `.train()` on
  `nn.Dropout` for `drop`, and on `nn.Dropout` + `StochasticDepth` for `full`; returns counts); `predict_rows` = xpert_arm's eval
  loop (batches of 64, no_grad, no autocast); `mc_predict` seeds `torch.manual_seed(1000·seed + k)` (and CUDA) before pass k,
  K = 8, and averages `y_pred` / `deg_pred`; `--identity_check SAVED.npz` asserts per-row Pearson(recomputed, saved) ≥ 0.9999 on
  every row and the dev per-row mean within 5e-5. Output `.npz` in xpert_arm's `--save_pred` format plus a JSON sidecar
  (arm, K, seed, switched counts, ckpt sha1).
- `model/v9/xpert_arm.py`: `wsd_factor(s, steps)` (today's lambda verbatim); `--snapshot_cycles N` (default 1): LR factor
  `wsd_factor(s % L, L)` with `L = (epochs // N) · (len(tr) // batch)` (asserts `epochs % N == 0`); the eval loop factored into
  `eval_model()`, run after the last epoch of each cycle (then `model.train()`); with N > 1 saves `_snap{k}`, `_last` and the main
  file = mean of the N snapshots; the reported metrics use the mean; tag `_snap{N}`; `snapshot_cycles` in `candidate_flags`.
  With N = 1: `L = steps`, one eval at the end at the same point as before, the saved arrays identical (a one-element mean cast
  back to float32).
- Tests (PI-run, `.venv-cuda`): `test_mc_infer_dev.py` 4/4 — module switching per arm; `drop` deterministic under its seeds
  and ≠ `det`; with dropout 0 and stochastic depth 0.5, `drop` == `det` exactly and `full` ≠ `det`; `wsd_factor` equals today's
  lambda on 100 steps and per-cycle on 120. **PI fix:** the worker's test set `cfg.stochastic_depth`, which is not a field
  (`stoch_depth` is) — the test passed for the wrong reason (default 0.1); corrected and checked that the built model's SD rates
  are then 0.0 / up to 0.5. `test_candidates_v9.py` 45/45 (adds the 128-row identity check on `v9dev_c8b_dev6s0_seed0` and a
  3-cycle smoke: three `_snap` files, `_last` = snap 2, main = mean of snaps); `test_xpert_arm_dev` pass; `test_c7_v9` 12/12;
  §88 suites 18/18.

## The V1 kernels (`kern_v1mc_s{0,1,2}`, CPU)
Mount `lincs-v9-bundle`, `lincs-v9-src` (the 684e6de upload), `xpert-mdmt-benchmark`, and the output of kernel
`lincs-v9dev-c8b` (the three checkpoints and saved predictions). Each pins its checkpoint's sha1 (bd88a0cd…, 481f1cc9…,
3c46faf9…), runs `det` with the full-row identity check first (SystemExit on failure), then `drop`, then `full`, K = 8. Scoring
afterwards, locally: `score_dev.py --centred --preds v1_{arm}_seed{0,1,2} --baseline v1_det_seed{0,1,2}` (paired, recomputed
deterministic baseline, §90.3), read by §90.3's rule.

## ASKS
1. Does `mc_infer_dev.py` implement §90.3 exactly — the switch, the seeding, the averaging, the identity check, and scoring the
   recomputed deterministic arm as the paired baseline?
2. Does `--snapshot_cycles` implement §90.4 exactly, and is N = 1 really today's run (schedule, eval point, saved arrays)?
   One thing I noted and did not change: with N > 1 the mid-run evaluations call `D.batch`, and if `D.batch` draws from the numpy
   RNG, later epochs' permutations differ from an N = 1 run — harmless for V2's reading, but say if it matters.
3. Anything in the V1 kernels that would waste their ~2.5 h of CPU each?
