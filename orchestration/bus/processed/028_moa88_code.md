# PACKET 028 — CODE: §88's trainer flags, probe and reader (W20 + PI glue), before any §88 kernel runs
packet_id: 028
created: 2026-09-26
repo_commit: 02c2c26 (code at 2c0bd5d; kernels at 02c2c26)
type: **CODE REVIEW before GPU spend.** Nothing has been trained or probed. The fold-0 seed-0 kernel
(`external/kaggle_kernels/kern_moa88_t0/lincs-moa88-t0.py`) will be pushed when a Kaggle GPU slot frees (~10:55 IST, when C3/C6
finish; C7 takes the other slot). If this review finds a MAJOR problem first, the push is held.

## What changed (all at 2c0bd5d)
- `model/v9/train_v9_gpu.py`: `--post_pathway`, `--dp_seed_mode {lockstep,distinct}` (default lockstep), `--tf32 {on,off}`
  (default on); per-epoch device-RNG equality recorded in `train_meta` with the device seeds, mode, tf32, post_pathway;
  SystemExit if states are equal after epoch 0 under `distinct`. The parser is now `build_parser()` (testable).
- `model/v9/dp_seeding.py`: `seed_devices` / `cuda_states_equal` moved verbatim from `xpert_arm.py` (which imports them);
  one addition: the reseed is skipped for `k ≥ len(torch.cuda.default_generators)` (CPU tests); on real devices it is a no-op
  and the epoch-0 guard catches any failure.
- `model/v9/probe_moa_88.py`: §88.1/88.2/88.6 — rows asserted `3e59a7ba…`; `post_delta()` = per-row ‖a_post(d) − a_post(mean
  drug)‖₂ (eval, no_grad), median over ≤ 4 rows; degeneracy → `void`; gate `rho_post < 0.95`; `score()` (from §86) and
  `score_within_strata()` (Null 1s) on the primary, output-projection and data-projection readouts; strata (seen/unseen;
  responsiveness via `responsiveness_groups()`; `no_landmark_target`), each with both nulls; `strength_quintiles()` returns the
  assignment's sha1, written to the JSON. Guards: `--cfg_from` must have `tcfg.fold == 0`; untrained state_dict keys and
  shapes must equal the cfg_from checkpoint's (`check_arch_match`); a trained checkpoint with `epoch != 11` is refused;
  `post_pathway` must be on.
- `model/v9/read_moa_88.py`: §88.3 applied mechanically to t0–t2 + u0–u4; INVALID on row / quintile / n / epoch / provenance
  mismatch or a void untrained; VOID on a void trained seed; gate failure → NULL; SIGNAL / SEEN-ONLY / NULL by the table. **The
  one form §88.3 leaves open, fixed in the file header before any output:** a reference projection "is target-aligned" iff on
  ANY of the three seeds its diff ≤ −0.02 with p < 0.05 (so a "beyond …" qualifier needs non-alignment on every seed).
- Tests (all run by the PI with `.venv-cuda`): `test_probe_moa_88.py` 8/8 — including a two-strata toy where the within-stratum
  null cannot exceed 0.45 while the unrestricted null does, exact equality with `score()`'s null for one stratum, the post layer
  drug-dependent and the pre-drug layer exactly drug-invariant on a tiny LincsV9, all four guards firing;
  `test_read_moa_88.py` 9/9 (every reading reachable, each guard); regression: `test_probe_moa_v9` 7/7, `test_c7_v9` 12/12,
  `test_candidates_v9` 40/40, `test_xpert_arm_dev` passed, `test_cuda_rng_map` passed.
- W20's own tests were replaced: three of five re-implemented the logic inside the test instead of calling it. W20 also ran
  the suites with the system Python 3.14 in two parallel copies; they were stopped.

## Stated, per review 027 ask 2 (op 2)
By inspection of `train_v9_gpu.main()`: `torch.manual_seed(seed)` is the last CPU-generator event before `LincsV9(...)` is
built (`probe_gpu` draws only on CUDA generators; `data_v9` makes no torch random calls). So u0–u2 share trained seeds 0–2's
initial weights; their quantisers differ (20,000 fitting rows in the probe as §86, 40,000 in the trainer).

## The training kernel (`kern_moa88_t{0,1,2}`)
`train_v9_gpu.py --post_pathway --seed S --epochs 12 --d_model 256 --budget_h 8.5 --gpus 2 --dp_seed_mode distinct --tf32 off`.
Guards: quantiser fix; W20 flags present in the mounted trainer; the C8b layer in the mounted model; ≥ 2 GPUs; after training,
the checkpoint's 16 architecture fields equal the screened C8b's, `tcfg.fold == 0`, `tcfg.batch == 48`, `train_meta` shows
distinct / TF32 off / unequal device states; the checkpoint is renamed `c8b_ckpt_v9_fold0_seed{S}.pt` and its sha1 printed.
The probe kernels (CPU) are generated after training, with each checkpoint's sha1 pinned.

## ASKS
1. Does the code implement §88 + §88.6 as committed? In particular `score_within_strata` (the permutation is within the
   quintile of each compound, quintiles computed once over all D compounds and reused for the strata subsets) and the reader's
   conjunctions.
2. Is the reader's fixed "target-aligned on ANY seed" the right conservative reading of §88.3's qualifiers?
3. Anything in the training kernel that would waste the ~6.5 GPU-h of seed 0?
