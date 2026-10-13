# PACKET 081 — CODE: D's GPU side (W40 + PI): the h5ad adapter, the cap patch, the inverse, the timing rule, and the timing / run kernels (RESULTS §98.19)
packet_id: 081
created: 2026-10-12
repo_commit: (this commit)
type: **CODE REVIEW** (nothing has run on a GPU; no D output exists. The caption kernels are running on free CPU.)

## What to review
- **`model/v9/p10_pertdit.py`** (W40 + PI):
  - `build_h5ad_d` reuses `p10_mlp.build_h5ad` (guards a–e), then applies the compact `pert_idx`, the float64 `dose_r4`, and
    guards (h1)–(h3). The timing variant has `check_timing_h5ad`, guards (h4)–(h5);
  - `make_cap_trainer` makes 98.16 C2's loop-control patch on the pinned trainer;
  - `check_trainer_cfg` and `check_embeddings` ((e1)–(e4));
  - `convert_outputs_d`: the 98.15 ruling 1 inverse with guards (v1)–(v5). (v3) is the alignment guard: it recomputes `ctrl_t`
    from the h5ad;
  - `parse_timing` and `timing_decision` (98.15 ruling 5).
- **`orchestration/make_p10_d_kernels.py`** (W40 + PI): the T4×2 timing and run kernels. Both are internet-off, on the pinned image,
  with seeds 117 / 118 running concurrently on cuda:0 / cuda:1.
- **`model/v9/test_p10_pertdit.py`** (W40 + PI): 19 tests.
- **The cap trainer's diff against their `train_pertdit.py`:**
```diff
diff --git a/external/bai_benchmark_1d750cd/experiments/_shared/training/train_pertdit.py b/C:/Users/Surya/AppData/Local/Temp/claude/C--Projects-LINCS-LINCS-project-scope-review-1b6bd9/1a2a87ef-e957-4ed6-bb10-ec2e7f9e1823/scratchpad/p10_train_pertdit_cap.py
index 2503930..2ac51f2 100644
--- a/external/bai_benchmark_1d750cd/experiments/_shared/training/train_pertdit.py
+++ b/C:/Users/Surya/AppData/Local/Temp/claude/C--Projects-LINCS-LINCS-project-scope-review-1b6bd9/1a2a87ef-e957-4ed6-bb10-ec2e7f9e1823/scratchpad/p10_train_pertdit_cap.py
@@ -1,3 +1,4 @@
+# Source sha1 a431f53ba18d43348161ee19b97b01148229f1ac; RESULTS 98.16 C2, P10 arm D
 """
 PertDiT training — supports per-sample dose for original-dataset reproduction.
 
@@ -114,6 +115,11 @@ def parse_args():
     p.add_argument("--ablation_mode", type=str, default="none",
                    choices=["none", "zero", "shuffle"])
 
+    p.add_argument("--cap_s", type=float, default=float("inf"))
+    p.add_argument("--t0", type=float, default=None)
+    p.add_argument("--t_test_s", type=float, default=0.0)
+    p.add_argument("--cap_margin_s", type=float, default=900.0)
+
     return p.parse_args()
 
 
@@ -303,12 +309,15 @@ def main():
     print(f"\n[5/7] Training (max {args.n_epochs} epochs, early stop patience="
           f"{args.early_stop_patience} after epoch {args.min_epochs}) ...")
     start_time = time.time()
+    _T0 = args.t0 if args.t0 is not None else start_time
+    stop_reason = 'max_epochs'
 
     best_mse = float('inf')
     best_epoch = 0
     epoch_hist = defaultdict(list)
 
     for epoch in range(args.n_epochs):
+        _ep_start = time.time()
         # ── Train ──
         model.train()
         train_losses = []
@@ -382,6 +391,14 @@ def main():
                 and epoch > args.min_epochs):
             print(f"  Early stopping at epoch {epoch} "
                   f"(no improvement since epoch {best_epoch})")
+            stop_reason = 'rule'
+            break
+
+        # P10 (RESULTS 98.16 C2): loop control only -- stop before the session cap; their best-checkpoint test prediction follows
+        _t_ep = time.time() - _ep_start
+        if time.time() - _T0 + _t_ep + args.t_test_s + args.cap_margin_s > args.cap_s:
+            stop_reason = 'cap'
+            print(f"  P10 cap: stopping after epoch {epoch} (elapsed {time.time() - _T0:.0f}s of cap {args.cap_s:.0f}s)")
             break
 
     train_time = time.time() - start_time
@@ -480,6 +497,8 @@ def main():
         "seed": args.seed,
         "ablation_mode": args.ablation_mode,
         "best_epoch": int(best_epoch),
+        "stop_reason": stop_reason,
+        "cap_s": (None if args.cap_s == float('inf') else args.cap_s),
         "n_epochs_run": len(epoch_hist['train_loss']),
         "wall_time_s": int(train_time),
         "param_count": param_count,
```

## The PI's changes to W40's code (disclosed)
1. **`make_cap_trainer` checks its source's sha1** against `TRAINER_SHA1`.
2. **The builder pins the compound map and dose list the embeddings were built from.** It refuses an `emb_record` whose
   `compound_map_sha1` or dose list differs. W40 had a fallback chain over guessed paths, and a mismatch would have given rows
   another compound's embedding silently.
3. **(h4)/(h5) moved into `check_timing_h5ad`,** with behaviour unchanged. W40's (h4) test re-implemented the guard inside the
   test, so it tested nothing.
4. **W40's two direct trainer tests passed `--dev cuda:0`.** The PI's brief said their code would fall back to CPU, but this laptop
   has a CUDA GPU, so those tiny test runs used it for a few minutes. They now pass `--dev cpu`, and the PI runs the suite with
   CUDA hidden.
5. **Tests added:**
   - h1's count and values, each isolated;
   - h4a / h4b / h5 on the real code, plus a check that `build_h5ad_d` really calls them;
   - each cfg condition;
   - e1's keys and coverage, and e4's keys and lookup, each isolated;
   - v2, v4, v5-seed and v5-n_test, plus the inverse's scale (not just its correlation);
   - in the kernel:
     - the GPU guard, with faked devices (1 × T4, T4 + P100, none);
     - the mounted sha1s and the cap-trainer pin;
     - the `--cap_s` argument;
     - the cfg check and the embedding check, each built so that only it can fire;
     - blinding on a real (tiny, CPU) timing run;
   - the builder's two cross-checks.

## The mutation check (each mutant run against the tests that should kill it; CPU only)
- **The first run, against W40's suite:** **23 of 47 survived.**
  - h1's count and values (each masked by the other);
  - h4a / h4b / h5 (W40's h4 test re-implemented the guard inside the test);
  - both cfg conditions (masked);
  - e1's keys, and e4's keys and lookup;
  - v2, v4, v5-seed and v5-n_test;
  - in the kernel:
    - the GPU device count;
    - the mounted sha1s;
    - the cap-trainer pin;
    - the cfg and embedding checks;
    - blinding;
    - the `--cap_s` argument;
  - the builder's compound-map cross-check.
  (One mutant, inverse-unscaled, wasn't unique in the first run.)
- **After the PI's refactor and 10 tests:** **0 of 51**, counting the 23 survivors plus 4 new mutants: timing-check-not-called,
  P100-allowed, e1-coverage and dose-cross-check.
- **The full CPU run:** 138 passed (19 D, 22 M, 62 score_p10, 35 score_p9).

## ASKS
1. **The cap patch:** is it 98.16 C2 exactly (loop control only)? In particular:
   - the check comes after their early-stop break, so a stop by their rule is never relabelled;
   - `stop_reason ∈ {rule, max_epochs, cap}`;
   - their best-checkpoint test prediction always follows.
2. **The inverse and (v3):**
   - is recomputing `ctrl_t` over all h5ad rows (scanpy's median of positive totals), to 1e-4, a sound alignment guard?
   - `T_train` is recomputed from the h5ad and asserted against 8242.431170.
3. **The timing kernel (98.16 C3):** its h5ad holds no real test row. The 13,445 `test` rows are copies of fit rows with
   `row_index = −(1 + r)`, and only `p10d_timing.json` persists. Anything else it could leak?
4. **The decision inputs:**
   - `t_pre` runs from the kernel's first statement to `[5/7] Training`, so it includes the installs and the h5ad build;
   - `t_epoch` is the (epoch 2 − epoch 0) / 2 of the stamped lines, which includes their per-epoch DDIM validation;
   - `t_test` runs from `[6/7]` to `[7/7]`;
   - each component is the max over the two concurrent seeds.
   Is that the registered P(E)?
5. **The run kernel's cap accounting:** `--t0` is the kernel's start, and `--t_test_s` comes from the timing kernel. Is that
   right?
6. **Blinding parity with M:** the echo allow-list, plus the persisted set, which has no `test_ground_truth`, `per_drug_pcc`,
   `test_sample_ids` or `train_x_pert_mean`.
7. **Anything that makes this not "their harness, a data adapter, and loop control".**
