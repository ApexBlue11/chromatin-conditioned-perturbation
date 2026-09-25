# TASK W17 — Implement the six §85.7 candidate flags for v9, exact when off, each with its acceptance test

## Read first
1. `model/results/RESULTS.md`, section `### 85.7` — **the binding definitions and acceptance tests. Implement exactly those.**
2. `model/v9/xpert_arm.py` (the dev harness), `model/v9/model_v9.py` (`LincsV9.forward`, `v9_loss`, `cfg_use`),
   `model/v9/modules_v9.py` (`ControlEncoder`, `NamedPathwayReadout`, `MultiTaskHeads`, `_GeneBlock`),
   `model/v9/config_v9.py`, and `model/v9/test_v9.py` (how a tiny `LincsV9` is built for tests).

## Deliverable A — the flags (in `xpert_arm.py`, threaded into `V9Config`, the model and the loss)
`--no_atoms`, `--l_control N` (default = today's value, 2; C2 is `--l_control 0`), `--listnet_w W` (default 0),
`--deg_adapt_k K` (default 0 = off), `--sign_head_w W` (default 0), `--post_pathway`. Definitions exactly as §85.7:
- C1: the drug sequence is `[global]` only; atom tokens reach nothing.
- C2: `ControlEncoder` with zero `_GeneBlock`s for BOTH control views; the expression embedding path is unchanged.
- C3: symmetric ListNet, `z = (x - row mean) / (row sd + 1e-3)`, tau = 1, on the delta loss's rows.
- C4: `w_delta * (a_all * L_all + a_DE * L_DE)`, `a_all = detach((L_all + L_DE) / (2 * L_all))`,
  `a_DE = detach((L_all + L_DE) / (2 * L_DE))`, per batch; `L_DE` = Huber over each row's top-K genes by `|y_delta|`.
- C6: `nn.Linear(d_model, 1)` on the final gene tokens; BCE-with-logits vs `1[y_delta > 0]` on each row's top-50 genes.
- C8b: a second `NamedPathwayReadout` (same `M`, `d_pathway`) after the LAST perturb block, `h = h + sd(path_delta)` with
  stochastic depth at `cfg.stoch_depth`; unsupervised; `aux['pathway_activations']` stays the PRE layer; the new
  activations go in `aux['post_pathway_activations']`.
Record every flag in the result JSON (`'candidate_flags'`); add an output-name tag only when a flag is non-default
(`_noatoms`, `_lctl0`, `_listnet`, `_degk50`, `_signhead`, `_postpath`).

## Deliverable B — `model/v9/calibrate_aux_weights.py` (§85.7 "Auxiliary weights by rule")
Builds the baseline configuration exactly as `xpert_arm.py` does (dev mode, `--dev_cells 6 --dev_seed 0`, seed 0, fresh
init, fp32, CPU, no training step taken), iterates the first 20 batches of seed 0's training order on the dev training
rows, and for each auxiliary term (C3 ListNet at unit weight; C6 sign BCE at unit weight) computes
`||grad_theta L_delta|| / ||grad_theta L_aux||` (L_delta = the weighted delta Huber term only), averages the ratio over
the 20 batches, and prints `w3 = 0.1 * ratio_listnet`, `w6 = 0.1 * ratio_sign` as JSON to
`model/results/v9_aux_weight_calibration.json`. **Do not run it** beyond a `--n_batches 2` smoke test; the PI runs it.

## Deliverable C — `model/v9/test_candidates_v9.py` (CPU, fast)
1. **Off = today:** all flags default → outputs and `v9_loss` on a fixed seeded batch bitwise equal to the committed code
   (import the committed modules from `git show HEAD:<path>` into temp modules; say how you did it).
2. The six acceptance tests of §85.7's table, each an assert, verbatim in intent.
3. Deliverable B's smoke test runs and writes finite ratios.
Run it and `model/v9/test_xpert_arm_dev.py`; paste full outputs in your final message.

## Rules
- Edit only `model/v9/xpert_arm.py`, `model/v9/model_v9.py`, `model/v9/modules_v9.py`, `model/v9/config_v9.py`; create
  the two new files. Interpreter `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`. No installs. No training.
- Do not change any default behaviour. Do not touch `probe_moa_v9.py`, `score_dev.py`, `external/`, `RESULTS.md`.
- Leave no scratch files in the repository root.
