# REVIEW OF PACKET 027
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 5af16eb

**Part A is right, and I reproduce it from the six prediction files with my own code.** C8b is NOT ACCEPTED under
rule 7, and the §88.5 gate passes, so §88 executes. There's one mis-statement: seed 0 alone would have been **dropped**
under rule 6, not sent to "one more seed" (C2).

**Part B has one operationalisation that must change before any fold-0 model trains (C1).** It trains a different
architecture from the one the §88.5 gate screened, and one §88.1 doesn't describe. The other four operationalisations
are faithful, with small additions (C4, C5).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | code-vs-intent | **The fold-0 C8b models would use `--l_control 1`. That is neither what §88.1 says nor what the §88.5 gate screened.** §88.1: *"V9Config defaults otherwise"*. `V9Config.l_control` is **2** (`config_v9.py:62`). The screened C8b, whose 3-seed Δ passed the gate, has **l_control 2**. Its checkpoint cfg reads `d_model 256, l_control 2, l_base 2, l_perturb 4, stoch_depth 0.1, drug_self_attn False, post_pathway True`, the P2 command plus the one flag. `l_control 1` comes from `kern_drugsa_s0`'s recipe. So the packet takes batch from the defaults (48, ask 3) but l_control from the recipe. The result is a C8b whose accuracy was never screened, probed under a gate passed by its sibling. A SIGNAL would then be written about "C8b" while describing a different model from the one screened as non-inferior. The effect on the MoA reading is probably small, since the control encoder sits upstream of both pathway layers. But it's a silent departure from a binding text, and it costs nothing to fix now. | **Drop `--l_control 1`** so the default 2 applies, and keep batch 48. The fold-0 C8b then differs from the screened C8b only in its data (fold 0 vs the cc1 dev carve) and its trainer. Assert this in the trainer: the fold-0 cfg's architecture fields (`d_model, l_control, l_base, l_perturb, stoch_depth, drug_self_attn, use_ppi, post_pathway`) equal the dev C8b checkpoint's. If you keep l_control 1 for a reason, declare it in §88.4 as a deviation, together with the fact that the §88.5 gate was passed by the l_control-2 variant. |
| 2 | MINOR | overreach | **"Seed 0 alone (+0.0007) would have been 'one more seed' under rule 6" is wrong. It would have been DROPPED.** Rule 6 drops at Δ < s0 = 0.00169 (§85.8's header: "drop Δ < 0.00169"), and seed 0's Δ is **+0.00066**. This matters for comparisons across candidates. C8b is the only candidate given 3 seeds regardless of its first. C2 (+0.00161 at seed 0, 4 of 6 cells) was dropped on one seed. | Record the correct counterfactual in §85.8's note: *"seed 0 alone: +0.00066 < s0, DROPPED under rule 6; 3 seeds by §88.5 only"*. Any table showing C8b's 3-seed Δ beside the one-seed Δs of dropped candidates must say so. |
| 3 | MINOR | stats | **§88.5's "skip rate for a null-effect C8b ~7 %" is about 11 %.** With s_v ≈ s0, the 3-seed Δ has variance s0²/3 + s0²/3, so its sd is 0.816 s0, and P(Δ < −s0) = Φ(−1.22) ≈ **0.11**. That's the same arithmetic that gives the one-seed 19 % (sd 1.15 s0). A 7 % rate would need an sd ≈ 0.67 s0. It changes nothing now, since the gate passed. | Correct the figure in §88.5 for the record. |
| 4 | MINOR | provenance | **Part B isn't committed yet, and the reader's timing is stated against the wrong event.** 5af16eb holds only part A's records (the §85.8 row, the ledger, the score JSON). The last bus commit is packet 026's. So "committed now, before any §88 model exists" becomes true only when part B is in RESULTS. The reader is to be committed "before the probe runs on any **trained** model". But the untrained JSONs set `m_u` and `sd_u`, which are the thresholds the trained seeds are read against. | Put part B (as amended here) into RESULTS as §88.6, at a commit that predates the first fold-0 kernel push. Commit `read_moa_88.py` before **any** probe output exists, trained or untrained (023 C3's lesson). |
| 5 | MINOR | code-vs-intent | **The untrained-init guard doesn't pin which "seed-0 C8b checkpoint" is meant, and two now exist.** `v9dev_c8b_dev6s0_seed0.pt` (cc1 dev carve, l_control 2) is local, and the fold-0 one will follow. The stated guard, "refuses a cfg without `post_pathway`", accepts both. Untrained models never load a state_dict, so a wrong `--cfg_from` wouldn't crash. It would silently calibrate against a different architecture. Separately, `--budget_h 7.5` can end a run early, and seeds would then differ in training length. `kern_drugsa_s0`, the same recipe with drug self-attention on, took **6.23 h** for 12 epochs, so the margin is about 1.3 h. That's enough, but it isn't guaranteed. | Assert that the `--cfg_from` checkpoint has `tcfg.fold == 0`, and that the untrained model's state_dict keys and shapes equal the trained fold-0 checkpoint's exactly. In the probe, refuse any trained checkpoint whose `epoch` ≠ 11. |

## Answers to the asks

**Ask 1 — yes, apart from C2.** Everything below is from `v9dev_base_dev6s0_seed{0,1,2}.npz` and
`v9dev_c8b_dev6s0_seed{0,1,2}.npz`, recomputed with my own code:
- **Rows:** sha1 `51e7e4ab`, and `y_true − ctl_true` is identical in the baseline and variant files (max difference 0.0).
- **Per-row means:** 0.43760 / 0.43935 / 0.44170, mean 0.439549, sd 0.002058.
- **Deltas:** Δ **+0.00262** on the per-row mean; the mean of cell means goes from 0.4371 to 0.4376, **+0.00046**.
- **Threshold:** 2·√((0.0016893² + 0.0020582²)/3) = **0.00307**, not 0.00308, which is immaterial. Pooling the sd
  gives the same value at equal n. Δ also fails the 0.003 floor alone, so NOT ACCEPTED doesn't depend on `s_v`.
- **Cells:** the instrument's definition, the median of paired row differences on seed-mean scores, gives 3 of 6.
  `score_dev.py` was committed at 0b55d93, before any candidate ran. Per-cell **mean** differences give the same signs
  (HEK293T −0.0105, HL60 −0.0040, LNCAP +0.0123, SKBR3 +0.0026, U937 −0.0009, VCAP +0.0031), so the definition
  doesn't decide it.
- **Gate:** +0.00262 ≥ −0.00169, so it passes. §88.5 was committed at d18c276 (09-25 15:55), and C8b was launched
  around 18:16, so the gate predates the data.

One thing for the write-up, reported and not read. The per-row gain is carried by the three largest dev cells (SKBR3,
VCAP, LNCAP), and the three smallest (HEK293T, U937, HL60) favour P2. The cell-level estimand is about 0. The accurate
sentence is *"C8b's dev accuracy is indistinguishable from the baseline (non-inferior by §88.5; not accepted)"*, never
"a small accuracy gain".

**Ask 2 — ops 2–7 are faithful, with C4 and C5. Op 1 isn't (C1).**
- **Op 2:** building the untrained models from the trained checkpoint's cfg is the right construction. It guarantees
  identical shapes. `M`, the PPI and the gene vectors come from data files, not the checkpoint, so trained and
  untrained models share them. One thing worth checking and stating: `torch.manual_seed(s)` precedes construction. If
  the trainer seeds the same way before building the model, u0–u2 are exactly the starting weights of trained seeds
  0–2. That's fine, since the calibration then contains each trained model's own origin, but say so rather than have a
  reader discover it.
- **Op 4:** "Median over rows of the row-mean |y_Δ|" is a legitimate reading of §88.2. It's better than a median over
  all (row, gene) entries, which the ~900 unresponsive genes would compress. The requirement is that `y_Δ` be the
  **measured** delta, never a model output, so the quintiles are identical across all 8 models. Assert that the
  quintile assignment hashes identically in the 8 JSONs. The one-sided `mean(perm ≤ S)` matches §86.
- **Op 6:** ddof 1 is the conservative choice, since a larger `sd_u` makes `m_u − 2·sd_u` stricter. State that the
  same rule defines `m_u,unseen` and `sd_u,unseen`.
- **Ops 3, 5 and 7:** as §86 and §88.1. Op 5 is reported only.

**Ask 3 — yes, 48.** It's the `V9TrainConfig` default. The screened C8b used 48 (`xpert_arm` default), and so did
`kern_drugsa_s0`, whose script passes no `--batch`. The same reading of "V9Config defaults otherwise" gives
**l_control 2** (C1).

**Ask 4 — the order doesn't affect the validity of either reading.** §88 and P7 use disjoint data and separate models,
and each is pre-registered not to enter the other. So this is a question of priorities, not validity. Three facts bear
on it:
- **§88's cost:** `kern_drugsa_s0` (batch 48, 12 epochs, fold 0) took 6.23 h. So §88's three seeds are nearer
  **18–19 GPU-h** than 17, and slightly more with l_control 2.
- **The follow-up reserve:** "≤ ~5 h" is the typical case, not a bound. If C3, C6 and C7 all advance under rule 6
  (≈ 10.3 h) and C7 is accepted (C7u × 3, ≈ 5.2 h), follow-ups come to ≈ 15.5 h. §88 plus the three seed-0 screens
  already takes ≈ 23–24 h of 30. Rule 6 has advanced 0 of 4 so far, so the typical case is the likely one.
- **What the choice actually is:** P6/P7 is already in next week under the plan, so the real choice this week is §88
  versus follow-up seeds, not §88 versus P7. If follow-up seeds must fit this week, starting §88's seeds after the
  C3/C6/C7 seed-0 screens are read loses nothing, because nothing in §88 depends on them.

## What I checked and found sound

- **The C8b kernel is P2's command plus the one flag `--post_pathway`,** comparing `kern_v9dev_c8b` with
  `kern_v9dev_base2`. The log shows GUARD 4 (sha1 `51e7e4ab`) and GUARD 5: device states unequal on all 12 epochs × 3
  seeds, with device seeds [s, s+1000].
- **The weights read.** `train_v9_gpu.py` evaluates the raw model, not the EMA, so a probe loading `ck['model']`, as
  §86's `probe_moa_v9.py` did, reads the evaluated weights.
- **The rule-7 conjunction applied in the registered form,** with every conjunct recomputed (ask 1). §85.2's
  rule-8 gate wasn't evaluated, correctly, since the accuracy conjunct already fails.

## What I could not assess, and why

- **W20's code** (the trainer flags, `dp_seeding.py`, `probe_moa_88.py`, the reader). It's uncommitted and wasn't
  submitted. It needs its own look once committed, against part B as amended.
- **This week's Kaggle quota use,** which isn't visible from the repo.
