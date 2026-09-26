# REVIEW OF PACKET 029
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 0fd2a0f

**The observation reproduces exactly, and it survives the check that matters most.** The seed ensemble's gain holds on
a drug-specific, cell-centred metric, so it's not a drift toward each cell's average response. Two parts of its framing
overreach (C4):
- "~10× the largest": it's ~3×.
- "between-basin": EMA's window under WSD can't test that.

The batch is reasonable. One requirement must be added before V1 is pre-registered (C1). Two definitions need fixing
(C2, C3).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **V1 can raise the per-row score without using the drug more, and nothing in the design would show it.** `StochasticDepth` drops a whole residual branch per row (`modules_v7.py:106-120`), and the perturb blocks' drug cross-attention branch has one (`self.sdc`, `model_v9.py:51`). With stochastic depth on at inference, some passes predict a row with its drug branch removed. The K-pass average is then pulled toward a drug-agnostic prediction. Per-row Pearson can rise from that pull alone, which is the failure the May-2026 benchmark (81e7075) reports for seven L1000 models. The check exists and is cheap. Subtract, within each dev cell, the cell-mean prediction and the cell-mean truth, then take per-row Pearson. That's invariant to uniform pull toward the cell's mean response. **The seed ensemble passes it:** centred single seeds 0.4749 → 3-seed average 0.5018, **+0.0270 of the raw +0.0293** (C8b: +0.0258 of +0.0281). So the criterion is not arbitrary, and the thing V1 is meant to imitate already meets it. | Pre-register the **cell-centred per-row score** as a required co-criterion for every variance-reduction arm (V1, V2): Δ_centred > 0 on the 3-seed mean, reported beside the raw Δ. Define V1's switch exactly: `model.eval()`, then `.train()` on `nn.Dropout` and `StochasticDepth` modules only. Report a **dropout-only** arm (stochastic depth off) alongside, as it's the variant that can't remove the drug branch. |
| 2 | MINOR | code-vs-intent | **V2 conflates a schedule change with ensembling, and "per-cycle WSD" isn't yet exact.** Three 4-epoch cycles change the single model too: the last snapshot has had three warm restarts and 4-epoch annealing. So V2 − P2 mixes "cyclic schedule" with "averaging 3 snapshots". It's also undefined whether each cycle has its own warmup at 0.03 of its steps and decay at 0.20 of its steps, or only cycle 1 warms up. | Report **V2-last** (the final snapshot alone) as well as V2 (the 3-snapshot average). V2 − V2-last is the ensembling gain; V2-last − P2 is the schedule effect. Write the LR function per step exactly (per-cycle warmup and decay fractions), with an acceptance test that plots or asserts it on a toy step count. |
| 3 | MINOR | stats | **V3's two extra P2 seeds mustn't move μ0 or s0.** §85.2 rule 5 lets the seed sd be "pooled across every completed 3-seed condition". A 5-seed P2 invites recomputing μ0 and s0, and that would change the thresholds under which C1–C4 and C8b were already judged, and C3/C6/C7 are being judged. | State that §85's μ0 = 0.43693 and s0 = 0.00169 are frozen at P2's seeds 0–2 for every §85 decision. V3's seeds are reported only, and pooled into nothing. |
| 4 | MINOR | overreach | **Two phrases in §85.9 overreach.** (a) *"~10× the largest"*: the largest candidate movement is C1's −0.0086, so the ensemble gain is **~3.4×** it; it's ~11× the largest *gain* (C8b +0.0026). (b) *"variance-dominated"*: seed predictions correlate **0.81** per row (P2: 0.808 / 0.807 / 0.814). An equicorrelated model, r_K = r_1·√(K/(1+(K−1)ρ)), fits the observed K = 2 and 3 within 0.0015 (predicted 0.4593 / 0.4676, observed 0.4583 / 0.4662). It puts the infinite-ensemble ceiling at **≈ 0.483–0.486**. So seed variance costs ≈ 0.05 of a 0.44 score, and the rest of the shortfall is shared across seeds. | Retitle §85.9: *"a seed-specific prediction component worth +0.029 at K = 3 (≈ +0.05 as K → ∞), larger than any candidate effect so far"*, and correct the multiplier. |

## Answers to the asks

**Ask 1 — the numbers are right. "Between-basin, not along-trajectory" is a hypothesis, and the EMA null can't license
it.**
- **Reproduced from the six committed dev npz:** P2 single 0.4350 / 0.4383 / 0.4375 against a 3-seed average of
  **0.4662 (+0.0293)**; C8b **0.4676 (+0.0281)**. K = 2 averages 0.4583 (P2) and 0.4601 (C8b).
- **The gain is present in every dev cell,** from +0.019 (LNCAP) to +0.035 (VCAP), unlike any candidate's.
- **On the EMA argument:** EMA at decay 0.999 remembers about 1,000 steps. A 12-epoch dev run has 42,888 / 48 ≈ 893
  steps per epoch, so ≈ 10,700 steps, and WSD's decay phase is the last 20 %, ≈ 2,140 steps. So the EMA window lies
  entirely inside the annealed tail, where the weights barely move, and EMA ≈ the final weights by construction. The
  null says nothing about averaging along the trajectory *before* the decay, which is exactly what V2 tests. Label the
  phrase as an explanation, with V2 as its test.

**Ask 2 — yes, V1's dev evidence may come from C8b, with three conditions.** V1 is an inference procedure, and C8b's
accuracy is indistinguishable from P2's (027). It differs by one extra stochastic-depth branch (`sd_post_path`), which
slightly changes V1's noise.
- **The right reading isn't rule 7's threshold.** `2·√(s0²/3 + s_v²/3)` prices independent retrainings, while V1's Δ
  is paired within each checkpoint. Keep rule 7's **0.003 floor** as the minimum effect. Accept V1 iff:
  - the 3-seed mean paired Δ ≥ 0.003, with Δ > 0 on **each** checkpoint;
  - Δ > 0 on the mean of cell means, and ≥ 4 of 6 cells;
  - Δ_centred > 0 (C1).
- **An identity check first:** the deterministic arm must reproduce `v9dev_c8b_dev6s0_seed{s}.npz` to float tolerance
  before any MC pass. That proves the checkpoint, the quantiser and the rows are the scored model's.
- **"Architecture-agnostic" is the claim that carries V1 into a stack that isn't C8b.** Replicate the sign on at least
  one other architecture's saved dev checkpoint. C3, C6 and C7 run with `--save_ckpt`, so this costs nothing.

**Ask 3 — keep 3 × 4, and make acceptance the same rule-7 form against μ0, not "recovers half the ensemble".** V2
would replace P2's single model at equal training cost, so the decision is V2's per-run score against P2's per-run
score:
- 3 V2 seeds, rule 7's threshold, cells, and Δ_centred > 0 (C1).
- "Fraction of the +0.029 recovered" is a reported quantity: `(V2 − μ0) / (P2 ensemble − μ0)`.

If V2 is adopted, §85.2 rule 10's *"never the ensemble"* needs a one-line amendment: a within-run snapshot average
counts as that run's prediction. The phrase was written to exclude averaging across seeds.

**Ask 4 — (i) is acceptable only with a hard rule, and V1 on XPert only as a labelled sensitivity row.**
- **The rule:** no table row or sentence sets a v9 *ensemble* (across seeds or snapshots) beside XPert's *single run*
  as a comparison. An ensemble row may appear only as v9's own labelled secondary. Any "v9 ensemble vs XPert" row needs
  option (ii), XPert's matched ensemble.
- **V1 on XPert:** "XPert as published" doesn't use MC inference, so V1 on XPert is a sensitivity analysis, not the
  comparison. If V1 is accepted on dev and stacked, it becomes part of v9's method. The headline then compares v9 (with
  V1) against XPert as published, and that's disclosed.
- **On V3's value,** as information for the priority call: the equicorrelated fit (C4) predicts K = 4 ≈ 0.470 and
  K = 5 ≈ 0.473 for P2. V3's 3.4 GPU-h would test that prediction to about ±0.002. It changes little unless the fit is
  wrong.

## What I checked and found sound

- **The observation's inputs:** the same six committed npz files as 027 (rows sha1 `51e7e4ab`, identical targets),
  with averaging on `deg_pred`, the scored quantity.
- **Train-mode behaviour in the model:** apart from `nn.Dropout` and `StochasticDepth`, nothing in `model_v9.py` or
  `modules_v9.py` branches on `self.training`. Every dropout is an `nn.Dropout` module, on branch outputs, or on
  attention weights at `modules_v9.py:275`. None uses SDPA's functional `dropout_p`. So module-type toggling (C1) captures exactly what V1 intends.
- **V1 is feasible on C8b:** the three `v9dev_c8b_dev6s0_seed{0,1,2}.pt` files are local (59 MB each). P2 saved none,
  so C8b is the only complete 3-seed set, as the packet says.

## What I could not assess, and why

- **Whether V1 helps at all.** MC averaging over K = 8 noisy passes can land below the deterministic network. That's
  the experiment, and it's free.
