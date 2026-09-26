# PACKET 029 — DESIGN: the dev score is variance-dominated (a 3-seed prediction average gains +0.029); a variance-reduction batch
packet_id: 029
created: 2026-09-26
repo_commit: 0fd2a0f
type: **OBSERVATION (reported, not read) + DESIGN for pre-registration.** Nothing below has been built or run except the
observation, which reuses committed dev predictions.

## A. Observation (computed after the fact from the committed dev `.npz`; reported, not read)
On the 4,043 dev rows (sha1 `51e7e4ab…`), the per-row mean delta Pearson of the **prediction average over seeds 0–2** vs the
mean of the three single-seed scores:

| condition | single seeds | mean | 3-seed prediction average | gain |
|---|---|---|---|---|
| P2 baseline | 0.4350 / 0.4383 / 0.4375 | 0.4369 | **0.4662** | **+0.0293** |
| C8b | 0.4376 / 0.4393 / 0.4417 | 0.4395 | **0.4676** | **+0.0281** |

Every §85 candidate so far moves the single-seed score by < 0.009 in either direction; the ensemble gain is ~10× the
largest. EMA was measured null twice (§21 v7, §38.4 v9: WSD already anneals to ~0 LR), so the gain is between-basin, not
along-trajectory. §85.2 rule 10 already reports "a 3-seed prediction average … separately, labelled as an ensemble" for P7.

## B. Proposed batch (to be pre-registered after review; nothing is built)
| id | what | cost | compared against |
|---|---|---|---|
| **V1** | **MC-dropout averaging at inference:** dropout and stochastic depth left ON at inference, K = 8 passes per row, predictions averaged; seeds of the passes fixed (1000·seed + pass). | **0 GPU-h** — inference on existing checkpoints (the three C8b dev checkpoints are the only saved 3-seed set; Kaggle CPU or ≤ 5 min local GPU inference) | the same checkpoints run deterministically, paired per row (within-model; seed variance does not enter the paired Δ, but the readout is over 3 seeds) |
| **V2** | **Snapshot ensemble within one run:** the same 12-epoch budget as 3 cycles of 4 epochs (WSD per cycle: warm restart to the base LR), predictions of the 3 cycle-end snapshots averaged. One flag in `xpert_arm.py`. | 1.7 GPU-h per seed (same as P2) | P2 (single-seed estimand) **and** P2's 3-seed prediction average (the ensemble it would replace) — both reported |
| **V3** | **Ensemble size:** P2 at seeds 3, 4 (two more single runs), to measure the ensemble curve at K = 1…5. | 2 × 1.7 GPU-h | reported only (it prices K for P7) |

## C. What a result would change
- A V1 gain would be free at P7 and applies to any architecture; V2 would buy a large share of the ensemble at one run's cost.
- For P7 and the head-to-head (§71.3 / §87), an ensemble v9 against a **single** XPert run is not a like-for-like model
  comparison. Options: (i) keep the registered per-seed estimand as the headline and report the ensemble as a labelled
  secondary (as §85.2 rule 10 says now); (ii) also report XPert's own 3-seed ensemble (2 more XPert runs at ~11.5 GPU-h each,
  as O2 took [§87]) so the ensemble rows are matched. I lean to (i) plus V1 applied to both models if V1 works (inference-only
  on XPert's checkpoint too).

## ASKS
1. Is the observation stated correctly, and is "between-basin, not along-trajectory" licensed by §21/§38.4, or is it an
   explanation that should be labelled as such?
2. V1 on C8b's checkpoints (the only saved 3-seed set): is a within-checkpoint paired Δ under §85.2 rule 7's form the right
   reading, given C8b itself was not accepted? (V1 is architecture-agnostic; the question is whether the dev evidence for it
   may come from a non-accepted architecture.)
3. V2's definition: is 3 × 4 epochs with per-cycle WSD the right fixed-budget snapshot design, or should it be pre-registered
   against the ensemble it replaces as its acceptance criterion (e.g. recovers ≥ half of +0.029)?
4. Priority and quota: V1 costs nothing; V2 and V3 would fall in next week's quota with P6/P7. Is the head-to-head framing
   in C (i) acceptable, or does an ensemble row need a matched XPert ensemble?
