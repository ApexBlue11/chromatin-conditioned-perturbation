# PACKET 004 — REVISED PRE-SPEND DESIGN: drug self-attention
packet_id: 004
created: 2026-09-20
repo_commit: c1cced5
type: **DESIGN + COST REVIEW, STILL BEFORE ANY GPU SPEND.** GPU hours committed to date: 0.
supersedes: packet 003 (NOT-SUPPORTED)

All six challenges from review 003 are discharged. **Two of them changed the answer, and one of the fixes
exposed a new problem with the design that I could not resolve myself — §ASK 1 is the crux.**

## ARTEFACTS
```
model/v9/modules_v9.py::_DrugAttention    diagonal-attention operator (C2)
model/v9/model_v9.py                       threads it through PerturbBlock and LincsV9
model/v9/test_drugsa_v9.py                 11/11 checks
model/v9/interaction_2x2.py                the four-cell harness (C1)
model/v9/test_interaction_2x2.py           4/4 checks
model/v9/atom_ablation_ci.py               the C4 gate
model/results/v9_atom_ablation_CI.json     C4 results, 3 seeds x 3 splits
model/results/v9_interaction_2x2_r0_*.json the refusal record
```
`git show --stat` for the four relevant commits is in `004_gitstat.txt`.

## WHAT CHANGED SINCE 003

### C4 — the effect is real, seed-stable, and ~half the size. Cost: 0 GPU-h.
Three fold0 checkpoints already on disk; local-GPU inference; `pearson_rows` copied verbatim;
`batch 48` held because the chunk mean makes batch part of the ablation.

| split | per-seed d_median (n=1500) | all 3 CIs exclude 0? |
|---|---|---|
| unseen_cell | −0.0032, −0.0039, −0.0015 | **NO — all span zero** |
| unseen_compound | −0.0156, −0.0176, −0.0107 | **yes** |
| unseen_both | −0.0245, −0.0109, −0.0107 | **yes** |

Cross-seed range is **smaller than the within-seed CI width on all three** splits.
§37's −0.00671 / −0.02549 / −0.02187 becomes **−0.0029 / −0.0146 / −0.0142**.
**`unseen_compound` is now the primary split; `unseen_cell` is out — there is no effect there to move.**

### C4 correction — 480 was a CAP, not the population
Your C4 said 480 is "the whole split, not a cap". All three splits reported **exactly 480**, from strata of
47,002 / 58,796 / 15,083 signatures, and this run finds ≥1500 eligible rows in `unseen_cell` alone.
`probe_v9.py` was invoked with `--n_eval 480`, and **the cap is not recorded in its JSON**, which is why it
read as a population size. Your reasoning was wrong; your conclusion was right and is now stronger. Logged
as the first factual error in 36 challenges, caused by our missing provenance. `interaction_2x2.py` records
`n_eligible` and `n_eval_bound`.

### C3 — mechanism restated, premise retracted
`drug_atom_reprs.npy` holds Uni-Mol **`atomic_reprs`**: per-atom transformer-encoder outputs with a 3D
distance bias. **Atom i already encodes its molecular environment.** "A bag with no intramolecular
structure" is retracted. What is under test is a **second, in-loop re-contextualisation that co-evolves
with the cell embedding across blocks** — a much weaker deficit.

### C2 — diagonal-attention operator, built and verified
`_DrugAttention(QKNormAttention)` delegates to `super()` unless `diagonal=True`, so the default path is
bit-identical **by construction**. Verified: cross-atom flow removed at **exactly 0.00e+00**; parameter
count **identical**; not a no-op (`|dY|max` 1.1266, SwiGLU and residual live); padding holds; `test_v9.py`
still 55/55. I additionally checked that a fully-masked padded query row does not produce NaN — 0
non-finite elements.

### C1 — the fourth cell exists
`interaction_2x2.py` computes `S11 / S01 / S10 / S00` on identical rows and identical chunks, and
`INTERACTION = (S11−S01) − (S10−S00)`, bootstrapped by one row-resample with all four medians recomputed on
those same rows. On an SA-off checkpoint it **refuses** and writes a refusal record, rather than silently
returning `S10 == S11` and fabricating a null of exactly 0.0.

### C5 — measured, and I am not quoting it as the cost
50 steps per arm, full width: **2.4636 vs 2.4858 s/step, ratio 1.009x**. But the local step is 2.464 s
where Kaggle T4×2 at batch 48 is 0.437 s — **33.8x slower per row**, a memory-bound 4 GB regime where a
large fixed cost **compresses ratios**. So the true ratio is plausibly larger. Budget **~5.7 h with stated
risk**; abort on the budget guard, not on this estimate.

### C6 — split choice
C1's fix makes the design self-contained (no comparison to the historical −0.025), so the split is now a
question of which regimes we want. `unseen_compound` is fold0-only, so **fold0 at ~5.7 h**. The cheap mdmt
splits would need `interaction_2x2.py` ported to take `--bundle`/`--split`, which is uncosted.

## 🔴 ASK 1 — THE CRUX: is this design powered for the outcome §56.1 makes likely?

The interaction is a difference of two atom-effects, each with a bootstrap CI width of ~0.0132 on
`unseen_compound` [C4]. Propagating, **the interaction's own CI width is ~0.0187**.

| scenario | interaction | vs a 0.0187-wide CI |
|---|---|---|
| **full rescue** (atoms stop hurting entirely) | ≈ +0.0146 | **3.06 σ — detectable** |
| **partial rescue** (half) | ≈ +0.0073 | **~1.5 σ — NOT detectable** |
| no effect | ≈ 0 | null, decisive per your §56.6 asymmetry |

**And C3 makes partial the likely outcome**: the atoms are already Uni-Mol-contextualised, so a second
in-loop pass plausibly recovers only part of the deficit.

So this is a 5.7-hour screen that is **well powered for the outcome I no longer expect and blind to the
one I do**. Your §56.6 point — that a null needs no capacity control and is decisive on its own — holds
only if the null is *informative*, and a null at 1.5σ sensitivity is not.

**Is this worth buying?** Concretely: (a) buy it as a full-rescue-only screen and pre-commit to reading a
null as "no large effect" rather than "no effect"; (b) find a way to tighten the interaction CI without
more training; (c) do not buy it. I lean (b) then (a), but I do not have a method for (b) and that is the
question.

## OTHER ASKS
2. Is there a variance-reduction route to (b)? The four cells share rows and weights, so the four per-row
   score vectors are highly correlated — does that mean the naive √2 propagation overstates the
   interaction's CI, and would a paired/blocked bootstrap on the per-row interaction contrast be tighter?
3. `unseen_both` shows the largest atom effect (−0.0142) and the widest seed spread (range 0.0138 vs CI
   width 0.0149). Is it a better or worse primary split than `unseen_compound`?
4. The C4 effect is measured on SA-**off** checkpoints. Is there any reason the atom effect in an SA-**on**
   model should be assumed comparable in magnitude, given the model was trained with a different drug path?
5. Anything in the discharged items that you consider not actually discharged.
