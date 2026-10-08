# PACKET 066 — PRE-REGISTRATION AMENDMENT: Stage B′ (RESULTS 96.8), before any Stage B′ code or P9 output
packet_id: 066
created: 2026-10-08
repo_commit: (this commit)
type: **PRE-REGISTRATION** (no code yet; the worker writes `score_p9.py` and `stage_b_prime.py` to this contract after
review)

## What changed
96.8 amends 96.4 item 2 ("beyond chemistry": T_v9 > T_R, with a paired (compound, cell) swap null).

1. **A scale confound, found by the PI before any code:**
   - **The problem:** A3's T is a mean of **raw** ULM activity differences (`mechanism_stage_a.run_a3`, lines ~797–818). So
     T_v9 − T_R carries each model's prediction amplitude. A model whose predictions are larger in amplitude would pass the swap
     test without a better direction, since the swap null centres on 0 while the observed difference keeps the amplitude term.
   - **Amended:** the comparison of record uses each model's standardised activities. That is `run_a3`'s `d_std` (per pathway,
     z-scored within the cell over the cell's labelled compounds, on that model's own predictions; review 061 C1).
2. **The swap null:**
   - **The elements:** every labelled (compound, cell) in the cells holding a unit, members **and** non-members.
   - **The swap:** the standardised vectors are exchanged with probability ½. There are 10,000 draws with `default_rng(9470)`,
     and p = (1 + #{Δ* ≥ Δ_obs}) / 10,001, read one-sided at p < 0.05.
   - **Run on:** the excluding reading of record (4 units) and the including reading (9 units). The code refuses unless the unit
     sets are exactly these.
3. **Which v9:** it must hold for the seed-mean **and** every seed, as item 1 does.
4. **The references, pinned:**
   - the training set, the same-cell rule, and the tie and fallback rules;
   - 5-NN similarity weighting;
   - the physchem descriptors, with an exact basic-amine SMARTS checked on 8 molecules, z-scored on training, s = 1/(1+d).
5. **No test response is read** by the references or by B3 on predictions, which reads predictions and labels only. The
   reference predictions and their B3 can be computed before P9 returns.

Also adopted: review 065a's optional ask (P7's `check_snapshot_identities` in the P9 scorer).

## ASKS
1. Is the scale confound real, and is standardising per model (before the swap) the right fix? Would a swap on raw activities
   ever be the better test here?
2. Is swapping members **and** non-members right? The non-members set each unit's baseline (`mu_out`).
3. Is anything in the reference construction under-specified (two implementers could differ) or biased for or against v9?
4. Is "the seed-mean and every seed" right for the comparison, or too strict given 4 units?
