# PACKET 010 — RESULT: the atom-only operator, its kill switch, and its null-key gate
packet_id: 010
created: 2026-09-23
repo_commit: c0a39d2
type: **RESULT.** 0 GPU-hours (local inference on the SA-on checkpoint). Independent of packet 009, which is still
open and blocking the XPert run; please answer 009 first if you are choosing an order.

## OBJECTIVE
Your review 006 C1 found that every drug-attention masking operator used so far also cut the **global token** off
from the atoms, because the global token is position 0 and the blend covered the whole matrix. You proposed the
operator that masks only atom-to-atom attention. This packet reports it.

## FUNCTION
`_DrugAttention.forward(..., atom_alpha)`, `model/v9/modules_v9.py`. Row 0 (the global token) is left exactly as the
learned softmax `A`. Each atom row `i >= 1` is `atom_alpha * A[i] + (1 - atom_alpha) * B[i]`, where `B[i]` is the
softmax of the same logits restricted to the key set `{0, i}`. `atom_alpha = 1` delegates to `super().forward()`.

Verified before use (delegated build W8, all tests rerun by the PI under the project venv):
- in one block at `atom_alpha = 0`: perturbing atom 3 moves atom 1 by exactly 0.0; perturbing the global token moves
  atom 1 by > 1e-4; perturbing atom 3 moves the global token by > 1e-4;
- under the **retired** full-matrix `alpha = 0`, those last two quantities are both exactly 0.0;
- rows sum to 1 within 1e-6; parameter count equal at every α; padding holds exactly; influence of atom 3 on atom 1
  is 0.0 at α = 0 and strictly increasing to α = 1 (0.093, 0.188, 0.284, 0.380);
- across a full multi-block `LincsV9` at α = 0, perturbing atom 3 moves the delta output by 0.515 (the path through
  the global token between blocks);
- regression: the pre-W8 `model/` tree exported from git and the new code, both running `alpha_sweep.py` on
  identical inputs with the default operator — zero differing fields, all 18 per-row arrays byte-identical.

`model/v9/alpha_sweep.py --operator atom_only`: for each α, atom effect = score(atoms present) − score(atoms
replaced by chunk mean), both at that α; estimand of record = median over rows of the per-row difference, plus a
sign test; one bootstrap resample matrix per split shared by all α; row sample drawn by a generator used for
nothing else.

## RULES, committed before the data

**§67.2 kill switch** (committed `3fc6781`, 2026-09-21 21:45 IST):
> *If `S11−S10` under the atom-only mask is not below +0.03 on `unseen_compound`, the operator is abandoned without
> computing any interaction.*

**§67.3 gate** (committed `3fc6781`):
> *The null key's endpoint span must be below 25 % of the hypothesis key's endpoint span on the primary split, in the
> estimand of record. At or above 25 %, the hypothesis curve is not read at all.*

**§67.6 operator refinement** (committed `a639b45`, 2026-09-23 09:29 IST, before the operator was written): the
first definition renormalised the atom submatrix, which would have let atom 3 reach atom 1 through the softmax
normaliser; replaced by the `{0, i}`-restricted softmax above.

**§62.3 reading** (committed `a4d2ef7`), inherited by §67.4: monotone across all five α, endpoint span larger than the
α = 1 estimand CI width, and the null key passing the gate → consistent with mechanism.

First atom-only output committed at `ff5d698`, 2026-09-23 10:09 IST.

## RESULTS
`rows_sha` identical across both keys and equal to the 2x2's row set on all three splits.

**Kill switch**, `score_full(α=1) − score_full(α=0)`:

| split | atom-only | retired full-matrix operator |
|---|---|---|
| unseen_cell | +0.00356 | +0.09050 |
| unseen_compound | **−0.01448** | +0.11679 |
| unseen_both | −0.01469 | +0.09796 |

**Curves**, `atom_effect_median_per_row`, α = 0, 0.25, 0.5, 0.75, 1:

| split | hypothesis key `atoms` | null key `x_cell` | null span / hyp. span |
|---|---|---|---|
| unseen_compound | −0.00322 −0.00453 −0.00591 −0.00775 −0.01129 | +0.01538 +0.01509 +0.01523 +0.01502 +0.01444 | **11.6 %** |
| unseen_both | −0.00626 −0.00738 −0.00883 −0.01002 −0.01414 | +0.00489 +0.00454 +0.00462 +0.00440 +0.00378 | 14.1 % |
| unseen_cell | +0.00100 +0.00160 +0.00186 +0.00255 +0.00290 | +0.00868 +0.00866 +0.00878 +0.00900 +0.00957 | 46.8 % |

For reference, the same null key under the retired operator: 21 % / 62 % / 66 % on cell / compound / both.

**Estimand CIs on `unseen_compound`**: α = 0 [−0.00433, −0.00226]; α = 1 [−0.01395, −0.00908], width 0.00487.
Endpoint span 0.00807. Sign-test p from 1.6e−10 (α = 0) to 1.0e−35 (α = 1). On `unseen_both` the α = 1 width is
0.00355 against a span of 0.00788. On `unseen_cell` the α = 1 width is 0.00202 against a span of 0.00190.

## WHAT WAS CONTROLLED
One set of weights; α changes only the drug self-attention matrix; identical rows and chunks across every α and both
keys; atom ablation is the chunk mean at `--batch 48` as in §57; the script was not modified between the hypothesis
and null-key runs.

## PRIOR RETRACTIONS IN SCOPE
- §60.9 item 2 and §65: the 2x2 and the full-matrix α sweep both failed their null key and the operator was retired.
- §66.2: that operator cut the global token off from the atoms.
- §63.3: a threshold that named the wrong statistic's interval.

## ASKS
1. Do the kill switch and the gate pass as written, on your own reading?
2. Is "consistent with mechanism" licensed here, and at what strength?
3. **The residual at α = 0.** The atom effect is still −0.00322 with atom-to-atom attention removed, sign p 1.6e−10.
   The atom tokens also reach the genes through gene→drug **cross**-attention, which this operator does not touch.
   Is that the likely carrier, and is there a clean way to test it?
4. The trained arm this points to — keep the atom tokens and global↔atom attention, drop atom-to-atom — costs one
   ~6 h run. What would it have to be compared against, and what should be pre-registered, for its result to be
   readable?
5. Anything else.
