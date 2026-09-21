# PACKET 006 — the gate result on your C3, and a DESIGN for review before any spend
packet_id: 006
created: 2026-09-21
repo_commit: 5076f9c
type: **RESULT (free) + DESIGN, BEFORE SPEND.** GPU hours this session: 6.23, all of it the run packet
004 approved. Everything in Part 1 cost zero.

Two parts. **Part 1** reports the dose-response you proposed in review 005 C3, and its null-key gate.
**Part 2** is a new design, put up before any hours are committed.

---

# PART 1 — the alpha dose-response and its gate

## OBJECTIVE
Whether the 2x2 interaction operator (masking drug self-attention to the identity) measures a drug-side
mechanism at all, or measures generic degradation. Review 005 C3 proposed the discriminator: blend the
post-softmax attention `alpha*A + (1-alpha)*I` across alpha in {0, 0.25, 0.5, 0.75, 1} and read the
shape. Review 005 C1 established that the previous estimand failed its own null key, so a gate was fixed
in advance this time.

## FUNCTION
`model/v9/alpha_sweep.py` (delegated W6, brief in `research/W6_alpha_doseresponse.md`).
`_DrugAttention.forward` gained an `alpha` argument blending the post-softmax attention matrix with the
identity; both have rows summing to 1 so no renormalisation is applied. Per split the script collates
chunks **once**, then loops alphas inside, and computes for each alpha:

```
atom_effect(alpha) = median_rows Pearson( model(batch,         drug_alpha=alpha) )
                   - median_rows Pearson( model(batch_ablated, drug_alpha=alpha) )
```
with `batch_ablated` replacing `batch['atoms']` by its chunk mean (`interaction_2x2.py::make_atom_ablated_batch`).
Emitted per alpha: that difference of medians with a bootstrap CI; `atom_effect_median_per_row` (the
**estimand of record** per your C2) ; a two-sided sign test on the per-row differences; `dY_max`.

Operator verification, run by the PI, not read from the worker's report:
- `alpha=1.0` **bit-identical** to the default path (0.00e+00) — it delegates to `super().forward()`.
- `alpha=0.0` **bit-identical** to the existing `diagonal=True` path (0.00e+00).
- Cross-atom information flow strictly increasing in alpha: 0, 0.858, 1.711, 2.562, 3.409.
- Parameter count asserted identical at all five alphas; padding holds at all five; no NaN.
- `test_drugsa_v9.py` 11/11, `test_interaction_2x2.py` 4/4, `test_v9.py` **55/55**.

Three defects in the delegated code were found and fixed by the PI before use: a capacity test that
counted once and asserted nothing; row selection and the per-alpha bootstrap drawn from **one** generator
with the bootstrap inside the alpha loop (so the row sample depended on how many alphas were requested);
and no row provenance.

## RESULTS

**Anchoring.** `rows_sha` matches `interaction_2x2.py`'s row set on all three splits for **both** the
hypothesis and the null-key run. Endpoints reproduce packet 005's 2x2 cells exactly, CIs included:

| | alpha=0 vs S10−S00 | alpha=1 vs S11−S01 |
|---|---|---|
| unseen_cell | −0.01012 = −0.01012 | +0.00530 = +0.00530 |
| unseen_compound | −0.01697 = −0.01697 | −0.01814 = −0.01814 |
| unseen_both | −0.01843 = −0.01843 | −0.02637 = −0.02637 |

**Hypothesis key (`atoms`), estimand of record `atom_effect_median_per_row`:**

| alpha | unseen_cell | unseen_compound | unseen_both |
|---|---|---|---|
| 0.00 | −0.00468 | −0.00637 | −0.01122 |
| 0.25 | −0.00093 | −0.00753 | −0.01195 |
| 0.50 | +0.00105 | −0.00831 | −0.01267 |
| 0.75 | +0.00208 | −0.00894 | −0.01266 |
| 1.00 | +0.00290 | −0.01129 | −0.01414 |
| successive diffs | +375 +198 +102 +82 (e−5) | −116 −78 −63 −235 (e−5) | −73 −72 **+1** −148 (e−5) |

**Null key (`x_cell`, the cell control expression), identical rows, chunks and weights:**

| alpha | unseen_cell | unseen_compound | unseen_both |
|---|---|---|---|
| 0.00 | +0.00797 | +0.01140 | +0.00186 |
| 0.25 | +0.01055 | +0.01303 | +0.00305 |
| 0.50 | +0.01032 | +0.01404 | +0.00398 |
| 0.75 | +0.00985 | +0.01502 | +0.00476 |
| 1.00 | +0.00957 | +0.01444 | +0.00378 |

| split | null-key endpoint span | hypothesis-key span | null as % of hypothesis |
|---|---|---|---|
| unseen_cell | 0.00160 | 0.00758 | 21 % |
| **unseen_compound** | **0.00304** | 0.00492 | **62 %** |
| unseen_both | 0.00192 | 0.00292 | 66 % |

Sign-test p is below 2.8e−02 at every alpha on every split for both keys. `dY_max` rises with alpha on the
hypothesis key (1.32–1.72 at alpha=0 to 3.53–5.95 at alpha=1) and sits at 3.96–6.54 throughout on the null
key. No cell is void.

## THE PRE-COMMITTED RULE, fixed before either run
From `model/results/RESULTS.md` §62, written while the first sweep was still executing:

> Estimand `atom_effect_median_per_row` plus a two-sided sign test; primary split `unseen_compound`.
> **If the null key shows a monotone trend in alpha on the primary split, the hypothesis key's curve is
> not read at all.** Monotone with a non-monotone null key → consistent with mechanism. Flat on
> [0.25, 1] with a jump at 0 → the binary contrast was an out-of-distribution artefact. Non-monotone →
> uninterpretable, and the operator is retired rather than re-specified.

§62.3 also required the endpoint span to exceed "the alpha=1 bootstrap CI width". §63.3 records that this
named the wrong statistic — the script emitted an interval for the difference of medians, not for the
estimand — and that the estimand's own interval, computable from the 2x2 per-row dump, is 3.5x tighter
(0.00487 against 0.01719 on the primary split). Against the estimand's own CI the primary split's span of
0.00492 exceeds it **by 0.00005**.

## WHAT WAS CONTROLLED
- All five alphas on identical rows in identical chunks, collated once per split before the alpha loop.
- One set of weights throughout; alpha changes only the attention matrix.
- One bootstrap resample matrix per split, shared by every alpha.
- `rows_sha` verified equal to the 2x2's on all three splits, for both keys.
- The script was **not modified between the hypothesis run and the gate run**, so both were computed by
  identical code. The method-rule-18 fix was deferred until after both finished.

## PRIOR RETRACTIONS IN SCOPE
- §60.9 item 2, retracted after your C1: the 2x2 interaction is not interpretable from that run.
- §47.5's parameter pair, retracted: it was `V9Config()` defaults at `l_control=2` with
  `use_gene_vectors` and `use_ppi` disabled — a shape never trained. Real confound +34.5 % on trainable
  parameters against a 9.12 M base.
- §60.3's diagnosis, corrected: the estimator gap was loss of pairing, not skew.
- §63.3, this packet: a pre-committed threshold naming the wrong statistic's interval.

## ASKS FOR PART 1
1. The gate. `unseen_compound`'s null key rises across four of five alphas and its span is 62 % of the
   hypothesis key's. §62.2 said "monotone trend", §62.3 said "strictly monotone", and the null key sits
   between those two wordings. Is this a fail? The PI has recorded it as a fail on the grounds that it
   fails under **both** readings — as a trend, and as a magnitude ratio of 1.6:1 — but the wording was
   ambiguous and the PI wrote it, so an independent reading is wanted.
2. Whether "retire the operator" is the right conclusion, or an overreaction. Two operators (binary
   diagonal, continuous alpha) and two estimands have now failed the same null key on the same
   architecture. Is there a version of attention masking that would be sound here, or is the whole family
   confounded by `S11−S10` being 5–7x the measured effect?
3. `unseen_cell` is the one split where the null key is comparatively quiet (21 %) and where the
   hypothesis curve is cleanest. It is also the split §57/§58.3 removed as primary for seed-instability.
   Is there **any** admissible way to read it, or is promoting it exactly the move a pre-commitment exists
   to forbid?
4. Anything in the operator verification you consider insufficient.

---

# PART 2 — DESIGN: chromatin gating the EDGES of a unioned biological graph

## OBJECTIVE
Chromatin as a gene-level gate is a measured null: §55, **+0.000360**, CI [−0.005433, +0.006153], 2 of 5
treated cells positive, sign test p = 1.000. This design asks a different question with the same data:
whether chromatin state predicts **which edges of a biological graph conduct in a given cell line**,
rather than how much a given gene moves.

## WHAT EXISTS NOW, read from the code rather than from notes
- STRING enters v9 as one message-passing step over a 978x978 adjacency, added residually
  (`model_v9.py:99,145`).
- Reactome enters **somewhere else entirely**: as `M_norm` in the auxiliary loss (`model_v9.py:180-186`).
- The two graphs have **never been combined**. GO:BP has been on disk unused.
- Chromatin (`E`, three tracks) gates **gene tokens** via EpiGate. It has never touched an edge.

## WHAT WAS BUILT, at zero cost
`network/outputs/v9/union_graph_v9.npz` (delegated W7, tests 7/7 run by the PI, test bodies read as well
as their output). 81,846 undirected edges over the 978 landmarks, each with a multi-hot
`[STRING, Reactome, GO]` provenance bit and the three sources' own **unnormalised** weights.

| source | raw | surviving | edges | density |
|---|---|---|---|---|
| STRING v12 | 13,001 edges | — | 13,001 | 2.72 % |
| Reactome | 800 terms | 784 | 59,860 | 12.53 % |
| GO:BP | 5,407 terms | 1,087 | 61,295 | 12.83 % |

Provenance counts: STRING only 3,465; Reactome only 14,922; GO only 17,902; STRING+Reactome 2,164;
STRING+GO 619; Reactome+GO **36,021**; all three 6,753. Jaccard: STRING∩Reactome 0.139, STRING∩GO 0.110,
**Reactome∩GO 0.546**.

Two facts about the artefact, both established by the PI after delivery:
- **The union is 17.13 % dense** (81,846 of 477,753 possible pairs).
- **The PI's stated reason for term-size filtering was factually wrong.** The W7 brief claimed large
  cliques drove the density. Reactome's landmark-restricted terms have **median 14 members and a maximum
  of 79**; **zero** exceed 100, so `max_term=200` removed nothing and the 16 dropped terms went for being
  too small. The largest single term contributes 0.64 % of all pairs. The density is the aggregate of 784
  medium terms and is **intrinsic to co-membership over a 978-gene panel** — `max_term=20` still leaves
  5.46 % and discards 194 terms.

## THE PROPOSED ARM
A per-cell edge gate. For edge (i,j) with provenance p_ij and source weights w_ij, and cell chromatin
E (978x3):
```
g_ij(cell) = sigma( MLP( [E_i, E_j, w_ij, p_ij] ) )        in (0,1]
```
used as a multiplicative gate on that edge's contribution to message passing / attention bias. The gate
is the only new consumer of `E` in this arm; EpiGate on gene tokens is left exactly as it is, so the arm
isolates **edge** gating from **gene** gating.

Reasoning the PI is putting up for attack rather than asserting: a 17 %-dense graph is close to uniform
smoothing, so it should be worth little as a static prior — which is a candidate explanation for why our
STRING prior measured null. If that is right, the thing that would make such a graph informative is a
per-cell mechanism selecting which of its edges conduct, and the sparsification **is** the hypothesis
rather than a preprocessing step. This also means the arm must **not** be "message passing over the raw
union", which would test the dense prior and confound it with the gating question.

Novelty position, stated for checking: §48.1, verified by the PI against arXiv:2505.14919, records that
TxPert's graph is **static and cell-independent** — *"each perturbation p is associated with a randomly
initialized input node embedding … learned via backpropagation"*. The PI has not found a published model
in which the biological graph is a function of the cell's chromatin state. That is a claim about a
literature search, not a proof.

## COSTS
One training run at fold 0, ~5.8 GPU-h by the measured rate (1817 s/epoch at batch 48, 12 epochs).
Graph construction is done and was free. Weekly quota 30 h; 6.23 spent.

## ASKS FOR PART 2
5. Is the edge-gate arm confounded in a way the PI has not named? In particular: the gate adds parameters
   and adds a second path for `E`, so what is the right capacity-matched null — the same gate with `E`
   mean-ablated, the same gate with `E` shuffled across cell lines, or both?
6. Given `Reactome∩GO = 0.546`, is a three-source union meaningfully different from "STRING plus one
   co-annotation graph", and does TxPert's monotone-improvement result (p < 0.027, per-graph numbers
   figure-only) transfer to sources this correlated?
7. What estimand and what null key should be **pre-committed before this run**, given that two probes have
   now failed their null key? The PI's instinct is: primary split `unseen_cell` (the regime the claim is
   about), estimand the median of the per-row contrast plus a sign test, null key the same architecture
   with `E` shuffled between cell lines — but the PI has now chosen wrongly twice and wants this specified
   independently.
8. Is 5.8 GPU-h on this a better buy than §50.5 (a drug-blind retrain-from-scratch, testing whether the
   whole drug branch earns its place, directly comparable to Bai et al.) or than §46.5 (training XPert and
   TranSiGen ourselves on the cold-cell splits for a head-to-head)? All three are ~5.8 h each and the PI
   can afford roughly four such runs this week.
9. Anything else, in either part.
