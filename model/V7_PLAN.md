# What to change next, ranked by expected effect (2026-08-15)

Written after v6's first full evaluation. Every "ours" number below is measured, not estimated. The short
version: **the modern-DL toolbox is real but it is not our bottleneck**, and one change implied by our own
baseline result is worth more than all of the tuning combined.

---

## 1. Why the pathway layer contributes nothing — diagnosed, not guessed

The unseen-cell ablation, all components ablated **to their mean** on identical signatures:

| component | Δpearson | ΔR² | \|dY\|max | reading |
|---|---|---|---|---|
| baseline expression | **+0.0257** | **+0.0391** | 10.66 | the only structural contributor |
| chromatin (whole modality) | −0.0001 | +0.0001 | 2.13 | live, contributes nothing |
| **pathway layer** | **+0.0003** | −0.0000 | 0.69 | live, contributes nothing |
| pathway chromatin gate | +0.0001 | +0.0001 | 0.24 | live, contributes nothing |

`|dY|max ≫ 0` everywhere, so every one of these is a **true null, not a dead ablation**. Four causes, in
order of how much they explain:

1. **There is almost no cell-specific headroom to capture on this split.** `Meandrug` — predict that drug's
   training-set average, using *no* cell information at all — scores **0.4475**, and v6 scores **0.4466**.
   Any cell-context module is competing for a slice of variance that nothing has yet shown exists in a
   learnable form. Chromatin, pathway, pathway-gate and lineage are all cell-context modules. They are all
   null *together*, which is the signature of a missing-signal problem, not four independent design faults.
2. **It is redundant with `X_base`.** The pathway layer reads gene tokens that were built from baseline
   expression and chromatin, and re-expresses them as a 360-dim masked average. Baseline is already in the
   residual stream and carries per-gene detail the averaging destroys. This is v5's `corr(conductance,
   X_base) = 0.74` finding [3.1a] reappearing in v6 *despite* the late-fusion fix.
3. **It is a bypassable side branch** [4.13]. `h = h + delta`, so gradient descent is free to ignore it —
   and there is a literature name for exactly this: residual **"network loafing"**, where residual branches
   under-contribute because the skip path is easier ([Stimulative Training](https://arxiv.org/pdf/2210.04153)).
4. **It sits before the drug enters**, so it can only ever encode cell context [4.12] — and cell context is
   what (1) says has no headroom here.

**So the pathway layer is not broken. It is placed where there is nothing left to win.**

---

## 2. The performance bar you asked for

Cross-paper numbers are not comparable and we already proved it: published SOTA PCCs of 0.743/0.870 fall
**inside our own convention-sweep curve**, where our *unchanged* predictions score 0.68–0.81 depending only
on how much baseline expression is mixed into the target (RESULTS §17). Quoting someone's 0.87 against our
0.45 is meaningless. Three bars that *are* meaningful:

| bar | value | where we are |
|---|---|---|
| **Noise ceiling** (replicate r 0.509–0.619 ⇒ max achievable corr ≈ √r) | **0.71 – 0.79** | 0.4466 = **57–63 %** of it |
| **Meandrug** (no cell information at all) | **0.4475** | 0.4466 — **we are at 100 % of it, and no more** |
| **Ridge on the same global inputs** | 0.4470 | tied |

Read together these say something sharp: **the drug-mean alone already reaches ~63 % of the achievable
ceiling, and everything we have built adds nothing on top.** The remaining 0.27–0.34 of correlation is
*entirely* the drug×cell interaction. That is the whole problem, and it is not a tuning problem.

The only way to get a real cross-paper number is to run a published model on **our** split — XPert's code is
public. That remains open item #2 and no amount of internal tuning substitutes for it.

---

## 3. Tier 1 — changes aimed at the actual bottleneck

### 3.1 Predict the residual over the drug mean ← highest expected value, ~1 line
If `Meandrug` gets 0.4475 and we get 0.4466, the network is spending its capacity re-deriving the drug mean.
Train it on what is left instead:

```
target  Y' = Y − mean_over_training_cells(Y | drug)      (fall back to the global mean for unseen drugs)
predict Ŷ  = Meandrug(drug) + f(cell, drug, dose, time)
```

Every parameter then goes to the **interaction**, which is the only thing left to learn, and the model
cannot score below the baseline by construction. This is ordinary boosting/residual-fitting, it costs
almost nothing, and it follows directly from our own measurement. **Do this first.**

### 3.2 Supervise the pathway layer and chromatin — your idea, and it is the right one
Right now nothing *asks* the pathway nodes to mean anything, so they are free to be ignored. Deep
supervision fixes precisely this: attach auxiliary heads to intermediate layers so they receive their own
gradient signal ([overview](https://www.emergentmind.com/topics/deep-supervision)).

We can build the targets from data we already have, and they are not circular:

| head | target (measured, never a model input) | teaches |
|---|---|---|
| pathway → response | mean\|Y\| over pathway *p*'s member genes, for this signature | "pathway *p* is responding here" |
| chromatin → responsiveness | per-gene mean\|Y\| in this cell, from chromatin alone | "chromatin says which genes *can* move" |

Total loss `L = L_main + λ_p·L_pathway + λ_e·L_epi`, with λ set by **uncertainty weighting**
([Kendall et al.](https://arxiv.org/abs/1705.07115); [UW-SO, IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02625-x))
rather than hand-tuned. This is the direct answer to "the interpretability is getting wasted": a supervised
pathway node is interpretable **and** load-bearing, and its readout becomes validatable rather than hoped-for.

### 3.3 Stop the branch being bypassable
Two options, cheap, and testable against each other:
- **Stochastic depth on the skip path** so the model cannot rely on the bypass being there — the
  established remedy for residual "loafing".
- **Make it a real bottleneck** for one ablation arm: route the gene tokens *through* the pathway layer
  instead of adding it, which is what P-NET actually does, and what would let us claim what our own docs
  wrongly claimed [4.13].

---

## 4. Tier 2 — the modern recipe. Real, but small: expect a few points, not a transformation

Adopt these because they are free and standard, not because they will close the gap.

| change | from | to | why it fits *us* |
|---|---|---|---|
| **EMA of weights** | none | EMA (decay ~0.999) | documented gains in **robustness to noisy labels** ([arXiv 2411.18704](https://arxiv.org/abs/2411.18704)) — LINCS is ~75 % inert noise, so this is unusually well matched. Best value-for-effort in this table |
| **schedule** | OneCycle | **WSD** (warmup → stable → decay) | now standard in large-scale recipes ([SOAP/Muon study](https://arxiv.org/abs/2607.20548)); also allows stopping early without wrecking the LR curve |
| **optimizer** | AdamW | **Muon** or **SOAP** | both consistently beat AdamW at scale in controlled comparisons ([ibid](https://arxiv.org/html/2607.20548v1)); gains shrink at our size, so treat as a cheap experiment, not a fix |
| **norm** | LayerNorm | **RMSNorm** | ~10 % faster, equal quality |
| **FFN** | GELU MLP | **SwiGLU** | gated FFNs consistently beat plain ones at matched FLOPs |
| **attention** | plain | **QK-norm** | bounds attention logits; better stability and convergence |
| **regularisation** | dropout 0.1 | + stochastic depth | doubles as the §3.3 fix |

**Honesty check:** none of these addresses §2. They are worth doing after Tier 1, and they are worth doing
*with seeds* (M.3) — at our current single-seed resolution a +0.005 gain is indistinguishable from noise.

---

## 5. Mamba: no, and the reason is structural

State-space models are genuinely good in transcriptomics — [GeneMamba](https://arxiv.org/abs/2504.16956),
[SC-MAMBA2](https://www.biorxiv.org/content/10.1101/2024.09.30.615775v1.full) — but look at *why* they win:
linear-time scaling over **tens of thousands of genes** and tens of millions of cells. Our situation:

- **978 gene tokens.** Attention is 978² ≈ 1M — trivial. Sequence length is not our cost.
- **We are not compute-bound.** 10 epochs = 2.58 h on T4×2, well inside budget. There is no training-time
  problem for Mamba to solve.
- **Genes are a set, not a sequence.** SSMs are inherently order-dependent; our gene axis has no meaningful
  order. Bi-Mamba mitigates but does not remove this. Attention is permutation-equivariant, which is the
  correct inductive bias for an unordered set of genes — and it is what lets the Reactome mask mean anything.

Mamba would trade a correct inductive bias for a speedup we do not need. **Recommend against.**

---

## 6. Loss function

We already tried a correlation loss and it changed nothing [6.x] — consistent with sitting at MSE-optimal
dispersion under noise [6.8/6.10]. So do not expect a loss swap to move the headline. What is worth doing:
- **uncertainty-weighted multi-task loss** for the auxiliary heads in §3.2 (this is loss engineering that
  serves a purpose, rather than swapping Huber for something else)
- keep reliability weighting — it is well-founded here and most of the field lacks it

---

## 7. Order of work

| # | change | cost | expected |
|---|---|---|---|
| 1 | **Residual-over-Meandrug target** (§3.1) | ~1 line + retrain | the only change targeting the real gap |
| 2 | **Auxiliary supervision of pathway + chromatin** (§3.2) | ~1 day + retrain | makes the priors load-bearing; rescues interpretability |
| 3 | **Seeds / k-fold** (M.3) | GPU | *required* before believing any of 1–2 |
| 4 | EMA + WSD + RMSNorm/SwiGLU/QK-norm (§4) | ~half a day | a few points, free |
| 5 | Stochastic depth / true bottleneck arm (§3.3) | small | tests [4.13] directly |
| 6 | XPert on our split (open item #2) | GPU | the only honest cross-paper bar |
| — | ~~Mamba~~ | — | declined, §5 |

**The discipline that must survive all of this:** measure against `Meandrug` and the ridge, on the
reproducible stratum, with all strata reported, ablating to the mean — and report the nulls as loudly as
this document reports them.
