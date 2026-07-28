# Why our interpretability failed, and what the literature does instead

Written 2026-07-27 after two of our interpretability claims were falsified by our own tests. This is the
design input for a v6 architecture. Evidence for every "ours" number is in `results/CLAIMS.md`.

---

## 1. The two failures, stated plainly

| Mechanism we built | What we hoped | What we measured |
|---|---|---|
| Atom→gene cross-attention, aggregated per gene | attention would point at the drug's target | **median target rank percentile 0.560 over 149 gold pairs — worse than chance**; per-drug 0.611; and the "targets don't move" excuse is dead (0.645 for the most-responsive targets, corr = −0.045) [4.1a/4.1b] |
| Gene↔gene attention with an additive prior bias `λ·log1p(prior)` | signal would flow along pathway edges | λ is non-zero (0.24–0.94, so the model *uses* the prior) but **prior heads land on-support only 1.1× random** [4.6] |
| Cell-conditional pathway conductance | chromatin gating pathway flow | structural contribution ΔR² **−0.003 / +0.006**; 84.7 % of its variance is a per-cell scalar; corr(c, X_base) = 0.74 [3.1a/3.1b] |

**Common cause:** all three are *soft* mechanisms whose interpretability we hoped to read out **post hoc**.
None of them is architecturally obliged to be interpretable, and empirically none is.

## 2. What the successful biologically-informed models do differently

**P-NET** (Elmarakeby et al., *Nature* 2021), **DCell / DrugCell** (Ma et al., *Nat Methods*) and the wider
"visible neural network" family invert the relationship: **the biology IS the architecture.**

- Layers correspond to biological entities: `features → genes → pathways → biological processes → outcome`.
  P-NET uses 1 gene layer + 5 pathway layers over ~3,007 Reactome pathways.
- Connections between layers are **masked to real membership** — a gene connects only to pathways that
  actually contain it. Information is *forced* through pathway nodes; it cannot route around them.
- Interpretation is then **by construction**: node *k* in the pathway layer literally *is* "MAPK signalling",
  so you read its activation or gradient. There is nothing to hope about.
- The mask is also a strong **regulariser** — far fewer parameters than a dense layer, which matters when
  training examples are scarce (our regime: 83 cell lines).

| | Visible networks (P-NET/DrugCell) | Ours (v5) |
|---|---|---|
| Prior enters as | a hard connectivity **mask** | a soft additive **bias** on attention logits |
| Can signal bypass the prior? | **no** | yes — and it does (1.1× on-support) |
| Named biological units? | **yes** — every node | no — only 978 gene tokens |
| Interpretability | by construction | hoped-for, **falsified** |
| Parameter cost | low (sparse) | full dense attention |

## 3. Design implications for v6

1. **Add a masked pathway layer, don't just bias attention.**
   `H_gene [B,G,d] → (mask M[G,P]) → pathway activations A[B,P,d'] → back to genes`, with `M` from
   Reactome/co-pathway membership. Gives named, readable pathway units and a real bottleneck.
2. **Gate chromatin AT THE PATHWAY LEVEL.** Our chromatin signal is real (+0.089, sign-correct,
   confound-controlled) but the per-gene conductance mostly re-encoded `X_base` (corr 0.74). Chromatin
   modulating *pathway* nodes is both more interpretable and less redundant with baseline expression.
3. **Stop relying on attention weights for attribution.** Our own null is evidence for a broader point:
   attention ≠ explanation. If drug→gene attribution matters, use a method with an axiomatic basis
   (integrated gradients / DeepLIFT) **and** validate it against a curated reference — which is exactly
   the test most papers skip.
4. **Keep what is validated:** the signed additive chromatin head (the one component with a measured,
   confound-controlled effect), reliability weighting, and the differential target.
5. **Sparsity is a feature, not a cost.** With 83 cell lines, the masked layer's parameter reduction is
   likely to help generalisation — the exact axis (unseen cells) where we are weakest.

## 4. What this does NOT fix

- Chromatin's benefit still does not transfer to unseen cells [2.3]; a pathway layer is not obviously a
  remedy for that, and we should not assume it is.
- Cell-specificity is bounded by MSE-optimal dispersion under noise [6.8/6.10] — an architecture change
  does not raise the reliability ceiling of the data.
- None of this is worth building until it is testable: any v6 must be judged on the **reproducible**
  stratum with the same splits, and its pathway readout validated against an independent pathway
  annotation, not inspected by eye.
