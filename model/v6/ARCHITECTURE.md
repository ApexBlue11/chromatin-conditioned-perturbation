# v6 — architecture, data provenance, and the evidence behind every design decision

**Self-contained.** Everything needed to understand what this model is, where its data comes from, how it
flows, and *why each piece is shaped the way it is* — is in this directory. Numbers cited as `[x.y]` are
claim ids in `../results/CLAIMS.md`; the measurement log is `../results/RESULTS.md`.

**v6 exists because v5's interpretability was falsified by our own tests.** This is not a tuning pass; two
structural mistakes were identified and corrected. Both corrections come from the published architectures
that make biological priors work (P-NET, DCell/DrugCell) and multi-omics integration (MOLI, DeepCDR).

---

## 1. The task

Predict the **differential transcriptional response** of a human cell line to a small molecule:
`Y ∈ ℝ^978` — the LINCS L1000 Level 5 MODZ z-score over the 978 landmark genes, for a
(cell line, compound, dose, time) condition.

We predict the **differential**, not absolute post-perturbation expression. Delta-style metrics are the
field standard, and absolute-convention correlations are inflated by the supplied baseline
(*control bias* / *signal dilution*) — established prior work, not our finding [6.6h].

## 2. Where every input comes from

| Tensor | Shape | Source | Built by |
|---|---|---|---|
| `Y` target | 312,438 × 978 | LINCS L1000 Level 5, GSE92742 (phase 1) + GSE70138 (phase 2) | `phase2_assembly/scripts/step6_extract_level5_target.py` |
| `sig_strength` | 312,438 | mean\|Y\| per signature | same |
| `X_base` baseline expression | 83 × 978 | CCLE / DepMap `OmicsExpressionTPMLogp1` | `baseline/scripts/step2_load_ccle.py` |
| `E` chromatin | 83 × 978 × 3 | ENCODE + ChIP-Atlas narrowPeak, ±10 kb TSS | `phase2_assembly/scripts/step10..step15` |
| `E_mask`, `E_reliability` | 83 × 978 × 3, 83 × 3 | per-(cell,mark) availability + ChIP quality | `step15_rebuild_reliability.py` |
| **`M_reactome`** | **360 × 978** | **Reactome pathway membership** (min size 10, root umbrellas excluded) | `network/` pipeline |
| `STRING_adj` | 978 × 978 | STRING v12, score ≥ 400 | `network/` pipeline |
| drug descriptors / ECFP4 | 21,220 × 20 / × 2048 | RDKit from canonical SMILES | `drug/scripts/step2_compute_descriptors.py` |
| Uni-Mol CLS / **atom tokens** | 21,220 × 512 / 703,851 × 512 | Uni-Mol v1, `remove_hs=True` | Kaggle GPU + `step8_integrate_atom_tokens.py` |
| `cell_lineage` | 83 × 16 | DepMap `OncotreeLineage` one-hot (col 0 = UNKNOWN) | `baseline/outputs/cellfeat/` |
| `scaffold_split` | 5 folds / 6,035 scaffolds | Bemis-Murcko + Tanimoto leakage audit | `drug/scripts/build_scaffold_split.py` |
| `dti_reference` | 19,174 edges | ChEMBL + STITCH v5, evidence-tiered | `drug/scripts/step4..7` — **validation only, never in the loss** |

Nothing is redistributed; all sources are public. Full run order: `../../METHODOLOGY.md` §9.

### The data fact that governs all evaluation
**~75 % of LINCS perturbations are biologically inert.** Replicate agreement is r = 0.127 over *all*
signatures but **0.509 / 0.619** for real responses (mean\|Y\| ≥ 1) [6.1]. Evaluating on the full set once
**inverted the sign** of the chromatin effect [6.2]. Every metric here is on the **reproducible stratum**.

## 3. What v5 got wrong (both fixed here)

### Mistake 1 — a soft prior instead of a hard mask
v5 biased gene↔gene attention with `λ·log1p(A_copathway)`. Measured: λ is non-zero, so the model *uses*
the prior, but prior heads land on-support only **1.1× random** [4.6] — signal routes around it. And
`A_copathway` [978,978] is the gene-gene **co-membership collapse** of `M_reactome` [360,978] — which we
had all along. Collapsing it **discards the named pathway axis**.

> **P-NET (Nature 2021), DCell/DrugCell (Nat Methods)** invert this: the prior is a hard **connectivity
> mask**, layers *are* biological entities (genes → pathways → processes), information is forced through
> named nodes. Interpretation is by construction, not by hoping attention means something — and the mask
> is strong regularisation, which matters with only **83 cell lines**.

### Mistake 2 — early fusion of modalities
v5 summed projected `X_base` and `E` straight into one gene token. Consequence: chromatin entangled with
baseline — the learned pathway conductance correlated **0.74 with X_base** and contributed ≈0 once its
scale was controlled [3.1a/3.1b].

> **MOLI (Bioinformatics 2019), DeepCDR** use **late integration**: a type-specific encoder per omic, then
> fuse. This respects each modality's distribution, allows per-omic normalisation, and — critically for us
> — makes each modality **cleanly ablatable**.

### Also corrected
- **Attention is not treated as explanation.** Our own null: atom→gene attribution puts known targets at
  median rank percentile **0.560 — worse than chance** over 149 gold pairs, and the "target doesn't move"
  confound was tested and rejected (corr = −0.045) [4.1a/4.1b]. v6 keeps cross-attention because the atom
  tokens are the largest *predictive* feature (Δpearson +0.163 [4.3]) — but makes **no interpretability
  claim** for it.
- **ChemBERTa removed** (ΔR² = +0.001, dead weight [5.2]).
- **Pathway conductance removed** in its v5 per-gene form; chromatin now gates at the **pathway** level.

## 4. v6 flow

```
  X_base [B,G]          E [B,G,3] + avail            atoms [B,M,512] + u_feats [B,2580]
      │                      │                                  │
  BaselineEncoder      ChromatinEncoder                  drug tokens
  (own LayerNorm)      (own LayerNorm, reliability-masked)       │
      └────────── ModalityFusion (LATE) ──────────┘              │
                        │                                        │
              gene tokens [B,G,d] + gene_emb                     │
                        │ + FiLM(dose, time, lineage)            │
                        ▼                                        │
        ┌──────── base encoder: gene↔gene ────────┐              │
        │                                          │             │
        ▼                                          │             │
  PathwayBottleneck  genes →(mask M)→ 360 NAMED    │             │
  Reactome nodes →(maskᵀ)→ member genes only       │             │
  chromatin gates AT PATHWAY LEVEL                 │             │
        │  └── pathway_activations [B,360,dp] ─────┼── READOUT   │
        ▼                                          │             │
      perturbation encoder: atom→gene cross-attn ◄─┴─────────────┘
                        │
              per-gene head + signed chromatin term
                        ▼
                    Ŷ [B,978]
```

## 5. Component rationale — each with its evidence

| Component | Why it is here |
|---|---|
| **PathwayBottleneck** | The fix for mistake 1. Hard mask ⇒ gene *g* reaches only pathways containing *g*; `pathway_activations[:,p]` **is** Reactome pathway *p*. 33.3k params, 2.28 % density. Verified: no-op at init, genes in no pathway get exactly 0 (243 genes), perturbing a gene changes only its 91 pathways |
| **Separate modality encoders + late fusion** | The fix for mistake 2. Each of baseline/chromatin keeps its own normalisation and stays independently ablatable |
| **Chromatin gating at pathway level** | v5 gated per gene and mostly re-encoded `X_base` (corr 0.74). Pathway-level gating is less redundant with baseline and directly interpretable as "which pathways are chromatin-permitted in this cell" |
| **Signed additive chromatin head** | **Kept — the one validated mechanism.** ΔR² +0.089 [2.1], correct sign [2.2], survives baseline-expression stratification in all 4 quartiles [2.2a]. A multiplicative gate alone cannot express a signed shift |
| **Atom tokens + ECFP4** | The two predictive pillars (Δpearson +0.163 / +0.147 [5.1]). Kept for accuracy, **not** for interpretability |
| **Reliability weighting** | LINCS is ~75 % inert; weighting by measured replicate reliability stops the model hedging to the drug-average [6.1] |
| **Dose/time FiLM + lineage** | Nonlinear (inverted-U-capable) conditioning; lineage is marginal [5.4] but free |
| **Differential target** | Field-standard delta metric; avoids control bias [6.6h] |

## 6. What v6 is expected NOT to fix
Honest, so we do not repeat the pattern of hoping:
- **Chromatin's benefit does not transfer to unseen cells** (+0.089 in-dist → ≈0 unseen cell [2.3]). A
  pathway layer is not an obvious remedy and should not be assumed to be one.
- **Cell-specificity is bounded by noise**, not architecture: the model already sits at MSE-optimal
  dispersion (std-ratio 0.47 ≈ corr 0.42) [6.8/6.10].
- **A pathway bottleneck may contribute nothing**, exactly as the conductance did. It must be judged by
  ablation-to-**mean** (never to 0/1 — that measures scale destruction, a 30× artefact last time [3.6]).

## 7. What the pathway readout is — and is NOT (verified 2026-07-30)

The readout comes off step (4), **before the drug enters at step (5)**. Measured on a live model by
substituting inputs one at a time, the change in `pathway_activations` is:

| swapped input | change in the readout |
|---|---|
| **a completely different drug** | **exactly 0.0** |
| baseline expression `X_base` | 2.54 |
| chromatin `E` | 1.28 |
| dose / time | 1.59 |

So `pathway_activations[:, p]` is a **cell × dose/time** quantity — *"is named Reactome pathway p
chromatin-permitted in this cell at this exposure"* — and carries **no drug information at all**.

Two consequences, stated plainly because the natural reading is the wrong one:
- **v6 does not repair the falsified drug-target claim [4.1a/4.1c].** That was a *drug*-level claim about
  atom→gene attention. This is a *cell*-level readout. v6 sidesteps that failure; it does not fix it.
- **Scoring this readout against drug→target annotation (`dti_reference.tsv`) is structurally guaranteed to
  return a null that means nothing.** Do not run it and do not report it.

The valid test uses the **measured LINCS response** as the independent annotation — it is never an input to
the readout: does pathway *p*'s activation in cell *c* track how much *p*'s member genes actually move in
*c* (reproducible signatures only)? With a gene-permutation null (preserves pathway size), a **cell-shuffle
null** (is it cell-specific, or a global pathway prior?), and train-vs-test-cell stratification.
Implemented in `probe_pathways_v6.py` [4.12].

## 8. How v6 must be judged
1. Reproducible stratum only, same three splits (unseen cell / unseen compound / unseen both) —
   `eval_v6.py`, which refuses to report a v5 comparison if the split sizes do not match v5's.
2. Every component ablated **to its mean**, on identical signatures — never to 0 or 1 [3.6].
3. Pathway readout validated against **independent** annotation, per §7 — not inspected by eye.
4. Unit tests (`test_v6.py`) pass before any accelerator time is spent. They include the positive **and**
   negative control for the ablation mechanism itself: at init the pathway layer, both chromatin gates and
   the FiLM are exact no-ops by design, so on an untrained model a broken ablation is indistinguishable
   from a genuine null.
5. Report negatives with the same prominence as positives.
