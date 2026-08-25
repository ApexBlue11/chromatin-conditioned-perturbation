# v8 — rebuild on Level 3, with priors done properly and interpretability kept

Decision taken 2026-08-18. This supersedes the v7 direction. Everything below is grounded in a measurement
already in `results/RESULTS.md`; where a claim is an expectation rather than a measurement it says so.

---

## 1. Why Level 3 — the right reason, which is NOT the one we first gave

§25 measured it: a replicate-averaged **Level-3 delta is LESS replicable than the Level-5 MODZ z-score**
(0.144 / 0.243 top-quartile vs 0.127 / 0.509–0.619 reproducible). So Level 3 is **not** a cleaner target,
and §24's argument for migrating was wrong and is retracted.

**The real advantage is on the INPUT side, and it is large:**

| | our baseline input today | what Level 3 gives |
|---|---|---|
| source | **CCLE** `X_base` | the DMSO wells **on the same plate** |
| assay | RNA-seq, different platform | **the same L1000 plate, same batch** |
| timing | a different experiment, years apart | **simultaneous with the treated well** |
| batch effects | uncontrolled | **cancelled by construction** |

Every SOTA LINCS model is handed a **plate-matched control**; we hand ours a proxy from a different
experiment. That is a categorical information advantage unrelated to noise, and it is the most plausible
remaining explanation for the gap in §25 once the HDACi-subset confound is accounted for.

**The cost, stated plainly:** the Level-3 delta target is noisier than MODZ. **Mitigation** — build the
target as a **MODZ-style correlation-weighted replicate average of Level-3 deltas** rather than a plain
mean. That keeps the matched control while recovering most of MODZ's denoising. Whether it does is
**measurable before any training**: rerun `level3/replicate_reliability.py` against the weighted aggregate
and require it to beat the plain mean (0.144 / 0.243). **Gate the whole migration on that number.**

---

## 2. "Proper" priors — what the other models actually do

Read from XPert's released `HG_data/`:

| | XPert | ours (v7) |
|---|---|---|
| PPI | **901,260** weighted edges over **19,392 genes** | 12,665 edges, restricted to the 978 landmarks |
| drug→target | 12,890 edges, **in the model** | validation only, never in the loss |
| drug↔drug similarity | 287,834 weighted edges | none |
| how consumed | heterogeneous graph **pretrained by link prediction**, embeddings become node features | raw adjacency, one graph-conv step |

**The defect is ours and it is specific: we restricted the graph to the 978 landmarks.** That deletes every
path routing *through* a non-landmark gene — and it is exactly why 231 of our landmarks sit in no Reactome
pathway at all and 66 have no STRING edge. The prior was never given a chance to connect them.

**v8 fix.** Build the graph over the **full gene set (~19k)**, pretrain node embeddings by link prediction,
then read off the 978 landmark rows as gene features. Orphan landmarks become reachable *through* the graph.
Add DTI and drug–drug edges so the graph is heterogeneous. This is a strictly better use of the same priors
we already have — STRING v12 and Reactome are both full-proteome sources that we truncated ourselves.

---

## 3. Epigenetics — there is no template to copy, and that is the point

**No SOTA LINCS model uses chromatin at all.** So "make it proper like the others" has no referent —
this is the part that is genuinely ours, and it has to be made to *work* rather than made to conform.

What v6/v7 measured: chromatin contributes ≈0 on unseen cells, +0.0061 on unseen compound, and a 16-dim
lineage one-hot beats it. What v8 changes:

- **Chromatin becomes node features on the gene graph**, not a separate encoder bolted alongside baseline
  expression. Gene *g* in cell *c* carries its accessibility / activation / repression state as attributes
  of the graph node. That is how the graph consumes every other prior, and it lets chromatin propagate
  along PPI edges instead of sitting in a parallel branch the model can ignore (which it did — v6 measured
  the branch live but worth +0.0003).
- **Keep the signed additive chromatin head.** It is the one chromatin mechanism that ever survived a test.

---

## 4. Interpretability must not dissolve into the latent space

Non-negotiable, and it constrains the design:

1. **Named pathway readout stays.** Reactome nodes remain explicit and named; `pathway_info.tsv` row order
   stays verified at load. What changes is that pathway nodes are now reachable *through* the graph.
2. **No purely latent bottleneck.** Whatever the graph learns must terminate in units that have names —
   genes, pathways, drug–target edges.
3. **Every readout ships with its null.** The lesson of [4.15]: on pathway-level target alignment, 0.5 is
   *not* chance — an untrained model scores 0.218 against a permutation null of 0.229. Any new readout gets
   a permutation null and a drug-invariance check before it is reported.
4. **Ablate to the mean, report `|dY|max`.** Non-negotiable [3.6].

---

## 5. Comparison against SOTA — the part that makes any of this legible

| model | code | status |
|---|---|---|
| **XPert** | Zenodo, downloaded | weights + data in `external/xpert/`; both conventions already computed §23 |
| **PRnet** | public | to obtain |
| **chemCPA** | public | to obtain |
| **TranSiGen** | public | to obtain |
| Meandrug / ridge | ours | already the honest floor [1.8/1.9] |

Rules, learned the hard way:
- **Same data level, same convention, same split, or the numbers do not compare** (§24/§25).
- **≥3 seeds or no difference is reported** — seed sd reaches 0.0232, larger than every architecture effect
  this project has ever claimed [M.10].
- Report **gain over the trivial baseline in each frame**, not raw headline correlations (§23).

---

## 6. Order of work, with the gate stated up front

| # | step | gate |
|---|---|---|
| 1 | Level-3 extraction: plate-matched controls + MODZ-style weighted replicate aggregation | — |
| 2 | **Measure the aggregated target's reliability** | **must beat 0.144 / 0.243, else stop and reconsider** |
| 3 | Full-proteome heterogeneous graph + link-prediction pretraining | orphan landmarks drop from 231 → ~0 |
| 4 | v8 model: graph node features, chromatin as node attributes, named pathway readout retained | 26/26-style design tests pass first |
| 5 | Train ×3 seeds | report mean ± range only |
| 6 | Head-to-head vs XPert/PRnet/chemCPA on identical conditions and splits | both conventions, gain over trivial |

**Step 2 is a real gate, not a formality.** If the weighted Level-3 aggregate cannot beat the plain mean, we
are trading a measurably better target for a measurably better input, and that trade needs its own decision
rather than an assumption.
