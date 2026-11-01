# IDEAS — the tracked backlog

Every idea this project has taken from the literature or from its own measurements, with where it came
from, whether the source claim was **verified by the PI** or only **reported by a worker**, and what it
would cost. Nothing is deleted from this file; items move to `status: DONE` with a pointer to the RESULTS
section that settled them, or `status: KILLED` with the measurement that killed them.

**Why this file exists.** Ideas were accumulating inside numbered RESULTS sections written for other
purposes and becoming invisible. Asked directly whether the combined STRING+Reactome graph had ever been
tried, the answer was buried in §48.3 item 1 and the answer was *no*.

Direction set by the principal, 2026-09-21: **pursue the mechanism/MoA line (A-group) while keeping the
load-bearing dissection (B-group) and the cold-cell head-to-head (C-group) alive.** Do not drop B and C.

**Direction set by the principal, 2026-11-01: the paper is framed around epigenetics, and section E (a cell representation
first, chosen by fast tests, then the drug model) is the current programme.** The goal is unchanged: mechanistic
interpretation through pathways, and the MoA of drugs the model has never seen.

---

## E. A cell representation first, then the drug — the principal's direction of 2026-11-01

**The direction, as the principal gave it:**
- **The paper is framed around epigenetics.**
- **Epigenetics is not an isolated feature.** It is part of a combined representation of the cell, together with pathways and
  baseline gene expression.
- **The full STRING network over all genes,** not only the 978 landmarks. Add a radius (k-hop) or another method, so the
  model learns how each landmark gene is connected through non-landmark genes.
- **Non-landmark genes have no L1000 baseline, and many cells have no RNA-seq, but they have epigenetics.** Chromatin can fill
  that gap.
- **Non-coding RNAs** (lncRNAs, miRNAs and other non-coding elements, none of them measured by L1000) may interact with drugs.
  Where they are part of Reactome or KEGG pathways, combine them with STRING, with epigenetics on them too.
- **These interactions are the model's picture of the drug's mechanism of action.**
- **Known knowledge seeds the graph:**
  - a drug's targets as the place the signal starts ("start from here"), or known endpoints where it ends;
  - knowledge gathered from the literature as well.
- **Train with that knowledge on some drugs, then test on drugs the model has never seen.** Read what it attributes to each
  molecule token: which pathways, genes or regulatory elements it reacts with.
- **The method:**
  - a full rerun per strategy is too slow;
  - how to construct the full cell representation is an open question;
  - so learn how others build similar representations, add novel ideas, compare them on a fast test, and put only the best
    candidates into the full model.
- **The fast test's target should check whether the model understands the cell.** Once the best representation is found, the
  target switches to drug-perturbation prediction.
- **Not attention everywhere:** CNNs, RNNs and other network types each have their own advantages. Combine them, weave them
  together, and create a new architecture.

**What the record already tested** (so it is not re-proposed, and so what is new is clear):

| idea | tested? | where | result |
|---|---|---|---|
| full STRING over all genes, with a radius | yes: v11's FG module, 19,496 genes, 345,962 experimental / database edges, K = 2 propagation rounds (a 2-hop radius) | §102 | R1 no claim (+0.0036 against 0.0042) |
| chromatin on every gene (the non-landmark gap) | yes, as an input: v11's node features and edge gate, 217 cells × 18,658 genes with a TSS | §102 | R2-chromatin no claim (+0.0024 against 0.0039). At inference, swapping chromatin between cells costs 0.0003–0.0004 and shuffling it across genes 0.0005–0.0008 |
| baseline expression for every gene | yes where CCLE exists: 131 LINCS cells (not HUVEC, NPC, BJAB, CD34) | §102 | R2-CCLE no claim (+0.0014) |
| a combined cell representation (graph + chromatin + expression + pathways) | yes, but only as inputs to a drug-response model (v11's FG on v9's pathway layer) | §102 | as above |
| more assays and coverage | yes: ChIP-Atlas re-search (90 cells with any axis, 43 with all three; 40 / 16 before), property-matched substitutes, CCLE RRBS methylation as a fourth axis, bigWig signal in 5 windows | §100 | T-A, T-B, T-C no claim |
| enhancers, TF-regulon accessibility | yes | §93.13, §101 | no gain over promoter accessibility. H3K27ac's BET signal is explained by expression |
| chromatin-gated edges | yes: C7 (landmark graph), v10-C, v11 | §89, §100, §102 | not accepted (+0.0020) / no claim |
| targets as the start of the signal | yes: zero-parameter diffusion (landmark graph, then the full graph), then trained in v11 (known or predicted signed targets, propagated 2 hops) | §82, §83, §102 | no signal / no claim |
| fast screens before GPU runs | yes, for chromatin: the §91 funnel | §91 | nothing advanced, but its target was the drug response, not the cell |
| train on some drugs' targets, test on unseen ones | yes: v11's DTI probe split | §102.3 | probe running 2026-11-01 |
| a pathway readout as MoA | yes: v9's pathway layer | §86.4, §88.8 | null against calibrated untrained models |
| non-coding RNAs | **no** | — | — |
| KEGG | **no** (Reactome and GO-BP only) | — | — |
| literature-derived knowledge | **no** (curated databases only: ChEMBL, STITCH, STRING, Reactome, GO) | — | — |
| known endpoints ("ends here") | **no** | — | — |
| a target that tests whether the model understands the cell | **no**: every test so far scored drug response | — | — |
| the cell representation trained and evaluated on its own | **no** | — | — |
| non-attention architectures (a CNN over chromatin profiles, recurrent / ODE propagation, a visible pathway network, a molecule GNN) | **no**: transformers plus 2 rounds of linear propagation | — | — |

**Why this is not a retry.** Every chromatin test so far asked one question: does chromatin improve a drug-response model's
predictions? That target's cell-specific drug signal is small next to its noise (§91.10; T-C bounds the gain at ≤ +0.0014).
The new programme first asks whether a representation understands the cell, on targets where cells differ a lot and the
truth is well measured. Only then does it ask whether that understanding transfers to drugs.

### E1. The fast tests — proposed, to be registered before any is run
**The interface:**
- a candidate builds a per-(cell, gene) embedding;
- the same small task heads score every candidate;
- every score is cold-cell (leave-cell-out) and, where meaningful, cold-gene.

**The proxy targets:**
- **T1, state:** impute a gene's baseline expression (CCLE RNA-seq, all genes, non-landmark included) in a held-out cell, from
  chromatin, the network and the other genes. This measures the gap-filling idea directly.
- **T2, wiring (the primary candidate): genetic perturbations in L1000.**
  - **The data:** 36,720 consensus shRNA knockdown signatures (`trt_sh.cgs`; 4,345 genes, 17 cells); 22,205 overexpression
    (`trt_oe`; 3,114 genes, 10 cells); 8,256 ligand signatures (`trt_lig`, 14 cells); GSE92742 `sig_info`.
  - **Why it fits:**
    - the perturbed gene is a known start node;
    - the readout is the same 978 landmarks as the drug task;
    - cold-gene ↔ cold-drug and leave-cell-out ↔ unseen cell, so it is the drug task with the targets given.
  - **Chromatin coverage:** 7 of the 9 core knockdown cells have chromatin and CCLE. PC3, MCF7, A375, VCAP, A549 and HEPG2 have
    all three axes; HT29 has two. HA1E and HCC515 have none.
- **T3, dependency:** DepMap CRISPR gene effect (which genes a cell needs to survive). H3K27ac super-enhancers are reported to
  mark lineage dependencies (to verify in the literature scan).
- **The control every candidate must beat:** the same heads on an expression-only representation. That is the control that
  explained T-E's pass (§101.6).
- **Data still to fetch, each needing the principal's approval:**
  - the Level 5 GCTX was deleted on 2026-09-20. `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` is 23.4 GB, so
    the plan is to extract its landmark rows for `trt_sh.cgs` / `trt_oe` / `trt_lig` inside a Kaggle kernel;
  - DepMap gene effect;
  - ncRNA interaction tables.
  - (LINCS 2020 reportedly holds many more CRISPR signatures; to verify.)

### E2. Candidate representations (menu)
- **expression only** (the control);
- **+ chromatin** (3 axes + methylation);
- **a full-STRING GNN** (relational, k-hop);
- **+ Reactome / KEGG hierarchy** as a visible network (DrugCell / P-NET style);
- **+ non-coding nodes** (lncRNA / miRNA):
  - their activity from chromatin at their TSS;
  - edges from ncRNA–target databases (miRTarBase, ENCORI / starBase, LncTarD, NPInter; licences to check);
- **pretrained gene embeddings from others:**
  - Geneformer / scGPT gene tokens;
  - GenePT text embeddings (a cheap route to literature knowledge);
  - ESM protein embeddings;
- **CellOracle-style chromatin-derived gene-regulatory networks** (ATAC peaks + TF motifs);
- **a 1D CNN over each gene's chromatin profile;**
- **combinations of these.**

Literature to read first (names, not verified claims): GEARS, CellOracle, SCENIC+, DrugCell, P-NET, TxPert (§48.1),
Geneformer, scGPT, GenePT, Enformer / Borzoi, Chemprop, gated graph networks, graph neural diffusion.

### E3. Architecture menu (not attention everywhere)
- a **1D CNN** over per-gene chromatin or sequence windows: local regulatory shapes, the Basenji / Enformer inductive bias;
- a **relational GNN** (R-GCN / GraphSAGE) over the heterogeneous gene–ncRNA–pathway graph;
- **recurrent or neural-ODE propagation** on the graph:
  - the drug perturbs its target nodes at t = 0;
  - a gated recurrent cell or an ODE carries the signal step by step, read at 6 h / 24 h;
  - the trajectory is the model's picture of the mechanism;
- a **visible neural network**: named pathway units are the only route to the output, so it is interpretable by construction;
- a **message-passing molecule encoder** (D-MPNN / Chemprop) instead of atom tokens (atom tokens hurt, §57 / §60);
- a **mixture of experts** gated by the drug's predicted targets or class;
- **hypernetworks**: the drug generates the cell graph's edge gates;
- **deep ensembles or a GP head** for calibrated confidence (the interpretability objective asks for confidence).

### E4. The MoA readout — the goal
- **The question:** on drugs whose targets were held out, does the route the model uses (target nodes → pathway units →
  landmark genes) match the annotation?
- **The nulls:** untrained models and a cell-agnostic prior (the methods lessons of §86 / §88).
- **Registered before it is read.**

**Status:** proposed 2026-11-01; nothing registered or run. **Next:**
1. a literature scan of how others build cell representations;
2. the T1–T3 harness and its registration;
3. the download approvals.

---

## A. Make chromatin and the drug atoms earn their place — the mechanism line

### A1. 🔵 Chromatin-modulated edges on a unioned biological graph — zero-parameter pre-test UNINFORMATIVE [§82.7]; trained arm NOT TRIED
**Status: NOT TRIED. Nothing like it has been built.** This is the principal's idea, raised 2026-09-21,
and it is a step beyond what §48.3 queued.

What v9 does today, confirmed by reading the code rather than from memory:
- **STRING** enters as `PPIMessagePassing(ppi, d)` over one [978,978] adjacency, added residually
  (`model_v9.py:99,145`).
- **Reactome** enters somewhere else entirely: as `M_norm` in the **auxiliary loss**
  (`model_v9.py:180-186`), and in the v5/v6 lineage as a pathway bottleneck.
- **They are never combined into one graph.**
- **Chromatin gates GENE tokens** (EpiGate), never edges.

Why the combination is the interesting version: §48.1 verified against arXiv:2505.14919 that TxPert's graph
is **static and cell-independent** — *"each perturbation p is associated with a randomly initialized input
node embedding … learned via backpropagation"*. So **no published model makes the biological graph a
function of the cell's chromatin state.** Making edge weights depend on accessibility/activation/repression
at both endpoints gives chromatin a **different job** from the one that measured null: not "how much does
this gene move" (§55: +0.000360, CI [−0.005433, +0.006153]) but **"which edges conduct in this cell"** —
i.e. which pathways are available to propagate a perturbation at all.

That is a mechanistically distinct hypothesis, not a retry of the dead one. It also directly serves the
MoA objective: a cell-specific conductive subgraph is an *interpretable object* — you can name the
pathways it opens and closes.

- **Cost:** one training run (~5.8 GPU-h) once the graph is built; graph construction is CPU-only and free.
- **Blocked on:** A2 (the union graph must exist first). ✅ A2 built (W7).
- **Pre-test pre-registered 2026-09-24 [RESULTS §82]:** zero-parameter random-walk-with-restart of each
  compound's DTI targets over the binary union graph, edges gated by the cell's own ATAC, against
  another cell's ATAC (null key), mean ATAC, and no gating; unit = cell line. Delegated as W10.
- **Result [§82.7]: UNINFORMATIVE on ATAC and H3K27ac** — ungated diffusion from landmark targets has no
  signal (median Spearman −0.0006 with |z|), so gating cannot be tested this way. Not refuted.
- **Next form, if pursued:** start from ALL annotated targets (not only landmark ones) on the full STRING
  graph (`string_graph_v9.npz`, 19,496 nodes), read out at the landmarks. The target table EXISTS:
  `drug/outputs/dti/chembl_dti_edges.tsv` (6,020 mechanism edges, 1,363 compounds, 546 targets) and
  `stitch_dti_edges.tsv`. Constraint: chromatin covers only the 978 landmarks, so gating acts on
  landmark-landmark edges only -- test ungated NONE vs DEGREE first. Free, CPU. The trained arm stays
  unpriced until something shows signal.
- **Result [§83.5]: NO_SIGNAL** — "no drug-specific signal from fixed propagation". NONE − RANDOM +0.002
  (57/78 cells, under the 75 % bar; the uniform-random null also mixes in target hub-ness), median ~0.
- **If A1's trained arm is ever priced (review 016 ask 4):** it rests on the hypothesis alone; it needs a
  mismatched-chromatin null key (a trained gate can use chromatin as a cell barcode) and `unseen_cell` as the
  pre-registered readout. **The diffusion route
  is closed;** A1 now needs a trained model (~5.8 GPU-h), queued behind O2.
- **Trained arm PRE-REGISTERED as C7 [RESULTS §89]** (review 026): 6-input gate with observed-indicators, C7u ungated control, mismatched-chromatin attribution requiring ≥ half the gain.
- **Null it must beat:** the same model with chromatin mean-ablated in the edge gate only, plus the
  existing gene-level EpiGate left untouched — so the arm isolates edge gating from gene gating.
- **Pre-register before spending:** method rules 13, 16, 17 all apply. Null key and estimand fixed first.

### A2. Union the four graphs with multi-hot provenance edge features — QUEUED since §48.3, never run
**Source:** TxPert, §48.1, **verified by PI** against arXiv full text: *"Exphormer-MG … adapted for
multi-graph learning via a union graph methodology"*, edges carry a **multi-hot feature** encoding which
source graph each came from. Their ablation: combining helps monotonically, all four beats the best three
(**p < 0.027**). ⚠️ Per-graph numbers are **figure-panel only** — not extractable from text or tables, worker
checked the LaTeX source.

We hold **STRING, Reactome, GO:BP and DTI** and have only ever used them separately or not at all.

- ✅ **The DTI trade turns out not to apply.** §48.3 called this "four graphs", but inspecting the file
  shows `dti_reference.tsv` is **bipartite drug-to-gene** (`pert_id → gene_symbol`, 19,174 edges), not
  gene-to-gene, so it cannot be an edge type in a gene-gene union at all. The union is **three** gene-gene
  sources — STRING, Reactome, GO:BP — and spends no validation set. Connecting drug targets into the gene
  stream is a **separate** mechanism (and closer to TxPert's actual move, §48.2: propagate from the drug's
  targets outward), tracked separately rather than folded in here.
- **Assets confirmed on disk, nothing to download:** `STRING_adj_978_v9.npy` (978², 2.72 % nonzero,
  weighted), `M_pathway_v9.npy` (800×978 int8), `GO_Biological_Process_2023.gmt` (5,406 terms, **never used
  by v9**), `landmark_symbols_v9.tsv` (the authoritative row order). Also present:
  `string_graph_v9.npz` with the **full** 19,496-node / 929,472-edge STRING graph and `landmark_idx` — so a
  2-hop variant through non-landmark intermediates is possible later, and is deliberately out of scope now.
- **Status:** delegated as **W7** (`research/W7_union_graph.md`), 2026-09-21. CPU-only. Free.
- **Two decisions made in the brief, not delegated:** term-size filtering is mandatory (a 400-gene pathway
  is a 79,800-edge clique with no specificity; `A_copathway.npy` at 12.83 % density is the symptom), and
  the three source weights are emitted **separately and unnormalised** because they are different
  quantities and collapsing them is a modelling choice for the consumer.

### A3. Chromatin direction/sign head — "the one form of the question this project has never put"
**Source:** §49.4, from Agrawal et al. (F1000Research), the closest flank found by the adversarial novelty
sweep. They reach **AUC 0.71–0.89** predicting **direction of change** on **extreme responders only** from
H3K27ac. We predicted **continuous magnitude for every gene** and got +0.004.

§49.4's reading, which still stands: chromatin may carry **which genes move and which way**, not **how
much**; and the signal may live in the **tails**, which the reproducible-stratum filter keeps but a per-gene
regression loss dilutes across all 978.

- **Build:** auxiliary direction/sign head conditioned on chromatin, scored as **AUC on top-k up vs top-k
  down genes**. Null: the same head with chromatin mean-ablated.
- **Cost:** one run, and it can ride along as an auxiliary head on any other arm — near-free if bundled.
- **Independent of** the token-vs-additive question, so it does not collide with A1/A2.

### A4. Dose and time as tokens INSIDE the drug stream — the one way atoms might yet matter
**Source:** §47.4, **verified by PI** in XPert's code. Their drug sequence is
`[dose, time, HG_embed, atom_1..atom_n]` (`unimol_Embeddings`, `model_utils.py:133`). Ours applies dose/time
as a **FiLM scale/shift on the gene tokens**, far from the atoms — same information, **structurally unable
to express the interaction** "this atom matters more at this exposure".

Relevant now: §61.7 established that atom tokens *hurt* even with self-attention. A4 is a different reason
they might be useful, and it is the only untested one that changes what the atoms can express.

- §47.4's own instruction: **test separately, do not bundle with 47.3.**
- **Cost:** one training run (~5.8 GPU-h).

### A5. Tail-weighted / top-k loss
**Source:** derived from §49.4's diagnosis, **not** from a specific paper — no citation yet, so it is an
idea, not a finding. If chromatin's signal lives in the responders, a loss averaging over all 978 genes
dilutes it. A magnitude-invariant correlation loss was tried once in a gentle finetune and failed to lift
correlation (RESULTS line 96) — that is evidence about *that* loss in *that* setting, not about tail
weighting.
- **Cost:** free to implement, one run to test. Cheapest paired with A3.

### A6. Token-vs-additive ablation for auxiliary biology
**Source:** §48.4, where a worker's novelty claim was **REJECTED** and the defensible residue recorded:
TxPert and State supply auxiliary biology as an **additive embedding**, PertAdapt as a **static attention
mask**, and **none of them ablates tokens-versus-additive.** That is a gap in the evidence, not an
established novelty — §48.4 is explicit that the worker's stronger claim is not accepted.
- ⚠️ State and PertAdapt claims are **REPORTED-NOT-VERIFIED** and must be checked before being cited.
- **Cost:** two runs, or one run with a flag.

### A8. 🔵 Keep the atoms, drop atom-to-atom self-attention — a trained arm, NOT YET PRICED
**Source:** RESULTS §74, the first mechanism reading in the atom line to pass its pre-committed null-key gate
(null span 11.6 % of hypothesis span, threshold 25 %). In the trained SA-on model, cutting atom→atom attention at
inference — global↔atom kept — removes ~71 % of the atom tokens' harm on unseen compounds and raises the model's
own score by +0.0145.
- **What it would test:** whether a model TRAINED with atoms + global↔atom attention but no atom→atom attention
  beats both the SA-on and SA-off references. §74.3: inference-time masking cannot answer that.
- **Cost:** one training run, ~6 GPU-h. Capacity-matched against SA-on by construction if the same `_DrugBlock`
  is used with the atom-only mask fixed at α=0 during training.
- **Pre-registered DISQUALIFIER [§75.2]:** a significant `unseen_cell` degradation disqualifies it, whatever it does
  on the compound splits — because at inference masking atom-to-atom attention **hurts** unseen cells (paired median
  +0.00298, 58 % of rows, p 6.3e−10) while helping unseen compounds.
- **Design per review 010 ask 4:** compare vs `sa0` (batch 48) per row, not r0-r2 (batch 96); pre-register vs the
  measured fold-0 between-seed sd (under ~2√2×sd is inconclusive); pre-register the predicted directions.
- **Cheaper question first:** for unseen-compound accuracy alone, the inference-time mask on `sa0` already delivers it.
- **Status:** logged 2026-09-23; to be pre-registered and priced through review before any spend.

### A9. Where the atom tokens' residual harm lives, with atom-to-atom attention removed — ⚫ RUN 2026-09-24: NOT ANSWERABLE at inference [§79.9]
**Source:** §75.6, review 010 ask 3. At α = 0 the atom effect is still −0.00322 (sign p 1.6e−10). Carriers: gene→drug
cross-attention; atom→global then global→genes; the cross-block global leak. Two inference-only cuts separate them.
Each needs a kill switch and the `x_cell` gate **committed before running**. Zero GPU-hours.
- **Pre-committed 2026-09-23 in RESULTS §79** as a 2×2 of cuts X (genes read the global key only) and G (the
  global row reads only itself), with a two-sided kill switch at 3×|E0| and the §67.3 gate. Delegated as W9.
- **Result [§79.9]:** the route inventory is complete (both cuts: `dY_max` exactly 0.0), but **both single
  cuts fail the kill switch** (+0.042, +0.046 against 0.00966). Not answerable by inference-time cuts;
  the question passes to a trained arm and is folded into A8's pricing.

### A10. ⚫ Do atom tokens memorise training compounds? — graded NOT SUPPORTED [§76.3]; binary NOT REFUTED, "most on warm" UNRESOLVED [§76.5]
**Source:** review 010, offered by the reviewer after seeing the data. Atoms help where test compounds were seen in
training (`unseen_cell`) and hurt where they were not (both compound splits). **Predicts atoms help most on a fully
warm split.** A test must commit its prediction first; nothing in §74 is read under it.
- **T2 run 2026-09-23 [§76.3–76.4]: GRADED memorisation not supported** (Spearman −0.119 over 858 compounds,
  p 0.0006; null key flat; confounded by reference compounds). The **binary** form — seen at all vs never seen — is
  still open and is exactly what **T1** tests.
- **T1 run 2026-09-25 [§76.5]: INCONCLUSIVE.** Warm median +0.0040 [+0.0032, +0.0050] vs the committed
  bar 0.00378 (`unseen_cell`'s CI upper bound): positive, bar not cleared. Per review 017 C1 the binary form was
  tested only by the refuted branch (not refuted — weak); the rival "atoms fail to extrapolate across
  scaffolds" is untouched. A new test needs a different drug fold. Does not gate any spend.

### A11. 🔵 A DRUG-SPECIFIC mechanistic readout — the project has none today [RESULTS §85.1, §85.3]
- v9's pathway alignment is cell-level (CLAIMS 4.16). Step 1, zero GPU: gradient importance `|a·∂Ŷ/∂a|`
  (CLAIMS 4.14) on existing checkpoints — does it reorder across drugs, and do target-containing pathways rank
  above a label-permutation AND a drug-shuffle null? Step 2 only if null: a trained post-perturbation named
  pathway readout (C8b). Pre-register before computing.
- **Step 1 RESULT [§86.4]: NULL** (valid probe; drug reorders importance, rho_del ~0.16; floor met on 1 of 3 seeds).
  **C8b is now the route**: a trained post-perturbation named readout, with its own MoA test pre-registered.
- **C8b's MoA test must pre-register (review 023 ask 3):** (1) ONE primary readout — the node's activation difference Δa = a(d) − a(mean drug), fixed mean drug; (2) whether the nodes are aux-supervised (then bounded near the data projection, which shows no alignment) or unsupervised; (3) calibration against ≥ 5 untrained inits — every trained seed's diff below the untrained minimum (or mean − 2 sd fixed from those inits), since −0.02 is inside the observed spread; (4) readouts compared by diff only; (5) a responsiveness split with usable sizes on both sides (the top-50 rule gave 483 vs 13); (6) the unseen-compound stratum licenses a mechanism claim, with its null sd stated; (7) run on v9 fold-0 test rows, trained separately from §85's cc1 dev work, and the MoA reading never influences C8b's accuracy acceptance or vice versa.

### A12. 🔵 Make chromatin matter — the brainstorm behind the §91 funnel (principal, 2026-10-02)
**Diagnosis** (§91.1 as amended by review 040 C3): v9's only **explicit** chromatin terms are drug-independent per-(cell, gene)
offsets; a drug × chromatin interaction is expressible only implicitly (gene tokens cross-attend to the drug) and §55 shows it is
not used; §44's ridge used chromatin as a cell-level block over ~40 cells. None of them makes "this drug's response at this gene
depends on this gene's chromatin in this cell" an explicit term.

| # | strategy | what it changes | cheapest test | status |
|---|---|---|---|---|
| S1 | **drug-conditioned chromatin interaction** — gain and shift on the drug's own response, `μ·(1 + v_d·f) + w_d·f`, f = this gene's chromatin | chromatin's effect depends on the drug; gene-local, so it transfers to new cells | T1 (closed form) | funnel |
| S2 | **retrieval by chromatin similarity** — an unseen cell borrows the responses of chromatin-similar training cells | chromatin as a cell-similarity metric, the way lineage helped | T2 | funnel |
| S3 | train on the **cell-specific component** (response minus the drug's mean in other cells) | the only variance cell features can explain gets its own loss | T1's cell-centred score | inside T1 |
| S4 | **residualised chromatin** — the part basal expression does not carry | removes the redundant part the model gets from `x_ctl` | T1 FBC⊥ | inside T1 |
| S5 | **which genes can move here** — chromatin predicts a gene's response spread in this cell | a per-gene response scale (not the rejected output gate) | T3 | funnel |
| S6 | **epigenetic drugs** (HDAC, BET, EZH2, DNMT, KDM inhibitors) | chromatin should matter most where the drug acts on chromatin | T1 stratum | reported |
| S7 | direction head conditioned on chromatin (A3) | sign, not magnitude, on extreme responders | T1's top tercile; A3 if T1 passes | deferred |
| S8 | chromatin-gated edges | — | C7, §89 | not accepted (+0.0020) |
| S9 | **richer features**: enhancer windows (±50 kb), TF-motif accessibility (chromVAR), enhancer–gene links | promoter peaks may be the wrong granularity | needs peak-level data (not on disk; bigWig access intractable here, 2026-07) | only if T1 shows a promoter-level signal worth extending |
| S10 | impute chromatin for uncovered cells from expression | harmonises inputs | — | rejected: adds no information beyond `x_ctl` |
| S11 | drop `x_ctl` at random in training so the model must use chromatin | forces a chromatin pathway | — | only if T1 shows non-redundant information |
| S12 | **fix the encoding**: failed-ChIP H3K27me3 tracks are z-scored back to full weight and `r` ignores `E_reliability.tsv` (§91.8) | v9 may be reading noise at |z| 10–28 | T1 primary (rank-normal, failed tracks missing) vs its v9-encoding secondary | inside T1 |

### A7. 🔵 "New startup mechanisms" — raised by the principal, no recorded source
The principal listed this alongside new loss functions on 2026-09-21. **There is no item in RESULTS it maps
onto**, so it is recorded here unattributed rather than invented. v9 currently uses a **WSD schedule** with
`warmup_frac 0.03`, AdamW, `decay_frac 0.2`.
- **Next step:** say which mechanism was meant, or run a literature pass. Not actionable as written.

---

## B. What is load-bearing — the dissection line (keep alive)

### B1. Bai et al.'s retrain-from-scratch drug ablation, on v9
**Source:** §50, Bai/Prince/Nitschke bioRxiv 2026-05-15, **verified by PI against the preprint full text**,
which also caught a worker error (§50.5). Their finding: a **drug-free MLP scores 0.637 against XPert's
0.633**, and zeroing drug features at inference moves ΔPCC_DEG by at most 0.012.
The distinction that saved CLAIMS 1.9 and 5.1: they **retrain from scratch** drug-blind; we ablated at
**inference**. Those are different experiments.
- **Now the strongest form of "the drug branch does not earn its place"**, given §61.7.
- **Open question they never asked:** whether drug features matter for unseen-**cell** generalisation.
- **Cost:** ~5.8 GPU-h for the v9 drug-blind retrain; their drug-free MLP baseline on our splits is cheap.

### B3. Does the drug-free MLP recall shared control profiles? (RESULTS §98.11, review 077)
**Source:** R1's reported strata. M is ahead on clean rows whose exact `X_ctl` also occurs in training (−0.011); v9 is ahead on
rows with a new control (+0.044). This replicates within each of the six largest cells. Plate/time composition is not ruled out.
- **(i) A control-substitution M:** retrain M with each row's `X_ctl` replaced by another control of the same cell, breaking
  the exact-control link. If recall is the mechanism, M's shared-row advantage should vanish. CPU only (≈ 4 h per seed, free).
- **(ii) A plate-disjoint split:** test controls never seen in training, by construction. That needs plate identities, which
  the bundle lacks; they would come from GSE metadata.
- **Either needs its own registration.** Not run until B1 / R2 is read.

### B2. Remove the control ENCODER on cold-cell
**Source:** §48.3 item 2, TxPert, **verified by PI**: *"no basal state encoder is by far the most effective
option"* for cross-cell-line transfer, with ŷ = x + gφ(Σ z_p). v9 runs **two** control encoders
(`ctl_enc`, `cell_enc`) mixed into the gene tokens. We already anchor absolute as control + delta (§43.1),
so we are half-way; the untested half is whether encoding the basal state **at all** hurts on unseen cells.
- **Cost:** cheap ablation, directly on the split that matters.

---

## C. Cold-cell head-to-head — the competitor line (keep alive)

### C1. Train XPert, TranSiGen and others ourselves on `split_cold_cell_1..5`
**Source:** §46.5. Published reference points (§46, **verified by PI** in XPert's Supplementary Table R8):
XPert warm **0.688 ± 0.011**, cold-drug **0.645 ± 0.008**, cold-cell **0.383 ± 0.027**; TranSiGen cold-cell
**0.293 ± 0.017**. Cold-cell is where the whole field is weakest and where §45.0 puts us **+0.178 over
ridge**. The same machinery admits TranSiGen, PRnet, DeepCE and CIGER.
- **Cost:** one training run per competitor per fold. The expensive item on this list; scope before buying.

---

## D. Open verification debts
- **MultiFlow's perturbation-type detail** — still needs the full text (§49.5).
- **State and PertAdapt** claims from W2 — REPORTED-NOT-VERIFIED (§48).
- **TxPert per-graph ablation numbers** — figure-panel only, not extractable (§48.3).

---

## Settled, kept so they are not re-proposed
| idea | outcome | where |
|---|---|---|
| Chromatin conditioning as the core contribution | **KILLED.** +0.000360, CI [−0.005433, +0.006153], 2/5 cells positive, sign test p=1.000. **2026-10-02 (§91.10):** that CI omits run-to-run variance; one run per arm could not detect effects below ≈ 0.006–0.016, so the kill covers large effects only | §54, §55, §91.10 |
| Sparse-vs-dense attention as XPert's advantage | **KILLED** — dead code in their repo; the real difference is drug self-attention | §47.6 |
| In-loop drug self-attention rescues the atom tokens | **KILLED.** Atoms still hurt at −0.01814 [−0.02635, −0.00916], 66 % of rows, with SA trained and present | §60.7, §61.7 |
| The 2x2 interaction as evidence of mechanism | **RETRACTED** — the pre-committed estimand failed its own null key | §60.9, §61.1 |
| Atom→gene attention recovers drug targets | **FALSIFIED** — median target rank percentile 0.560, worse than chance | v5 era, README |
| Auxiliary biology as attended tokens is "completely novel" | **REJECTED** as a worker overclaim; the defensible residue is A6 | §48.4 |
