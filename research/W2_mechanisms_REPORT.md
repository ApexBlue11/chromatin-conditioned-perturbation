# Mechanisms from 2026 Perturbation Papers

## Paper 1 — TxPert
**Q1.1: How exactly are MULTIPLE knowledge graphs combined?**
TxPert combines multiple graphs (e.g., STRINGdb, GO) using a "union graph methodology." The operator is the mathematical union of all edges across the source graphs to form a unified graph. To prevent information loss and encode provenance, they introduce a multi-hot edge feature vector $\mathcal{E}_e \in \{0, 1\}^k$, where the $j$-th component is 1 if the edge exists in the $j$-th source graph. This unified graph and multi-hot edge representation is then processed using an Exphormer (Graph Transformer) variant they call "Exphormer-MG" (Section 2.1, Eq in Section "Detailed evaluation...").

**Q1.2: Which graphs do they use, and what does the ablation say each one contributes individually vs in combination?**
They use four graphs: STRINGdb, Gene Ontology (GO), PxMap, and TxMap. 
Individually, STRINGdb performs the best, followed by PxMap. Incrementally integrating multiple graphs starting from STRINGdb consistently improves predictive performance, peaking when all four are combined (achieving a significant improvement over the best 3-graph combination with a T-test $p < 0.027$).
**Numbers:** `UNKNOWN`. The exact MSE/Pearson numbers for the ablation are plotted in Figure 1C/1D (referred to as `fig:graph_ablation_main` in `main.tex`), but the raw quantitative values are omitted from the text and tables in the LaTeX source.

**Q1.3: How do they represent a cell line the model has never seen?**
For unseen cell lines, the model relies on learning a perturbation delta vector that is applied directly to the empirical control profile of the target cell line. While perturbation data is held out, the model is trained on the unperturbed control cells from that context. At inference, they use the raw expression profile of the unseen cell line's control cells as the basal state ($\mathbf{s} \coloneqq \mathbf{x}$), and the decoder adds the learned perturbation embedding shift directly to it: $\mathbf{\hat{y}} = \mathbf{x} + g_\phi(\sum \mathbf{z}_p)$ (Section "Encoders" / "OOD Perturbation effect prediction tasks").

**Q1.4: Is the graph used as a static prior, or is it conditioned on the perturbation?**
It is used as a static prior. The GNN processes randomly initialized node embeddings ($\mathbf{h}_p^0$) that are treated as learnable parameters. The message passing over the gene-gene interaction graphs refines these embeddings into final perturbation encodings ($\mathbf{z}_p$). This refinement depends entirely on the static graph structure and the perturbation token itself, completely independent of the cell's context or basal state. The resulting static embedding is then extracted and summed with the basal state representation.

## Paper 2 — State
**Q2.1: What exactly are the tokens — individual cells? What is the set? How is the set constructed at inference?**
- **Tokens:** The tokens are individual cells. Each cell's log-normalized expression vector (or its embedding from the State Embedding model) is projected via a 4-layer MLP into a hidden dimension $d_h$, acting as a single sequence token.
- **The Set:** The set is a non-overlapping group of a fixed number of control cells (e.g., $S=256$) that share the same biological covariates (cell line and batch). 
- **Inference Construction:** At inference, the available unperturbed control cells for a given cell context are partitioned into non-overlapping subsets of size $S$. If the number of cells is not perfectly divisible by $S$, the final subset is padded to size $S$ by sampling additional cells with replacement from itself (Section 4.3.1).

**Q2.2: How is a perturbation represented and injected into that stream?**
Perturbations are passed through a 4-layer MLP to produce a perturbation embedding of dimension $d_h$. This embedding is injected **additively**: it is broadcast and summed element-wise into every control cell token's embedding (along with batch embeddings) prior to being passed into the transformer backbone. $\mathbf{H} = \mathbf{H}_{cell} + \mathbf{H}_{pert} + \mathbf{H}_{batch}$ (Section 4.3.3, Eq 16).

**Q2.3: How is cell context encoded, and what makes it transfer to a context not seen in training?**
Cell context is either directly encoded from raw expression using an MLP, or encoded using a pre-trained State Embedding (SE) model. The SE model is a transformer that represents a cell by attending over its top expressed genes (using ESM-2 protein language model embeddings summed with soft-binned expression magnitudes). The `[CLS]` token from this SE model serves as the rich transcriptomic state of the cell. It transfers to unseen contexts because the SE model is pre-trained via self-supervised gene expression reconstruction on an observational dataset of 167 million cells across ~14,000 datasets, learning generalizable features robust to technical noise (Section 4.4 / Fig 3).

## Paper 3 — PertAdapt
**Q3.1: What is "condition-sensitive adaptation" mechanically? Which parameters are adapted and by what?**
Mechanically, this is implemented via a plug-in "perturbation-conditional adapter". The adapter takes the frozen contextualized cell representations from the foundation model ($F(x_c)$) and adds the perturbation embedding $p(c)$ row-wise: $Q = \text{LayerNorm}(F(x_c) + \frac{1}{N} p(c)^\top)$. This acts as the Query, Key, and Value for a multi-head self-attention layer, where the attention mechanism is restricted by a static binary mask derived from Gene Ontology (genes only attend to each other if they share a GO term). 
- **Parameters adapted:** The perturbation encoder (a GNN from GEARS), the adapter modules (masked MHA, layer norm, feed-forward networks), and the final gene-wise output MLPs. The backbone foundation model is strictly frozen.
- **Adapted by what:** These parameters are updated by backpropagation via an "adaptive loss" function that dynamically reweights the MSE of all readout genes against the MSE of the top $k$ differentially expressed (perturbation-sensitive) genes (Section 2.2.2 / 2.3).

**Q3.2: What was failing in the un-adapted foundation model that this fixes?**
Two primary failures are cited: 
1. **Shallow Knowledge Transfer:** Existing approaches merely attach a linear head or reuse existing models (like GEARS) over the FM representations, leaving the pre-trained knowledge under-utilized for out-of-distribution shifts. 
2. **Gene Imbalance Diluting Gradients:** For a given perturbation, only a tiny fraction of the ~20,000 readout genes actually exhibit a response. Un-adapted models treat all genes equally using a uniform MSE loss, meaning gradients are diluted by thousands of uninformative targets, preventing effective optimization for the genes that actually change (Section 1).

## Cross-cutting question
**Q4. Is there evidence that auxiliary biological information contributes more when supplied as its own attended token set than when summed into an existing representation?**

The literature **does not address this**. Across these three papers, auxiliary biological information is never provided as an independent token set for the core self-attention mechanism to attend over:
- **TxPert** processes biological graphs using a completely separate GNN to derive a single perturbation embedding, which is then *added* to the basal cell representation. 
- **State** uses protein language model (ESM-2) features, but *sums* them with expression magnitude embeddings (Eq 27). Perturbations are also *summed* directly into the cell tokens before the transformer (Eq 16). 
- **PertAdapt** utilizes Gene Ontology, but solely as a static binary *attention mask* to zero-out interactions between unrelated genes, rather than encoding pathways as distinct tokens.

Because none of these state-of-the-art architectures model auxiliary biological data (like named pathways or chromatin tracks) as independent tokens within the attention sequence, there are no ablations comparing this approach to additive injection. Formulating auxiliary information as its own attended token set would be a completely novel comparison against the current literature.

## What I could NOT determine
- For **TxPert (Q1.2)**, I could not determine the exact numerical values (MSE/Pearson) for the graph ablation contributions. These values are plotted visually in the paper's Figure 1 (`fig:graph_ablation_main` and `stringdb_ablations_supp_v2`), but the explicit numbers are omitted from the text and tables in the LaTeX source.

## Which version of each paper I read
- **TxPert:** LaTeX source from arXiv preprint (arXiv:2505.14919).
- **State:** Full-text PDF from bioRxiv preprint (doi: 10.1101/2025.06.26.661135).
- **PertAdapt:** Plain-text XML extraction from EuropePMC (PMCID: PMC13341120).
