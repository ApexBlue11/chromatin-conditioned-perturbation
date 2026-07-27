# Design Notes — Model Architecture & Related Work

Companion to the branch `report.md` files. Records the model-side design directions and what we
took from the literature. **Core objective: interpretability — associations with proper confidence,
not vague/black-box associations.**

## Branch structure (clarification)
Two orthogonal, composable "branch" axes:
- **Modality axis ("Multi-Encoder", ≥4 branches):** cell baseline · epigenetics · network/pathway ·
  drug · (dose+time condition tokens).
- **Pre/post axis (XPert-style, optional to adopt):** a base encoder (pre-perturbation state) + a
  perturbation-effect encoder predicting the residual `xdeg = xpert − xbase`.

We are richer than "dual/triple" on the modality axis; we can additionally adopt the pre/post
residual formulation on top.

## Where attention lands (for interpretability)
Cross-attention should be inspectable at every hop:
- **drug-atom ↔ gene** (SAR-level attribution — XPert's validated win)
- **gene ↔ gene**, biased/masked by the **pathway/PPI prior** (network branch)
- **epigenetics ↔ gene** (why a gene is poised to respond in *this* cell)

Epigenetics plays two composable roles: the specified multiplicative **gate**
`x_mod = x_base · sigmoid(MLP_epi([ATAC, H3K27ac, H3K27me3]))` **and** an attention input.
Target story: *drug substructure → target → pathway propagation → gated by this cell's chromatin →
gene response*, with defensible weights throughout.

## MoA grounding
Both reference papers: **chemical structure alone cannot resolve MoA** (same-MoA drugs scatter in
chemical space). We reconstruct **cell-specific MoA** by combining drug-atom features + pathway prior
+ epigenetics + gene–gene regulation — an alternative to XPert's explicit (cell-agnostic) drug-MoA
knowledge graph, and arguably stronger because it is conditioned on the actual cell line.
Gap to consider: no explicit **drug→target (DTI)** prior yet. **Plan (endorsed):** primary =
*learn-then-validate* — let the model learn drug→gene, expose it cleanly, and validate against a DTI
database (DrugBank/STITCH/ChEMBL); a recovered+validated DTI is a stronger interpretability/discovery
claim than a hardcoded prior. Secondary/ablation = inject DTI as a prior (XPert-style) to measure the
performance gain. Try both if ETAs allow. Caveat: LINCS = downstream transcriptional effects, not direct
binding, so a learned "DTI" overlaps with but ≠ a binding-DTI database.

We also adopt XPert's pre/post split (predict `xdeg = xpert − xbase`) on top of the multi-encoder inputs.

## Related work
- **XPert** (`s42256-025-01165-w.pdf`, Nat. Mach. Intell. Jan 2026): dual-branch transformer on L1000;
  UniMol + drug-MoA knowledge graph (DTI/PPI/DDS); nonlinear dose/time tokens; **interpretable**
  attention (atom-level SAR, resistance biomarkers). The model to emulate for our objective.
- **StateXDiff** (`statexdiff.pdf`, preprint May 2026): conditional latent diffusion; multimodal cell
  state (RNA + pseudo-protein); mechanism-aware drug template; **bidirectional cross-attention**;
  triplet + reliability weighting vs spurious signals. Borrow the bidirectional coupling and
  anti-spurious weighting; avoid the diffusion black box (hurts interpretability).

## Warnings carried forward
1. **VAE over-denoising** — we are a VAE; VAE baselines fail cold-cell (negative R² despite good
   correlation). Evaluate cold-cell explicitly; report R²/magnitude, not just correlation; consider
   supervised signal/noise separation.
2. **Structure ≠ MoA** → biological priors mandatory.
3. **Dose** — no one-hot; nonlinear (inverted-U); aggregate pharmacologically-equivalent doses.
4. **Spurious correlations / low SNR** → reliability-weight unreliable inputs; map the epigenetics
   confidence flags (`tissue_type_only`, `imputed`, `related_line`) to reliability weights.
5. **Multimodal representation collapse** — dominant RNA modality can drown weaker ones (epigenetics);
   use disentangled alignment.
6. **OOD: expressive ≈ simple baselines** → always include Mean / Meancell / Meandrug baselines.
7. **Batch/plate effects** — a real generalization hazard; handle explicitly.
