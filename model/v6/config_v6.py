# -*- coding: utf-8 -*-
"""v6 configuration. Every non-obvious value carries the measurement that justifies it.
Design rationale: ARCHITECTURE.md (same directory)."""
from dataclasses import dataclass


@dataclass
class V6Config:
    # ---- dimensions ----
    n_genes: int = 978
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    l_base: int = 2           # drug-free context encoder
    l_perturb: int = 4        # atom->gene + gene<->gene
    max_atoms: int = 96       # heavy-atom cap (p99 = 59); FIXED padding for XLA/TPU static shapes
    dropout: float = 0.1

    # ---- input dims (frozen by the data) ----
    d_atom: int = 512         # Uni-Mol per-atom representation
    d_epi: int = 3            # ATAC / H3K27ac / H3K27me3
    d_global: int = 512 + 20 + 2048   # UniMol CLS + descriptors + ECFP4.
    # ChemBERTa (384) REMOVED: drug-feature ablation measured its contribution at dR2 = +0.001 [5.2].
    d_cell_ctx: int = 16      # lineage one-hot, col 0 = UNKNOWN (26/83 cells lack a DepMap lineage) [5.4]

    # ---- v6: the biological bottleneck (replaces v5's soft prior bias) ----
    d_pathway: int = 64       # width of each of the 360 named Reactome pathway nodes
    pathway_epi_gate: bool = True   # chromatin gates at the PATHWAY level; v5 gated per gene, where it
                                    # largely re-encoded X_base (corr 0.74) and contributed ~0 [3.1a/3.1b]

    # ---- the soft prior bias is OFF by default ----
    # v5 measurement: prior heads land on-support only 1.1x random [4.6] -- it does not steer. Kept
    # switchable so v6 can be compared against it directly rather than on assertion.
    use_prior_bias: bool = False
    n_prior_heads: int = 4

    d_epi_hidden: int = 16
    huber_delta: float = 1.0


@dataclass
class V6DataConfig:
    """Paths are relative to the repo root. Provenance for each file: ARCHITECTURE.md 2."""
    root: str = "C:/Projects/LINCS"
    y_path: str = "phase2_assembly/outputs/Y_target_level5_978.npy"
    sig_path: str = "phase2_assembly/outputs/signatures_usable.tsv"
    strength_path: str = "phase2_assembly/outputs/sig_strength.npy"
    xbase_path: str = "baseline/outputs/ccle_baseline_lincs_v5/X_base_lincs.npy"
    e_path: str = "phase2_assembly/outputs/E_final.npy"
    e_mask_path: str = "phase2_assembly/outputs/E_final_mask.npy"
    e_reliability_path: str = "phase2_assembly/outputs/E_reliability.tsv"
    lineage_path: str = "baseline/outputs/cellfeat/cell_lineage.npy"

    # THE biological backbone of v6: 360 named Reactome pathways x 978 landmark genes.
    # v5 used the gene-gene COLLAPSE of this (A_copathway), discarding the named pathway axis.
    m_reactome_path: str = "network/outputs/M_reactome.npy"
    # Row p of M_reactome <-> row p of pathway_info.tsv (pathway_id, pathway_name, member symbols).
    # Without this the "360 NAMED nodes" claim is not checkable, so the mapping is VERIFIED at load
    # (eval_v6._load_pathway_names re-derives each row's gene set from the symbols and asserts equality).
    pathway_info_path: str = "network/outputs/pathway_info.tsv"
    gene_order_path: str = "Data Info/pathway_landmark_genes.txt"   # canonical 978-gene order
    cop_path: str = "network/outputs/A_copathway.npy"     # only if use_prior_bias
    ppi_path: str = "network/outputs/STRING_adj_978.npy"  # only if use_prior_bias

    drug_index_path: str = "drug/outputs/drug_feature_index.json"
    desc_path: str = "drug/outputs/drug_descriptors.npy"
    fp_path: str = "drug/outputs/drug_fingerprints.npy"
    unimol_cls_path: str = "drug/outputs/drug_unimol.npy"
    chemberta_path: str = "drug/outputs/drug_chemberta.npy"
    atom_reprs_path: str = "drug/outputs/drug_atom_reprs.npy"
    atom_offsets_path: str = "drug/outputs/drug_atom_offsets.npy"
    scaffold_split_path: str = "drug/outputs/splits/scaffold_split.json"

    use_chemberta: bool = False        # keep in sync with V6Config.d_global
    center_epi: bool = True            # per-(cell,mark) standardise: strips the per-cell technical offset
    reliability_weighting: bool = True # LINCS is ~75% inert [6.1]
    min_strength: float = 0.0
    eval_min_strength: float = 1.0     # ALL reported metrics are on the reproducible stratum

    # cold-CELL x cold-COMPOUND: one run yields unseen-cell / unseen-compound / unseen-both
    cold_cell_test: tuple = ("MCF10A", "NPC")
    n_cell_folds: int = 5
    cell_fold: int = 0
    drug_fold: int = 0
    val_frac: float = 0.05
    seed: int = 0
