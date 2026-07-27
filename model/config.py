# -*- coding: utf-8 -*-
"""Model + data configuration. Dims/hparams match model/MODEL_MATH.md."""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # dimensions
    n_genes: int = 978
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    l_base: int = 2          # base/context encoder layers (gene<->gene, drug-free)
    l_perturb: int = 4       # perturbation encoder layers (atom->gene + gene<->gene)
    max_atoms: int = 96      # M: heavy-atom token cap (p99=59)
    dropout: float = 0.1

    # input feature dims (frozen by the data)
    d_atom: int = 512        # UniMol atomic_reprs / CLS dim
    d_epi: int = 3           # ATAC, H3K27ac, H3K27me3
    # unimol CLS + [chemberta] + descriptors + fingerprint. ChemBERTa is DROPPED by default: the drug-feature
    # ablation measured its contribution at ΔR²=+0.001 / Δpearson=+0.002 (dead weight, redundant with the
    # fingerprint+UniMol pillars). MUST stay in sync with DataConfig.use_chemberta (which builds u_feats).
    d_global: int = 512 + 20 + 2048

    # v5: per-cell CONTEXT for FiLM (lineage one-hot, dim0=UNKNOWN for the 26 non-cancer/primary lines absent
    # from DepMap). Measured premise: partial corr(lineage, response | X_base) = +0.092, p<0.0005 (small but
    # real). 0 disables. Cell features are FAIR for cold-cell eval: observable without any perturbation.
    d_cell_ctx: int = 16

    # gene<->gene prior bias
    n_prior_heads: int = 4   # heads that receive the additive biological bias (rest are free/discovery)
    prior_transform: str = "log1p"   # transform applied to the (nonneg) prior matrices before biasing

    # epigenetic gate
    d_epi_hidden: int = 16

    # v5: cell-conditional PATHWAY CONDUCTANCE. The prior bias was static (same lambda every cell); this
    # lets chromatin/baseline decide per (cell,gene) how much a gene listens to its pathway neighbours.
    # Measured motivation: pathway module activity varies a lot across cells (across-cell std ~0.77), and
    # the pathway interpretability leg was the weakest (prior heads only 1.1x random on-support).
    pathway_conductance: bool = True

    # output chromatin gate (multiply Yhat by s); default OFF per design choice
    output_gate: bool = False

    # v2: additive SIGNED epigenetic head  Yhat_g += MLP_epi_out(E) * r_{c,g}. The multiplicative gate is
    # a magnitude modulator and cannot express the measured signed effect (low H3K27ac/high H3K27me3 ->
    # gene UP); this head can, and is directly interpretable as "chromatin's contribution to gene g".
    epi_additive: bool = True

    # loss
    huber_delta: float = 1.0


@dataclass
class DataConfig:
    root: str = "C:/Projects/LINCS"
    y_path: str = "phase2_assembly/outputs/Y_target_level5_978.npy"
    sig_path: str = "phase2_assembly/outputs/signatures_usable.tsv"
    keep_mask_path: str = "phase2_assembly/outputs/Y_keep_mask_smiles.npy"
    xbase_path: str = "baseline/outputs/ccle_baseline_lincs_v5/X_base_lincs.npy"
    background_path: str = "baseline/outputs/ccle_baseline_lincs_v5/ccle_full_background.npy"
    e_path: str = "phase2_assembly/outputs/E_final.npy"
    e_mask_path: str = "phase2_assembly/outputs/E_final_mask.npy"
    e_reliability_path: str = "phase2_assembly/outputs/E_reliability.tsv"
    cop_path: str = "network/outputs/A_copathway.npy"
    ppi_path: str = "network/outputs/STRING_adj_978.npy"
    # drug features (pert_id -> row via drug_feature_index.json)
    drug_index_path: str = "drug/outputs/drug_feature_index.json"
    desc_path: str = "drug/outputs/drug_descriptors.npy"
    fp_path: str = "drug/outputs/drug_fingerprints.npy"
    unimol_cls_path: str = "drug/outputs/drug_unimol.npy"
    chemberta_path: str = "drug/outputs/drug_chemberta.npy"
    atom_reprs_path: str = "drug/outputs/drug_atom_reprs.npy"     # (sum_p count_p, 512)
    atom_offsets_path: str = "drug/outputs/drug_atom_offsets.npy"  # (n_drug+1,)
    # v3: per-signature reliability weighting. LINCS L5 is mostly irreproducible (replicate r=0.127
    # median; ~75% of sigs reproduce at r<0.14). Weighting by MEASURED reliability(strength) stops the
    # model hedging toward the drug-average signature (it under-expressed the drug x cell interaction:
    # 25.4% predicted vs 47.9% true). strength_path = mean|Y| per signature.
    strength_path: str = "phase2_assembly/outputs/sig_strength.npy"
    lineage_path: str = "baseline/outputs/cellfeat/cell_lineage.npy"   # [83,16] one-hot, col0=UNKNOWN
    # drop ChemBERTa from u_feats (ablation: dead weight). Keep in sync with ModelConfig.d_global.
    use_chemberta: bool = False
    reliability_weighting: bool = True
    min_strength: float = 0.0        # optional hard filter on TRAIN (0 = keep all, rely on weights)
    eval_min_strength: float = 1.0   # report metrics on REPRODUCIBLE sigs (replicate r ~0.37+)

    # v2: per-(cell,mark) standardize E across genes to remove the technical per-cell offset
    # (35-69% of E variance is a uniform per-cell shift = batch effect the v1 gate latched onto).
    center_epi: bool = True

    # splits -- cold-CELL k-fold: all cells partitioned into n_cell_folds balanced by signature
    # count; `cold_cell_test` cells are pinned into fold 0 (the originally designated test lines).
    # Test = cells in `cell_fold`. Run other folds to get a k-fold cold-cell average.
    cold_cell_test: tuple = ("MCF10A", "NPC")
    # unseen-COMPOUND holdout: Bemis-Murcko scaffold fold held out of training (None = disabled).
    # NOTE (measured): scaffold splitting alone does NOT make compounds truly unseen -- median max
    # Tanimoto from a test drug to its nearest train drug is 0.655, 39% >= 0.70. Always report the
    # leakage audit in drug/outputs/splits/scaffold_split.json alongside any unseen-compound number.
    scaffold_split_path: str = "drug/outputs/splits/scaffold_split.json"
    drug_fold: int = 0
    n_cell_folds: int = 5
    cell_fold: int = 0
    val_frac: float = 0.05
    seed: int = 0
