# -*- coding: utf-8 -*-
"""
v9 configuration. Every non-default value carries the measurement that put it there; where a choice is
NOT yet measured it says so rather than implying it is settled.

Design: V9_HANDOFF.md §D, amended by RESULTS §27 where §27 measured the handoff wrong.
"""
import os, sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
from config_v6 import V6DataConfig                    # noqa: F401  (re-exported; v9 extends it)


@dataclass
class V9DataConfig(V6DataConfig):
    """v9 replaces the data substrate. Paths are relative to root unless absolute."""
    # --- the Level-3 substrate, signature-aligned, 99.65 % coverage [RESULTS 27.1] ---
    l3_dir: str = 'phase2_assembly/outputs/level3_sig'
    x_trt_path: str = 'phase2_assembly/outputs/level3_sig/X_trt_l3.npy'
    x_ctl_path: str = 'phase2_assembly/outputs/level3_sig/X_ctl_l3.npy'
    l3_covered_path: str = 'phase2_assembly/outputs/level3_sig/l3_covered.npy'
    l3_yrow_path: str = 'phase2_assembly/outputs/level3_sig/l3_yrow.npy'

    # --- v9 priors: full proteome, HGNC-current symbols [RESULTS 27.3] ---
    m_pathway_path: str = 'network/outputs/v9/M_pathway_v9.npy'
    pathway_info_v9_path: str = 'network/outputs/v9/pathway_info_v9.tsv'
    ppi_v9_path: str = 'network/outputs/v9/STRING_adj_978_v9.npy'
    gene_vec_path: str = 'network/outputs/v9/gene_vectors_978.npy'
    landmark_symbols_path: str = 'network/outputs/v9/landmark_symbols_v9.tsv'

    # --- inputs ---
    # CCLE is REDUNDANT, not harmful: `both - matched` lands in [-0.0104, +0.0003] on every target and
    # split, and a per-cell mean of L1000 DMSO controls dominates it (identical on seen cells, +0.037 /
    # +0.027 better on unseen cells) [RESULTS 27.4]. Default off; flip to True to re-run the ablation.
    use_ccle: bool = False
    # Both control views, because they help in OPPOSITE regimes [RESULTS 27.5]: the matched control is
    # worth +0.103 over a per-cell mean on unseen COMPOUNDS, but on unseen CELLS an independent plate
    # control is 0.08 WORSE than the per-cell mean and its apparent gain is noise cancellation.
    use_matched_ctl: bool = True
    use_cell_ctl: bool = True

    # --- target ---
    # Multi-task, as XPert does. The Level-5 head is kept so every number in this project's history stays
    # comparable; §27.2 shows the Level-3 delta is NOT the noisier target on our evaluation stratum
    # (0.5283 self-agreement vs Level-5 MODZ 0.509-0.619), which is why this migration is affordable.
    predict_abs: bool = True
    predict_delta: bool = True
    predict_l5: bool = True

    cache_in_ram: bool = False
    eval_min_strength: float = 1.0


@dataclass
class V9Config:
    n_genes: int = 978
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    l_control: int = 2          # the separate control encoder (XPert's ctl_structure is SA x4)
    l_base: int = 2
    l_perturb: int = 4
    max_atoms: int = 96
    dropout: float = 0.1
    d_atom: int = 512
    d_epi: int = 3
    d_global: int = 512 + 20 + 2048
    d_cell_ctx: int = 16
    d_pathway: int = 32         # 800 named nodes now (was 360), so the per-node width comes down
    n_pathways: int = 800
    d_epi_hidden: int = 16
    huber_delta: float = 1.0
    stoch_depth: float = 0.1

    # --- expression encoding: THE v9 A/B [handoff §D.2] ---
    # 'raw'    : a linear layer on the continuous value, as v3-v7 did
    # 'binned' : quantise to n_bins levels and embed, as XPert does (n_bins: 128)
    # Gated: model/v9/ab_encoder.py must show it before a full train commits to it.
    # RESULTS 47: XPert's crossEncoder runs drug_SA over the drug tokens INSIDE every cross-encoder block,
    # so gene queries attend over a molecule whose atoms have been mutually contextualised. v9 built
    # D = [global; linear(atoms)] once, outside the block loop, and reused it -- our atoms never saw each
    # other. That is the leading explanation for [RESULTS 37] "removing atom tokens IMPROVES accuracy"
    # (-0.007 / -0.025 / -0.022): uncontextualised per-atom vectors are noise.
    # DEFAULT FALSE so the A/B changes exactly one thing and every v9 checkpoint still loads.
    drug_self_attn: bool = False
    expr_encoder: str = 'binned'
    n_bins: int = 128
    bin_mode: str = 'global'    # 'global' keeps values comparable ACROSS genes (XPert's setting);
                                # 'per_gene' equalises each gene's dynamic range but destroys that.

    # --- priors in the gene representation ---
    use_gene_vectors: bool = True     # pretrained STRING vectors, link-prediction AUC 0.9198 [27.3]
    freeze_gene_vectors: bool = False
    d_gene_vec: int = 128
    use_ppi: bool = True              # message passing on the induced 978x978 graph
    # chromatin as a PER-GENE EMBEDDING summed into the gene representation [handoff §D.4] -- the natural
    # home given that the PPI vector enters the same way. v6/v7 ran it as a parallel encoder the model
    # could ignore, and it did: -0.0001 / +0.0061 / -0.0001.
    epi_as_gene_embedding: bool = True
    keep_epi_head: bool = True        # the SIGNED ADDITIVE chromatin head: the one chromatin mechanism
                                      # that ever survived a test [2.1/2.2/2.2a]. Not removed.

    # --- auxiliary supervision ---
    # v7 measured learned (Kendall) uncertainty weighting sending ~90 % of the gradient to the auxiliary
    # tasks, and --no_aux then beat it 6/6 [RESULTS 22]. So: small FIXED lambdas, never learned weights.
    use_aux: bool = True
    aux_pathway_w: float = 0.05
    aux_epi_w: float = 0.05
    # abs, delta, l5, pcc(delta).
    # THE ABS TERM IS EXACTLY THE DELTA TERM. Because MultiTaskHeads emits abs = x_ctl + delta and the
    # dataset builds y_delta = y_abs - x_ctl from the same x_ctl, the two residuals are identical element
    # for element -- verified numerically (both 1.5578 on the same batch). So the EFFECTIVE delta weight
    # here is 2.0, not 1.0, and w_abs is not a second task.
    # Left as (1.0, 1.0, ...) rather than (0.0, 2.0, ...) because the two are the same objective and the
    # seed runs were already queued against this tuple; changing it would alter nothing except which seeds
    # are comparable with which. If the absolute head is ever UN-anchored, these weights stop being
    # equivalent and test_v9.py's identity check is what will say so.
    task_w: tuple = (1.0, 1.0, 0.3, 0.5)

    # --- §85.7 candidate flags (all default-off / default-today) ---
    no_atoms: bool = False          # C1: drug sequence is [global] only
    # l_control: int = 2            # C2: already above; --l_control 0 removes all gene-blocks
    listnet_w: float = 0.0          # C3: symmetric ListNet weight (0 = off)
    deg_adapt_k: int = 0            # C4: adaptive DE weighting, K genes (0 = off)
    sign_head_w: float = 0.0        # C6: sign-prediction BCE weight (0 = off)
    post_pathway: bool = False      # C8b: second NamedPathwayReadout after last perturb block


@dataclass
class V9TrainConfig:
    epochs: int = 12
    batch: int = 48
    lr: float = 4e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'
    schedule: str = 'wsd'
    warmup_frac: float = 0.03
    decay_frac: float = 0.20
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    budget_h: float = 8.5
    fold: int = 0
    workers: int = 2
    seed: int = 0
