# -*- coding: utf-8 -*-
"""
LincsV6 -- rebuilt after v5's interpretability claims were falsified by our own tests.
Full rationale, data provenance and the evidence for every choice: ARCHITECTURE.md (same directory).

Two structural corrections vs v5:
  1. PATHWAY PRIOR IS A HARD MASK, NOT A SOFT BIAS (P-NET / DCell / DrugCell).
     v5 biased attention with lambda*log1p(A_copathway); prior heads landed on-support only 1.1x random,
     and A_copathway is the gene-gene COLLAPSE of the real membership matrix M_reactome [360,978] we
     already had. v6 routes information through 360 NAMED Reactome nodes, so the readout is a pathway
     activation by construction.
  2. LATE MODALITY INTEGRATION, NOT EARLY FUSION (MOLI / DeepCDR).
     v5 summed projected X_base and E into one gene token; chromatin entangled with baseline (corr 0.74).
     v6 encodes each modality separately, then fuses -- keeping them independently ablatable.

Retained because it survived every test: the SIGNED additive chromatin head (+0.089 R^2, correct sign,
robust across all four baseline-expression quartiles).
Retained for accuracy only, with NO interpretability claim: atom->gene cross-attention (our own test put
known targets at median rank percentile 0.560 -- worse than chance).
"""
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import (DoseTimeFiLM, MultiHeadCrossAttention, BiasedSelfAttention,
                     FeedForward, PathwayBottleneck)
from modules_v6 import BaselineEncoder, ChromatinEncoder, ModalityFusion, GeneHead


class GeneBlock(nn.Module):
    """gene<->gene self-attention + FFN. The prior bias is OPTIONAL here (default off): v6 gets its
    biological structure from the masked PathwayBottleneck instead, because the soft bias was measured
    not to steer (1.1x random on-support). Kept switchable so the two can be compared directly."""
    def __init__(self, cfg, n_priors):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model); self.n2 = nn.LayerNorm(cfg.d_model)
        self.attn = BiasedSelfAttention(cfg.d_model, cfg.n_heads, n_priors,
                                        cfg.n_prior_heads if cfg.use_prior_bias else 0,
                                        dropout=cfg.dropout)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, h, priors, support, need_weights=False):
        a, attn = self.attn(self.n1(h), priors, support, need_weights=need_weights)
        h = h + a
        return h + self.ff(self.n2(h)), attn


class PerturbBlock(nn.Module):
    """atom->gene cross-attention, then gene<->gene, then FFN."""
    def __init__(self, cfg, n_priors):
        super().__init__()
        self.n_ca = nn.LayerNorm(cfg.d_model)
        self.cross = MultiHeadCrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.gene = GeneBlock(cfg, n_priors)

    def forward(self, h, D, key_mask, priors, support, need_weights=False):
        ca, a_drug = self.cross(self.n_ca(h), D, key_mask, need_weights=need_weights)
        h = h + ca
        h, a_gene = self.gene(h, priors, support, need_weights=need_weights)
        return h, a_drug, a_gene


class LincsV6(nn.Module):
    def __init__(self, cfg, M_reactome, cop=None, ppi=None):
        """M_reactome: [P,G] binary Reactome membership -- the biological backbone.
        cop/ppi: optional [G,G] priors, only used if cfg.use_prior_bias (default False)."""
        super().__init__()
        self.cfg = cfg
        G, d = cfg.n_genes, cfg.d_model

        if cfg.use_prior_bias and cop is not None:
            tf = lambda M: torch.log1p(torch.as_tensor(M, dtype=torch.float32).clamp(min=0))
            self.register_buffer("priors", torch.stack([tf(cop), tf(ppi)], 0))
            self.register_buffer("support", (torch.as_tensor(cop) > 0) | (torch.as_tensor(ppi) > 0))
            n_priors = 2
        else:
            self.register_buffer("priors", torch.zeros(1, 1, 1)); self.register_buffer("support", torch.zeros(1, 1, dtype=torch.bool))
            n_priors = 1

        self.gene_emb = nn.Parameter(torch.randn(G, d) * 0.02)
        self.enc_base = BaselineEncoder(d)
        self.enc_epi = ChromatinEncoder(d, cfg.d_epi)
        self.fuse = ModalityFusion(d)
        self.film = DoseTimeFiLM(d, d_extra=cfg.d_cell_ctx)

        self.w_a = nn.Linear(cfg.d_atom, d); self.w_u = nn.Linear(cfg.d_global, d)
        self.type_atom = nn.Parameter(torch.zeros(d)); self.type_drug = nn.Parameter(torch.zeros(d))
        self.ln_atom = nn.LayerNorm(d); self.ln_u = nn.LayerNorm(d)

        self.base = nn.ModuleList([GeneBlock(cfg, n_priors) for _ in range(cfg.l_base)])
        self.pathway = PathwayBottleneck(M_reactome, d, cfg.d_pathway, cfg.d_epi,
                                         gate_with_epi=cfg.pathway_epi_gate)
        self.perturb = nn.ModuleList([PerturbBlock(cfg, n_priors) for _ in range(cfg.l_perturb)])
        self.head = GeneHead(d, cfg.d_epi, cfg.d_epi_hidden, G)

    def forward(self, batch, return_attn=False, return_interp=False):
        """return_interp gives the INTERPRETABLE readouts that v6 is built to produce:
        pathway_activations [B,360,dp] (named Reactome nodes) and the signed chromatin contribution."""
        X, E, r = batch["X"], batch["E"], batch["r"]
        atoms, atom_valid = batch["atoms"], batch["atom_mask"]
        u, dose, time = batch["u_feats"], batch["dose"], batch["time"]
        B = X.shape[0]
        dev = X.device.type
        with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
            # (1) LATE modality integration -- each encoder keeps its own normalisation
            h = self.gene_emb.unsqueeze(0) + self.fuse(self.enc_base(X), self.enc_epi(E, r))[0]
            h = self.film(h, dose, time, batch.get("cell_ctx"))

            # (2) drug tokens: [pooled global ; per-atom]
            D = torch.cat([(self.ln_u(self.w_u(u)) + self.type_drug).unsqueeze(1),
                           self.ln_atom(self.w_a(atoms)) + self.type_atom], dim=1)
            key_mask = ~torch.cat([torch.ones(B, 1, dtype=torch.bool, device=X.device), atom_valid], 1)

            # (3) drug-free context encoder
            a_gene = None
            for blk in self.base:
                h, a_gene = blk(h, self.priors, self.support, need_weights=return_attn)

            # (4) BIOLOGICAL BOTTLENECK: genes -> 360 named Reactome pathways -> member genes only
            delta, pathways = self.pathway(h, E, return_pathways=(return_attn or return_interp))
            h = h + delta

            # (5) perturbation encoder
            a_drug = None
            for blk in self.perturb:
                h, a_drug, a_gene = blk(h, D, key_mask, self.priors, self.support, need_weights=return_attn)

            # (6) per-gene head + signed chromatin term
            yhat, epi_contrib = self.head(h, E, r)

        if return_attn or return_interp:
            return yhat, {"pathway_activations": pathways, "epi_contrib": epi_contrib,
                          "alpha_drug": a_drug, "alpha_gene": a_gene}
        return yhat
