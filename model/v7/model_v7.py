# -*- coding: utf-8 -*-
"""
LincsV7 = v6's verified-correct pieces + the modern transformer recipe + priors that are FORCED to be used.

v6 measured every one of its cell-context modules at ~0 on unseen cells (chromatin -0.0001, pathway +0.0003,
pathway-gate +0.0001) while being live in every case, and tied the `Meandrug` baseline overall. v7 therefore
changes three things that the v6 measurement actually implicates, rather than tuning:

  1. THE PRIORS ARE SUPERVISED (AuxHeads + UncertaintyWeighting). Nothing in v6 asked the pathway nodes or
     the chromatin encoder to mean anything, so gradient descent ignored them. Each auxiliary head reads
     ONLY its own branch, against a target derived from the MEASURED response.
  2. THE BRANCHES CANNOT BE BYPASSED FOR FREE (StochasticDepth on the residual branches). v6's pathway
     layer is a side branch [4.13]; dropping the skip path some of the time removes the option to ignore it.
  3. STRING IS ACTUALLY USED (PPIMessagePassing). 12,665 edges that v6 never loaded, consumed as a sparse
     graph convolution rather than v5's soft attention bias, which was measured not to steer [4.6].

Kept unchanged because they are measured-good: the signed additive chromatin head [2.1/2.2/2.2a], atom->gene
cross-attention for accuracy with no interpretability claim [4.3/4.1a], late modality fusion, reliability
weighting, and the differential target.
"""
import os, sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v6"))
sys.path.insert(0, os.path.dirname(HERE))

from modules import DoseTimeFiLM, PathwayBottleneck
from modules_v6 import BaselineEncoder, ChromatinEncoder, ModalityFusion, GeneHead
from modules_v7 import (RMSNorm, SwiGLU, QKNormAttention, CrossAttention, StochasticDepth,
                        PPIMessagePassing, AuxHeads, UncertaintyWeighting)


class GeneBlock(nn.Module):
    """pre-norm gene<->gene block: RMSNorm + QK-norm attention + SwiGLU, each residual stochastic-depthed."""
    def __init__(self, cfg, p_drop=0.0):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = QKNormAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.sd1, self.sd2 = StochasticDepth(p_drop), StochasticDepth(p_drop)

    def forward(self, h):
        h = h + self.sd1(self.attn(self.n1(h)))
        return h + self.sd2(self.ff(self.n2(h)))


class PerturbBlock(nn.Module):
    def __init__(self, cfg, p_drop=0.0):
        super().__init__()
        self.nc = RMSNorm(cfg.d_model)
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.sdc = StochasticDepth(p_drop)
        self.gene = GeneBlock(cfg, p_drop)

    def forward(self, h, D, key_mask):
        h = h + self.sdc(self.cross(self.nc(h), D, key_mask))
        return self.gene(h)


class LincsV7(nn.Module):
    def __init__(self, cfg, M_reactome, ppi=None):
        super().__init__()
        self.cfg = cfg
        G, d = cfg.n_genes, cfg.d_model

        self.gene_emb = nn.Parameter(torch.randn(G, d) * 0.02)
        self.enc_base = BaselineEncoder(d)
        self.enc_epi = ChromatinEncoder(d, cfg.d_epi)
        self.fuse = ModalityFusion(d)
        self.film = DoseTimeFiLM(d, d_extra=cfg.d_cell_ctx)

        self.w_a = nn.Linear(cfg.d_atom, d); self.w_u = nn.Linear(cfg.d_global, d)
        self.type_atom = nn.Parameter(torch.zeros(d)); self.type_drug = nn.Parameter(torch.zeros(d))
        self.ln_atom, self.ln_u = RMSNorm(d), RMSNorm(d)

        # linearly increasing drop rate with depth is the standard stochastic-depth schedule
        L = cfg.l_base + cfg.l_perturb
        rates = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        self.base = nn.ModuleList([GeneBlock(cfg, rates[i]) for i in range(cfg.l_base)])
        self.perturb = nn.ModuleList([PerturbBlock(cfg, rates[cfg.l_base + i]) for i in range(cfg.l_perturb)])

        self.ppi = PPIMessagePassing(ppi, d, cfg.dropout) if (cfg.use_ppi and ppi is not None) else None
        self.sd_ppi = StochasticDepth(cfg.stoch_depth)
        self.pathway = PathwayBottleneck(M_reactome, d, cfg.d_pathway, cfg.d_epi,
                                         gate_with_epi=cfg.pathway_epi_gate)
        self.sd_path = StochasticDepth(cfg.stoch_depth)
        self.head = GeneHead(d, cfg.d_epi, cfg.d_epi_hidden, G)
        self.aux = AuxHeads(cfg.d_pathway, d) if cfg.use_aux else None
        self.task_weights = UncertaintyWeighting(3) if cfg.use_aux else None

    def forward(self, batch, return_aux=False, return_interp=False):
        X, E, r = batch["X"], batch["E"], batch["r"]
        atoms, atom_valid = batch["atoms"], batch["atom_mask"]
        u, dose, time = batch["u_feats"], batch["dose"], batch["time"]
        B = X.shape[0]

        h_epi = self.enc_epi(E, r)                                  # kept: the aux head reads THIS only
        h = self.gene_emb.unsqueeze(0) + self.fuse(self.enc_base(X), h_epi)[0]
        h = self.film(h, dose, time, batch.get("cell_ctx"))

        D = torch.cat([(self.ln_u(self.w_u(u)) + self.type_drug).unsqueeze(1),
                       self.ln_atom(self.w_a(atoms)) + self.type_atom], dim=1)
        key_mask = ~torch.cat([torch.ones(B, 1, dtype=torch.bool, device=X.device), atom_valid], 1)

        for blk in self.base:
            h = blk(h)
        if self.ppi is not None:                                    # STRING message passing [NEW in v7]
            h = h + self.sd_ppi(self.ppi(h))

        delta, pathways = self.pathway(h, E, return_pathways=True)
        h = h + self.sd_path(delta)

        for blk in self.perturb:
            h = blk(h, D, key_mask)
        yhat, epi_contrib = self.head(h, E, r)

        if return_aux or return_interp:
            aux = {"pathway_activations": pathways, "epi_contrib": epi_contrib}
            if self.aux is not None:
                aux["pathway_pred"], aux["epi_pred"] = self.aux(pathways, h_epi)
            return yhat, aux
        return yhat


def aux_targets(Y, M_norm):
    """Targets for deep supervision, both derived from the MEASURED response (never a branch input).
      pathway [B,P] : mean|Y| over each pathway's member genes -- "is this pathway moving?"
      epi     [B,G] : |Y| per gene                             -- "which genes CAN move in this cell?"
    """
    absY = Y.abs()
    return torch.einsum("pg,bg->bp", M_norm, absY), absY
