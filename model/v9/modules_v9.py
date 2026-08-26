# -*- coding: utf-8 -*-
"""
v9 modules. Each exists because a measurement asked for it; the docstrings say which.
Blocks that were already measured-good (RMSNorm, SwiGLU, QK-norm attention, cross-attention, stochastic
depth, PPI message passing) are imported from v7 rather than reimplemented.
"""
import os, sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v7'))
from modules_v7 import RMSNorm, SwiGLU, QKNormAttention, CrossAttention, StochasticDepth  # noqa: F401
from modules_v7 import PPIMessagePassing                                                   # noqa: F401


class BinnedExpression(nn.Module):
    """Quantise expression to n_bins levels and EMBED, instead of a linear layer on the raw scalar.

    This is the one architectural element V9_HANDOFF §C identifies as a concrete difference from the field
    (`n_bins: 128`, `exp_vocab_size`), and the handoff gates it on an A/B before a full train, which is
    what model/v9/ab_encoder.py runs.

    Bin edges are fitted on TRAINING ROWS ONLY and stored in the module, so evaluation cannot see the test
    distribution through the quantiser. `global` bins share one set of edges across all 978 genes, which
    keeps values comparable across genes (XPert's setting); `per_gene` gives each gene its own edges, which
    equalises dynamic range but destroys that comparability.
    """

    def __init__(self, d_model, n_bins=128, n_genes=978, mode='global'):
        super().__init__()
        self.n_bins, self.mode, self.n_genes = n_bins, mode, n_genes
        self.emb = nn.Embedding(n_bins, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        edges = torch.zeros(1 if mode == 'global' else n_genes, n_bins - 1)
        self.register_buffer('edges', edges)
        self.register_buffer('fitted', torch.zeros(1))

    @torch.no_grad()
    def fit(self, X):
        """X: [N, G] training values. Quantile edges, so bins carry equal mass rather than equal width."""
        q = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        if self.mode == 'global':
            e = np.percentile(np.asarray(X, np.float32).ravel(), q)[None, :]
        else:
            e = np.percentile(np.asarray(X, np.float32), q, axis=0).T
        self.edges.copy_(torch.as_tensor(np.ascontiguousarray(e), dtype=self.edges.dtype))
        self.fitted.fill_(1.0)
        return self

    def bucket(self, x):
        """x: [B, G] -> [B, G] long bin indices."""
        if self.mode == 'global':
            return torch.bucketize(x, self.edges[0].to(x.dtype), right=False).clamp_(0, self.n_bins - 1)
        idx = torch.zeros_like(x, dtype=torch.long)
        for g in range(x.shape[1]):                      # per-gene edges; only used in the A/B arm
            idx[:, g] = torch.bucketize(x[:, g], self.edges[g].to(x.dtype), right=False)
        return idx.clamp_(0, self.n_bins - 1)

    def forward(self, x):
        # HARD GUARD. With unfitted edges every value falls in bin 0, so the expression input becomes a
        # silent no-op: the model trains, converges, and reports numbers while ignoring expression
        # entirely. test_v9.py caught exactly that (|dY|max 0.0000 for the matched control). Refuse
        # instead, in the spirit of train_v7_gpu.check_inputs().
        if float(self.fitted) != 1.0:
            raise RuntimeError(
                'BinnedExpression used before fit(): every value would land in bin 0 and the expression '
                'input would be silently ignored. Call model.fit_bins(X_train) with TRAINING rows only.')
        return self.emb(self.bucket(x))                  # [B, G, d]


class RawExpression(nn.Module):
    """The v3-v7 encoding, kept as the A/B's control arm: one linear layer on the continuous value."""

    def __init__(self, d_model, n_in=1):
        super().__init__()
        self.lin = nn.Linear(n_in, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.norm(self.lin(x.unsqueeze(-1)))      # [B, G, d]


class GeneRepresentation(nn.Module):
    """The gene token, before any expression value is added.

    learned embedding + pretrained STRING vector + chromatin, ALL SUMMED. The handoff's §D.4 point is that
    chromatin must enter the same way the PPI vector does -- as part of what a gene IS -- rather than as a
    parallel branch the model can route around, which is exactly what v6/v7 measured it doing
    (-0.0001 / +0.0061 / -0.0001, while a 16-dim lineage one-hot beat it at +0.0215).
    """

    def __init__(self, cfg, gene_vectors=None):
        super().__init__()
        G, d = cfg.n_genes, cfg.d_model
        self.emb = nn.Parameter(torch.randn(G, d) * 0.02)
        self.use_vec = cfg.use_gene_vectors and gene_vectors is not None
        if self.use_vec:
            v = torch.as_tensor(np.asarray(gene_vectors, np.float32))
            assert v.shape[0] == G, f'gene vectors are [{v.shape[0]}, .], expected {G} rows'
            self.vec = nn.Parameter(v, requires_grad=not cfg.freeze_gene_vectors)
            self.vec_proj = nn.Linear(v.shape[1], d)
        self.use_epi = cfg.epi_as_gene_embedding
        if self.use_epi:
            # per-gene, per-cell chromatin -> a vector added to the gene token. Reliability multiplies it,
            # so a cell with no ChIP for a mark contributes nothing rather than a zero that means "closed".
            self.epi_proj = nn.Sequential(nn.Linear(cfg.d_epi, cfg.d_epi_hidden), nn.GELU(),
                                          nn.Linear(cfg.d_epi_hidden, d))
        self.norm = RMSNorm(d)

    def forward(self, E=None, r=None):
        h = self.emb.unsqueeze(0)
        if self.use_vec:
            h = h + self.vec_proj(self.vec).unsqueeze(0)
        if self.use_epi and E is not None:
            e = self.epi_proj(E)
            if r is not None:
                e = e * r.unsqueeze(-1)
            h = h + e
        return self.norm(h)


class ControlEncoder(nn.Module):
    """A SEPARATE stack over the control profile (XPert's `ctl_structure`, SA x4).

    v9 is the first version where the control is a real input rather than a CCLE proxy, so it gets its own
    encoder instead of being concatenated into the treated stream. Both control views are encoded: the
    plate-matched control and the per-cell aggregate, which RESULTS 27.5 shows help in opposite regimes.
    """

    def __init__(self, cfg, expr_encoder):
        super().__init__()
        self.expr = expr_encoder
        self.blocks = nn.ModuleList([_GeneBlock(cfg, cfg.stoch_depth * i / max(cfg.l_control - 1, 1))
                                     for i in range(cfg.l_control)])
        self.out = RMSNorm(cfg.d_model)

    def forward(self, x, gene_tok):
        h = gene_tok + self.expr(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


class _GeneBlock(nn.Module):
    def __init__(self, cfg, p_drop=0.0):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = QKNormAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.sd1, self.sd2 = StochasticDepth(p_drop), StochasticDepth(p_drop)

    def forward(self, h):
        h = h + self.sd1(self.attn(self.n1(h)))
        return h + self.sd2(self.ff(self.n2(h)))


class NamedPathwayReadout(nn.Module):
    """800 NAMED nodes (Reactome + GO:BP), one row of M per named term, verified against pathway_info at
    load. Nothing here is a latent bottleneck: every unit has a curated name and a member gene list.

    v6/v7 measured the pathway LAYER at +0.0003 -- a true null with |dY|max 0.69, i.e. live but worthless
    for accuracy. It is retained for INTERPRETABILITY, which is the project's actual deliverable, and it is
    reported with its own measured chance level (an untrained model scores 0.218 against a permutation null
    of 0.229), never against an assumed 0.5.
    """

    def __init__(self, M, d_model, d_pathway):
        super().__init__()
        M = torch.as_tensor(np.asarray(M, np.float32))
        self.register_buffer('M', M)                                        # [P, G]
        self.register_buffer('M_norm', M / M.sum(1, keepdim=True).clamp(min=1))
        self.proj = nn.Linear(d_model, d_pathway)
        self.act = nn.GELU()
        self.back = nn.Linear(d_pathway, d_model)

    def forward(self, h):
        p = self.act(torch.einsum('pg,bgd->bpd', self.M_norm, self.proj(h)))   # [B, P, d_pathway]
        return torch.einsum('pg,bpd->bgd', self.M, self.back(p)), p


class SignedChromatinHead(nn.Module):
    """The signed ADDITIVE chromatin head, unchanged from v6/v7 -- the one chromatin mechanism that ever
    survived a test. It contributes an additive, per-gene, signed term whose magnitude is reported so a
    true null stays distinguishable from a dead branch."""

    def __init__(self, d_model, d_epi, d_hidden, n_genes):
        super().__init__()
        self.gene = nn.Linear(d_model, 1)
        self.epi = nn.Sequential(nn.Linear(d_epi, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, h, E, r):
        base = self.gene(h).squeeze(-1)
        contrib = self.epi(E).squeeze(-1) * (r if r is not None else 1.0)
        return base + contrib, contrib


class MultiTaskHeads(nn.Module):
    """Absolute Level-3 expression, the delta, and the Level-5 z-score.

    XPert trains MSE(absolute) + MSE(control) + MSE(delta) + PCC(delta). The Level-5 head is ours and is
    the reason v9 stays comparable with v3-v7: it is the only target every previous number used.

    THE ABSOLUTE HEAD IS ANCHORED ON THE CONTROL. Predicting absolute expression from scratch is mostly
    predicting the cell's baseline, which is why the absolute convention self-agrees at 0.93 and "copy the
    control" scores 0.9200 [RESULTS 23, 27.2]. Emitting abs = ctl + delta by construction makes the model
    spend its capacity on the delta and makes the absolute number honest about what it contains.
    """

    def __init__(self, cfg):
        super().__init__()
        self.delta = SignedChromatinHead(cfg.d_model, cfg.d_epi, cfg.d_epi_hidden, cfg.n_genes) \
            if cfg.keep_epi_head else nn.Linear(cfg.d_model, 1)
        self.keep_epi_head = cfg.keep_epi_head
        self.l5 = nn.Linear(cfg.d_model, 1)

    def forward(self, h, E, r, x_ctl):
        if self.keep_epi_head:
            d, epi_contrib = self.delta(h, E, r)
        else:
            d, epi_contrib = self.delta(h).squeeze(-1), torch.zeros_like(h[..., 0])
        return {'delta': d, 'abs': x_ctl + d, 'l5': self.l5(h).squeeze(-1)}, epi_contrib
