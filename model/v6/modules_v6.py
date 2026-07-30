# -*- coding: utf-8 -*-
"""
v6 building blocks. Design rationale and the evidence for each choice: ARCHITECTURE.md (same directory).

New in v6 (both are corrections of measured v5 failures, not speculative changes):
  BaselineEncoder / ChromatinEncoder / ModalityFusion
      LATE integration, per MOLI (Bioinformatics 2019) and DeepCDR. v5 summed projected X_base and E
      straight into one gene token (early fusion); chromatin then entangled with baseline (learned
      conductance corr 0.74 with X_base) and contributed ~0 once scale was controlled.
  PathwayBottleneck  (imported from ../modules.py)
      P-NET-style hard connectivity MASK over 360 named Reactome pathways, replacing v5's soft
      lambda*log1p(prior) attention bias, which measurement showed steers attention only 1.1x random.
"""
import torch
import torch.nn as nn


class BaselineEncoder(nn.Module):
    """Per-gene encoder for CCLE baseline expression, with its own normalisation.

    Separate from the chromatin encoder on purpose (late integration): the two modalities have different
    distributions and different missingness, and keeping them apart is what makes each independently
    ABLATABLE -- v5's early fusion is why chromatin's contribution could not be cleanly separated from
    baseline expression."""
    def __init__(self, d_model, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x:[B,G] gene-level baseline expression -> [B,G,d]
        return self.norm(self.net(x.unsqueeze(-1)))


class ChromatinEncoder(nn.Module):
    """Per-gene encoder for the 3 chromatin tracks (ATAC / H3K27ac / H3K27me3), with its own normalisation.

    Takes the availability indicator explicitly, so "this mark was never measured in this cell" is a
    distinct input state from "this mark is zero". 45/83 cells have >=1 mark and we deliberately do NOT
    impute the rest -- cells with no chromatin data must fall back to a neutral contribution, which the
    reliability mask enforces (output is exactly 0 where r == 0)."""
    def __init__(self, d_model, d_epi=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_epi + 1, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, E, r):
        # E:[B,G,3]  r:[B,G] reliability -> [B,G,d], exactly 0 where the cell has no chromatin data
        avail = (r > 0).to(E.dtype).unsqueeze(-1)
        h = self.norm(self.net(torch.cat([E, avail], dim=-1)))
        return h * avail


class ModalityFusion(nn.Module):
    """Fuse the separately-encoded modalities into gene tokens (the 'late integration' join).

    A learned gate per modality rather than a plain sum, so the model can down-weight a modality instead of
    being forced to add it -- and so we can read off how much each is used. Chromatin starts at zero
    contribution (zero-init gate) and must EARN its way in, which is the same discipline that made the v5
    signed chromatin head trustworthy."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        self.epi_gate = nn.Parameter(torch.zeros(1))       # sigmoid(0)=0.5 scaled below -> starts small
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_base, h_epi):
        g = torch.sigmoid(self.epi_gate)                   # scalar in (0,1), inspectable
        fused = torch.cat([h_base, g * h_epi], dim=-1)
        return self.norm(self.proj(fused)), float(g.detach()) if not self.training else None


class GeneHead(nn.Module):
    """Per-gene output head + a SIGNED additive chromatin term.

    The signed term is kept from v5 because it is the ONE mechanism that survived every test: DeltaR^2
    +0.089 in-distribution, mechanistically correct sign (low activation / high Polycomb => larger
    response), and robust to the floor-effect confound across ALL four baseline-expression quartiles.
    A multiplicative gate cannot express a signed shift, which is why this is additive."""
    def __init__(self, d_model, d_epi=3, hidden=16, n_genes=978):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.epi_head = nn.Sequential(nn.Linear(d_epi, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.epi_head[-1].weight); nn.init.zeros_(self.epi_head[-1].bias)

    def forward(self, h, E, r):
        y = self.head(self.norm(h)).squeeze(-1) + self.gene_bias
        epi_contrib = self.epi_head(E).squeeze(-1) * r      # reliability-masked, signed
        return y + epi_contrib, epi_contrib
