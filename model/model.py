# -*- coding: utf-8 -*-
"""
LincsCrossAttn -- deterministic, fully-attributable predictor of the 978-gene differential response.
Flow (model/MODEL_MATH.md): gate -> gene tokens (+FiLM) -> base/context encoder (gene<->gene, drug-free)
-> perturbation encoder (atom->gene + gene<->gene) -> per-gene head. Predicts Y directly; X_base is context.
forward() never sees Y (no target leakage). Set return_attn=True to get alpha^drug / alpha^gene / gate.
"""
import torch
import torch.nn as nn

from modules import (EpiGate, DoseTimeFiLM, MultiHeadCrossAttention,
                     BiasedSelfAttention, FeedForward, PathwayConductance)


class BaseLayer(nn.Module):
    """Pre-norm gene<->gene (biased) self-attention + FFN. The drug-free context encoder."""
    def __init__(self, cfg, n_priors):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model); self.n2 = nn.LayerNorm(cfg.d_model)
        self.attn = BiasedSelfAttention(cfg.d_model, cfg.n_heads, n_priors, cfg.n_prior_heads,
                                        dropout=cfg.dropout)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, h, priors, support, need_weights=False, cond=None):
        a, attn = self.attn(self.n1(h), priors, support, need_weights=need_weights)
        if cond is not None:                      # cell-conditional pathway conductance [B,G]
            a = a * cond.unsqueeze(-1)
        h = h + a
        h = h + self.ff(self.n2(h))
        return h, attn


class PerturbLayer(nn.Module):
    """Pre-norm atom->gene cross-attn, then gene<->gene biased self-attn, then FFN."""
    def __init__(self, cfg, n_priors):
        super().__init__()
        self.n_ca = nn.LayerNorm(cfg.d_model)
        self.cross = MultiHeadCrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.n_sa = nn.LayerNorm(cfg.d_model)
        self.attn = BiasedSelfAttention(cfg.d_model, cfg.n_heads, n_priors, cfg.n_prior_heads,
                                        dropout=cfg.dropout)
        self.n_ff = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, h, D, key_mask, priors, support, need_weights=False, cond=None):
        ca, a_drug = self.cross(self.n_ca(h), D, key_mask, need_weights=need_weights)
        h = h + ca
        sa, a_gene = self.attn(self.n_sa(h), priors, support, need_weights=need_weights)
        if cond is not None:                      # cell-conditional pathway conductance [B,G]
            sa = sa * cond.unsqueeze(-1)
        h = h + sa
        h = h + self.ff(self.n_ff(h))
        return h, a_drug, a_gene, ca      # ca [B,G,d] = atom->gene cross-attn contribution (for DTI attribution)


class LincsCrossAttn(nn.Module):
    def __init__(self, cfg, cop, ppi):
        """cop, ppi: [G,G] nonneg prior matrices (numpy/tensor)."""
        super().__init__()
        self.cfg = cfg
        G, d = cfg.n_genes, cfg.d_model

        # priors -> transformed, nonneg buffer [P,G,G] + support mask
        cop = torch.as_tensor(cop, dtype=torch.float32)
        ppi = torch.as_tensor(ppi, dtype=torch.float32)
        tf = (lambda M: torch.log1p(M.clamp(min=0))) if cfg.prior_transform == "log1p" else (lambda M: M.clamp(min=0))
        self.register_buffer("priors", torch.stack([tf(cop), tf(ppi)], 0))     # [2,G,G]
        self.register_buffer("support", (cop > 0) | (ppi > 0))                  # [G,G] bool
        n_priors = self.priors.shape[0]

        # embeddings / projections
        self.gene_emb = nn.Parameter(torch.randn(G, d) * 0.02)
        self.w_x = nn.Linear(1, d)                        # gated baseline scalar
        self.w_e = nn.Linear(cfg.d_epi + 1, d)            # [E(3); availability]
        self.w_a = nn.Linear(cfg.d_atom, d)               # atom token
        self.w_u = nn.Linear(cfg.d_global, d)             # global drug token
        self.type_atom = nn.Parameter(torch.zeros(d))
        self.type_drug = nn.Parameter(torch.zeros(d))
        self.ln_atom = nn.LayerNorm(d); self.ln_u = nn.LayerNorm(d)

        self.gate = EpiGate(cfg.d_epi, cfg.d_epi_hidden)
        self.film = DoseTimeFiLM(d, d_extra=getattr(cfg, "d_cell_ctx", 0))
        # v5: cell-conditional pathway conductance (chromatin decides how much a gene listens to its
        # pathway neighbours). None -> static priors only (v3/v4 behaviour).
        self.pathway_cond = PathwayConductance(cfg.d_epi, cfg.d_epi_hidden) \
            if getattr(cfg, "pathway_conductance", False) else None

        self.base = nn.ModuleList([BaseLayer(cfg, n_priors) for _ in range(cfg.l_base)])
        self.perturb = nn.ModuleList([PerturbLayer(cfg, n_priors) for _ in range(cfg.l_perturb)])

        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.gene_bias = nn.Parameter(torch.zeros(G))

        # v2: additive SIGNED epigenetic contribution  += MLP_epi_out(E) * r  (zero-init -> starts as no-op)
        if cfg.epi_additive:
            self.epi_head = nn.Sequential(nn.Linear(cfg.d_epi, cfg.d_epi_hidden), nn.GELU(),
                                          nn.Linear(cfg.d_epi_hidden, 1))
            nn.init.zeros_(self.epi_head[-1].weight); nn.init.zeros_(self.epi_head[-1].bias)

    def forward(self, batch, return_attn=False, return_attr=False):
        # return_attn -> full per-head weights (manual attention, materializes [B,H,G,G]; for alpha maps).
        # return_attr -> lightweight attribution (ca_gene_norm + epi_contrib) via the FAST SDPA path;
        #                use this for DTI recall@k signals #1/#3 without paying for the weight tensors.
        # autocast is enabled INSIDE forward (not just in the train loop) so nn.DataParallel's worker
        # threads run AMP too -- autocast state is thread-local and does not propagate into the threads
        # DP spawns. enabled=False on CPU keeps the unit-test path in fp32. Nesting under the train
        # loop's autocast is a harmless no-op.
        dev = batch["X"].device.type
        with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
            X = batch["X"]; E = batch["E"]; r = batch["r"]                 # [B,G],[B,G,3],[B,G]
            atoms = batch["atoms"]; atom_valid = batch["atom_mask"]        # [B,M,512],[B,M] True=valid
            u = batch["u_feats"]; dose = batch["dose"]; time = batch["time"]
            B, G = X.shape

            # (1) chromatin gate + gated baseline
            s = self.gate(E, r)                                            # [B,G] in (0,1]
            x_mod = X * s                                                  # gated baseline (context)

            # (2) gene tokens
            avail = (r > 0).float().unsqueeze(-1)                          # [B,G,1]
            h = (self.gene_emb.unsqueeze(0)
                 + self.w_x(x_mod.unsqueeze(-1))
                 + self.w_e(torch.cat([E, avail], dim=-1)))                # [B,G,d]
            h = self.film(h, dose, time, batch.get("cell_ctx"))

            # (3) drug tokens: [global ; atoms]
            atom_tok = self.ln_atom(self.w_a(atoms)) + self.type_atom      # [B,M,d]
            glob_tok = (self.ln_u(self.w_u(u)) + self.type_drug).unsqueeze(1)  # [B,1,d]
            D = torch.cat([glob_tok, atom_tok], dim=1)                     # [B,1+M,d]
            valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=X.device), atom_valid], dim=1)
            key_mask = ~valid                                             # True = PAD

            # cell-conditional pathway conductance (chromatin gates pathway listening), [B,G] or None
            cond = self.pathway_cond(E, x_mod, r) if self.pathway_cond is not None else None

            # (4a) base/context encoder
            a_gene_last = None
            for layer in self.base:
                h, a_gene_last = layer(h, self.priors, self.support, need_weights=return_attn, cond=cond)

            # (4b) perturbation encoder
            a_drug_last = None
            ca_gene_norm = None                       # [B,G] total atom->gene displacement of each gene
            for layer in self.perturb:
                h, a_drug_last, a_gene_last, ca = layer(h, D, key_mask, self.priors, self.support,
                                                        need_weights=return_attn, cond=cond)
                if return_attn or return_attr:
                    n = ca.float().norm(dim=-1)        # per-gene L2 of the cross-attn contribution
                    ca_gene_norm = n if ca_gene_norm is None else ca_gene_norm + n

            # (5) per-gene head
            yhat = self.head(self.head_norm(h)).squeeze(-1) + self.gene_bias   # [B,G]
            epi_contrib = None
            if self.cfg.epi_additive:                 # signed, reliability-masked chromatin contribution
                epi_contrib = self.epi_head(E).squeeze(-1) * r                 # [B,G]
                yhat = yhat + epi_contrib
            if self.cfg.output_gate:
                yhat = yhat * s

        if return_attn:
            return yhat, {"gate": s, "alpha_drug": a_drug_last, "alpha_gene": a_gene_last,
                          "epi_contrib": epi_contrib, "ca_gene_norm": ca_gene_norm,
                          "pathway_cond": cond}
        if return_attr:
            return yhat, {"ca_gene_norm": ca_gene_norm, "epi_contrib": epi_contrib, "gate": s,
                          "pathway_cond": cond}
        return yhat
