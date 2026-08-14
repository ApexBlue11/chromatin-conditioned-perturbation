# -*- coding: utf-8 -*-
"""
Model building blocks. Each maps to an equation in model/MODEL_MATH.md:
  EpiGate                -> (1) s = 1 - r*(1 - sigmoid(MLP_epi(E)))     [chromatin gate, s=1 fallback]
  DoseTimeFiLM           -> (2) FiLM (gamma,beta) from nonlinear dose/time features
  MultiHeadCrossAttention-> (4b) atom->gene attention (exposes alpha^drug)
  BiasedSelfAttention    -> (4a/4b) gene<->gene attention with biological prior bias (exposes alpha^gene)
Attention is implemented manually so per-head weights are returned for interpretability.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bias_mask(q, key_mask, add_bias):
    """Combine an additive prior bias and a key-padding mask into one float attn_mask [B,H,Lq,Lk]."""
    m = None
    if add_bias is not None:
        # [H,Lq,Lk] -> [1,H,Lq,Lk]; broadcast over batch instead of materializing B copies (the prior
        # bias is identical for every example). SDPA accepts a broadcastable attn_mask.
        m = add_bias if add_bias.dim() == 4 else add_bias.unsqueeze(0)
    if key_mask is not None:
        km = torch.zeros(q.shape[0], 1, 1, key_mask.shape[-1], device=q.device, dtype=q.dtype)
        km = km.masked_fill(key_mask[:, None, None, :], float("-inf"))
        m = km if m is None else m + km
    # CUDA's fused SDPA kernels require the mask's last dimension to be contiguous; a broadcast/expanded
    # bias has stride 0 there and raises. CPU silently accepts it, so this only ever bites on a GPU kernel.
    return m if m is None or m.stride(-1) == 1 else m.contiguous()


def masked_softmax_attention(q, k, v, key_mask=None, add_bias=None):
    """Manual attention that RETURNS per-head weights (for interpretability). Shapes: q:[B,H,Lq,dh],
    k,v:[B,H,Lk,dh], key_mask:[B,Lk] True=PAD, add_bias:[B,H,Lq,Lk] or [H,Lq,Lk]. -> (out, attn)."""
    dh = q.shape[-1]
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
    if add_bias is not None:
        logits = logits + (add_bias if add_bias.dim() == 4 else add_bias.unsqueeze(0))
    if key_mask is not None:
        logits = logits.masked_fill(key_mask[:, None, None, :], float("-inf"))
    attn = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)   # fully-masked rows -> 0
    return torch.matmul(attn, v), attn


def attention(q, k, v, key_mask=None, add_bias=None, need_weights=False):
    """Dispatch: memory-efficient SDPA (no weights) for training; manual path when weights are needed."""
    if need_weights:
        return masked_softmax_attention(q, k, v, key_mask, add_bias)
    attn_mask = _bias_mask(q, key_mask, add_bias)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out, None


class EpiGate(nn.Module):
    """s_{c,g} = 1 - r*(1 - sigmoid(MLP_epi(E))). r=0 -> s=1 exactly (neutral fallback)."""
    def __init__(self, d_epi=3, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_epi, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, E, r):
        # E:[B,G,3]  r:[B,G] in [0,1]
        sig = torch.sigmoid(self.mlp(E).squeeze(-1))        # [B,G] in (0,1)
        s = 1.0 - r * (1.0 - sig)                           # r=0 -> 1 ; r=1 -> sigmoid
        return s


class DoseTimeFiLM(nn.Module):
    """Nonlinear (inverted-U-capable) dose/time -> per-example FiLM (gamma,beta). Identity at init.
    d_extra>0 appends a per-example CELL-CONTEXT vector (e.g. lineage one-hot) to the conditioning, so the
    model can modulate all gene tokens by *what kind of cell this is*, not only per-gene X_base/E."""
    def __init__(self, d_model, n_freq=8, d_extra=0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(2, n_freq) * 2.0)   # fixed random Fourier
        feat = 2 + 4 * n_freq + d_extra                                # [dose,time] + sin/cos + cell context
        self.net = nn.Sequential(nn.Linear(feat, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, 2 * d_model))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # -> gamma=beta=0 -> identity
        self.d = d_model; self.d_extra = d_extra

    def _feat(self, dose, time, ctx=None):
        x = torch.stack([dose, time], dim=-1)               # [B,2]
        ang = x.unsqueeze(-1) * self.freqs                  # [B,2,n_freq]
        ang = ang.flatten(1)                                # [B,2*n_freq]
        out = torch.cat([x, torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.d_extra:
            out = torch.cat([out, ctx.to(out.dtype)], dim=-1)
        return out

    def forward(self, h, dose, time, ctx=None):
        # h:[B,G,d]  dose,time:[B]  ctx:[B,d_extra] or None
        gb = self.net(self._feat(dose, time, ctx))          # [B,2d]
        gamma, beta = gb[:, :self.d], gb[:, self.d:]
        return (1.0 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)


class PathwayConductance(nn.Module):
    """Cell-conditional PATHWAY CONDUCTANCE  c_{cell,g} = 1 + tanh(MLP([E_g ; x_g ; avail_g])).

    Motivation (measured): cells differ a lot in which pathways are active, and the model's pathway priors
    were only a STATIC bias (same lambda for every cell) -- it had no way to say "this pathway channel is
    open in this cell, closed in that one". Per-EDGE cell conditioning is infeasible ([B,H,G,G] ~ 10GB/batch),
    so we gate how much each gene LISTENS to its pathway neighbours: scale the gene<->gene attention OUTPUT
    per (cell, gene). Biologically: a gene in closed/repressed chromatin cannot be driven by its pathway
    partners, however strong the prior edge.

    Zero-init -> c=1 exactly at start (no-op), so it must EARN its use. Inspectable: c<1 = damped pathway
    input (closed), c>1 = amplified (open) -- a per-cell, per-gene readout of pathway conductance.
    NOTE: a measured caveat -- co-pathway module activity correlates ~0.85 with a gene's own baseline, so
    this must add value via the EPIGENETIC part (chromatin), not by re-encoding X_base."""
    def __init__(self, d_epi=3, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_epi + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)   # -> c = 1 at init

    def forward(self, E, x, r):
        # E:[B,G,3]  x:[B,G] gated baseline  r:[B,G] epi reliability -> c:[B,G]
        feat = torch.cat([E, x.unsqueeze(-1), (r > 0).to(E.dtype).unsqueeze(-1)], dim=-1)
        return 1.0 + torch.tanh(self.mlp(feat).squeeze(-1))          # in (0,2), ==1 at init


class MultiHeadCrossAttention(nn.Module):
    """Genes (query) attend to drug tokens (key/value). Returns out + attn [B,H,G,T] (alpha^drug)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)   # [B,H,L,dh]

    def forward(self, q_in, kv_in, key_mask=None, need_weights=False):
        q, k, v = self._split(self.q(q_in)), self._split(self.k(kv_in)), self._split(self.v(kv_in))
        out, attn = attention(q, k, v, key_mask=key_mask, need_weights=need_weights)
        B, H, Lq, dh = out.shape
        out = out.transpose(1, 2).reshape(B, Lq, H * dh)
        return self.drop(self.o(out)), attn


class BiasedSelfAttention(nn.Module):
    """Gene<->gene self-attention with additive biological prior bias on the first n_prior_heads heads.
    priors: [P,G,G] nonneg (registered by the model). lambda per (prior_head, prior) via softplus (>=0).
    Optional hard mask on head 0 (attend only on prior support) for a high-confidence pathway head."""
    def __init__(self, d_model, n_heads, n_priors, n_prior_heads=4, masked_head=False, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0 and n_prior_heads <= n_heads
        self.h, self.dh = n_heads, d_model // n_heads
        self.n_prior_heads, self.masked_head = n_prior_heads, masked_head
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.log_lambda = nn.Parameter(torch.zeros(n_prior_heads, n_priors))  # softplus -> lambda>=0
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def _bias(self, priors, support):
        # priors:[P,G,G] ; returns [H,G,G], or None when there is no prior to add.
        # v6 sets use_prior_bias=False -> n_prior_heads=0 and `priors` is a DUMMY [1,1,1] buffer. Building a
        # bias from it produced a [H,1,1] tensor that SDPA broadcasts to [B,H,G,G] with stride 0 on the last
        # dim. CPU tolerates that; CUDA raises "(*bias): last dimension must be contiguous" -- so the whole
        # test suite passed locally and the model died on the first GPU batch. Return None instead: correct,
        # and it skips a pointless einsum every layer.
        if self.n_prior_heads == 0 and not self.masked_head:
            return None
        G = priors.shape[-1]
        bias = priors.new_zeros(self.h, G, G)
        lam = F.softplus(self.log_lambda)                       # [n_prior_heads,P]
        biased = torch.einsum("hp,pij->hij", lam, priors)       # [n_prior_heads,G,G]
        bias[:self.n_prior_heads] = biased
        if self.masked_head:
            neg = torch.zeros(G, G, device=priors.device)
            neg = neg.masked_fill(~support, float("-inf"))
            bias[0] = bias[0] + neg                             # head 0: on-support only
        return bias

    def forward(self, x, priors, support, need_weights=False):
        q, k, v = self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x))
        out, attn = attention(q, k, v, add_bias=self._bias(priors, support), need_weights=need_weights)
        B, H, L, dh = out.shape
        out = out.transpose(1, 2).reshape(B, L, H * dh)
        return self.drop(self.o(out)), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)

class PathwayBottleneck(nn.Module):
    """P-NET-style VISIBLE pathway layer: genes -> NAMED Reactome pathway nodes -> back to genes.

    Why this replaces the v5 approach. v5 biased gene<->gene attention with `lambda * log1p(A_copathway)`,
    a SOFT prior: signal may route around it, and measurement showed it does (prior heads land on-support
    only 1.1x random). It also used A_copathway [G,G], which is the gene-gene CO-MEMBERSHIP COLLAPSE of the
    real membership matrix M [P,G] -- discarding the named pathway axis we actually had.

    Here the prior is a hard connectivity MASK, as in P-NET (Nature 2021) / DCell / DrugCell: gene g
    contributes ONLY to pathways containing g, and a pathway returns signal ONLY to its member genes.
    `pathway_activation[:, p]` IS a masked aggregation of Reactome pathway p's member genes -- that much is
    interpretability by construction rather than post-hoc hope, and it is verified by test_v6 (perturbing a
    gene moves exactly its 91 pathways; the 243 genes in no pathway get exactly 0).

    HONEST LIMIT vs P-NET (name notwithstanding, this is NOT a bottleneck). P-NET stacks pathway layers as
    the ONLY route to the outcome, so information genuinely cannot bypass them. Here the caller adds the
    output residually (`h = h + delta` in model_v6), which makes this a masked SIDE BRANCH: the mask governs
    what the branch may read and write, but the main residual stream can route around it entirely -- and at
    init, with w_out zero-init, it does exactly that. So "pathway p's activation is faithful to pathway p"
    holds; "the prediction had to pass through pathway p" does NOT. How much it actually contributes is
    precisely what eval_v6's `mean_pathway` ablation measures. See CLAIMS 4.13.

    Chromatin gates at the PATHWAY level (v5 gated per gene, where it mostly re-encoded X_base, corr 0.74).
    Sparse by design: with 83 training cell lines the parameter reduction is itself useful regularisation.
    Zero-init output projection => exact no-op at initialisation, so the layer must earn its contribution.
    """
    def __init__(self, M, d_model, d_path=64, d_epi=3, gate_with_epi=True):
        super().__init__()
        M = torch.as_tensor(M, dtype=torch.float32)          # [P,G] binary membership
        if M.shape[0] > M.shape[1]:
            M = M.t()
        P, G = M.shape
        self.P, self.G = P, G
        # row/col normalised masks: mean over a pathway's genes, mean over a gene's pathways
        self.register_buffer("M_in", M / M.sum(1, keepdim=True).clamp(min=1))       # [P,G] gene -> pathway
        self.register_buffer("M_out", (M / M.sum(0, keepdim=True).clamp(min=1)).t())  # [G,P] pathway -> gene
        self.register_buffer("has_pathway", (M.sum(0) > 0).float().unsqueeze(-1))   # [G,1]

        self.w_in = nn.Linear(d_model, d_path)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_path)
        self.w_out = nn.Linear(d_path, d_model)
        nn.init.zeros_(self.w_out.weight); nn.init.zeros_(self.w_out.bias)          # no-op at init
        self.epi_gate = nn.Sequential(nn.Linear(d_epi, 16), nn.GELU(), nn.Linear(16, 1)) if gate_with_epi else None
        if self.epi_gate is not None:
            nn.init.zeros_(self.epi_gate[-1].weight); nn.init.zeros_(self.epi_gate[-1].bias)  # gate == 1 at init

    def forward(self, h, E=None, return_pathways=False):
        # h:[B,G,d]  E:[B,G,d_epi] -> (h_delta [B,G,d], pathway activations [B,P,d_path])
        a = self.act(self.w_in(h))                               # [B,G,dp]
        a = torch.einsum("pg,bgd->bpd", self.M_in, a)            # masked gene -> pathway
        a = self.norm(a)
        if self.epi_gate is not None and E is not None:
            e_path = torch.einsum("pg,bgk->bpk", self.M_in, E)   # chromatin summarised PER PATHWAY
            a = a * (1.0 + torch.tanh(self.epi_gate(e_path)))    # in (0,2), ==1 at init
        out = torch.einsum("gp,bpd->bgd", self.M_out, a)         # masked pathway -> gene
        out = self.w_out(out) * self.has_pathway                 # genes in no pathway get exactly 0
        return (out, a) if return_pathways else (out, None)
