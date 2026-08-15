# -*- coding: utf-8 -*-
"""
v7 building blocks: the modern-transformer recipe + the two mechanisms that make the biological priors
LOAD-BEARING instead of decorative. Rationale and sources: ../V7_PLAN.md.

Why each piece is here (v6 measured all four of its cell-context modules at ~0, so this is not tuning):
  RMSNorm      ~10% faster than LayerNorm at equal quality; standard in modern stacks
  SwiGLU       gated FFNs consistently beat plain MLPs at matched FLOPs
  QK-norm      bounds attention logits -> better stability and convergence
  StochasticDepth  regularisation AND the established remedy for residual "loafing", where a residual
               branch under-contributes because the skip path is easier. v6's pathway branch is exactly
               that case: live (|dY|max 0.69) but worth +0.0003 [4.13]
  PPIMessagePassing  STRING has 12,665 edges over our 978 genes and v6 used NONE of them
               (use_prior_bias=False means the buffer is never even loaded). A hard sparse graph layer is
               how a PPI prior is normally used; v5's soft attention bias was measured not to steer (1.1x)
  AuxHeads     deep supervision. Nothing in v6 ASKED the pathway nodes or the chromatin encoder to mean
               anything, so they were free to be ignored -- and they were. Each head reads ONLY its own
               branch, so the gradient has to make that branch informative
  UncertaintyWeighting  Kendall et al. 2018 homoscedastic weighting, so the auxiliary losses are balanced
               by learned uncertainty rather than by hand-tuned lambdas
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """Gated FFN. d_ff is scaled by 2/3 so the parameter count matches a plain MLP of width d_ff."""
    def __init__(self, d, d_ff, dropout=0.1):
        super().__init__()
        h = int(2 * d_ff / 3)
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class QKNormAttention(nn.Module):
    """Self-attention with query/key RMS normalisation and a learned per-head temperature.

    QK-norm keeps attention logits bounded, which is the cheapest known fix for the logit blow-up that
    destabilises training. No prior bias here on purpose: v5's `lambda*log1p(prior)` was measured to steer
    attention only 1.1x random [4.6], so v7 puts its priors in HARD structures (the Reactome mask and the
    STRING graph layer) instead."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.dh); self.k_norm = RMSNorm(self.dh)
        self.scale = nn.Parameter(torch.full((n_heads, 1, 1), math.log(1.0 / math.sqrt(self.dh))))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask=None):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        sp = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
        q, k, v = self.q_norm(sp(q)), self.k_norm(sp(k)), sp(v)
        q = q * self.scale.exp().unsqueeze(0)
        mask = None
        if key_mask is not None:
            mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
            mask = mask.masked_fill(key_mask[:, None, None, :], float("-inf")).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        return self.drop(self.o(out.transpose(1, 2).reshape(B, L, self.h * self.dh)))


class CrossAttention(nn.Module):
    """atom -> gene cross-attention. Kept from v6 for ACCURACY only: it is the largest predictive drug
    feature (Δpearson +0.163 [4.3]) and carries NO interpretability claim (our own null: known targets at
    median rank percentile 0.560, worse than chance [4.1a])."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d, bias=False); self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.dh); self.k_norm = RMSNorm(self.dh)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, mem_mask):
        B, L, _ = x.shape; M = mem.shape[1]
        k, v = self.kv(mem).chunk(2, dim=-1)
        q = self.q_norm(self.q(x).view(B, L, self.h, self.dh).transpose(1, 2))
        k = self.k_norm(k.view(B, M, self.h, self.dh).transpose(1, 2))
        v = v.view(B, M, self.h, self.dh).transpose(1, 2)
        mask = torch.zeros(B, 1, 1, M, device=x.device, dtype=q.dtype)
        mask = mask.masked_fill(mem_mask[:, None, None, :], float("-inf")).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.drop(self.o(out.transpose(1, 2).reshape(B, L, self.h * self.dh)))


class StochasticDepth(nn.Module):
    """Drop a whole residual branch per-example with probability p during training.

    Here it does double duty. It regularises, and it removes the model's option to IGNORE a branch: if the
    skip path is sometimes unavailable, the branch has to carry signal. v6's pathway branch was live but
    worth +0.0003 -- the textbook symptom of residual under-use."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        m = torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) < keep
        return x * m / keep                      # rescaled so the expectation is unchanged


class PPIMessagePassing(nn.Module):
    """One graph-convolution step over the STRING PPI graph: h_g <- h_g + W( sum_j A_norm[g,j] h_j ).

    STRING v12 at score >= 400 gives 12,665 edges over the 978 landmarks (2.6 % density). v6 never used
    them -- `use_prior_bias=False` means the buffer is not even registered. This is the way a PPI prior is
    normally consumed (a sparse message-passing step), as opposed to v5's additive attention bias, which
    measurement showed does not steer [4.6]. Row-normalised so node degree does not set the scale, and
    zero-init output so the layer starts as an exact no-op and must earn its contribution."""
    def __init__(self, A, d_model, dropout=0.1):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32).clone()
        A.fill_diagonal_(0)
        self.register_buffer("A", A / A.sum(1, keepdim=True).clamp(min=1e-6))
        self.register_buffer("has_edge", (A.sum(1) > 0).float().unsqueeze(-1))
        self.norm = RMSNorm(d_model)
        self.w = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.w.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        m = torch.einsum("gj,bjd->bgd", self.A, self.norm(h))
        return self.drop(self.w(m)) * self.has_edge      # genes with no STRING edge get exactly 0


class AuxHeads(nn.Module):
    """Deep supervision for the two priors, each reading ONLY its own branch.

    That restriction is the whole point: if the head could see the fused representation it would be solved
    by the main path and teach the branch nothing.
      pathway head : pathway activations   -> per-pathway mean|Y| over that pathway's member genes
      chromatin head: chromatin encoding   -> per-gene |Y| ("which genes CAN move in this cell")
    Both targets are derived from the MEASURED response, which is never an input to either branch."""
    def __init__(self, d_pathway, d_model, hidden=64):
        super().__init__()
        self.path = nn.Sequential(nn.Linear(d_pathway, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.epi = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, pathway_act, h_epi):
        return self.path(pathway_act).squeeze(-1), self.epi(h_epi).squeeze(-1)   # [B,P], [B,G]


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla 2018), so auxiliary losses are balanced
    by a LEARNED uncertainty instead of hand-tuned lambdas that we would inevitably tune on the test set.

        L = sum_i  exp(-s_i) * L_i + s_i        with s_i = log(sigma_i^2), learned

    A task the model cannot do drives s_i up and quietly switches itself off, which is the honest failure
    mode: if supervising the pathway branch is impossible, the weighting says so rather than dragging the
    main loss down. Read the weights off after training -- they are a diagnostic."""
    def __init__(self, n_tasks):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = sum(torch.exp(-self.log_var[i]) * l + self.log_var[i] for i, l in enumerate(losses))
        return total, {f"w{i}": float(torch.exp(-self.log_var[i]).detach()) for i in range(len(losses))}
