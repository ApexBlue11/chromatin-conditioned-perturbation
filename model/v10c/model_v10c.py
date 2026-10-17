"""v10-C: the v10 base (F4-F7, F9 of V10_AUDIT_AND_PLAN.md) with chromatin as its own stream that acts on the drug's
effect (RESULTS 100, 100.4). Every chromatin pathway starts as an exact no-op, so v10-C at initialisation computes what
v10 without chromatin computes, and the "off" arm keeps all of the machinery (training-mean chromatin and mask).

Per gene g in a cell, c_g = ChromatinStream(F_g, m_g) enters in four places, and nowhere else:
  1. the drug query      q_g = W_q( n(h_g) + W_c c_g )             (which drug features a gene reads; W_c = 0 at init)
  2. the drug gate       h_g += s(dose | drug) * gamma_g * attn_g,  gamma_g = 2 sigma(MLP([c_g; drug; dose])) (= 1 at init)
  3. STRING propagation  A_eff = diag(sqrt a) A diag(sqrt a),       a_g = sigma(w c_g + b)   (factorised; no B x G x G tensor)
  4. the basal offset    b_g = w_h h_basal,g + w_c c_g              (w_c = 0 at init)
The gene token itself carries no chromatin (the principal's "keep them separate").

The v10 base: F4 the named pathway readout after the drug blocks; F5 a chemCPA-style dose scale
s = sigma(a_d (log dose - b_d)), zero dose -> zero effect, with time entering the drug token, not the cell tokens;
F6 delta = basal(cell) + Delta(drug, dose, time, cell), the plate contrast is carried by Delta alone (the basal term is
identical for plate-mates); F7 an MLP drug encoder with drug self-attention on; F9 no stochastic depth on the drug path.
"""
import math, os, sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
V9 = os.path.join(os.path.dirname(HERE), 'v9')
for p in (V9, os.path.join(os.path.dirname(HERE), 'v7'), os.path.join(os.path.dirname(HERE), 'v6'), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules_v9 import (RMSNorm, CrossAttention, StochasticDepth, BinnedExpression, RawExpression,  # noqa: E402
                        ControlEncoder, NamedPathwayReadout, _GeneBlock, _DrugBlock)

N_FEAT, N_MASK, N_WIN = 16, 4, 5


class V10CConfig:
    n_genes = 978
    d_model = 256
    n_heads = 8
    d_ff = 1024
    l_control = 2
    l_base = 2
    l_perturb = 4
    dropout = 0.1
    stoch_depth = 0.1            # gene path only (F9)
    d_atom = 512
    d_global = 512 + 20 + 2048
    d_drug_hidden = 512
    n_lineage = 21
    d_pathway = 32
    d_chrom_hidden = 128
    d_gate_hidden = 64
    n_bins = 128
    bin_mode = 'global'
    expr_encoder = 'binned'
    drug_self_attn = True        # _DrugBlock reads it
    huber_delta = 1.0
    # loss weights: delta Huber, delta PCC, contrast Huber, contrast PCC
    w_delta, w_pcc, w_con, w_conpcc = 1.0, 0.5, 1.0, 0.5
    min_mates = 2

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)
        self.d_ff = kw.get('d_ff', 4 * self.d_model)


def mask_per_feature(M):
    """[.., 4] availability (ACC, ACT, PCG, METH) -> [.., 16] per feature (5 windows per axis, then methylation)."""
    return torch.cat([M[..., :3].repeat_interleave(N_WIN, -1), M[..., 3:4]], -1)


class ChromatinStream(nn.Module):
    """c_g = RMSNorm(MLP([F_g * m_g ; m_g])): a cell's chromatin at gene g, its own d-wide stream."""

    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT + N_MASK, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.norm = RMSNorm(d)

    def forward(self, F, M):
        M = M.to(F.dtype)
        return self.norm(self.net(torch.cat([F * mask_per_feature(M), M], -1)))


class GatedPPI(nn.Module):
    """One STRING step whose edge (i, j) is weighted sqrt(a_i a_j), a_g = sigma(w c_g + b) from chromatin.
    Row-normalised A and a zero-init output as v7's PPIMessagePassing; b = logit(0.95) and w = 0 at init."""

    def __init__(self, A, d, dropout):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32).clone()
        A.fill_diagonal_(0)
        self.register_buffer('A', A / A.sum(1, keepdim=True).clamp(min=1e-6))
        self.register_buffer('has_edge', (A.sum(1) > 0).float().unsqueeze(-1))
        self.norm = RMSNorm(d)
        self.w = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.w.weight)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.95 / 0.05))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, c):
        r = torch.sigmoid(self.gate(c)).sqrt()                       # [B, G, 1]
        m = torch.einsum('gj,bjd->bgd', self.A.to(h.dtype), r * self.norm(h)) * r
        return self.drop(self.w(m)) * self.has_edge, r.squeeze(-1) ** 2


class CellFiLM(nn.Module):
    """Lineage -> (1 + gamma) h + beta on the cell tokens; identity at init. Dose and time do not enter here (F5)."""

    def __init__(self, n_lineage, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_lineage, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.d = d

    def forward(self, h, ctx):
        gb = self.net(ctx.to(h.dtype))
        return (1.0 + gb[:, :self.d]).unsqueeze(1) * h + gb[:, self.d:].unsqueeze(1)


class DoseScale(nn.Module):
    """F5, chemCPA-style: s = sigma(a_d (ld - b_d)), a_d = softplus(.) + 0.1 > 0, from the drug's global token. As the
    (standardised log10) dose goes to -inf the drug's effect goes to 0."""

    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, 1)
        self.b = nn.Linear(d, 1)

    def forward(self, g, ld):
        a = nn.functional.softplus(self.a(g)).squeeze(-1) + 0.1
        return torch.sigmoid(a * (ld - self.b(g).squeeze(-1)))      # [B]


class PerturbBlockC(nn.Module):
    def __init__(self, cfg, p_drop):
        super().__init__()
        d = cfg.d_model
        self.drug_sa = _DrugBlock(cfg, 0.0)                         # F9: no stochastic depth on the drug path
        self.nc = RMSNorm(d)
        self.cross = CrossAttention(d, cfg.n_heads, cfg.dropout)
        self.wc = nn.Linear(d, d, bias=False)                       # chromatin -> the gene's query
        nn.init.zeros_(self.wc.weight)
        self.gate = nn.Sequential(nn.Linear(2 * d + 1, cfg.d_gate_hidden), nn.GELU(), nn.Linear(cfg.d_gate_hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)                          # gamma = 2 sigma(0) = 1
        self.gene = _GeneBlock(cfg, p_drop)

    def forward(self, h, D, key_mask, c, s, ld):
        D = self.drug_sa(D, key_mask=key_mask)
        a = self.cross(self.nc(h) + self.wc(c), D, key_mask)
        G = h.shape[1]
        g_in = torch.cat([c, D[:, :1].expand(-1, G, -1), ld.to(h.dtype)[:, None, None].expand(-1, G, 1)], -1)
        gamma = 2.0 * torch.sigmoid(self.gate(g_in))               # [B, G, 1]
        h = h + s.to(h.dtype)[:, None, None] * gamma * a
        return self.gene(h), D, gamma.squeeze(-1)


class LincsV10C(nn.Module):
    def __init__(self, cfg, M_pathway, ppi, gene_vectors):
        super().__init__()
        self.cfg = cfg
        G, d = cfg.n_genes, cfg.d_model
        self.emb = nn.Parameter(torch.randn(G, d) * 0.02)
        gv = torch.as_tensor(gene_vectors, dtype=torch.float32)
        assert gv.shape[0] == G
        self.vec = nn.Parameter(gv)
        self.vec_proj = nn.Linear(gv.shape[1], d)
        self.tok_norm = RMSNorm(d)
        mk = (lambda: BinnedExpression(d, cfg.n_bins, G, cfg.bin_mode)) if cfg.expr_encoder == 'binned' \
            else (lambda: RawExpression(d))
        self.expr_ctl, self.expr_cell = mk(), mk()
        self.ctl_enc = ControlEncoder(cfg, self.expr_ctl)
        self.cell_enc = ControlEncoder(cfg, self.expr_cell)
        self.ctl_mix = nn.Linear(2 * d, d)
        self.film = CellFiLM(cfg.n_lineage, d)
        self.chrom = ChromatinStream(d, cfg.d_chrom_hidden)

        self.w_u = nn.Sequential(nn.Linear(cfg.d_global, cfg.d_drug_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(cfg.d_drug_hidden, d))
        self.w_a = nn.Linear(cfg.d_atom, d)
        self.ln_u, self.ln_a = RMSNorm(d), RMSNorm(d)
        self.type_drug, self.type_atom = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.w_dt = nn.Linear(3, d)                                 # [ld, time, ld*time] -> the drug token (F5)
        nn.init.zeros_(self.w_dt.weight)
        nn.init.zeros_(self.w_dt.bias)
        self.dose = DoseScale(d)

        L = cfg.l_base + cfg.l_perturb
        rates = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        self.base = nn.ModuleList([_GeneBlock(cfg, rates[i]) for i in range(cfg.l_base)])
        self.ppi = GatedPPI(ppi, d, cfg.dropout)
        self.sd_ppi = StochasticDepth(cfg.stoch_depth)
        self.perturb = nn.ModuleList([PerturbBlockC(cfg, rates[cfg.l_base + i]) for i in range(cfg.l_perturb)])
        self.pathway = NamedPathwayReadout(M_pathway, d, cfg.d_pathway)
        self.sd_path = StochasticDepth(cfg.stoch_depth)
        self.basal_h = nn.Linear(d, 1)
        self.basal_c = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.basal_c.weight)
        self.drug_head = nn.Linear(d, 1)

    def fit_bins(self, X_ctl_train, X_cell_train):
        if self.cfg.expr_encoder == 'binned':
            self.expr_ctl.fit(X_ctl_train)
            self.expr_cell.fit(X_cell_train)
        return self

    @property
    def bins_fitted(self):
        qs = [m for m in self.modules() if isinstance(m, BinnedExpression)]
        return all(q.discriminates() for q in qs) if qs else True

    def forward(self, b, return_aux=False):
        x_ctl, x_cell = b['x_ctl'], b['x_cell']
        B = x_ctl.shape[0]
        tok = self.tok_norm(self.emb + self.vec_proj(self.vec)).unsqueeze(0).expand(B, -1, -1)
        h = tok + self.ctl_mix(torch.cat([self.ctl_enc(x_ctl, tok), self.cell_enc(x_cell, tok)], -1))
        h = self.film(h, b['cell_ctx'])
        c = self.chrom(b['F'], b['M'])                              # [B, G, d]
        for blk in self.base:
            h = blk(h)
        m, a_ppi = self.ppi(h, c)
        h = h + self.sd_ppi(m)
        h_basal = h

        ld, t = b['dose'], b['time']
        g = self.ln_u(self.w_u(b['u_feats'])) + self.type_drug
        g = g + self.w_dt(torch.stack([ld, t, ld * t], -1).to(g.dtype))
        D = torch.cat([g.unsqueeze(1), self.ln_a(self.w_a(b['atoms'])) + self.type_atom], 1)
        key_mask = ~torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x_ctl.device), b['atom_mask']], 1)
        s = self.dose(g, ld)
        gammas = []
        for blk in self.perturb:
            h, D, gam = blk(h, D, key_mask, c, s, ld)
            gammas.append(gam)
        pd, pathways = self.pathway(h)                              # F4: after the drug
        h = h + self.sd_path(pd)
        basal = (self.basal_h(h_basal) + self.basal_c(c)).squeeze(-1)
        drug = self.drug_head(h).squeeze(-1)
        delta = basal + drug
        out = {'delta': delta, 'abs': x_ctl + delta, 'drug': drug, 'basal': basal}
        if return_aux:
            return out, {'gamma': torch.stack(gammas, 1), 'a_ppi': a_ppi, 'dose_scale': s, 'pathways': pathways}
        return out


def batch_contrast(Y, plate, pert, min_mates=2):
    """Row i minus the mean of the other-compound rows on its plate within this set; ok where >= min_mates mates."""
    same = (plate[:, None] == plate[None, :]) & (pert[:, None] != pert[None, :])
    cnt = same.sum(1)
    mates = (same.to(Y.dtype) @ Y) / cnt.clamp(min=1).to(Y.dtype)[:, None]
    return Y - mates, cnt >= min_mates


def row_pcc(p, t):
    p = p - p.mean(1, keepdim=True)
    t = t - t.mean(1, keepdim=True)
    return (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp(min=1e-6)


def v10c_loss(out, b, cfg):
    hub = nn.functional.huber_loss
    y = b['y_delta']
    parts = {}
    l_d = hub(out['delta'], y, delta=cfg.huber_delta)
    l_p = 1.0 - row_pcc(out['delta'], y).mean()
    loss = cfg.w_delta * l_d + cfg.w_pcc * l_p
    parts['delta'], parts['pcc'] = float(l_d.detach()), float(l_p.detach())
    cy, ok = batch_contrast(y, b['plate'], b['pert'], cfg.min_mates)
    if bool(ok.any()):
        cp, _ = batch_contrast(out['delta'], b['plate'], b['pert'], cfg.min_mates)
        l_c = hub(cp[ok], cy[ok], delta=cfg.huber_delta)
        l_cp = 1.0 - row_pcc(cp[ok], cy[ok]).mean()
        loss = loss + cfg.w_con * l_c + cfg.w_conpcc * l_cp
        parts['con'], parts['conpcc'] = float(l_c.detach()), float(l_cp.detach())
    parts['n_contrast_rows'] = int(ok.sum())
    return loss, parts
