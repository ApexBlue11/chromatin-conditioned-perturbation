# -*- coding: utf-8 -*-
"""
LincsV9 = the field's data substrate + this project's interpretability, with every change traceable to a
measurement rather than to a paper.

What changed from v7, and why:
  1. THE CONTROL IS A REAL INPUT, with its own encoder. Two views, because they help in opposite regimes:
     the plate-matched control is worth +0.103 over a per-cell mean on unseen COMPOUNDS, while on unseen
     CELLS an independent plate control is 0.08 WORSE than the per-cell mean and its apparent advantage is
     the model cancelling the control's own noise out of the target [RESULTS 27.5]. CCLE is dropped by
     default -- redundant, not harmful, and dominated by the per-cell L1000 mean [27.4].
  2. EXPRESSION IS BINNED AND EMBEDDED (128 levels), the field's encoding, gated on ab_encoder.py.
  3. GENE IDENTITY CARRIES A PRETRAINED STRING VECTOR from the full 19,496-node proteome graph, rather
     than being learned from scratch over 978 landmarks [27.3].
  4. CHROMATIN ENTERS AS A PER-GENE EMBEDDING summed into the gene token, the same way the PPI vector
     does -- not as a parallel branch the model can ignore, which is what v6/v7 measured it doing. The
     signed additive chromatin HEAD is kept as well; it is the one chromatin mechanism that ever survived.
  5. MULTI-TASK TARGET: absolute Level-3, delta, and the Level-5 z-score, so that every number in this
     project's history stays comparable while the reported convention matches the field's.
  6. AUXILIARY SUPERVISION USES SMALL FIXED LAMBDAS. v7 measured learned uncertainty weighting handing
     ~90 % of the gradient to the auxiliaries, and --no_aux then beat it 6/6 [RESULTS 22].

Kept unchanged because measured-good: atom->gene cross-attention (accuracy only, no interpretability claim
after the localisation retraction), late fusion, reliability weighting, stochastic depth, STRING message
passing, and named pathway nodes that terminate every readout in units with curated names.
"""
import os, sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v7'))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from modules import DoseTimeFiLM
from modules_v9 import (RMSNorm, CrossAttention, StochasticDepth, PPIMessagePassing, BinnedExpression,
                        _DrugBlock,
                        RawExpression, GeneRepresentation, ControlEncoder, NamedPathwayReadout,
                        MultiTaskHeads, _GeneBlock)


class PerturbBlock(nn.Module):
    def __init__(self, cfg, p_drop=0.0):
        super().__init__()
        self.nc = RMSNorm(cfg.d_model)
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.sdc = StochasticDepth(p_drop)
        self.gene = _GeneBlock(cfg, p_drop)
        # RESULTS 47.2 -- contextualise the drug tokens before the genes attend over them, as XPert's
        # crossEncoder does. Default off; `--drug_self_attn` turns it on for the A/B.
        self.drug_sa = _DrugBlock(cfg, p_drop) if getattr(cfg, 'drug_self_attn', False) else None

    def forward(self, h, D, key_mask, return_attn=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0, drug_atom_alpha=1.0, drug_global_self_only=False, xattn_global_only=False):
        if drug_global_self_only and self.drug_sa is None:
            raise ValueError("drug_global_self_only=True requested but self.drug_sa is None")
            
        # The updated D is RETURNED, so contextualisation compounds across blocks exactly as it does in
        # XPert, where crossEncoder.forward emits drug_SA_embed alongside the cell output.
        # TASK W4: diagonal=True masks drug_sa to the identity for the zero-cross-atom ablation arm.
        diag = diagonal or drug_diagonal
        if self.drug_sa is not None:
            D = self.drug_sa(D, key_mask=key_mask, diagonal=diag, alpha=drug_alpha, atom_alpha=drug_atom_alpha, global_self_only=drug_global_self_only)
            
        # RESULTS 79, cut X: the genes read the global key only, so atom content can reach them solely through
        # the global token. key_mask itself is untouched: drug self-attention still sees the real padding.
        xmask = key_mask
        if xattn_global_only:
            xmask = key_mask.clone()
            xmask[:, 1:] = True
        a = self.cross(self.nc(h), D, xmask)
        
        h = h + self.sdc(a)
        h = self.gene(h)
        return (h, D, a) if return_attn else (h, D)


class LincsV9(nn.Module):
    def __init__(self, cfg, M_pathway, ppi=None, gene_vectors=None):
        super().__init__()
        self.cfg = cfg
        G, d = cfg.n_genes, cfg.d_model
        assert M_pathway.shape[1] == G, 'M_pathway must be [P, 978]'

        self.gene_repr = GeneRepresentation(cfg, gene_vectors)
        mk = (lambda: BinnedExpression(d, cfg.n_bins, G, cfg.bin_mode)) if cfg.expr_encoder == 'binned' \
            else (lambda: RawExpression(d))
        self.expr_ctl = mk()
        self.expr_cell = mk()
        self.ctl_enc = ControlEncoder(cfg, self.expr_ctl)
        self.cell_enc = ControlEncoder(cfg, self.expr_cell)
        self.ctl_mix = nn.Linear(2 * d, d)

        self.film = DoseTimeFiLM(d, d_extra=cfg.d_cell_ctx)
        self.w_a = nn.Linear(cfg.d_atom, d)
        self.w_u = nn.Linear(cfg.d_global, d)
        self.type_atom = nn.Parameter(torch.zeros(d))
        self.type_drug = nn.Parameter(torch.zeros(d))
        self.ln_atom, self.ln_u = RMSNorm(d), RMSNorm(d)

        L = cfg.l_base + cfg.l_perturb
        rates = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        self.base = nn.ModuleList([_GeneBlock(cfg, rates[i]) for i in range(cfg.l_base)])
        self.perturb = nn.ModuleList([PerturbBlock(cfg, rates[cfg.l_base + i])
                                      for i in range(cfg.l_perturb)])

        self.ppi = PPIMessagePassing(ppi, d, cfg.dropout) if (cfg.use_ppi and ppi is not None) else None
        self.sd_ppi = StochasticDepth(cfg.stoch_depth)
        self.pathway = NamedPathwayReadout(M_pathway, d, cfg.d_pathway)
        self.sd_path = StochasticDepth(cfg.stoch_depth)
        self.heads = MultiTaskHeads(cfg)
        if cfg.use_aux:
            self.aux_path = nn.Linear(cfg.d_pathway, 1)
            self.aux_epi = nn.Linear(d, 1)

    # ---- fitting the quantiser is a DATA step, not a training step: training rows only ----
    def fit_bins(self, X_ctl_train, X_cell_train=None):
        if self.cfg.expr_encoder != 'binned':
            return self
        self.expr_ctl.fit(X_ctl_train)
        self.expr_cell.fit(X_cell_train if X_cell_train is not None else X_ctl_train)
        return self

    def forward(self, batch, return_aux=False, return_interp=False, diagonal=False, drug_diagonal=False, drug_alpha=1.0, drug_atom_alpha=1.0, drug_global_self_only=False, drug_xattn_global_only=False):
        diag = (diagonal or drug_diagonal or
                (batch.get('drug_diagonal', False) if isinstance(batch, dict) else False) or
                (batch.get('diagonal', False) if isinstance(batch, dict) else False))
        alpha = batch.get('drug_alpha', drug_alpha) if isinstance(batch, dict) else drug_alpha
        atom_alpha = batch.get('drug_atom_alpha', drug_atom_alpha) if isinstance(batch, dict) else drug_atom_alpha
        batch_drug_global_self_only = batch.get('drug_global_self_only', drug_global_self_only) if isinstance(batch, dict) else drug_global_self_only
        batch_drug_xattn_global_only = batch.get('drug_xattn_global_only', drug_xattn_global_only) if isinstance(batch, dict) else drug_xattn_global_only
        E, r = batch['E'], batch['r']
        x_ctl = batch['x_ctl']
        x_cell = batch.get('x_cell', x_ctl)
        atoms, atom_valid = batch['atoms'], batch['atom_mask']
        u, dose, time = batch['u_feats'], batch['dose'], batch['time']
        B = x_ctl.shape[0]

        gene_tok = self.gene_repr(E, r)                                  # [1 or B, G, d]
        if gene_tok.shape[0] == 1:
            gene_tok = gene_tok.expand(B, -1, -1)

        h_ctl = self.ctl_enc(x_ctl, gene_tok) if self.cfg_use('use_matched_ctl', batch) else \
            torch.zeros_like(gene_tok)
        h_cell = self.cell_enc(x_cell, gene_tok) if self.cfg_use('use_cell_ctl', batch) else \
            torch.zeros_like(gene_tok)
        h = gene_tok + self.ctl_mix(torch.cat([h_ctl, h_cell], -1))
        h = self.film(h, dose, time, batch.get('cell_ctx'))

        D = torch.cat([(self.ln_u(self.w_u(u)) + self.type_drug).unsqueeze(1),
                       self.ln_atom(self.w_a(atoms)) + self.type_atom], dim=1)
        key_mask = ~torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x_ctl.device), atom_valid], 1)

        for blk in self.base:
            h = blk(h)
        if self.ppi is not None:
            h = h + self.sd_ppi(self.ppi(h))

        path_delta, pathways = self.pathway(h)
        h = h + self.sd_path(path_delta)

        attn = None
        for i, blk in enumerate(self.perturb):
            if return_interp and i == len(self.perturb) - 1:
                h, D, attn = blk(h, D, key_mask, return_attn=True, diagonal=diag, drug_alpha=alpha, drug_atom_alpha=atom_alpha, drug_global_self_only=batch_drug_global_self_only, xattn_global_only=batch_drug_xattn_global_only)
            else:
                h, D = blk(h, D, key_mask, diagonal=diag, drug_alpha=alpha, drug_atom_alpha=atom_alpha, drug_global_self_only=batch_drug_global_self_only, xattn_global_only=batch_drug_xattn_global_only)

        out, epi_contrib = self.heads(h, E, r, x_ctl)
        if return_aux or return_interp:
            aux = {'pathway_activations': pathways, 'epi_contrib': epi_contrib, 'atom_gene': attn}
            if self.cfg.use_aux:
                aux['pathway_pred'] = self.aux_path(pathways).squeeze(-1)
                aux['epi_pred'] = self.aux_epi(h).squeeze(-1)
            return out, aux
        return out

    @property
    def bins_fitted(self):
        from modules_v9 import BinnedExpression
        qs = [m for m in self.modules() if isinstance(m, BinnedExpression)]
        # `fitted == 1` only says fit() was CALLED. A NaN-poisoned fit sets it and still collapses every
        # value to one bin, which is how a whole set of runs trained with no expression input at all.
        return all(q.discriminates() for q in qs) if qs else True

    def cfg_use(self, name, batch):
        """A batch may carry only one control view (the A/B arms do). Falls back to the config."""
        return batch.get(name, getattr(self.cfg, name, True))


def aux_targets(delta, M_norm):
    """Both derived from the MEASURED response, never from a branch input.
      pathway [B,P] : mean|delta| over each named term's member genes -- "is this term moving?"
      epi     [B,G] : |delta| per gene                                -- "which genes CAN move here?"
    """
    a = delta.abs()
    return torch.einsum('pg,bg->bp', M_norm, a), a


def v9_loss(out, batch, cfg, M_norm, aux=None):
    """abs + delta + l5 + PCC(delta), with the auxiliaries at small FIXED weights.

    Rows are masked per target: a signature can have a Level-3 substrate and no usable Level-5 row, or the
    reverse, and averaging over a NaN would poison the whole batch silently.
    """
    w_abs, w_delta, w_l5, w_pcc = cfg.task_w
    losses, parts = 0.0, {}
    hub = nn.functional.huber_loss

    def masked(pred, tgt, m):
        if m is None or bool(m.all()):
            return hub(pred, tgt, delta=cfg.huber_delta)
        if not bool(m.any()):
            return pred.sum() * 0.0
        return hub(pred[m], tgt[m], delta=cfg.huber_delta)

    if cfg_pred(cfg, 'predict_delta') and 'y_delta' in batch:
        l = masked(out['delta'], batch['y_delta'], batch.get('m_l3'))
        losses = losses + w_delta * l; parts['delta'] = float(l.detach())
        p = out['delta'] - out['delta'].mean(1, keepdim=True)
        t = batch['y_delta'] - batch['y_delta'].mean(1, keepdim=True)
        pcc = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp(min=1e-6)
        m = batch.get('m_l3')
        pcc = pcc[m] if (m is not None and bool(m.any()) and not bool(m.all())) else pcc
        lp = 1.0 - pcc.mean()
        losses = losses + w_pcc * lp; parts['pcc'] = float(lp.detach())
    if cfg_pred(cfg, 'predict_abs') and 'y_abs' in batch:
        l = masked(out['abs'], batch['y_abs'], batch.get('m_l3'))
        losses = losses + w_abs * l; parts['abs'] = float(l.detach())
    if cfg_pred(cfg, 'predict_l5') and 'y_l5' in batch:
        l = masked(out['l5'], batch['y_l5'], batch.get('m_l5'))
        losses = losses + w_l5 * l; parts['l5'] = float(l.detach())

    if aux is not None and cfg.use_aux and 'y_delta' in batch:
        tp, te = aux_targets(batch['y_delta'], M_norm)
        lp = hub(aux['pathway_pred'], tp, delta=cfg.huber_delta)
        le = hub(aux['epi_pred'], te, delta=cfg.huber_delta)
        losses = losses + cfg.aux_pathway_w * lp + cfg.aux_epi_w * le
        parts['aux_pathway'], parts['aux_epi'] = float(lp.detach()), float(le.detach())
    return losses, parts


def cfg_pred(cfg, name):
    return getattr(cfg, name, True)
