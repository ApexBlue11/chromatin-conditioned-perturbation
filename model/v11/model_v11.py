"""v11 (RESULTS 102 / 102.1): T-B's off arm (LincsV10C on training-mean chromatin, unchanged) plus a full-graph module FG.
The base is constructed first, so under the same seed its initial weights are v10-C's; FG and the structure head are
initialised under a forked CPU generator seeded seed + FG_SEED_OFFSET (the CPU generator only: torch.manual_seed would
also re-seed every CUDA generator, which fork_rng(devices=[]) does not restore; review 099 C1), so the CPU and CUDA streams
after construction are the ones T-B off's same-seed run starts training with; FG has no random op in its forward; FG's
readout is zero at init, so v11 at init computes exactly what T-B off computes (tested with identical base weights,
102.1 C1 / C7).

FG, per unique cell in a batch (shared by that cell's rows), over the 19,496-gene experimental + database graph:
  x_v = [gene vector_v ; CCLE rank_v, mask_v ; chromatin_v (16) , axis masks (4)]  ->  MLP  ->  z_v  (d_g = 64)
  a_v = sigma(w . [chromatin, masks, CCLE, mask] + b),  w = 0, b = logit 0.95        (the chromatin / CCLE gate)
  K = 2 rounds  z <- z + W_k sqrt(a) * (A_hat (sqrt(a) * RMSNorm(z))),  W_k = 0       (factorised gating, 102.1 C7)
Per row, the drug's target signal t_d over the nodes in 2 channels (any target; action sign), from the known ChEMBL
targets where allowed, else the structure head's predicted probabilities (detached; sign 0); hops 0, 1, 2 over the
row's cell's gated graph -> s_v, 6 channels, each divided by a fixed outcome-free scale.
Readout at the 978 landmarks, after h_basal (so the basal term stays drug-free, F6) and before the perturb blocks:
  h_g += U [z_g ; s_g ; z_g * P s_g],  U = 0 at init.
The structure head: the drug's global token before dose and time -> logits over the known compounds' target genes,
BCE (lambda 0.1) on known compounds only; off with the targets in the -targets arm (102.1 C2 / C4).
"""
import math, os, sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
if V10C not in sys.path:
    sys.path.insert(0, V10C)

from model_v10c import LincsV10C, RMSNorm, mask_per_feature, v10c_loss  # noqa: E402

D_G, K_ROUNDS = 64, 2
FG_SEED_OFFSET = 7919


class DeviceTables:
    """FG's fixed tables, kept out of the module's buffers: DataParallel would broadcast ~170 MB of buffers on every
    step. replicate() shallow-copies a module's __dict__, so every replica shares this object and each device's copy is
    made once, on first use."""

    def __init__(self, **arrays):
        self.cpu = {k: torch.as_tensor(v) for k, v in arrays.items()}
        self.dev = {}

    def get(self, device):
        key = str(device)
        if key not in self.dev:
            d = {k: v.to(device) for k, v in self.cpu.items()}
            n = int(self.cpu['n_nodes'])
            d['A'] = torch.sparse_coo_tensor(d['A_index'], d['A_value'], (n, n)).coalesce()
            self.dev[key] = d
        return self.dev[key]

    def replace(self, **arrays):
        """Swap tables (the reported permutations) and drop every device copy."""
        for k, v in arrays.items():
            self.cpu[k] = torch.as_tensor(v)
        self.dev = {}


def spmm(A, X):
    """A [N, N] sparse (fp32) times X [U, N, C] -> [U, N, C], in fp32 outside autocast."""
    U, N, C = X.shape
    with torch.autocast(device_type=X.device.type, enabled=False):
        Y = torch.sparse.mm(A, X.float().permute(1, 0, 2).reshape(N, U * C))
    return Y.reshape(N, U, C).permute(1, 0, 2)


class FullGraphModule(nn.Module):
    def __init__(self, d_model, fg):
        super().__init__()
        self.tables = DeviceTables(A_index=fg['A_index'], A_value=fg['A_value'], n_nodes=fg['n_nodes'],
                                   gv=fg['gene_vec'], F=fg['F'], M=fg['M'], E=fg['E'], Em=fg['Em'],
                                   lm=fg['landmark_idx'], s_scale=fg['s_scale'], vocab=fg['vocab'])
        self.n_nodes = int(fg['n_nodes'])
        self.targets_on = bool(fg['targets_on'])
        d_gv = fg['gene_vec'].shape[1]
        self.node_in = nn.Sequential(nn.Linear(d_gv + 2 + 20, 2 * D_G), nn.GELU(), nn.Linear(2 * D_G, D_G))
        self.gate = nn.Linear(20 + 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.95 / 0.05))
        self.norms = nn.ModuleList([RMSNorm(D_G) for _ in range(K_ROUNDS)])
        self.rounds = nn.ModuleList([nn.Linear(D_G, D_G, bias=False) for _ in range(K_ROUNDS)])
        for w in self.rounds:
            nn.init.zeros_(w.weight)
        self.z_norm = RMSNorm(D_G)
        self.P = nn.Linear(6, D_G)
        self.readout = nn.Linear(D_G + 6 + D_G, d_model)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def cell_state(self, T, uc):
        F, M = T['F'][uc].float(), T['M'][uc].float()
        chrom = torch.cat([F * mask_per_feature(M), M], -1)                     # [U, N, 20]
        ce = torch.stack([T['E'][uc].float(), T['Em'][uc].float()], -1)          # [U, N, 2]
        U = len(uc)
        x = torch.cat([T['gv'].float().unsqueeze(0).expand(U, -1, -1), ce, chrom], -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.node_in(x)
            r = torch.sigmoid(self.gate(torch.cat([chrom, ce], -1))).sqrt()     # [U, N, 1]
            for norm, W in zip(self.norms, self.rounds):
                z = z + W(r * spmm(T['A'], r * norm(z)))
        return z, r

    def target_signal(self, b, p_pred, T):
        """[B, N, 2]: known targets (any = 1, sign) for rows with tgt_known, else the predicted probabilities (sign 0)."""
        B, N = b['cell_row'].shape[0], self.n_nodes
        dev = b['cell_row'].device
        known = b['tgt_known'].float()[:, None]
        t_any = torch.zeros(B, N + 1, device=dev).scatter_(1, b['tgt_node'], 1.0)[:, :N]
        t_sign = torch.zeros(B, N + 1, device=dev).scatter_(1, b['tgt_node'], b['tgt_sign'].float())[:, :N]
        p_any = torch.zeros(B, N, device=dev)
        p_any[:, T['vocab']] = p_pred.float()
        return torch.stack([known * t_any + (1 - known) * p_any, known * t_sign], -1)

    def forward(self, b, p_pred):
        T = self.tables.get(b['cell_row'].device)
        uc, inv = torch.unique(b['cell_row'], return_inverse=True)
        z, r = self.cell_state(T, uc)
        lm = T['lm']
        zl = self.z_norm(z[:, lm])[inv]                                          # [B, 978, d_g]
        B = inv.shape[0]
        if self.targets_on:
            with torch.autocast(device_type=zl.device.type, enabled=False):
                t = self.target_signal(b, p_pred, T)
                rr = r[inv]                                                      # the row's cell's gate
                hops = [t]
                for _ in range(2):
                    hops.append(rr * spmm(T['A'], rr * hops[-1]))
                s = torch.cat([h[:, lm] for h in hops], -1) / T['s_scale'].clamp(min=1e-12)
        else:
            s = torch.zeros(B, len(lm), 6, device=zl.device)
        with torch.autocast(device_type=zl.device.type, enabled=False):
            out = self.readout(torch.cat([zl, s, zl * self.P(s)], -1))
        return out, {'gate': (r ** 2).squeeze(-1), 'n_cells': len(uc), 'inv': inv}


class LincsV11(LincsV10C):
    def __init__(self, cfg, M_pathway, ppi, gene_vectors, fg):
        super().__init__(cfg, M_pathway, ppi, gene_vectors)                      # the base first: v10-C's RNG stream
        seed = torch.initial_seed()
        with torch.random.fork_rng(devices=[]):                     # leaves the global stream where v10-C leaves it
            torch.default_generator.manual_seed((seed + FG_SEED_OFFSET) % 2 ** 63)     # CPU only (review 099 C1)
            self.fg = FullGraphModule(cfg.d_model, fg)
            self.target_head = nn.Linear(cfg.d_model, len(fg['vocab']))
        with torch.no_grad():                                       # each gene's prior rate among the known compounds
            q = torch.as_tensor(fg['vocab_prior'], dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
            self.target_head.bias.copy_(torch.log(q / (1 - q)))

    def forward(self, b, return_aux=False):
        # LincsV10C.forward with FG inserted after h_basal (marked); everything else is the base's, op for op
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
        logits = self.target_head(g)                                # v11: structure -> targets, from the dose-free token
        fgh, fga = self.fg(b, torch.sigmoid(logits.float()).detach())   # v11: FG
        h = h + fgh.to(h.dtype)                                     # v11: zero at init
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
        out = {'delta': delta, 'abs': x_ctl + delta, 'drug': drug, 'basal': basal, 'dose_scale': s,
               'target_logits': logits, 'fg_gate_mean': fga['gate'].mean(1)[fga['inv']]}
        if return_aux:
            return out, {'gamma': torch.stack(gammas, 1), 'a_ppi': a_ppi, 'dose_scale': s, 'pathways': pathways,
                         'fg_gate': fga['gate'], 'fg_cells': fga['n_cells'], 'fg_readout': fgh}
        return out


def v11_loss(out, b, cfg, lam=0.1, targets_on=True):
    loss, parts = v10c_loss(out, b, cfg)
    ok = b['tgt_known']
    if targets_on and bool(ok.any()):
        l_t = nn.functional.binary_cross_entropy_with_logits(out['target_logits'][ok].float(), b['head_y'][ok])
        loss = loss + lam * l_t
        parts['target_bce'] = float(l_t.detach())
    parts['n_target_rows'] = int(ok.sum())
    return loss, parts
