# -*- coding: utf-8 -*-
"""
v7 code-vs-design checks. Each asserts a property V7_PLAN claims. Run BEFORE any accelerator time:
    python model/v7/test_v7.py
Exits nonzero on failure.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v6"))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import torch

from config_v7 import V7Config, V6DataConfig
from model_v7 import LincsV7, aux_targets
from modules_v7 import StochasticDepth, PPIMessagePassing, UncertaintyWeighting, QKNormAttention

R = []
def check(name, cond, detail=""):
    R.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def batch(cfg, B=4, dead=50, n_atoms=30):
    b = {"X": torch.randn(B, cfg.n_genes), "E": torch.randn(B, cfg.n_genes, 3),
         "r": torch.rand(B, cfg.n_genes), "atoms": torch.randn(B, cfg.max_atoms, cfg.d_atom),
         "atom_mask": torch.zeros(B, cfg.max_atoms, dtype=torch.bool),
         "u_feats": torch.randn(B, cfg.d_global),
         "cell_ctx": torch.eye(cfg.d_cell_ctx)[torch.randint(0, cfg.d_cell_ctx, (B,))],
         "dose": torch.rand(B), "time": torch.rand(B), "Y": torch.randn(B, cfg.n_genes)}
    b["atom_mask"][:, :n_atoms] = True
    b["r"][:, :dead] = 0.0
    return b


def main():
    cfg = V7Config(); dc = V6DataConfig()
    M = np.load(os.path.join(dc.root, dc.m_reactome_path))
    ppi = np.load(os.path.join(dc.root, dc.ppi_path))
    torch.manual_seed(0)
    m = LincsV7(cfg, M, ppi).eval()
    b = batch(cfg)

    with torch.no_grad():
        y = m(b); y2, aux = m(b, return_aux=True)
    check("Yhat shape [B,G] and finite", tuple(y.shape) == (4, cfg.n_genes) and bool(torch.isfinite(y).all()))
    check("return_aux does not change the prediction", torch.allclose(y, y2, atol=1e-6))

    # --- the auxiliary heads that make the priors load-bearing ---
    check("pathway aux head outputs one score per NAMED pathway [B,P]",
          tuple(aux["pathway_pred"].shape) == (4, M.shape[0]), str(tuple(aux["pathway_pred"].shape)))
    check("chromatin aux head outputs one score per gene [B,G]",
          tuple(aux["epi_pred"].shape) == (4, cfg.n_genes))
    t_path, t_epi = aux_targets(b["Y"], torch.as_tensor(M, dtype=torch.float32) /
                                torch.as_tensor(M, dtype=torch.float32).sum(1, keepdim=True).clamp(min=1))
    check("aux targets match head shapes", t_path.shape == aux["pathway_pred"].shape
          and t_epi.shape == aux["epi_pred"].shape)
    check("pathway target is a masked mean of |Y| (non-negative, bounded by max|Y|)",
          bool((t_path >= 0).all() and (t_path <= b["Y"].abs().max() + 1e-5).all()))

    # the chromatin aux head must read ONLY the chromatin branch, or it teaches that branch nothing
    b2 = dict(b); b2["X"] = torch.randn_like(b["X"])
    with torch.no_grad():
        _, aux2 = m(b2, return_aux=True)
    check("chromatin aux head is INDEPENDENT of baseline expression (reads only its own branch)",
          torch.allclose(aux["epi_pred"], aux2["epi_pred"], atol=1e-6))
    b3 = dict(b); b3["E"] = torch.randn_like(b["E"])
    with torch.no_grad():
        _, aux3 = m(b3, return_aux=True)
    check("chromatin aux head DOES respond to chromatin",
          not torch.allclose(aux["epi_pred"], aux3["epi_pred"], atol=1e-5))

    # --- STRING PPI actually used, and hard-masked ---
    check("PPI layer present and used", m.ppi is not None)
    pp = PPIMessagePassing(ppi, cfg.d_model).eval()   # eval(): dropout would make repeat calls differ
    h = torch.randn(2, cfg.n_genes, cfg.d_model)
    check("PPI layer is an exact no-op at init (zero-init out)", float(pp(h).abs().max()) == 0.0)
    with torch.no_grad():
        pp.w.weight.normal_(0, 0.1)
    out = pp(h)
    iso = (pp.has_edge.squeeze(-1) == 0)
    check("genes with NO STRING edge receive exactly 0", float(out[:, iso].abs().max()) == 0.0,
          f"{int(iso.sum())} isolated genes of {cfg.n_genes}")
    A = torch.as_tensor(ppi, dtype=torch.float32).clone(); A.fill_diagonal_(0)
    g = int(torch.nonzero(A.sum(1) > 0)[0])
    nbr = (A[g] > 0)
    h2 = h.clone(); h2[:, ~nbr] += 5.0; h2[:, g] = h[:, g]
    check("a gene's PPI message depends only on its NEIGHBOURS",
          torch.allclose(pp(h)[:, g], pp(h2)[:, g], atol=1e-4)
          if int(nbr.sum()) > 0 else True, f"gene {g} has {int(nbr.sum())} neighbours")

    # --- stochastic depth: identity in eval, stochastic in train ---
    sd = StochasticDepth(0.5); x = torch.randn(64, 8)
    sd.eval(); check("stochastic depth is identity at eval", torch.allclose(sd(x), x))
    sd.train()
    drops = [float((sd(x) == 0).all(dim=1).float().mean()) for _ in range(20)]
    check("stochastic depth drops whole examples during training", 0.2 < float(np.mean(drops)) < 0.8,
          f"mean drop rate {np.mean(drops):.2f}")

    # --- uncertainty weighting ---
    uw = UncertaintyWeighting(3)
    l = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    tot, w = uw(l)
    check("uncertainty weighting starts at unit weights", all(abs(v - 1.0) < 1e-6 for v in w.values()))
    tot.backward()
    check("uncertainty weighting log_var receives gradient", uw.log_var.grad is not None)

    # --- QK-norm attention sanity ---
    at = QKNormAttention(cfg.d_model, cfg.n_heads)
    o = at(torch.randn(2, 50, cfg.d_model))
    check("QK-norm attention output finite and shaped", tuple(o.shape) == (2, 50, cfg.d_model)
          and bool(torch.isfinite(o).all()))

    # --- kept-from-v6 invariants that must not regress ---
    check("signed chromatin term is exactly 0 where reliability r==0",
          float(aux["epi_contrib"][:, :50].abs().max()) == 0.0)
    check("pathway activations are [B,P,d_pathway] over NAMED nodes",
          tuple(aux["pathway_activations"].shape) == (4, M.shape[0], cfg.d_pathway))

    # --- gradients reach every new component, THROUGH THE REAL MULTI-TASK LOSS ---
    # Composed exactly as train_v7_gpu does, because `task_weights` lives in the trainer's loss and not in
    # forward(): a forward-only backward would leave its gradient None and the check would be vacuous.
    import losses as Lz
    import torch.nn.functional as Fz
    m.train(); yg, auxg = m(b, return_aux=True)
    Mn = torch.as_tensor(M, dtype=torch.float32)
    Mn = Mn / Mn.sum(1, keepdim=True).clamp(min=1)
    tp, te = aux_targets(b["Y"], Mn)
    l_main = Lz.weighted_huber(yg, b["Y"], delta=cfg.huber_delta)
    l_path = Fz.huber_loss(auxg["pathway_pred"], tp)
    l_epi = Fz.huber_loss(auxg["epi_pred"], te)
    total, _ = m.task_weights([l_main, l_path, l_epi])
    total.backward()
    check("the multi-task loss is finite and backprops", bool(torch.isfinite(total)))
    watched = {"ppi.w": m.ppi.w.weight, "aux.path": m.aux.path[0].weight, "aux.epi": m.aux.epi[0].weight,
               "task_weights": m.task_weights.log_var, "pathway.w_in": m.pathway.w_in.weight,
               "enc_epi": m.enc_epi.net[0].weight}
    for k, p in watched.items():
        check(f"gradient present + finite: {k}", p.grad is not None and bool(torch.isfinite(p.grad).all()))

    n_fail = sum(1 for r in R if not r)
    print(f"\n{'='*58}\n{len(R)-n_fail}/{len(R)} checks passed" + (f" | {n_fail} FAILED" if n_fail else " | ALL PASS"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
