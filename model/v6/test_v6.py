# -*- coding: utf-8 -*-
"""
v6 code-vs-design checks. Each asserts a property ARCHITECTURE.md claims. Run BEFORE any accelerator time:
    python model/v6/test_v6.py
Exits nonzero on failure.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from config_v6 import V6Config, V6DataConfig
from model_v6 import LincsV6
from modules import PathwayBottleneck

R = []
def check(name, cond, detail=""):
    R.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def batch(cfg, B=4, dead_genes=50, n_atoms=30):
    b = {"X": torch.randn(B, cfg.n_genes), "E": torch.randn(B, cfg.n_genes, 3),
         "r": torch.rand(B, cfg.n_genes), "atoms": torch.randn(B, cfg.max_atoms, cfg.d_atom),
         "atom_mask": torch.zeros(B, cfg.max_atoms, dtype=torch.bool),
         "u_feats": torch.randn(B, cfg.d_global),
         "cell_ctx": torch.eye(cfg.d_cell_ctx)[torch.randint(0, cfg.d_cell_ctx, (B,))],
         "dose": torch.rand(B), "time": torch.rand(B)}
    b["atom_mask"][:, :n_atoms] = True
    b["r"][:, :dead_genes] = 0.0
    return b


def main():
    cfg = V6Config(); dc = V6DataConfig()
    M = np.load(os.path.join(dc.root, dc.m_reactome_path))
    torch.manual_seed(0)
    m = LincsV6(cfg, M).eval()
    b = batch(cfg)

    with torch.no_grad():
        y = m(b); y2, aux = m(b, return_interp=True)
    check("Yhat shape [B,G] and finite", tuple(y.shape) == (4, cfg.n_genes) and bool(torch.isfinite(y).all()))
    check("return_interp does not change the prediction", torch.allclose(y, y2, atol=1e-6))

    # --- the interpretable readout v6 exists to produce ---
    pa = aux["pathway_activations"]
    check("pathway activations are [B,P,d_pathway] over NAMED Reactome nodes",
          tuple(pa.shape) == (4, M.shape[0], cfg.d_pathway), str(tuple(pa.shape)))

    # --- chromatin discipline: no data => no contribution ---
    check("signed chromatin term is exactly 0 where reliability r==0",
          float(aux["epi_contrib"][:, :50].abs().max()) == 0.0)
    hb = m.enc_epi(b["E"], b["r"])
    check("chromatin ENCODER output is exactly 0 where r==0", float(hb[:, :50].abs().max()) == 0.0)

    # --- the hard mask: information cannot route around pathway membership ---
    pb = PathwayBottleneck(M, cfg.d_model, cfg.d_pathway, cfg.d_epi)
    h = torch.randn(2, cfg.n_genes, cfg.d_model); E = torch.randn(2, cfg.n_genes, 3)
    out0, _ = pb(h, E)
    check("pathway bottleneck is an exact no-op at init (zero-init out)", float(out0.abs().max()) == 0.0)
    nop = (pb.has_pathway.squeeze(-1) == 0)
    with torch.no_grad():
        pb.w_out.weight.normal_(0, 0.1); pb.w_out.bias.normal_(0, 0.1)
    out1, p1 = pb(h, E, return_pathways=True)
    check("genes in NO pathway receive exactly 0", float(out1[:, nop].abs().max()) == 0.0,
          f"{int(nop.sum())} such genes")
    g = int(torch.nonzero(pb.M_in[0] > 0)[0])
    h2 = h.clone(); h2[:, g] += 10.0
    _, p2 = pb(h2, E, return_pathways=True)
    changed = (p2 - p1).abs().amax(dim=(0, 2)) > 1e-5
    member = pb.M_in[:, g] > 0
    check("perturbing a gene changes ONLY pathways containing it (hard mask holds)",
          bool((changed == member).all()), f"changed {int(changed.sum())} == member of {int(member.sum())}")

    # --- late integration: modalities must be separately ablatable ---
    b_noepi = dict(b); b_noepi["E"] = torch.zeros_like(b["E"]); b_noepi["r"] = torch.zeros_like(b["r"])
    with torch.no_grad():
        y_noepi = m(b_noepi)
    check("ablating chromatin alone changes the prediction (modalities are separable)",
          not torch.allclose(y, y_noepi, atol=1e-5))

    # --- XLA/TPU: static shapes must not alter results ---
    b_pad = dict(b)   # already padded to max_atoms; a tighter batch must agree
    n = int(b["atom_mask"].sum(1).max())
    b_tight = dict(b); b_tight["atoms"] = b["atoms"][:, :n]; b_tight["atom_mask"] = b["atom_mask"][:, :n]
    with torch.no_grad():
        check("prediction invariant to atom padding (no attention leak onto pads)",
              torch.allclose(m(b_tight), m(b_pad), atol=1e-5),
              f"max dev {float((m(b_tight)-m(b_pad)).abs().max()):.2e}")

    # --- gradients ---
    m.train(); yg = m(b); yg.sum().backward()
    watched = {"pathway.w_in": m.pathway.w_in.weight, "pathway.w_out": m.pathway.w_out.weight,
               "enc_epi": m.enc_epi.net[0].weight, "enc_base": m.enc_base.net[0].weight,
               "fuse.epi_gate": m.fuse.epi_gate, "head.epi_head": m.head.epi_head[0].weight}
    for k, p in watched.items():
        check(f"gradient present + finite: {k}", p.grad is not None and bool(torch.isfinite(p.grad).all()))

    n_fail = sum(1 for r in R if not r)
    print(f"\n{'='*56}\n{len(R)-n_fail}/{len(R)} checks passed" + (f" | {n_fail} FAILED" if n_fail else " | ALL PASS"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
