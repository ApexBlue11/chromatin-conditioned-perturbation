# -*- coding: utf-8 -*-
"""
moa.py — pathway-shaped information flow = a crude MoA (per the design: gene<->gene attention is biased,
not bounded, by the co-pathway/STRING priors, so signal is softly channeled along the biology). Two Qs:

  1. Does the model actually LEAN ON the priors? Report the learned bias strength lambda =
     softplus(log_lambda) per gene<->gene layer, per prior-head, per prior (co-pathway vs STRING-PPI).
     lambda ~ 0 => that head ignores the prior (became a free "discovery" head); large => flow is shaped
     by that biological graph.
  2. Do the PRIOR-biased heads actually concentrate attention ON pathway/PPI edges (support) more than the
     FREE heads and more than the random density? If yes, the gene->gene flow follows the biology (the MoA
     substrate works), not just "which pathway lit up".

Kaggle CPU. Writes moa.json.
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig
from data import LincsDataset, collate
from model import LincsCrossAttn
from train import make_dataconfig


def main():
    device = "cpu"
    dc = make_dataconfig(); mc = ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).to(device).eval()
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    model.load_state_dict(torch.load(ck[0], map_location=device)["model"])
    print("loaded", ck[0], flush=True)

    # ---- 1. lambda: does the model lean on the priors? ----
    print("\n== lambda (prior-bias strength) per gene<->gene layer  [prior 0=co-pathway, 1=STRING-PPI] ==",
          flush=True)
    layers = [("base", i, l) for i, l in enumerate(model.base)] + \
             [("perturb", i, l) for i, l in enumerate(model.perturb)]
    lam = {}
    for tag, i, layer in layers:
        ll = F.softplus(layer.attn.log_lambda).detach().cpu().numpy()   # [n_prior_heads, 2]
        lam[f"{tag}_{i}"] = ll.tolist()
        for h in range(ll.shape[0]):
            print(f"  {tag} layer{i} prior-head{h}: copathway={ll[h, 0]:.3f}  STRING={ll[h, 1]:.3f}", flush=True)

    # ---- 2. on-support attention mass: prior heads vs free heads (last perturb layer) ----
    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    support = model.support.cpu().numpy().astype(np.float32)            # [G,G]
    idx = np.where(ds.strength >= 1.0)[0]
    idx = np.random.default_rng(0).choice(idx, 16, replace=False)
    b = collate([ds[i] for i in idx], mc.max_atoms)
    with torch.no_grad():
        _, aux = model({k: v.to(device) for k, v in b.items()}, return_attn=True)
    ag = aux["alpha_gene"].detach().cpu().numpy()                       # [B,H,G,G] last perturb layer
    onfrac = (ag * support[None, None]).sum(-1).mean((0, 2))            # per head: mean on-support mass
    dens = float(support.mean())
    print("\n== on-support attention mass by head (last perturb layer) ==", flush=True)
    print(f"   support density (random baseline) = {dens:.4f}", flush=True)
    for h in range(len(onfrac)):
        tag = "PRIOR" if h < mc.n_prior_heads else "free "
        print(f"  head{h} ({tag}): on-support frac = {onfrac[h]:.3f}  ({onfrac[h] / dens:.1f}x random)", flush=True)
    prior_avg = float(onfrac[:mc.n_prior_heads].mean()); free_avg = float(onfrac[mc.n_prior_heads:].mean())
    print(f"  => PRIOR heads {prior_avg:.3f} ({prior_avg/dens:.1f}x) vs FREE heads {free_avg:.3f} "
          f"({free_avg/dens:.1f}x) vs random {dens:.4f}", flush=True)
    print("  (PRIOR >> FREE >~ random => flow is shaped by the biology = MoA substrate works)", flush=True)

    json.dump({"lambda": lam, "onsupport_frac_per_head": onfrac.tolist(),
               "prior_heads_avg": prior_avg, "free_heads_avg": free_avg, "support_density": dens},
              open(("/kaggle/working/moa.json" if os.path.isdir("/kaggle/working") else "moa.json"), "w"),
              indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
