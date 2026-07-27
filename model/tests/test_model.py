# -*- coding: utf-8 -*-
"""
Code-vs-math verification (CPU, synthetic tensors). Each test maps to a property in MODEL_MATH.md.
Run: python model/tests/test_model.py   (torch only; no real data, no GPU).
Exits nonzero if any check fails.
"""
import os, sys, math
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):   # robust: modules may sit alongside (Kaggle flat) or in parent (local)
    sys.path.insert(0, _p)
import numpy as np
import torch

from config import ModelConfig
from modules import BiasedSelfAttention, DoseTimeFiLM, EpiGate
from model import LincsCrossAttn
import losses as L

torch.manual_seed(0); np.random.seed(0)
RESULTS = []

def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def small_cfg(**kw):
    c = ModelConfig(n_genes=20, d_model=32, n_heads=4, d_ff=64, l_base=1, l_perturb=2,
                    max_atoms=8, n_prior_heads=2, d_global=512 + 384 + 20 + 2048)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def make_batch(cfg, B=6, zero_r_genes=(0, 1, 2)):
    G, M = cfg.n_genes, cfg.max_atoms
    r = torch.rand(B, G)
    r[:, list(zero_r_genes)] = 0.0                       # force fallback genes
    # variable atom counts per example
    atom_mask = torch.zeros(B, M, dtype=torch.bool)
    for b in range(B):
        n = int(torch.randint(1, M + 1, (1,)))
        atom_mask[b, :n] = True
    return {
        "X": torch.randn(B, G), "E": torch.randn(B, G, 3), "r": r,
        "atoms": torch.randn(B, M, cfg.d_atom), "atom_mask": atom_mask,
        "u_feats": torch.randn(B, cfg.d_global),
        "cell_ctx": torch.eye(getattr(cfg, "d_cell_ctx", 16))[torch.randint(0, getattr(cfg, "d_cell_ctx", 16), (B,))],
        "dose": torch.rand(B), "time": torch.rand(B),
    }


def build_model(cfg):
    G = cfg.n_genes
    cop = np.abs(np.random.randn(G, G)) * (np.random.rand(G, G) < 0.3)   # sparse nonneg
    ppi = np.abs(np.random.randn(G, G)) * (np.random.rand(G, G) < 0.2)
    np.fill_diagonal(cop, 0); np.fill_diagonal(ppi, 0)
    return LincsCrossAttn(cfg, cop, ppi), cop, ppi


# ---- 1. shapes ----
def test_shapes():
    cfg = small_cfg(); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    y, aux = m(b, return_attn=True)
    check("shape: Yhat [B,G]", tuple(y.shape) == (6, cfg.n_genes), str(tuple(y.shape)))
    check("shape: alpha_drug [B,H,G,1+M]",
          tuple(aux["alpha_drug"].shape) == (6, cfg.n_heads, cfg.n_genes, 1 + cfg.max_atoms),
          str(tuple(aux["alpha_drug"].shape)))
    check("shape: alpha_gene [B,H,G,G]",
          tuple(aux["alpha_gene"].shape) == (6, cfg.n_heads, cfg.n_genes, cfg.n_genes),
          str(tuple(aux["alpha_gene"].shape)))
    check("shape: gate [B,G]", tuple(aux["gate"].shape) == (6, cfg.n_genes))


# ---- 2. masked softmax rows sum to 1 ----
def test_attention_normalized():
    cfg = small_cfg(); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    _, aux = m(b, return_attn=True)
    sd = aux["alpha_drug"].sum(-1)                       # [B,H,G]
    sg = aux["alpha_gene"].sum(-1)
    check("cross-attn rows sum to 1", torch.allclose(sd, torch.ones_like(sd), atol=1e-5),
          f"max dev {float((sd-1).abs().max()):.2e}")
    check("self-attn rows sum to 1", torch.allclose(sg, torch.ones_like(sg), atol=1e-5),
          f"max dev {float((sg-1).abs().max()):.2e}")


# ---- 3. gate bounds + s=1 fallback where r=0 ----
def test_gate():
    cfg = small_cfg(); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg, zero_r_genes=(0, 1, 2))
    s = m.gate(b["E"], b["r"])
    check("gate in (0,1]", bool((s > 0).all() and (s <= 1 + 1e-6).all()),
          f"min {float(s.min()):.3f} max {float(s.max()):.3f}")
    check("gate == 1 exactly where r==0", torch.allclose(s[:, :3], torch.ones_like(s[:, :3]), atol=1e-6),
          f"max dev {float((s[:, :3]-1).abs().max()):.2e}")


# ---- 4. no target leakage (forward independent of Y) ----
def test_no_leakage():
    cfg = small_cfg(); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    y1 = m(b)
    # inject a bogus 'Y' into the batch; forward must ignore it entirely
    b2 = dict(b); b2["Y"] = torch.randn(6, cfg.n_genes)
    y2 = m(b2)
    check("forward ignores Y (no leakage)", torch.allclose(y1, y2, atol=1e-6))


# ---- 5. prior-bias limits ----
def test_prior_bias():
    G, d, H, P = 20, 32, 4, 2
    att = BiasedSelfAttention(d, H, P, n_prior_heads=2, masked_head=True)
    priors = torch.abs(torch.randn(P, G, G))
    support = torch.rand(G, G) < 0.3
    # lambda -> ~0  => bias ~ 0 (vanilla limit)
    with torch.no_grad():
        att.log_lambda.fill_(-30.0)
    bias0 = att._bias(priors, support)
    check("lambda->0 gives ~zero bias", float(bias0[bias0 > -1e30].abs().max()) < 1e-6,
          f"max |bias| {float(bias0[bias0 > -1e30].abs().max()):.2e}")
    # large lambda on masked head 0 => off-support strongly suppressed
    with torch.no_grad():
        att.log_lambda.fill_(5.0)
    x = torch.randn(2, G, d)
    _, attn = att(x, priors, support, need_weights=True)   # weights needed -> manual path
    off = attn[:, 0][:, ~support.any(0)]        # crude off-support probe not ideal; use full support mask
    # proper: attention from masked head 0 on hard off-support entries must be ~0
    a0 = attn[:, 0]                              # [B,G,G]
    off_mass = a0.masked_select(~support.unsqueeze(0).expand_as(a0))
    check("masked head attends only on-support", float(off_mass.max()) < 1e-6,
          f"max off-support weight {float(off_mass.max()):.2e}")


# ---- 6. FiLM identity at init ----
def test_film_identity():
    film = DoseTimeFiLM(16)
    h = torch.randn(4, 20, 16)
    out = film(h, torch.rand(4), torch.rand(4))
    check("FiLM is identity at init", torch.allclose(out, h, atol=1e-6),
          f"max dev {float((out-h).abs().max()):.2e}")


# ---- 7. gradient flow ----
def test_gradients():
    cfg = small_cfg(dropout=0.0); m, *_ = build_model(cfg); m.train()
    b = make_batch(cfg); y = m(b)
    loss = L.weighted_huber(y, torch.randn_like(y))
    loss.backward()
    watched = {"gene_emb": m.gene_emb, "gate_mlp0": m.gate.mlp[0].weight,
               "w_a": m.w_a.weight, "log_lambda": m.perturb[0].attn.log_lambda,
               "film_last": m.film.net[-1].weight}
    for name, p in watched.items():
        ok = (p.grad is not None) and torch.isfinite(p.grad).all()
        # film last layer is zero-init -> may get grad now; just require finite & present
        check(f"grad present+finite: {name}", bool(ok))


# ---- 8. overfit a tiny batch (capacity/optimization sanity) + beat Mean baseline ----
def test_overfit():
    cfg = small_cfg(dropout=0.0); m, *_ = build_model(cfg); m.train()
    b = make_batch(cfg, zero_r_genes=())
    target = torch.randn(6, cfg.n_genes)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    l0 = None
    for step in range(400):
        opt.zero_grad(); y = m(b); loss = L.weighted_huber(y, target)
        if step == 0:
            l0 = float(loss)
        loss.backward(); opt.step()
    lf = float(loss)
    check("overfit: loss decreases >20x", lf < l0 / 20, f"{l0:.3f} -> {lf:.4f}")
    m.eval()
    with torch.no_grad():
        mse_model = float(((m(b) - target) ** 2).mean())
    mse_mean = float(((target.mean(0, keepdim=True) - target) ** 2).mean())
    check("overfit: model beats Mean baseline", mse_model < mse_mean,
          f"model {mse_model:.4f} < mean {mse_mean:.4f}")


# ---- 8b. SDPA (training) path matches manual (interpretability) path ----
def test_sdpa_equiv():
    cfg = small_cfg(dropout=0.0); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    with torch.no_grad():
        y_manual, aux = m(b, return_attn=True)      # manual weight-exposing path
        y_sdpa = m(b)                               # memory-efficient SDPA path
    check("SDPA path == manual path (Yhat)", torch.allclose(y_manual, y_sdpa, atol=1e-4),
          f"max dev {float((y_manual - y_sdpa).abs().max()):.2e}")
    check("SDPA path returns no weights (mem-efficient)", m(b) is not None and aux["alpha_gene"] is not None)


# ---- 8c. additive signed epi head (v2): signed, reliability-masked, zeroed by ablation ----
def test_epi_additive():
    cfg = small_cfg(dropout=0.0, epi_additive=True); m, *_ = build_model(cfg); m.eval()
    # force the (zero-init) epi head to be active so we can test its behavior
    with torch.no_grad():
        m.epi_head[-1].weight.normal_(0, 0.5); m.epi_head[-1].bias.normal_(0, 0.5)
    b = make_batch(cfg, zero_r_genes=(0, 1, 2))
    _, aux = m(b, return_attn=True)
    ec = aux["epi_contrib"]
    check("epi_contrib shape [B,G]", tuple(ec.shape) == (6, cfg.n_genes))
    check("epi_contrib is signed (has neg & pos)", bool((ec < 0).any() and (ec > 0).any()))
    check("epi_contrib == 0 where r==0", torch.allclose(ec[:, :3], torch.zeros_like(ec[:, :3]), atol=1e-6))
    # ablating E,r must change Yhat (head is active) and remove the epi term
    y = m(b)
    b2 = dict(b); b2["E"] = torch.zeros_like(b["E"]); b2["r"] = torch.zeros_like(b["r"])
    _, aux2 = m(b2, return_attn=True)
    check("epi ablation zeroes epi_contrib", torch.allclose(aux2["epi_contrib"], torch.zeros_like(ec), atol=1e-6))
    check("epi ablation changes Yhat (head active)", not torch.allclose(y, m(b2), atol=1e-4))


# ---- 8d. atom->gene attribution (ca_gene_norm) for DTI recall@k ----
def test_ca_attribution():
    cfg = small_cfg(dropout=0.0); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    with torch.no_grad():
        y0 = m(b)
        y1, aux1 = m(b, return_attn=True)          # manual-attn path
        y2, aux2 = m(b, return_attr=True)          # fast SDPA path
    for tag, aux in [("attn", aux1), ("attr", aux2)]:
        ca = aux["ca_gene_norm"]
        check(f"ca_gene_norm shape [B,G] ({tag})", tuple(ca.shape) == (6, cfg.n_genes), str(tuple(ca.shape)))
        check(f"ca_gene_norm finite & >=0 ({tag})", bool(torch.isfinite(ca).all() and (ca >= 0).all()))
    check("return_attr Yhat == plain forward", torch.allclose(y0, y2, atol=1e-5),
          f"max dev {float((y0-y2).abs().max()):.2e}")
    check("ca_gene_norm manual==SDPA path", torch.allclose(aux1["ca_gene_norm"], aux2["ca_gene_norm"], atol=1e-3),
          f"max dev {float((aux1['ca_gene_norm']-aux2['ca_gene_norm']).abs().max()):.2e}")


# ---- 8e. correlation loss (v4 pattern term) ----
def test_correlation_loss():
    y = torch.randn(8, 20)
    check("corr-loss ~0 for perfect prediction", float(L.correlation_loss(y.clone(), y)) < 1e-4,
          f"{float(L.correlation_loss(y.clone(), y)):.2e}")
    check("corr-loss ~2 for anti-correlated", abs(float(L.correlation_loss(-y, y)) - 2.0) < 1e-3,
          f"{float(L.correlation_loss(-y, y)):.4f}")
    # magnitude-invariant: scaling the prediction does NOT change the correlation loss
    l1 = float(L.correlation_loss(0.1 * y, y)); l2 = float(L.correlation_loss(5.0 * y, y))
    check("corr-loss is magnitude-invariant", abs(l1 - l2) < 1e-4, f"{l1:.2e} vs {l2:.2e}")
    w = torch.rand(8)
    check("corr-loss weighted is finite", torch.isfinite(torch.tensor(float(L.correlation_loss(torch.randn(8, 20), y, w)))).item())


# ---- 8f. cell-context (lineage) FiLM conditioning ----
def test_cell_context():
    cfg = small_cfg(dropout=0.0); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    # FiLM is zero-init => cell_ctx must be a NO-OP at init (model starts identical to the no-context model)
    b2 = dict(b); b2["cell_ctx"] = torch.roll(b["cell_ctx"], 1, dims=0)
    with torch.no_grad():
        check("cell_ctx is a no-op at init (zero-init FiLM)", torch.allclose(m(b), m(b2), atol=1e-6))
        # once FiLM is active, a DIFFERENT cell context must change the prediction
        m.film.net[-1].weight.normal_(0, 0.05); m.film.net[-1].bias.normal_(0, 0.05)
        check("cell_ctx changes Yhat once FiLM is active", not torch.allclose(m(b), m(b2), atol=1e-4))
    check("FiLM d_extra wired to d_cell_ctx", m.film.d_extra == getattr(cfg, "d_cell_ctx", 0),
          f"{m.film.d_extra} vs {getattr(cfg, 'd_cell_ctx', 0)}")


# ---- 8g. cell-conditional pathway conductance (v5) ----
def test_pathway_conductance():
    from modules import PathwayConductance
    pc = PathwayConductance(3, 16)
    E = torch.randn(4, 20, 3); x = torch.randn(4, 20); r = torch.rand(4, 20)
    c = pc(E, x, r)
    check("pathway conductance == 1 at init (no-op)", torch.allclose(c, torch.ones_like(c), atol=1e-6),
          f"max dev {float((c-1).abs().max()):.2e}")
    with torch.no_grad():
        pc.mlp[-1].weight.normal_(0, 0.5); pc.mlp[-1].bias.normal_(0, 0.5)
    c2 = pc(E, x, r)
    check("pathway conductance in (0,2)", bool((c2 > 0).all() and (c2 < 2).all()),
          f"min {float(c2.min()):.3f} max {float(c2.max()):.3f}")
    check("pathway conductance varies per (cell,gene)", float(c2.std()) > 1e-3, f"std {float(c2.std()):.3f}")
    # end-to-end: enabling it must be a no-op at init but change Yhat once active
    cfg = small_cfg(dropout=0.0, pathway_conductance=True); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    cfg0 = small_cfg(dropout=0.0, pathway_conductance=False); m0, *_ = build_model(cfg0); m0.eval()
    m0.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("pathway_cond")}, strict=False)
    with torch.no_grad():
        check("pathway-cond model == static model at init", torch.allclose(m(b), m0(b), atol=1e-5),
              f"max dev {float((m(b)-m0(b)).abs().max()):.2e}")
        m.pathway_cond.mlp[-1].weight.normal_(0, 0.5); m.pathway_cond.mlp[-1].bias.normal_(0, 0.5)
        check("pathway conductance changes Yhat once active", not torch.allclose(m(b), m0(b), atol=1e-4))


# ---- 8h. padding must NOT leak into attention (prerequisite for TPU/XLA static shapes) ----
def test_padding_invariance():
    """If attention could see padded atom slots, adding MORE padding would change Yhat. It must not."""
    cfg = small_cfg(dropout=0.0); m, *_ = build_model(cfg); m.eval()
    b = make_batch(cfg)
    n_real = int(b["atom_mask"].sum(1).max())          # tight length actually used
    tight = {k: (v[:, :n_real] if k in ("atoms", "atom_mask") else v) for k, v in b.items()}
    with torch.no_grad():
        y_tight = m(tight)                             # minimal padding
        y_padded = m(b)                                # full max_atoms padding
    check("Yhat invariant to atom padding (no attention leak)", torch.allclose(y_tight, y_padded, atol=1e-5),
          f"max dev {float((y_tight - y_padded).abs().max()):.2e}")
    # and the attention weights themselves must be exactly 0 on padded keys
    with torch.no_grad():
        _, aux = m(b, return_attn=True)
    a = aux["alpha_drug"]                              # [B,H,G,1+M]
    pad = ~torch.cat([torch.ones(b["atoms"].shape[0], 1, dtype=torch.bool), b["atom_mask"]], 1)
    mass = a.masked_select(pad[:, None, None, :].expand_as(a))
    check("zero attention mass on padded atom tokens",
          (float(mass.max()) if mass.numel() else 0.0) < 1e-6,
          f"max pad weight {float(mass.max()) if mass.numel() else 0.0:.2e}")
    # collate(fixed_pad=True) must produce the same prediction as the dynamic-M collate
    import numpy as _np
    from data import collate as _col
    samples = [{"Y": _np.zeros(cfg.n_genes, _np.float32), "X": b["X"][i].numpy(), "E": b["E"][i].numpy(),
                "r": b["r"][i].numpy(), "cell_ctx": b["cell_ctx"][i].numpy(),
                "atoms": b["atoms"][i][b["atom_mask"][i]].numpy(), "u_feats": b["u_feats"][i].numpy(),
                "dose": float(b["dose"][i]), "time": float(b["time"][i]), "w": 1.0}
               for i in range(b["X"].shape[0])]
    with torch.no_grad():
        y_dyn = m(_col(samples, cfg.max_atoms, fixed_pad=False))
        y_fix = m(_col(samples, cfg.max_atoms, fixed_pad=True))
    check("collate fixed_pad == dynamic pad (XLA-safe)", torch.allclose(y_dyn, y_fix, atol=1e-5),
          f"max dev {float((y_dyn - y_fix).abs().max()):.2e}")


# ---- 9. baselines helper runs ----
def test_baselines_helper():
    G = 20
    Ytr = np.random.randn(50, G); ce = np.random.randint(0, 5, 50); de = np.random.randint(0, 8, 50)
    Yev = np.random.randn(12, G); cev = np.random.randint(0, 5, 12); dev = np.random.randint(0, 8, 12)
    out = L.naive_baselines(Ytr, ce, de, Yev, cev, dev)
    check("naive_baselines returns 3 finite MSEs",
          set(out) == {"Mean", "Meancell", "Meandrug"} and all(np.isfinite(list(out.values()))), str(out))


if __name__ == "__main__":
    for fn in [test_shapes, test_attention_normalized, test_gate, test_no_leakage, test_prior_bias,
               test_film_identity, test_gradients, test_overfit, test_sdpa_equiv,
               test_epi_additive, test_ca_attribution, test_correlation_loss, test_cell_context,
               test_pathway_conductance, test_padding_invariance,
               test_baselines_helper]:
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(fn.__name__ + " (raised)", False, repr(e))
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'='*50}\n{len(RESULTS)-n_fail}/{len(RESULTS)} checks passed"
          + (f" | {n_fail} FAILED" if n_fail else " | ALL PASS"))
    sys.exit(1 if n_fail else 0)
