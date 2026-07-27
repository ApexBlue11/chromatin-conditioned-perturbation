# -*- coding: utf-8 -*-
"""
Consolidated + optimized re-run of the two analyses whose results were never persisted as artifacts:
  (A) CORRECTED pathway ablation  c->mean  (separates STRUCTURE from learned SCALE)  [the retraction evidence]
  (B) Metric-convention anchor sweep       (PCC vs basal:delta variance ratio)

OPTIMIZATIONS vs the original one-off scripts:
  1. ONE dataset load (`load_shared` parses a 300k-row TSV + big npys) instead of three.
  2. ONE forward pass per (split, mode) reused across BOTH analyses — the sweep needs only the `full`
     pass, so it costs ZERO extra forwards (previously a separate script + separate load).
  3. Predictions CACHED to results/preds_cache.npz -> any future metric is instant, no re-forward.
  4. torch.inference_mode() + all 12 cores (torch defaults to 6) + float32 matmul fast path.
  5. Resumable: if the cache exists, forwards are skipped entirely.
Net: ~6 forward passes total instead of ~10 plus 3 dataset loads.

Run: python model/finalize_analyses.py [--n 900]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def pcc_med(a, b):
    ac = a - a.mean(1, keepdims=True); bc = b - b.mean(1, keepdims=True)
    return float(np.median((ac * bc).sum(1) / np.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1) + 1e-8)))


def r2(pred, true):
    return float(1 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--force", action="store_true", help="ignore the prediction cache")
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)          # default was 6 of 12
    torch.set_float32_matmul_precision("high")
    cache = os.path.join(RES, "preds_cache.npz")
    t0 = time.time()

    if os.path.exists(cache) and not a.force:
        z = np.load(cache)
        print(f"loaded cached predictions from {cache} (skipping forwards)", flush=True)
        P = {k: z[k] for k in z.files}
    else:
        from config import ModelConfig, DataConfig
        from data import LincsDataset, collate, build_splits
        from model import LincsCrossAttn
        dc, mc = DataConfig(), ModelConfig()
        R = lambda p: os.path.join(dc.root, p)
        model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).eval()
        model.load_state_dict(torch.load(os.path.join(RES, "v5_ckpt.pt"), map_location="cpu")["model"])
        shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
        sp = build_splits(ds, dc)
        print(f"data loaded ({time.time()-t0:.0f}s)", flush=True)

        C = np.load(os.path.join(RES, "pathway_conductance.npy"))
        CMEAN = float(C.mean())
        print(f"learned conductance mean = {CMEAN:.4f}  (c->1 would rescale by {1/CMEAN:.2f}x)", flush=True)

        class ConstCond(torch.nn.Module):
            def __init__(self, v): super().__init__(); self.v = v
            def forward(self, E, x, r): return torch.full(x.shape, self.v, dtype=x.dtype)

        rng = np.random.default_rng(0); P = {"cmean": np.array([CMEAN])}
        for split, tag in [("test_coldcell", "cell"), ("test_colddrug", "drug")]:
            idx = sp[split]; idx = idx[ds.strength[idx] >= 1.0]
            idx = rng.choice(idx, min(a.n, len(idx)), replace=False)
            for mode in ["full", "c_to_1", "c_to_mean"]:
                keep = model.pathway_cond
                if mode == "c_to_1":    model.pathway_cond = None
                elif mode == "c_to_mean": model.pathway_cond = ConstCond(CMEAN)
                yh, yt, xb = [], [], []
                with torch.inference_mode():
                    for s in range(0, len(idx), a.batch):
                        b = collate([ds[i] for i in idx[s:s + a.batch]], mc.max_atoms)
                        yh.append(model(b).float().numpy())
                        if mode == "full":
                            yt.append(b["Y"].numpy()); xb.append(b["X"].numpy())
                model.pathway_cond = keep
                P[f"{tag}_{mode}_yh"] = np.concatenate(yh)
                if mode == "full":
                    P[f"{tag}_yt"] = np.concatenate(yt); P[f"{tag}_xb"] = np.concatenate(xb)
                print(f"  {tag}/{mode} done ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(cache, **P)
        print(f"cached predictions -> {cache}", flush=True)

    CMEAN = float(P["cmean"][0])
    out = {"conductance_mean": CMEAN, "rescale_if_ablated_to_1": 1 / CMEAN}

    # ---------- (A) corrected pathway ablation ----------
    print("\n=== (A) PATHWAY ABLATION: structure vs scale ===", flush=True)
    for tag, label in [("cell", "unseen CELL"), ("drug", "unseen COMPOUND")]:
        yt = P[f"{tag}_yt"]
        f_r2, f_p = r2(P[f"{tag}_full_yh"], yt), pcc_med(P[f"{tag}_full_yh"], yt)
        o_r2, o_p = r2(P[f"{tag}_c_to_1_yh"], yt), pcc_med(P[f"{tag}_c_to_1_yh"], yt)
        m_r2, m_p = r2(P[f"{tag}_c_to_mean_yh"], yt), pcc_med(P[f"{tag}_c_to_mean_yh"], yt)
        print(f"{label} (n={len(yt)}):")
        print(f"  full                      R2={f_r2:+.4f}  p={f_p:.4f}")
        print(f"  c->1   (naive ablation)   R2={o_r2:+.4f}  p={o_p:.4f}   dR2={f_r2-o_r2:+.4f}  <- CONFOUNDED (breaks scale)")
        print(f"  c->mean (scale preserved) R2={m_r2:+.4f}  p={m_p:.4f}   dR2={f_r2-m_r2:+.4f}  <- TRUE structural effect")
        out[label] = {"full": {"r2": f_r2, "pcc": f_p},
                      "c_to_1": {"r2": o_r2, "pcc": o_p, "delta_r2": f_r2 - o_r2},
                      "c_to_mean": {"r2": m_r2, "pcc": m_p, "delta_r2": f_r2 - m_r2}}

    # ---------- (B) metric-convention sweep (reuses the `full` pass: zero extra forwards) ----------
    print("\n=== (B) METRIC CONVENTION: PCC vs basal:delta variance ratio ===", flush=True)
    sweep = {}
    for tag, label in [("cell", "unseen CELL"), ("drug", "unseen COMPOUND")]:
        yh, yt, xb = P[f"{tag}_full_yh"], P[f"{tag}_yt"], P[f"{tag}_xb"]
        rows = [("0 (differential = OUR metric)", 0.0, pcc_med(yh, yt))]
        for al in [0.5, 1, 2, 3, 5, 10, 20, 50]:
            ratio = float((al * xb).var() / yt.var())
            rows.append((f"alpha={al}", ratio, pcc_med(al * xb + yh, al * xb + yt)))
        sweep[label] = [{"anchor": r[0], "basal_delta_var_ratio": r[1], "pcc": r[2]} for r in rows]
        print(f"{label}:  " + "  ".join(f"[ratio {r[1]:.1f} -> PCC {r[2]:.3f}]" for r in rows))
    out["metric_sweep"] = sweep
    out["caveat"] = ("Anchor is the CCLE baseline (no L1000 controls in our data). The sweep shows PCC under "
                     "the absolute convention is a function of the basal:delta variance ratio; it does NOT "
                     "let us infer another paper's ratio or claim parity with it.")

    dest = os.path.join(RES, "finalized_analyses.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}   (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
