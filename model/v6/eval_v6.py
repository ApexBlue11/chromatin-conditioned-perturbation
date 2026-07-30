# -*- coding: utf-8 -*-
"""
v6 evaluation: accuracy on the reproducible stratum + ABLATE-TO-MEAN attribution of every component.
Runs LOCALLY on CPU (torch 2.11 is installed) -- no accelerator time needed. Rationale: ARCHITECTURE.md 7.

PROTOCOL IS MATCHED TO v5, or the comparison is meaningless (HANDOFF 2):
  * the same three splits from one run -- unseen CELL / unseen COMPOUND / unseen BOTH (build_splits, fold 0)
  * REPRODUCIBLE STRATUM ONLY (mean|Y| >= 1). ~75 % of LINCS perturbations are inert; evaluating on all
    signatures once INVERTED the sign of the chromatin effect [6.2].
  * the same metric functions v5 reported (losses.r2_overall / r2_per_gene / pearson_per_row)
  * SPLIT-IDENTITY CHECK: v5 measured these strata at 7296 / 6000 / 2998 signatures. Different counts mean
    different splits and an INVALID comparison -- this script says so loudly instead of quietly printing a
    number that looks comparable.
  v5 reference (pearson_median): unseen cell 0.440 | unseen compound 0.471 | unseen both 0.451

EVERY component is ablated TO ITS MEAN over the same signatures -- never to 0 and never to 1.
Forcing a learned multiplicative component to 1 destroys its learned SCALE, not its information: in v5 that
reported the pathway conductance at +0.103 when the true structural effect was +0.006, a 30x artefact [3.6].
Ablating to the mean removes what the component KNOWS while preserving what it CONTRIBUTES in magnitude,
which is the quantity the claim is about.

Components ablated (each to its mean on the evaluated signatures):
  mean_baseline      X_base            -> per-gene mean            "does baseline expression inform this?"
  mean_chromatin     E                 -> per-(gene,mark) mean     THE surviving v5 mechanism [2.1]
  mean_pathway       PathwayBottleneck output -> its mean          THE v6 novelty -- may well be null
  mean_pathway_gate  chromatin's PATHWAY-level gate -> mean        v6's replacement for v5's per-gene gate
  mean_lineage       cell_ctx          -> mean                     marginal in v5 [5.4]
  mean_drug_global   u_feats (UniMol CLS + descriptors + ECFP4)    a predictive pillar [5.1]
  mean_atoms         valid atom tokens -> mean atom vector         the other pillar (+0.163 [4.3])

`r` (chromatin reliability) is deliberately NOT touched by mean_chromatin. Setting r=0 asserts "this cell
was never assayed", which is a different intervention from "this cell's chromatin looks average"; only the
latter is an information ablation.

Run:  python model/v6/eval_v6.py --ckpt model/results/ckpt_v6_fold0.pt
"""
import os, sys, csv, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch

from config_v6 import V6Config, V6DataConfig
from model_v6 import LincsV6
from data import LincsDataset, build_splits, collate
import losses as L

# v5 reference, same splits/stratum/metrics. `cap` is v5's max_n (train.py: evaluate(max_n=...) subsamples
# with a FRESH default_rng(0)); reproduced here so the headline number is comparable signature-for-signature.
# `n` is what v5 REPORTED = min(stratum, cap), which makes it a split-identity check -- exact when v5 was
# not capped (unseen cell/both), only a lower bound when it was (unseen compound: 6000 of a larger stratum).
V5_REF = {"unseen_cell":     {"pearson": 0.440, "r2": 0.273, "n": 7296, "cap": 8000},
          "unseen_compound": {"pearson": 0.471, "r2": 0.188, "n": 6000, "cap": 6000},
          "unseen_both":     {"pearson": 0.451, "r2": 0.173, "n": 2998, "cap": 6000}}
SPLIT_KEY = {"unseen_cell": "test_coldcell", "unseen_compound": "test_colddrug",
             "unseen_both": "test_coldboth"}
MODES = ["mean_baseline", "mean_chromatin", "mean_pathway", "mean_pathway_gate",
         "mean_lineage", "mean_drug_global", "mean_atoms"]


# --------------------------------------------------------------------------------------------------
def load_pathway_names(dc, M):
    """The 360 (pathway_id, pathway_name) in M's ROW ORDER -- after VERIFYING the mapping.

    v6's entire interpretability claim is that `pathway_activations[:, p]` IS Reactome pathway p, so the
    row order must be CHECKED, not assumed: each row's gene set is re-derived from the member symbols in
    pathway_info.tsv and compared against M[p]. A silent off-by-one here would mislabel every pathway we
    ever report, which is exactly the class of error this project keeps catching late."""
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    with open(R(dc.gene_order_path), encoding="utf-8") as f:
        gi = {g.strip(): i for i, g in enumerate(l for l in f if l.strip())}
    with open(R(dc.pathway_info_path), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != M.shape[0]:
        raise RuntimeError(f"pathway_info.tsv has {len(rows)} rows, M_reactome has {M.shape[0]}")
    bad = [p for p, r in enumerate(rows)
           if set(np.where(M[p] > 0)[0].tolist()) !=
              {gi[g] for g in r["landmark_gene_symbols"].split(",") if g in gi}]
    if bad:
        raise RuntimeError(f"pathway_info.tsv row order does NOT match M_reactome ({len(bad)}/{len(rows)} "
                           f"rows differ) -- the pathway readout would be MISLABELLED. Refusing to report.")
    return [(r["pathway_id"], r["pathway_name"]) for r in rows]


def load_v6(dc, cfg, M, ckpt=None, fold=0, random_init=False):
    """Build v6 and load a checkpoint STRICTLY. A silently partial load (renamed or missing keys) would
    make every component look inert and manufacture a clean null result, so a mismatch is fatal here."""
    model = LincsV6(cfg, M).eval()
    if random_init:
        print("!! random_init: UNTRAINED model, mechanics check only, numbers are meaningless", flush=True)
        return model, "RANDOM_INIT"
    ck = ckpt or os.path.join(os.path.dirname(HERE), "results", f"ckpt_v6_fold{fold}.pt")
    sd = torch.load(ck, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing={list(missing)[:6]} "
                           f"unexpected={list(unexpected)[:6]}")
    print(f"loaded {ck}", flush=True)
    return model, os.path.basename(ck)


class MeanAblate:
    """Collect a module output's MEAN over the evaluated signatures, then substitute that mean.

    Two-pass, because the mean must be taken over the SAME signatures the ablation is scored on:
      pass 1 (`collect`) -- accumulate the running mean over the batch dimension
      pass 2 (`ablate`)  -- return that mean in place of the real output, broadcast over the batch
    `transform`/`inverse` let us average in the space that is actually multiplied in: the pathway gate is
    `1 + tanh(g)`, so we average tanh(g) and hand back atanh(mean), making the multiplier EXACTLY its mean
    rather than the tanh of a mean.
    """

    def __init__(self, module, tuple_index=None, transform=None, inverse=None):
        self.ti, self.tf, self.inv = tuple_index, transform, inverse
        self.mode, self.sum, self.n = "off", None, 0
        self.handle = module.register_forward_hook(self._hook)

    def _pick(self, out):
        return out if self.ti is None else out[self.ti]

    def _hook(self, mod, inp, out):
        t = self._pick(out)
        if t is None:
            return None
        if self.mode == "collect":
            v = self.tf(t) if self.tf else t
            s = v.detach().sum(0)
            self.sum = s if self.sum is None else self.sum + s
            self.n += t.shape[0]
        elif self.mode == "ablate":
            m = self.mean
            m = self.inv(m) if self.inv else m
            new = m.unsqueeze(0).expand(t.shape[0], *([-1] * m.dim())).to(t.dtype)
            if self.ti is None:
                return new
            o = list(out); o[self.ti] = new
            return tuple(o)
        return None

    @property
    def mean(self):
        if self.sum is None:
            raise RuntimeError("MeanAblate: nothing collected -- run the collect pass first")
        return self.sum / max(self.n, 1)


def metrics(yh, yt, phase=None):
    """Exactly the quantities v5 reported, so the two are directly comparable, plus Common-DEGs (M.5)."""
    out = {"r2_overall": L.r2_overall(yh, yt),
           "r2_gene_median": float(L.r2_per_gene(yh, yt).median()),
           "pearson_median": float(L.pearson_per_row(yh, yt).median()),
           "mse": float(((yh - yt) ** 2).mean()), "n": int(len(yt))}
    for k in (50, 100):
        out[f"common_degs@{k}"] = float(L.common_degs(yh, yt, k).median())
        out[f"common_degs@{k}_chance"] = k / yt.shape[1]
    if phase is not None:
        for p in np.unique(phase):                     # residual batch effects: we do NOT model plate
            m = torch.from_numpy(phase == p)
            if int(m.sum()) > 50:
                out[f"r2_phase_{p}"] = L.r2_overall(yh[m], yt[m])
    return out


# Strength bins for the all-strata report (M.2). Reporting ONLY mean|Y| >= 1 makes the headline
# unauditable: the stratum is defined by the ground truth we then score against, which Ahlmann-Eltze
# (Nature Methods 2025) call out as inapplicable in real use. We keep the stratified metric -- LINCS is
# ~75 % inert and dilution once inverted a real effect's sign [6.2] -- but we now show every bin, which is
# what the in-the-wild benchmark recommends. Equal n per bin so precision is comparable across bins.
STRATA = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]


def substitute(b, mode, mu):
    """Replace one component's INPUT with its mean over the evaluated signatures (in place)."""
    if mode == "mean_baseline":
        b["X"] = mu["X"].expand_as(b["X"])
    elif mode == "mean_chromatin":
        b["E"] = mu["E"].unsqueeze(0).expand_as(b["E"])          # r untouched -- see module docstring
    elif mode == "mean_lineage":
        b["cell_ctx"] = mu["cell_ctx"].expand_as(b["cell_ctx"])
    elif mode == "mean_drug_global":
        b["u_feats"] = mu["u_feats"].expand_as(b["u_feats"])
    elif mode == "mean_atoms":
        # every VALID atom becomes the mean atom vector; atom_mask is unchanged, so molecule size is kept
        # and the ablation is of atom CHEMISTRY only
        b["atoms"] = mu["atom"].view(1, 1, -1).expand_as(b["atoms"])
    return b


# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_split(model, ds, idx, cfg, batch, hooks, collect_interp):
    """One pass over `idx`. Returns (yhat, ytrue, accumulators). Accumulators are only filled when
    collect_interp is True (the full-model pass), which is also when the hooks collect their means."""
    yh, yt = [], []
    acc = {"X": None, "E": None, "cell_ctx": None, "u_feats": None, "atom": None,
           "atom_n": 0, "n": 0, "pa_abs": None, "epi_abs": 0.0}
    for s in range(0, len(idx), batch):
        b = collate([ds[i] for i in idx[s:s + batch]], cfg.max_atoms)
        if collect_interp:
            for k in ["X", "E", "cell_ctx", "u_feats"]:
                v = b[k].sum(0)
                acc[k] = v if acc[k] is None else acc[k] + v
            m = b["atom_mask"]
            acc["atom"] = (b["atoms"] * m.unsqueeze(-1)).sum((0, 1)) if acc["atom"] is None else \
                          acc["atom"] + (b["atoms"] * m.unsqueeze(-1)).sum((0, 1))
            acc["atom_n"] += int(m.sum())
            acc["n"] += b["X"].shape[0]
            out, aux = model(b, return_interp=True)
            pa = aux["pathway_activations"].abs().mean(-1).sum(0)        # [P]
            acc["pa_abs"] = pa if acc["pa_abs"] is None else acc["pa_abs"] + pa
            acc["epi_abs"] += float(aux["epi_contrib"].abs().sum())
        else:
            out = model(b)
        yh.append(out.float()); yt.append(b["Y"])
    return torch.cat(yh), torch.cat(yt), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="default: model/results/ckpt_v6_fold0.pt")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=2000,
                    help="signatures per split for the ABLATION study (identical across all modes). "
                         "0 = whole reproducible stratum (slow on CPU)")
    ap.add_argument("--headline_n", type=int, default=0,
                    help="signatures for the protocol-matched ACCURACY number. 0 = whole reproducible "
                         "stratum, which is what v5 reported (7296/6000/2998)")
    ap.add_argument("--strata_n", type=int, default=1500,
                    help="signatures per mean|Y| bin for the all-strata report (M.2). 0 disables it")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--random_init", action="store_true",
                    help="mechanics check with an UNTRAINED model. Numbers are meaningless by "
                         "construction and are labelled as such in the output.")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    dc, cfg = V6DataConfig(), V6Config()
    dc.cell_fold = a.fold
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_reactome_path))
    pathways = load_pathway_names(dc, M)
    print(f"pathway_info.tsv <-> M_reactome row order VERIFIED ({len(pathways)} named nodes)", flush=True)

    model, ckpt_name = load_v6(dc, cfg, M, a.ckpt, a.fold, a.random_init)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par/1e6:.2f}M", flush=True)

    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    sp = build_splits(ds, dc)

    hooks = {
        "mean_pathway": MeanAblate(model.pathway, tuple_index=0),
        "mean_pathway_gate": (MeanAblate(model.pathway.epi_gate, transform=torch.tanh,
                                         inverse=lambda m: torch.atanh(m.clamp(-0.999, 0.999)))
                              if model.pathway.epi_gate is not None else None),
    }
    modes = [m for m in a.modes.split(",") if m and hooks.get(m, "ok") is not None]
    if "mean_pathway_gate" not in modes and hooks["mean_pathway_gate"] is None:
        print("note: pathway_epi_gate is disabled in this config -> mean_pathway_gate skipped", flush=True)

    out = {"ckpt": ckpt_name, "fold": a.fold, "params": n_par, "eval_min_strength": dc.eval_min_strength,
           "random_init": bool(a.random_init), "v5_reference": V5_REF,
           "fusion_epi_gate": float(torch.sigmoid(model.fuse.epi_gate)), "splits": {}}
    print(f"late-fusion chromatin gate sigmoid(epi_gate) = {out['fusion_epi_gate']:.4f}  "
          f"(0.5 = untouched from init; chromatin must EARN its way in)", flush=True)

    for name, key in SPLIT_KEY.items():
        idx = sp[key]
        rep = idx[ds.strength[idx] >= dc.eval_min_strength]        # THE reproducible stratum
        ref, cap = V5_REF[name], V5_REF[name]["cap"]
        # v5 reported min(stratum, cap). When it was NOT capped, that pins our stratum size exactly.
        exact = ref["n"] < cap
        ok = (min(len(rep), cap) == ref["n"]) and (len(rep) == ref["n"] if exact else len(rep) >= ref["n"])
        verdict = ("MATCH -> comparable" if ok and exact else
                   "consistent (v5 was capped at this n, so this is a lower bound)" if ok else
                   "MISMATCH -> the splits differ and the v5 COMPARISON IS INVALID")
        print(f"\n{'='*78}\n=== {name}   reproducible stratum n={len(rep)}   (v5 reported n={ref['n']}, "
              f"cap={cap}: {verdict})\n{'='*78}", flush=True)
        if len(rep) < 500:
            print("  too few signatures -- skipped", flush=True); continue

        rec = {"n_reproducible": int(len(rep)), "split_matches_v5": bool(ok),
               "split_check_is_exact": bool(exact), "ablation": {}}

        # ---------- protocol-matched accuracy: v5's own cap + fresh rng(0), signature-for-signature ----------
        hn = min(len(rep), cap if a.headline_n in (0, None) else a.headline_n)
        h_idx = np.random.default_rng(0).choice(rep, hn, replace=False) if hn < len(rep) else rep
        t0 = time.time()
        yh, yt, _ = run_split(model, ds, h_idx, cfg, a.batch, hooks, collect_interp=False)
        rec["headline"] = metrics(yh, yt, ds.phase[h_idx])
        print(f"  ACCURACY (n={hn}, {time.time()-t0:.0f}s)  pearson={rec['headline']['pearson_median']:.4f}"
              f"  R2={rec['headline']['r2_overall']:+.4f}"
              f"   |  v5: {ref['pearson']:.3f} / {ref['r2']:+.3f}"
              f"   |  delta_pearson={rec['headline']['pearson_median']-ref['pearson']:+.4f}", flush=True)

        # ---------- all-strata report (M.2): the >=1 number is only auditable next to the others -------
        rec["strata"] = {}
        if a.strata_n:
            print(f"  ALL STRATA (equal n per bin; the reproducible stratum is the last two)", flush=True)
        for lo, hi in (STRATA if a.strata_n else []):
            band = idx[(ds.strength[idx] >= lo) & (ds.strength[idx] < hi)]
            if len(band) < 200:
                print(f"    mean|Y| {lo}-{hi}: n={len(band)} too few, skipped", flush=True); continue
            sn = min(a.strata_n, len(band))
            s_idx = np.sort(np.random.default_rng(11).choice(band, sn, replace=False)) if sn < len(band) else band
            yhs, yts, _ = run_split(model, ds, s_idx, cfg, a.batch, hooks, collect_interp=False)
            m = metrics(yhs, yts)
            rec["strata"][f"{lo}-{hi}"] = {**m, "n_available": int(len(band))}
            print(f"    mean|Y| {lo:>4}-{str(hi):<4} n={sn:<5} (of {len(band):>6})  "
                  f"pearson={m['pearson_median']:.4f}  R2={m['r2_overall']:+.4f}  "
                  f"cDEG@100={m['common_degs@100']:.3f} (chance {m['common_degs@100_chance']:.3f})", flush=True)

        # ---------- ablation study: identical signatures for full and every mode ----------
        an = len(rep) if a.n in (0, None) else min(a.n, len(rep))
        # own rng: the requirement is that FULL and every ablation mode see IDENTICAL signatures, not that
        # they match the headline set. Sorting keeps phase/plate order representative rather than row-ordered.
        ab_idx = np.sort(np.random.default_rng(7).choice(rep, an, replace=False)) if an < len(rep) else rep
        for h in hooks.values():
            if h: h.mode, h.sum, h.n = "collect", None, 0
        yh, yt, acc = run_split(model, ds, ab_idx, cfg, a.batch, hooks, collect_interp=True)
        for h in hooks.values():
            if h: h.mode = "off"
        base = metrics(yh, yt)
        rec["ablation"]["full"] = base
        mu = {k: acc[k] / max(acc["n"], 1) for k in ["X", "E", "cell_ctx", "u_feats"]}
        mu["atom"] = acc["atom"] / max(acc["atom_n"], 1)
        pa_abs = (acc["pa_abs"] / max(acc["n"], 1)).numpy()
        rec["pathway_readout"] = {
            "mean_abs_activation_top20": [
                {"rank": i + 1, "pathway_id": pathways[p][0], "pathway_name": pathways[p][1],
                 "mean_abs_activation": float(pa_abs[p]), "n_genes": int(M[p].sum())}
                for i, p in enumerate(np.argsort(-pa_abs)[:20])],
            "n_pathways_below_1pct_of_max": int((pa_abs < 0.01 * pa_abs.max()).sum()),
            "activation_gini_like_top10_share": float(np.sort(pa_abs)[-10:].sum() / (pa_abs.sum() + 1e-9)),
        }
        rec["epi_contrib_mean_abs"] = acc["epi_abs"] / max(acc["n"] * cfg.n_genes, 1)
        print(f"\n  ABLATE TO MEAN (n={an}, identical signatures; positive delta => CONTRIBUTES)")
        print(f"    {'full':18s} pearson={base['pearson_median']:.4f}  R2={base['r2_overall']:+.4f}", flush=True)
        for mode in modes:
            if mode in hooks:
                hooks[mode].mode = "ablate"
                yh2, yt2, _ = run_split(model, ds, ab_idx, cfg, a.batch, hooks, collect_interp=False)
                hooks[mode].mode = "off"
            else:
                yh2, yt2 = [], []
                for s in range(0, len(ab_idx), a.batch):
                    b = collate([ds[i] for i in ab_idx[s:s + a.batch]], cfg.max_atoms)
                    yh2.append(model(substitute(b, mode, mu)).float()); yt2.append(b["Y"])
                yh2, yt2 = torch.cat(yh2), torch.cat(yt2)
            m2 = metrics(yh2, yt2)
            m2["delta_pearson"] = base["pearson_median"] - m2["pearson_median"]
            m2["delta_r2"] = base["r2_overall"] - m2["r2_overall"]
            # Does the ablation actually MOVE the prediction? Without this, "delta = 0.000" is ambiguous
            # between "the component carries no information" and "the substitution never took effect" --
            # and this project's recurring failure mode is a valid computation of the wrong quantity.
            # test_v6.py holds the positive control that each mode CAN move a live component.
            m2["max_abs_delta_yhat"] = float((yh2 - yh).abs().max())
            rec["ablation"][mode] = m2
            flag = "   <- prediction UNCHANGED: component is inert in this checkpoint" \
                   if m2["max_abs_delta_yhat"] == 0.0 else ""
            print(f"    {mode:18s} pearson={m2['pearson_median']:.4f} (d={m2['delta_pearson']:+.4f})  "
                  f"R2={m2['r2_overall']:+.4f} (d={m2['delta_r2']:+.4f})  "
                  f"|dY|max={m2['max_abs_delta_yhat']:.3f}{flag}", flush=True)

        print(f"\n    signed chromatin head |contribution| = {rec['epi_contrib_mean_abs']:.4f} per gene")
        print(f"    pathway nodes effectively unused (<1 % of max activation): "
              f"{rec['pathway_readout']['n_pathways_below_1pct_of_max']}/{len(pathways)}"
              f"   top-10 share of total activation: "
              f"{rec['pathway_readout']['activation_gini_like_top10_share']:.3f}", flush=True)
        out["splits"][name] = rec

    dest = os.path.join(os.path.dirname(HERE), "results", f"eval_v6_fold{a.fold}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}", flush=True)
    print("Report the negatives with the same prominence as the positives -- a null pathway bottleneck is "
          "a result, not a failure to hide.", flush=True)


if __name__ == "__main__":
    main()
