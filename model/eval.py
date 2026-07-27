# -*- coding: utf-8 -*-
"""
DTI recall@k -- the interpretability deliverable (Strategy B). Tests the project's ACTUAL objective:
does the model's atom->gene attribution localize to KNOWN drug target genes (dti_reference.tsv)?

For each drug (keyed by pert_id) with >=1 REPRODUCIBLE signature (mean|Y|>=STRENGTH) and >=1 target
gene_idx in the reference, rank the 978 genes by each attribution signal and measure recall@k against
the drug's reference targets. Three signals (one forward pass, manual-attn path exposes all):
  1. ca_gene_norm  -- L2 of the summed atom->gene cross-attention CONTRIBUTION per gene (primary,
                      faithful "how much the drug's atoms displace gene g").
  3. |Yhat|        -- predicted differential magnitude per gene (primary, end-to-end readout).
  2. atom_attn     -- gene's attention mass on ATOM tokens (excl. pooled global) (exploratory).
Reported vs a random-ranking baseline (expected recall@k = k/978) and as enrichment = recall/random,
on ALL reference edges and on a HIGH-CONFIDENCE subset (stitch>=700 or chembl_direct). Stratified to
reproducible signatures throughout (per the operating rule: never judge on inert perturbations).

Run inside a Kaggle CPU kernel (needs model-src + train bundle + fresh ckpt + dti_reference.tsv).
"""
import os, sys, glob, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
import numpy as np
import torch

from config import ModelConfig
from data import LincsDataset, collate
from model import LincsCrossAttn
from train import make_dataconfig

STRENGTH = 1.0            # reproducible threshold (replicate r ~0.37+)
MAX_SIG_PER_DRUG = 6      # signatures averaged per drug (keeps the manual-attn memory bounded)
KS = [5, 10, 20, 50]
SIGNALS = ["ca_gene_norm", "yhat_abs", "atom_attn"]


def find_dti():
    hits = glob.glob("/kaggle/input/**/dti_reference.tsv", recursive=True)
    if hits:
        return hits[0]
    return os.path.join(make_dataconfig().root, "drug/outputs/dti/dti_reference.tsv")


def recall_at_k(score, targets, ks):
    """score:[978] higher=stronger; targets:set of gene_idx. -> {k: recall}, plus target rank percentiles."""
    order = np.argsort(-score)                       # gene indices best->worst
    rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    T = len(targets)
    out = {k: len(set(order[:k].tolist()) & targets) / T for k in ks}
    pct = [float(rank[g]) / len(order) for g in targets]     # 0=top, 1=bottom
    return out, pct


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = make_dataconfig(); mc = ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).to(device).eval()
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    if not ck:
        raise FileNotFoundError("ckpt.pt not found under /kaggle/input")
    model.load_state_dict(torch.load(ck[0], map_location=device)["model"])
    print("loaded ckpt", ck[0], flush=True)

    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    dindex = json.load(open(R(dc.drug_index_path)))          # pert_id -> drug_row

    # ---- DTI reference, split by evidence quality (target-gene != responding-gene is a known confound,
    # so isolate the biologically meaningful subsets):
    #   all    - every edge (98% noisy STITCH co-occurrence)
    #   chembl - ChEMBL CURATED direct-mechanism edges (the real drug->target genes)
    #   both   - both-source gold core (ChEMBL AND STITCH agree)
    SUBSETS = ["all", "chembl", "both"]
    ref = {sub: defaultdict(set) for sub in SUBSETS}
    with open(find_dti(), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                gi = int(row["gene_idx"])
            except (KeyError, ValueError):
                continue
            if not (0 <= gi < mc.n_genes):
                continue
            pid = row["pert_id"]; ev = (row.get("evidence") or "").lower()
            is_chembl = ("chembl" in ev) or (int(float(row.get("chembl_direct") or 0)) == 1) \
                        or bool((row.get("chembl_actions") or "").strip())
            ref["all"][pid].add(gi)
            if is_chembl:
                ref["chembl"][pid].add(gi)
            if ev.startswith("chembl+stitch"):        # both-source gold (chembl+stitch / chembl+stitch_high)
                ref["both"][pid].add(gi)
    for sub in SUBSETS:
        print(f"DTI '{sub}' drugs: {sum(1 for v in ref[sub].values() if v):,} "
              f"edges: {sum(len(v) for v in ref[sub].values()):,}", flush=True)

    # ---- per-drug reproducible signature rows ----
    drug_sigs = defaultdict(list)
    for j in range(len(ds.y_row)):
        if ds.strength[j] >= STRENGTH:
            drug_sigs[int(ds.drug_row[j])].append(j)

    agg = {(sub, s): {k: [] for k in KS} for sub in SUBSETS for s in SIGNALS}
    pcts = {(sub, s): [] for sub in SUBSETS for s in SIGNALS}
    ndrug = {sub: 0 for sub in SUBSETS}
    n_fwd = 0

    for pid in ref["all"]:
        d = dindex.get(pid)
        if d is None:
            continue
        rows = drug_sigs.get(d, [])
        if not rows:
            continue
        rows = rows[:MAX_SIG_PER_DRUG]
        b = collate([ds[j] for j in rows], mc.max_atoms)
        bd = {k: v.to(device) for k, v in b.items()}
        with torch.no_grad():
            yhat, aux = model(bd, return_attn=True)          # exposes alpha_drug + ca_gene_norm
        sig = {
            "ca_gene_norm": aux["ca_gene_norm"].float().mean(0).cpu().numpy(),
            "yhat_abs": yhat.float().abs().mean(0).cpu().numpy(),
            # alpha_drug [B,H,G,1+M]: drop global token 0, sum over atoms, mean over heads & sigs
            "atom_attn": aux["alpha_drug"][..., 1:].sum(-1).mean(1).mean(0).cpu().numpy(),
        }
        for sub in SUBSETS:
            targets = ref[sub].get(pid, set())
            if not targets:
                continue
            for s in SIGNALS:
                r, p = recall_at_k(sig[s], targets, KS)
                for k in KS:
                    agg[(sub, s)][k].append(r[k])
                pcts[(sub, s)].extend(p)
            ndrug[sub] += 1
        n_fwd += 1
        if n_fwd % 50 == 0:
            print(f"  evaluated {n_fwd} drugs", flush=True)

    # ---- report ----
    def summarize(subset):
        n = ndrug[subset]
        print(f"\n===== {subset.upper()} edges  (drugs evaluated: {n}) =====", flush=True)
        print(f"  random baseline recall@k = k/978: " + "  ".join(f"@{k}={k/mc.n_genes:.3f}" for k in KS))
        res = {}
        for s in SIGNALS:
            line = []
            for k in KS:
                v = float(np.mean(agg[(subset, s)][k])) if agg[(subset, s)][k] else float("nan")
                enr = v / (k / mc.n_genes) if k else float("nan")
                line.append(f"@{k}={v:.3f}(x{enr:.1f})")
            medpct = float(np.median(pcts[(subset, s)])) if pcts[(subset, s)] else float("nan")
            print(f"  {s:14s} recall " + "  ".join(line) + f"   | median target rank pctile={medpct:.3f}",
                  flush=True)
            res[s] = {"recall": {k: (float(np.mean(agg[(subset, s)][k])) if agg[(subset, s)][k] else None)
                                  for k in KS}, "median_rank_pctile": medpct}
        return res

    out = {"n_drugs": ndrug, "strength": STRENGTH, "max_sig_per_drug": MAX_SIG_PER_DRUG,
           "random_recall": {k: k / mc.n_genes for k in KS},
           **{sub: summarize(sub) for sub in SUBSETS}}
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    json.dump(out, open(f"{work}/dti_eval.json", "w"), indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
