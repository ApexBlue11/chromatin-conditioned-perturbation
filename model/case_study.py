# -*- coding: utf-8 -*-
"""
Mechanistic case studies: for named drugs with textbook targets, where does the model's atom->gene
attribution place that target among the 978 landmark genes?

Aggregate enrichment (2.6x on gold edges) is abstract; "panobinostat's attribution puts HDAC2 in the top
N" is something a reader can check. This is the figure that makes the interpretability claim concrete.

For each drug we report the target's RANK PERCENTILE (0 = top of 978, 0.5 = chance) under two signals:
  ca_gene_norm  -- L2 of the summed atom->gene cross-attention contribution  (structure -> gene)
  |Yhat|        -- predicted differential magnitude                          (end-to-end readout)
averaged over that drug's reproducible signatures.

Run: python model/case_study.py
"""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
import numpy as np
import torch

from config import ModelConfig, DataConfig
from data import LincsDataset, collate
from model import LincsCrossAttn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# textbook mechanisms drawn from the both-source GOLD tier of dti_reference.tsv
CASES = [
    ("BRD-K02130563", "panobinostat", "HDAC2",  "HDAC inhibitor"),
    ("BRD-K09416995", "lovastatin",   "HMGCR",  "HMG-CoA reductase inhibitor"),
    ("BRD-K11433652", "aspirin",      "PTGS2",  "COX inhibitor"),
    ("BRD-A88254928", "salbutamol",   "ADRB2",  "beta-2 agonist"),
    ("BRD-A10070317", "propranolol",  "ADRB2",  "beta blocker"),
    ("BRD-A13084692", "troglitazone", "PPARG",  "PPAR-gamma agonist"),
    ("BRD-K10799896", "clobetasol",   "NR3C1",  "glucocorticoid agonist"),
    ("BRD-K00259736", "colchicine",   "TUBB6",  "tubulin inhibitor"),
    ("BRD-K08799216", "pelitinib",    "EGFR",   "EGFR inhibitor"),
    ("BRD-K07881437", "danusertib",   "AURKA",  "Aurora kinase inhibitor"),
    ("BRD-K12184916", "NVP-BEZ235",   "PIK3CA", "PI3K inhibitor"),
    ("BRD-K07265709", "dexrazoxane",  "TOP2A",  "topoisomerase II inhibitor"),
]
MAX_SIG = 8


def main():
    dc, mc = DataConfig(), ModelConfig()
    R = lambda p: os.path.join(dc.root, p)
    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).eval()
    model.load_state_dict(torch.load(os.path.join(RES, "v5_ckpt.pt"), map_location="cpu")["model"])
    shared = LincsDataset.load_shared(dc); ds = LincsDataset(dc, _shared=shared)
    dindex = json.load(open(R(dc.drug_index_path)))

    # gene_idx -> symbol, from the DTI reference
    sym = {}
    for r in csv.DictReader(open(R("drug/outputs/dti/dti_reference.tsv"), encoding="utf-8"), delimiter="\t"):
        try:
            sym[int(r["gene_idx"])] = r["gene_symbol"]
        except (KeyError, ValueError):
            pass
    sym2idx = {v: k for k, v in sym.items()}

    drug_sigs = defaultdict(list)
    for j in range(len(ds.y_row)):
        if ds.strength[j] >= 1.0:
            drug_sigs[int(ds.drug_row[j])].append(j)

    out = []
    print(f"{'drug':14s} {'target':7s} {'mechanism':30s} {'atom>gene':>11s} {'|Yhat|':>9s}  {'n':>3s}")
    print("-" * 84)
    for pid, name, target, mech in CASES:
        d = dindex.get(pid); gi = sym2idx.get(target)
        rows = drug_sigs.get(d, []) if d is not None else []
        if d is None or gi is None or not rows:
            print(f"{name:14s} {target:7s} {mech:30s} {'--- no reproducible signatures ---':>21s}")
            continue
        rows = rows[:MAX_SIG]
        b = collate([ds[i] for i in rows], mc.max_atoms)
        with torch.inference_mode():
            yhat, aux = model(b, return_attr=True)
        ca = aux["ca_gene_norm"].float().mean(0).numpy()
        ya = yhat.float().abs().mean(0).numpy()

        def pctile(score):
            order = np.argsort(-score)
            rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
            return float(rank[gi]) / len(order)

        p_ca, p_y = pctile(ca), pctile(ya)
        top_ca = [sym.get(int(i), f"g{int(i)}") for i in np.argsort(-ca)[:5]]
        out.append({"drug": name, "pert_id": pid, "target": target, "mechanism": mech,
                    "n_signatures": len(rows), "pctile_atom_gene": p_ca, "pctile_yhat": p_y,
                    "rank_atom_gene": int(p_ca * 978), "rank_yhat": int(p_y * 978),
                    "top5_atom_gene": top_ca})
        print(f"{name:14s} {target:7s} {mech:30s} {p_ca:>10.3f} {p_y:>9.3f}  {len(rows):>3d}")

    if out:
        ca_m = float(np.median([o["pctile_atom_gene"] for o in out]))
        y_m = float(np.median([o["pctile_yhat"] for o in out]))
        print("-" * 84)
        print(f"{'MEDIAN over cases':14s} {'':7s} {'(0.5 = chance, lower = better)':30s} {ca_m:>10.3f} {y_m:>9.3f}")
        json.dump({"cases": out, "median_pctile_atom_gene": ca_m, "median_pctile_yhat": y_m},
                  open(os.path.join(RES, "case_study.json"), "w"), indent=2)
        print(f"\nwrote {os.path.join(RES, 'case_study.json')}")


if __name__ == "__main__":
    main()
