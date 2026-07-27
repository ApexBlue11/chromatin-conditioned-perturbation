# -*- coding: utf-8 -*-
"""
step7_dti_merge.py -- unify ChEMBL (curated, action-typed) + STITCH (confidence-scored) into ONE
drug->gene DTI reference over the 978 landmark genes, with provenance. This is the ground-truth for
the learn-then-validate interpretability plan: for a drug, which landmark genes are known targets,
by which evidence, and how strong.

One row per (pert_id, gene) that has ANY evidence:
  pert_id, pert_iname, gene_symbol, entrez, gene_idx,
  chembl (0/1), chembl_actions, chembl_direct (0/1), stitch_score (max, ''=none), evidence
Output: ../outputs/dti/dti_reference.tsv + dti_reference_summary.json
"""
import csv, json, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
OUT = "../outputs/dti"

lm = {r["symbol"]: (r["entrez"], r["gene_idx"]) for r in
      csv.DictReader(open(f"{OUT}/landmark_genes.tsv", encoding="utf-8"), delimiter="\t")}
iname = {r["pert_id"]: r["pert_iname"] for r in
         csv.DictReader(open("../outputs/drug_list.tsv", encoding="utf-8"), delimiter="\t")}

# --- ChEMBL landmark edges ---
chembl = defaultdict(lambda: {"actions": set(), "direct": 0})
cpath = f"{OUT}/chembl_dti_edges.tsv"
if os.path.exists(cpath):
    for r in csv.DictReader(open(cpath, encoding="utf-8"), delimiter="\t"):
        if r["is_landmark"] != "1" or not r["gene_symbol"]:
            continue
        k = (r["pert_id"], r["gene_symbol"])
        if r["action_type"]:
            chembl[k]["actions"].add(r["action_type"])
        if str(r["direct_interaction"]).lower() in ("true", "1"):
            chembl[k]["direct"] = 1
else:
    print("WARN: chembl edges not found -- run step5 first (STITCH-only merge)")

# --- STITCH edges (max score per drug-gene) ---
stitch = {}
for r in csv.DictReader(open(f"{OUT}/stitch_dti_edges.tsv", encoding="utf-8"), delimiter="\t"):
    k = (r["pert_id"], r["gene_symbol"])
    s = int(r["combined_score"])
    if s > stitch.get(k, -1):
        stitch[k] = s

# --- union ---
keys = set(chembl) | set(stitch)
rows = []
for (pid, sym) in keys:
    entrez, gi = lm.get(sym, ("", ""))
    c = chembl.get((pid, sym))
    s = stitch.get((pid, sym), "")
    ev = []
    if c: ev.append("chembl")
    if s != "" and s >= 700: ev.append("stitch_high")
    elif s != "": ev.append("stitch")
    rows.append([pid, iname.get(pid, ""), sym, entrez, gi,
                 1 if c else 0,
                 "|".join(sorted(c["actions"])) if c else "",
                 c["direct"] if c else 0,
                 s, "+".join(ev)])

rows.sort(key=lambda r: (r[0], -(r[8] if isinstance(r[8], int) else 0)))
hdr = ["pert_id", "pert_iname", "gene_symbol", "entrez", "gene_idx",
       "chembl", "chembl_actions", "chembl_direct", "stitch_score", "evidence"]
with open(f"{OUT}/dti_reference.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(hdr); w.writerows(rows)

both = sum(1 for r in rows if r[5] == 1 and isinstance(r[8], int))
both_high = sum(1 for r in rows if r[5] == 1 and isinstance(r[8], int) and r[8] >= 700)
summary = {
    "pairs_total": len(rows),
    "pairs_chembl": len(chembl), "pairs_stitch": len(stitch),
    "pairs_both_sources": both, "pairs_both_chembl_and_stitch_high": both_high,
    "drugs_covered": len({r[0] for r in rows}),
    "genes_covered": len({r[2] for r in rows}),
    "drugs_chembl": len({k[0] for k in chembl}),
    "drugs_stitch": len({k[0] for k in stitch}),
}
json.dump(summary, open(f"{OUT}/dti_reference_summary.json", "w"), indent=2)
print("SUMMARY:", json.dumps(summary, indent=2))
print("wrote dti_reference.tsv")
