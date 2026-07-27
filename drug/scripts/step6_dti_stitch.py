# -*- coding: utf-8 -*-
"""
step6_dti_stitch.py -- broader drug->protein associations (DTI) from STITCH v5.0 (human, 9606).
Complements ChEMBL's curated direct targets with STITCH's confidence-scored links (which include
indirect/downstream associations -- a good match for LINCS being downstream transcriptional effects).

Fully local: LINCS gives each drug a PubChem CID (pert_info), STITCH keys chemicals by zero-padded CID
(CIDm/CIDs), and STITCH proteins are 9606.ENSP ids we already have for the landmark genes (step4, via
STRING). Streams the 74 MB links file; keeps rows where protein in the 978 landmark set AND chemical
maps to one of our drugs; records the STITCH combined_score (0-1000).

Input : ../data/9606.protein_chemical.links.v5.0.tsv.gz  (downloaded)
Output: ../outputs/dti/stitch_dti_edges.tsv + stitch_dti_summary.json
"""
import gzip, csv, glob, json, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

ROOT = "../.."
OUT = "../outputs/dti"
LINKS = "../data/9606.protein_chemical.links.v5.0.tsv.gz"

def di(pat):
    return glob.glob(f"{ROOT}/Data Info/{pat}/*.txt")[0]

# --- pert_id -> pubchem_cid (both phases; GSE92742 carries pubchem_cid) ---
pid2cid = {}
for p in (di("GSE92742_Broad_LINCS_pert_info.txt"), di("GSE70138_Broad_LINCS_pert_info_2017-03-06.txt")):
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            c = r.get("pubchem_cid", "")
            if c and c not in ("-666", "", "NA") and r["pert_id"] not in pid2cid:
                pid2cid[r["pert_id"]] = c

# restrict to our usable drugs + capture inames
ours = list(csv.DictReader(open("../outputs/drug_list.tsv", encoding="utf-8"), delimiter="\t"))
pid2iname = {r["pert_id"]: r["pert_iname"] for r in ours}
our_pids = set(pid2iname)

# STITCH chemical id (CIDm/CIDs + 8-digit) -> [pert_ids]
chem2pids = defaultdict(list)
n_cid = 0
for pid in our_pids:
    c = pid2cid.get(pid)
    if not c:
        continue
    try:
        n = int(c)
    except ValueError:
        continue
    n_cid += 1
    for pre in ("CIDm", "CIDs"):
        chem2pids[f"{pre}{n:08d}"].append(pid)
print(f"drugs with usable CID: {n_cid:,}/{len(our_pids):,} | STITCH chem keys: {len(chem2pids):,}")

# landmark ensp (9606.ENSP...) -> (gene_idx, symbol, entrez)
ensp2gene = {}
for r in csv.DictReader(open(f"{OUT}/landmark_genes.tsv", encoding="utf-8"), delimiter="\t"):
    if r["ensp"]:
        ensp2gene[r["ensp"]] = (r["gene_idx"], r["symbol"], r["entrez"])
print(f"landmark proteins (ensp): {len(ensp2gene):,}")

# --- stream the links file ---
if not os.path.exists(LINKS):
    sys.exit(f"missing {LINKS} -- download it first")
edges = []
seen_prot = set()
n_rows = 0
with gzip.open(LINKS, "rt", encoding="utf-8") as f:
    header = f.readline()  # 'chemical\tprotein\tcombined_score'
    for line in f:
        n_rows += 1
        chem, prot, score = line.rstrip("\n").split("\t")
        g = ensp2gene.get(prot)
        if g is None:
            continue
        pids = chem2pids.get(chem)
        if not pids:
            continue
        seen_prot.add(prot)
        gi, sym, entrez = g
        for pid in pids:
            edges.append([pid, pid2iname.get(pid, ""), pid2cid.get(pid, ""), chem, prot,
                          sym, entrez, gi, int(score)])
print(f"streamed {n_rows:,} link rows; matched edges: {len(edges):,}")

edges.sort(key=lambda e: (-e[8], e[0]))
hdr = ["pert_id", "pert_iname", "pubchem_cid", "stitch_chemical", "ensp",
       "gene_symbol", "entrez", "gene_idx", "combined_score"]
with open(f"{OUT}/stitch_dti_edges.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(hdr); w.writerows(edges)

def at(th):
    e = [x for x in edges if x[8] >= th]
    return {"edges": len(e), "drugs": len({x[0] for x in e}), "genes": len({x[5] for x in e})}
summary = {
    "drugs_total": len(our_pids), "drugs_with_cid": n_cid,
    "landmark_proteins_mapped": len(ensp2gene), "landmark_proteins_hit": len(seen_prot),
    "edges_total": len(edges),
    "drugs_with_any_edge": len({e[0] for e in edges}),
    "genes_hit": len({e[5] for e in edges}),
    "at_score_ge_400": at(400), "at_score_ge_700": at(700),
}
json.dump(summary, open(f"{OUT}/stitch_dti_summary.json", "w"), indent=2)
print("SUMMARY:", json.dumps(summary, indent=2))
print("wrote stitch_dti_edges.tsv")
