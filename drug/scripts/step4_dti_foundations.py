# -*- coding: utf-8 -*-
"""
step4_dti_foundations.py -- shared keys for the DTI validation reference (ChEMBL + STITCH).
Builds:
  outputs/dti/landmark_genes.tsv   gene_idx, symbol, entrez, ensp   (978 genes, canonical order)
      symbol+entrez from LINCS gene_info (pr_is_lm==1); ensp (9606.ENSPxxxx, STITCH/STRING id space)
      from the network branch STRING v12 protein.info (preferred_name -> string id).
  outputs/dti/drug_inchikeys.tsv   pert_id, inchikey, block1   (InChIKey = cross-DB structure key;
      block1 = 14-char connectivity layer for looser matching). RDKit from canonical_smiles.
Run with .venv-drug python (RDKit). Reproducible, no network.
"""
import os, csv, sys
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = "../.."
GENE_INFO = f"{ROOT}/Data Info/GSE92742_Broad_LINCS_gene_info.txt/GSE92742_Broad_LINCS_gene_info.txt"
LANDMARK_ORDER = f"{ROOT}/Data Info/pathway_landmark_genes.txt"   # canonical 978 gene-axis order
STRING_INFO = f"{ROOT}/network/data/9606.protein.info.v12.0.txt"
DRUGS = "../outputs/drug_list.tsv"
OUT = "../outputs/dti"
os.makedirs(OUT, exist_ok=True)

# --- canonical 978 gene order (symbols) ---
order = [l.strip() for l in open(LANDMARK_ORDER, encoding="utf-8") if l.strip()]
print(f"canonical landmark order: {len(order)} symbols")

# --- symbol -> entrez from LINCS gene_info (landmark rows) ---
sym2entrez = {}
with open(GENE_INFO, encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r.get("pr_is_lm") == "1":
            sym2entrez[r["pr_gene_symbol"]] = r["pr_gene_id"]
print(f"landmark symbols with entrez: {len(sym2entrez)}")

# --- symbol -> ensp (9606.ENSP...) from STRING v12 protein.info ---
sym2ensp = {}
with open(STRING_INFO, encoding="utf-8") as f:
    next(f)  # header '#string_protein_id\tpreferred_name\t...'
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            sym2ensp.setdefault(p[1], p[0])  # preferred_name -> string id (first wins)

# --- write landmark_genes.tsv ---
n_ensp = 0
with open(f"{OUT}/landmark_genes.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["gene_idx", "symbol", "entrez", "ensp"])
    for i, sym in enumerate(order):
        ensp = sym2ensp.get(sym, "")
        if ensp: n_ensp += 1
        w.writerow([i, sym, sym2entrez.get(sym, ""), ensp])
print(f"landmark_genes.tsv: {len(order)} genes | entrez {sum(1 for s in order if s in sym2entrez)} | ensp {n_ensp}")

# --- drug InChIKeys ---
rows = list(csv.DictReader(open(DRUGS, encoding="utf-8"), delimiter="\t"))
ok = fail = 0
with open(f"{OUT}/drug_inchikeys.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["pert_id", "inchikey", "block1"])
    for r in rows:
        m = Chem.MolFromSmiles(r["canonical_smiles"])
        ik = Chem.MolToInchiKey(m) if m is not None else ""
        if ik: ok += 1
        else: fail += 1
        w.writerow([r["pert_id"], ik, ik[:14] if ik else ""])
print(f"drug_inchikeys.tsv: {ok} ok, {fail} fail (of {len(rows)})")
print("wrote", OUT)
