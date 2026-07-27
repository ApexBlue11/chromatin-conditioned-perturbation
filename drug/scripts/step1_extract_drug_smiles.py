# -*- coding: utf-8 -*-
"""
step1_extract_drug_smiles.py -- drug branch foundation. From the usable signatures (post no-SMILES
exclusion), gather the unique drugs (keyed by pert_id) + their canonical SMILES + iname + #signatures.
This is the per-pert_id lookup the model joins by at train time. Outputs drug/outputs/drug_list.tsv.
"""
import sys, csv, glob
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
ROOT="../.."
SIGS=f"{ROOT}/phase2_assembly/outputs/signatures_usable.tsv"
def di(pat): return glob.glob(f"{ROOT}/Data Info/{pat}/*.txt")[0]
PERT=[di("GSE92742_Broad_LINCS_pert_info.txt"), di("GSE70138_Broad_LINCS_pert_info_2017-03-06.txt")]

# unique pert_ids in usable signatures (+ iname, sig count)
iname={}; nsig=Counter()
with open(SIGS,encoding="utf-8") as f:
    for r in csv.DictReader(f,delimiter="\t"):
        iname.setdefault(r["pert_id"],r["pert_iname"]); nsig[r["pert_id"]]+=1
print(f"unique usable drugs (pert_id): {len(nsig):,} ; total usable sigs: {sum(nsig.values()):,}")

# canonical SMILES from pert_info (both phases; prefer a non-empty value)
smiles={}
for p in PERT:
    with open(p,encoding="utf-8") as f:
        for r in csv.DictReader(f,delimiter="\t"):
            sm=r.get("canonical_smiles","")
            if sm and sm not in ("-666","restricted","NA") and r["pert_id"] not in smiles:
                smiles[r["pert_id"]]=sm

rows=[(pid,iname[pid],nsig[pid],smiles.get(pid,"")) for pid in sorted(nsig)]
missing=[pid for pid,_,_,sm in rows if not sm]
with open("../outputs/drug_list.tsv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t"); w.writerow(["pert_id","pert_iname","n_sigs","canonical_smiles"])
    for r in rows: w.writerow(r)
print(f"drugs with SMILES: {len(rows)-len(missing):,}/{len(rows):,} ; missing SMILES (should be ~0): {len(missing)}")
if missing[:5]: print("  e.g. missing:", missing[:5])
print("wrote ../outputs/drug_list.tsv")
