# -*- coding: utf-8 -*-
"""
step2_compute_descriptors.py -- physicochemical descriptors + Morgan (ECFP4) fingerprints for every
usable drug, keyed by pert_id. Interpretable descriptor set (Lipinski/shape/complexity). Run with
.venv-drug python. Outputs: drug_descriptors.npy (N x D), drug_fingerprints.npy (N x 2048),
drug_feature_index.json (pert_id -> row), drug_descriptor_names.json, drug_featurize_log.txt.
"""
import sys, csv, json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.stdout.reconfigure(encoding="utf-8")

# curated interpretable descriptor set
DESC = [
 ("MolWt", Descriptors.MolWt), ("LogP", Crippen.MolLogP), ("TPSA", Descriptors.TPSA),
 ("HBD", Descriptors.NumHDonors), ("HBA", Descriptors.NumHAcceptors),
 ("RotatableBonds", Descriptors.NumRotatableBonds), ("AromaticRings", Descriptors.NumAromaticRings),
 ("AliphaticRings", Descriptors.NumAliphaticRings), ("RingCount", Descriptors.RingCount),
 ("FractionCSP3", Descriptors.FractionCSP3), ("HeavyAtoms", Descriptors.HeavyAtomCount),
 ("Heteroatoms", Descriptors.NumHeteroatoms), ("SaturatedRings", Descriptors.NumSaturatedRings),
 ("QED", QED.qed), ("MolMR", Crippen.MolMR), ("LabuteASA", Descriptors.LabuteASA),
 ("BertzCT", Descriptors.BertzCT), ("NumNO", rdMolDescriptors.CalcNumLipinskiHBA),
 ("FormalCharge", lambda m: Chem.GetFormalCharge(m)), ("NumStereo", rdMolDescriptors.CalcNumAtomStereoCenters),
]
NAMES=[n for n,_ in DESC]; FP_BITS=2048; FP_RADIUS=2  # ECFP4

rows=list(csv.DictReader(open("../outputs/drug_list.tsv",encoding="utf-8"),delimiter="\t"))
N=len(rows)
D=np.full((N,len(DESC)),np.nan,dtype=np.float32)
FP=np.zeros((N,FP_BITS),dtype=np.uint8)
index={}; log=[]; fails=0
for i,r in enumerate(rows):
    pid=r["pert_id"]; index[pid]=i
    m=Chem.MolFromSmiles(r["canonical_smiles"])
    if m is None:
        fails+=1; log.append(f"PARSE_FAIL {pid} {r['canonical_smiles'][:60]}"); continue
    for j,(_,fn) in enumerate(DESC):
        try: D[i,j]=float(fn(m))
        except Exception: pass
    try:
        bv=AllChem.GetMorganFingerprintAsBitVect(m,FP_RADIUS,nBits=FP_BITS)
        FP[i]=np.frombuffer(bytes(bv.ToBitString(),"ascii"),dtype=np.uint8)-ord("0")
    except Exception: pass

np.save("../outputs/drug_descriptors.npy",D); np.save("../outputs/drug_fingerprints.npy",FP)
json.dump(index,open("../outputs/drug_feature_index.json","w"))
json.dump(NAMES,open("../outputs/drug_descriptor_names.json","w"),indent=1)
open("../outputs/drug_featurize_log.txt","w",encoding="utf-8").write("\n".join(log))
valid=~np.isnan(D).any(axis=1)
print(f"drugs featurized: {N:,} | SMILES parse failures: {fails} | fully-valid descriptor rows: {int(valid.sum()):,}")
print(f"descriptors: {len(NAMES)} ({', '.join(NAMES[:6])}, ...) | fingerprint: ECFP4 {FP_BITS}-bit")
print(f"FP mean bits set: {FP.sum(1).mean():.1f} | descriptor NaN cols: {int(np.isnan(D).any(axis=0).sum())}")
print("wrote drug_descriptors.npy, drug_fingerprints.npy, drug_feature_index.json, names, log")
