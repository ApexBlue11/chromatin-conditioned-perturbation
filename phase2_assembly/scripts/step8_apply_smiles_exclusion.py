# -*- coding: utf-8 -*-
"""
step8_apply_smiles_exclusion.py -- exclude the 79 drugs with NO canonical_smiles (2,324 sigs;
un-featurizable by the drug branch). Produces the usable signature list + a row keep-mask for
Y_target. The doc's 2 PROPRIETARY drugs (legal exclusion) are STILL PENDING identification --
recorded as a TODO, not yet applied.
"""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

sig=pd.read_csv("../outputs/signatures.tsv",sep="\t",dtype=str,keep_default_na=False)
n0=len(sig)
sig["restricted_smiles"]=sig["restricted_smiles"].astype(int)
keep = sig["restricted_smiles"]==0
usable=sig[keep].copy()
excl_drugs=sorted(sig.loc[~keep,"pert_id"].unique())
print(f"total trt_cp sigs:        {n0:,}")
print(f"no-SMILES sigs excluded:  {(~keep).sum():,}  ({len(excl_drugs)} drugs)")
print(f"usable sigs:              {len(usable):,}")

# row keep-mask aligned to Y_target rows (row column == original index)
mask=keep.to_numpy()
np.save("../outputs/Y_keep_mask_smiles.npy", mask)
usable.to_csv("../outputs/signatures_usable.tsv",sep="\t",index=False)
with open("../outputs/EXCLUSIONS_TODO.md","w",encoding="utf-8") as f:
    f.write("# Dataset exclusions\n\n")
    f.write(f"## Applied: no-SMILES drugs ({len(excl_drugs)} drugs, {(~keep).sum()} sigs)\n")
    f.write("Un-featurizable (empty canonical_smiles). Row keep-mask: Y_keep_mask_smiles.npy.\n\n")
    f.write("Excluded pert_ids:\n"+", ".join(excl_drugs)+"\n\n")
    f.write("## PENDING: 2 proprietary drugs (legal exclusion from the architecture doc)\n")
    f.write("Not yet identifiable (doc not on disk). MUST be excluded from train/val/test once IDs are known.\n")
print("wrote signatures_usable.tsv, Y_keep_mask_smiles.npy, EXCLUSIONS_TODO.md")
print("NOTE: the 2 proprietary-drug legal exclusion is STILL PENDING (needs the architecture doc).")
