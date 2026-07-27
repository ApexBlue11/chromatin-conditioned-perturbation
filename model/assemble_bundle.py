# -*- coding: utf-8 -*-
"""
Stage the Kaggle training bundle: every file model/data.py needs, flat, into one dir for upload as the
dataset `apexblue/lincs-train-bundle`. Big arrays (Y, atom reprs) -> fp16 to halve the upload (fine for
z-score targets + features under AMP; data.py upcasts to fp32 on gather). Run with drug/.venv-drug python.
Usage: python assemble_bundle.py <out_dir>
"""
import os, sys, shutil, json
import numpy as np

ROOT = "C:/Projects/LINCS"
OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Surya/AppData/Local/Temp/claude/C--Projects-LINCS/b52b181c-6c61-417b-b2c3-13a1547fd791/scratchpad/kaggle/bundle"
os.makedirs(OUT, exist_ok=True)

# (src relative to ROOT, dst basename, mode)  mode: copy | fp16
FILES = [
    ("phase2_assembly/outputs/Y_target_level5_978.npy", "Y_target_level5_978.npy", "fp16"),
    ("phase2_assembly/outputs/signatures_usable.tsv", "signatures_usable.tsv", "copy"),
    ("baseline/outputs/ccle_baseline_lincs_v5/X_base_lincs.npy", "X_base_lincs.npy", "copy"),
    ("baseline/outputs/ccle_baseline_lincs_v5/ccle_full_background.npy", "ccle_full_background.npy", "copy"),
    ("baseline/outputs/ccle_baseline_lincs_v5/lincs_cell_index.json", "lincs_cell_index.json", "copy"),
    ("phase2_assembly/outputs/E_final.npy", "E_final.npy", "copy"),
    ("phase2_assembly/outputs/E_final_mask.npy", "E_final_mask.npy", "copy"),
    ("phase2_assembly/outputs/E_reliability.tsv", "E_reliability.tsv", "copy"),
    ("network/outputs/A_copathway.npy", "A_copathway.npy", "copy"),
    ("network/outputs/STRING_adj_978.npy", "STRING_adj_978.npy", "copy"),
    ("drug/outputs/drug_feature_index.json", "drug_feature_index.json", "copy"),
    ("drug/outputs/drug_descriptors.npy", "drug_descriptors.npy", "copy"),
    ("drug/outputs/drug_fingerprints.npy", "drug_fingerprints.npy", "copy"),
    ("drug/outputs/drug_unimol.npy", "drug_unimol.npy", "copy"),
    ("drug/outputs/drug_chemberta.npy", "drug_chemberta.npy", "copy"),
    ("drug/outputs/drug_atom_reprs.npy", "drug_atom_reprs.npy", "fp16"),
    ("drug/outputs/drug_atom_offsets.npy", "drug_atom_offsets.npy", "copy"),
]

total = 0
for src, dst, mode in FILES:
    s = os.path.join(ROOT, src); d = os.path.join(OUT, dst)
    if mode == "fp16":
        a = np.load(s, mmap_mode="r")
        np.save(d, np.asarray(a, np.float16)); del a
    else:
        shutil.copy2(s, d)
    sz = os.path.getsize(d); total += sz
    print(f"{dst:34s} {sz/1e6:8.1f} MB  ({mode})")

json.dump({"title": "LINCS Train Bundle", "id": "apexblue/lincs-train-bundle",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(OUT, "dataset-metadata.json"), "w"), indent=2)
print(f"\nTOTAL: {total/1e9:.2f} GB  ->  {OUT}")
