# -*- coding: utf-8 -*-
"""
step3_integrate_kaggle_embeddings.py -- validate + fold in the drug embeddings computed on Kaggle
(UniMol 3D CLS + ChemBERTa "MolBERTa" + optional MoLFormer). The heavy compute runs in the Kaggle
GPU kernel `apexblue/lincs-drug-embeddings-unimol-molberta` (T4x2); this script only ingests its
downloaded output, checks pert_id alignment against drug_feature_index.json, and copies the arrays
into drug/outputs/ as normalized pert_id-keyed lookups.

Usage:  python step3_integrate_kaggle_embeddings.py <download_dir>
where <download_dir> holds the kernel output: drug_unimol.npy, drug_unimol_mask.npy,
drug_chemberta.npy, [drug_molformer.npy], drug_embed_meta.json, drug_pert_ids.json
"""
import sys, os, json, shutil
import numpy as np

DL = sys.argv[1] if len(sys.argv) > 1 else "../outputs/_kaggle_download"
OUT = "../outputs"
INDEX = f"{OUT}/drug_feature_index.json"          # pert_id -> row (from step2)
LIST = f"{OUT}/drug_list.tsv"

def load_order():
    # canonical row order = drug_list.tsv order (== drug_feature_index.json)
    pids = []
    with open(LIST, encoding="utf-8") as f:
        next(f)
        for line in f:
            pids.append(line.split("\t", 1)[0])
    idx = json.load(open(INDEX))
    assert len(idx) == len(pids), f"index {len(idx)} != list {len(pids)}"
    for i, p in enumerate(pids):
        assert idx.get(p) == i, f"order mismatch at {i}: {p}"
    return pids

def main():
    pids = load_order()
    N = len(pids)
    print(f"canonical drugs: {N:,}")

    # 1) alignment check against the kernel's saved pert_id order
    kp_path = f"{DL}/drug_pert_ids.json"
    if os.path.exists(kp_path):
        kp = json.load(open(kp_path))
        assert kp == pids, "kernel pert_id order does NOT match drug_list order — alignment broken"
        print("alignment OK: kernel pert_id order == drug_list order")
    else:
        print("WARN: no drug_pert_ids.json from kernel; assuming row order preserved")

    if os.path.exists(f"{DL}/drug_embed_meta.json"):
        print("kernel meta:", json.load(open(f"{DL}/drug_embed_meta.json")))

    # 2) validate + copy each embedding
    specs = [
        ("drug_unimol.npy", "UniMol v1 CLS (3D)"),
        ("drug_chemberta.npy", "ChemBERTa-77M (MolBERTa)"),
        ("drug_molformer.npy", "MoLFormer-XL"),
    ]
    for fn, desc in specs:
        p = f"{DL}/{fn}"
        if not os.path.exists(p):
            print(f"-- {fn}: MISSING (skipped)"); continue
        a = np.load(p)
        assert a.shape[0] == N, f"{fn} rows {a.shape[0]} != {N}"
        nan = int(np.isnan(a).any(axis=1).sum())
        zero = int((~a.any(axis=1)).sum())
        print(f"++ {fn:24s} {str(a.shape):14s} {desc} | all-NaN rows={nan} all-zero rows={zero}")
        shutil.copy2(p, f"{OUT}/{fn}")

    # 3) mask
    mp = f"{DL}/drug_unimol_mask.npy"
    if os.path.exists(mp):
        m = np.load(mp)
        assert m.shape[0] == N
        print(f"++ drug_unimol_mask.npy  {str(m.shape):14s} ok(conformer)={int(m.sum()):,}/{N:,} "
              f"failed={int((~m).sum()):,}")
        shutil.copy2(mp, f"{OUT}/drug_unimol_mask.npy")

    for extra in ("drug_embed_meta.json",):
        if os.path.exists(f"{DL}/{extra}"):
            shutil.copy2(f"{DL}/{extra}", f"{OUT}/{extra}")
    print("integrated into", OUT)

if __name__ == "__main__":
    main()
