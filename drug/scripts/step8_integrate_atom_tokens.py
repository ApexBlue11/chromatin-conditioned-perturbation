# -*- coding: utf-8 -*-
"""
step8_integrate_atom_tokens.py -- validate + fold in the per-atom UniMol tokens from the Kaggle kernel
`apexblue/lincs-atom-tokens` (remove_hs=True; atomic_reprs). These are the drug ATOM TOKENS the model
attends over (atom->gene), distinct from the pooled CLS. Ragged storage: reprs[off[j]:off[j+1]] = drug j.

Usage: python step8_integrate_atom_tokens.py [download_dir]
Checks pert_id alignment + offset integrity, compares token counts vs RDKit heavy-atom counts
(detects any extra global node), then moves the arrays into drug/outputs/.
"""
import sys, os, json, csv
import numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
sys.stdout.reconfigure(encoding="utf-8")

DL = sys.argv[1] if len(sys.argv) > 1 else "../outputs/_atom_dl"
OUT = "../outputs"

def main():
    reprs = np.load(f"{DL}/drug_atom_reprs.npy", mmap_mode="r")
    off = np.load(f"{DL}/drug_atom_offsets.npy")
    counts = np.load(f"{DL}/drug_atom_counts.npy")
    cls = np.load(f"{DL}/drug_atom_cls.npy")
    meta = json.load(open(f"{DL}/drug_atom_meta.json"))
    kp = json.load(open(f"{DL}/drug_pert_ids.json"))
    print("kernel meta:", meta)

    rows = list(csv.DictReader(open(f"{OUT}/drug_list.tsv", encoding="utf-8"), delimiter="\t"))
    pids = [r["pert_id"] for r in rows]
    N = len(pids)

    # alignment + integrity
    assert kp == pids, "kernel pert_id order != drug_list order"
    idx = json.load(open(f"{OUT}/drug_feature_index.json"))
    assert all(idx[p] == i for i, p in enumerate(pids)), "drug_feature_index order mismatch"
    assert len(counts) == N and len(off) == N + 1, (len(counts), len(off), N)
    assert int(off[-1]) == reprs.shape[0] == int(counts.sum()), (off[-1], reprs.shape[0], counts.sum())
    assert cls.shape == (N, 512), cls.shape
    print(f"alignment OK: {N:,} drugs | tokens={reprs.shape[0]:,} dim={reprs.shape[1]}")

    # token count stats + off-by-one check vs RDKit heavy atoms
    heavy = np.array([Chem.MolFromSmiles(r["canonical_smiles"]).GetNumAtoms()
                      if Chem.MolFromSmiles(r["canonical_smiles"]) else 0 for r in rows])
    cap = int(meta.get("max_atoms", 96))
    heavy_cap = np.minimum(heavy, cap)
    diff = counts.astype(int) - heavy_cap
    zero = int((counts == 0).sum())
    print(f"tokens/mol: mean={counts.mean():.1f} median={np.median(counts):.0f} max={counts.max()} | "
          f"failed(0 tokens)={zero}")
    print(f"count - rdkit_heavy(capped): mode={np.bincount(diff - diff.min()).argmax() + diff.min()} "
          f"| ==0: {int((diff==0).sum())}/{N} | ==+1: {int((diff==1).sum())} (extra global node if large)")

    # release the memmap handle (Windows can't rename an open/mmapped file), then move
    del reprs
    import gc; gc.collect()
    for fn in ("drug_atom_reprs.npy", "drug_atom_offsets.npy", "drug_atom_counts.npy",
               "drug_atom_cls.npy", "drug_atom_meta.json"):
        os.replace(f"{DL}/{fn}", f"{OUT}/{fn}")
    print("integrated atom tokens into", OUT)

if __name__ == "__main__":
    main()
