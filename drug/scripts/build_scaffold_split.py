# -*- coding: utf-8 -*-
"""
Bemis-Murcko SCAFFOLD split for the unseen-COMPOUND benchmark (+ a Tanimoto leakage audit).

Why not a random drug split: SOTA papers report "unseen compound" numbers, but a random holdout leaves
structurally near-identical analogues on both sides (e.g. a series of β-blockers), so the model can win by
memorising a scaffold rather than generalising chemistry. We split by Bemis-Murcko scaffold so a whole
chemical series lands on ONE side, then AUDIT the residual leakage: max ECFP4/Tanimoto similarity from each
test drug to its nearest TRAIN drug. Report that distribution -- it is the honest measure of how "unseen"
the unseen-compound split really is (and lets us compare our split's strictness to anyone else's).

Output: drug/outputs/splits/scaffold_split.json  {folds, per-drug scaffold, leakage audit}
Run: drug/.venv-drug/Scripts/python.exe drug/scripts/build_scaffold_split.py
"""
import os, sys, csv, json
from collections import defaultdict
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator, DataStructs

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIST = os.path.join(ROOT, "drug", "outputs", "drug_list.tsv")
OUT = os.path.join(ROOT, "drug", "outputs", "splits")
N_FOLDS = 5


def main():
    rows = list(csv.DictReader(open(LIST, encoding="utf-8"), delimiter="\t"))
    print(f"drugs: {len(rows):,}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    scaf, fps, pids, bad = {}, {}, [], 0
    for r in rows:
        pid, smi = r["pert_id"], r["canonical_smiles"]
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            bad += 1
            continue
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            s = ""
        scaf[pid] = s or "__ACYCLIC__"       # acyclic/scaffold-less compounds share one bucket
        fps[pid] = gen.GetFingerprint(m)
        pids.append(pid)
    print(f"parsed: {len(pids):,}  (unparseable: {bad})")

    groups = defaultdict(list)
    for pid in pids:
        groups[scaf[pid]].append(pid)
    print(f"distinct Bemis-Murcko scaffolds: {len(groups):,}")
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"largest scaffold groups: {sizes[:8]}  | singletons: {sum(1 for s in sizes if s == 1):,}")

    # greedy balanced assignment: biggest scaffold group -> currently lightest fold
    folds = {f: [] for f in range(N_FOLDS)}
    load = {f: 0 for f in range(N_FOLDS)}
    for s in sorted(groups, key=lambda k: -len(groups[k])):
        f = min(load, key=lambda k: load[k])
        folds[f].extend(groups[s]); load[f] += len(groups[s])
    print(f"fold sizes: {dict(load)}")

    # ---- leakage audit on fold 0 as the test set ----
    test = folds[0]
    train = [p for f in range(1, N_FOLDS) for p in folds[f]]
    tr_fps = [fps[p] for p in train]
    nn = []
    for p in test:
        sims = DataStructs.BulkTanimotoSimilarity(fps[p], tr_fps)
        nn.append(max(sims) if sims else 0.0)
    nn = np.array(nn)
    print("\n=== LEAKAGE AUDIT (fold 0 = test): max Tanimoto to any TRAIN drug ===")
    print(f"  median={np.median(nn):.3f}  mean={nn.mean():.3f}  p90={np.percentile(nn,90):.3f}  max={nn.max():.3f}")
    for thr in (0.4, 0.5, 0.6, 0.7, 0.85):
        print(f"  test drugs with a train neighbour >= {thr:.2f} Tanimoto: "
              f"{(nn >= thr).mean()*100:5.1f}%  ({int((nn >= thr).sum())}/{len(nn)})")
    print("  (a RANDOM drug split would sit far higher -- this is the honesty check on 'unseen compound')")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"n_folds": N_FOLDS, "fold_sizes": load,
               "folds": {str(f): folds[f] for f in folds},
               "scaffold_of": scaf,
               "leakage_audit_fold0": {
                   "median_max_tanimoto": float(np.median(nn)), "mean": float(nn.mean()),
                   "p90": float(np.percentile(nn, 90)), "max": float(nn.max()),
                   "frac_ge_0.5": float((nn >= 0.5).mean()), "frac_ge_0.7": float((nn >= 0.7).mean())}},
              open(os.path.join(OUT, "scaffold_split.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'scaffold_split.json')}")


if __name__ == "__main__":
    main()
