# -*- coding: utf-8 -*-
"""
Validate data.py on the REAL tensors (numpy-only; no torch/GPU). Checks gather-by-key shapes,
alignment of atom slices to offsets, condition parsing, and cold-cell splits (MCF10A/NPC).
Run: python model/tests/test_data.py
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
import numpy as np
from config import DataConfig
from data import LincsDataset, build_splits

dc = DataConfig()
shared = LincsDataset.load_shared(dc)
ds = LincsDataset(dc, _shared=shared)
print(f"resolvable signatures: {len(ds):,}")
print(f"Y {ds.Y.shape} | X_base {ds.Xb.shape} | E {ds.E.shape} | r_cell {ds.r_cell.shape}")
print(f"u_feats {ds.u_feats.shape} | atom_reprs {ds.atom_reprs.shape} | offsets {ds.atom_off.shape}")

# splits (cold-cell k-fold)
sp = build_splits(ds, dc)
print("splits:", {k: f"{len(sp[k]):,}" for k in ("train", "val", "test_coldcell")})
print("fold signature loads:", sp["_fold_load"])
row2cell = {v: k for k, v in ds.cell_idx.items()}
test_cells = sorted(row2cell[c] for c in set(ds.cell_row[sp["test_coldcell"]]))
print(f"cold-cell test = fold {dc.cell_fold}: {len(test_cells)} cells -> {test_cells}")

# __getitem__ shapes/values
s = ds[0]
print("sample shapes:", {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in s.items()})
assert s["Y"].shape == (978,) and not np.isnan(s["Y"]).any()
assert s["X"].shape == (978,) and s["E"].shape == (978, 3) and s["r"].shape == (978,)
assert s["u_feats"].shape == (dc_dg := 512 + 384 + 20 + 2048,) and not np.isnan(s["u_feats"]).any()
assert s["atoms"].ndim == 2 and s["atoms"].shape[1] == 512

# atom slice matches offsets for several samples
for k in [0, 1, 100, len(ds) // 2, len(ds) - 1]:
    i = ds.indices[k]; d = ds.drug_row[i]
    exp = int(ds.atom_off[d + 1] - ds.atom_off[d])
    got = ds[k]["atoms"].shape[0]
    assert got == exp, (k, got, exp)

# condition normalization sane (zero-mean-ish, finite)
assert np.isfinite(ds.dose_n).all() and np.isfinite(ds.time_n).all()
print(f"dose_n mean/std {ds.dose_n.mean():.2f}/{ds.dose_n.std():.2f} | "
      f"time_n mean/std {ds.time_n.mean():.2f}/{ds.time_n.std():.2f}")

# split disjointness + coverage
allidx = np.concatenate([sp["train"], sp["val"], sp["test_coldcell"]])
assert len(allidx) == len(np.unique(allidx)) == len(ds), "splits not a disjoint partition"
# designated lines are pinned into fold 0
assert set(dc.cold_cell_test).issubset(set(test_cells)), "pinned cold cells missing from fold 0"
# cold-cell test cells must be ABSENT from train (that is what makes it cold)
train_cells = set(ds.cell_row[sp["train"]])
assert not (set(ds.cell_row[sp["test_coldcell"]]) & train_cells), "cold-cell leak: test cell seen in train"
print(f"phase distribution: {dict(zip(*np.unique(ds.phase, return_counts=True)))}")
print("\nALL DATA CHECKS PASS")
