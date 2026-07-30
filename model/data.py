# -*- coding: utf-8 -*-
"""
LincsDataset + collate + splits. NORMALIZED design: no per-row copies of cell/drug tensors -- each
__getitem__ GATHERS Y[i], X[cell], E[cell], r[cell], atom tokens+globals[drug], dose, time BY KEY.
Dataset uses numpy only (so split/parse logic is testable without torch); collate builds torch tensors.

Keys: signatures_usable.tsv (row->Y index, cell_id, pert_id, dose, time). Cell order = lincs_cell_index
(== epigenetics order, verified). Drug order = drug_feature_index.json. dose/time are raw strings
("10 µM","6 h") -> parsed -> log-dose standardized, time standardized.
"""
import os, csv, json, re
import numpy as np


def _num(s):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(m.group()) if m else np.nan


def _dose_um(s):
    """Dose in MICROMOLAR, unit-aware.

    BUG FIXED 2026-07-30: the raw `dose` field mixes units ('10 uM', '500 nM' -- 110 distinct strings), and
    reading only the leading number turned 500 nM into 500 uM. That is 1000x too large on **13,910 rows
    (4.49 %)**, and because it is a *multiplicative* error on a log axis it placed the LOWEST doses at the
    TOP of the standardised log-dose feature -- worse than dropping them. Only uM and nM occur (no mM)."""
    t = str(s).lower().replace("µ", "u").replace("μ", "u")
    v = _num(t)
    if np.isnan(v):
        return v
    if "nm" in t:
        return v / 1000.0
    if "mm" in t:
        return v * 1000.0
    return v                       # uM, or a bare number taken as uM


class LincsDataset:
    def __init__(self, dcfg, indices=None, _shared=None):
        self.dcfg = dcfg
        R = lambda p: os.path.join(dcfg.root, p)

        if _shared is None:
            _shared = LincsDataset.load_shared(dcfg)
        self.__dict__.update(_shared)                       # arrays shared across splits (built once)

        self.indices = np.arange(len(self.y_row)) if indices is None else np.asarray(indices)

    # ---- one-time shared load (call once, pass _shared to each split to avoid re-loading) ----
    @staticmethod
    def load_shared(dcfg):
        R = lambda p: os.path.join(dcfg.root, p)
        cell_idx = json.load(open(R(dcfg.xbase_path).replace("X_base_lincs.npy", "lincs_cell_index.json")))
        cell_idx = cell_idx.get("cell_id_to_row", cell_idx)

        Y = np.load(R(dcfg.y_path), mmap_mode="r")           # [N,978]
        Xb = np.load(R(dcfg.xbase_path)).astype(np.float32)  # [83,978]
        E = np.load(R(dcfg.e_path)).astype(np.float32)       # [83,978,3]
        Emask = np.load(R(dcfg.e_mask_path))                 # [83,978,3] bool

        if getattr(dcfg, "center_epi", False):
            # per-(cell,mark) standardize across the 978 genes -> removes the technical per-cell offset
            # (batch effect), keeps the informative cell x gene interaction. Only where the cell has the mark.
            for k in range(E.shape[2]):
                has = Emask[:, :, k].any(1)                   # [83] cells with mark k
                for c in np.where(has)[0]:
                    v = E[c, :, k]; m, s = v.mean(), v.std()
                    E[c, :, k] = (v - m) / (s + 1e-6)

        # per-(cell,mark) reliability -> per-(cell,gene) r via available marks
        rel = {}
        with open(R(dcfg.e_reliability_path), encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rel[row["cell_id"]] = [float(row["ATAC-seq"]), float(row["H3K27ac"]), float(row["H3K27me3"])]
        relmat = np.zeros((Xb.shape[0], 3), np.float32)      # [83,3]
        row2cell = {v: k for k, v in cell_idx.items()}
        for ci in range(Xb.shape[0]):
            relmat[ci] = rel.get(row2cell.get(ci, ""), [0, 0, 0])
        avail = Emask.astype(np.float32)                     # [83,978,3]
        num = (avail * relmat[:, None, :]).sum(-1)           # sum reliab over available marks
        den = avail.sum(-1)
        r_cell = np.where(den > 0, num / np.maximum(den, 1), 0.0).astype(np.float32)  # [83,978]

        # Dose parsing. `legacy_dose_parsing=True` restores the unit-BLIND parse (500 nM -> 500 uM) that
        # trained every checkpoint before 2026-07-30. It exists because evaluating such a checkpoint with
        # the corrected parse would feed 13,910 rows (4.49 %) a log-dose the model never saw during
        # training, quietly understating it. Match the flag to how the checkpoint was trained.
        _dose = _num if getattr(dcfg, "legacy_dose_parsing", False) else _dose_um

        # drug features -> global u_feats + atom tokens
        dindex = json.load(open(R(dcfg.drug_index_path)))    # pert_id -> row
        desc = np.load(R(dcfg.desc_path)).astype(np.float32)
        desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)  # standardize physchem descriptors
        fp = np.load(R(dcfg.fp_path)).astype(np.float32)
        ucls = np.load(R(dcfg.unimol_cls_path)).astype(np.float32)
        # ChemBERTa dropped by default (ablation: ΔR²=+0.001, dead weight). Keep the flag on ModelConfig.
        blocks = [ucls]
        if getattr(dcfg, "use_chemberta", False):
            blocks.append(np.load(R(dcfg.chemberta_path)).astype(np.float32))
        blocks += [desc, fp]
        u_feats = np.concatenate(blocks, axis=1).astype(np.float32)   # [n_drug, 2580] (2964 with chemberta)
        atom_reprs = np.load(R(dcfg.atom_reprs_path), mmap_mode="r")   # [sum,512]
        atom_off = np.load(R(dcfg.atom_offsets_path))                  # [n_drug+1]

        # signatures -> per-row keys (only usable rows with resolvable cell+drug)
        y_row, cell_row, drug_row, dose_raw, time_raw, phase = [], [], [], [], [], []
        with open(R(dcfg.sig_path), encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                c = cell_idx.get(row["cell_id"]); d = dindex.get(row["pert_id"])
                if c is None or d is None:
                    continue
                y_row.append(int(row["row"])); cell_row.append(c); drug_row.append(d)
                dose_raw.append(_dose(row["dose"])); time_raw.append(_num(row["time"]))
                phase.append(row.get("phase", ""))
        y_row = np.array(y_row); cell_row = np.array(cell_row); drug_row = np.array(drug_row)
        dose = np.array(dose_raw, np.float32); time = np.array(time_raw, np.float32)
        # normalize condition tokens (log-dose standardized; time standardized)
        ld = np.log10(np.clip(dose, 1e-4, None)); ld = np.nan_to_num(ld, nan=np.nanmean(ld))
        dose_n = ((ld - ld.mean()) / (ld.std() + 1e-6)).astype(np.float32)
        tt = np.nan_to_num(time, nan=np.nanmean(time))
        time_n = ((tt - tt.mean()) / (tt.std() + 1e-6)).astype(np.float32)

        # v3: per-signature reliability weight from the MEASURED strength->replicate-r curve
        # (bins: mean|Y| 0.48/0.65/0.80/1.11/1.63/2.5 -> replicate r .074/.088/.136/.365/.646/.751)
        sp = R(getattr(dcfg, "strength_path", ""))
        if getattr(dcfg, "reliability_weighting", False) and os.path.exists(sp):
            st_all = np.load(sp).astype(np.float32)
            strength = st_all[y_row]
            rel = np.interp(strength, [0.48, 0.65, 0.80, 1.11, 1.63, 2.50],
                            [0.074, 0.088, 0.136, 0.365, 0.646, 0.751]).astype(np.float32)
            weight = np.clip(rel, 0.05, 1.0)
        else:
            strength = np.ones(len(y_row), np.float32); weight = np.ones(len(y_row), np.float32)

        # scaffold fold per DRUG (for the unseen-COMPOUND benchmark). -1 = no scaffold assignment.
        drug_fold = np.full(u_feats.shape[0], -1, np.int64)
        sp_path = R(getattr(dcfg, "scaffold_split_path", ""))
        if sp_path and os.path.exists(sp_path):
            sj = json.load(open(sp_path))
            for f, pids in sj.get("folds", {}).items():
                for pid in pids:
                    d = dindex.get(pid)
                    if d is not None:
                        drug_fold[d] = int(f)

        # per-cell context (lineage one-hot); zeros if the file is absent so older bundles still load
        lp = R(getattr(dcfg, "lineage_path", ""))
        if lp and os.path.exists(lp):
            cell_ctx = np.load(lp).astype(np.float32)
        else:
            cell_ctx = np.zeros((Xb.shape[0], 16), np.float32); cell_ctx[:, 0] = 1.0

        return dict(cell_ctx=cell_ctx, drug_fold=drug_fold, strength=strength, weight=weight,
                    Y=Y, Xb=Xb, E=E, Emask=Emask, r_cell=r_cell, u_feats=u_feats,
                    atom_reprs=atom_reprs, atom_off=atom_off, y_row=y_row, cell_row=cell_row,
                    drug_row=drug_row, dose_n=dose_n, time_n=time_n, cell_idx=cell_idx,
                    phase=np.array(phase))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        d = self.drug_row[i]; c = self.cell_row[i]
        a0, a1 = int(self.atom_off[d]), int(self.atom_off[d + 1])
        return {
            "Y": np.asarray(self.Y[self.y_row[i]], np.float32),
            "X": self.Xb[c], "E": self.E[c], "r": self.r_cell[c], "cell_ctx": self.cell_ctx[c],
            "atoms": np.asarray(self.atom_reprs[a0:a1], np.float32),   # (n_atoms,512)
            "u_feats": self.u_feats[d],
            "dose": self.dose_n[i], "time": self.time_n[i],
            "w": self.weight[i],
        }


def collate(samples, max_atoms=96, fixed_pad=False):
    """fixed_pad=True always pads atoms to exactly `max_atoms` (STATIC shape) instead of the batch max.
    Needed for XLA/TPU, where a varying M triggers a recompile every batch. SAFETY: padded positions are
    False in `atom_mask` -> the model sets key_mask=~valid -> those keys get -inf before the softmax, so
    attention CANNOT land on padding. The global drug token is always valid, so no query row is fully
    masked (which would produce NaN). Verified by test_padding_invariance: identical Yhat for
    fixed_pad=True vs False."""
    import torch
    B = len(samples)
    M = max_atoms if fixed_pad else min(max_atoms, max(1, max(s["atoms"].shape[0] for s in samples)))
    atoms = np.zeros((B, M, 512), np.float32)
    amask = np.zeros((B, M), bool)
    for b, s in enumerate(samples):
        n = min(M, s["atoms"].shape[0])
        atoms[b, :n] = s["atoms"][:n]; amask[b, :n] = True
    # force float32: python-float scalars (dose/time/w) would otherwise stack to float64 and blow up in
    # the first Linear ("mat1 and mat2 must have the same dtype")
    t = lambda key: torch.from_numpy(np.stack([s[key] for s in samples]).astype(np.float32))
    return {
        "Y": t("Y"), "X": t("X"), "E": t("E"), "r": t("r"), "u_feats": t("u_feats"),
        "cell_ctx": t("cell_ctx"), "dose": t("dose"), "time": t("time"), "w": t("w"),
        "atoms": torch.from_numpy(atoms), "atom_mask": torch.from_numpy(amask),
    }


def cell_folds(ds, dcfg):
    """Partition cells into n_cell_folds balanced by signature count (greedy: assign the largest
    remaining cell to the lightest fold). dcfg.cold_cell_test cells are pinned into fold 0 so the
    originally designated test lines stay in fold 0. Returns {fold: [cell_row,...]}."""
    counts = np.bincount(ds.cell_row, minlength=max(ds.cell_idx.values()) + 1)
    pinned = [ds.cell_idx[c] for c in dcfg.cold_cell_test if c in ds.cell_idx]
    folds = {f: [] for f in range(dcfg.n_cell_folds)}
    load = {f: 0 for f in range(dcfg.n_cell_folds)}
    for c in pinned:
        folds[0].append(int(c)); load[0] += int(counts[c])
    for c in np.argsort(-counts):
        if c in pinned or counts[c] == 0:
            continue
        f = min(load, key=lambda k: load[k])
        folds[f].append(int(c)); load[f] += int(counts[c])
    return folds, load


def build_splits(ds, dcfg):
    """Cold-CELL x cold-COMPOUND split. Test cells = cells in `dcfg.cell_fold`; test drugs = drugs in the
    Bemis-Murcko scaffold fold `dcfg.drug_fold` (drug/outputs/splits/scaffold_split.json). Training excludes
    BOTH, which yields three disjoint generalization tests from ONE run:
        test_coldcell  - unseen CELL,     seen drug     (the classic cold-cell number)
        test_colddrug  - seen cell,       unseen COMPOUND (comparable to SOTA "unseen compound")
        test_coldboth  - unseen cell AND unseen compound (the hardest)
    Set dcfg.drug_fold=None to disable the compound holdout (v3/v4 behaviour)."""
    folds, load = cell_folds(ds, dcfg)
    test_cells = folds[dcfg.cell_fold]
    is_test_cell = np.isin(ds.cell_row, test_cells)

    df = getattr(dcfg, "drug_fold", None)
    if df is None or not hasattr(ds, "drug_fold") or (ds.drug_fold >= 0).sum() == 0:
        is_test_drug = np.zeros(len(ds.y_row), bool)
    else:
        is_test_drug = ds.drug_fold[ds.drug_row] == int(df)

    coldcell = np.where(is_test_cell & ~is_test_drug)[0]
    colddrug = np.where(~is_test_cell & is_test_drug)[0]
    coldboth = np.where(is_test_cell & is_test_drug)[0]
    rest = np.where(~is_test_cell & ~is_test_drug)[0]
    rng = np.random.default_rng(dcfg.seed); rng.shuffle(rest)
    n_val = int(len(rest) * dcfg.val_frac)
    return {"val": rest[:n_val], "train": rest[n_val:],
            "test_coldcell": coldcell, "test_colddrug": colddrug, "test_coldboth": coldboth,
            "_test_cells": test_cells, "_fold_load": load}
