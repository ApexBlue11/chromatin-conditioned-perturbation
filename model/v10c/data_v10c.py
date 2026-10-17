"""v10-C data: the full-release substrate (RESULTS 100.2) in batch form, with chromatin looked up per cell (100.1),
the leave-own-plate-out cell baseline (L6) and plate-complete batches for the contrast loss (L5). RESULTS 100.4.
"""
import glob, json, os

import numpy as np
import torch

DEV_POOL_ROWS = (500, 5000)
DEV_K, DEV_SEED = 6, 0
# The benchmark's dose for each of the full release's 10 dose indices (one-to-one on all 68,830 benchmark rows; measured
# 2026-10-17). `pert_dose` itself can list several doses ('0.12;0.04') where replicates were pooled.
# Review 087 C1 / 088 C1: one cell line under two LINCS names (as build_full_substrate.CELL_ALIASES).
CELL_ALIASES = {'NCIH1975': 'H1975'}
DOSE_OF_IDX = {0: 0.1, 1: 0.37, 2: 0.62, 3: 0.77, 4: 1.11, 5: 2.31, 6: 3.33, 7: 5.0, 8: 10.0, 9: 40.0}


def find(name, roots):
    for r in roots:
        hit = sorted(glob.glob(os.path.join(r, '**', name), recursive=True))
        if hit:
            return hit[0]
    raise SystemExit(f'FATAL: {name} not found under {roots}')


def dev_carve(train_cells, rows_per_cell, chrom_axes, k=DEV_K, seed=DEV_SEED, band=DEV_POOL_ROWS):
    """RESULTS 100.4: pool = training cells with ACC and ACT chromatin and band[0] <= rows <= band[1]; k by RandomState."""
    pool = sorted(c for c in set(train_cells) if chrom_axes.get(c, (False, False))[:2] == (True, True)
                  and band[0] <= rows_per_cell[c] <= band[1])
    if len(pool) < k:
        raise ValueError(f'dev pool {len(pool)} < {k}')
    rs = np.random.RandomState(seed)
    return sorted(pool[i] for i in rs.choice(len(pool), k, replace=False)), pool


def leave_own_plate_out(cell, ctl_id, ctl):
    """x_cell for each row: the mean of its cell's DISTINCT control profiles other than its own (L6). A cell with a single
    distinct control gets that control (flagged)."""
    out = np.empty((len(cell), ctl.shape[1]), np.float32)
    single = np.zeros(len(cell), bool)
    for c in np.unique(cell):
        rows = np.flatnonzero(cell == c)
        ids = np.unique(ctl_id[rows])
        S = ctl[ids].astype(np.float64).sum(0)
        n = len(ids)
        if n == 1:
            out[rows] = ctl[ids[0]]
            single[rows] = True
            continue
        own = ctl[ctl_id[rows]].astype(np.float64)
        out[rows] = ((S[None, :] - own) / (n - 1)).astype(np.float32)
    return out, single


def plate_batches(plates, batch, rng, cap=16):
    """Index batches over rows grouped by plate. Each plate's rows are shuffled and split into chunks of at most `cap`
    (a plate of <= cap rows is one chunk); chunks are shuffled and packed whole into batches of at most `batch` rows.
    Every row appears exactly once per call (100.4 amendment: an earlier version kept only `cap` rows per plate, which
    on the real data used 75k of 255k training rows per epoch). Returns a list of index arrays."""
    order = np.argsort(plates, kind='stable')
    ps = plates[order]
    cuts = np.flatnonzero(np.diff(ps)) + 1
    groups = []
    for g in np.split(order, cuts):
        g = rng.permutation(g)
        groups.extend(np.array_split(g, -(-len(g) // cap)))
    rng.shuffle(groups)
    out, cur, n = [], [], 0
    for g in groups:
        if n + len(g) > batch and cur:
            out.append(np.concatenate(cur))
            cur, n = [], 0
        cur.append(g)
        n += len(g)
    if cur:
        out.append(np.concatenate(cur))
    return [b for b in out if len(b) == batch] + [b for b in out if len(b) != batch]


class V10CData:
    def __init__(self, roots, split='cold_cell_1', dev=False, chrom_off=False, substrate=None, chromatin=None):
        sub = np.load(substrate or find('v10c_full_substrate.npz', roots))
        ch = np.load(chromatin or find('v10c_chromatin.npz', roots))
        di = json.load(open(find('drug_feature_index.json', roots), encoding='utf-8'))
        tissue = json.load(open(find('cell_tissue.json', roots), encoding='utf-8'))['tissue']
        lab = sub['split_' + split]
        pert_all = sub['meta_pert_id']
        feat = np.array([p in di for p in pert_all])
        cell_all = sub['meta_cell_iname']
        tr = (lab == 'train') & feat
        te = (lab == 'test') & feat
        self.n_dropped = {'train': int(((lab == 'train') & ~feat).sum()), 'test': int(((lab == 'test') & ~feat).sum())}
        cells = [str(x) for x in ch['cells']]
        cidx = {c: i for i, c in enumerate(cells)}
        M = ch['M']
        axes = {c: tuple(bool(M[i, :, k].any()) for k in range(3)) for i, c in enumerate(cells)}
        self.dev_cells = None
        if dev:
            rows_per_cell = {c: int(n) for c, n in zip(*np.unique(cell_all[tr], return_counts=True))}
            self.dev_cells, self.dev_pool = dev_carve(cell_all[tr], rows_per_cell, axes)
            ev = tr & np.isin(cell_all, self.dev_cells)
            tr = tr & ~ev
        else:
            ev = te
        keep = np.flatnonzero(tr | ev)
        self.tr = np.flatnonzero(tr[keep])
        self.ev = np.flatnonzero(ev[keep])
        canon = lambda cs: {CELL_ALIASES.get(str(c), str(c)) for c in cs}
        if canon(cell_all[keep][self.tr]) & canon(cell_all[keep][self.ev]) and split.startswith('cold_cell'):
            raise SystemExit('FATAL: a held-out cell (under any of its names) is in training')
        self.sub_row = keep
        self.X = np.asarray(sub['X'][keep], np.float32)
        self.ctl = np.asarray(sub['ctl'], np.float32)
        self.ctl_id = sub['ctl_id'][keep]
        self.cell = cell_all[keep]
        self.pert = pert_all[keep]
        self.plate = sub['plate_id'][keep].astype(np.int64)
        self.bench_row = sub['bench_row'][keep]
        _, self.pert_code = np.unique(self.pert, return_inverse=True)
        dose = np.array([DOSE_OF_IDX[int(k)] for k in sub['meta_pert_dose_idx'][keep]], np.float32)
        time = sub['meta_pert_time'][keep].astype(np.float32)
        ldose = np.log10(np.clip(dose, 1e-4, None))
        self.dose_n = ((ldose - ldose[self.tr].mean()) / (ldose[self.tr].std() + 1e-6)).astype(np.float32)
        self.time_n = ((time - time[self.tr].mean()) / (time[self.tr].std() + 1e-6)).astype(np.float32)
        self.x_cell, self.single_ctl = leave_own_plate_out(self.cell, self.ctl_id, self.ctl)

        self.cell_row = np.array([cidx[c] for c in self.cell])
        F = ch['F'].astype(np.float32)
        Mf = ch['M'].astype(np.float32)
        if chrom_off:
            # RESULTS 45 convention: training-row-weighted mean chromatin AND mask for every cell; machinery intact
            w = np.bincount(self.cell_row[self.tr], minlength=len(cells)).astype(np.float64)
            w /= w.sum()
            F = np.broadcast_to(np.tensordot(w, F, 1).astype(np.float32), F.shape).copy()
            Mf = np.broadcast_to(np.tensordot(w, Mf, 1).astype(np.float32), Mf.shape).copy()
        self.F, self.M = F, Mf
        self.chrom_off = chrom_off
        tis = sorted(set(tissue.values()))
        self.lineage_names = tis
        L = np.zeros((len(cells), len(tis)), np.float32)
        for c, i in cidx.items():
            L[i, tis.index(tissue[c])] = 1.0
        self.L = L

        rows = np.array([di[p] for p in self.pert])
        desc = np.load(find('drug_descriptors.npy', roots)).astype(np.float32)
        desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)          # as v9 (xpert_arm.py)
        fp = np.load(find('drug_fingerprints.npy', roots)).astype(np.float32)
        ucls = np.load(find('drug_unimol.npy', roots)).astype(np.float32)
        self.drug_row = rows
        uniq, inv = np.unique(rows, return_inverse=True)
        self.u_tab = np.concatenate([ucls[uniq], desc[uniq], fp[uniq]], 1)
        self.u_inv = inv
        self.atom_reprs = np.load(find('drug_atom_reprs.npy', roots), mmap_mode='r')
        self.atom_off = np.load(find('drug_atom_offsets.npy', roots))

    def fit_rows(self, n=40000, seed=0):
        """Training rows for the quantiser (100.4: training rows only)."""
        r = self.tr if len(self.tr) <= n else np.sort(np.random.default_rng(seed).choice(self.tr, n, replace=False))
        return self.ctl[self.ctl_id[r]], self.x_cell[r]

    def batch(self, idx, device, max_atoms=96):
        d = self.drug_row[idx]
        n = min(max_atoms, max(1, int(max(self.atom_off[k + 1] - self.atom_off[k] for k in d))))
        atoms = np.zeros((len(idx), n, 512), np.float32)
        amask = np.zeros((len(idx), n), bool)
        for b, k in enumerate(d):
            a0, a1 = int(self.atom_off[k]), int(self.atom_off[k + 1])
            m = min(n, a1 - a0)
            atoms[b, :m] = np.asarray(self.atom_reprs[a0:a0 + m], np.float32)
            amask[b, :m] = True
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(device)
        C = self.ctl[self.ctl_id[idx]]
        cr = self.cell_row[idx]
        return {'x_ctl': t(C), 'x_cell': t(self.x_cell[idx]), 'y_abs': t(self.X[idx]), 'y_delta': t(self.X[idx] - C),
                'F': t(self.F[cr]), 'M': t(self.M[cr]), 'cell_ctx': t(self.L[cr]),
                'atoms': t(atoms), 'atom_mask': t(amask), 'u_feats': t(self.u_tab[self.u_inv[idx]]),
                'dose': t(self.dose_n[idx]), 'time': t(self.time_n[idx]),
                'plate': t(self.plate[idx]), 'pert': t(self.pert_code[idx])}
