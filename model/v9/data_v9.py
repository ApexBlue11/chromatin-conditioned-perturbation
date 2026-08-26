# -*- coding: utf-8 -*-
"""
v9 dataset: the Level-3 substrate on top of v6's proven key/split machinery.

What is new relative to `model/data.py`:
  * X_ctl (the plate-matched DMSO median) and X_cell (the per-cell aggregate of those controls) replace
    the CCLE proxy as the baseline INPUT;
  * three targets per row -- absolute Level-3, its delta, and the Level-5 z-score -- each with its own
    row mask, because a signature can have one and not the other and averaging over a NaN would poison a
    batch silently;
  * the bridge from dataset position to Level-3 row goes through y_row, never through position.
    `signatures_usable.tsv` has 310,114 lines but its `row` column indexes a 312,438-row Level-5 target,
    so position-based indexing pairs a signature's inputs with a DIFFERENT signature's labels.

Splits, dose parsing, drug features, reliability weighting and the scaffold holdout are inherited
unchanged: they are measured-good and re-deriving them here would be a silent divergence risk.
"""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))

from data import LincsDataset, build_splits, cell_folds          # noqa: F401


class LincsV9Dataset(LincsDataset):
    """LincsDataset + the Level-3 arrays. `load_shared_v9` is called once and passed to every split."""

    @staticmethod
    def load_shared_v9(dcfg):
        shared = LincsDataset.load_shared(dcfg)
        R = lambda p: p if os.path.isabs(p) else os.path.join(dcfg.root, p)
        cache = getattr(dcfg, 'cache_in_ram', False)
        mm = None if cache else 'r'
        Xtrt = np.load(R(dcfg.x_trt_path), mmap_mode=mm)
        Xctl = np.load(R(dcfg.x_ctl_path), mmap_mode=mm)
        cov = np.load(R(dcfg.l3_covered_path))
        yrow_l3 = np.load(R(dcfg.l3_yrow_path))

        y2pos = {int(y): i for i, y in enumerate(yrow_l3)}
        ds_to_l3 = np.array([y2pos.get(int(y), -1) for y in shared['y_row']], np.int64)
        has_l3 = (ds_to_l3 >= 0) & cov[np.clip(ds_to_l3, 0, None)]

        # Per-cell aggregate control -- the quantity the A/B called `cellmean`, which RESULTS 27.5 shows
        # is the better input on unseen CELLS. It aggregates that cell's DMSO controls, which carry no
        # drug response and are available at inference for any cell you can plate, so it is a legitimate
        # input rather than a label. It is nevertheless taken LEAVE-ONE-OUT in __getitem__: a cell with
        # few signatures would otherwise have its aggregate dominated by the very row being predicted.
        cell_row = shared['cell_row']
        n_cell = shared['Xb'].shape[0]
        acc = np.zeros((n_cell, Xctl.shape[1]), np.float64)
        cnt = np.zeros(n_cell, np.float64)
        idx = np.flatnonzero(has_l3)
        for lo in range(0, len(idx), 20000):
            sub = idx[lo:lo + 20000]
            rows = ds_to_l3[sub]
            o = np.argsort(rows)
            np.add.at(acc, cell_row[sub][o], np.asarray(Xctl[np.sort(rows)], np.float64))
            np.add.at(cnt, cell_row[sub][o], 1.0)
        x_cell = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
        x_cell[cnt == 0] = np.nan

        shared.update(Xtrt=Xtrt, Xctl=Xctl, ds_to_l3=ds_to_l3, has_l3=has_l3, x_cell=x_cell,
                      cell_acc=acc, cell_ctl_count=cnt.astype(np.int64))
        return shared

    def __getitem__(self, k):
        i = self.indices[k]
        d = self.drug_row[i]
        c = self.cell_row[i]
        a0, a1 = int(self.atom_off[d]), int(self.atom_off[d + 1])
        j = int(self.ds_to_l3[i])
        ok = bool(self.has_l3[i])
        G = self.Xctl.shape[1]

        if ok:
            ctl = np.asarray(self.Xctl[j], np.float32)
            trt = np.asarray(self.Xtrt[j], np.float32)
        else:
            # never a NaN in an INPUT: the per-cell aggregate stands in, and m_l3 turns off both L3 losses
            ctl = self.x_cell[c] if np.isfinite(self.x_cell[c]).all() else np.zeros(G, np.float32)
            trt = ctl
        n_c = int(self.cell_ctl_count[c])
        if ok and n_c > 1:                       # leave-one-out, so a sparse cell's aggregate is not
            cell = ((self.cell_acc[c] - ctl) / (n_c - 1)).astype(np.float32)   # mostly this row's control
        elif np.isfinite(self.x_cell[c]).all():
            cell = self.x_cell[c]
        else:
            cell = ctl

        y5 = np.asarray(self.Y[self.y_row[i]], np.float32)
        return {
            'x_ctl': ctl, 'x_cell': cell,
            'y_abs': trt, 'y_delta': (trt - ctl).astype(np.float32), 'y_l5': y5,
            'm_l3': np.float32(ok), 'm_l5': np.float32(np.isfinite(y5).all()),
            'E': self.E[c], 'r': self.r_cell[c], 'cell_ctx': self.cell_ctx[c],
            'atoms': np.asarray(self.atom_reprs[a0:a1], np.float32),
            'u_feats': self.u_feats[d],
            'dose': self.dose_n[i], 'time': self.time_n[i], 'w': self.weight[i],
        }


def collate_v9(samples, max_atoms=96, fixed_pad=False):
    import torch
    B = len(samples)
    M = max_atoms if fixed_pad else min(max_atoms, max(1, max(s['atoms'].shape[0] for s in samples)))
    atoms = np.zeros((B, M, 512), np.float32)
    amask = np.zeros((B, M), bool)
    for b, s in enumerate(samples):
        n = min(M, s['atoms'].shape[0])
        atoms[b, :n] = s['atoms'][:n]
        amask[b, :n] = True
    t = lambda key: torch.from_numpy(np.stack([s[key] for s in samples]).astype(np.float32))
    out = {k: t(k) for k in ['x_ctl', 'x_cell', 'y_abs', 'y_delta', 'y_l5', 'E', 'r', 'cell_ctx',
                             'u_feats', 'dose', 'time', 'w']}
    out['m_l3'] = t('m_l3').bool()
    out['m_l5'] = t('m_l5').bool()
    out['atoms'] = torch.from_numpy(atoms)
    out['atom_mask'] = torch.from_numpy(amask)
    return out


def check_inputs_v9(ds, sp, dcfg):
    """Refuse to train on a silently degraded setup. Same contract as train_v7_gpu.check_inputs, plus the
    Level-3 substrate: data.py falls back to neutral defaults when a file is missing, and a rebuilt Kaggle
    dataset once dropped the compound holdout AND reliability weighting without any log line saying so."""
    problems = []
    for k in ('test_colddrug', 'test_coldboth'):
        if len(sp[k]) == 0:
            problems.append(f'{k} EMPTY -> scaffold_split.json missing; no compound holdout (leakage)')
    if float(ds.strength.min()) == 1.0 and float(ds.strength.max()) == 1.0:
        problems.append('strength identically 1.0 -> sig_strength.npy missing; reliability weighting OFF '
                        'and the reproducible-stratum filter passes everything')
    frac = float(ds.has_l3.mean())
    if frac < 0.90:
        problems.append(f'Level-3 coverage {100 * frac:.1f}% < 90% -> the substrate gate is not met; '
                        f'expected 99.65% from outputs/level3_sig')
    if not np.isfinite(np.asarray(ds.Xctl[:64])).all():
        problems.append('X_ctl has non-finite values in its first rows -> wrong array or partial write')
    lo, hi = float(np.min(ds.Xctl[:2000])), float(np.max(ds.Xctl[:2000]))
    if not (-0.1 <= lo and hi <= 15.5):
        problems.append(f'X_ctl range [{lo:.2f}, {hi:.2f}] is not Level-3 log expression')
    n_cell_missing = int((ds.cell_ctl_count == 0).sum())
    if n_cell_missing > 5:
        problems.append(f'{n_cell_missing} cells have no matched control at all -> x_cell is NaN for them')
    if problems:
        raise SystemExit('FATAL: inputs missing or degraded:\n  - ' + '\n  - '.join(problems)
                         + '\nRefusing to train.')
    print(f'input check OK: L3 coverage {100 * frac:.2f}%, X_ctl in [{lo:.2f}, {hi:.2f}], '
          f'strength {ds.strength.min():.2f}-{ds.strength.max():.2f}, compound holdout present',
          flush=True)
