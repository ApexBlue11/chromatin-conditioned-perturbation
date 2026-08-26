# -*- coding: utf-8 -*-
"""
THE SOTA ARM THE GATE ADMITS: v9 trained on XPert's own data, their split, their metric.

V9_HANDOFF §D: "any SOTA comparison | same data level + convention + split, else report non-comparability."
Nothing else in this project satisfies that. Their reported numbers cannot be set against ours directly --
§28 shows their splits leave 100 % of test cells and 89.4 % of test (cell, compound) pairs in the training
set, so the tasks differ by more than any model does. Running OUR model on THEIR benchmark removes every
difference except the model.

They report two numbers, and both are reproduced here:
    Pearson      on absolute Level-3 expression      (their `metrics['Pearson']`,     reported 0.9804)
    Pearson_deg  on the delta                        (their `metrics['Pearson_deg']`, reported 0.8440)
plus the two nulls without which neither is interpretable: copy-the-control (0.9200 on their own released
predictions) and the drug mean.

Honest limits, stated rather than buried:
  * 5.4 % of the rows their splits touch use compounds we have no UniMol/ECFP4 features for and are dropped;
  * 143 of their 217 cell lines have no chromatin or lineage in our data, so those rows carry the UNKNOWN
    lineage and zero chromatin reliability -- v9 handles that by construction, but it is less input than
    their model gets;
  * their numbers come from their trained weights on their own preprocessing; ours come from this code.

    python model/v9/xpert_arm.py --split split_lung_1 --seeds 3
"""
import os, sys, json, time, math, argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9, v9_loss

BUNDLE = 'xpert_splits.npz'


def find(name, roots):
    import glob
    for r in roots:
        hit = glob.glob(os.path.join(r, '**', name), recursive=True)
        if hit:
            return hit[0]
    raise SystemExit(f'FATAL: {name} not found under {roots}')


def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


class XPertData:
    """Their rows, presented in the v9 batch format."""

    def __init__(self, npz, roots, split):
        z = np.load(npz, allow_pickle=True)
        lab = z[f'split_{split}']
        self.tr = np.flatnonzero(lab == 'train')
        self.te = np.flatnonzero(lab == 'test')
        if len(self.te) == 0:
            raise SystemExit(f'FATAL: split {split} has no test rows in the bundle')
        keep = np.concatenate([self.tr, self.te])
        self.X = np.asarray(z['X'][keep], np.float32)
        self.C = np.asarray(z['X_ctl'][keep], np.float32)
        self.pert = z['meta_pert_id'][keep]
        self.cell = z['meta_cell'][keep]
        self.dose = z['meta_dose'][keep].astype(np.float32)
        self.time = z['meta_time'][keep].astype(np.float32)
        n_tr = len(self.tr)
        self.tr = np.arange(n_tr)
        self.te = np.arange(n_tr, len(keep))

        # --- our drug features, keyed by their pert_id ---
        di = json.load(open(find('drug_feature_index.json', roots)))
        rows = np.array([di[p] for p in self.pert])
        desc = np.load(find('drug_descriptors.npy', roots)).astype(np.float32)
        desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)
        fp = np.load(find('drug_fingerprints.npy', roots)).astype(np.float32)
        ucls = np.load(find('drug_unimol.npy', roots)).astype(np.float32)
        self.u = np.concatenate([ucls[rows], desc[rows], fp[rows]], 1)
        self.atom_reprs = np.load(find('drug_atom_reprs.npy', roots), mmap_mode='r')
        self.atom_off = np.load(find('drug_atom_offsets.npy', roots))
        self.drug_row = rows

        # --- chromatin / lineage where we have the cell line, neutral where we do not ---
        cidx = json.load(open(find('lincs_cell_index.json', roots)))
        cidx = cidx.get('cell_id_to_row', cidx)
        E = np.load(find('E_final.npy', roots)).astype(np.float32)
        Em = np.load(find('E_final_mask.npy', roots))
        for k in range(E.shape[2]):
            has = Em[:, :, k].any(1)
            for c in np.where(has)[0]:
                v = E[c, :, k]
                E[c, :, k] = (v - v.mean()) / (v.std() + 1e-6)
        lin = np.load(find('cell_lineage.npy', roots)).astype(np.float32)
        G = self.X.shape[1]
        self.E = np.zeros((len(self.X), G, E.shape[2]), np.float32)
        self.r = np.zeros((len(self.X), G), np.float32)
        self.ctx = np.zeros((len(self.X), lin.shape[1]), np.float32)
        self.ctx[:, 0] = 1.0
        known = 0
        for i, c in enumerate(self.cell):
            j = cidx.get(str(c))
            if j is None:
                continue
            known += 1
            self.E[i] = E[j]
            self.r[i] = Em[j].any(-1).astype(np.float32)
            self.ctx[i] = lin[j]
        self.known_cell_frac = known / len(self.cell)

        ld = np.log10(np.clip(self.dose, 1e-4, None))
        self.dose_n = ((ld - ld[self.tr].mean()) / (ld[self.tr].std() + 1e-6)).astype(np.float32)
        self.time_n = ((self.time - self.time[self.tr].mean()) /
                       (self.time[self.tr].std() + 1e-6)).astype(np.float32)

        # per-cell aggregate control, from TRAINING rows only
        self.cell_mean = {}
        for c in np.unique(self.cell):
            m = (self.cell == c)
            m_tr = np.zeros(len(self.cell), bool)
            m_tr[self.tr] = True
            sel = m & m_tr
            self.cell_mean[c] = self.C[sel].mean(0) if sel.any() else self.C[m].mean(0)
        self.x_cell = np.stack([self.cell_mean[c] for c in self.cell])

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
        return {'x_ctl': t(self.C[idx]), 'x_cell': t(self.x_cell[idx]),
                'y_abs': t(self.X[idx]), 'y_delta': t(self.X[idx] - self.C[idx]),
                'y_l5': t(np.zeros((len(idx), self.X.shape[1]), np.float32)),
                'm_l3': torch.ones(len(idx), dtype=torch.bool, device=device),
                'm_l5': torch.zeros(len(idx), dtype=torch.bool, device=device),
                'E': t(self.E[idx]), 'r': t(self.r[idx]), 'cell_ctx': t(self.ctx[idx]),
                'atoms': t(atoms), 'atom_mask': torch.as_tensor(amask).to(device),
                'u_feats': t(self.u[idx]), 'dose': t(self.dose_n[idx]), 'time': t(self.time_n[idx])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='split_lung_1')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--expr_encoder', default='binned')
    a = ap.parse_args()

    roots = ['/kaggle/input', os.path.join(r'C:\Projects\LINCS'), os.path.join(r'C:\Projects\LINCS',
                                                                              'external')]
    roots = [r for r in roots if os.path.isdir(r)]
    npz = find(BUNDLE, roots)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda' and any('P100' in torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())):
        raise SystemExit('FATAL: P100 assigned; Kaggle torch has no sm_60 kernels.')
    D = XPertData(npz, roots, a.split)
    print(f'{a.split}: train {len(D.tr)} test {len(D.te)} | rows with a cell line we know: '
          f'{100 * D.known_cell_frac:.1f}% | device {dev}', flush=True)

    M = np.load(find('M_pathway_v9.npy', roots))
    ppi = np.load(find('STRING_adj_978_v9.npy', roots))
    gv = np.load(find('gene_vectors_978.npy', roots))

    # nulls first: an absolute number is uninterpretable without them
    Xte, Cte = D.X[D.te], D.C[D.te]
    nulls = {'copy_ctl_abs': round(float(np.nanmedian(pearson_rows(Cte, Xte))), 4),
             'zero_delta': 0.0}
    dtr = D.X[D.tr] - D.C[D.tr]
    gm = dtr.mean(0)
    per = {}
    for p in np.unique(D.pert[D.tr]):
        per[p] = dtr[D.pert[D.tr] == p].mean(0)
    pred = np.stack([per.get(p, gm) for p in D.pert[D.te]])
    nulls['mean_drug_delta'] = round(float(np.nanmedian(pearson_rows(pred, Xte - Cte))), 4)
    print(f'nulls on THEIR test rows: {nulls}', flush=True)

    runs = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = V9Config()
        cfg.d_model, cfg.d_ff, cfg.expr_encoder = a.d_model, 4 * a.d_model, a.expr_encoder
        cfg.n_pathways = M.shape[0]
        cfg.predict_l5 = False
        core = LincsV9(cfg, M, ppi, gv)
        if cfg.expr_encoder == 'binned':
            core.fit_bins(D.C[D.tr])          # THEIR training rows only
            if not core.bins_fitted:
                raise SystemExit('FATAL: quantiser did not fit.')
        core = core.to(dev)
        model = torch.nn.DataParallel(core) if torch.cuda.device_count() > 1 else core
        opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.95))
        steps = a.epochs * max(1, len(D.tr) // a.batch)
        warm = max(1, int(0.03 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm else max(0.0, 1 - math.sqrt(
                max(0, s - 0.8 * steps) / max(1, 0.2 * steps))))
        Mn = torch.as_tensor(M, dtype=torch.float32)
        Mn = (Mn / Mn.sum(1, keepdim=True).clamp(min=1)).to(dev)
        scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda')
        t0 = time.time()
        model.train()
        for ep in range(a.epochs):
            order = np.random.permutation(D.tr)
            for it in range(len(order) // a.batch):
                idx = order[it * a.batch:(it + 1) * a.batch]
                b = D.batch(idx, dev)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=dev == 'cuda'):
                    out, aux = model(b, return_aux=True)
                    loss, _ = v9_loss(out, b, cfg, Mn, aux)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                if it % 200 == 0:
                    print(f'  seed{seed} e{ep} it{it}/{len(order) // a.batch} loss {float(loss):.4f} '
                          f'{time.time() - t0:.0f}s', flush=True)
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for s in range(0, len(D.te), 64):
                idx = D.te[s:s + 64]
                b = D.batch(idx, dev)
                o = model(b)
                P.append(o['abs'].float().cpu().numpy())
                T.append(o['delta'].float().cpu().numpy())
        pa, pd = np.concatenate(P), np.concatenate(T)
        rec = {'seed': seed,
               'Pearson': round(float(np.nanmedian(pearson_rows(pa, Xte))), 4),
               'Pearson_deg': round(float(np.nanmedian(pearson_rows(pd, Xte - Cte))), 4),
               'seconds': round(time.time() - t0, 1)}
        runs.append(rec)
        print(f'  [seed {seed}] Pearson {rec["Pearson"]:.4f}  Pearson_deg {rec["Pearson_deg"]:.4f}',
              flush=True)

    def mmr(k):
        v = [r[k] for r in runs]
        return f'{np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]'

    print('\n' + '=' * 92)
    print(f'v9 ON XPERT\'S OWN BENCHMARK ({a.split}), mean [min, max] over {a.seeds} seeds')
    print('=' * 92)
    print(f'  Pearson      (absolute) : {mmr("Pearson")}   | their reported 0.9804 | '
          f'copy-the-control {nulls["copy_ctl_abs"]:.4f}')
    print(f'  Pearson_deg  (delta)    : {mmr("Pearson_deg")}   | their reported 0.8440 | '
          f'mean-drug {nulls["mean_drug_delta"]:.4f}')
    print(f'\n  {100 * (1 - D.known_cell_frac):.1f}% of rows use a cell line we have no chromatin or lineage '
          f'for; 5.4% of their split rows were dropped for lack of drug features.')
    WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(
        os.path.dirname(os.path.dirname(HERE)), 'model', 'results')
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, f'v9_xpert_arm_{a.split}.json')
    json.dump({'split': a.split, 'runs': runs, 'nulls': nulls,
               'their_reported': {'Pearson': 0.9804, 'Pearson_deg': 0.8440},
               'known_cell_frac': round(D.known_cell_frac, 4),
               'n_train': int(len(D.tr)), 'n_test': int(len(D.te))}, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
