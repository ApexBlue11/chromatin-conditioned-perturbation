# -*- coding: utf-8 -*-
"""
THE DIRECT HEAD-TO-HEAD: our model and XPert's, scored on IDENTICAL ROWS with IDENTICAL TARGETS.

Every comparison in this project so far has been indirect -- our number on our rows against their number on
their rows, differing in drug class, split regime, target level and stratum at once. §39 narrowed that to a
matched *construction* (warm regime, epi-drugs) but still on different rows.

Their HDACi release makes the direct version possible. `reproducing/fig4/` ships
`l1000_mdmt_HDACi.h5ad` (3,439 conditions with `X`, `obsm['X_ctl']` and full metadata) alongside their
`y_pred.npy` for exactly those rows. So: feed OUR model their control, score BOTH predictions against THEIR
target, on the SAME rows. The only difference left is the model and what each was trained on.

Their published numbers are reproduced here first as a check on the artefact: Pearson 0.9804,
Pearson_deg 0.8440, copy-the-control 0.9200 -- all three match §23 exactly.

WHAT IS STILL NOT MATCHED, stated rather than buried:
  * 16 of 3,439 rows use a compound we cannot featurise; 1,159 use a cell line we have no chromatin or
    lineage for (those rows get UNKNOWN lineage and zero chromatin reliability, which v9 handles but which
    is less input than their model had).
  * 15.6 % of their rows pool MULTIPLE doses into one condition (median 1, max 8). Ours never do.
  * Their model was trained on the L1000 mdmt corpus these rows come from; ours was trained on our own
    Level-3 substrate. Neither was trained on the other's preprocessing, and 46.6 % of their
    (cell, compound) pairs also appear somewhere in our data -- so the comparison is reported SPLIT BY
    whether the pair was in our training set, because for pairs we trained on our number is optimistic.

    python model/v9/head_to_head_hdaci.py --ckpt ckpt_v9_fold0_seed0.pt
"""
import os, sys, csv, json, argparse

import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9
from data_v9 import LincsV9Dataset, build_splits

ROOT = r'C:\Projects\LINCS'
FIG4 = os.path.join(ROOT, 'external', 'xpert', 'code', 'XPert', 'reproducing', 'fig4')


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


def boot(r, n=4000, seed=0):
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(seed)
    bs = np.array([np.median(rng.choice(r, len(r), replace=True)) for _ in range(n)])
    return float(np.median(r)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(r)


def num_multi(x):
    parts = [p for p in str(x).replace(',', ';').split(';') if p.strip()]
    v = []
    for p in parts:
        try:
            v.append(float(p))
        except ValueError:
            pass
    return float(np.mean(v)) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--require_known_cell', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    f = h5py.File(os.path.join(FIG4, 'l1000_mdmt_HDACi.h5ad'), 'r')
    Y = np.asarray(f['X'][:], np.float32)
    C = np.asarray(f['obsm/X_ctl'][:], np.float32)

    def col(n):
        g = f['obs'][n]
        if isinstance(g, h5py.Group):
            c = dec(g['categories'][:]); k = g['codes'][:]
            return np.array([c[i] if i >= 0 else 'NA' for i in k])
        return np.array(dec(g[:]))

    pert, cell = col('pert_id'), col('cell_iname')
    dose = np.array([num_multi(x) for x in col('pert_dose')], np.float32)
    time = np.array([num_multi(x) for x in col('pert_time')], np.float32)
    f.close()
    XP = np.load(os.path.join(FIG4, 'hdaci_predict', 'y_pred.npy'))

    # ---- reproduce their published numbers as a check on the artefact ----
    print('XPert on all %d of their HDACi rows (should be 0.9804 / 0.8440 / 0.9200):' % len(Y))
    print('  Pearson %.4f | Pearson_deg %.4f | copy-the-control %.4f'
          % (pearson_rows(Y, XP).mean(), pearson_rows(Y - C, XP - C).mean(), pearson_rows(Y, C).mean()))

    # ---- our model, on their rows ----
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = V9Config(**{k: v for k, v in ck['cfg'].items() if k in V9Config.__dataclass_fields__})
    dc = V9DataConfig()
    dc.cache_in_ram = False
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_pathway_path)); ppi = np.load(R(dc.ppi_v9_path)); gv = np.load(R(dc.gene_vec_path))
    model = LincsV9(cfg, M, ppi, gv)
    model.load_state_dict(ck['model'])
    model = model.to(dev).eval()

    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)

    di = json.load(open(R(dc.drug_index_path)))
    cidx = json.load(open(R(dc.xbase_path).replace('X_base_lincs.npy', 'lincs_cell_index.json')))
    cidx = cidx.get('cell_id_to_row', cidx)

    keep = np.array([p in di for p in pert])
    if a.require_known_cell:
        keep &= np.array([c in cidx for c in cell])
    idx = np.flatnonzero(keep)
    print('\nrows we can run: %d/%d (%d compounds we cannot featurise; %d rows have a cell we do not know)'
          % (len(idx), len(Y), int((~np.isin(pert, list(di.keys()))).sum()),
             int((~np.isin(cell, list(cidx.keys()))).sum())))

    # dose/time MUST use the same standardisation the model trained under. data.py derives it from the
    # whole signature table, so it is recomputed here from that same table with that same parser rather
    # than re-standardised on their rows -- which would hand the model a dose axis it never saw.
    from data import _dose_um, _num
    ld_all, tt_all = [], []
    with open(R(dc.sig_path), encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='	'):
            ld_all.append(_dose_um(r['dose'])); tt_all.append(_num(r['time']))
    ld_all = np.log10(np.clip(np.array(ld_all, np.float64), 1e-4, None))
    ld_all = np.nan_to_num(ld_all, nan=np.nanmean(ld_all))
    tt_all = np.nan_to_num(np.array(tt_all, np.float64), nan=np.nanmean(tt_all))
    d_mu, d_sd = ld_all.mean(), ld_all.std() + 1e-6
    t_mu, t_sd = tt_all.mean(), tt_all.std() + 1e-6
    print('  training dose transform: log10 dose mu %.4f sd %.4f | time mu %.2f sd %.2f'
          % (d_mu, d_sd, t_mu, t_sd))
    ld = np.log10(np.clip(dose[idx], 1e-4, None))
    dose_n = ((ld - d_mu) / d_sd).astype(np.float32)
    time_n = ((time[idx] - t_mu) / t_sd).astype(np.float32)

    E = np.load(R(dc.e_path)).astype(np.float32)
    Em = np.load(R(dc.e_mask_path))
    for k in range(E.shape[2]):
        has = Em[:, :, k].any(1)
        for c in np.where(has)[0]:
            v = E[c, :, k]
            E[c, :, k] = (v - v.mean()) / (v.std() + 1e-6)
    lin = np.load(R(dc.lineage_path)).astype(np.float32)

    fpm = np.load(R(dc.fp_path)).astype(np.float32)
    desc = np.load(R(dc.desc_path)).astype(np.float32)
    desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)
    ucls = np.load(R(dc.unimol_cls_path)).astype(np.float32)
    atom_reprs = np.load(R(dc.atom_reprs_path), mmap_mode='r')
    atom_off = np.load(R(dc.atom_offsets_path))

    drow = np.array([di[p] for p in pert[idx]])
    # per-cell aggregate control, from THEIR rows for that cell
    cmean = {}
    for c in np.unique(cell[idx]):
        cmean[c] = C[idx][cell[idx] == c].mean(0)

    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), a.batch):
            sl = slice(s, min(s + a.batch, len(idx)))
            b_idx = idx[sl]
            d = drow[sl]
            n_at = min(cfg.max_atoms, max(1, int(max(atom_off[k + 1] - atom_off[k] for k in d))))
            atoms = np.zeros((len(d), n_at, 512), np.float32)
            amask = np.zeros((len(d), n_at), bool)
            for j, k in enumerate(d):
                a0, a1 = int(atom_off[k]), int(atom_off[k + 1])
                m = min(n_at, a1 - a0)
                atoms[j, :m] = np.asarray(atom_reprs[a0:a0 + m], np.float32)
                amask[j, :m] = True
            Eb = np.zeros((len(d), 978, E.shape[2]), np.float32)
            rb = np.zeros((len(d), 978), np.float32)
            ctx = np.zeros((len(d), lin.shape[1]), np.float32); ctx[:, 0] = 1.0
            for j, c in enumerate(cell[b_idx]):
                q = cidx.get(str(c))
                if q is not None:
                    Eb[j] = E[q]; rb[j] = Em[q].any(-1).astype(np.float32); ctx[j] = lin[q]
            t = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(dev)
            batch = {'x_ctl': t(C[b_idx]), 'x_cell': t(np.stack([cmean[c] for c in cell[b_idx]])),
                     'E': t(Eb), 'r': t(rb), 'cell_ctx': t(ctx),
                     'atoms': t(atoms), 'atom_mask': torch.as_tensor(amask).to(dev),
                     'u_feats': t(np.concatenate([ucls[d], desc[d], fpm[d]], 1)),
                     'dose': t(dose_n[sl]), 'time': t(time_n[sl])}
            preds.append(model(batch)['abs'].float().cpu().numpy())
    ours = np.concatenate(preds)

    # ---- was the (cell, compound) pair in OUR training set? ----
    tr = sp['train']
    row2cell = {v: k for k, v in cidx.items()}
    inv_di = {v: k for k, v in di.items()}
    tr_pairs = {(row2cell.get(int(c), ''), inv_di.get(int(d), ''))
                for c, d in zip(ds.cell_row[tr], ds.drug_row[tr])}
    seen = np.array([(c, p) in tr_pairs for c, p in zip(cell[idx], pert[idx])])
    print('  of those rows, (cell, compound) pair was in OUR training set: %d (%.1f%%)'
          % (seen.sum(), 100 * seen.mean()))

    yt, ct, xp = Y[idx], C[idx], XP[idx]
    out = {}
    print('\n' + '=' * 104)
    print('HEAD TO HEAD ON IDENTICAL ROWS (their targets, their controls; median row Pearson, 95% CI)')
    print('=' * 104)
    print('  %-34s %6s  %-26s %-26s %s' % ('subset', 'n', 'XPert', 'v9 (ours)', 'copy-the-control'))
    for lab, m in [('all runnable rows', np.ones(len(idx), bool)),
                   ('pair NOT in our training set', ~seen),
                   ('pair in our training set', seen)]:
        if m.sum() < 30:
            continue
        for conv, ya, yb, yc in [('delta', yt[m] - ct[m], xp[m] - ct[m], ours[m] - ct[m]),
                                 ('abs', yt[m], xp[m], ours[m])]:
            x_med, x_lo, x_hi, n = boot(pearson_rows(ya, yb))
            o_med, o_lo, o_hi, _ = boot(pearson_rows(ya, yc))
            c_med = float(np.nanmedian(pearson_rows(yt[m], ct[m]))) if conv == 'abs' else 0.0
            print('  %-34s %6d  %.4f [%.4f,%.4f]  %.4f [%.4f,%.4f]  %.4f'
                  % (f'{lab} / {conv}', n, x_med, x_lo, x_hi, o_med, o_lo, o_hi, c_med))
            out[f'{lab}/{conv}'] = {'n': n, 'xpert': [x_med, x_lo, x_hi], 'v9': [o_med, o_lo, o_hi],
                                    'copy_ctl': c_med}
    dst = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'model', 'results',
                       'v9_head_to_head_hdaci.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
