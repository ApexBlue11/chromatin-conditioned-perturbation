# -*- coding: utf-8 -*-
"""
Extract the rows XPert's 15 released splits actually use, into a compact bundle we can train on.

This is the setup for the only SOTA comparison the handoff's gate admits: same data level, same convention,
same split. Their h5ad is 5.1 GB and lives in gitignored `external/`; the three tissues their splits touch
are 120,488 of its 336,852 rows, which fp16 to ~470 MB and can be shipped to a GPU kernel.

Verified before extracting, not assumed:
  * their `var/gene_id` is identical to our canonical landmark order, ENTRY BY ENTRY -- so their columns
    need no remapping. (Their `gene_symbol` differs from ours because ours are 2012 L1000 names; the Entrez
    ids agree exactly.)
  * 93.9 % of their rows use a compound we have UniMol/ECFP4 features for; rows we cannot featurise are
    dropped and the loss is reported rather than hidden.

    python model/v9/xpert_extract.py
"""
import os, sys, json, csv, argparse

import numpy as np
import h5py

ROOT = r'C:\Projects\LINCS'
H5 = os.path.join(ROOT, 'external', 'xpert', 'l1000_mdmt_full_336852.h5ad')
OUT = os.path.join(ROOT, 'external', 'xpert_split_bundle')      # gitignored, like the rest of external/
GENE_INFO = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_gene_info.txt',
                         'GSE92742_Broad_LINCS_gene_info.txt')
GENE_ORDER = os.path.join(ROOT, 'Data Info', 'pathway_landmark_genes.txt')


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', default=H5)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = h5py.File(a.h5ad, 'r')

    # ---- gene axis: assert alignment rather than trusting it ----
    order = [l.strip() for l in open(GENE_ORDER, encoding='utf-8') if l.strip()]
    sym2ent = {r['pr_gene_symbol']: r['pr_gene_id']
               for r in csv.DictReader(open(GENE_INFO, encoding='utf-8'), delimiter='\t')
               if r['pr_is_lm'] == '1'}
    ours = [sym2ent[g] for g in order]
    theirs = dec(f['var/gene_id'][:])
    if theirs != ours:
        pos = {g: i for i, g in enumerate(theirs)}
        missing = [g for g in ours if g not in pos]
        if missing:
            raise SystemExit(f'FATAL: {len(missing)} of our landmarks are absent from their var: {missing[:5]}')
        gidx = np.array([pos[g] for g in ours])
        print(f'gene axis differs in ORDER; remapping {len(gidx)} columns')
    else:
        gidx = np.arange(len(ours))
        print('gene axis is identical to ours entry by entry -- no remapping needed')

    # ---- which rows do the splits touch, and which can we featurise ----
    splits = sorted(k for k in f['obs'].keys() if k.startswith('split_'))
    used = np.zeros(f['X'].shape[0], bool)
    split_codes = {}
    for s in splits:
        cats = dec(f[f'obs/{s}/categories'][:])
        k = f[f'obs/{s}/codes'][:]
        split_codes[s] = dict(categories=cats, codes=k)
        used |= (k >= 0)
    print(f'rows touched by at least one split: {used.sum()} / {len(used)}')

    pcat = dec(f['obs/pert_id/categories'][:])
    pcode = f['obs/pert_id/codes'][:]
    ccat = dec(f['obs/cell_iname/categories'][:])
    ccode = f['obs/cell_iname/codes'][:]
    dcat = dec(f['obs/pert_dose/categories'][:])
    dcode = f['obs/pert_dose/codes'][:]
    tcat = dec(f['obs/pert_time/categories'][:])
    tcode = f['obs/pert_time/codes'][:]

    drug_index = json.load(open(os.path.join(ROOT, 'drug', 'outputs', 'drug_feature_index.json')))
    have_drug = np.array([c in drug_index for c in pcat])
    keep = used & have_drug[pcode]
    lost = int(used.sum() - keep.sum())
    print(f'usable rows: {keep.sum()} (dropped {lost} = {100 * lost / max(used.sum(), 1):.1f}% with no '
          f'drug features of ours)')

    idx = np.flatnonzero(keep)
    X = np.zeros((len(idx), len(gidx)), np.float16)
    C = np.zeros((len(idx), len(gidx)), np.float16)
    for lo in range(0, len(idx), 20000):
        sl = idx[lo:lo + 20000]
        X[lo:lo + len(sl)] = np.asarray(f['X'][sl.min():sl.max() + 1], np.float32)[sl - sl.min()][:, gidx]
        C[lo:lo + len(sl)] = np.asarray(f['obsm/X_ctl'][sl.min():sl.max() + 1],
                                        np.float32)[sl - sl.min()][:, gidx]
        print(f'  {lo + len(sl)}/{len(idx)}', flush=True)

    def num(x):
        """Their pert_dose/pert_time can carry SEMICOLON-SEPARATED values -- a condition aggregates
        replicates whose recorded dose differed slightly ('0.04;0.12;0.09;0.03;0.01'). Averaging the parts
        is the faithful reading; taking the first would silently pick one replicate's dose."""
        parts = [p for p in str(x).replace(',', ';').split(';') if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                pass
        return float(np.mean(vals)) if vals else np.nan

    dvals = np.array([num(dcat[c]) for c in dcode[idx]], np.float32)
    tvals = np.array([num(tcat[c]) for c in tcode[idx]], np.float32)
    n_multi = sum(1 for c in set(dcode[idx].tolist()) if ';' in str(dcat[c]))
    print(f'dose categories that are multi-valued: {n_multi} (averaged); '
          f'unparseable dose rows: {int(np.isnan(dvals).sum())}, time rows: {int(np.isnan(tvals).sum())}')
    meta = dict(
        pert_id=np.array([pcat[c] for c in pcode[idx]]),
        cell=np.array([ccat[c] for c in ccode[idx]]),
        dose=np.nan_to_num(dvals, nan=float(np.nanmedian(dvals))),
        time=np.nan_to_num(tvals, nan=float(np.nanmedian(tvals))),
    )
    sp = {}
    for s in splits:
        cats, k = split_codes[s]['categories'], split_codes[s]['codes'][idx]
        lab = np.full(len(idx), '', dtype=object)
        for i, c in enumerate(cats):
            lab[k == i] = c
        sp[s] = np.array(lab)

    np.savez_compressed(os.path.join(a.out, 'xpert_splits.npz'), X=X, X_ctl=C, row=idx,
                        **{f'meta_{k}': v for k, v in meta.items()},
                        **{f'split_{k}': v for k, v in sp.items()})
    d = np.asarray(X[:5000], np.float32) - np.asarray(C[:5000], np.float32)
    prov = dict(source=a.h5ad, n_rows=int(len(idx)), n_genes=int(len(gidx)),
                dropped_no_drug_features=lost, gene_axis_identical=bool(theirs == ours),
                splits=len(splits), mean_abs_delta=round(float(np.abs(d).mean()), 4),
                ours_for_reference=0.3860)
    json.dump(prov, open(os.path.join(a.out, 'xpert_splits_provenance.json'), 'w'), indent=2)
    f.close()
    sz = os.path.getsize(os.path.join(a.out, 'xpert_splits.npz'))
    print(f'\nwrote {len(idx)} rows x {len(gidx)} genes, {sz / 1e6:.0f} MB -> {a.out}')
    print(f'their mean|delta| {prov["mean_abs_delta"]:.4f} vs ours 0.3860')


if __name__ == '__main__':
    main()
