# -*- coding: utf-8 -*-
"""
Bundle XPert's MAIN mdmt benchmark (68,830 conditions) into the compact form our GPU kernels can train on.

`model/v9/xpert_extract.py` bundled the fifteen TISSUE splits that live in their 336,852-row h5ad. Those
are not the benchmark their baseline table is built on. The benchmark is `l1000_mdmt_68830_subset.h5ad`,
whose obs carries the three split families the field quotes:

    split_1..5              warm      55,064 train / 13,766 test, an exact 80/20 five-fold CV
    split_cold_cell_1..5    unseen cell lines
    split_cold_drug_1..5    unseen compounds

and it is the corpus their released `l1000_mdmt_warm_split.pth` was trained on. Bundling it lets the SAME
rows be scored by their checkpoint (model/v9/xpert_native_eval.py) and by ours, which is the only form of
comparison this project has not already got wrong once.

`row_index` is carried through so a prediction from either model can be aligned back to their h5ad row --
without it, two 13,766-row arrays in different orders would silently compare row i to row j.

    python model/v9/xpert_mdmt_extract.py
"""
import os, sys, json, csv, argparse

import numpy as np

ROOT = r'C:\Projects\LINCS'
H5 = os.path.join(ROOT, 'external', 'xpert', 'code', 'XPert', 'processed_data',
                  'l1000_mdmt_68830_subset.h5ad')
OUT = os.path.join(ROOT, 'external', 'xpert_split_bundle')
GENE_INFO = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_gene_info.txt',
                         'GSE92742_Broad_LINCS_gene_info.txt')
GENE_ORDER = os.path.join(ROOT, 'Data Info', 'pathway_landmark_genes.txt')

FAMILIES = ['split_%d', 'split_cold_cell_%d', 'split_cold_drug_%d']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', default=H5)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    import anndata as ad
    os.makedirs(a.out, exist_ok=True)

    print('reading %s' % a.h5ad, flush=True)
    d = ad.read_h5ad(a.h5ad)
    n, G = d.shape
    print('  %d conditions x %d genes' % (n, G), flush=True)

    # ---- gene axis: assert, do not trust ----
    order = [l.strip() for l in open(GENE_ORDER, encoding='utf-8') if l.strip()]
    sym2ent = {r['pr_gene_symbol']: r['pr_gene_id']
               for r in csv.DictReader(open(GENE_INFO, encoding='utf-8'), delimiter='\t')
               if r['pr_is_lm'] == '1'}
    ours = [sym2ent[g] for g in order]
    var = d.var
    col = 'gene_id' if 'gene_id' in var.columns else var.columns[0]
    theirs = [str(x) for x in var[col]]
    if theirs == ours:
        gidx = np.arange(G)
        print('  gene axis identical to ours entry by entry -- no remapping', flush=True)
    else:
        pos = {g: i for i, g in enumerate(theirs)}
        missing = [g for g in ours if g not in pos]
        if missing:
            raise SystemExit('FATAL: %d landmarks absent from their var (%s...)' % (len(missing), missing[:5]))
        gidx = np.array([pos[g] for g in ours])
        print('  gene axis differs in ORDER; remapping %d columns' % len(gidx), flush=True)

    X = np.asarray(d.X, np.float32)[:, gidx]
    C = np.asarray(d.obsm['X_ctl'], np.float32)[:, gidx]
    if not (np.isfinite(X).all() and np.isfinite(C).all()):
        raise SystemExit('FATAL: non-finite values in their expression or control matrix')

    obs = d.obs

    # ---- DOSE. 18.9 % of their rows pool 2-8 distinct doses into one condition ('0.12;0.04;0.01'), so
    # pert_dose is not a number. Their model never sees it: MyDataset reads `pert_dose_idx`, which is
    # single-valued 0..9, because obs already carries that column and their fallback binning is skipped.
    # To keep the comparison about the MODEL, v9 must see the same dose resolution and no more -- so the
    # continuous dose we hand it is a per-bin representative, taken as the median true dose of the
    # SINGLE-dose rows in that bin. That is a bijection from the bin index, carrying no extra information.
    dose_raw = np.asarray(obs['pert_dose']).astype(str)
    pooled = np.char.find(dose_raw, ';') >= 0
    dose_idx = np.asarray(obs['pert_dose_idx']).astype(np.int64)
    rep = {}
    for b in np.unique(dose_idx):
        clean = (dose_idx == b) & ~pooled
        vals = dose_raw[clean].astype(np.float64) if clean.any() else None
        if vals is None or not len(vals):
            vals = np.array([float(s.split(';')[0]) for s in dose_raw[dose_idx == b]])
        rep[int(b)] = float(np.median(vals))
    print('  dose bin -> representative uM: %s' % {k: round(v, 4) for k, v in sorted(rep.items())},
          flush=True)
    dose = np.array([rep[int(b)] for b in dose_idx], np.float32)

    out = dict(X=X, X_ctl=C, row_index=np.arange(n, dtype=np.int64),
               meta_pert_id=np.asarray(obs['pert_id']).astype(str),
               meta_cell=np.asarray(obs['cell_iname']).astype(str),
               meta_dose=dose,
               meta_dose_idx=dose_idx,
               meta_dose_pooled=pooled,
               meta_time=np.asarray(obs['pert_time']).astype(np.float32),
               meta_pert_idx=np.asarray(obs['pert_idx']).astype(np.int64),
               meta_cell_idx=np.asarray(obs['cell_idx']).astype(np.int64))

    kept = []
    for fam in FAMILIES:
        for k in range(1, a.folds + 1):
            c = fam % k
            if c not in obs.columns:
                continue
            lab = np.asarray(obs[c]).astype(str)
            out['split_' + c] = lab
            kept.append((c, int((lab == 'train').sum()), int((lab == 'test').sum())))
    if not kept:
        raise SystemExit('FATAL: none of the expected split columns are present')
    for c, ntr, nte in kept:
        print('  %-22s train %6d  test %6d' % (c, ntr, nte), flush=True)

    dst = os.path.join(a.out, 'xpert_mdmt_splits.npz')
    np.savez(dst, **out)
    prov = {'source': a.h5ad, 'n_rows': int(n), 'n_genes': int(G),
            'gene_axis_identical': bool(np.array_equal(gidx, np.arange(G))),
            'splits': [c for c, _, _ in kept],
            'mean_abs_delta': round(float(np.abs(X - C).mean()), 4),
            'n_cells': int(len(set(out['meta_cell'].tolist()))),
            'n_compounds': int(len(set(out['meta_pert_id'].tolist()))),
            'rows_with_pooled_dose': int(out['meta_dose_pooled'].sum()),
            'frac_pooled_dose': round(float(out['meta_dose_pooled'].mean()), 4)}
    json.dump(prov, open(os.path.join(a.out, 'xpert_mdmt_provenance.json'), 'w'), indent=1)
    print('\nwrote %s (%.0f MB)' % (dst, os.path.getsize(dst) / 1e6))
    print(json.dumps(prov, indent=1))


if __name__ == '__main__':
    main()
