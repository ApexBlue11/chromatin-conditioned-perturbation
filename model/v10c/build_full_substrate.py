"""v10-C L1: the full-release training substrate (RESULTS 100.2). Outcome-free: rows are chosen by cell and compound
identity only; no expression value decides membership (values are read only to verify the benchmark-row match).

  python model/v10c/build_full_substrate.py --out external/v10c_full/v10c_full_substrate.npz
"""
import argparse, csv, hashlib, json, os, sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5AD = os.path.join(REPO, 'external', 'xpert', 'l1000_mdmt_full_336852.h5ad')
BUNDLE = os.path.join(REPO, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz')
DRUG_INDEX = os.path.join(REPO, 'drug', 'outputs', 'drug_feature_index.json')
DRUG_LIST = os.path.join(REPO, 'drug', 'outputs', 'drug_list.tsv')
DRUG_FP = os.path.join(REPO, 'drug', 'outputs', 'drug_fingerprints.npy')
PERT_INFO = [os.path.join(REPO, 'Data Info', 'GSE92742_Broad_LINCS_pert_info.txt', 'GSE92742_Broad_LINCS_pert_info.txt'),
             os.path.join(REPO, 'Data Info', 'GSE70138_Broad_LINCS_pert_info_2017-03-06.txt',
                          'GSE70138_Broad_LINCS_pert_info.txt')]
TANI_DUP = 0.999  # RESULTS 96.6: the same molecule is an InChIKey first-block match or ECFP4 Tanimoto >= 0.999
# Review 087 C1: one cell line under two LINCS names. NCIH1975 (unseen-cell TRAINING in XPert's split) is test cell H1975:
# the same 54 (compound, dose, time) and mean-X r 0.9997 (the next test / training pair in the 217 cells is 0.89).
CELL_ALIASES = {'NCIH1975': 'H1975'}


def same_cell(c):
    return CELL_ALIASES.get(str(c), str(c))
SPLITS = ('cold_cell_1', 'cold_drug_1')
CHUNK = 20_000


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def obs_col(obs, k):
    g = obs[k]
    if isinstance(g, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g['categories'][()]])
        return cats[g['codes'][()]]
    v = g[()]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v]) if v.dtype.kind in 'OS' else v


def plate_codes(obs, rows):
    """det_plate (the replicate plate set; up to ~4,000 characters) as a compact id per row, plus sha1[:16] of each used
    set's name, without materialising the long strings per row."""
    g = obs['det_plate']
    if isinstance(g, h5py.Group):
        codes = g['codes'][()][rows]
        cats = g['categories']
        used, plate_id = np.unique(codes, return_inverse=True)
        names = [cats[int(c)] for c in used]
    else:
        v = obs_col(obs, 'det_plate')[rows]
        names, plate_id = np.unique(v, return_inverse=True)
    key = np.array([hashlib.sha1((n.decode() if isinstance(n, bytes) else str(n)).encode()).hexdigest()[:16] for n in names])
    return plate_id.astype(np.int64), key


def bench_keys(b):
    return np.array([f'{c}:{p}:{int(t)} h:{int(d)}' for c, p, t, d in
                     zip(b['meta_cell'], b['meta_pert_id'], b['meta_time'], b['meta_dose_idx'])])


def structure_groups(drug_list):
    """pert_id -> canonical SMILES (drug_list.tsv), so IDs of one structure can be excluded together."""
    out = {}
    with open(drug_list, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            s = (r.get('canonical_smiles') or '').strip()
            if s:
                out[r['pert_id']] = s
    return out


def inchi_blocks(paths=PERT_INFO):
    """pert_id -> InChIKey first block, from the LINCS pert_info tables (as RESULTS 96.6)."""
    out = {}
    for p in paths:
        with open(p, encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                k = (r.get('inchi_key') or '').strip()
                if k and k not in ('-666', 'nan') and r['pert_id'] not in out:
                    out[r['pert_id']] = k.split('-')[0]
    return out


def same_molecule(test_perts, candidates, smiles, inchi, fp_of, tani=TANI_DUP):
    """The candidates that are a test compound's molecule under another ID: same canonical SMILES, same InChIKey first
    block, or ECFP4 Tanimoto >= tani (RESULTS 96.6). fp_of(pert) -> a 0/1 vector or None."""
    test_smiles = {smiles[p] for p in test_perts if p in smiles}
    test_ik = {inchi[p] for p in test_perts if p in inchi}
    T = [fp_of(p) for p in sorted(test_perts)]
    T = np.array([t for t in T if t is not None], np.float32)
    tcount = T.sum(1) if len(T) else None
    out = set()
    for p in candidates:
        if p in test_perts:
            continue
        if smiles.get(p) in test_smiles or inchi.get(p) in test_ik:
            out.add(p)
            continue
        v = fp_of(p)
        if v is None or tcount is None:
            continue
        v = np.asarray(v, np.float32)
        inter = T @ v
        union = tcount + v.sum() - inter
        if np.any(np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0) >= tani):
            out.add(p)
    return out


def assign_splits(full_cell, full_pert, bench_pos, bench_split, has_feat, smiles, inchi=None, fp_of=None):
    """bench_pos[i] = the full row of benchmark row i. Returns {split: labels over full rows} with labels
    train / test / excluded_test_cell / excluded_test_compound / excluded_no_features.

    Order (review 087 C2): the identity exclusions (a test cell under any of its names; a test compound's molecule under
    any ID) rank ABOVE "keep the benchmark's own training rows", so a benchmark leak is not carried over."""
    n = len(full_cell)
    out = {}
    for name, sp in bench_split.items():
        lab = np.full(n, 'train', dtype=object)
        test_rows = bench_pos[sp == 'test']
        if name.startswith('cold_cell'):
            test_cells = {same_cell(c) for c in full_cell[test_rows]}
            ident = np.array([same_cell(c) in test_cells for c in full_cell])
            reason = 'excluded_test_cell'
        else:
            test_perts = set(full_pert[test_rows])
            same = same_molecule(test_perts, set(full_pert), smiles, inchi or {}, fp_of or (lambda p: None))
            ident = np.isin(full_pert, sorted(test_perts | same))
            reason = 'excluded_test_compound'
        lab[ident] = reason
        lab[~has_feat & (lab == 'train')] = 'excluded_no_features'
        lab[test_rows] = 'test'
        # the benchmark's own training rows are kept, unless an identity rule excludes them (review 087 C2)
        bt = bench_pos[sp == 'train']
        lab[bt[~ident[bt]]] = 'train'
        out[name] = lab.astype(str)
    return out


def build(out_path, h5ad=H5AD, bundle=BUNDLE, drug_index=DRUG_INDEX, drug_list=DRUG_LIST, verify_all=True,
          drug_fp=DRUG_FP, pert_info=PERT_INFO):
    b = np.load(bundle, allow_pickle=True)
    f = h5py.File(h5ad, 'r')
    obs = f['obs']
    key = obs_col(obs, 'cell_drug_dose_time')
    if len(set(key)) != len(key):
        raise SystemExit('FATAL: full-release keys are not unique')
    pos = {k: i for i, k in enumerate(key)}
    bk = bench_keys(b)
    bench_pos = np.array([pos.get(k, -1) for k in bk])
    if (bench_pos < 0).any():
        raise SystemExit(f'FATAL: {(bench_pos < 0).sum()} benchmark rows not in the full release')
    if len(set(bench_pos)) != len(bench_pos):
        raise SystemExit('FATAL: two benchmark rows map to one full row')
    cell, pert = obs_col(obs, 'cell_iname'), obs_col(obs, 'pert_id')
    if not ((cell[bench_pos] == b['meta_cell']).all() and (pert[bench_pos] == b['meta_pert_id']).all()):
        raise SystemExit('FATAL: matched rows disagree on cell or compound')
    feat = json.load(open(drug_index, encoding='utf-8'))
    has_feat = np.array([p in feat for p in pert])
    smiles = structure_groups(drug_list)
    inchi = inchi_blocks(pert_info)
    fps = np.load(drug_fp, mmap_mode='r')
    fp_of = lambda p: (np.asarray(fps[feat[p]]) > 0) if p in feat else None
    labels = assign_splits(cell, pert, bench_pos, {s: b['split_split_' + s] for s in SPLITS}, has_feat, smiles,
                           inchi, fp_of)
    keep = np.zeros(len(key), bool)
    for lab in labels.values():
        keep |= np.isin(lab, ['train', 'test'])
    rows = np.where(keep)[0]
    n = len(rows)
    X = np.empty((n, 978), np.float32)
    ctl_id = np.empty(n, np.int64)
    ctl_rows, ctl_lookup = [], {}
    Xh, Ch = f['X'], f['obsm']['X_ctl']
    for s in range(0, n, CHUNK):
        r = rows[s:s + CHUNK]
        X[s:s + len(r)] = Xh[r[0]:r[-1] + 1][r - r[0]].astype(np.float32)
        C = Ch[r[0]:r[-1] + 1][r - r[0]].astype(np.float32)
        for j, c in enumerate(C):
            h = c.tobytes()
            k = ctl_lookup.get(h)
            if k is None:
                k = ctl_lookup[h] = len(ctl_rows)
                ctl_rows.append(c)
            ctl_id[s + j] = k
        print(f'{s + len(r)} / {n}', flush=True)
    ctl = np.stack(ctl_rows)
    inv = np.full(len(key), -1, np.int64)
    inv[rows] = np.arange(n)
    bsub = inv[bench_pos]
    nb = np.arange(len(bk)) if verify_all else np.arange(0, len(bk), 20)
    if not (np.array_equal(X[bsub[nb]], b['X'][nb]) and np.array_equal(ctl[ctl_id[bsub[nb]]], b['X_ctl'][nb])):
        raise SystemExit('FATAL: benchmark X / X_ctl differ from the full release on matched rows')
    meta = {k: obs_col(obs, k)[rows] for k in ('cell_iname', 'pert_id', 'pert_dose', 'pert_dose_idx', 'pert_time',
                                               'cell_drug_dose_time')}
    plate_id, plate_key = plate_codes(obs, rows)
    br = np.full(n, -1, np.int64)  # the benchmark row each substrate row is, or -1
    br[bsub] = np.arange(len(bk))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, X=X, ctl=ctl, ctl_id=ctl_id, full_row=rows, bench_row=br, plate_id=plate_id, plate_key=plate_key,
             **{'meta_' + k: v.astype(str) for k, v in meta.items()},
             **{'split_' + s: labels[s][rows] for s in SPLITS})
    counts = {s: {k: int(v) for k, v in zip(*np.unique(labels[s], return_counts=True))} for s in SPLITS}
    kept = {s: {'train_cells': int(len(set(meta['cell_iname'][labels[s][rows] == 'train']))),
                'train_compounds': int(len(set(meta['pert_id'][labels[s][rows] == 'train'])))} for s in SPLITS}
    rec = {'rule': 'RESULTS 100.2', 'h5ad_sha1': sha1(h5ad), 'bundle_sha1': sha1(bundle),
           'drug_index_sha1': sha1(drug_index), 'drug_list_sha1': sha1(drug_list), 'drug_fp_sha1': sha1(drug_fp),
           'pert_info_sha1': [sha1(x) for x in pert_info], 'n_rows': n,
           'n_unique_controls': int(len(ctl)), 'label_counts_full_release': counts, 'train_sets': kept,
           'compounds_without_features': int(len(set(pert[~has_feat]))), 'out_sha1': sha1(out_path)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--record', required=True)
    a = ap.parse_args()
    rec = build(a.out)
    with open(a.record, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
