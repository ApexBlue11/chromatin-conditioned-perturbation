# -*- coding: utf-8 -*-
"""RESULTS 98.20 (review 081 C5): the caption-collision stratum, fixed before any D output exists.

A clean test molecule whose MolT5 caption equals a training compound's caption gets that compound's BioLinkBERT embedding
exactly, so to PertDiT it is a known compound although the molecule is unseen (clean-subset deduplication is by InChIKey /
ECFP4, not by caption). This writes the stratum's molecules and rows; D - v9 and D's own score on it, against the rest, are
REPORTED ONLY, never read (like 98.12 (b)). Reads only: P9's row_index (no predictions), the bundle's split / pert columns,
the units and clean JSONs, the compact compound map and the five caption caches.
"""
import hashlib, io, json, os, pickle, sys

import numpy as np

R = r'C:\Projects\LINCS'
sys.path.insert(0, os.path.join(R, 'model', 'v9'))
import score_p9  # noqa: E402

z = np.load(os.path.join(R, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz'), allow_pickle=False)
assert score_p9.sha1(os.path.join(R, 'external', 'xpert_split_bundle', 'xpert_mdmt_splits.npz')) == '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464'
split = z['split_split_cold_drug_1'].astype(str)
pert_idx = z['meta_pert_idx'].astype(np.int64)
pert_id = z['meta_pert_id'].astype(str)
row_index = z['row_index'].astype(np.int64)

idx2smi = np.load(os.path.join(R, 'model', 'results', 'p10', 'p10d_inputs', 'idx2smi_all.npy'), allow_pickle=True).item()
cmap = {int(k): v for k, v in json.load(open(os.path.join(R, 'model', 'results', 'p10', 'p10d_inputs', 'compound_map.json'),
                                             encoding='utf-8')).items()}
caps = {}
for k in range(5):
    with open(os.path.join(R, 'external', 'kaggle_out', 'p10d_cap', f's{k}', f'_caption_cache_s{k}.pkl'), 'rb') as f:
        caps.update(pickle.load(f))
assert len(caps) == 1970
caption_of = {xp: caps[idx2smi[c]] for xp, c in cmap.items()}

# P9's scored rows (row_index only) and the clean subset of record
p9 = np.load(os.path.join(R, 'external', 'kaggle_out', 'v9p9', 'v9p9_seed0.npz'), allow_pickle=False)
p9_ri = np.asarray(p9['row_index'], dtype=np.int64)
pos = {int(r): i for i, r in enumerate(row_index)}
p9_pert = np.array([pert_id[pos[int(r)]] for r in p9_ri])
p9_pidx = np.array([pert_idx[pos[int(r)]] for r in p9_ri])
clean = json.load(open(os.path.join(R, score_p9.CLEAN_JSON), encoding='utf-8'))
assert score_p9.sha1(os.path.join(R, score_p9.CLEAN_JSON)) == score_p9.CLEAN_SHA1
dirty = set(clean['dirty_compounds'])
cmask = np.array([p not in dirty for p in p9_pert])
assert int(cmask.sum()) == score_p9.EXPECT_N_CLEAN and score_p9.row_set_sha1(p9_ri[cmask]) == score_p9.EXPECT_SHA1_CLEAN
units = json.load(open(os.path.join(R, score_p9.UNITS_JSON), encoding='utf-8'))
p2m = units['pert_to_molecule']

train_caps = {caption_of[int(p)] for p in np.unique(pert_idx[split == 'train'])}
coll = np.array([caption_of[int(p)] in train_caps for p in p9_pidx])
rows_c = np.sort(p9_ri[cmask & coll])
mol = np.array([p2m[p] for p in p9_pert])
mols_c = sorted(set(mol[cmask & coll]))
mols_all = sorted(set(mol[cmask]))
# a molecule is in the stratum if ANY of its clean rows' compounds collide; report whether that is all of its rows
mixed = sorted(m for m in mols_c if not coll[cmask & (mol == m)].all())
out = {
    'rule': 'clean test row whose compound\'s MolT5 caption (merged caption cache) equals the caption of any compound with a '
            'split_cold_drug_1 training row; molecule = InChIKey block 1 (P9 units)',
    'label': 'REPORTED ONLY, never read (RESULTS 98.20; review 081 C5)',
    'n_clean_rows': int(cmask.sum()), 'n_clean_molecules': len(mols_all),
    'n_rows': int(len(rows_c)), 'n_molecules': len(mols_c), 'molecules': mols_c, 'molecules_with_mixed_rows': mixed,
    'rows_sha1': score_p9.row_set_sha1(rows_c), 'row_index': rows_c.tolist(),
    'n_train_compounds': int(len(np.unique(pert_idx[split == 'train']))), 'n_distinct_train_captions': len(train_caps),
    'inputs': {'caption_cache_sha1s': {f's{k}': score_p9.sha1(os.path.join(R, 'external', 'kaggle_out', 'p10d_cap', f's{k}',
                                                                          f'_caption_cache_s{k}.pkl')) for k in range(5)},
               'compound_map_sha1': score_p9.sha1(os.path.join(R, 'model', 'results', 'p10', 'p10d_inputs', 'compound_map.json')),
               'clean_json_sha1': score_p9.CLEAN_SHA1, 'units_json_sha1': score_p9.UNITS_SHA1},
}
op = os.path.join(R, 'model', 'results', 'p10', 'p10d_caption_collision.json')
assert not os.path.exists(op), op
with open(op, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(out, f, indent=1)
print({k: v for k, v in out.items() if k not in ('row_index', 'molecules')})
