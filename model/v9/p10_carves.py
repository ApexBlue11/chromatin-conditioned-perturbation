# -*- coding: utf-8 -*-
"""RESULTS 98.2 / 98.6: P10's two fixed row sets, written once before any P10 output exists. No response is read.

  - the validation carve: 10 % of split_cold_drug_1's TRAINING molecules (InChIKey block 1; a pert without one is its own
    molecule), drawn by default_rng(9810); the carve is the training rows of those molecules. Same carve for every M seed and D.
  - the distinct-control stratum (review 073 C2b): clean scored rows whose X_ctl is not byte-identical to any training row's.

    python p10_carves.py OUT.json
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Projects\LINCS'
SPLIT = 'split_split_cold_drug_1'
VAL_FRAC = 0.10
VAL_SEED = 9810
EXPECT_N_SCORED, EXPECT_SHA1_SCORED = 13364, '5f85ef0b5bec'   # P9's row guard (96.2), prefix
EXPECT_N_CLEAN, EXPECT_SHA1_CLEAN = 11983, '6024dbf8a8d5301e68174d678e69f28b31f3aea7'


def rows_sha1(r):
    return hashlib.sha1(np.sort(np.asarray(r, dtype=np.int64)).tobytes()).hexdigest()


z = np.load(ROOT + r'\external\xpert_split_bundle\xpert_mdmt_splits.npz', allow_pickle=True)
pert = np.asarray(z['meta_pert_id']).astype(str)
lab = np.asarray(z[SPLIT]).astype(str)
ri = np.asarray(z['row_index'], dtype=np.int64)
tr, te = lab == 'train', lab == 'test'
assert tr.sum() == 55385 and te.sum() == 13445 and (tr | te).all()

# --- the scored and clean rows, re-derived and checked against P9's guards ---
di = json.load(open(ROOT + r'\drug\outputs\drug_feature_index.json'))
scored = te & np.array([p in di for p in pert])
assert scored.sum() == EXPECT_N_SCORED and rows_sha1(ri[scored]).startswith(EXPECT_SHA1_SCORED)
clean_json = json.load(open(ROOT + r'\model\results\mechanism96\cold_drug_clean_subset.json', encoding='utf-8'))
dirty = set(clean_json['dirty_compounds'])
clean = scored & ~np.isin(pert, sorted(dirty))
assert clean.sum() == EXPECT_N_CLEAN and rows_sha1(ri[clean]) == EXPECT_SHA1_CLEAN

# --- the validation carve ---
info = pd.concat([pd.read_csv(ROOT + r'\Data Info\GSE92742_Broad_LINCS_pert_info.txt\GSE92742_Broad_LINCS_pert_info.txt', sep='\t'),
                  pd.read_csv(ROOT + r'\Data Info\GSE70138_Broad_LINCS_pert_info_2017-03-06.txt\GSE70138_Broad_LINCS_pert_info.txt',
                              sep='\t')]).drop_duplicates('pert_id').set_index('pert_id')


def mol(p):
    k = str(info['inchi_key'].get(p, '')) if p in info.index else ''
    return k.split('-')[0] if k not in ('', '-666', 'nan') else 'PERT:' + p


tr_mol = np.array([mol(p) for p in pert[tr]])
mols = np.array(sorted(set(tr_mol)))
n_val = int(round(VAL_FRAC * len(mols)))
pick = set(mols[np.sort(np.random.default_rng(VAL_SEED).choice(len(mols), n_val, replace=False))].tolist())
val = np.zeros(len(pert), bool)
val[np.flatnonzero(tr)[np.isin(tr_mol, sorted(pick))]] = True
te_mol = {mol(p) for p in pert[te]}
# cold-drug: a test molecule may share a key with a training one (the leak, 96.6); the overlap is reported, not used

# --- the distinct-control stratum ---
Xc = np.ascontiguousarray(z['X_ctl'])
tr_ctl = {Xc[i].tobytes() for i in np.flatnonzero(tr)}
shared = np.array([Xc[i].tobytes() in tr_ctl for i in range(len(Xc))])
distinct_clean = clean & ~shared

out = {
    'rule': 'RESULTS 98.2 / 98.6; written once before any P10 output',
    'split': SPLIT,
    'validation_carve': {'frac_of_training_molecules': VAL_FRAC, 'rng': 'default_rng(%d)' % VAL_SEED,
                         'n_training_molecules': int(len(mols)), 'n_val_molecules': n_val,
                         'n_val_rows': int(val.sum()), 'n_fit_rows': int((tr & ~val).sum()),
                         'val_rows_sha1': rows_sha1(ri[val]), 'val_row_index': sorted(ri[val].tolist()),
                         'n_val_molecules_also_a_test_key': len(pick & te_mol)},
    'distinct_control_stratum': {'n_scored_rows_sharing_a_training_ctl': int((scored & shared).sum()),
                                 'n_clean_rows_sharing': int((clean & shared).sum()),
                                 'n_clean_distinct': int(distinct_clean.sum()),
                                 'n_clean_distinct_molecules': len({mol(p) for p in pert[distinct_clean]}),
                                 'clean_distinct_rows_sha1': rows_sha1(ri[distinct_clean]),
                                 'clean_distinct_row_index': sorted(ri[distinct_clean].tolist())},
}
with open(sys.argv[1], 'x', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print({k: {kk: vv for kk, vv in v.items() if not kk.endswith('row_index')} for k, v in out.items() if isinstance(v, dict)})
