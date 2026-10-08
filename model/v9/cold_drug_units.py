# -*- coding: utf-8 -*-
"""RESULTS 96.7 (review 065 C1, C3a, C3c; identity level, no response read): the molecule units of split_cold_drug_1's scored
test rows. Each featurisable test pert_id maps to a molecule key (InChIKey first block, else the pert_id itself); the full and
clean scored sets with their row counts and molecule counts; and each molecule's max-Tanimoto stratum (max over its pert_ids).

    python model/v9/cold_drug_units.py OUT.json
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Projects\LINCS'
z = np.load(ROOT + r'\external\xpert_split_bundle\xpert_mdmt_splits.npz', allow_pickle=True)
pert = np.asarray(z['meta_pert_id']).astype(str)
ri = np.asarray(z['row_index'])
lab = z['split_split_cold_drug_1']
te = lab == 'test'
di = json.load(open(ROOT + r'\drug\outputs\drug_feature_index.json'))
audit = json.load(open(ROOT + r'\model\results\mechanism96\cold_drug_molecule_audit.json'))['split_cold_drug_1']['compounds']
clean = json.load(open(ROOT + r'\model\results\mechanism96\cold_drug_clean_subset.json'))
info = pd.concat([pd.read_csv(ROOT + r'\Data Info\GSE92742_Broad_LINCS_pert_info.txt\GSE92742_Broad_LINCS_pert_info.txt', sep='\t'),
                  pd.read_csv(ROOT + r'\Data Info\GSE70138_Broad_LINCS_pert_info_2017-03-06.txt\GSE70138_Broad_LINCS_pert_info.txt', sep='\t')])
ik = info.drop_duplicates('pert_id').set_index('pert_id')['inchi_key'].astype(str).to_dict()
mx = {r['pert_id']: r['max_tanimoto'] for r in audit}
excluded = set(clean['dirty_compounds'])


def key(p):
    k = ik.get(p, '')
    return k.split('-')[0] if k not in ('', '-666', 'nan') else 'PERT:' + p


def stratum(t):
    if t is None:
        return None
    return 'lt_0.6' if t < 0.6 else ('0.6_0.8' if t < 0.8 else ('0.8_0.999' if t < 0.999 else 'ge_0.999'))


scored = te & np.isin(pert, list(di))
full_p = sorted(set(pert[scored]))
mol = {p: key(p) for p in full_p}
groups = {}
for p, k in mol.items():
    groups.setdefault(k, []).append(p)
dups = {k: v for k, v in groups.items() if len(v) > 1}
out = {'split': 'split_cold_drug_1', 'molecule_key': 'InChIKey first block (else PERT:<pert_id>)', 'pert_to_molecule': mol,
       'test_internal_duplicates': dups}
for name, ps in (('full', full_p), ('clean', [p for p in full_p if p not in excluded])):
    rows = scored & np.isin(pert, ps)
    ms = sorted({mol[p] for p in ps})
    mol_max = {m: max(mx[p] for p in groups[m] if p in ps and mx.get(p) is not None) for m in ms}
    strata = pd.Series({m: stratum(t) for m, t in mol_max.items()}).value_counts().to_dict()
    out[name] = {'n_pert_ids': len(ps), 'n_molecules': len(ms), 'n_rows': int(rows.sum()),
                 'row_index_sha1': hashlib.sha1(np.sort(ri[rows]).astype(np.int64).tobytes()).hexdigest(),
                 'molecule_max_tanimoto_strata': strata}
    if name == 'full':
        out['excluded_rows_of_scored'] = int((scored & np.isin(pert, list(excluded))).sum())
out['molecule_stratum'] = {m: stratum(max(mx[p] for p in groups[m] if mx.get(p) is not None)) for m in groups}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('pert_to_molecule', 'molecule_stratum')}, indent=1))
