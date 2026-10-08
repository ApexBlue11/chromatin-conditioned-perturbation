# -*- coding: utf-8 -*-
"""RESULTS 96.4 item 4 (identity level, before any P9 prediction): the cold-drug A3 member compounds, each with its max Tanimoto
(ECFP4, r=2, 2048 bits) to the training compounds and its nearest training analogue. Run with drug/.venv-drug (RDKit)."""
import json
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = r'C:\Projects\LINCS'
z = np.load(ROOT + r'\external\xpert_split_bundle\xpert_mdmt_splits.npz', allow_pickle=True)
lab = z['split_split_cold_drug_1']
pert = np.asarray(z['meta_pert_id']).astype(str)
cell = np.asarray(z['meta_cell']).astype(str)
te, tr = lab == 'test', lab == 'train'
di = json.load(open(ROOT + r'\drug\outputs\drug_feature_index.json'))
info = pd.concat([pd.read_csv(ROOT + r'\Data Info\GSE92742_Broad_LINCS_pert_info.txt\GSE92742_Broad_LINCS_pert_info.txt', sep='\t'),
                  pd.read_csv(ROOT + r'\Data Info\GSE70138_Broad_LINCS_pert_info_2017-03-06.txt\GSE70138_Broad_LINCS_pert_info.txt', sep='\t')])
smi = info.drop_duplicates('pert_id').set_index('pert_id')['canonical_smiles'].to_dict()
names = info.drop_duplicates('pert_id').set_index('pert_id')['pert_iname'].to_dict()
d = pd.read_csv(ROOT + r'\drug\outputs\dti\chembl_dti_edges.tsv', sep='\t')
d = d[(d.direct_interaction == 1) & (d.organism == 'Homo sapiens')]
moa = d.groupby('pert_id')['mechanism_of_action'].apply(set).to_dict()
tgt = d.groupby('pert_id').apply(lambda g: set(zip(g.gene_symbol.astype(str), g.action_type.astype(str)))).to_dict()

SCORED = ['MCF7', 'PC3', 'A375', 'HA1E', 'HT29', 'A549']
TP53WT = {'MCF7', 'A549', 'A375'}
members = {}
for p in sorted(set(pert[te]) & set(di)):
    cells_p = set(cell[te & (pert == p)]) & set(SCORED)
    if 'DNA inhibitor' in moa.get(p, set()) and cells_p & TP53WT:
        members.setdefault(p, set()).add('DNA')
    if any(g in ('EGFR', 'ERBB2') and a.upper() in ('INHIBITOR', 'ANTAGONIST', 'NEGATIVE MODULATOR', 'BLOCKER')
           for g, a in tgt.get(p, set())) and cells_p:
        members.setdefault(p, set()).add('EGFR')


def fp(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s not in ('-666', '') else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None


train_ids = [p for p in sorted(set(pert[tr])) if fp(smi.get(p)) is not None]
train_fps = [fp(smi[p]) for p in train_ids]
rows = []
for p, cls in sorted(members.items()):
    f = fp(smi.get(p))
    if f is None:
        rows.append({'pert_id': p, 'name': names.get(p), 'classes': sorted(cls), 'max_tanimoto': None, 'nearest_train': None})
        continue
    sims = DataStructs.BulkTanimotoSimilarity(f, train_fps)
    j = int(np.argmax(sims))
    rows.append({'pert_id': p, 'name': names.get(p), 'classes': sorted(cls), 'max_tanimoto': round(float(sims[j]), 3),
                 'nearest_train': train_ids[j], 'nearest_train_name': names.get(train_ids[j]),
                 'flag_gt_0_8': bool(sims[j] > 0.8)})
out = {'n_members': len(rows), 'n_train_compounds_fingerprinted': len(train_ids), 'members': rows,
       'fingerprint': 'Morgan radius 2, 2048 bits (ECFP4), RDKit'}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for r in rows:
    print(r)
