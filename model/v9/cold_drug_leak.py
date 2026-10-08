# -*- coding: utf-8 -*-
"""Identity-level audit: how many of XPert's cold-drug TEST compounds are the same molecule as a TRAINING compound under a
different pert_id? (InChIKey, InChIKey first block, ECFP4 Tanimoto.) No response is read. Run with drug/.venv-drug.

    python cold_drug_leak.py OUT.json [SPLIT ...]    # default: split_cold_drug_1 split_cold_cell_1 (96.6)
"""
import json
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = r'C:\Projects\LINCS'
z = np.load(ROOT + r'\external\xpert_split_bundle\xpert_mdmt_splits.npz', allow_pickle=True)
pert = np.asarray(z['meta_pert_id']).astype(str)
info = pd.concat([pd.read_csv(ROOT + r'\Data Info\GSE92742_Broad_LINCS_pert_info.txt\GSE92742_Broad_LINCS_pert_info.txt', sep='\t'),
                  pd.read_csv(ROOT + r'\Data Info\GSE70138_Broad_LINCS_pert_info_2017-03-06.txt\GSE70138_Broad_LINCS_pert_info.txt', sep='\t')])
info = info.drop_duplicates('pert_id').set_index('pert_id')
out = {}
for sp in (sys.argv[2:] or ['split_cold_drug_1', 'split_cold_cell_1']):
    lab = z['split_' + sp]
    te = sorted(set(pert[lab == 'test']))
    tr = sorted(set(pert[lab == 'train']))
    ik = lambda p: str(info['inchi_key'].get(p, '')) if p in info.index else ''        # noqa: E731
    ik1 = lambda p: ik(p).split('-')[0] if ik(p) not in ('', '-666', 'nan') else ''  # noqa: E731
    tr_ik = {ik(p) for p in tr if ik(p) not in ('', '-666', 'nan')}
    tr_ik1 = {ik1(p) for p in tr if ik1(p)}

    def fp(p):
        s = info['canonical_smiles'].get(p) if p in info.index else None
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s not in ('-666', '') else None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None
    tr_fp = [f for f in (fp(p) for p in tr) if f is not None]
    rows = []
    for p in te:
        f = fp(p)
        mx = float(max(DataStructs.BulkTanimotoSimilarity(f, tr_fp))) if f is not None else None
        rows.append({'pert_id': p, 'name': info['pert_iname'].get(p) if p in info.index else None,
                     'same_inchikey_in_train': ik(p) in tr_ik, 'same_inchikey_block1_in_train': ik1(p) in tr_ik1 if ik1(p) else None,
                     'max_tanimoto': mx})
    df = pd.DataFrame(rows)
    n = len(df)
    rows_per = pd.Series(pert[lab == 'test']).value_counts()
    df['n_test_rows'] = df['pert_id'].map(rows_per)
    summ = {'n_test_compounds': n,
            'same_inchikey': int(df.same_inchikey_in_train.sum()),
            'same_inchikey_block1': int(df.same_inchikey_block1_in_train.fillna(False).sum()),
            'tanimoto_eq_1': int((df.max_tanimoto >= 0.999).sum()),
            'tanimoto_gt_0_8': int((df.max_tanimoto > 0.8).sum()),
            'tanimoto_gt_0_6': int((df.max_tanimoto > 0.6).sum()),
            'test_rows_in_tanimoto_eq_1_compounds': int(df.loc[df.max_tanimoto >= 0.999, 'n_test_rows'].sum()),
            'test_rows_total': int(df.n_test_rows.sum()),
            'median_max_tanimoto': float(df.max_tanimoto.median())}
    out[sp] = {'summary': summ, 'compounds': df.to_dict('records') if sp.startswith('split_cold_drug') else None}
    print(sp, summ)
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=str)
