# -*- coding: utf-8 -*-
"""
Proof that the proposed MyDataset memory patch changes NO value, only storage. [RESULTS 73, packet 009]

Their MyDataset.load_data does, per row, `tensor(self.drug_feat[pert_idx], dtype=torch.float32)`: a fresh
245 KB copy of that drug's (122, 514) block for every row. Measured: 268 KB per row, 245 KB of it the drug
block, every drug stored as separate per-row copies, 24.7 GB for train + val + test on split_cold_cell_1 --
which is why launch v3's one-batch probe was killed at row ~18,500 of the second test build.

The patch computes that SAME tensor once per drug and reuses it. This script builds their dataset twice on
identical rows -- once from their file verbatim, once from the patched copy -- and requires every tensor in
every item to be torch.equal with identical dtype and shape. It also reports the storage saving.
"""
import importlib.util
import io
import os
import sys

import numpy as np
import torch

HERE = r'C:\Projects\LINCS\model\v9'
sys.path.insert(0, HERE)
import xpert_native_eval as X  # noqa: E402

ORIG = 'drug_feat = tensor(drug_feat, dtype=torch.float32) if self.args.drug_feat != \'smi\' else drug_feat'
PATCH = ('# [LINCS memory patch, RESULTS 73] one tensor per drug, reused: same values, shared storage.\n'
         '            if self.args.drug_feat != \'smi\':\n'
         '                _k = pert_id if self.args.dataset == \'transigen_sdst\' else pert_idx\n'
         '                _c = self.__dict__.setdefault(\'_drug_tensor_cache\', {})\n'
         '                if _k not in _c:\n'
         '                    _c[_k] = tensor(drug_feat, dtype=torch.float32)\n'
         '                drug_feat = _c[_k]')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    E = X.load_everything('cpu')
    src_path = os.path.join(X.XPERT, 'datasets', 'MyDataset.py')
    src = io.open(src_path, encoding='utf-8').read()
    assert src.count(ORIG) == 1, 'the line to patch must occur exactly once'
    patched_src = src.replace(ORIG, PATCH, 1)
    tmp = os.path.join(os.environ['TEMP'], 'MyDataset_patched.py')
    io.open(tmp, 'w', encoding='utf-8').write(patched_src)
    A = load_module('MyDataset_orig', src_path).MyDataset
    B = load_module('MyDataset_patched', tmp).MyDataset

    adata = E['ad'].read_h5ad(X.H5AD)
    feats = X.drug_feat_dict(X.UNIMOL)
    sp = adata.obs['split_cold_cell_1'].values
    idx = np.where(sp == 'test')[0][:int(os.environ.get('N_ROWS', '600'))]
    sub = X.subset(E, adata, np.isin(np.arange(adata.n_obs), idx))
    args = X.their_args(nfold='split_cold_cell_1', device='cpu')
    kw = dict(max_value=E['cfg']['dataset']['max_value'], min_value=E['cfg']['dataset']['min_value'])
    da = A(sub, feats, args, E['cfg'], E['logger'], **kw).data
    db = B(sub, feats, args, E['cfg'], E['logger'], **kw).data

    assert len(da) == len(db), (len(da), len(db))
    n_tensors = 0
    for i, (ra, rb) in enumerate(zip(da, db)):
        assert len(ra) == len(rb), 'row %d arity' % i
        for j, (a, b) in enumerate(zip(ra, rb)):
            if torch.is_tensor(a):
                assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape, (i, j)
                assert torch.equal(a, b), 'row %d field %d differs' % (i, j)
                n_tensors += 1
            else:
                assert a == b, 'row %d field %d (non-tensor) differs' % (i, j)

    def storage_bytes(data):
        seen, total = set(), 0
        for r in data:
            for t in r:
                if torch.is_tensor(t):
                    p = t.untyped_storage().data_ptr()
                    if p not in seen:
                        seen.add(p)
                        total += t.untyped_storage().nbytes()
        return total

    sa, sb = storage_bytes(da), storage_bytes(db)
    print('rows %d | tensors compared %d | ALL torch.equal, dtypes and shapes identical' % (len(da), n_tensors))
    print('unique storage: original %.1f MB, patched %.1f MB (%.1fx less)' % (sa / 1e6, sb / 1e6, sa / sb))
    per_row_a, per_row_b = sa / len(da), sb / len(db)
    n = 47509 + 21321 + 21321
    print('projected for train + val + test (%d rows): original %.1f GB, patched %.2f GB'
          % (n, n * per_row_a / 1e9, n * per_row_b / 1e9 + 1970 * 122 * 514 * 4 / 1e9))
    print('PROOF PASSED')


if __name__ == '__main__':
    main()
