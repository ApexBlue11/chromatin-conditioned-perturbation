# -*- coding: utf-8 -*-
"""RESULTS 76.1, T2: does the atom tokens' benefit on unseen_cell grow with the compound's training exposure?

Pre-committed at 55a3098 before this ran. Unit = compound. Null key = x_cell, same construction, same rows.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = r'C:\Projects\LINCS\model\v9'
sys.path.insert(0, HERE)
from config_v9 import V9DataConfig  # noqa: E402
from data_v9 import LincsV9Dataset  # noqa: E402
from train_v9_gpu import resolve_v9  # noqa: E402
from data import build_splits  # noqa: E402

RES = r'C:\Projects\LINCS\model\results'
A = np.load(os.path.join(RES, 'v9_interaction_2x2_sa0_ckpt_v9_fold0_seed0_rows.npz'))
N = np.load(os.path.join(RES, 'v9_interaction_2x2_sa0_ckpt_v9_fold0_seed0_key-x_cell_rows.npz'))
rows = A['unseen_cell__rows']
assert np.array_equal(rows, N['unseen_cell__rows']), 'hypothesis and null-key rows differ'

dc = resolve_v9(V9DataConfig())
dc.cache_in_ram = False
ds = LincsV9Dataset(dc, _shared=LincsV9Dataset.load_shared_v9(dc))
sp = build_splits(ds, dc)
assert np.all(np.isin(rows, sp['test_coldcell'])), 'rows are not all test_coldcell'
drug = np.asarray(ds.drug_row)
train_drug = drug[sp['train']]
u, c = np.unique(train_drug, return_counts=True)
n_train = dict(zip(u.tolist(), c.tolist()))
row_drug = drug[rows]
missing = int(sum(d not in n_train for d in row_drug))
print('unseen_cell rows %d | distinct compounds %d | rows whose compound has NO training rows: %d'
      % (len(rows), len(np.unique(row_drug)), missing))


def compound_level(e):
    ok = np.isfinite(e)
    comp, E, n = [], [], []
    for d in np.unique(row_drug[ok]):
        m = ok & (row_drug == d)
        if d in n_train:
            comp.append(d); E.append(np.median(e[m])); n.append(n_train[d])
    return np.array(E), np.log(np.array(n, float))


def perm_test(E, logn, n_perm=20000, seed=0):
    rho = spearmanr(E, logn).correlation
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(E, rng.permutation(logn)).correlation for _ in range(n_perm)])
    return rho, float((null >= rho).mean()), float((np.abs(null) >= abs(rho)).mean())


out = {}
for tag, Z in (('atoms', A), ('x_cell', N)):
    e = Z['unseen_cell__r11'] - Z['unseen_cell__r01']
    E, logn = compound_level(e)
    rho, p_one, p_two = perm_test(E, logn)
    out[tag] = {'n_compounds': int(len(E)), 'spearman_rho': round(float(rho), 4),
                'perm_p_one_sided': p_one, 'perm_p_two_sided': p_two,
                'median_row_effect': round(float(np.nanmedian(e)), 5)}
    print('%-7s compounds %d | Spearman rho(E_c, log n_c) = %+.4f | one-sided perm p = %.4f | two-sided %.4f'
          % (tag, len(E), rho, p_one, p_two))

# The RESULTS 76.1 table, row by row, in its own order.
ra, pa = out['atoms']['spearman_rho'], out['atoms']['perm_p_one_sided']
rn, pn = out['x_cell']['spearman_rho'], out['x_cell']['perm_p_two_sided']
gate_passes = (pn >= 0.05) and (ra > 0) and (abs(rn) < 0.5 * ra)
out['null_key_gate_passes'] = bool(gate_passes)
if ra <= 0 or pa >= 0.05:
    verdict = 'NOT SUPPORTED'
elif pa < 0.01 and gate_passes:
    verdict = 'CONSISTENT WITH MEMORISATION'
elif pa < 0.01 and not gate_passes:
    verdict = 'NULL KEY FAILS ITS GATE: the exposure correlation is generic, not attributed to atoms'
else:
    verdict = 'INCONCLUSIVE'
out['verdict'] = verdict
out['exposure_quantiles'] = [float(np.quantile([n_train[d] for d in np.unique(row_drug) if d in n_train], q))
                             for q in (0, 0.25, 0.5, 0.75, 1)]
print('training-exposure quantiles over these compounds (rows):', out['exposure_quantiles'])
print('\nVERDICT (RESULTS 76.1):', verdict)
json.dump(out, open(os.path.join(RES, 'v9_memorisation_T2_unseen_cell.json'), 'w'), indent=2)
print('wrote v9_memorisation_T2_unseen_cell.json')
