# -*- coding: utf-8 -*-
"""RESULTS 98.15 (review 078 C1 / C3): the PI's check of the input-only inverse of PertDiT's transform, on split_cold_drug_1 test rows.
Mirrors train_pertdit.py:144-159 (clip -> normalize_total(target_sum=None: median row total over the AnnData passed) -> log1p,
separately for X and X_ctl)."""
import numpy as np

z = np.load(r'C:\Projects\LINCS\external\xpert_split_bundle\xpert_mdmt_splits.npz', allow_pickle=True)
X = np.clip(z['X'].astype(np.float64), 0, 1e3)
C = np.clip(z['X_ctl'].astype(np.float64), 0, 1e3)
sp = z['split_split_cold_drug_1'].astype(str)
print('splits', {k: int((sp == k).sum()) for k in np.unique(sp)})
print('X range', X.min(), X.max(), 'negatives clipped', int((z['X'] < 0).sum()), int((z['X_ctl'] < 0).sum()))
S, Sc = X.sum(1), C.sum(1)
T_all, Tc_all = np.median(S), np.median(Sc)
tr, te = sp == 'train', sp == 'test'
T_tr = np.median(S[tr])
xt = np.log1p(X * (T_all / S)[:, None])
ct = np.log1p(C * (Tc_all / Sc)[:, None])
ratio = S[te] / Sc[te]
print('S/S_ctl test: mean %.5f sd %.4f p1 %.4f p99 %.4f' % (ratio.mean(), ratio.std(), *np.percentile(ratio, [1, 99])))
print('T_all/T_train %.6f   Tc_all %.3f T_all %.3f' % (T_all / T_tr, Tc_all, T_all))


def rowr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


d_raw = X[te] - C[te]
p = xt[te] - ct[te]                      # a perfect transformed-space prediction (their saved test_ground_truth)
for name, T in (('T_train', T_tr), ('T_all', T_all)):
    Xh = np.expm1(p + ct[te]) * (Sc[te] / T)[:, None]
    r = rowr(Xh - C[te], d_raw)
    print('oracle inverse (%s): mean %.5f median %.5f p1 %.4f min %.4f' % (name, r.mean(), np.median(r), np.percentile(r, 1), r.min()))
r2 = rowr(p, d_raw)
print('transformed vs raw delta, per row: mean %.4f median %.4f p5 %.4f' % (r2.mean(), np.median(r2), np.percentile(r2, 5)))
print('doses', np.unique(z['meta_dose']).tolist(), 'frac 10uM %.3f' % np.mean(np.isclose(z['meta_dose'], 10.0)))
print('test doses', np.unique(z['meta_dose'][te]).tolist())
print('n compounds (pert_idx)', len(np.unique(z['meta_pert_idx'])), 'max', int(z['meta_pert_idx'].max()))
