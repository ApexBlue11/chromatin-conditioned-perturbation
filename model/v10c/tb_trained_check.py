"""RESULTS 100.9: the "drug path trained" criterion, fixed before it is applied. A run's drug path is UNTRAINED iff the
median dose scale s over 480 training rows is below 1e-4 (100.8's floor). The rows are drawn by np.random.default_rng(0)
exactly as in diag_dose_collapse.py (480 dev rows first, then 480 training rows); the training rows are the same in both
arms. Also reported (review 090 C3): the min and max of a_d (ld - b_d), and how many rows have s < 1e-6 or s == 0.
The rows come from the checkpoint's own split, read from its name (v10c_<split>_test_...): cold_cell_1 for T-C, cold_drug_1
for the gap arm (100.11 C3b: the check runs on every T-C and gap checkpoint).

    python model/v10c/tb_trained_check.py --out model/results/v10c/tb_trained_check.json CKPT.pt [CKPT.pt ...]
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_v10c import V10CData, find  # noqa: E402
from model_v10c import LincsV10C, V10CConfig  # noqa: E402

FLOOR, N_ROWS, BATCH = 1e-4, 480, 48
ROOTS = ['external/kaggle_v10c_src', 'external/v10c_full', 'external/lightning/v9payload/data']


def training_rows(D, n=N_ROWS, seed=0):
    rng = np.random.default_rng(seed)
    rng.choice(D.ev, n, replace=False)                    # consumed first, as in diag_dose_collapse.py
    return np.sort(rng.choice(D.tr, n, replace=False))


def dose_terms(m, b):
    """(s, z) for one batch, where s = sigmoid(z) and z = a_d (ld - b_d), captured at the DoseScale's input."""
    got = {}
    h = m.dose.register_forward_hook(lambda mod, inp, out: got.update(g=inp[0], ld=inp[1], s=out))
    try:
        m(b)
    finally:
        h.remove()
    a = torch.nn.functional.softplus(m.dose.a(got['g'])).squeeze(-1) + 0.1
    z = a * (got['ld'] - m.dose.b(got['g']).squeeze(-1))
    return got['s'], z


def verdict(s_all, z_all, floor=FLOOR):
    med = float(np.median(s_all))
    return {'n_rows': int(len(s_all)), 's_median': med, 's_min': float(s_all.min()), 's_max': float(s_all.max()),
            'frac_s_below_1e-6': float((s_all < 1e-6).mean()), 'n_s_exactly_0': int((s_all == 0).sum()),
            'z_min': float(z_all.min()), 'z_max': float(z_all.max()), 'drug_path_trained': bool(med >= floor)}


def split_of(path):
    name = os.path.basename(path)
    hits = [s for s in ('cold_cell_1', 'cold_drug_1') if f'_{s}_' in name]
    if len(hits) != 1:
        raise SystemExit(f'FATAL: cannot tell the split of {name}')
    return hits[0]


def check(path, data_cache, roots=ROOTS):
    off = '_off_' in os.path.basename(path)
    key = (split_of(path), off)
    if key not in data_cache:
        data_cache[key] = V10CData(roots, split=key[0], dev=True, chrom_off=off)
    D = data_cache[key]
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['cfg'] if isinstance(ck['cfg'], V10CConfig) else V10CConfig(**ck['cfg'])
    m = LincsV10C(cfg, np.load(find('M_pathway_v9.npy', roots)), np.load(find('STRING_adj_978_v9.npy', roots)),
                  np.load(find('gene_vectors_978.npy', roots)))
    Xc, Xm = D.fit_rows()
    m.fit_bins(Xc, Xm)
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()})
    m.eval()
    rows = training_rows(D)
    S, Z = [], []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            s, z = dose_terms(m, D.batch(rows[i:i + BATCH], 'cpu'))
            S.append(s.double().numpy())
            Z.append(z.double().numpy())
    return {'checkpoint': os.path.basename(path), 'split': key[0], 'arm': 'off' if off else 'on', 'd_model': cfg.d_model,
            **verdict(np.concatenate(S), np.concatenate(Z))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('ckpts', nargs='+')
    a = ap.parse_args()
    torch.set_num_threads(4)
    cache, recs = {}, []
    for p in a.ckpts:
        recs.append(check(p, cache))
        print(json.dumps(recs[-1]), flush=True)
    json.dump({'rule': 'RESULTS 100.9: untrained iff median s over 480 training rows < 1e-4', 'runs': recs},
              open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
