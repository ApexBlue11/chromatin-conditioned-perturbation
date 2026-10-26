"""T-C's registered reported item (RESULTS 100.5; review 100 C5): the drug gate gamma and the STRING gate a against each
test cell's OWN chromatin, by mark, for the chromatin-on checkpoints T-C used. tb_read.mechanism's measure on T-C's test
rows: 960 test rows drawn by np.random.default_rng(0), gamma averaged over the perturb blocks, then by tercile of the
cell's prom_1kb value per axis (ACC, ACT, PCG), on all test cells with the axis and on the 7 clean cells. Reported, never
read.

    python model/v10c/tc_mechanism.py --out model/results/v10c/tc_mechanism.json CKPT.pt [CKPT.pt ...]
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_v10c import V10CData, find  # noqa: E402
from model_v10c import LincsV10C, V10CConfig  # noqa: E402
import tc_read  # noqa: E402

AXES = {'ACC': 0, 'ACT': 5, 'PCG': 10}           # prom_1kb of each axis in v10c_chromatin.npz (as tb_read.AXES)
N_ROWS, BATCH = 960, 48
ROOTS = ['external/kaggle_v10c_src', 'external/v10c_full', 'external/lightning/v9payload/data']


def terciles(Gm, Ap, cr, ch, cells_sel):
    """Per axis: gamma and a by tercile of the row's cell's own prom_1kb, over rows whose cell has the axis."""
    F, M = ch['F'], ch['M'].astype(bool)
    names = [str(c) for c in ch['cells']]
    out = {}
    for ax, f in AXES.items():
        k = list(AXES).index(ax)
        gb, ab, used = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}, set()
        for i, c in enumerate(cr):
            if names[c] not in cells_sel or not M[c, :, k].any():
                continue
            used.add(names[c])
            ter = np.minimum((F[c, :, f] * 3).astype(int), 2)
            ok = M[c, :, k]
            for q in range(3):
                gb[q].append(Gm[i, ok & (ter == q)])
                ab[q].append(Ap[i, ok & (ter == q)])
        if not used:
            out[ax] = None
            continue
        g3 = [float(np.concatenate(gb[q]).mean()) for q in range(3)]
        a3 = [float(np.concatenate(ab[q]).mean()) for q in range(3)]
        out[ax] = {'cells': sorted(used), 'gamma_by_tercile': g3, 'gamma_top_minus_bottom': g3[2] - g3[0],
                   'a_ppi_by_tercile': a3, 'a_ppi_top_minus_bottom': a3[2] - a3[0]}
    return out


def run(path, D, ch, roots):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['cfg'] if isinstance(ck['cfg'], V10CConfig) else V10CConfig(**ck['cfg'])
    m = LincsV10C(cfg, np.load(find('M_pathway_v9.npy', roots)), np.load(find('STRING_adj_978_v9.npy', roots)),
                  np.load(find('gene_vectors_978.npy', roots)))
    m.fit_bins(*D.fit_rows())
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()})
    m.eval()
    rows = np.sort(np.random.default_rng(0).choice(D.ev, N_ROWS, replace=False))
    Gm, Ap = [], []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            _, aux = m(D.batch(rows[i:i + BATCH], 'cpu'), return_aux=True)
            Gm.append(aux['gamma'].float().mean(1).numpy())
            Ap.append(aux['a_ppi'].float().numpy())
    Gm, Ap, cr = np.concatenate(Gm), np.concatenate(Ap), D.cell_row[rows]
    names = {str(c) for c in ch['cells']}
    return {'checkpoint': os.path.basename(path), 'gamma_mean': float(Gm.mean()), 'gamma_sd_over_genes_rows': float(Gm.std()),
            'a_ppi_mean': float(Ap.mean()),
            'all_test_cells': terciles(Gm, Ap, cr, ch, names),
            'clean7': terciles(Gm, Ap, cr, ch, set(tc_read.CLEAN))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--roots', nargs='+', default=ROOTS)
    ap.add_argument('ckpts', nargs='+')
    a = ap.parse_args()
    torch.set_num_threads(4)
    D = V10CData(a.roots, split='cold_cell_1', dev=False, chrom_off=False)
    ch = np.load(find('v10c_chromatin.npz', a.roots))
    recs = []
    for p in a.ckpts:
        recs.append(run(p, D, ch, a.roots))
        print(json.dumps({k: recs[-1][k] for k in ('checkpoint', 'gamma_mean', 'a_ppi_mean')}), flush=True)
    json.dump({'rule': 'RESULTS 100.5 reported item (review 100 C5): gamma and a by tercile of the cell\'s own prom_1kb, '
                       'per mark; reported, never read', 'runs': recs}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
