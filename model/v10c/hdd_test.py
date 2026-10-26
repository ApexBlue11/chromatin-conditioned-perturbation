"""H-DD (RESULTS 100.15): in MCF7, does chromatin improve the prediction of DNA-damage compounds on rows T-C never saw?

The six T-C checkpoints (on: seeds 20, 11, 2; off: 0, 1, 2), unchanged, score the four clean cells' excluded_test_cell rows.
T-C's loader (data_v10c, pinned, not edited) is used with split_cold_cell_1 relabelled in memory:
  train                                  unchanged (so dose / time normalisation, the quantiser and the off arm's
                                         training-mean chromatin are T-C's)
  excluded_test_cell of MCF7, HT29,      -> 'test' (the held-out set: x_cell is leave-own-plate-out over these plates)
  MDAMB231, THP1
  everything else (T-C's own test rows)  -> dropped
Mates: other-compound held-out rows on the same plate (so within the cell's excluded rows), >= 2 (train_v10c's scorer).

The rule (100.15), on MCF7's DNA-damage rows with a contrast:
  (i)  all three on slot means > all three off slot means, AND
  (ii) the compound-cluster bootstrap one-sided 95 % lower bound of delta > 0 (compounds resampled, 20,000 draws,
       default_rng(0)).
Reported: the same per cell for HT29, MDAMB231, THP1, and pooled over the four.

    python model/v10c/hdd_test.py --tc_runs external/kaggle_out/v10c_tc --out model/results/v10c/hdd_reading.json
"""
import argparse, contextlib, glob, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_v10c  # noqa: E402
from data_v10c import V10CData, find  # noqa: E402
from model_v10c import LincsV10C, V10CConfig  # noqa: E402
from train_v10c import heldout_scores, predict_by_plate  # noqa: E402

CELLS = ('MCF7', 'HT29', 'MDAMB231', 'THP1')
PRIMARY = 'MCF7'
CLASS = 'dna_damage'
EXPECTED = {'MCF7': (78, 28)}            # 100.15's outcome-free count (rows with >= 2 mates, compounds)
CKPTS = {'on': [('tc_on_s0', 'tc_on_s0_retry2', 20), ('tc_on_s1', 'tc_on_s1_retry1', 11), ('tc_on_s2', 'tc_on_s2', 2)],
         'off': [('tc_off_s0', 'tc_off_s0', 0), ('tc_off_s1', 'tc_off_s1', 1), ('tc_off_s2', 'tc_off_s2', 2)]}
N_BOOT = 20000
ROOTS = ['external/kaggle_v10c_src', 'external/v10c_full', 'external/lightning/v9payload/data']


def relabel(lab, cell, cells=None):
    """split_cold_cell_1 -> train unchanged; the cells' excluded_test_cell rows -> 'test'; the rest 'dropped'."""
    cells = CELLS if cells is None else cells
    out = np.full(lab.shape, 'dropped', dtype='<U20')
    out[lab == 'train'] = 'train'
    out[(lab == 'excluded_test_cell') & np.isin(cell, cells)] = 'test'
    return out


def count_rows(sub, feature_index, cls_of, cells=None):
    """Outcome-free: per cell, the class rows of excluded_test_cell (featureless dropped) with >= 2 other-compound
    excluded rows of that cell on the same plate, and their compounds (100.15's count)."""
    cells = CELLS if cells is None else cells
    lab, cell, pert, plate = (sub[k] for k in ('split_cold_cell_1', 'meta_cell_iname', 'meta_pert_id', 'plate_id'))
    feat = np.array([p in feature_index for p in pert])
    is_cls = np.array([cls_of.get(p) == CLASS for p in pert])
    out = {}
    for c in cells:
        ex = (lab == 'excluded_test_cell') & (cell == c) & feat
        n, comps = 0, set()
        for i in np.flatnonzero(ex & is_cls):
            if (ex & (plate == plate[i]) & (pert != pert[i])).sum() >= 2:
                n += 1
                comps.add(pert[i])
        out[c] = (n, len(comps))
    return out


class _Relabelled:
    def __init__(self, z, lab):
        self.z, self.lab, self.files = z, lab, z.files

    def __getitem__(self, k):
        return self.lab if k == 'split_cold_cell_1' else self.z[k]


@contextlib.contextmanager
def relabelled_substrate(cells=None):
    """Within the block, np.load of the substrate returns it with split_cold_cell_1 relabelled (data_v10c untouched)."""
    cells = CELLS if cells is None else cells
    real = np.load

    def patched(path, *a, **k):
        obj = real(path, *a, **k)
        if os.path.basename(str(path)) == 'v10c_full_substrate.npz':
            return _Relabelled(obj, relabel(obj['split_cold_cell_1'], obj['meta_cell_iname'], cells))
        return obj
    data_v10c.np.load = patched
    try:
        yield
    finally:
        data_v10c.np.load = real


def load_data(roots, chrom_off, substrate=None, chromatin=None):
    with relabelled_substrate():
        return V10CData(roots, split='cold_cell_1', dev=False, chrom_off=chrom_off, substrate=substrate, chromatin=chromatin)


def scored_rows(D, cls_of):
    """The held-out rows on plates that hold a class row (mates are same-plate rows, so these suffice for the class rows'
    contrasts; x_cell was computed over every held-out plate when D was built)."""
    is_cls = np.array([cls_of.get(p) == CLASS for p in D.pert[D.ev]])
    plates = np.unique(D.plate[D.ev][is_cls])
    return D.ev[np.isin(D.plate[D.ev], plates)]


def build(ck_path, D, roots):
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    cfg = ck['cfg'] if isinstance(ck['cfg'], V10CConfig) else V10CConfig(**ck['cfg'])
    m = LincsV10C(cfg, np.load(find('M_pathway_v9.npy', roots)), np.load(find('STRING_adj_978_v9.npy', roots)),
                  np.load(find('gene_vectors_978.npy', roots)))
    m.fit_bins(*D.fit_rows())
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()})
    return m.eval()


def per_row_r(m, D, rows):
    pd, _ = predict_by_plate(m, D, rows, 'cpu', 64)
    T = D.X[rows] - D.ctl[D.ctl_id[rows]]
    _, r_con, _ = heldout_scores(pd, T, D.plate[rows], D.pert_code[rows], D.cell[rows])
    return r_con


def lower_bound(delta_row, comp, n_boot=N_BOOT, seed=0):
    """Compound-cluster bootstrap one-sided 95 % lower bound of the row mean of delta_row."""
    u, inv = np.unique(comp, return_inverse=True)
    s = np.bincount(inv, weights=delta_row, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n_boot, len(u)))
    return float(np.percentile(s[idx].sum(1) / n[idx].sum(1), 5))


def rule(Ron, Roff, comp, n_boot=N_BOOT):
    """Ron, Roff [3, n] per-row contrast r on the stratum's rows (finite); comp [n]."""
    a, b = Ron.mean(1), Roff.mean(1)
    d_row = Ron.mean(0) - Roff.mean(0)
    lb = lower_bound(d_row, comp, n_boot)
    up = -lower_bound(-d_row, comp, n_boot)
    i_, ii = bool(a.min() > b.max()), lb > 0
    mi, mii = bool(b.min() > a.max()), up < 0
    verdict = ('in MCF7, chromatin improves the prediction of DNA-damage compounds on untouched rows' if i_ and ii else
               'mirror: chromatin worsens it' if mi and mii else 'no claim')
    return {'n_rows': int(Ron.shape[1]), 'n_compounds': int(len(np.unique(comp))), 'slot_means_on': a.tolist(),
            'slot_means_off': b.tolist(), 'delta': float(a.mean() - b.mean()), 'cluster_one_sided_lower_95': lb,
            'cluster_one_sided_upper_95': up, 'cond_i_all_on_above_all_off': i_, 'cond_ii_lower_bound_above_0': ii,
            'verdict': verdict}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tc_runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--roots', nargs='+', default=ROOTS)
    ap.add_argument('--classes', default='model/results/v10c/compound_classes.json')
    ap.add_argument('--n_boot', type=int, default=N_BOOT)
    ap.add_argument('--expected', default=json.dumps(EXPECTED))
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- H-DD is read once')
    torch.set_num_threads(4)
    cls_of = json.load(open(a.classes, encoding='utf-8'))['class']
    R, meta = {}, None
    for arm, slots in CKPTS.items():
        D = load_data(a.roots, chrom_off=arm == 'off')
        rows = scored_rows(D, cls_of)
        if meta is None:
            meta = (rows, D.cell[rows], D.pert[rows], D.sub_row[rows])
        elif not np.array_equal(D.sub_row[rows], meta[3]):
            raise SystemExit('FATAL: the two arms are not on the same rows')
        R[arm] = []
        for slot, sub, seed in slots:
            hits = glob.glob(os.path.join(a.tc_runs, slot, sub, f'v10c_cold_cell_1_test_{arm}_d*_s{seed}.pt'))
            if len(hits) != 1:
                raise SystemExit(f'FATAL: expected one checkpoint for {arm} seed {seed}, found {hits}')
            R[arm].append(per_row_r(build(hits[0], D, a.roots), D, rows))
            print('scored', os.path.basename(hits[0]), flush=True)
    R = {k: np.stack(v) for k, v in R.items()}
    _, cell, pert, sub_row = meta
    is_cls = np.array([cls_of.get(p) == CLASS for p in pert])
    ok = is_cls & np.isfinite(R['on']).all(0) & np.isfinite(R['off']).all(0)
    if not np.array_equal(np.isfinite(R['on'][0]), np.isfinite(R['off'][0])):
        raise SystemExit('FATAL: the contrast mask differs between arms')
    expected = json.loads(a.expected)
    counts = {c: (int((ok & (cell == c)).sum()), int(len(np.unique(pert[ok & (cell == c)])))) for c in CELLS}
    for c, (n, k) in expected.items():
        if tuple(counts[c]) != (n, k):
            raise SystemExit(f'FATAL: {c} has {counts[c]} rows / compounds, the registration counted {(n, k)}')
    res = {'rule': 'RESULTS 100.15: (i) all 3 on slot means > all 3 off AND (ii) compound-cluster one-sided 95 % lower '
                   'bound > 0, MCF7 DNA-damage rows of excluded_test_cell; read once', 'counts': counts}
    for c in CELLS:
        m = ok & (cell == c)
        res[c] = rule(R['on'][:, m], R['off'][:, m], pert[m], a.n_boot) if m.any() else None
    res['pooled_reported'] = rule(R['on'][:, ok], R['off'][:, ok], np.char.add(cell[ok].astype(str), pert[ok].astype(str)),
                                  a.n_boot)
    res['primary'] = PRIMARY
    res['verdict'] = res[PRIMARY]['verdict']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({'verdict': res['verdict'], **{k: res[PRIMARY][k] for k in ('delta', 'slot_means_on', 'slot_means_off',
                                                                                    'cluster_one_sided_lower_95')}}))
    return res


if __name__ == '__main__':
    main()
