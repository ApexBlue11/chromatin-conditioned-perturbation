"""T-B, read once when all six d = 256 runs exist (RESULTS 100.4's rule; 100.4b's reported items; 100.7 / 100.9's post hoc
trained-runs view, the collapse counts). Nothing here selects anything.

  registered   3-seed mean dev contrast r, on - off >= max(0.003, 2 sqrt(s_on^2/3 + s_off^2/3)), the mean of per-cell
               means favours on, and >= 4 of 6 dev cells favour on (3-seed per-cell means)
  post hoc     the same over the runs whose drug path trained (tb_trained_check.py), unequal-n threshold; never a pass
  reported     delta r; the gene-shuffle and cell-swap permutations of each on run; collapse counts per arm; the
               ChIP-Atlas tier composition of each dev cell; with --mechanism, the gate gamma, the STRING gate a and the
               dose scale against the cell's own chromatin terciles (prom_1kb of ACC, ACT, PCG) and against dose, also on
               the cells whose tracks are all ENCODE / explicit-control (tiers E, C) as 087 C3's sensitivity

    python model/v10c/tb_read.py --runs_dir external/kaggle_out/v10c_tb --out model/results/v10c/tb_reading.json \
        --trained model/results/v10c/tb_trained_check_first3.json model/results/v10c/tb_trained_check_next2.json ... \
        [--mechanism]
"""
import argparse, hashlib, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ARMS, SEEDS = ('on', 'off'), (0, 1, 2)
LEDGER = 'model/results/v10c/tb_ledger.jsonl'
SELECTION = 'model/results/v10c/chipatlas_selection.json'
AXES = {'ACC': 0, 'ACT': 5, 'PCG': 10}           # the prom_1kb feature of each axis in v10c_chromatin.npz
N_MECH_ROWS = 960


def tag_of(arm, seed, d=256):
    return f'cold_cell_1_dev_{arm}_d{d}_s{seed}'


def load_runs(runs_dir, ledger=LEDGER):
    """Every T-B record, checked against the ledger's sha1 (a record that differs from what was logged is refused)."""
    led = {}
    for line in open(ledger, encoding='utf-8'):
        e = json.loads(line)
        led[e['run']] = e
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            t, tag = f'tb_{arm}_s{s}', tag_of(arm, s)
            p = os.path.join(runs_dir, t, t, f'V10C_COMPLETE_{tag}.json')
            raw = open(p, 'rb').read()
            if tag not in led or hashlib.sha1(raw).hexdigest() != led[tag]['record_sha1']:
                raise SystemExit(f'FATAL: {p} is not the record in the ledger')
            runs[(arm, s)] = json.loads(raw)
    return runs


def rule(on, off, cells_on, cells_off, min_cells=4):
    """100.4's rule on lists of per-run row-mean contrast r and per-run {cell: contrast r}. The threshold is the unequal-n
    form, which equals the registered one when n_on = n_off = 3."""
    on, off = np.asarray(on, float), np.asarray(off, float)
    d = float(on.mean() - off.mean())
    s_on = float(on.std(ddof=1)) if len(on) > 1 else float('nan')
    s_off = float(off.std(ddof=1)) if len(off) > 1 else float('nan')
    thr = max(0.003, 2 * math.sqrt(s_on ** 2 / len(on) + s_off ** 2 / len(off)))
    cells = sorted(cells_on[0])
    cm_on = {c: float(np.mean([r[c] for r in cells_on])) for c in cells}
    cm_off = {c: float(np.mean([r[c] for r in cells_off])) for c in cells}
    cell_mean_delta = float(np.mean(list(cm_on.values())) - np.mean(list(cm_off.values())))
    fav = [c for c in cells if cm_on[c] > cm_off[c]]
    return {'n_on': int(len(on)), 'n_off': int(len(off)), 'mean_on': float(on.mean()), 'mean_off': float(off.mean()),
            'delta': d, 's_on': s_on, 's_off': s_off, 'threshold': thr, 'delta_clears_threshold': bool(d >= thr),
            'cell_mean_delta': cell_mean_delta, 'per_cell_on': cm_on, 'per_cell_off': cm_off,
            'cells_favouring_on': fav, 'n_cells_favouring_on': len(fav),
            'passes': bool(d >= thr and cell_mean_delta > 0 and len(fav) >= min_cells)}


def per_cell(rec):
    return {c: v['contrast_r'] for c, v in rec['scores']['per_cell'].items()}


def trained_flags(paths):
    flags = {}
    for p in paths:
        for r in json.load(open(p, encoding='utf-8'))['runs']:
            flags[r['checkpoint'].replace('v10c_', '').replace('.pt', '')] = r['drug_path_trained']
    return flags


def tiers(cells, selection=SELECTION):
    sel = json.load(open(selection, encoding='utf-8'))
    tier = {r['srx']: r['tier'] for r in sel['records']}
    out = {}
    for c in cells:
        out[c] = {ax: [tier.get(x, '?') for x in srx] for ax, srx in sel['selection'].get(c, {}).items()}
    return out


def mechanism(runs_dir, cells_tiers, roots):
    """Gate gamma (mean over the perturb blocks), STRING gate a and dose scale s on N_MECH_ROWS dev rows per run, by the
    cell's OWN prom_1kb tercile per axis (also for the off arm, whose model saw the training-mean chromatin)."""
    import torch
    from data_v10c import V10CData, find
    from model_v10c import LincsV10C, V10CConfig
    torch.set_num_threads(4)
    ch = np.load(find('v10c_chromatin.npz', roots))
    F_true, M_true = ch['F'], ch['M'].astype(bool)
    out, cache = {}, {}
    for arm in ARMS:
        if arm not in cache:
            cache[arm] = V10CData(roots, split='cold_cell_1', dev=True, chrom_off=arm == 'off')
        D = cache[arm]
        rows = np.sort(np.random.default_rng(0).choice(D.ev, N_MECH_ROWS, replace=False))
        for s in SEEDS:
            t = f'tb_{arm}_s{s}'
            p = os.path.join(runs_dir, f'{t}_ckpt', t, f'v10c_{tag_of(arm, s)}.pt')
            ck = torch.load(p, map_location='cpu', weights_only=False)
            cfg = ck['cfg'] if isinstance(ck['cfg'], V10CConfig) else V10CConfig(**ck['cfg'])
            m = LincsV10C(cfg, np.load(find('M_pathway_v9.npy', roots)), np.load(find('STRING_adj_978_v9.npy', roots)),
                          np.load(find('gene_vectors_978.npy', roots)))
            Xc, Xm = D.fit_rows()
            m.fit_bins(Xc, Xm)
            m.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()})
            m.eval()
            Gm, Ap, S, ld, cr = [], [], [], [], []
            with torch.no_grad():
                for i in range(0, len(rows), 48):
                    b = D.batch(rows[i:i + 48], 'cpu')
                    _, aux = m(b, return_aux=True)
                    Gm.append(aux['gamma'].float().mean(1).numpy())
                    Ap.append(aux['a_ppi'].float().numpy())
                    S.append(aux['dose_scale'].float().numpy())
                    ld.append(b['dose'].numpy())
                    cr.append(D.cell_row[rows[i:i + 48]])
            Gm, Ap, S, ld, cr = map(np.concatenate, (Gm, Ap, S, ld, cr))
            rec = {'gamma_mean': float(Gm.mean()), 'a_ppi_mean': float(Ap.mean()), 'by_axis': {}}
            for ax, f in AXES.items():
                k = list(AXES).index(ax)
                for subset in ('all', 'EC_only'):
                    gb, ab = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}
                    n_cells = set()
                    for i in range(len(cr)):
                        c = cr[i]
                        name = str(ch['cells'][c])
                        if not M_true[c, :, k].any():
                            continue
                        if subset == 'EC_only' and not all(x in ('E', 'C') for x in cells_tiers.get(name, {}).get(ax, ['?'])):
                            continue
                        n_cells.add(name)
                        ter = np.minimum((F_true[c, :, f] * 3).astype(int), 2)
                        ok = M_true[c, :, k]
                        for q in range(3):
                            sel = ok & (ter == q)
                            gb[q].append(Gm[i, sel])
                            ab[q].append(Ap[i, sel])
                    if not n_cells:
                        rec['by_axis'][f'{ax}_{subset}'] = None
                        continue
                    g3 = [float(np.concatenate(gb[q]).mean()) for q in range(3)]
                    a3 = [float(np.concatenate(ab[q]).mean()) for q in range(3)]
                    rec['by_axis'][f'{ax}_{subset}'] = {'cells': sorted(n_cells), 'gamma_by_tercile': g3,
                                                       'gamma_top_minus_bottom': g3[2] - g3[0], 'a_ppi_by_tercile': a3,
                                                       'a_ppi_top_minus_bottom': a3[2] - a3[0]}
            order = np.argsort(np.argsort(ld))
            rs = np.argsort(np.argsort(S))
            rec['dose_scale'] = {'median': float(np.median(S)),
                                 'spearman_with_dose': float(np.corrcoef(order, rs)[0, 1]),
                                 'mean_by_dose_quintile': [float(S[(order * 5 // len(order)) == q].mean()) for q in range(5)]}
            out[t] = rec
            print(t, json.dumps({k: rec[k] for k in ('gamma_mean', 'a_ppi_mean')}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', required=True)
    ap.add_argument('--trained', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mechanism', action='store_true')
    ap.add_argument('--roots', nargs='+', default=['external/kaggle_v10c_src', 'external/v10c_full',
                                                    'external/lightning/v9payload/data'])
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- T-B is read once')
    runs = load_runs(a.runs_dir)
    flags = trained_flags(a.trained)
    missing = [tag_of(arm, s) for arm in ARMS for s in SEEDS if tag_of(arm, s) not in flags]
    if missing:
        raise SystemExit(f'FATAL: no trained-run check for {missing}')
    cr = {k: r['scores']['contrast_r_row_mean'] for k, r in runs.items()}
    pc = {k: per_cell(r) for k, r in runs.items()}
    reg = rule([cr[('on', s)] for s in SEEDS], [cr[('off', s)] for s in SEEDS],
               [pc[('on', s)] for s in SEEDS], [pc[('off', s)] for s in SEEDS])
    tr = {k: flags[tag_of(*k)] for k in runs}
    ton, toff = [s for s in SEEDS if tr[('on', s)]], [s for s in SEEDS if tr[('off', s)]]
    post = None
    if len(ton) >= 2 and len(toff) >= 2:
        post = rule([cr[('on', s)] for s in ton], [cr[('off', s)] for s in toff],
                    [pc[('on', s)] for s in ton], [pc[('off', s)] for s in toff])
        post['label'] = 'POST HOC (100.9), motivated by on s0; dev scores of on s0, on s1, off s0 were visible; never a pass'
        post['seeds_on'], post['seeds_off'] = ton, toff
    perm = {}
    for s in SEEDS:
        r = runs[('on', s)]
        pm = r['reported_permutations']
        perm[f'on_s{s}'] = {'real': cr[('on', s)], 'trained': tr[('on', s)],
                            **{k: pm[k]['contrast_r_row_mean'] for k in pm},
                            **{f'real_minus_{k}': cr[('on', s)] - pm[k]['contrast_r_row_mean'] for k in pm}}
    cells = sorted(pc[('on', 0)])
    ct = tiers(cells)
    rec = {'rule': 'RESULTS 100.4 (registered); 100.9 (post hoc view)',
           'runs': {f'{arm}_s{s}': {'contrast_r_row_mean': cr[(arm, s)], 'contrast_r_cell_mean': runs[(arm, s)]['scores']['contrast_r_cell_mean'],
                                    'delta_r_row_mean': runs[(arm, s)]['scores']['delta_r_row_mean'], 'per_cell': pc[(arm, s)],
                                    'drug_path_trained': tr[(arm, s)], 'record_pred_sha1': runs[(arm, s)]['pred_sha1']}
                    for arm in ARMS for s in SEEDS},
           'registered': reg, 'post_hoc_trained_runs': post,
           'collapse_counts': {arm: {'untrained': sum(not tr[(arm, s)] for s in SEEDS), 'of': len(SEEDS)} for arm in ARMS},
           'reported_delta_r': {arm: float(np.mean([runs[(arm, s)]['scores']['delta_r_row_mean'] for s in SEEDS])) for arm in ARMS},
           'reported_permutations': perm, 'tier_composition': ct}
    if a.mechanism:
        rec['mechanism'] = mechanism(a.runs_dir, ct, a.roots)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({'registered_passes': reg['passes'], 'delta': reg['delta'], 'threshold': reg['threshold'],
                      'post_hoc': None if post is None else {k: post[k] for k in ('delta', 'threshold', 'cell_mean_delta',
                                                                                 'n_cells_favouring_on', 'passes')},
                      'collapse_counts': rec['collapse_counts']}, indent=1))


if __name__ == '__main__':
    main()
