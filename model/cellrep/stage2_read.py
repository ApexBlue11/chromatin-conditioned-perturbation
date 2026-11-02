"""Stage 2's reader (RESULTS 103.6 / 103.7), committed before any stage-2 output is opened. Two phases, read once each:

  inject   rung 3 only: for each family (M2 - M1; M3 - M4), shape (dense / graph; raw ACT reported) and level, the fold rule
           on LOCO: per fold the 3-seed mean delta of mean per-row r; mean over the 6 folds >= max(0.003, 2 SD / sqrt 6) and
           >= 5 of 6 folds positive -> detectable. The smallest detectable level per family and shape. Recovery reported.
  real     refuses to run unless the committed inject reading exists. LOCO d: R-chrom-MLP (M2 - M1), R-chrom-graph
           (M3 - M4), R-understand (M1, M2, M3 each against the per-fold maximum over baselines) by the fold rule.
           Cold-gene y: R-graph-cold "propagation against embedding" (M3 - M1), R-graph-real (M3 - M5), R-understand-cold
           (each against the best cold baseline): a paired gene-cluster bootstrap CI (10,000, default_rng(0)) of the mean
           over genes of each gene's 3-seed delta averaged over core cells must exclude 0 (>= 5 of 6 cells reported).
           Reported: K = 4, distance strata, tas strata, usage, fractions of the ceiling (>= 0.43), baselines.

    python model/cellrep/stage2_read.py --phase inject --outs external/kaggle_out/cellrep_s2 --out model/results/cellrep/stage2_inject.json
    python model/cellrep/stage2_read.py --phase real --outs external/kaggle_out/cellrep_s2 --out model/results/cellrep/stage2_real.json
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stage1 as s1  # noqa: E402

CORE = s1.CORE
SEEDS = (0, 1, 2)
CEILING = 0.43
INJECT_READING = 'model/results/cellrep/stage2_inject.json'


def fold_rule(deltas):
    """deltas: the 6 per-fold 3-seed mean deltas. Passes if mean >= max(0.003, 2 SD / sqrt 6) and >= 5 folds positive."""
    d = np.asarray(deltas, np.float64)
    thr = max(0.003, 2 * d.std(ddof=1) / np.sqrt(len(d)))
    return {'mean_delta': float(d.mean()), 'threshold': float(thr), 'folds_positive': int((d > 0).sum()),
            'per_fold': d.round(6).tolist(), 'passes': bool(d.mean() >= thr and (d > 0).sum() >= 5)}


def gene_bootstrap(delta_by_gene, n_boot=10000, seed=0):
    """Paired gene-cluster bootstrap of the mean over genes (review 110 C3)."""
    v = np.asarray(list(delta_by_gene.values()), np.float64)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    ci = [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]
    return {'n_genes': len(v), 'mean_delta': float(v.mean()), 'ci95': ci, 'passes': bool(ci[0] > 0 or ci[1] < 0)}


def load(outs, inject):
    """Opens only the injection outputs (inject=True) or only the real ones, chosen by file name before any file is read."""
    recs = {}
    for p in glob.glob(os.path.join(outs, '**', '*.json'), recursive=True):
        name = os.path.basename(p)
        if name.startswith('STAGE2_COMPLETE') or (('_inj-' in name) != inject):
            continue
        r = json.load(open(p))
        if 'job' in r:
            recs[os.path.basename(p)[:-5]] = r
    return recs


def tag(model, setting, seed, K=3, inj=None):
    t = [model, setting.replace(':', '-'), f's{seed}', f'K{K}']
    if inj:
        t.append(inj)
    return '_'.join(t)


def fold_deltas(recs, a, b, inj=None, K=3, key='mean_r'):
    """Per LOCO fold, the 3-seed mean of (a - b); None if any job is missing."""
    out = []
    for c in CORE:
        s = f'loco:{c}'
        da = [recs.get(tag(a, s, k, K if a in ('M3',) else 3, inj)) for k in SEEDS]
        db = [recs.get(tag(b, s, k, 3, inj)) for k in SEEDS]
        if any(x is None for x in da + db):
            return None
        out.append(np.mean([x[key] for x in da]) - np.mean([x[key] for x in db]))
    return out


def phase_inject(recs):
    res = {}
    for fam, (a, b), shapes, levels in (('MLP', ('M2', 'M1'), ('dense', 'graph', 'graph-nopsi'), (0.02, 0.05, 0.1)),
                                        ('graph', ('M3', 'M4'), ('graph-nopsi',), (0.05,))):
        for shape in shapes:
            rows = {}
            for lv in (levels if shape != 'graph-nopsi' else (0.05,)):
                inj = f"inj-{shape.replace('-nopsi', '')}-{lv}" + ('-nopsi' if shape.endswith('-nopsi') else '')
                d = fold_deltas(recs, a, b, inj)
                rows[str(lv)] = None if d is None else fold_rule(d)
                if rows[str(lv)] is not None:
                    rec_a = [np.mean(recs[tag(a, f'loco:{c}', k, 3, inj)]['injected_recovery_r']) for c in CORE for k in SEEDS]
                    rec_b = [np.mean(recs[tag(b, f'loco:{c}', k, 3, inj)]['injected_recovery_r']) for c in CORE for k in SEEDS]
                    rows[str(lv)]['recovery_r'] = {a: float(np.mean(rec_a)), b: float(np.mean(rec_b))}
            det = [float(lv) for lv in rows if rows[lv] and rows[lv]['passes']]
            res[f'{fam}_{shape}'] = {'levels': rows, 'smallest_detectable': min(det) if det else None}
        if fam == 'MLP':
            d = fold_deltas(recs, a, b, 'inj-dense-0.05-raw')
            res['MLP_dense_rawACT_reported'] = None if d is None else fold_rule(d)
    return res


def per_gene_deltas(recs, a, b, Ka=3, key='r'):
    """Cold-gene: each held-out gene's 3-seed mean of (a - b) per row, averaged over its core-cell rows."""
    acc = defaultdict(list)
    for k in SEEDS:
        ra, rb = recs[tag(a, 'cold', k, Ka)], recs[tag(b, 'cold', k, 3)]
        assert ra['rows'] == rb['rows']
        for (g, c), x, y in zip(ra['rows'], ra[key], rb[key]):
            acc[(g, c)].append(x - y)
    by_gene = defaultdict(list)
    for (g, c), v in acc.items():
        by_gene[g].append(np.mean(v))
    per_cell = defaultdict(list)
    for (g, c), v in acc.items():
        per_cell[c].append(np.mean(v))
    return {g: float(np.mean(v)) for g, v in by_gene.items()}, {c: float(np.mean(v)) for c, v in per_cell.items()}


def phase_real(recs, baselines):
    res = {'loco': {}, 'cold': {}}
    res['loco']['R-chrom-MLP'] = fold_rule(fold_deltas(recs, 'M2', 'M1'))
    res['loco']['R-chrom-graph'] = fold_rule(fold_deltas(recs, 'M3', 'M4'))
    best = [max(baselines['loco'][c].values()) for c in CORE]
    for m in ('M1', 'M2', 'M3'):
        mine = [np.mean([recs[tag(m, f'loco:{c}', k)]['mean_r'] for k in SEEDS]) for c in CORE]
        res['loco'][f'R-understand-{m}'] = fold_rule(np.array(mine) - np.array(best))
        res['loco'][f'R-understand-{m}']['fraction_of_ceiling'] = float(np.mean(mine) / CEILING)
    for name, (a, b, Ka) in {'R-graph-cold (propagation against embedding)': ('M3', 'M1', 3),
                             'R-graph-real': ('M3', 'M5', 3)}.items():
        g, cells = per_gene_deltas(recs, a, b, Ka)
        res['cold'][name] = dict(gene_bootstrap(g), cells_positive=int(sum(v > 0 for v in cells.values())), per_cell=cells)
    for m in ('M1', 'M2', 'M3'):
        g = defaultdict(list)
        for k in SEEDS:
            r = recs[tag(m, 'cold', k)]
            for (gg, c), x in zip(r['rows'], r['r']):
                g[gg].append(x - baselines['cold_best_by_row'][f'{gg}|{c}'])
        res['cold'][f'R-understand-cold-{m}'] = gene_bootstrap({k: float(np.mean(v)) for k, v in g.items()})
    res['reported'] = {
        'K4_minus_K3_loco': fold_deltas(recs, 'M3', 'M3', K=4),
        'mean_r': {f'{m}_{s}': float(np.mean([recs[tag(m, s, k)]['mean_r'] for k in SEEDS]))
                   for m in ('M1', 'M2', 'M3', 'M4', 'M5') for s in [f'loco:{c}' for c in CORE] + ['cold']},
        'usage': {t: r.get('usage') for t, r in recs.items() if r.get('usage')}, 'baselines': baselines['loco']}
    return res


def baselines_loco(D, c0):
    """Per-row r of each baseline on the held-out cell's d (103 harness): gene mean (0 by convention), nearest cell and k = 3
    nearest cells by CCLE, ridge on cell features (expression; + chromatin)."""
    import stage2 as s2
    tr, te, _ = D.loco(c0)
    E = {c: D.cellfeat['CE'][D.cell_index[c]][:978] for c in s2.TRAIN_CELLS}
    sim = {c: np.corrcoef(E[c0], E[c])[0, 1] for c in s2.TRAIN_CELLS if c != c0}
    out = {}
    def nn_pred(g, k):
        cand = sorted(((sim[c], c) for c in sim if (g, c) in D.idx), reverse=True)[:k]
        if not cand:
            return None
        w = np.array([max(s_, 1e-6) for s_, _ in cand])
        ref = [x for (gg, x) in D.idx if gg == g and x != c0]
        refm = D.Y[[D.idx[(g, x)] for x in ref]].mean(0)
        return (w[:, None] * np.stack([D.Y[D.idx[(g, c)]] for _, c in cand])).sum(0) / w.sum() - refm
    for name, k in (('nearest_cell', 1), ('knn3_cells', 3)):
        rs = [s1.row_r(p, d, D.lm_pos.get(g)) for (g, _, d) in te for p in [nn_pred(g, k)] if p is not None]
        out[name] = float(np.mean(rs))
    from sklearn.linear_model import Ridge
    cells = sorted({c for _, c, _ in tr})
    mean_d = np.stack([np.mean([d for _, cc, d in tr if cc == c], 0) for c in cells])
    wts = np.array([sum(cc == c for _, cc, _ in tr) for c in cells], np.float64)
    for name, feats in (('ridge_expression', lambda c: D.cellfeat['CE'][D.cell_index[c]]),
                        ('ridge_expression_chromatin', lambda c: np.r_[D.cellfeat['CE'][D.cell_index[c]],
                                                                        D.cellfeat['CL'][D.cell_index[c]].ravel()])):
        # cell features vary only across the training cells, so the row-level ridge equals a count-weighted ridge on
        # the cells' mean d; alpha by leave-one-cell-out
        X = np.stack([feats(c) for c in cells]).astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu) / sd
        def loco_err(al):
            e = 0.0
            for i in range(len(cells)):
                k = np.arange(len(cells)) != i
                m_ = Ridge(alpha=al).fit(Xs[k], mean_d[k], sample_weight=wts[k])
                e += wts[i] * ((m_.predict(Xs[i:i + 1])[0] - mean_d[i]) ** 2).mean()
            return e
        alpha = min(np.logspace(-1, 5, 13), key=loco_err)
        m = Ridge(alpha=alpha).fit(Xs, mean_d, sample_weight=wts)
        P = m.predict(((feats(c0) - mu) / sd)[None])[0]
        out[name] = float(np.mean([s1.row_r(P, d, D.lm_pos.get(g)) for g, _, d in te]))
    out['gene_mean'] = 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('inject', 'real'), required=True)
    ap.add_argument('--outs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--inject_sha1', default=None, help='the committed injection reading sha1 (real phase)')
    ap.add_argument('--roots', nargs='*', default=['external/kaggle_out/genetic_extract', 'external/cellrep_data',
                                                   'external/kaggle_alt2_stage/lincs-v11-src'])
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- read once')
    recs = load(a.outs, inject=(a.phase == 'inject'))
    if a.phase == 'inject':
        rec = {'rule': 'RESULTS 103.6 / 103.7 rung 3', 'n_jobs': len(recs), 'readings': phase_inject(recs)}
    else:
        import hashlib, subprocess
        if not os.path.exists(INJECT_READING) or subprocess.run(['git', 'ls-files', '--error-unmatch', INJECT_READING],
                                                                capture_output=True).returncode != 0:
            raise SystemExit('FATAL: the injection reading must be committed first (103.7)')
        got = hashlib.sha1(open(INJECT_READING, 'rb').read()).hexdigest()
        if got != a.inject_sha1:
            raise SystemExit(f'FATAL: injection reading sha1 {got} != pinned {a.inject_sha1} (review 111 C2)')
        import stage2 as s2
        D = s2.Data(a.roots)
        base = {'loco': {c: baselines_loco(D, c) for c in CORE}, 'cold_best_by_row': cold_baselines(D)}
        rec = {'rule': 'RESULTS 103.6 / 103.7 rungs 4-5', 'n_jobs': len(recs), 'readings': phase_real(recs, base)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w'), indent=1)
    print(json.dumps(rec['readings'], indent=1, default=str)[:8000])


def cold_baselines(D):
    """Cold-gene baselines per test row (y): ridge from [gene vector; cell features] and the cell offset (mean y of the
    cell's training genes); the best of the two per row is the comparator (by mean over rows of each)."""
    from sklearn.linear_model import RidgeCV
    tr, te, _ = D.cold()
    f = lambda g, c: np.r_[D.gvec[D.node_of[g]], D.cellfeat['CE'][D.cell_index[c]]]
    m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(np.stack([f(g, c) for g, c, _ in tr]), np.stack([t for _, _, t in tr]))
    P_ridge = m.predict(np.stack([f(g, c) for g, c, _ in te]))
    off = {c: np.mean([t for _, cc, t in tr if cc == c], 0) for c in CORE}
    r_ridge = [s1.row_r(p, t, D.lm_pos.get(g)) for (g, c, t), p in zip(te, P_ridge)]
    r_off = [s1.row_r(off[c], t, D.lm_pos.get(g)) for g, c, t in te]
    use = r_ridge if np.mean(r_ridge) >= np.mean(r_off) else r_off
    return {f'{g}|{c}': r for (g, c, _), r in zip(te, use)}


if __name__ == '__main__':
    main()
