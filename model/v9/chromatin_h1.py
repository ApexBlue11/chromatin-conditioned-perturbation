# -*- coding: utf-8 -*-
"""RESULTS 93.3 / 93.10: H1 on richer accessibility features in T1 (promoter, enhancer, regulon/TF motif).
Linear T1 with incremental arm FB + F_prom + {F_enh, F_reg} vs FB + F_prom.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402
import chromatin_power as cp  # noqa: E402

PIS = (0.005, 0.02, 0.05)


def install_c93(ctx, arrays):
    """arrays = npz contents as dict (cells, genes, F_prom, F_enh, F_reg, has, etc.).
    Sets ctx.enc['c93'] with b, Ez, has, cov_dt, cov_dev."""
    rn = ctx.enc['rank_normal']
    b = rn['b']
    G = cf.G
    n = len(ctx.cells)
    Ez = np.zeros((n, G, 3), dtype=np.float32)
    has = np.zeros((n, 3), dtype=bool)

    c93_cells = [str(c) if not isinstance(c, bytes) else c.decode('utf-8') for c in arrays['cells']]
    c93_has = np.asarray(arrays['has'], bool)
    c93_cell_map = {c: i for i, c in enumerate(c93_cells)}

    for i, c in enumerate(ctx.cells):
        idx = c93_cell_map.get(c)
        if idx is not None and c93_has[idx]:
            Ez[i, :, 0] = cf.rank_normal(np.asarray(arrays['F_prom'][idx], np.float64))
            Ez[i, :, 1] = cf.rank_normal(np.asarray(arrays['F_enh'][idx], np.float64))
            Ez[i, :, 2] = cf.rank_normal(np.asarray(arrays['F_reg'][idx], np.float64))
            has[i, :] = True

    cov = [c for i, c in enumerate(ctx.cells) if has[i].any()]
    cov_dt = [c for c in cov if c not in cf.DEV_CELLS]
    cov_dev = [c for c in cov if c in cf.DEV_CELLS]

    assert not (set(cov_dt) & set(cf.DEV_CELLS)), "GUARD: dev cell in cov_dt"
    assert set(cov_dev).issubset(set(cf.DEV_CELLS)), "GUARD: non-dev cell in cov_dev"

    ctx.enc['c93'] = {
        'b': b,
        'Ez': Ez,
        'has': has,
        'cov_dt': cov_dt,
        'cov_dev': cov_dev,
    }
    return ctx.enc['c93']


def h1_builder(ctx, enc, kind, fit_cells, lam=None):
    """Feature builder for H1 kinds:
    'B'  = [b, P]
    'C'  = [b, P, E, R]
    'Cp' = [b, E, R]
    'N1' = [b, P, mean E, mean R]
    'S'  = [b, P, (1 - λ) E + λ mean E, (1 - λ) R + λ mean R]
    """
    e = ctx.enc[enc]
    b, Ez, has = e['b'], e['Ez'], e['has']
    G = cf.G

    def build(h):
        out = {}
        if kind in ('N1', 'S'):
            src = [cf.cell_index(ctx, c) for c in fit_cells if c != h]
            mean_E = np.zeros(G, np.float32)
            mean_R = np.zeros(G, np.float32)
            idx = [i for i in src if has[i, 0]]
            if idx:
                mean_E = Ez[idx, :, 1].mean(0)
                mean_R = Ez[idx, :, 2].mean(0)

        for i, c in enumerate(ctx.cells):
            b_col = b[i][:, None]
            c_has = has[i, 0]
            if kind == 'B':
                P_col = Ez[i, :, 0][:, None] if c_has else np.zeros((G, 1), np.float32)
                f = np.concatenate([b_col, P_col], axis=1)
            elif kind == 'C':
                if c_has:
                    f = np.concatenate([b_col, Ez[i]], axis=1)
                else:
                    f = np.concatenate([b_col, np.zeros((G, 3), np.float32)], axis=1)
            elif kind == 'Cp':
                if c_has:
                    f = np.concatenate([b_col, Ez[i, :, 1][:, None], Ez[i, :, 2][:, None]], axis=1)
                else:
                    f = np.concatenate([b_col, np.zeros((G, 2), np.float32)], axis=1)
            elif kind == 'N1':
                if c_has:
                    f = np.concatenate([b_col, Ez[i, :, 0][:, None], mean_E[:, None], mean_R[:, None]], axis=1)
                else:
                    f = np.concatenate([b_col, np.zeros((G, 3), np.float32)], axis=1)
            elif kind == 'S':
                assert lam is not None, "lam must be provided for kind 'S'"
                if c_has:
                    E_shrunk = (1.0 - lam) * Ez[i, :, 1] + lam * mean_E
                    R_shrunk = (1.0 - lam) * Ez[i, :, 2] + lam * mean_R
                    f = np.concatenate([b_col, Ez[i, :, 0][:, None], E_shrunk[:, None], R_shrunk[:, None]], axis=1)
                else:
                    f = np.concatenate([b_col, np.zeros((G, 3), np.float32)], axis=1)
            else:
                raise ValueError(f"Unknown kind {kind}")
            out[c] = f.astype(np.float32)
        return out
    return build


def cell_rule(n_measured):
    assert n_measured <= 5, f'More than 5 measured dev cells cannot happen: {n_measured}'
    if n_measured == 5:
        return 4
    elif n_measured == 4:
        return 3
    else:
        return None


def paired_on(a, b, cells):
    """Paired comparison restricted to the given cells."""
    d = {c: a['per_cell'][c] - b['per_cell'][c] for c in cells}
    dc = {c: a['centred_per_cell'][c] - b['centred_per_cell'][c] for c in cells}
    return {
        'all': a['all'] - b['all'],
        'top': a['top'] - b['top'],
        'per_cell': d,
        'cells_pos': int(sum(v > 0 for v in d.values())),
        'centred_all': a['centred_all'] - b['centred_all'],
        'centred_per_cell': dc,
        'centred_cells_pos': int(sum(v > 0 for v in dc.values()))
    }


def h1_reading(sC, sB, sN1, bars, cells, k):
    """RESULTS §93.10 item 6 reading logic."""
    d = paired_on(sC, sB, cells)
    dn = paired_on(sC, sN1, cells)
    conj = {
        'delta_all_or_top_ge_bar': bool(d['all'] >= bars['t1'] or d['top'] >= bars['t1_top']),
        'cells_delta_pos_ge_k': bool(d['cells_pos'] >= k),
        'centred_all_ge_bar': bool(d['centred_all'] >= bars['t1_centred']),
        'cells_centred_pos_ge_k': bool(d['centred_cells_pos'] >= k),
        'vs_N1_all_ge_bar': bool(dn['all'] >= bars['t1'] / 2.0),
        'cells_C_gt_N1_ge_k': bool(dn['cells_pos'] >= k)
    }
    return {
        'delta': d,
        'vs_N1': dn,
        'conjuncts': conj,
        'bars': bars,
        'pass': bool(all(conj.values()))
    }


def run_h1(ctx, y, enc, report_extras=True):
    fit = ctx.enc[enc]['cov_dt']
    cov_dev = ctx.enc[enc]['cov_dev']
    cf.assert_no_dev(ctx, np.isin(ctx.rows['cell'], fit) & ctx.fit_mask, 'h1 fitting pool')
    assert not (set(fit) & set(cf.DEV_CELLS)), 'GUARD: dev cell in fit'

    k = cell_rule(len(cov_dev))
    bars = cf.BARS['known']

    specs = {
        'B': (h1_builder(ctx, enc, 'B', fit), ['full']),
        'C': (h1_builder(ctx, enc, 'C', fit), ['full']),
        'N1': (h1_builder(ctx, enc, 'N1', fit), ['full']),
    }
    if report_extras:
        specs['Cp'] = (h1_builder(ctx, enc, 'Cp', fit), ['full'])
        for lam in (0.25, 0.5, 0.75):
            specs[f'S_{lam:g}'] = (h1_builder(ctx, enc, 'S', fit, lam=lam), ['full'])

    out, mu, level = cf.run_t1(ctx, y, specs, fit)

    dev_rows = np.flatnonzero(ctx.dev_mask)
    sub = (level[dev_rows] <= 2) & np.isin(ctx.rows['cell'][dev_rows], cov_dev)

    sc = {}
    for arm in specs:
        sc[arm] = cf.score(ctx, y, out[(arm, 'full')]['y_hat_dev'], sub)

    reading = h1_reading(sc['C'], sc['B'], sc['N1'], bars, cov_dev, k) if k is not None else None

    res = {
        'reading': reading,
        'pass': reading['pass'] if reading else False,
        'delta': reading['delta'] if reading else None,
        'known': reading,
        'scores': {arm: cf.strip(sc[arm]) for arm in sc},
        'k': k,
        'measured_dev_cells': cov_dev,
        'fitting_cells': fit,
    }

    if report_extras:
        reading_cp = h1_reading(sc['Cp'], sc['B'], sc['N1'], bars, cov_dev, k) if k is not None else None
        res['Cp_vs_B'] = {
            'delta': paired_on(sc['Cp'], sc['B'], cov_dev),
            'reading': reading_cp,
        }

        locos = {arm: out[(arm, 'full')]['loco'] for arm in specs}
        res['loco'] = locos
        res['loco_C_minus_B'] = locos['C'] - locos['B']
        res['loco_C_minus_N1'] = locos['C'] - locos['N1']

        s_lams = (0.25, 0.5, 0.75)
        best_lam = max(s_lams, key=lambda l: locos[f'S_{l:g}'])
        s_best = f'S_{best_lam:g}'
        reading_s = h1_reading(sc[s_best], sc['B'], sc['N1'], bars, cov_dev, k) if k is not None else None
        res['shrinkage'] = {
            'best_lambda': best_lam,
            'reading_vs_B': paired_on(sc[s_best], sc['B'], cov_dev),
            'reading_vs_N1': paired_on(sc[s_best], sc['N1'], cov_dev),
            'reading': reading_s,
        }

    return res


def calibrate(ctx, slot, draws=3):
    assert slot in (1, 2), f"Slot must be 1 or 2, got {slot}"
    c93 = ctx.enc['c93']
    fit_cells = list(c93['cov_dt'])
    cov_dev = list(c93['cov_dev'])
    all_has_cells = fit_cells + cov_dev
    G = cf.G

    mu_e, _ = cf.condition_means(ctx, ctx.y, ctx.fit_mask)
    records = []

    for d in range(draws):
        rhos = []
        for c in fit_cells:
            i = cf.cell_index(ctx, c)
            r_cb = np.corrcoef(c93['Ez'][i, :, slot], c93['b'][i])[0, 1]
            if np.isfinite(r_cb):
                rhos.append(r_cb)
        rho = float(np.median(rhos)) if rhos else 0.0

        perm = np.random.default_rng(1000 + d).permutation(G)
        scale = np.sqrt(max(1.0 - rho * rho, 0.0))

        fstar = {}
        for c in all_has_cells:
            i = cf.cell_index(ctx, c)
            b_c = c93['b'][i].astype(np.float64)
            v_c = c93['Ez'][i, :, slot].astype(np.float64)
            fstar[c] = (rho * b_c + scale * v_c[perm]).astype(np.float32)

        Ez_calib = np.zeros_like(c93['Ez'])
        for k in range(3):
            if k == slot:
                for c in all_has_cells:
                    i = cf.cell_index(ctx, c)
                    Ez_calib[i, :, slot] = fstar[c]
            else:
                perm_k = np.random.default_rng(3000 + 10 * d + k).permutation(G)
                for c in all_has_cells:
                    i = cf.cell_index(ctx, c)
                    Ez_calib[i, :, k] = c93['Ez'][i, perm_k, k]

        ctx.enc['c93_calib'] = {
            'b': c93['b'],
            'Ez': Ez_calib.astype(np.float32),
            'has': c93['has'].copy(),
            'cov_dt': list(c93['cov_dt']),
            'cov_dev': list(c93['cov_dev']),
        }

        fbar = np.mean(np.stack(list(fstar.values())), axis=0)
        fg = {c: fbar for c in fstar}

        cases = (
            [('null', None, None, None)]
            + [(f, pi, f, fstar) for pi in PIS for f in ('P1', 'P2')]
            + [('P3', 0.05, 'P3', fstar)]
            + [('G', pi, 'P1', fg) for pi in (0.02, 0.05)]
        )

        rec = {'draw': d, 'rho': rho, 'cases': {}}
        for cid, pi, form, fmap in cases:
            if cid == 'null':
                y_case, alpha = ctx.y, 0.0
            else:
                y_case, alpha = cp.plant(ctx, ctx.y, mu_e, fmap, form, pi, d)
            r = run_h1(ctx, y_case, 'c93_calib', report_extras=False)
            r['alpha'] = float(alpha)
            rec['cases'][cid if pi is None else f'{cid}_{pi:g}'] = r
        records.append(rec)

    return records


def faults_and_mde(records):
    def _get_r(case):
        if 'reading' in case:
            return case['reading']
        if 'known' in case:
            return case['known']
        return case

    p = lambda r, k: bool(_get_r(r['cases'][k])['pass'])
    delta_all = lambda r, k: float(_get_r(r['cases'][k])['delta']['all'])

    t1_bar = cf.BARS['known']['t1']
    g_inf = any(all(delta_all(r, f'G_{pi:g}') >= t1_bar for r in records) for pi in (0.02, 0.05))

    out = {
        'VOID_null': any(p(r, 'null') for r in records),
        'VOID_P3': any(p(r, 'P3_0.05') for r in records),
        'VOID_G': any(p(r, f'G_{pi:g}') for r in records for pi in (0.02, 0.05)),
        'G_informative': g_inf,
        'G_raw_delta': [delta_all(r, f'G_{pi:g}') for r in records for pi in (0.02, 0.05)],
        'MDE': {},
    }
    for form in ('P1', 'P2'):
        out['MDE'][form] = next((pi for pi in PIS if all(p(r, f'{form}_{pi:g}') for r in records)), None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--c93', required=True)
    ap.add_argument('--c93_sha1', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--dti', default=None)
    a = ap.parse_args()

    got_c93_sha1 = cf.sha1_file(a.c93)
    if got_c93_sha1 != a.c93_sha1:
        raise SystemExit(f'REFUSED: --c93 sha1 {got_c93_sha1} does not match --c93_sha1 ({a.c93_sha1})')

    t0 = time.time()
    ctx = cf.prepare(a.data_dir, a.provenance, a.dti)
    arrays = dict(np.load(a.c93, allow_pickle=True))
    install_c93(ctx, arrays)

    cov_dev = ctx.enc['c93']['cov_dev']
    fitting_cells = ctx.enc['c93']['cov_dt']
    k = cell_rule(len(cov_dev))

    inputs = dict(ctx.inputs)
    inputs[os.path.basename(a.c93)] = a.c93_sha1

    if k is None:
        res = {
            'not_run': 'too few measured dev cells',
            'measured_dev_cells': cov_dev,
            'fitting_cells': fitting_cells,
            'k': None,
            'inputs': inputs,
            'seconds': time.time() - t0,
        }
        json.dump(cf.jsonable(res), open(a.out, 'w'), indent=1)
        marker = {'complete': True, 'outputs': {os.path.basename(a.out): cf.sha1_file(a.out)}, 'inputs': inputs}
        json.dump(marker, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'CHROMATIN93_H1_COMPLETE.json'), 'w'), indent=1)
        return

    calib = {}
    fam = {}
    for slot in (1, 2):
        calib[slot] = calibrate(ctx, slot, draws=a.draws)
        fam[slot] = faults_and_mde(calib[slot])

    real = run_h1(ctx, ctx.y, 'c93', report_extras=True)

    res = {
        'real': real,
        'calibration': calib,
        'faults_and_mde': fam,
        'measured_dev_cells': cov_dev,
        'fitting_cells': fitting_cells,
        'k': k,
        'inputs': inputs,
        'seconds': time.time() - t0,
    }
    json.dump(cf.jsonable(res), open(a.out, 'w'), indent=1)
    marker = {'complete': True, 'outputs': {os.path.basename(a.out): cf.sha1_file(a.out)}, 'inputs': inputs}
    json.dump(marker, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'CHROMATIN93_H1_COMPLETE.json'), 'w'), indent=1)
    # No reading is printed


if __name__ == '__main__':
    main()
