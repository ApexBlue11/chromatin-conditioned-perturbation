# -*- coding: utf-8 -*-
"""Apply RESULTS 88.3 mechanically to the eight 88 probe outputs (trained t0-t2, untrained u0-u4); 88.6 item 6.
Committed before any probe output exists. The one form 88.3 leaves open is fixed HERE, before any output:
a reference projection "is target-aligned" iff on ANY of the three seeds its diff <= -0.02 with p < 0.05 (the
conservative reading: the "beyond ..." qualifier needs the projection to be non-aligned on every seed).

    python model/v9/read_moa_88.py DIR [OUT]     # DIR/t{0,1,2}/probe_moa_88_*.json, DIR/u{0..4}/probe_moa_88_*.json
"""
import glob
import json
import os
import sys

import numpy as np

FLOOR, PMAX, GATE = -0.02, 0.05, 0.95
ROWS = '3e59a7ba7832775596f032dbab06c2c36a7723b4'


def load(D, tag):
    f = glob.glob(os.path.join(D, tag, 'probe_moa_88_*.json'))
    assert len(f) == 1, (tag, f)
    return json.load(open(f[0]))


def read(T, U):
    """T: 3 trained probe dicts (seeds 0-2); U: 5 untrained. Returns the reading record."""
    out = {'rules': {'floor': FLOOR, 'p': PMAX, 'gate_rho_post': GATE, 'sd_ddof': 1}, 'checks': {}, 'seeds': {}}
    allj = list(T) + list(U)
    # identity checks (88.6 items 1-4): rows, quintiles, compounds, epochs, provenance
    ok_rows = all(j.get('row_sha1') == ROWS for j in allj)
    live = [j for j in allj if not j.get('void')]
    ok_q = len({j.get('quintile_sha1') for j in live}) == 1          # a void record has no quintiles; VOID below
    ok_n = len({j.get('n_compounds') for j in live}) == 1
    ok_epoch = all(j.get('epoch') == 11 and not j.get('untrained') for j in T)
    ok_u = all(j.get('untrained') and j.get('cfg_from') == 'c8b_ckpt_v9_fold0_seed0.pt' for j in U)
    out['checks'] = {'rows_sha1': ok_rows, 'quintile_sha1_identical': ok_q, 'n_compounds_identical': ok_n,
                     'trained_epoch_11': ok_epoch, 'untrained_from_fold0_seed0': ok_u,
                     'untrained_void': [bool(j.get('void')) for j in U], 'trained_void': [bool(j.get('void')) for j in T]}
    if not (ok_rows and ok_epoch and ok_u) or any(j.get('void') for j in U):
        out['reading'] = 'INVALID'
        return out
    if not (ok_q and ok_n):
        out['reading'] = 'INVALID'
        return out

    ud = np.array([u['primary']['score']['diff'] for u in U])
    uu = np.array([u['strata']['unseen']['score']['diff'] for u in U])
    m_u, sd_u = float(ud.mean()), float(ud.std(ddof=1))
    m_un, sd_un = float(uu.mean()), float(uu.std(ddof=1))
    out['untrained'] = {'diffs': ud.tolist(), 'm_u': m_u, 'sd_u': sd_u, 'min': float(ud.min()),
                        'unseen_diffs': uu.tolist(), 'm_u_unseen': m_un, 'sd_u_unseen': sd_un,
                        'unseen_min': float(uu.min()), 'rho_post': [u['rho_post'] for u in U]}
    thr_all, thr_un = min(float(ud.min()), m_u - 2 * sd_u), min(float(uu.min()), m_un - 2 * sd_un)

    all_ok, un_ok, gate_ok = [], [], []
    for s, t in enumerate(T):
        if t.get('void'):
            out['seeds'][s] = {'void': True}
            all_ok.append(False); un_ok.append(False); gate_ok.append(False)
            continue
        g = t['gate_pass'] and t['rho_post'] < GATE
        p, ps, un = t['primary']['score'], t['primary']['null1s'], t['strata']['unseen']['score']
        a = [p['diff'] <= float(ud.min()), p['diff'] <= m_u - 2 * sd_u, p['p'] < PMAX, ps['p_s'] < PMAX,
             p['S'] < p['null2_mean']]
        b = [un['diff'] <= float(uu.min()), un['diff'] <= m_un - 2 * sd_un, un['p'] < PMAX, un['diff'] <= FLOOR]
        gate_ok.append(g); all_ok.append(all(a)); un_ok.append(all(b))
        out['seeds'][s] = {
            'gate_pass': g, 'rho_post': t['rho_post'], 'max_da': t['max_da'],
            'all_rows': {'S': p['S'], 'null1_mean': p['null1_mean'], 'diff': p['diff'], 'p': p['p'],
                         'null2_mean': p['null2_mean'], 'diff_s': ps['diff_s'], 'p_s': ps['p_s'],
                         'conditions': dict(zip(['<=min_u', '<=m_u-2sd_u', 'p<0.05', 'p_s<0.05', 'S<null2'], a))},
            'unseen': {'n': t['strata']['unseen']['n'], 'S': un['S'], 'diff': un['diff'], 'p': un['p'],
                       'null1_sd': un['null1_sd'],
                       'conditions': dict(zip(['<=min_u', '<=m_u-2sd_u', 'p<0.05', 'diff<=-0.02'], b))},
            'output_projection': {k: t['output_projection']['score'][k] for k in ('S', 'diff', 'p')},
            'data_projection': {k: t['data_projection']['score'][k] for k in ('S', 'diff', 'p')},
            'strata_reported_not_read': {k: (None if v is None else
                                             {'n': v['n'], 'diff': v['score']['diff'], 'p': v['score']['p']})
                                         for k, v in t['strata'].items()}}
    out['thresholds'] = {'all_rows': thr_all, 'unseen': thr_un}

    if any(t.get('void') for t in T):
        reading = 'VOID'                                   # 88.2 step 1: max ||da|| <= 1e-12 is void, not null
    elif not all(gate_ok):
        reading = 'NULL'
    elif all(all_ok) and all(un_ok):
        reading = 'SIGNAL'
    elif all(all_ok):
        reading = 'SEEN-ONLY'
    else:
        reading = 'NULL'
    out['reading'] = reading
    if reading == 'SIGNAL':
        aligned = lambda key: any(t[key]['score']['diff'] <= FLOOR and t[key]['score']['p'] < PMAX for t in T)
        data_al, out_al = aligned('data_projection'), aligned('output_projection')
        q = ['a trained drug-dependent named pathway layer aligns with annotated mechanism, including for unseen compounds']
        if not data_al:
            q.append("beyond the data's own target alignment")
        q.append("beyond the model's predicted signature" if not out_al
                 else 'a named readout of a predicted signature that is itself target-aligned')
        out['wording'] = {'data_projection_aligned': data_al, 'output_projection_aligned': out_al, 'sentence': q}
    return out


def main():
    D = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join('model', 'results', 'probe_moa_88_reading.json')
    T = [load(D, 't%d' % s) for s in range(3)]
    U = [load(D, 'u%d' % s) for s in range(5)]
    out = read(T, U)
    json.dump(out, open(dst, 'w'), indent=1)
    print(json.dumps({'reading': out['reading'], 'checks': out['checks']}, indent=1))
    if 'untrained' in out:
        u = out['untrained']
        print('untrained diffs %s  m_u %+.4f sd_u %.4f | unseen %s m %+.4f sd %.4f' % (
            np.round(u['diffs'], 4).tolist(), u['m_u'], u['sd_u'], np.round(u['unseen_diffs'], 4).tolist(),
            u['m_u_unseen'], u['sd_u_unseen']))
        for s, r in out['seeds'].items():
            if r.get('void'):
                print('seed %s VOID' % s)
                continue
            a, b = r['all_rows'], r['unseen']
            print('seed %s | gate %s rho %.3f | all S %.4f diff %+.4f p %.3f p_s %.3f null2 %.4f %s | unseen n %d diff '
                  '%+.4f p %.3f %s' % (s, r['gate_pass'], r['rho_post'], a['S'], a['diff'], a['p'], a['p_s'],
                                       a['null2_mean'], a['conditions'], b['n'], b['diff'], b['p'], b['conditions']))
    if 'wording' in out:
        print('WORDING', json.dumps(out['wording']))


if __name__ == '__main__':
    main()
