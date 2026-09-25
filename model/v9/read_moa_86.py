# -*- coding: utf-8 -*-
"""Apply RESULTS 86.3 mechanically to the six probe outputs. Rules copied from 86.3; the one numeric form 86 left open
(what it means for a stratum to "show it") is fixed HERE, before reading any output: the stratum meets the per-seed
conditions diff <= -0.02 and p < 0.05 on all three seeds.

    python model/v9/read_moa_86.py external/kaggle_out/moa
"""
import glob
import json
import os
import sys

D = sys.argv[1]
FLOOR, PMAX, GATE, SCOPE_MARGIN = -0.02, 0.05, 0.95, 0.01


def load(tag):
    f = glob.glob(os.path.join(D, tag, 'probe_moa_v9_*.json'))
    assert len(f) == 1, (tag, f)
    return json.load(open(f[0]))


def cond3(r):
    """The three per-seed conditions shared by trained and untrained: diff <= floor, p < 0.05, S below Null 2."""
    return r['diff'] <= FLOOR and r['p'] < PMAX and r['S'] < r['null2_mean']


def reading(per_seed_signal, per_seed_partial_extra):
    n = sum(per_seed_signal)
    if n == 3:
        return 'SIGNAL'
    if n == 2 or all(per_seed_partial_extra):
        return 'PARTIAL'
    return 'NULL'


out = {'rules': {'floor': FLOOR, 'p': PMAX, 'gate_rho_del': GATE, 'scope_margin': SCOPE_MARGIN}, 'seeds': {}}
U = {s: load('u%d' % s) for s in range(3)}
T = {s: load('r%d' % s) for s in range(3)}

# untrained control: valid only if it reads NULL under the same table (86.2 step 4)
u_sig, u_part = [], []
for s in range(3):
    u = U[s]
    g = u['scores']['gradient_readout']
    ok_gate = (not u['degeneracy']) and u['rho_del'] < GATE
    u_sig.append(ok_gate and cond3(g))
    u_part.append(ok_gate and g['p'] < PMAX and FLOOR < g['diff'] < 0)
untrained_reading = reading(u_sig, u_part)
valid = untrained_reading == 'NULL'

t_sig, t_part = [], []
for s in range(3):
    t, u = T[s], U[s]
    g, gu = t['scores']['gradient_readout'], U[s]['scores']['gradient_readout']
    ok_gate = (not t['degeneracy']) and t['rho_del'] < GATE
    d_vs_u = g['diff'] - gu['diff']
    sig = ok_gate and cond3(g) and d_vs_u <= FLOOR
    t_sig.append(sig)
    t_part.append(ok_gate and g['p'] < PMAX and FLOOR < g['diff'] < 0)
    out['seeds'][s] = {
        'trained': {'degenerate': t['degeneracy'], 'rho_raw': t['rho_raw'], 'rho_del': t['rho_del'], 'gate_pass': ok_gate,
                    'n_compounds': t['n_compounds'], 'S': g['S'], 'null1_mean': g['null1_mean'], 'diff': g['diff'],
                    'p': g['p'], 'null2_mean': g['null2_mean'], 'diff_minus_untrained': d_vs_u,
                    'output_projection_S': t['scores']['output_projection']['S'],
                    'data_projection_S': t['scores']['data_projection']['S'],
                    'strata': {k: (None if v is None else {kk: v[kk] for kk in ('S', 'diff', 'p', 'n')})
                               for k, v in t['scores'].items() if k.startswith('stratum') or k == 'landmark_tier'}},
        'untrained': {'rho_raw': u['rho_raw'], 'rho_del': u['rho_del'], 'S': gu['S'], 'diff': gu['diff'], 'p': gu['p'],
                      'null2_mean': gu['null2_mean']},
        'signal_conditions_met': sig}
trained_reading = reading(t_sig, t_part) if valid else 'NOT READ (untrained control is not NULL)'
out['untrained_reading'] = untrained_reading
out['probe_valid'] = valid
out['reading'] = trained_reading if valid else 'INVALID'

if valid and trained_reading == 'SIGNAL':
    internal = all(T[s]['scores']['gradient_readout']['S'] < T[s]['scores']['output_projection']['S'] - SCOPE_MARGIN
                   for s in range(3))
    unseen = all((T[s]['scores']['stratum_unseen'] or {}).get('diff', 1) <= FLOOR and
                 (T[s]['scores']['stratum_unseen'] or {}).get('p', 1) < PMAX for s in range(3))
    notresp = all((T[s]['scores']['stratum_target_not_responsive'] or {}).get('diff', 1) <= FLOOR and
                  (T[s]['scores']['stratum_target_not_responsive'] or {}).get('p', 1) < PMAX for s in range(3))
    data_enriched = all(T[s]['scores']['data_projection']['diff'] <= FLOOR and T[s]['scores']['data_projection']['p'] < PMAX
                        for s in range(3))
    out['scope'] = {'model_internal': internal, 'unseen_compound_stratum_shows_it': unseen,
                    'non_responsive_target_stratum_shows_it': notresp, 'data_projection_enriched': data_enriched,
                    'mechanism_claim_licensed': unseen and notresp}
json.dump(out, open(os.path.join('model', 'results', 'probe_moa_v9_reading_86.json'), 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('untrained_reading', 'probe_valid', 'reading')}, indent=1))
for s in range(3):
    t, u = out['seeds'][s]['trained'], out['seeds'][s]['untrained']
    print('seed %d | trained: rho_raw %.3f rho_del %.3f gate %s n %d | S %.4f null1 %.4f diff %+.4f p %.3f null2 %.4f | '
          'diff-untrained %+.4f | outproj S %.4f dataproj S %.4f' % (
              s, t['rho_raw'], t['rho_del'], t['gate_pass'], t['n_compounds'], t['S'], t['null1_mean'], t['diff'], t['p'],
              t['null2_mean'], t['diff_minus_untrained'], t['output_projection_S'], t['data_projection_S']))
    print('        | untrained: rho_raw %.3f rho_del %.3f | S %.4f diff %+.4f p %.3f null2 %.4f' % (
        u['rho_raw'], u['rho_del'], u['S'], u['diff'], u['p'], u['null2_mean']))
    for k, v in t['strata'].items():
        if v:
            print('        |   %-32s n %4d S %.4f diff %+.4f p %.3f' % (k, v['n'], v['S'], v['diff'], v['p']))
if 'scope' in out:
    print('SCOPE', json.dumps(out['scope']))
