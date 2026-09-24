# -*- coding: utf-8 -*-
"""RESULTS 80.3 as amended in 80.5, applied offline to the four gradient dumps v6 left behind when its own comparison
stopped. Criterion 1's loss clause is NOT evaluated here: the losses were in the per-probe JSON the crash lost.

Order, as committed: structure (extras exactly zero; compare on the reference key set) -> the fp32 floor rule
(Amendment C) -> criterion 1 (fp32, gradients) -> finiteness equality and the void rule (Amendment B) -> criterion 2
(fp16, 32 rows) -> reported-only 128-row comparisons. Timing and memory come from the v6 log (80.6)."""
import io
import json
import os

import torch

D = r'C:\Projects\LINCS\external\kaggle_out\cc1_v6\dp_shared'
OUT = r'C:\Projects\LINCS\model\results\xpert_dp_v6_offline.json'
MODES = ('single_ckpt', 'single_ckpt_repeat', 'dp', 'dp_ckpt')
EXPECTED_UNUSED = sorted(['attnEncoder_trt.crossEncoders.0.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.0.LayerNorm.gamma',
                          'attnEncoder_trt.crossEncoders.1.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.1.LayerNorm.gamma',
                          'cell_emb.linear.bias', 'cell_emb.linear.weight', 'ctl_fc.0.bias', 'ctl_fc.0.weight',
                          'ctl_fc.3.bias', 'ctl_fc.3.weight'])
G = {m: torch.load(os.path.join(D, m + '_grads.pt')) for m in MODES}
out = {'source': 'v6 gradient dumps', 'criteria': 'RESULTS 80.3 as amended by 80.5', 'tags': {}}


def finite(g):
    return all(bool(torch.isfinite(v).all()) for v in g.values())


def rel_l2(ga, gb, keys):
    num = sum(float(((ga[n] - gb[n]) ** 2).sum()) for n in keys)
    den = sum(float((gb[n] ** 2).sum()) for n in keys)
    return (num / den) ** 0.5


# ---- structure ------------------------------------------------------------------------------------------------------
struct = {}
for tag, ref in G['single_ckpt'].items():
    rk = set(ref)
    for m in MODES[1:]:
        extra = sorted(set(G[m][tag]) - rk)
        struct['%s/%s' % (tag, m)] = {
            'missing_from_variant': len(rk - set(G[m][tag])),
            'extra': len(extra),
            'extra_is_expected_unused_set': extra == ([] if m == 'single_ckpt_repeat' else EXPECTED_UNUSED),
            'extra_all_exactly_zero': all(float(G[m][tag][n].abs().max()) == 0.0 for n in extra)}
out['structure'] = struct
out['structure_pass'] = all(v['missing_from_variant'] == 0 and v['extra_is_expected_unused_set']
                            and v['extra_all_exactly_zero'] for v in struct.values())

# ---- per tag -------------------------------------------------------------------------------------------------------
for tag, ref in G['single_ckpt'].items():
    keys = sorted(ref)
    t = {'reference_finite': finite(ref)}
    for m in MODES[1:]:
        g = {n: G[m][tag][n] for n in keys}
        t[m] = {'finite': finite(g), 'rel_l2_vs_single': rel_l2(g, ref, keys) if finite(g) and t['reference_finite'] else None}
    out['tags'][tag] = t

fp32_floor = max(out['tags'][t]['single_ckpt_repeat']['rel_l2_vs_single'] for t in ('fp32_e0', 'fp32_e70'))
out['amendment_C_fp32_floor'] = fp32_floor
out['amendment_C_criterion1_discriminates'] = fp32_floor < 3e-6

prec = {}
for e in ('e0', 'e70'):
    a, b = G['single_ckpt']['fp16_' + e], G['single_ckpt']['fp32_' + e]
    keys = sorted(b)
    prec[e] = rel_l2(a, b, keys) if finite(a) and finite(b) else None
out['precision_noise_fp16_vs_fp32'] = prec

verdict = {}
for v in ('dp', 'dp_ckpt'):
    c1 = all(out['tags']['fp32_' + e][v]['rel_l2_vs_single'] is not None
             and out['tags']['fp32_' + e][v]['rel_l2_vs_single'] < 1e-5 for e in ('e0', 'e70'))
    fin = all(out['tags'][t][v]['finite'] == out['tags'][t]['reference_finite'] for t in out['tags'])
    c2 = {}
    for e in ('e0', 'e70'):
        tg = 'fp16_' + e
        if not out['tags'][tg]['reference_finite'] or prec[e] is None:
            c2[tg] = 'VOID (reference non-finite, Amendment B)'
        else:
            c2[tg] = out['tags'][tg][v]['rel_l2_vs_single'] <= prec[e]
    verdict[v] = {'criterion1_fp32_gradients': c1 if out['amendment_C_criterion1_discriminates'] else 'NOT READ (floor, Amendment C)',
                  'criterion1_loss_clause': 'NOT EVALUABLE on v6 (losses lost with the crash)',
                  'finiteness_equals_reference': fin, 'criterion2_fp16': c2,
                  'structure': out['structure_pass']}
out['verdict'] = verdict
print(json.dumps({k: out[k] for k in ('structure_pass', 'amendment_C_fp32_floor', 'amendment_C_criterion1_discriminates',
                                      'precision_noise_fp16_vs_fp32', 'verdict')}, indent=1))
for tag, t in out['tags'].items():
    print('%-13s ref finite %-5s | ' % (tag, t['reference_finite']) + ' | '.join(
        '%s %s %s' % (m, 'finite' if t[m]['finite'] else 'NONFINITE',
                      '%.2e' % t[m]['rel_l2_vs_single'] if t[m]['rel_l2_vs_single'] is not None else '-') for m in MODES[1:]))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(json.dumps(out, indent=2) + '\n')
print('wrote', OUT)
