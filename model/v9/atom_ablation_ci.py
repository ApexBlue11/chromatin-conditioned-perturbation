# -*- coding: utf-8 -*-
"""C4 from adversarial review 003: put an INTERVAL and a CROSS-SEED SPREAD on the atom-token ablation.

Why
---
`probe_v9.py` reports the atom-token ablation as -0.00671 / -0.02549 / -0.02187 (unseen cell / compound /
both) and contains NO bootstrap and NO interval -- `grep -c "bootstrap|ci95|percentile"` returns 0. Those
figures come from ONE checkpoint, ONE seed, on 480 signatures per regime (and 480 is the whole split after
the strength and coverage filters, not a cap: --n_eval defaults to 1500).

RESULTS 47 was about to commit ~5.6 GPU-h to explain that number. Review 003 C4: compute its uncertainty
first, from checkpoints already on disk. Inference only, no training, no quota.

Method
------
The reported statistic is a difference of MEDIANS: median(row Pearson | intact) - median(row Pearson |
atoms ablated to the chunk mean). So the bootstrap resamples ROWS and recomputes both medians, which is
the interval for the statistic actually quoted. The paired-mean version is reported alongside because it
is the more stable estimator and the two disagreeing is itself informative.

Chunking is replicated exactly (same --batch, same rng seed, same filters) because ablate_to_mean uses the
CHUNK mean over dim 0 -- a different batch size is a different ablation.

Run:  python model/v9/atom_ablation_ci.py --ckpt_dir external/v9_checkpoints
"""
import argparse, csv, io, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
os.chdir(HERE)

from config_v9 import V9Config, V9DataConfig                     # noqa: E402
from model_v9 import LincsV9                                       # noqa: E402
from data_v9 import LincsV9Dataset, collate_v9, build_splits       # noqa: E402
from train_v9_gpu import resolve_v9                                # noqa: E402


def pearson_rows(a, b):
    """Byte-for-byte the definition in probe_v9.py, so these intervals attach to THAT number."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def rows_for(model, chunks, key):
    """Per-row Pearson on the delta, intact and with `key` ablated to its CHUNK mean. Paired by row."""
    r0, r1, dymax = [], [], 0.0
    with torch.no_grad():
        for c in chunks:
            tgt = c['y_delta'].float().cpu().numpy()
            y0 = model(c)['delta']
            ref = c[key]
            patched = dict(c)
            patched[key] = ref.mean(dim=0, keepdim=True).expand_as(ref)
            y1 = model(patched)['delta']
            dymax = max(dymax, float((y0 - y1).abs().max()))
            r0.append(pearson_rows(y0.float().cpu().numpy(), tgt))
            r1.append(pearson_rows(y1.float().cpu().numpy(), tgt))
    return np.concatenate(r0), np.concatenate(r1), dymax


def boot(r0, r1, n=20000, seed=0):
    """Bootstrap the DIFFERENCE OF MEDIANS (the quoted statistic) and the paired mean."""
    rng = np.random.default_rng(seed)
    ok = ~(np.isnan(r0) | np.isnan(r1))
    a, b = r0[ok], r1[ok]
    idx = rng.integers(0, len(a), size=(n, len(a)))
    dmed = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    dmean = (a - b)[idx].mean(axis=1)
    return {
        'n_rows': int(len(a)),
        'd_median': round(float(np.median(a) - np.median(b)), 5),
        'd_median_ci95': [round(float(np.percentile(dmed, 2.5)), 5),
                          round(float(np.percentile(dmed, 97.5)), 5)],
        'd_paired_mean': round(float((a - b).mean()), 5),
        'd_paired_mean_ci95': [round(float(np.percentile(dmean, 2.5)), 5),
                               round(float(np.percentile(dmean, 97.5)), 5)],
        'frac_rows_intact_better': round(float((a > b).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default=os.path.join(ROOT, 'external', 'v9_checkpoints'))
    ap.add_argument('--key', default='atoms')
    ap.add_argument('--batch', type=int, default=48, help='MUST match probe_v9 (48) or it is a different ablation')
    ap.add_argument('--n_eval', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0, help='row-selection seed; probe_v9 uses 0')
    ap.add_argument('--n_boot', type=int, default=20000)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cks = sorted(f for f in os.listdir(a.ckpt_dir) if f.endswith('.pt') and 'fold0' in f)
    # one checkpoint per distinct seed, preferring the r-series (the 12-epoch 3-seed round, RESULTS 33)
    by_seed = {}
    for f in cks:
        s = f.split('seed')[-1].split('.')[0]
        if s not in by_seed or f.startswith('r'):
            by_seed[s] = f
    print('device %s | checkpoints: %s' % (dev, ', '.join('seed%s=%s' % kv for kv in sorted(by_seed.items()))))

    dc = resolve_v9(V9DataConfig())
    dc.cache_in_ram = False
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M, ppi, gv = np.load(R(dc.m_pathway_path)), np.load(R(dc.ppi_v9_path)), np.load(R(dc.gene_vec_path))
    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)

    out = {'key': a.key, 'batch': a.batch, 'n_boot': a.n_boot, 'device': dev,
           'note': 'C4 of adversarial review 003. probe_v9.py emits no interval; these are the missing ones.',
           'seeds': {}}

    for sd, fn in sorted(by_seed.items()):
        ck = torch.load(os.path.join(a.ckpt_dir, fn), map_location='cpu', weights_only=False)
        cfg = V9Config(**{k: v for k, v in ck['cfg'].items() if k in V9Config.__dataclass_fields__})
        torch.manual_seed(0)
        model = LincsV9(cfg, M, ppi, gv)
        model.load_state_dict(ck['model'])
        model = model.to(dev).eval()
        print('\n=== %s (epoch %s) ===' % (fn, ck.get('epoch')), flush=True)
        rec = {}
        rng = np.random.default_rng(a.seed)
        for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                          ('unseen_both', 'test_coldboth')]:
            idx = sp[key][ds.strength[sp[key]] >= dc.eval_min_strength]
            idx = idx[ds.has_l3[idx]]
            if len(idx) < 300:
                continue
            idx = np.sort(rng.choice(idx, min(a.n_eval, len(idx)), replace=False))
            chunks = [collate_v9([ds[i] for i in idx[s:s + a.batch]])
                      for s in range(0, len(idx), a.batch)]
            chunks = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in c.items()} for c in chunks]
            r0, r1, dymax = rows_for(model, chunks, a.key)
            st = boot(r0, r1, a.n_boot, seed=0)
            st['dY_max'] = round(dymax, 4)
            st['fired'] = bool(dymax > 1e-5)
            rec[name] = st
            sign = 'SPANS ZERO' if st['d_median_ci95'][0] <= 0 <= st['d_median_ci95'][1] else 'excludes 0'
            print('  %-17s n=%4d  d_median %+.5f  CI95 [%+.5f, %+.5f]  %-10s | paired_mean %+.5f '
                  'CI [%+.5f, %+.5f] | dY_max %.3f'
                  % (name, st['n_rows'], st['d_median'], st['d_median_ci95'][0], st['d_median_ci95'][1],
                     sign, st['d_paired_mean'], st['d_paired_mean_ci95'][0], st['d_paired_mean_ci95'][1],
                     st['dY_max']), flush=True)
        out['seeds'][sd] = {'ckpt': fn, 'epoch': int(ck.get('epoch', -1)), 'splits': rec}
        del model
        if dev == 'cuda':
            torch.cuda.empty_cache()

    # ---- cross-seed spread: is the effect stable across training runs? ----
    print('\n=== CROSS-SEED (the question probe_v9 never asked) ===')
    out['cross_seed'] = {}
    for name in ('unseen_cell', 'unseen_compound', 'unseen_both'):
        vals = [out['seeds'][s]['splits'][name]['d_median']
                for s in out['seeds'] if name in out['seeds'][s]['splits']]
        if len(vals) < 2:
            continue
        w = [out['seeds'][s]['splits'][name]['d_median_ci95'] for s in out['seeds']
             if name in out['seeds'][s]['splits']]
        mean_w = float(np.mean([x[1] - x[0] for x in w]))
        out['cross_seed'][name] = {
            'per_seed': [round(v, 5) for v in vals], 'range': round(float(max(vals) - min(vals)), 5),
            'sd': round(float(np.std(vals, ddof=1)), 5), 'mean_ci_width': round(mean_w, 5),
            'range_exceeds_ci_width': bool((max(vals) - min(vals)) > mean_w),
        }
        cs = out['cross_seed'][name]
        print('  %-17s per-seed %s  range %.5f  sd %.5f  vs mean CI width %.5f  -> %s'
              % (name, cs['per_seed'], cs['range'], cs['sd'], cs['mean_ci_width'],
                 'RANGE EXCEEDS ITS OWN CI WIDTH' if cs['range_exceeds_ci_width'] else 'stable'))

    dst = os.path.join(ROOT, 'model', 'results', 'v9_atom_ablation_CI.json')
    with io.open(dst, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(out, indent=2) + '\n')
    print('\nwrote %s' % dst)


if __name__ == '__main__':
    main()
