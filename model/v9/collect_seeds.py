# -*- coding: utf-8 -*-
"""
Collect the v9 seed runs and report **mean [min, max]** — the only form this project reports a number in.

Method rule 3: seed sd here reaches 0.0232, so a 2-sd band is ±0.046 and a single run is not a result.
Six architectures (v3→v7) produced no difference distinguishable from that band, and the encoder A/B made
seven, so this script refuses to print a comparison it cannot separate from seed noise: any difference
whose magnitude is inside the seed range is labelled NO DIFFERENCE rather than reported as a gain.

It also refuses to average seeds that did not run the same schedule. A run truncated by the budget guard
stops at whatever epoch it reached, and averaging a 6-epoch seed with a 4-epoch seed produces a number that
means nothing.

    python model/v9/collect_seeds.py --dir <downloaded kernel outputs>
"""
import os, json, glob, argparse

import numpy as np

REF = {  # what v9 has to be read against
    'v7_no_aux_l5': {'unseen_cell': 0.4549, 'unseen_compound': 0.4985, 'unseen_both': 0.4825},
    'seed_sd_max': 0.0232,
    'two_sd_band': 0.046,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='directory holding metrics_v9_fold0_seed*.json')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, '**', 'metrics_v9_fold*_seed*.json'), recursive=True))
    if not files:
        raise SystemExit(f'no metrics_v9_fold*_seed*.json under {a.dir}')
    runs = {}
    for f in files:
        hist = json.load(open(f))
        if not hist:
            continue
        seed = os.path.basename(f).split('seed')[1].split('.')[0]
        runs[seed] = hist
    print(f'{len(runs)} seed(s): ' + ', '.join(f'seed{s} ({len(h)} epochs)' for s, h in sorted(runs.items())))

    epochs = {len(h) for h in runs.values()}
    if len(epochs) != 1:
        print('\n!! SEEDS DID NOT RUN THE SAME SCHEDULE: ' +
              ', '.join(f'seed{s}={len(h)}' for s, h in sorted(runs.items())) +
              '\n!! Averaging these would produce a number that means nothing. Truncating all seeds to the '
              'shortest schedule; re-run if that is too few epochs to be worth reporting.')
    n_ep = min(len(h) for h in runs.values())
    last = {s: h[n_ep - 1] for s, h in runs.items()}

    print(f'\nreporting at epoch {n_ep - 1} (0-indexed), the last epoch every seed completed')
    out = {'n_seeds': len(runs), 'epoch': n_ep - 1, 'splits': {}}
    hdr = '  %-16s %-8s %26s %26s %26s' % ('split', 'metric', 'mean [min, max]', '', '')
    for split in ['unseen_cell', 'unseen_compound', 'unseen_both']:
        if not all(split in r for r in last.values()):
            continue
        print(f'\n--- {split} ---')
        rec = {}
        for metric in ['delta', 'abs', 'l5', 'copy_ctl_abs']:
            v = [r[split].get(metric) for r in last.values() if r[split].get(metric) is not None]
            if not v:
                continue
            rec[metric] = {'mean': round(float(np.mean(v)), 4), 'min': round(float(min(v)), 4),
                           'max': round(float(max(v)), 4), 'range': round(float(max(v) - min(v)), 4),
                           'seeds': v}
            print(f'  {metric:14s} {np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]  range {max(v)-min(v):.4f}')
        if 'abs' in rec and 'copy_ctl_abs' in rec:
            va = rec['abs']['mean'] - rec['copy_ctl_abs']['mean']
            rec['abs_value_added_over_copying_the_control'] = round(va, 4)
            print(f'  {"value added":14s} {va:+.4f}   (absolute minus copy-the-control; XPert\'s released '
                  f'predictions add +0.060)')
        if 'l5' in rec and split in REF['v7_no_aux_l5']:
            d = rec['l5']['mean'] - REF['v7_no_aux_l5'][split]
            band = max(rec['l5']['range'], REF['two_sd_band'])
            verdict = 'NO DIFFERENCE (inside the seed band)' if abs(d) <= band else \
                ('v9 ahead' if d > 0 else 'v7 --no_aux ahead')
            rec['vs_v7_no_aux_on_l5'] = {'diff': round(d, 4), 'band': round(band, 4), 'verdict': verdict}
            print(f'  vs v7 --no_aux on the SAME target (l5): {REF["v7_no_aux_l5"][split]:.4f} -> '
                  f'{rec["l5"]["mean"]:.4f}  ({d:+.4f})  {verdict}')
        out['splits'][split] = rec

    print('\n' + '=' * 100)
    print('Only the l5 row is comparable with v3-v7: those versions never had a Level-3 target. The delta')
    print('and abs rows are v9\'s own conventions and must be read against their nulls, not against history.')
    print('The absolute number means nothing without copy-the-control beside it [RESULTS 23, 27.2, 28.2].')
    dst = a.out or os.path.join(a.dir, 'v9_seeds_summary.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
