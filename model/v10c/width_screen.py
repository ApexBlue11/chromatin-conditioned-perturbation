"""The d = 384 width screen: as registered (100.4) and as read (100.9, post hoc, fixed before d384 was pushed).

  100.4  d384 on s0 - d256 on s0 >= max(0.003, 2 s_on), s_on over the three d256 on seeds -> 384, else 256
  100.9  comparator = mean of the d256 on seeds whose drug path trained, s_on over those seeds; if d384's own drug path
         did not train, the screen is uninformative and T-C uses 256

    python model/v10c/width_screen.py --trained model/results/v10c/tb_trained_check_*.json --out model/results/v10c/width_screen.json
"""
import argparse, json, os

import numpy as np

LEDGER = 'model/results/v10c/tb_ledger.jsonl'


def screen(r384, r256, trained384, trained256):
    """r256, trained256: dicts seed -> contrast r / drug-path-trained flag (the d256 on seeds)."""
    seeds = sorted(r256)
    s_all = float(np.std([r256[s] for s in seeds], ddof=1))
    reg_delta = r384 - r256[0]
    reg_thr = max(0.003, 2 * s_all)
    registered = {'delta': reg_delta, 'threshold': reg_thr, 'width': 384 if reg_delta >= reg_thr else 256,
                  'note': 'comparator is d256 on s0, whose drug path did not train; can err only toward 256 (review 090 C1)'}
    ok = [s for s in seeds if trained256[s]]
    if not trained384:
        read = {'width': 256, 'reason': "d384's drug path did not train: the screen is uninformative (100.9)"}
    elif len(ok) < 2:
        read = {'width': 256, 'reason': 'fewer than two trained d256 on seeds (100.9)'}
    else:
        comp = float(np.mean([r256[s] for s in ok]))
        thr = max(0.003, 2 * float(np.std([r256[s] for s in ok], ddof=1)))
        read = {'comparator_seeds': ok, 'comparator': comp, 'delta': r384 - comp, 'threshold': thr,
                'width': 384 if r384 - comp >= thr else 256}
    return {'registered_100_4': registered, 'read_100_9': read, 'T_C_width': read['width']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trained', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- the screen is read once')
    led = {}
    for line in open(LEDGER, encoding='utf-8'):
        e = json.loads(line)
        led[e['run']] = e
    flags = {}
    for p in a.trained:
        for r in json.load(open(p, encoding='utf-8'))['runs']:
            flags[r['checkpoint'].replace('v10c_', '').replace('.pt', '')] = r['drug_path_trained']
    t384 = 'cold_cell_1_dev_on_d384_s0'
    r256 = {s: led[f'cold_cell_1_dev_on_d256_s{s}']['contrast_r_row_mean'] for s in (0, 1, 2)}
    tr256 = {s: flags[f'cold_cell_1_dev_on_d256_s{s}'] for s in (0, 1, 2)}
    rec = screen(led[t384]['contrast_r_row_mean'], r256, flags[t384], tr256)
    rec.update({'d384_contrast_r': led[t384]['contrast_r_row_mean'], 'd384_drug_path_trained': flags[t384],
                'd256_on_contrast_r': r256, 'd256_on_trained': tr256})
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
