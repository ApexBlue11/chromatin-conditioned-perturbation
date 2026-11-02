"""Stage 2's job list (RESULTS 103.6 / 103.7), split over six Kaggle sessions (two per account), injection jobs first.

    python model/cellrep/stage2_jobs.py --out model/cellrep/stage2_jobs.json
"""
import argparse, json

CORE = ('PC3', 'MCF7', 'A375', 'A549', 'HEPG2', 'HT29')
SEEDS = (0, 1, 2)
SESSIONS = ('apexblue-a', 'apexblue-b', 'wwhhhhhhhh-a', 'wwhhhhhhhh-b', 'suryababu179-a', 'suryababu179-b')
COST = {'M1': 1, 'M2': 1, 'M3': 10, 'M4': 10, 'M5': 10}


def jobs():
    loco = [f'loco:{c}' for c in CORE]
    inj, real = [], []
    for name in ('M1', 'M2'):                                           # rung 3, MLP family: both shapes, three levels
        for shape in ('dense', 'graph'):
            for share in (0.02, 0.05, 0.10):
                inj += [{'model': name, 'setting': s, 'seed': k, 'inject': {'shape': shape, 'share': share}} for s in loco for k in SEEDS]
        inj += [{'model': name, 'setting': s, 'seed': k, 'inject': {'shape': 'dense', 'share': 0.05, 'residualise': False}}
                for s in loco for k in SEEDS]                           # raw ACT, reported
    for name in ('M3', 'M4'):                                           # rung 3, graph family: both shapes at 5 %
        for shape in ('dense', 'graph'):
            inj += [{'model': name, 'setting': s, 'seed': k, 'inject': {'shape': shape, 'share': 0.05}} for s in loco for k in SEEDS]
    for name in ('M1', 'M2', 'M3', 'M4', 'M5'):                          # the real targets
        real += [{'model': name, 'setting': s, 'seed': k} for s in loco + ['cold'] for k in SEEDS]
    real += [{'model': 'M3', 'setting': s, 'seed': k, 'K': 4} for s in loco + ['cold'] for k in SEEDS]   # K = 4 sensitivity
    return inj, real


def split(js):
    """Greedy balance by cost; within each session the injection jobs keep their place before the real ones."""
    load = {s: 0 for s in SESSIONS}
    out = {s: [] for s in SESSIONS}
    for j in sorted(js, key=lambda j: -COST[j['model']]):
        s = min(load, key=load.get)
        out[s].append(j)
        load[s] += COST[j['model']]
    for s in SESSIONS:
        out[s].sort(key=lambda j: ('inject' not in j, -COST[j['model']]))
    return out, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    inj, real = jobs()
    out, load = split(inj + real)
    json.dump({'n_inject': len(inj), 'n_real': len(real), 'sessions': out, 'cost': load}, open(a.out, 'w'), indent=0)
    print({'n_inject': len(inj), 'n_real': len(real), 'cost_per_session': load})


if __name__ == '__main__':
    main()
