# -*- coding: utf-8 -*-
"""
THE GATE on the Level-3 migration (V8_PLAN §1, step 2).

Migrating to Level 3 buys a plate-matched control as INPUT but costs target reliability: a plain mean of
replicate deltas measures 0.144 (0.243 top quartile) against Level-5 MODZ's 0.127 / 0.509-0.619 [RESULTS 25].
The proposed mitigation is to aggregate replicates the way cmap's MODZ does -- weighting each replicate by
how much it agrees with the others -- instead of averaging flat.

This measures whether that actually helps, as a WITHIN-CONDITION comparison so the two methods see identical
wells: split each condition's wells into two disjoint PLATE groups, aggregate each half by both methods, and
correlate the halves. Different plates only, since same-plate wells share a control vector.

Decision rule, fixed in advance:
    MODZ split-half reliability MUST beat the plain mean's 0.144 / 0.243.
    If it does not, we are trading a measurably better target for a measurably better input, and that trade
    needs its own decision rather than an assumption.
"""
import os, csv, json, argparse, random
from collections import defaultdict

import numpy as np
import h5py

ROOT = r'C:\Projects\LINCS'
GCTX = os.path.join(ROOT, 'level3 files', 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
INST = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_inst_info.txt',
                    'GSE92742_Broad_LINCS_inst_info.txt')
OUT = os.path.join(ROOT, 'phase2_assembly', 'outputs', 'level3')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_level3 import modz_weights, landmark_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_conditions', type=int, default=1500)
    ap.add_argument('--min_wells', type=int, default=6)
    ap.add_argument('--min_ctl', type=int, default=3)
    ap.add_argument('--max_ctl', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    f = h5py.File(GCTX, 'r')
    gi, _ = landmark_rows(f)
    col_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/COL/id'][:]]
    col_pos = {c: i for i, c in enumerate(col_ids)}

    trt, ctl_by_plate = {}, defaultdict(list)
    with open(INST, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['pert_type'] == 'trt_cp':
                trt[r['inst_id']] = (r['rna_plate'], r['cell_id'], r['pert_id'],
                                     r['pert_dose'], r['pert_time'])
            elif r['pert_type'] == 'ctl_vehicle':
                ctl_by_plate[r['rna_plate']].append(r['inst_id'])
    ctl_by_plate = {p: [i for i in v if i in col_pos][:a.max_ctl] for p, v in ctl_by_plate.items()}
    ctl_by_plate = {p: v for p, v in ctl_by_plate.items() if len(v) >= a.min_ctl}

    need, conditions = set(), defaultdict(list)
    for inst, (plate, cell, pert, dose, ptime) in trt.items():
        if plate in ctl_by_plate and inst in col_pos:
            conditions[(cell, pert, dose, ptime)].append((plate, inst))
            need.add(inst)
    for p, v in ctl_by_plate.items():
        need.update(v)
    need_rows = sorted(col_pos[i] for i in need)
    slot = {r: n for n, r in enumerate(need_rows)}
    cache = np.memmap(os.path.join(OUT, 'wells_cache.dat'), dtype=np.float32, mode='r',
                      shape=(len(need_rows), len(gi)))
    print("cache %s | conditions %d" % (str(cache.shape), len(conditions)))

    # conditions with enough wells spread over >=2 plates
    usable = {}
    for k, v in conditions.items():
        plates = defaultdict(list)
        for p, i in v:
            plates[p].append(i)
        if len(plates) >= 2 and len(v) >= a.min_wells:
            usable[k] = plates
    print("conditions with >=%d wells on >=2 plates: %d" % (a.min_wells, len(usable)))

    rnd = random.Random(a.seed)
    chosen = rnd.sample(sorted(usable), min(a.n_conditions, len(usable)))

    plate_med = {}
    def med(p):
        if p not in plate_med:
            rows = [slot[col_pos[i]] for i in ctl_by_plate[p]]
            plate_med[p] = np.median(np.asarray(cache[rows], np.float64), axis=0)
        return plate_med[p]

    r_mean, r_modz, strengths = [], [], []
    for n, k in enumerate(chosen):
        plates = usable[k]
        pl = sorted(plates)
        half = len(pl) // 2
        gA, gB = pl[:half], pl[half:2 * half] if half else []
        if not gA or not gB:
            continue
        def deltas(group):
            D = []
            for p in group:
                m = med(p)
                for i in plates[p]:
                    D.append(np.asarray(cache[slot[col_pos[i]]], np.float64) - m)
            return np.array(D)
        DA, DB = deltas(gA), deltas(gB)
        if len(DA) < 1 or len(DB) < 1:
            continue
        # plain mean
        A1, B1 = DA.mean(0), DB.mean(0)
        # MODZ-weighted
        wA, wB = modz_weights(DA), modz_weights(DB)
        A2, B2 = (wA[:, None] * DA).sum(0), (wB[:, None] * DB).sum(0)
        if min(A1.std(), B1.std(), A2.std(), B2.std()) <= 0:
            continue
        r_mean.append(np.corrcoef(A1, B1)[0, 1])
        r_modz.append(np.corrcoef(A2, B2)[0, 1])
        strengths.append((np.abs(A2).mean() + np.abs(B2).mean()) / 2)
        if (n + 1) % 200 == 0:
            print("  %d/%d  n=%d" % (n + 1, len(chosen), len(r_mean)), flush=True)

    r_mean = np.array(r_mean); r_modz = np.array(r_modz); strengths = np.array(strengths)
    print("\n" + "=" * 78)
    print("GATE: Level-3 aggregation, split-half over disjoint plate groups (n=%d conditions)" % len(r_mean))
    print("=" * 78)
    print("  %-34s %9s %9s" % ("", "mean r", "median"))
    print("  %-34s %9.4f %9.4f" % ("plain mean of replicate deltas", r_mean.mean(), np.median(r_mean)))
    print("  %-34s %9.4f %9.4f" % ("MODZ-weighted (this migration)", r_modz.mean(), np.median(r_modz)))
    print("  %-34s %+9.4f" % ("MODZ advantage", r_modz.mean() - r_mean.mean()))
    print()
    qs = np.quantile(strengths, [0, .25, .5, .75, 1.0])
    print("  %26s %6s %10s %10s" % ("|delta| bin", "n", "mean-agg", "MODZ-agg"))
    rows = []
    for i in range(4):
        m = (strengths >= qs[i]) & (strengths <= qs[i + 1])
        print("  %11.3f-%-13.3f %6d %10.4f %10.4f" % (qs[i], qs[i + 1], m.sum(),
                                                      r_mean[m].mean(), r_modz[m].mean()))
        rows.append({'lo': float(qs[i]), 'hi': float(qs[i + 1]), 'n': int(m.sum()),
                     'plain_mean': float(r_mean[m].mean()), 'modz': float(r_modz[m].mean())})
    print()
    print("  REFERENCE  Level-3 plain mean [RESULTS 25]: 0.1441 overall / 0.2429 top quartile")
    print("  REFERENCE  Level-5 MODZ       [6.1]       : 0.127 all / 0.509-0.619 reproducible")
    top_modz = rows[-1]['modz']
    verdict = ("PASS -- MODZ aggregation improves the Level-3 target"
               if (r_modz.mean() > 0.1441 and top_modz > 0.2429) else
               "FAIL -- MODZ aggregation does NOT beat the plain mean; the migration trades a better "
               "target for a better input and needs an explicit decision")
    print("\n  VERDICT: %s" % verdict)
    json.dump({'n': int(len(r_mean)), 'plain_mean': float(r_mean.mean()), 'modz': float(r_modz.mean()),
               'modz_advantage': float(r_modz.mean() - r_mean.mean()), 'by_strength': rows,
               'reference_l3_plain': [0.1441, 0.2429], 'reference_l5_modz': [0.127, 0.509, 0.619],
               'verdict': verdict},
              open(os.path.join(ROOT, 'model', 'results', 'level3_gate_modz.json'), 'w'), indent=2)
    print("wrote model/results/level3_gate_modz.json")


if __name__ == '__main__':
    main()
