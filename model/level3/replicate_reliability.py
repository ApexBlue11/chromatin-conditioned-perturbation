# -*- coding: utf-8 -*-
"""
THE decisive data-vs-model measurement: how replicable is a LEVEL-3 delta compared with the LEVEL-5 MODZ
z-score we train on?

Why it matters. On 153,478 matched conditions the two targets agree at only ~0.64 on our reproducible
stratum, and that agreement sits at 83-92 % of the ceiling implied by OUR OWN measured replicate reliability
(RESULTS 24). That is consistent with the two targets measuring the same biology while ours is the noisier
measurement -- but it was INFERRED from our reliability numbers, not measured on Level 3 directly. This
measures it directly.

Method -- the honest construction of a Level-3 delta:
  * a well's delta = its expression MINUS the median of the ctl_vehicle (DMSO) wells ON THE SAME PLATE.
    Plate-matched, which is the whole point of Level 3 and the thing our Level-5 data cannot give us.
  * replicates = wells sharing (cell_id, pert_id, pert_dose, pert_time) but on DIFFERENT plates. Same-plate
    pairs would share the identical control vector and inflate agreement, so they are excluded.
  * reliability = pairwise Pearson between replicate deltas, stratified by delta strength so it is directly
    comparable to our Level-5 curve (mean|Y| 0.48/0.65/0.80/1.11/1.63/2.50 -> rho
    .074/.088/.136/.365/.646/.751, claim 6.1).

If Level-3 deltas are markedly more replicable than Level-5 z-scores at matched strength, then every model
trained on Level 3 is solving an intrinsically easier problem, and our headline numbers carry a noise
penalty those models never pay.
"""
import os, csv, json, argparse, random
from collections import defaultdict

import numpy as np
import h5py

ROOT = r'C:\Projects\LINCS'
GCTX = os.path.join(ROOT, 'level3 files', 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
INST = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_inst_info.txt',
                    'GSE92742_Broad_LINCS_inst_info.txt')
GENE = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_gene_info.txt',
                    'GSE92742_Broad_LINCS_gene_info.txt')


def landmark_rows(f):
    """GCTX gene-axis indices of the 978 landmarks."""
    lm = []
    with open(GENE, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['pr_is_lm'] == '1':
                lm.append(r['pr_gene_id'])
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/ROW/id'][:]]
    pos = {g: i for i, g in enumerate(ids)}
    idx = np.array(sorted(pos[g] for g in lm if g in pos))
    print("landmark genes located in GCTX: %d/978" % len(idx))
    return idx


def load_meta():
    trt, ctl_by_plate = {}, defaultdict(list)
    with open(INST, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            t = r['pert_type']
            if t == 'trt_cp':
                trt[r['inst_id']] = (r['rna_plate'], r['cell_id'], r['pert_id'],
                                     r['pert_dose'], r['pert_time'])
            elif t == 'ctl_vehicle':
                ctl_by_plate[r['rna_plate']].append(r['inst_id'])
    print("trt_cp wells: %d | plates with DMSO controls: %d" % (len(trt), len(ctl_by_plate)))
    return trt, ctl_by_plate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_conditions', type=int, default=400)
    ap.add_argument('--max_ctl', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split_half', action='store_true',
                    help='FAIR comparison: average replicate deltas within two disjoint PLATE groups and '
                         'correlate the halves. A single Level-3 well is not comparable to a Level-5 MODZ '
                         'signature, which is itself a replicate average -- that is what MODZ is for.')
    ap.add_argument('--min_wells', type=int, default=4)
    a = ap.parse_args()

    f = h5py.File(GCTX, 'r')
    mat = f['0/DATA/0/matrix']
    gi = landmark_rows(f)
    col_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/COL/id'][:]]
    col_pos = {c: i for i, c in enumerate(col_ids)}
    print("GCTX wells: %d" % len(col_ids))

    trt, ctl_by_plate = load_meta()

    groups = defaultdict(list)
    for inst, (plate, cell, pert, dose, time) in trt.items():
        if plate in ctl_by_plate and inst in col_pos:
            groups[(cell, pert, dose, time)].append((plate, inst))

    multi = {}
    for k, v in groups.items():
        seen, keep = set(), []
        for p, i in v:
            if p not in seen:
                seen.add(p)
                keep.append((p, i))
        if len(keep) >= (a.min_wells if a.split_half else 2):
            multi[k] = keep
    print("conditions with >=2 wells on DIFFERENT plates: %d" % len(multi))

    rnd = random.Random(a.seed)
    chosen = rnd.sample(sorted(multi), min(a.n_conditions, len(multi)))

    need = sorted({p for k in chosen for p, _ in multi[k]})
    print("computing DMSO medians for %d plates ..." % len(need), flush=True)
    plate_med = {}
    for n, p in enumerate(need):
        cols = sorted({col_pos[i] for i in ctl_by_plate[p][:a.max_ctl] if i in col_pos})
        if len(cols) >= 3:
            plate_med[p] = np.median(mat[cols, :][:, gi], axis=0)
        if (n + 1) % 50 == 0:
            print("  %d/%d" % (n + 1, len(need)), flush=True)

    rs, strengths = [], []
    for n, k in enumerate(chosen):
        wells = [(p, i) for p, i in multi[k] if p in plate_med]
        if len(wells) < (a.min_wells if a.split_half else 2):
            continue
        d = []
        for p, i in wells[:8]:
            d.append(mat[col_pos[i], :][gi] - plate_med[p])
        d = np.array(d)
        if a.split_half:
            # disjoint PLATE groups, averaged within each -- the direct analogue of a MODZ signature
            h = len(d) // 2
            A, B = d[:h].mean(0), d[h:2 * h].mean(0)
            if A.std() > 0 and B.std() > 0:
                rs.append(np.corrcoef(A, B)[0, 1])
                strengths.append((np.abs(A).mean() + np.abs(B).mean()) / 2)
        else:
            for x in range(len(d)):
                for y in range(x + 1, len(d)):
                    if d[x].std() > 0 and d[y].std() > 0:
                        rs.append(np.corrcoef(d[x], d[y])[0, 1])
                        strengths.append((np.abs(d[x]).mean() + np.abs(d[y]).mean()) / 2)
        if (n + 1) % 50 == 0:
            print("  condition %d/%d  pairs %d" % (n + 1, len(chosen), len(rs)), flush=True)

    rs = np.array(rs); strengths = np.array(strengths)
    print("\n" + "=" * 78)
    print("LEVEL-3 %s RELIABILITY  (%d comparisons, different plates)" % ("SPLIT-HALF (replicate-averaged)" if a.split_half else "single-well", len(rs)))
    print("=" * 78)
    print("  overall mean r = %.4f   median %.4f" % (rs.mean(), np.median(rs)))
    print("  LEVEL-5 reference [6.1]: 0.127 over ALL signatures, 0.509-0.619 on reproducible\n")
    qs = np.quantile(strengths, [0, .25, .5, .75, 1.0])
    print("  %26s %6s %13s" % ("level-3 |delta| bin", "n", "replicate r"))
    rows = []
    for i in range(4):
        m = (strengths >= qs[i]) & (strengths <= qs[i + 1])
        print("  %11.3f-%-13.3f %6d %13.4f" % (qs[i], qs[i + 1], m.sum(), rs[m].mean()))
        rows.append({'lo': float(qs[i]), 'hi': float(qs[i + 1]), 'n': int(m.sum()),
                     'r': float(rs[m].mean())})
    out = os.path.join(ROOT, 'model', 'results', ('level3_splithalf_reliability.json' if a.split_half else 'level3_replicate_reliability.json'))
    json.dump({'n_pairs': int(len(rs)), 'mean_r': float(rs.mean()),
               'median_r': float(np.median(rs)),
               'level5_reference': {'all': 0.127, 'reproducible': [0.509, 0.619]},
               'by_strength': rows}, open(out, 'w'), indent=2)
    print("\nwrote %s" % out)


if __name__ == '__main__':
    main()
