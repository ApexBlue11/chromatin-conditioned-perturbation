# -*- coding: utf-8 -*-
"""
Build the v8 training substrate from LINCS Level 3: PLATE-MATCHED controls + MODZ-style weighted replicate
aggregation. Rationale: ../V8_PLAN.md §1.

The point of the migration is the INPUT, not the target. Today our baseline is CCLE -- a different assay on
a different platform from a different experiment. Level 3 gives the DMSO wells from the SAME PLATE in the
SAME BATCH as the treated well, which is what every SOTA LINCS model is handed and we are not.

The target's noise is the cost, and it is handled explicitly. A plain mean of replicate deltas measured
0.144 (0.243 top quartile) against MODZ's 0.127 / 0.509-0.619 [RESULTS 25]. So the aggregation here is
MODZ-style: replicates are weighted by how much each agrees with the others, exactly as cmap's MODZ does,
rather than averaged flat. Whether that recovers the denoising is the GATE on the whole migration and is
measured separately by `replicate_reliability.py --target l3_modz`.

Outputs (phase2_assembly/outputs/level3/):
    Y_delta_l3.npy   [n_cond, 978]  MODZ-weighted mean of (well - plate DMSO median)
    X_ctl_l3.npy     [n_cond, 978]  the matched control profile itself -- the new baseline INPUT
    conditions_l3.tsv               cell, pert_id, dose, time, n_wells, n_plates
"""
import os, csv, json, argparse, time
from collections import defaultdict

import numpy as np
import h5py

ROOT = r'C:\Projects\LINCS'
GCTX = os.path.join(ROOT, 'level3 files', 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
INST = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_inst_info.txt',
                    'GSE92742_Broad_LINCS_inst_info.txt')
GENE = os.path.join(ROOT, 'Data Info', 'GSE92742_Broad_LINCS_gene_info.txt',
                    'GSE92742_Broad_LINCS_gene_info.txt')
OUT = os.path.join(ROOT, 'phase2_assembly', 'outputs', 'level3')


def landmark_rows(f):
    lm = []
    with open(GENE, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['pr_is_lm'] == '1':
                lm.append((r['pr_gene_id'], r['pr_gene_symbol']))
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/ROW/id'][:]]
    pos = {g: i for i, g in enumerate(ids)}
    keep = [(pos[g], s) for g, s in lm if g in pos]
    keep.sort()
    idx = np.array([k[0] for k in keep])
    syms = [k[1] for k in keep]
    print("landmarks located: %d/978" % len(idx))
    return idx, syms


def modz_weights(D):
    """cmap MODZ: weight each replicate by its summed Spearman agreement with the others.
    k==1 -> [1]; k==2 -> equal. Weights floored at 0.01 and renormalised, as in the reference method."""
    k = len(D)
    if k == 1:
        return np.array([1.0])
    if k == 2:
        return np.array([0.5, 0.5])
    R = np.zeros((k, k))
    ranks = np.apply_along_axis(lambda v: np.argsort(np.argsort(v)).astype(np.float64), 1, D)
    for i in range(k):
        for j in range(i + 1, k):
            c = np.corrcoef(ranks[i], ranks[j])[0, 1]
            R[i, j] = R[j, i] = 0.0 if np.isnan(c) else c
    w = R.sum(1)
    w = np.clip(w, 0.01, None)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', type=int, default=20000, help='GCTX rows per sequential read')
    ap.add_argument('--min_ctl', type=int, default=3, help='min DMSO wells for a usable plate')
    ap.add_argument('--max_ctl', type=int, default=32)
    ap.add_argument('--rebuild_cache', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    f = h5py.File(GCTX, 'r')
    mat = f['0/DATA/0/matrix']
    gi, syms = landmark_rows(f)
    col_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/COL/id'][:]]
    col_pos = {c: i for i, c in enumerate(col_ids)}

    # ---- metadata ----
    trt, ctl_by_plate = {}, defaultdict(list)
    with open(INST, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['pert_type'] == 'trt_cp':
                trt[r['inst_id']] = (r['rna_plate'], r['cell_id'], r['pert_id'],
                                     r['pert_dose'], r['pert_time'])
            elif r['pert_type'] == 'ctl_vehicle':
                ctl_by_plate[r['rna_plate']].append(r['inst_id'])
    ctl_by_plate = {p: [i for i in v if i in col_pos][:a.max_ctl]
                    for p, v in ctl_by_plate.items()}
    ctl_by_plate = {p: v for p, v in ctl_by_plate.items() if len(v) >= a.min_ctl}
    print("trt_cp wells: %d | usable plates (>=%d DMSO): %d" % (len(trt), a.min_ctl, len(ctl_by_plate)))

    # ---- which wells do we need, and where does each land in the cache ----
    need = set()
    conditions = defaultdict(list)
    for inst, (plate, cell, pert, dose, ptime) in trt.items():
        if plate in ctl_by_plate and inst in col_pos:
            conditions[(cell, pert, dose, ptime)].append((plate, inst))
            need.add(inst)
    for p, v in ctl_by_plate.items():
        need.update(v)
    need_rows = sorted(col_pos[i] for i in need)
    slot = {r: n for n, r in enumerate(need_rows)}
    print("conditions: %d | wells to read: %d" % (len(conditions), len(need_rows)))

    # ---- pass 1: ONE sequential scan, landmark columns only, into a disk-backed cache ----
    cache_path = os.path.join(OUT, 'wells_cache.dat')
    meta_path = os.path.join(OUT, 'wells_cache.json')
    want = {'n': len(need_rows), 'g': len(gi)}
    have = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    if a.rebuild_cache or have != want or not os.path.exists(cache_path):
        cache = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=(len(need_rows), len(gi)))
        need_arr = np.array(need_rows)
        t0 = time.time()
        for start in range(0, mat.shape[0], a.block):
            stop = min(start + a.block, mat.shape[0])
            lo = np.searchsorted(need_arr, start); hi = np.searchsorted(need_arr, stop)
            if hi <= lo:
                continue
            blk = mat[start:stop, :]
            rows = need_arr[lo:hi] - start
            cache[lo:hi] = blk[rows, :][:, gi]
            if (start // a.block) % 10 == 0:
                el = time.time() - t0
                print("  scan %.0f%%  %.0fs" % (100 * stop / mat.shape[0], el), flush=True)
        cache.flush()
        json.dump(want, open(meta_path, 'w'))
        print("cache built in %.0fs" % (time.time() - t0))
    else:
        print("reusing existing well cache")
    cache = np.memmap(cache_path, dtype=np.float32, mode='r', shape=(len(need_rows), len(gi)))

    # ---- pass 2: plate DMSO medians ----
    plate_med = {}
    for n, (p, wells) in enumerate(ctl_by_plate.items()):
        rows = [slot[col_pos[i]] for i in wells]
        plate_med[p] = np.median(np.asarray(cache[rows]), axis=0)
        if (n + 1) % 200 == 0:
            print("  plate medians %d/%d" % (n + 1, len(ctl_by_plate)), flush=True)

    # ---- pass 3: per-condition MODZ-weighted delta + matched control ----
    keys = sorted(conditions)
    Y = np.zeros((len(keys), len(gi)), np.float32)
    C = np.zeros((len(keys), len(gi)), np.float32)
    meta = []
    for n, k in enumerate(keys):
        wells = conditions[k]
        D, ctls = [], []
        for plate, inst in wells:
            v = np.asarray(cache[slot[col_pos[inst]]], np.float64)
            D.append(v - plate_med[plate]); ctls.append(plate_med[plate])
        D = np.array(D)
        w = modz_weights(D)
        Y[n] = (w[:, None] * D).sum(0)
        C[n] = np.average(np.array(ctls), axis=0)          # the matched baseline INPUT
        meta.append((k[0], k[1], k[2], k[3], len(wells), len({p for p, _ in wells})))
        if (n + 1) % 20000 == 0:
            print("  conditions %d/%d" % (n + 1, len(keys)), flush=True)

    np.save(os.path.join(OUT, 'Y_delta_l3.npy'), Y)
    np.save(os.path.join(OUT, 'X_ctl_l3.npy'), C)
    with open(os.path.join(OUT, 'conditions_l3.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['cell_id', 'pert_id', 'dose', 'time', 'n_wells', 'n_plates'])
        w.writerows(meta)
    with open(os.path.join(OUT, 'genes_l3.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(syms))
    st = np.abs(Y).mean(1)
    print("\nwrote %d conditions x %d genes" % Y.shape)
    print("  mean|delta| : median %.3f  p90 %.3f  max %.3f" % (np.median(st), np.percentile(st, 90), st.max()))
    print("  n_wells     : median %d  max %d" % (int(np.median([m[4] for m in meta])), max(m[4] for m in meta)))
    print("  -> %s" % OUT)


if __name__ == '__main__':
    main()
