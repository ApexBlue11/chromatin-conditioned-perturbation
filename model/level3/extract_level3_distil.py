# -*- coding: utf-8 -*-
"""
Build the v9 Level-3 substrate, SIGNATURE-ALIGNED, from BOTH LINCS phases.

Why this exists rather than `extract_level3.py`
-----------------------------------------------
`extract_level3.py` joins Level 5 to Level 3 on the key (cell_id, pert_id, dose, time). Measured, that
join reaches 66.3 % of our P1 signatures and 4.1 % of our P2 signatures -- and those 4.1 % are FALSE
matches, because its condition table contains GSE92742 wells only, so a P2 signature keyed to it gets a
"plate-matched" control from a different experiment entirely. Extracting GSE70138 under the same key logic
projects to ~66 % overall, which fails the >=90 % coverage gate in V9_HANDOFF.md §D.

LINCS ships the exact mapping we were approximating: `sig_info.distil_id` lists the Level-3 wells that MODZ
aggregated into each Level-5 signature. Using it, projected coverage is 100.0 % (P1) and 99.0 % (P2), the
control is drawn from the SAME wells' plates as the target, and cross-phase pairing becomes impossible by
construction. That is the only deviation from the handoff's step 1; the one-sequential-pass GCTX scan it
tells us to reuse is kept.

What is written (row-aligned to phase2_assembly/outputs/signatures_usable.tsv, gene-aligned to
Data Info/pathway_landmark_genes.txt):

    X_trt_l3.npy   [310114, 978] float32   absolute Level-3 expression, mean over the signature's own wells
    X_ctl_l3.npy   [310114, 978] float32   plate-matched DMSO median, averaged over those same wells' plates
    l3_covered.npy [310114]      bool      which rows are real
    l3_yrow.npy    [310114]      int32     row of Y_target_level5_978.npy for the SAME signature
    l3_meta.tsv                            sig_row, sig_id, phase, y_row, n_wells_total/used, n_plates
    l3_coverage.json                       the gate number and its decomposition

INDEX SEMANTICS -- the trap here is real. `signatures_usable.tsv` has 310,114 lines, but its `row` column
indexes `Y_target_level5_978.npy`, which has 312,438 rows; `row` is NOT the line position. These arrays are
indexed by LINE POSITION (`sig_row`), the same order a dataset gets when it reads the tsv. To pair them with
the Level-5 target use `Y_target[np.load('l3_yrow.npy')]`. Indexing these arrays with a y_row would be
silently, subtly wrong -- so `l3_yrow.npy` is emitted and `test_extract_l3.py` asserts the pairing.

The DELTA IS NOT STORED. delta = X_trt - X_ctl, exactly, because both sides use the same flat mean over the
same wells (linearity). Storing it would cost 1.2 GB and create a third array that could drift out of sync
with the identity. Uncovered rows are NaN, not 0, so that using them without the mask fails loudly.

Aggregation is a FLAT MEAN, not MODZ. Measured [RESULTS 25 / gate_modz.py]: MODZ-weighted Level-3 deltas
score 0.157 split-half against a plain mean's 0.164. `--agg modz` is kept for a controlled re-test; it
breaks the exact delta identity above and says so.

Usage:
    python model/level3/extract_level3_distil.py --phase P1
    python model/level3/extract_level3_distil.py --phase P2
"""
import os, csv, json, argparse, time, sys
from collections import defaultdict

import numpy as np
import h5py

csv.field_size_limit(10 ** 9)

ROOT = r'C:\Projects\LINCS'
DI = os.path.join(ROOT, 'Data Info')
# A SEPARATE directory from the old key-join pipeline's `level3/`. The first draft of this script
# wrote X_ctl_l3.npy into `level3/` and clobbered that pipeline's array of the same name; the
# artefacts of two different joins must not share a namespace.
OUT = os.path.join(ROOT, 'phase2_assembly', 'outputs', 'level3_sig')

# Per-phase sources. The PLATE KEY DIFFERS BY PHASE and is not guessable from the inst_id:
# for GSE92742 the inst_id prefix disagrees with rna_plate on 20000/20000 sampled wells, while for
# GSE70138 there is no rna_plate column at all and det_plate matches the prefix 20000/20000.
PHASE = {
    'P1': dict(
        gctx=os.path.join(ROOT, 'level3 files',
                          'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx'),
        inst=os.path.join(DI, 'GSE92742_Broad_LINCS_inst_info.txt',
                          'GSE92742_Broad_LINCS_inst_info.txt'),
        sig=os.path.join(DI, 'GSE92742_Broad_LINCS_sig_info.txt',
                         'GSE92742_Broad_LINCS_sig_info.txt'),
        plate_col='rna_plate'),
    'P2': dict(
        gctx=os.path.join(ROOT, 'level3 files',
                          'GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328.gctx'),
        inst=os.path.join(DI, 'GSE70138_Broad_LINCS_inst_info_2017-03-06.txt',
                          'GSE70138_Broad_LINCS_inst_info.txt'),
        sig=os.path.join(DI, 'GSE70138_Broad_LINCS_sig_info_2017-03-06.txt',
                         'GSE70138_Broad_LINCS_sig_info.txt'),
        plate_col='det_plate'),
}
GENE_INFO = os.path.join(DI, 'GSE92742_Broad_LINCS_gene_info.txt',
                         'GSE92742_Broad_LINCS_gene_info.txt')
GENE_ORDER = os.path.join(DI, 'pathway_landmark_genes.txt')
SIGS = os.path.join(ROOT, 'phase2_assembly', 'outputs', 'signatures_usable.tsv')

N_GENES = 978


def canonical_gene_rows(f):
    """Return GCTX row indices for the 978 landmarks IN THE CANONICAL PROJECT ORDER.

    The old extractor wrote genes in gene_info order and left the reindexing to each consumer
    (`test_matched_control.align_genes`). Emitting the canonical order here deletes that whole class of
    silent misalignment: every downstream array already uses this order.
    """
    sym2id = {}
    with open(GENE_INFO, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['pr_is_lm'] == '1':
                sym2id[r['pr_gene_symbol']] = r['pr_gene_id']
    order = [l.strip() for l in open(GENE_ORDER, encoding='utf-8') if l.strip()]
    assert len(order) == N_GENES == len(set(order)), 'canonical gene order is not 978 unique symbols'
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/ROW/id'][:]]
    pos = {g: i for i, g in enumerate(ids)}
    missing = [g for g in order if g not in sym2id or sym2id[g] not in pos]
    assert not missing, 'landmarks absent from this GCTX: %s' % missing[:10]
    return np.array([pos[sym2id[g]] for g in order]), order


def load_our_signatures():
    """In LINE-POSITION order. `y_row` is the tsv's own `row` column: the index into the Level-5 target."""
    rows, phase, y_row = [], [], []
    with open(SIGS, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            rows.append(r['sig_id'])
            phase.append(r['phase'])
            y_row.append(int(r['row']))
    return rows, phase, np.array(y_row, np.int32)


def modz_weights(D):
    """cmap MODZ: weight each replicate by summed Spearman agreement with the others. Kept only so the
    flat-mean choice can be re-tested head to head; it breaks delta == trt - ctl."""
    k = len(D)
    if k == 1:
        return np.array([1.0])
    if k == 2:
        return np.array([0.5, 0.5])
    ranks = np.apply_along_axis(lambda v: np.argsort(np.argsort(v)).astype(np.float64), 1, D)
    R = np.corrcoef(ranks)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 0.0)
    w = np.clip(R.sum(1), 0.01, None)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['P1', 'P2'], required=True)
    ap.add_argument('--block', type=int, default=6000,
                    help='GCTX well-rows per sequential read; 6000 x 12328 x 4B = 296 MB (RAM is 15 GB)')
    ap.add_argument('--min_ctl', type=int, default=3, help='min DMSO wells for a usable plate')
    ap.add_argument('--agg', choices=['mean', 'modz'], default='mean')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P = PHASE[a.phase]
    t00 = time.time()

    # ---------------- our signatures, and the wells behind them ----------------
    sig_ids, sig_phase, y_row = load_our_signatures()
    n_all = len(sig_ids)
    np.save(os.path.join(OUT, 'l3_yrow.npy'), y_row)
    mine = {s: i for i, (s, p) in enumerate(zip(sig_ids, sig_phase)) if p == a.phase}
    print('[%s] our signatures: %d of %d' % (a.phase, len(mine), n_all))

    distil = {}
    with open(P['sig'], encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['sig_id'] in mine:
                distil[r['sig_id']] = r['distil_id'].split('|')
    assert len(distil) == len(mine), 'sig_info is missing %d of our signatures' % (len(mine) - len(distil))
    need_trt = set()
    for v in distil.values():
        need_trt.update(v)
    print('  distil wells: %d unique across %d signatures' % (len(need_trt), len(distil)))

    # ---------------- inst_info: plate of each needed well + DMSO wells per plate ----------------
    # Only wells we actually need are retained -- keeping all 1.3M would cost ~400 MB of dicts.
    plate_of, dmso_by_plate = {}, defaultdict(list)
    with open(P['inst'], encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            iid = r['inst_id']
            if r['pert_type'] == 'ctl_vehicle':
                dmso_by_plate[r[P['plate_col']]].append(iid)
                plate_of[iid] = r[P['plate_col']]
            elif iid in need_trt:
                plate_of[iid] = r[P['plate_col']]
    dmso_by_plate = {p: v for p, v in dmso_by_plate.items() if len(v) >= a.min_ctl}
    usable_plates = {p: i for i, p in enumerate(sorted(dmso_by_plate))}
    n_dmso = sum(len(v) for v in dmso_by_plate.values())
    print('  plates with >=%d DMSO: %d   DMSO wells: %d' % (a.min_ctl, len(usable_plates), n_dmso))

    # ---------------- GCTX columns ----------------
    f = h5py.File(P['gctx'], 'r')
    mat = f['0/DATA/0/matrix']
    gi, gene_order = canonical_gene_rows(f)
    col_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/COL/id'][:]]
    col_pos = {c: i for i, c in enumerate(col_ids)}
    print('  gctx %s  wells=%d genes=%d' % (os.path.basename(P['gctx']), mat.shape[0], mat.shape[1]))

    # ---------------- what to read, and where each read lands ----------------
    # treated wells: (gctx_row, out_row, plate_idx); a well is used only if its plate has enough DMSO.
    trt_rows, trt_sig, trt_plate = [], [], []
    n_total, n_used = np.zeros(n_all, np.int32), np.zeros(n_all, np.int32)
    plates_per_sig = defaultdict(set)
    lost_no_well = lost_no_dmso = 0
    for s, wells in distil.items():
        out_row = mine[s]
        n_total[out_row] = len(wells)
        keep = [w for w in wells if w in col_pos and w in plate_of and plate_of[w] in usable_plates]
        if not keep:
            if not any(w in col_pos and w in plate_of for w in wells):
                lost_no_well += 1
            else:
                lost_no_dmso += 1
            continue
        for w in keep:
            trt_rows.append(col_pos[w]); trt_sig.append(out_row)
            trt_plate.append(usable_plates[plate_of[w]])
            plates_per_sig[out_row].add(plate_of[w])
        n_used[out_row] = len(keep)
    trt_rows = np.array(trt_rows, np.int64); trt_sig = np.array(trt_sig, np.int64)
    trt_plate = np.array(trt_plate, np.int64)
    covered_rows = np.array(sorted(mine[s] for s in distil if n_used[mine[s]] > 0), np.int64)
    print('  wells to read: %d treated + %d DMSO = %d  (%.1f%% of the file)'
          % (len(trt_rows), n_dmso, len(trt_rows) + n_dmso,
             100.0 * (len(trt_rows) + n_dmso) / mat.shape[0]))
    print('  covered signatures: %d/%d  (lost: %d no well in gctx, %d plate <%d DMSO)'
          % (len(covered_rows), len(mine), lost_no_well, lost_no_dmso, a.min_ctl))

    # ---------------- ONE sequential pass ----------------
    # Accumulate treated sums per signature, and buffer the (small) DMSO wells. Plate medians cannot be
    # known until the scan ends, so the control is reconstructed afterwards from a per-(signature, plate)
    # well count -- exact, and it costs one int per well instead of a 4 GB well cache.
    dmso_slot = {}
    dmso_need = []
    for p, wells in dmso_by_plate.items():
        for w in wells:
            if w in col_pos:
                dmso_slot[col_pos[w]] = (len(dmso_need), usable_plates[p])
                dmso_need.append(col_pos[w])
    acc = np.zeros((len(mine), N_GENES), np.float32)
    acc_row = {r: i for i, r in enumerate(sorted(mine.values()))}
    acc_idx = np.array([acc_row[r] for r in trt_sig], np.int64)
    dmso_buf = np.zeros((len(dmso_need), N_GENES), np.float32)
    dmso_plate = np.zeros(len(dmso_need), np.int64)
    for r, (slot, pidx) in dmso_slot.items():
        dmso_plate[slot] = pidx
    modz_store = defaultdict(list) if a.agg == 'modz' else None

    want = np.zeros(mat.shape[0], np.bool_)
    want[trt_rows] = True
    want[np.array(dmso_need, np.int64)] = True
    t0 = time.time()
    for start in range(0, mat.shape[0], a.block):
        stop = min(start + a.block, mat.shape[0])
        loc = np.nonzero(want[start:stop])[0]
        if len(loc) == 0:
            continue
        blk = mat[start:stop, :][loc][:, gi]
        for k, l in enumerate(loc):
            g = start + int(l)
            if g in dmso_slot:
                dmso_buf[dmso_slot[g][0]] = blk[k]
        m = (trt_rows >= start) & (trt_rows < stop)
        if m.any():
            take = np.searchsorted(loc, trt_rows[m] - start)   # loc is sorted; every trt row is in it
            np.add.at(acc, acc_idx[m], blk[take])
            if modz_store is not None:
                for j, row in zip(acc_idx[m], blk[take]):
                    modz_store[int(j)].append(row.astype(np.float64))
        if (start // a.block) % 20 == 0:
            el = time.time() - t0
            print('  scan %5.1f%%  %6.0fs' % (100.0 * stop / mat.shape[0], el), flush=True)
    f.close()
    print('  scan done in %.0fs' % (time.time() - t0))

    # ---------------- plate medians, then the matched control ----------------
    n_pl = len(usable_plates)
    med = np.zeros((n_pl, N_GENES), np.float32)
    for p in range(n_pl):
        sel = dmso_plate == p
        med[p] = np.median(dmso_buf[sel], axis=0) if sel.any() else np.nan
    del dmso_buf

    # Sum the plate medians a signature's wells came from, in bounded chunks: a [n_sig, n_plate] count
    # matrix would be 1.6 GB for P1 and med[trt_plate] materialised at once would be 2.6 GB, on a 15 GB box.
    ctl = np.zeros((len(mine), N_GENES), np.float32)
    for lo in range(0, len(acc_idx), 50000):
        hi = min(lo + 50000, len(acc_idx))
        np.add.at(ctl, acc_idx[lo:hi], med[trt_plate[lo:hi]])

    used = np.array([n_used[r] for r in sorted(mine.values())], np.float32)
    ok = used > 0
    ctl[ok] /= used[ok][:, None]
    trt = np.full_like(acc, np.nan)
    trt[ok] = acc[ok] / used[ok][:, None]
    if modz_store is not None:                   # re-weight; breaks delta == trt - ctl, and says so
        print('  WARNING: --agg modz breaks the exact delta identity; measured worse than mean (0.157/0.164)')
        for j, rows in modz_store.items():
            D = np.array(rows) - ctl[j]
            w = modz_weights(D)
            trt[j] = (w[:, None] * np.array(rows)).sum(0)
    ctl[~ok] = np.nan

    # ---------------- merge into the full, row-aligned arrays ----------------
    paths = {k: os.path.join(OUT, k + '.npy') for k in ['X_trt_l3', 'X_ctl_l3']}
    mode = 'r+' if all(os.path.exists(p) for p in paths.values()) else 'w+'
    Xt = np.lib.format.open_memmap(paths['X_trt_l3'], mode=mode, dtype=np.float32, shape=(n_all, N_GENES))
    Xc = np.lib.format.open_memmap(paths['X_ctl_l3'], mode=mode, dtype=np.float32, shape=(n_all, N_GENES))
    if mode == 'w+':
        Xt[:] = np.nan; Xc[:] = np.nan
    dst = np.array(sorted(mine.values()), np.int64)
    Xt[dst] = trt; Xc[dst] = ctl
    Xt.flush(); Xc.flush()

    cov_path = os.path.join(OUT, 'l3_covered.npy')
    covered = np.load(cov_path) if os.path.exists(cov_path) else np.zeros(n_all, np.bool_)
    covered[dst] = ok
    np.save(cov_path, covered)

    meta_path = os.path.join(OUT, 'l3_meta.tsv')
    FIELDS = ['sig_row', 'sig_id', 'phase', 'y_row', 'n_wells_total', 'n_wells_used', 'n_plates', 'covered']
    prev = {}
    if os.path.exists(meta_path):
        # tolerate the older schema (a bare `row` column, no `y_row`) so one phase can be re-run alone
        for r in csv.DictReader(open(meta_path, encoding='utf-8'), delimiter='\t'):
            k = int(r.get('sig_row') or r['row'])
            keep = {f: r.get(f, '') for f in FIELDS}
            keep['sig_row'] = k
            keep['y_row'] = keep['y_row'] or int(y_row[k])
            prev[k] = keep
    for s, out_row in mine.items():
        prev[out_row] = dict(sig_row=out_row, sig_id=s, phase=a.phase, y_row=int(y_row[out_row]),
                             n_wells_total=int(n_total[out_row]), n_wells_used=int(n_used[out_row]),
                             n_plates=len(plates_per_sig[out_row]), covered=int(n_used[out_row] > 0))
    with open(meta_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, delimiter='\t', fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for k in sorted(prev):
            w.writerow(prev[k])
    with open(os.path.join(OUT, 'genes_l3_canonical.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(gene_order))

    cov_json = os.path.join(OUT, 'l3_coverage.json')
    rec = json.load(open(cov_json)) if os.path.exists(cov_json) else {}
    rec[a.phase] = dict(n_signatures=len(mine), covered=int(ok.sum()),
                        pct=round(100.0 * ok.sum() / len(mine), 2), lost_no_well=lost_no_well,
                        lost_plate_lt_min_ctl=lost_no_dmso, min_ctl=a.min_ctl, agg=a.agg,
                        plate_col=P['plate_col'], n_plates=len(usable_plates),
                        n_treated_wells=int(len(trt_rows)), n_dmso_wells=n_dmso,
                        seconds=round(time.time() - t00, 1))
    rec['TOTAL'] = dict(n_signatures=n_all, covered=int(covered.sum()),
                        pct=round(100.0 * covered.sum() / n_all, 2))
    json.dump(rec, open(cov_json, 'w'), indent=2)

    d = trt[ok] - ctl[ok]
    print('\n[%s] wrote %d rows into %d x %d arrays' % (a.phase, int(ok.sum()), n_all, N_GENES))
    print('  X_trt  min %.2f  median %.2f  max %.2f' % (np.nanmin(trt), np.nanmedian(trt), np.nanmax(trt)))
    print('  X_ctl  min %.2f  median %.2f  max %.2f' % (np.nanmin(ctl), np.nanmedian(ctl), np.nanmax(ctl)))
    print('  delta  mean|d| median %.3f  p90 %.3f  max %.3f'
          % (np.median(np.abs(d).mean(1)), np.percentile(np.abs(d).mean(1), 90), np.abs(d).max()))
    print('  COVERAGE %s: %.2f%%   TOTAL SO FAR: %.2f%%  (gate: >=90%%)'
          % (a.phase, rec[a.phase]['pct'], rec['TOTAL']['pct']))
    print('  -> %s' % OUT)


if __name__ == '__main__':
    main()
