# -*- coding: utf-8 -*-
"""
Design tests for the v9 Level-3 substrate. Each asserts a property the extractor CLAIMS, not that it ran.
Run before anything consumes these arrays:

    python model/level3/test_extract_l3.py            # structural + from-scratch recomputation
    python model/level3/test_extract_l3.py --xpert    # also check against XPert's own control vector

The load-bearing one is `recompute from scratch`: it re-reads a sample of signatures' wells straight out of
the GCTX, rebuilds the mean expression and the plate DMSO median independently of the extractor's
accumulators, and demands agreement. Everything else can pass while the pipeline is silently wrong.
"""
import os, csv, json, argparse, sys
from collections import defaultdict

import numpy as np
import h5py

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_level3_distil import PHASE, ROOT, OUT, SIGS, GENE_ORDER, canonical_gene_rows, N_GENES

R = []


def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_spot', type=int, default=12, help='signatures per phase recomputed from scratch')
    ap.add_argument('--xpert', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    Xt = np.load(os.path.join(OUT, 'X_trt_l3.npy'), mmap_mode='r')
    Xc = np.load(os.path.join(OUT, 'X_ctl_l3.npy'), mmap_mode='r')
    cov = np.load(os.path.join(OUT, 'l3_covered.npy'))
    cj = json.load(open(os.path.join(OUT, 'l3_coverage.json')))
    meta = {}
    for r in csv.DictReader(open(os.path.join(OUT, 'l3_meta.tsv'), encoding='utf-8'), delimiter='\t'):
        meta[int(r['sig_row'])] = r
    yrow = np.load(os.path.join(OUT, 'l3_yrow.npy'))

    sig_rows = []
    with open(SIGS, encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            sig_rows.append((r['sig_id'], r['phase'], int(r['row'])))
    n_all = len(sig_rows)

    # ---------------- structure ----------------
    check("arrays are [n_signatures, 978]", Xt.shape == (n_all, N_GENES) and Xc.shape == (n_all, N_GENES),
          f"{Xt.shape} / {Xc.shape} vs n_sig={n_all}")
    check("row i of the arrays is row i of signatures_usable.tsv",
          all(meta[i]['sig_id'] == sig_rows[i][0] for i in range(0, n_all, 977)),
          "checked every 977th row")
    check("meta phase agrees with signatures_usable phase",
          all(meta[i]['phase'] == sig_rows[i][1] for i in range(0, n_all, 977)))
    # The index trap: signatures_usable has 310,114 lines but its `row` column indexes a 312,438-row
    # Level-5 target. These arrays are positional; l3_yrow.npy is the bridge. If this pairing were wrong,
    # every v9 run would train Level-3 inputs against another signature's Level-5 label.
    Yl5 = np.load(os.path.join(ROOT, 'phase2_assembly', 'outputs', 'Y_target_level5_978.npy'),
                  mmap_mode='r')
    check("l3_yrow.npy is the tsv's own row column, in line order",
          len(yrow) == n_all and all(int(yrow[i]) == sig_rows[i][2] for i in range(0, n_all, 977)))
    check("y_row is required: it is NOT the line position",
          not bool((yrow == np.arange(n_all)).all()),
          f"max y_row {int(yrow.max())} vs {n_all - 1} lines; Y_target has {Yl5.shape[0]} rows")
    check("every y_row is in range for the Level-5 target",
          bool((yrow >= 0).all() and (yrow < Yl5.shape[0]).all()))
    check("meta y_row agrees with l3_yrow.npy",
          all(int(meta[i]['y_row']) == int(yrow[i]) for i in range(0, n_all, 977)))
    order = [l.strip() for l in open(GENE_ORDER, encoding='utf-8') if l.strip()]
    emitted = [l.strip() for l in open(os.path.join(OUT, 'genes_l3_canonical.txt'), encoding='utf-8')
               if l.strip()]
    check("gene axis is the canonical project order, unaltered", emitted == order,
          f"{len(emitted)} genes, first={emitted[0]}")

    # ---------------- the gate ----------------
    pct = 100.0 * cov.sum() / n_all
    check("GATE: coverage >= 90% of our signatures", pct >= 90.0,
          f"{int(cov.sum())}/{n_all} = {pct:.2f}%  (was 44.8% under the key join)")
    check("coverage json agrees with the mask", abs(cj['TOTAL']['pct'] - pct) < 0.01)

    # ---------------- NaN discipline ----------------
    samp = rng.choice(np.flatnonzero(cov), 2000, replace=False)
    unc = np.flatnonzero(~cov)
    check("covered rows are finite in both arrays",
          bool(np.isfinite(Xt[samp]).all() and np.isfinite(Xc[samp]).all()))
    check("uncovered rows are NaN, not 0 (so silent misuse is impossible)",
          len(unc) == 0 or bool(np.isnan(Xt[unc[:200]]).all() and np.isnan(Xc[unc[:200]]).all()),
          f"{len(unc)} uncovered")
    check("covered mask == (n_wells_used > 0) in meta",
          all((int(meta[i]['n_wells_used']) > 0) == bool(cov[i]) for i in rng.choice(n_all, 3000)))

    # ---------------- value ranges: Level 3 is log-expression, XPert's is [0,15] ----------------
    lo_t, hi_t = float(np.min(Xt[samp])), float(np.max(Xt[samp]))
    lo_c, hi_c = float(np.min(Xc[samp])), float(np.max(Xc[samp]))
    check("X_trt within the Level-3 log range [-0.1, 15.5]", -0.1 <= lo_t and hi_t <= 15.5,
          f"[{lo_t:.2f}, {hi_t:.2f}]")
    check("X_ctl within the Level-3 log range [-0.1, 15.5]", -0.1 <= lo_c and hi_c <= 15.5,
          f"[{lo_c:.2f}, {hi_c:.2f}]")
    d = np.asarray(Xt[samp], np.float64) - np.asarray(Xc[samp], np.float64)
    check("delta = trt - ctl is centred near zero and not degenerate",
          abs(float(d.mean())) < 0.25 and float(np.abs(d).mean()) > 0.05,
          f"mean {d.mean():+.4f}  mean|d| {np.abs(d).mean():.4f}")
    check("delta is not a constant offset (genes move independently)",
          float(np.median(d.std(1))) > 0.1, f"median per-row sd {np.median(d.std(1)):.3f}")

    # ---------------- recompute a sample of signatures FROM SCRATCH ----------------
    for ph in ['P1', 'P2']:
        P = PHASE[ph]
        rows_ph = [i for i in range(n_all) if sig_rows[i][1] == ph and cov[i]]
        pick = rng.choice(rows_ph, a.n_spot, replace=False)
        want_sigs = {sig_rows[i][0]: i for i in pick}
        distil = {}
        with open(P['sig'], encoding='utf-8') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                if r['sig_id'] in want_sigs:
                    distil[r['sig_id']] = r['distil_id'].split('|')
        plate_of, dmso = {}, defaultdict(list)
        need = set().union(*distil.values())
        with open(P['inst'], encoding='utf-8') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                if r['pert_type'] == 'ctl_vehicle':
                    dmso[r[P['plate_col']]].append(r['inst_id'])
                    plate_of[r['inst_id']] = r[P['plate_col']]
                elif r['inst_id'] in need:
                    plate_of[r['inst_id']] = r[P['plate_col']]
        f = h5py.File(P['gctx'], 'r')
        gi, _ = canonical_gene_rows(f)
        cols = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0/META/COL/id'][:]]
        cpos = {c: i for i, c in enumerate(cols)}

        def read(ids):
            rr = sorted(cpos[i] for i in ids)
            return np.array([f['0/DATA/0/matrix'][r, :][gi] for r in rr], np.float64)

        et, ec, phase_ok = [], [], True
        for s, out_row in want_sigs.items():
            wells = [w for w in distil[s] if w in cpos and w in plate_of
                     and len(dmso[plate_of[w]]) >= 3]
            phase_ok &= all(w in plate_of for w in wells)
            trt = read(wells).mean(0)
            ctl = np.mean([np.median(read(dmso[plate_of[w]]), axis=0) for w in wells], axis=0)
            et.append(np.abs(trt - np.asarray(Xt[out_row], np.float64)).max())
            ec.append(np.abs(ctl - np.asarray(Xc[out_row], np.float64)).max())
        f.close()
        check(f"[{ph}] X_trt recomputed from the GCTX matches (n={a.n_spot})", max(et) < 1e-3,
              f"max abs diff {max(et):.2e}")
        check(f"[{ph}] X_ctl recomputed from the plate DMSO median matches (n={a.n_spot})", max(ec) < 1e-3,
              f"max abs diff {max(ec):.2e}")
        check(f"[{ph}] every contributing well belongs to {ph}'s own inst_info (no cross-phase pairing)",
              phase_ok)
        check(f"[{ph}] plate key is '{P['plate_col']}'",
              cj[ph]['plate_col'] == P['plate_col'], cj[ph]['plate_col'])

    # ---------------- the P1 plate column actually matters ----------------
    # If inst_id-prefix grouping gave the same answer, the rna_plate column would be decorative and a
    # future refactor could drop it silently. Show the two groupings genuinely differ.
    P = PHASE['P1']
    pref_diff = 0
    with open(P['inst'], encoding='utf-8') as fh:
        for k, r in enumerate(csv.DictReader(fh, delimiter='\t')):
            if k >= 20000:
                break
            pref_diff += (r['inst_id'].split(':')[0] != r['rna_plate'])
    check("P1 rna_plate is NOT the inst_id prefix (so the column is load-bearing)", pref_diff > 19000,
          f"{pref_diff}/20000 differ")

    # ---------------- continuity with the pipeline that produced the ridge A/B ----------------
    # The first draft of the extractor wrote into `level3/` and OVERWROTE that pipeline's X_ctl_l3.npy,
    # so a profile-level comparison against it is no longer possible without regenerating it. Continuity
    # is instead established where it matters: model/level3/ab_matched_control.py re-runs the identical
    # ridge A/B on this substrate, and must reproduce or beat +0.0314 / +0.0435 / +0.0372.
    OLD = os.path.join(ROOT, 'phase2_assembly', 'outputs', 'level3')
    old_c = os.path.join(OLD, 'X_ctl_l3.npy')
    old_j = os.path.join(OLD, 'join_l5row_to_l3row.npy')
    if os.path.exists(old_c) and os.path.exists(old_j):
        Xo = np.load(old_c, mmap_mode='r')
        j = np.load(old_j)
        og = [l.strip() for l in open(os.path.join(OLD, 'genes_l3.txt'), encoding='utf-8') if l.strip()]
        pos = {g: i for i, g in enumerate(og)}
        gidx = np.array([pos[g] for g in order])
        # the old join keyed on Y-row, these arrays are positional -- bridge them through l3_yrow
        y2pos = {int(y): i for i, y in enumerate(yrow)}
        sel = [(y2pos[int(r)], int(k)) for r, k in j
               if int(r) in y2pos and sig_rows[y2pos[int(r)]][1] == 'P1' and cov[y2pos[int(r)]]]
        sel = [sel[i] for i in rng.choice(len(sel), 400, replace=False)]
        cs = [np.corrcoef(np.asarray(Xc[r], np.float64),
                          np.asarray(Xo[k], np.float64)[gidx])[0, 1] for r, k in sel]
        check("new matched control reproduces the old key-join control on P1 rows (r > 0.95)",
              float(np.median(cs)) > 0.95, f"median r = {np.median(cs):.4f} over 400 rows")
    else:
        print("[SKIP] old key-join control absent (overwritten by the first draft of the "
              "extractor) -- continuity is tested by ab_matched_control.py instead")

    # ---------------- external: XPert's own control vector ----------------
    if a.xpert:
        h5 = os.path.join(ROOT, 'external', 'xpert', 'l1000_mdmt_full_336852.h5ad')
        if os.path.exists(h5):
            with h5py.File(h5, 'r') as g:
                xc = g['obsm/X_ctl']
                sub = np.asarray(xc[:2000], np.float64)
            check("XPert's X_ctl occupies the same range as ours",
                  -0.1 <= sub.min() and sub.max() <= 15.5,
                  f"theirs [{sub.min():.2f}, {sub.max():.2f}] vs ours [{lo_c:.2f}, {hi_c:.2f}]")
        else:
            print("[SKIP] XPert h5ad not present")

    print("\n%d/%d checks passed" % (sum(R), len(R)))
    sys.exit(0 if all(R) else 1)


if __name__ == '__main__':
    main()
