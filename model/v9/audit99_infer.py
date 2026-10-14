# -*- coding: utf-8 -*-
"""RESULTS 99 / model/V10_AUDIT_AND_PLAN.md §5 (as amended by review 083): Phase 1's two inference tests. CPU only.

  --mode xcell  (C2b, split_cold_cell_1, P7's final-snapshot checkpoints). Two modified passes:
                  ctl    : x_cell := the row's own X_ctl (inductive; keeps cell identity, removes the transductive
                           cell average) -- OF RECORD (review 083 C5);
                  global : x_cell := the mean control over ALL TRAINING ROWS (removes all cell-level baseline) -- reported.
  --mode swap   (C3, split_cold_drug_1, P9's final-snapshot checkpoints). Four modified passes on one intact pass:
                the two swaps below, and -- review 084 C2, the seen-cell bound on C2b's input shift -- the same two x_cell
                passes as xcell mode ('ctl', 'global') on these SEEN cells, where x_cell is training-derived.
                Each swap is a COMPOUND-LEVEL
                substitution within cell (one substitute compound per (compound, cell), the same for all its rows):
                  tani   : the substitute is drawn (seed 0) from the other test compounds in that cell with ECFP4
                           Tanimoto < 0.4 to it -- OF RECORD (review 083 C6);
                  free   : drawn from all the other test compounds in that cell -- reported.
                Only the drug inputs move (drug_row -> atoms, atom_mask; u_feats); control, dose, time and cell stay.
                A (compound, cell) with no allowed substitute keeps its own drug and is excluded, counted.

Each run predicts the test rows intact and then under each modification, with one checkpoint, deterministically
(model.eval()). Before any of it, the intact predictions on the rows of --identity (row_index + the saved final-snapshot
deg_pred) must match: min per-row Pearson >= 0.9999 and every row finite. The identity pass is also timed: if the
projected wall time of the rest exceeds --budget_s (default 11 h; Kaggle's CPU session limit is 12 h and a timeout
writes no manifest, review 084 ask 6), the run refuses before predicting anything else.

Out: <out>/<mode>_seed<k>.npz with row_index, ctl, y, deg_intact, deg_<variant> per variant (and donor_row_index_<variant>
for swap, -1 where unswapped), plus a .json record.
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(os.path.dirname(HERE), 'v6'), os.path.join(os.path.dirname(HERE), 'v7'), os.path.dirname(HERE)):
    sys.path.insert(0, p)

SPLIT = {'xcell': 'split_cold_cell_1', 'swap': 'split_cold_drug_1'}
VARIANTS = {'xcell': ('ctl', 'global'), 'swap': ('tani', 'free', 'ctl', 'global')}
BUDGET_S = 11 * 3600
SWAP_SEED = 0
TANI_MAX = 0.4
FP_SLICE = slice(512 + 20, 512 + 20 + 2048)      # u = [Uni-Mol CLS 512 | descriptors 20 | ECFP4 2048 bits] (xpert_arm.py)
BATCH = 32
ID_MIN_R = 0.9999


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def row_r(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def tanimoto(A, B):
    """A [n, bits], B [m, bits] of 0/1 -> [n, m]."""
    A = (np.asarray(A) > 0.5).astype(np.float64)
    B = (np.asarray(B) > 0.5).astype(np.float64)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def substitutes(cell, pert, fp, rng, tani_max=None):
    """Compound-level substitution within cell. Returns donor (index into the given rows, -1 = no substitute).
    For each cell, each compound k gets ONE substitute compound s(k) != k drawn uniformly from the allowed set
    (other compounds of that cell; with tani_max, only those with Tanimoto(k, s) < tani_max), and every row of k
    takes the drug inputs of s(k)'s first row."""
    donor = np.full(len(cell), -1, np.int64)
    for c in np.unique(cell):
        ix = np.flatnonzero(cell == c)
        comps, first = np.unique(pert[ix], return_index=True)
        rep = ix[first]                                         # one representative row per compound
        T = tanimoto(fp[rep], fp[rep]) if tani_max is not None else None
        for a, k in enumerate(comps):
            allowed = np.flatnonzero(comps != k)
            if tani_max is not None:
                allowed = allowed[T[a, allowed] < tani_max]
            if len(allowed) == 0:
                continue
            s = rng.choice(allowed)
            donor[ix[pert[ix] == k]] = rep[s]
    ok = donor >= 0
    assert (cell[donor[ok]] == cell[ok]).all() and (pert[donor[ok]] != pert[ok]).all()
    return donor


def predict(model, D, idx, dev):
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            out.append(model(D.batch(idx[s:s + BATCH], dev))['delta'].float().cpu().numpy())
    return np.concatenate(out)


def build(ckpt_path, roots, split):
    from config_v9 import V9Config
    from model_v9 import LincsV9
    from xpert_arm import XPertData, find
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if ck['split'] != split:
        raise SystemExit('FATAL: checkpoint split %r, expected %r' % (ck['split'], split))
    if ck.get('ablate_epi'):
        raise SystemExit('FATAL: checkpoint was trained chromatin-ablated')
    D = XPertData(find('xpert_mdmt_splits.npz', roots), roots, split)
    cfg = V9Config()
    for k, v in ck['cfg'].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = LincsV9(cfg, np.load(find('M_pathway_v9.npy', roots)), np.load(find('STRING_adj_978_v9.npy', roots)),
                    np.load(find('gene_vectors_978.npy', roots)))
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return ck, D, model


def identity(model, D, ident_path, dev):
    z = np.load(ident_path)
    pos = {int(r): i for i, r in enumerate(D.row_index)}
    missing = [int(r) for r in z['row_index'] if int(r) not in pos]
    if missing:
        raise SystemExit('FATAL: %d identity rows are not in the data' % len(missing))
    idx = np.array([pos[int(r)] for r in z['row_index']], np.int64)
    t0 = time.time()
    r = row_r(predict(model, D, idx, dev), np.asarray(z['deg_pred'], np.float64))
    res = {'n': int(len(r)), 'min_r': float(np.nanmin(r)), 'n_nonfinite': int((~np.isfinite(r)).sum()),
           's_per_row': (time.time() - t0) / max(1, len(r))}
    if res['n_nonfinite'] or res['min_r'] < ID_MIN_R:
        raise SystemExit('FATAL: identity check failed %s' % json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=sorted(SPLIT))
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--identity', required=True, help='slim npz: row_index + deg_pred of the saved final snapshot')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='smoke only: score the first N test rows')
    ap.add_argument('--budget_s', type=float, default=BUDGET_S)
    ap.add_argument('--roots', nargs='*', default=['/kaggle/input', r'C:\Projects\LINCS', r'C:\Projects\LINCS\external'])
    a = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    dev = 'cpu'
    ck, D, model = build(a.ckpt, a.roots, SPLIT[a.mode])
    seed = int(ck['seed'])
    out_npz = os.path.join(a.out, '%s_seed%d.npz' % (a.mode, seed))
    if os.path.exists(out_npz):
        raise SystemExit('FATAL: %s exists; run once' % out_npz)
    ident = identity(model, D, a.identity, dev)
    print('IDENTITY OK', json.dumps(ident), flush=True)

    te = D.te[:a.limit] if a.limit else D.te
    projected = ident['s_per_row'] * len(te) * (1 + len(VARIANTS[a.mode]))
    print('PROJECTED %.0f s for %d rows x %d passes (%.3f s/row)' % (projected, len(te), 1 + len(VARIANTS[a.mode]),
                                                                   ident['s_per_row']), flush=True)
    if projected > a.budget_s:
        raise SystemExit('FATAL: projected %.0f s exceeds the %.0f s budget; nothing predicted' % (projected, a.budget_s))
    arrays = {'row_index': D.row_index[te], 'ctl': D.C[te].astype(np.float32), 'y': (D.X[te] - D.C[te]).astype(np.float32),
              'deg_intact': predict(model, D, te, dev).astype(np.float32)}
    rec = {'mode': a.mode, 'split': SPLIT[a.mode], 'seed': seed, 'ckpt_sha1': sha1(a.ckpt), 'identity': ident,
           'n_rows': int(len(te)), 'limit': a.limit, 'code_sha1': sha1(os.path.abspath(__file__)), 'variants': {}}
    def xcell_passes():
        x_cell0 = D.x_cell
        for v in ('ctl', 'global'):
            if v == 'ctl':
                D.x_cell = D.C.copy()
                rec['variants'][v] = "x_cell := each row's own X_ctl"
            else:
                g = D.C[D.tr].mean(0).astype(np.float32)
                D.x_cell = np.broadcast_to(g[None, :], x_cell0.shape).copy()
                rec['variants'][v] = 'x_cell := mean control over all %d training rows' % len(D.tr)
            arrays['deg_' + v] = predict(model, D, te, dev).astype(np.float32)
            D.x_cell = x_cell0

    if a.mode == 'xcell':
        xcell_passes()
    else:
        cell, pert = np.asarray(D.cell).astype(str)[te], np.asarray(D.pert).astype(str)[te]
        fp = D.u[te][:, FP_SLICE]
        dr0, u0 = D.drug_row.copy(), D.u.copy()
        for v in ('tani', 'free'):
            donor_local = substitutes(cell, pert, fp, np.random.default_rng(SWAP_SEED),
                                      tani_max=TANI_MAX if v == 'tani' else None)
            sw = donor_local >= 0
            donor = te[np.where(sw, donor_local, np.arange(len(te)))]
            D.drug_row, D.u = dr0.copy(), u0.copy()
            D.drug_row[te[sw]] = dr0[donor[sw]]
            D.u[te[sw]] = u0[donor[sw]]
            arrays['deg_' + v] = predict(model, D, te, dev).astype(np.float32)
            arrays['donor_row_index_' + v] = np.where(sw, D.row_index[donor], -1)
            rec['variants'][v] = {'rule': ('compound-level within cell, ECFP4 Tanimoto < %.1f' % TANI_MAX if v == 'tani'
                                           else 'compound-level within cell, unconstrained') + ', seed %d' % SWAP_SEED,
                                  'n_unswapped': int((~sw).sum())}
        D.drug_row, D.u = dr0, u0
        xcell_passes()                                   # review 084 C2: the seen-cell bound, drug inputs restored first
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    rec['wall_s'] = time.time() - t0
    with open(out_npz.replace('.npz', '.json'), 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1)
    print('DONE', out_npz, json.dumps({'n_rows': rec['n_rows'], 'wall_s': round(rec['wall_s'], 1)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
