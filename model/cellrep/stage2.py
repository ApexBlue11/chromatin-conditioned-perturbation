"""Cell-representation programme, stage 2 (RESULTS 103.6, amended by 103.7): candidate cell representations on the L1000
knockdown proxy. One job = (model, setting, seed, injection, K); it trains, scores and saves per-row values.

  settings   loco:<cell>  targets d with per-fold references (stage1.fold_targets); training rows from the 9 knockdown cells
                          with CCLE expression minus the held-out cell; early stopping on 10 % of training ROWS
             cold         20 % of graph knockdown genes held out (default_rng(0)); targets y; training rows in the 9 cells,
                          test rows in the 6 core cells; early stopping on 10 % of training GENES
  inputs     never an L1000-derived input from the held-out cell; CCLE principal components are fit on the 131 LINCS CCLE
             cells (transductive in feature space only)
  models     M1 expression MLP [gene vector; cell features]; M2 = M1 + global chromatin + a perturbation-conditioned
             chromatin term per landmark; M3 recurrent gated propagation from ONE shared start vector at the knocked-down
             node over the binary STRING expdb graph, gates from each node's expression + chromatin; M4 = M3 with
             expression-only gates; M5 = M3 on the degree-preserving rewired graph. K = 3 (4 as a sensitivity run).
             The 31 landmarks with no graph edge are read from the mean-pooled node state.
  injection  d' = d + s u; u = phi(c, j) psi(g) (dense) or phi psi [A^3 e_g]_j / SD (graph); phi = ACT:prom_1kb residualised on
             CCLE expression within cell (raw ACT reported); psi ~ N(0, 1) per gene (default_rng(2)); s sets Var(s u) to a
             share of Var(target) over training rows; train and test targets alike.

    python model/cellrep/stage2.py --job '{"model": "M3", "setting": "loco:PC3", "seed": 0}' --roots ... --out_dir out/
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stage1 as s1  # noqa: E402

CORE = s1.CORE
TRAIN_CELLS = ('PC3', 'MCF7', 'A375', 'A549', 'HEPG2', 'HT29', 'HCC515', 'SW480', 'VCAP')   # knockdown cells with CCLE
N_PC = 50
D_HID = 512
D_NODE = 32
MAX_EPOCHS = 30
PATIENCE = 3


def find(name, roots):
    hits = sorted({h for r in roots for h in glob.glob(os.path.join(r, '**', name), recursive=True)})
    if not hits:
        raise SystemExit(f'FATAL: {name} not found under {roots}')
    return hits[0]


# --- the data -----------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, roots):
        z = np.load(find('l1000_genetic_lm978.npz', roots))
        self.Y = z['Y'].astype(np.float32)
        kd96 = (z['pert_type'] == 'trt_sh.cgs') & (z['pert_itime'] == '96 h')
        self.idx = s1.index_signatures(z['cell_id'], z['pert_iname'], kd96)
        self.lm_sym = [str(s) for s in z['landmark_symbol']]
        self.lm_pos = {s: i for i, s in enumerate(self.lm_sym)}
        gp = np.load(find('stage2_graph.npz', roots), allow_pickle=True)
        self.nodes = [str(s) for s in gp['nodes']]
        self.node_of = {s: i for i, s in enumerate(self.nodes)}
        self.lm_node = gp['landmark_idx'].astype(np.int64)
        self.edges = {'real': gp['edges'].astype(np.int64), 'rewired': gp['edges_rewired'].astype(np.int64)}
        deg = np.bincount(self.edges['real'].ravel(), minlength=len(self.nodes))
        self.lm_reach = deg[self.lm_node] > 0                              # 947 of 978
        self.gvec = np.load(find('gene_vectors_v11_full.npy', roots)).astype(np.float32)
        self.genes = sorted({g for g, _ in self.idx if g in self.node_of})
        ce = np.load(find('ccle_expr_v11.npz', roots), allow_pickle=True)
        epos = {str(c): i for i, c in enumerate(ce['cells'])}
        En = np.asarray(ce['expr_nodes'], np.float64)
        filled = np.where(np.isfinite(En), En, np.nanmean(En, 0, keepdims=True))
        filled = np.where(np.isfinite(filled), filled, 0.0)
        U, S, _ = np.linalg.svd((filled - filled.mean(0)) / (filled.std(0) + 1e-6), full_matrices=False)
        pcs = U[:, :N_PC] * S[:N_PC]                                       # fit on the 131 LINCS CCLE cells
        El = np.asarray(ce['expr_landmarks'], np.float64)
        ch_l = np.load(find('v10c_chromatin.npz', roots), allow_pickle=True)
        ch_n = np.load(find('v11_chromatin.npz', roots), allow_pickle=True)
        clp = {str(c): i for i, c in enumerate(ch_l['cells'])}
        cnp = {str(c): i for i, c in enumerate(ch_n['cells'])}
        ex = np.load(find('ccle_extra.npz', roots), allow_pickle=True)
        xp = {str(c): i for i, c in enumerate(ex['cells'])}
        self.fn = [str(f) for f in ch_l['feature_names']]
        N = len(self.nodes)
        self.cell_index = {c: k for k, c in enumerate(TRAIN_CELLS)}
        CE, CG, CL, CLM, CNE, CNC, self.act_lm, self.expr_lm = [], [], [], [], [], [], {}, {}
        for c in TRAIN_CELLS:
            el = np.nan_to_num(El[epos[c]], nan=0.0)
            Fl = np.nan_to_num(ch_l['F'][clp[c]].astype(np.float32), nan=0.0) if c in clp else np.zeros((978, 16), np.float32)
            Ml = ch_l['M'][clp[c]].astype(np.float32) if c in clp else np.zeros((978, 4), np.float32)
            Fn = np.nan_to_num(ch_n['F'][cnp[c]].astype(np.float32), nan=0.0) if c in cnp else np.zeros((N, 16), np.float32)
            Mn = ch_n['M'][cnp[c]].astype(np.float32) if c in cnp else np.zeros((N, 4), np.float32)
            gc = ex['global_chromatin'][xp[c]] if c in xp else np.full(42, np.nan)
            en = En[epos[c]]
            CE.append(np.r_[el, pcs[epos[c]]])
            CG.append(np.r_[np.nan_to_num(gc, nan=0.0), float(np.isfinite(gc).any())])
            CL.append(Fl)
            CLM.append(Ml)
            CNE.append(np.c_[np.nan_to_num(en, nan=0.0), np.isfinite(en)])
            CNC.append(np.c_[Fn, Mn])
            self.act_lm[c] = np.where(Ml[:, 1] > 0, Fl[:, self.fn.index('ACT:prom_1kb')], np.nan)
            self.expr_lm[c] = el
        self.cellfeat = {'CE': np.stack(CE).astype(np.float32), 'CG': np.stack(CG).astype(np.float32),
                         'CL': np.stack(CL), 'CLM': np.stack(CLM), 'CNE': np.stack(CNE).astype(np.float32),
                         'CNC': np.stack(CNC).astype(np.float32)}

    def loco(self, c0):
        tr, te, dropped = s1.fold_targets(self.Y, self.idx, c0, core=TRAIN_CELLS)
        keep = lambda rows: [(g, c, d) for g, c, d in rows if g in self.node_of]
        return keep(tr), keep(te), dropped

    def cold(self):
        rng = np.random.default_rng(0)
        test_g = set(rng.permutation(list(self.genes))[: int(round(0.2 * len(self.genes)))].tolist())
        tr = [(g, c, self.Y[i]) for (g, c), i in self.idx.items() if c in TRAIN_CELLS and g in self.node_of and g not in test_g]
        te = [(g, c, self.Y[i]) for (g, c), i in self.idx.items() if c in CORE and g in test_g]
        return tr, te, sorted(test_g)

    def d_reference(self, g, c):
        """Target-side d for a cold-gene test row: y minus the mean over g's other knockdown cells (>= 3), else None."""
        ref = [x for (gg, x) in self.idx if gg == g and x != c]
        if len(ref) < s1.MIN_REF:
            return None
        return self.Y[self.idx[(g, c)]] - self.Y[[self.idx[(g, x)] for x in ref]].mean(0)


def inner_split(rows, by, seed=1, frac=0.1):
    """Early-stopping split: 10 % of training rows (LOCO) or of training genes (cold), default_rng(seed)."""
    rng = np.random.default_rng(seed)
    if by == 'rows':
        m = np.zeros(len(rows), bool)
        m[rng.permutation(len(rows))[: int(round(frac * len(rows)))]] = True
        return m
    genes = sorted({g for g, _, _ in rows})
    vg = set(rng.permutation(genes)[: int(round(frac * len(genes)))].tolist())
    return np.array([g in vg for g, _, _ in rows])


# --- the injection ------------------------------------------------------------------------------------------------------
def phi_matrix(D, residualise=True):
    """phi(c, j): ACT:prom_1kb centred by landmark over the cells that have it, residualised on centred CCLE expression
    within cell (103.7 C4) unless residualise=False, standardised; 0 where missing."""
    A = np.stack([D.act_lm[c] for c in TRAIN_CELLS]).astype(np.float64)
    E = np.stack([D.expr_lm[c] for c in TRAIN_CELLS]).astype(np.float64)
    with np.errstate(all='ignore'):
        A = A - np.nanmean(A, 0, keepdims=True)
    E = E - E.mean(0, keepdims=True)
    out = np.zeros_like(A)
    for k in range(len(TRAIN_CELLS)):
        ok = np.isfinite(A[k])
        if ok.sum() < 10:
            continue
        a = A[k, ok]
        if residualise:
            X = np.c_[np.ones(ok.sum()), E[k, ok]]
            a = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
        out[k, ok] = a if a.std() > 1e-8 * (np.abs(A[k, ok]).std() + 1e-12) else 0.0   # no orthogonal variation: 0
    nz = out != 0
    out[nz] = (out[nz] - out[nz].mean()) / out[nz].std()
    return {c: out[k] for k, c in enumerate(TRAIN_CELLS)}


def graph_shape(D, A_norm, gene):
    """[A^3 e_g] at the landmark nodes."""
    v = np.zeros(len(D.nodes))
    v[D.node_of[gene]] = 1.0
    for _ in range(3):
        v = A_norm @ v
    return v[D.lm_node]


def inject(D, train, test, share, shape, residualise=True, A_norm=None):
    """Adds s u to every target (train and test); s puts Var(s u) at `share` of Var(target) over training rows."""
    phi = phi_matrix(D, residualise)
    genes = sorted({g for g, _, _ in train + test})
    psi = dict(zip(genes, np.random.default_rng(2).normal(size=len(genes))))
    gs = {}

    def u(g, c):
        base = phi[c] * psi[g]
        if shape == 'graph':
            if g not in gs:
                gs[g] = graph_shape(D, A_norm, g)
            base = base * gs[g]
        return base
    U_tr = np.stack([u(g, c) for g, c, _ in train])
    U_te = np.stack([u(g, c) for g, c, _ in test])
    T_tr = np.stack([t for _, _, t in train])
    s = np.sqrt(share * T_tr.var() / U_tr.var())
    new_tr = [(g, c, t + s * uu) for (g, c, t), uu in zip(train, U_tr)]
    new_te = [(g, c, t + s * uu) for (g, c, t), uu in zip(test, U_te)]
    return new_tr, new_te, s * U_te


# --- the models ---------------------------------------------------------------------------------------------------------
def norm_adj_scipy(edges, n):
    from scipy.sparse import coo_matrix
    e = np.asarray(edges)
    r, c = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    deg = np.bincount(r, minlength=n).astype(np.float64)
    return coo_matrix((1.0 / np.sqrt(deg[r] * deg[c]), (r, c)), shape=(n, n)).tocsr()


def norm_adj(edges, n):
    """Binary symmetric D^-1/2 A D^-1/2 as a torch sparse tensor."""
    a = norm_adj_scipy(edges, n).tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.stack([a.row, a.col])), torch.tensor(a.data, dtype=torch.float32),
                                   (n, n)).coalesce()


class CellFeatures(nn.Module):
    """The per-cell inputs, resident on the device; rows index them by cell."""
    def __init__(self, D):
        super().__init__()
        for k, v in D.cellfeat.items():
            self.register_buffer(k, torch.as_tensor(v))


class MLP1(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.cf = CellFeatures(D)
        self.net = nn.Sequential(nn.Linear(128 + 978 + N_PC, D_HID), nn.GELU(), nn.Dropout(0.1), nn.Linear(D_HID, D_HID),
                                 nn.GELU(), nn.Linear(D_HID, 978))

    def forward(self, b):
        return self.net(torch.cat([b['gvec'], self.cf.CE[b['cell']]], 1))


class MLP2(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.cf = CellFeatures(D)
        self.trunk = nn.Sequential(nn.Linear(128 + 978 + N_PC + 43, D_HID), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(D_HID, D_HID), nn.GELU())
        self.a = nn.Linear(D_HID, 978)
        self.b = nn.Linear(D_HID, 20)
        self.s = nn.Parameter(torch.zeros(978))

    def forward(self, b):
        cc = b['chrom_cell']
        h = self.trunk(torch.cat([b['gvec'], self.cf.CE[b['cell']], self.cf.CG[cc]], 1))
        ch = torch.cat([self.cf.CL[cc], self.cf.CLM[cc]], 2)                       # [B, 978, 20]
        if b.get('perm_l') is not None:
            ch = ch[:, b['perm_l']]
        return self.a(h) + self.s * torch.einsum('bjk,bk->bj', ch, self.b(h))


class GraphProp(nn.Module):
    """M3 / M4 / M5: one shared start vector at the knocked-down node; K gated steps X <- X + g (A X W_k)."""
    def __init__(self, D, A, K=3, chrom_gates=True):
        super().__init__()
        self.cf = CellFeatures(D)
        self.register_buffer('A', A)
        self.register_buffer('lm_node', torch.as_tensor(D.lm_node))
        self.register_buffer('lm_reach', torch.as_tensor(D.lm_reach))
        self.K, self.chrom_gates = K, chrom_gates
        self.start = nn.Parameter(torch.randn(D_NODE) * 0.1)
        self.W = nn.ParameterList([nn.Parameter(torch.eye(D_NODE) + 0.01 * torch.randn(D_NODE, D_NODE)) for _ in range(K)])
        self.gate = nn.Sequential(nn.Linear(2 + (20 if chrom_gates else 0), 16), nn.GELU(), nn.Linear(16, 1))
        self.w = nn.Linear(D_NODE, 1)
        self.c = nn.Parameter(torch.zeros(978))
        self.pool = nn.Linear(D_NODE, 978)

    def cell_gates(self, b):
        """Gates for each distinct (cell, chromatin cell) pair in the batch, indexed back to rows: [B, N]."""
        pairs = torch.stack([b['cell'], b['chrom_cell']], 1)
        uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
        x = self.cf.CNE[uniq[:, 0]]
        if self.chrom_gates:
            ch = self.cf.CNC[uniq[:, 1]]
            if b.get('perm_n') is not None:
                ch = ch[:, b['perm_n']]
            x = torch.cat([x, ch], 2)
        return torch.sigmoid(self.gate(x)).squeeze(-1)[inv]

    def forward(self, b, return_gates=False):
        B, N = b['node'].shape[0], self.A.shape[0]
        g = self.cell_gates(b)
        X = torch.zeros(B, N, D_NODE, device=self.A.device)
        X[torch.arange(B, device=X.device), b['node']] = self.start
        for k in range(self.K):
            AX = torch.sparse.mm(self.A, X.permute(1, 0, 2).reshape(N, B * D_NODE)).reshape(N, B, D_NODE).permute(1, 0, 2)
            X = X + g.unsqueeze(-1) * (AX @ self.W[k])
        node_read = self.w(X[:, self.lm_node]).squeeze(-1)
        out = torch.where(self.lm_reach, node_read, self.pool(X.mean(1))) + self.c
        return (out, g) if return_gates else out


def build_model(name, D, K=3, device='cpu'):
    if name == 'M1':
        return MLP1(D).to(device)
    if name == 'M2':
        return MLP2(D).to(device)
    A = norm_adj(D.edges['rewired' if name == 'M5' else 'real'], len(D.nodes))
    return GraphProp(D, A, K=K, chrom_gates=(name != 'M4')).to(device)


def batch(D, rows, device, swap_cell=None, perm=None):
    """Row indices for rows [(g, c, target)]; swap_cell and perm are rung 4's usage variants (inference only): the held-out
    cell's chromatin replaced by another cell's, or shuffled across landmarks / nodes."""
    cells = [D.cell_index[c] for _, c, _ in rows]
    chrom = [D.cell_index[swap_cell] for _ in rows] if swap_cell else cells
    nodes = [D.node_of[g] for g, _, _ in rows]
    b = {'gvec': torch.as_tensor(D.gvec[nodes]).to(device), 'node': torch.as_tensor(nodes).to(device),
         'cell': torch.as_tensor(cells).to(device), 'chrom_cell': torch.as_tensor(chrom).to(device),
         'y': torch.as_tensor(np.stack([np.asarray(r[2], np.float32) for r in rows])).to(device)}
    if perm is not None:
        b['perm_l'], b['perm_n'] = torch.as_tensor(perm[0]).to(device), torch.as_tensor(perm[1]).to(device)
    return b


def row_scores(D, rows, P):
    """Per-row r over all 978 landmarks and over the 947 reachable ones, own coordinate excluded."""
    r_all, r_reach = [], []
    for (g, _, t), p in zip(rows, P):
        sk = D.lm_pos.get(g)
        r_all.append(s1.row_r(p, t, sk))
        keep = D.lm_reach.copy()
        if sk is not None:
            keep[sk] = False
        r_reach.append(s1.row_r(np.asarray(p)[keep], np.asarray(t)[keep]))
    return np.array(r_all), np.array(r_reach)


def predict(model, D, rows, device, bs, **kw):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(rows), bs):
            out.append(model(batch(D, rows[k:k + bs], device, **kw)).float().cpu().numpy())
    return np.concatenate(out)


def train(model, D, tr, va, device, bs, seed, lr=1e-3):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -np.inf, None, 0, []
    for ep in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(tr))
        for k in range(0, len(order), bs):
            b = batch(D, [tr[i] for i in order[k:k + bs]], device)
            loss = nn.functional.mse_loss(model(b), b['y'])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = float(np.mean(row_scores(D, va, predict(model, D, va, device, bs))[0]))
        hist.append(v)
        if v > best:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return hist


def job_tag(job):
    inj = job.get('inject')
    parts = [job['model'], job['setting'].replace(':', '-'), f"s{job['seed']}", f"K{job.get('K', 3)}"]
    if inj:
        parts.append(f"inj-{inj['shape']}-{inj['share']}" + ('' if inj.get('residualise', True) else '-raw'))
    return '_'.join(parts)


def run_job(job, roots, out_dir, device, D=None):
    t0 = time.time()
    D = D or Data(roots)
    setting, name, seed, K = job['setting'], job['model'], int(job['seed']), int(job.get('K', 3))
    loco = setting.startswith('loco:')
    if loco:
        c0 = setting.split(':')[1]
        tr, te, dropped = D.loco(c0)
    else:
        tr, te, _ = D.cold()
        dropped = 0
    inj, inj_te = job.get('inject'), None
    if inj:
        A_s = norm_adj_scipy(D.edges['real'], len(D.nodes)) if inj['shape'] == 'graph' else None
        tr, te, inj_te = inject(D, tr, te, float(inj['share']), inj['shape'], inj.get('residualise', True), A_s)
    m = inner_split(tr, 'rows' if loco else 'genes')
    va, tr2 = [r for r, x in zip(tr, m) if x], [r for r, x in zip(tr, m) if not x]
    torch.manual_seed(seed)
    model = build_model(name, D, K, device)
    bs = 256 if name in ('M1', 'M2') else 32
    hist = train(model, D, tr2, va, device, bs, seed)
    P = predict(model, D, te, device, bs)
    r_all, r_reach = row_scores(D, te, P)
    rec = {'job': job, 'n_train': len(tr2), 'n_val': len(va), 'n_test': len(te), 'dropped_min_ref': int(dropped),
           'val_curve': hist, 'mean_r': float(r_all.mean()), 'mean_r_reachable': float(r_reach.mean()),
           'rows': [[g, c] for g, c, _ in te], 'r': r_all.round(6).tolist(), 'r_reachable': r_reach.round(6).tolist()}
    if inj_te is not None:
        rec['injected_recovery_r'] = [round(s1.row_r(p, u), 6) for p, u in zip(P, inj_te)]
    if not loco:
        dref = [D.d_reference(g, c) for g, c, _ in te]
        rec['r_on_d'] = [round(s1.row_r(p - (np.asarray(t) - dd), dd, D.lm_pos.get(g)), 6) if dd is not None else None
                         for (g, _, t), p, dd in zip(te, P, dref)]
    if loco and not inj:                                                   # rung 4: usage variants, inference only
        other = [c for c in CORE if c != c0][0]
        rng = np.random.default_rng(4)
        perm = (rng.permutation(978), rng.permutation(len(D.nodes)))
        rec['usage'] = {f'swap_chromatin_with_{other}': float(row_scores(D, te, predict(model, D, te, device, bs, swap_cell=other))[0].mean()),
                        'shuffle_chromatin': float(row_scores(D, te, predict(model, D, te, device, bs, perm=perm))[0].mean())}
        if isinstance(model, GraphProp):
            with torch.no_grad():
                _, gts = model(batch(D, te[:32], device), return_gates=True)
            rec['usage'].update(gate_mean=float(gts.mean()), gate_sd=float(gts.std()))
    rec['elapsed_s'] = round(time.time() - t0)
    os.makedirs(out_dir, exist_ok=True)
    tag = job_tag(job)
    json.dump(rec, open(os.path.join(out_dir, tag + '.json'), 'w'))
    np.save(os.path.join(out_dir, tag + '_pred.npy'), P.astype(np.float16))
    return tag, rec['mean_r']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    print(run_job(json.loads(a.job), a.roots, a.out_dir, a.device))


if __name__ == '__main__':
    main()
