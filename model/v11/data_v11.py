"""v11 data (RESULTS 102 / 102.1): v10-C's data exactly as T-B's off arm (training-mean chromatin at the landmarks), so the
base sees byte-identical inputs, plus the tables of the full-graph module FG. Every new source enters FG only (102.1 C1).

FG's tables, outcome-free (no response data is read to build them):
  graph     the experimental + database STRING graph over 19,496 genes, symmetric-normalised D^-1/2 A D^-1/2
  gene vec  its re-pretrained gene vectors, standardised per column
  chromatin v10-C's encoding at every gene (16 features + 4 axis masks), per cell; the -chromatin arm gets the
            training-row-weighted mean of F and of M for every cell (100.4's convention)
  CCLE      per cell, the within-cell rank in [0, 1] of log2(TPM + 1) over the genes it measures, and a mask; zero and
            masked for cells without CCLE and, in the -CCLE arm, for every cell
  targets   ChEMBL mechanism targets (Homo sapiens, genes in the graph) with the frozen action-sign map; a compound's
            known targets are an input ("known") iff it is annotated, not a probe compound (dti_probe_split.json) and not
            unseen (it has no training row) or a same-molecule duplicate (InChIKey first block) of an unseen compound;
            the structure head trains on exactly those compounds (102 item 2). Every other compound is "predicted".
"""
import collections, csv, json

import numpy as np
import scipy.sparse as sp
import torch

from data_v10c import V10CData, find

ARMS = ('full', 'no_chrom', 'no_targets', 'no_ccle')
GENE_VEC_FG = 'gene_vectors_v11_full.npy'   # network/outputs/v11/gene_vectors_full.npy, renamed: v9 has a file of that name
STATUS = ('known', 'probe', 'unseen_annotated', 'unannotated')   # status codes 0..3 (102.1 C5 strata)


def ccle_ranks(expr):
    """[C, N] log2(TPM + 1) with NaN where unmeasured -> within-row rank in [0, 1] over finite entries (te2_tests'
    rank_rows_nan convention) and the finite mask."""
    E = np.zeros(expr.shape, np.float32)
    m = np.isfinite(expr)
    for i in range(expr.shape[0]):
        f = m[i]
        if f.sum() > 1:
            E[i, f] = np.argsort(np.argsort(expr[i, f], kind='stable'), kind='stable') / (f.sum() - 1)
    return E, m


def sym_norm(edge_index, weight, n):
    """Symmetric A from an i < j edge list, D^-1/2 A D^-1/2 with isolated nodes left at zero."""
    i, j = edge_index
    A = sp.coo_matrix((np.r_[weight, weight], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    return (sp.diags(dinv) @ A @ sp.diags(dinv)).tocoo()


def combine_signs(signs):
    """One (compound, gene) pair's ChEMBL rows -> one sign: all +1 -> +1, all -1 -> -1, anything else 0."""
    s = set(signs)
    return 1.0 if s == {1} else (-1.0 if s == {-1} else 0.0)


def hop_scale(A, t_any, t_sign, landmark_idx):
    """Fixed per-channel scales of the target signal (outcome-free): for hops 0, 1, 2 of the any and sign channels over
    the ungated graph, the RMS of the nonzero landmark entries over the known compounds. Order: any0, sign0, any1, sign1,
    any2, sign2 (the order FG concatenates)."""
    out = []
    X = np.stack([t_any, t_sign], -1)                      # [P, N, 2]
    hops = [X]
    for _ in range(2):
        hops.append(np.stack([(A @ hops[-1][..., c].T).T for c in range(2)], -1))
    for h in hops:
        for c in range(2):
            v = h[:, landmark_idx, c]
            v = v[v != 0]
            out.append(float(np.sqrt(np.mean(v ** 2))) if v.size else 1.0)
    return np.array(out, np.float32)


class V11Data(V10CData):
    def __init__(self, roots, split='cold_cell_1', dev=False, arm='full', substrate=None, chromatin=None):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(roots, split=split, dev=dev, chrom_off=True, substrate=substrate, chromatin=chromatin)
        self.arm = arm
        base_cells = [str(c) for c in np.load(chromatin or find('v10c_chromatin.npz', roots))['cells']]
        g = np.load(find('string_graph_v11_expdb.npz', roots))
        nodes = [str(x) for x in g['nodes']]
        N = self.n_nodes = len(nodes)
        self.landmark_idx = g['landmark_idx'].astype(np.int64)
        if len(self.landmark_idx) != self.X.shape[1] or len(set(self.landmark_idx.tolist())) != self.X.shape[1]:
            raise SystemExit('FATAL: the graph does not map the 978 landmarks one-to-one')
        A = sym_norm(g['edge_index'], g['weight'].astype(np.float64), N)
        self.A_index = np.stack([A.row, A.col]).astype(np.int64)
        self.A_value = A.data.astype(np.float32)
        gv = np.load(find(GENE_VEC_FG, roots)).astype(np.float32)
        if gv.shape[0] != N:
            raise SystemExit('FATAL: gene vectors do not match the graph')
        self.gene_vec = (gv - gv.mean(0)) / (gv.std(0) + 1e-6)

        ch = np.load(find('v11_chromatin.npz', roots))
        if [str(c) for c in ch['cells']] != base_cells or [str(x) for x in ch['genes']] != nodes:
            raise SystemExit('FATAL: v11 chromatin is not in the base cell order and the graph node order')
        F = ch['F'].astype(np.float16)
        M = ch['M'].astype(np.float16)
        if arm == 'no_chrom':
            w = np.bincount(self.cell_row[self.tr], minlength=len(base_cells)).astype(np.float64)
            w /= w.sum()
            Fm = np.zeros(F.shape[1:], np.float64)
            Mm = np.zeros(M.shape[1:], np.float64)
            for i in np.flatnonzero(w):
                Fm += w[i] * F[i].astype(np.float64)
                Mm += w[i] * M[i].astype(np.float64)
            F = np.broadcast_to(Fm.astype(np.float16), F.shape).copy()
            M = np.broadcast_to(Mm.astype(np.float16), M.shape).copy()
        self.fg_F, self.fg_M = F, M

        cz = np.load(find('ccle_expr_v11.npz', roots))
        if [str(x) for x in cz['nodes']] != nodes:
            raise SystemExit('FATAL: CCLE is not in the graph node order')
        expr = np.full((len(base_cells), N), np.nan, np.float32)
        cidx = {c: i for i, c in enumerate(base_cells)}
        for j, c in enumerate(cz['cells']):
            expr[cidx[str(c)]] = cz['expr_nodes'][j]
        E, Em = ccle_ranks(expr)
        self.ccle_cells = sorted(str(c) for c in cz['cells'])
        if arm == 'no_ccle':
            E, Em = np.zeros_like(E), np.zeros_like(Em)
        self.fg_E, self.fg_Em = E.astype(np.float16), Em.astype(np.float16)

        self._targets(roots, nodes)

    def _targets(self, roots, nodes):
        nid = {n: i for i, n in enumerate(nodes)}
        sign_map = json.load(open(find('action_sign_map.json', roots), encoding='utf-8'))['sign']
        probe = set(json.load(open(find('dti_probe_split.json', roots), encoding='utf-8'))['probe'])
        block1 = {r['pert_id']: r['block1'] for r in
                  csv.DictReader(open(find('drug_inchikeys.tsv', roots), encoding='utf-8'), delimiter='\t') if r.get('block1')}
        pairs = collections.defaultdict(list)
        for r in csv.DictReader(open(find('chembl_dti_edges.tsv', roots), encoding='utf-8'), delimiter='\t'):
            if r['organism'] == 'Homo sapiens' and r['gene_symbol'] in nid:
                pairs[(r['pert_id'], nid[r['gene_symbol']])].append(int(sign_map.get(r['action_type'], 0)))
        tg = collections.defaultdict(dict)
        for (p, n), s in pairs.items():
            tg[p][n] = combine_signs(s)
        uniq = np.unique(self.pert)                           # pert_code indexes this
        train_perts = set(self.pert[self.tr])
        unseen = set(uniq) - train_perts
        unseen_mol = {block1[p] for p in unseen if p in block1}
        status = np.full(len(uniq), 3, np.int8)
        for k, p in enumerate(uniq):
            if p not in tg:
                continue
            if p in probe:
                status[k] = 1
            elif p in unseen or block1.get(p) in unseen_mol:
                status[k] = 2
            else:
                status[k] = 0
        self.status = status
        self.n_unseen_compounds = len(unseen)
        known = np.flatnonzero(status == 0)
        self.vocab = np.array(sorted({n for k in known for n in tg[uniq[k]]}), np.int64)
        vpos = {n: i for i, n in enumerate(self.vocab)}
        T = max([len(v) for v in tg.values()] + [1])
        self.tgt_node = np.full((len(uniq), T), self.n_nodes, np.int64)       # pad: a dummy node
        self.tgt_sign = np.zeros((len(uniq), T), np.float32)
        self.tgt_vocab = np.full((len(uniq), T), -1, np.int64)
        self.tgt_count = np.zeros(len(uniq), np.int64)
        for k, p in enumerate(uniq):
            if p in tg and status[k] != 1:                   # probe compounds' targets never enter (102 item 2)
                for j, (n, s) in enumerate(sorted(tg[p].items())):
                    self.tgt_node[k, j], self.tgt_sign[k, j] = n, s
                    self.tgt_vocab[k, j] = vpos.get(n, -1)
                self.tgt_count[k] = len(tg[p])
        self.known_input = status == 0
        if self.arm == 'no_targets':
            self.known_input = np.zeros_like(self.known_input)
        t_any = np.zeros((len(known), self.n_nodes), np.float32)
        t_sign = np.zeros((len(known), self.n_nodes), np.float32)
        for i, k in enumerate(known):
            m = self.tgt_node[k] < self.n_nodes
            t_any[i, self.tgt_node[k][m]] = 1.0
            t_sign[i, self.tgt_node[k][m]] = self.tgt_sign[k][m]
        prior = np.zeros(len(self.vocab), np.float64)
        for k in known:
            v = self.tgt_vocab[k][:self.tgt_count[k]]
            prior[v[v >= 0]] += 1.0
        self.vocab_prior = (prior / max(1, len(known))).astype(np.float32)
        A = sp.coo_matrix((self.A_value, (self.A_index[0], self.A_index[1])), shape=(self.n_nodes,) * 2).tocsr()
        self.s_scale = hop_scale(A, t_any, t_sign, self.landmark_idx)

    def fg_inputs(self):
        """Everything LincsV11 needs to build FG (numpy; the model moves it to each device once)."""
        return {'A_index': self.A_index, 'A_value': self.A_value, 'n_nodes': self.n_nodes, 'gene_vec': self.gene_vec,
                'F': self.fg_F, 'M': self.fg_M, 'E': self.fg_E, 'Em': self.fg_Em, 'landmark_idx': self.landmark_idx,
                's_scale': self.s_scale, 'vocab': self.vocab, 'vocab_prior': self.vocab_prior,
                'targets_on': self.arm != 'no_targets'}

    def batch(self, idx, device, max_atoms=96):
        b = super().batch(idx, device, max_atoms)
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(device)
        k = self.pert_code[idx]
        T = max(1, int(self.tgt_count[k].max()))
        known = self.known_input[k]
        y = np.zeros((len(idx), len(self.vocab)), np.float32)
        for r, kk in enumerate(k):
            if known[r]:
                v = self.tgt_vocab[kk][:self.tgt_count[kk]]
                y[r, v[v >= 0]] = 1.0
        b.update({'cell_row': t(self.cell_row[idx].astype(np.int64)), 'tgt_node': t(self.tgt_node[k, :T]),
                  'tgt_sign': t(self.tgt_sign[k, :T]), 'tgt_known': t(known), 'head_y': t(y)})
        return b
