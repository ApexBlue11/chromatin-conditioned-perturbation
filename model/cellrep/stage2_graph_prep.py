"""Stage 2's outcome-free graph inputs (RESULTS 103.6 / 103.7): the binary symmetric STRING expdb graph, its
degree-preserving rewiring for M5 (networkx double_edge_swap, 10 x |E| swaps, seed 3), and hop distances (<= 6) from every
96 h knocked-down gene to the 978 landmark nodes.

    python -I model/cellrep/stage2_graph_prep.py --out external/cellrep_data/stage2_graph.npz
"""
import argparse, glob, hashlib, json, os, sys

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

GRAPH = 'external/kaggle_alt2_stage/lincs-v11-src/string_graph_v11_expdb.npz'
KD = 'external/kaggle_out/genetic_extract/l1000_genetic_lm978.npz'


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def undirected_edges(edge_index):
    a, b = np.minimum(edge_index[0], edge_index[1]), np.maximum(edge_index[0], edge_index[1])
    keep = a != b
    e = np.unique(np.stack([a[keep], b[keep]], 1), axis=0)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    g = np.load(GRAPH, allow_pickle=True)
    nodes = [str(x) for x in g['nodes']]
    N = len(nodes)
    E = undirected_edges(g['edge_index'].astype(np.int64))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(map(tuple, E))
    deg0 = np.array([d for _, d in sorted(G.degree())])
    R = G.copy()
    nx.double_edge_swap(R, nswap=10 * len(E), max_tries=200 * len(E), seed=3)
    Er = undirected_edges(np.array(list(R.edges())).T)
    degr = np.array([d for _, d in sorted(R.degree())])
    kept = len(set(map(tuple, E)) & set(map(tuple, Er))) / len(E)

    z = np.load(KD)
    m = (z['pert_type'] == 'trt_sh.cgs') & (z['pert_itime'] == '96 h')
    pos = {s: i for i, s in enumerate(nodes)}
    kd_genes = sorted({s for s in z['pert_iname'][m].tolist() if s in pos})
    lm = g['landmark_idx'].astype(np.int64)
    A = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(N, N)).tocsr()
    D = np.full((len(kd_genes), len(lm)), 127, np.int8)
    src = np.array([pos[s] for s in kd_genes])
    for k in range(0, len(src), 256):
        sp = dijkstra(A, directed=False, unweighted=True, indices=src[k:k + 256], limit=6)
        sub = sp[:, lm]
        D[k:k + 256] = np.where(np.isfinite(sub), sub, 127).astype(np.int8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, nodes=np.array(nodes), edges=E.astype(np.int32), edges_rewired=Er.astype(np.int32),
                        landmark_idx=lm.astype(np.int32), kd_genes=np.array(kd_genes), hop=D)
    within = [(D <= k).mean(1) for k in (1, 2, 3, 4)]
    rec = {'nodes': N, 'edges': int(len(E)), 'edges_rewired': int(len(Er)), 'degree_preserved': bool((deg0 == degr).all()),
           'kept_edge_fraction': float(kept), 'kd_genes_in_graph': len(kd_genes),
           'landmarks_within_k_hops_median': {k: float(np.median(w)) for k, w in zip((1, 2, 3, 4), within)},
           'unreachable_pairs_within_6': float((D == 127).mean()),
           'inputs': {os.path.basename(p): sha1(p) for p in (GRAPH, KD)}, 'out_sha1': sha1(a.out)}
    json.dump(rec, open(a.out.replace('.npz', '_record.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != 'inputs'}, indent=1))


if __name__ == '__main__':
    main()
