"""v11's STRING graph (RESULTS 101.1 C6; review 093): experimental + database channels only, no co-expression, text-mining,
neighbourhood, fusion or co-occurrence. Outcome-free.

Score: STRING's own channel combination, restricted to the two channels -- each channel score s (in [0, 1]) has the prior
p = 0.041 removed, s' = max(0, (s - p) / (1 - p)); the channels combine as 1 - (1 - e')(1 - d'); the prior is added back,
c = c' + p (1 - c'). Edges with c >= 0.400 are kept (the v9 cut, on the combined score).
Nodes: v9's node list (19,496 genes, aligned with model/results/v11/tss_hg38_string.tsv), so every feature stays aligned;
a gene with no kept edge is an isolated node.

    python -I network/scripts/build_string_v11.py --detailed ... --info network/data/9606.protein.info.v12.0.txt \
        --v9 network/outputs/v9/string_graph_v9.npz --out network/outputs/v11/string_graph_v11_expdb.npz
"""
import argparse, gzip, hashlib, json, os

import numpy as np

PRIOR = 0.041
THRESH = 0.400


def combine(e, d, p=PRIOR):
    e2, d2 = max(0.0, (e - p) / (1 - p)), max(0.0, (d - p) / (1 - p))
    c = 1 - (1 - e2) * (1 - d2)
    return c + p * (1 - c)


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detailed', required=True)
    ap.add_argument('--info', required=True)
    ap.add_argument('--v9', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    v9 = np.load(a.v9, allow_pickle=True)
    nodes = [str(x) for x in v9['nodes']]
    idx = {g: i for i, g in enumerate(nodes)}
    ensp2sym = {}
    with open(a.info, encoding='utf-8') as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip('\n').split('\t')
            ensp2sym[p[0]] = p[1]
    edges, w, seen, n_pairs, n_kept_any = [], [], set(), 0, 0
    with gzip.open(a.detailed, 'rt', encoding='utf-8') as fh:
        head = fh.readline().split()
        ce, cd = head.index('experimental'), head.index('database')
        for line in fh:
            f = line.split()
            n_pairs += 1
            e, d = int(f[ce]) / 1000.0, int(f[cd]) / 1000.0
            if e == 0 and d == 0:
                continue
            c = combine(e, d)
            if c < THRESH:
                continue
            sx, sy = ensp2sym.get(f[0]), ensp2sym.get(f[1])
            if sx is None or sy is None or sx == sy or sx not in idx or sy not in idx:
                continue
            u, v = sorted((idx[sx], idx[sy]))
            if (u, v) in seen:
                continue
            seen.add((u, v))
            edges.append((u, v))
            w.append(c)
    E = np.array(edges, np.int32).T
    W = np.array(w, np.float32)
    deg = np.bincount(E.ravel(), minlength=len(nodes))
    lm = v9['landmark_idx']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, nodes=np.array(nodes), edge_index=E, weight=W, landmark_idx=lm)
    v9E = set(map(tuple, np.sort(v9['edge_index'].T, 1).tolist()))
    mine = set(map(tuple, np.sort(E.T, 1).tolist()))
    prov = {'rule': 'RESULTS 101.1 C6: experimental + database channels, prior-corrected combination, c >= 0.400',
            'inputs': {os.path.basename(a.detailed): sha1(a.detailed), os.path.basename(a.info): sha1(a.info),
                       os.path.basename(a.v9): sha1(a.v9)},
            'prior': PRIOR, 'threshold': THRESH, 'pairs_in_file': n_pairs, 'nodes': len(nodes), 'edges': int(E.shape[1]),
            'isolated_nodes': int((deg == 0).sum()), 'isolated_landmarks': int((deg[lm] == 0).sum()),
            'edges_shared_with_v9_combined': len(mine & v9E), 'v9_edges': len(v9E),
            'landmark_landmark_edges': int(np.isin(E, lm).all(0).sum()),
            'out_sha1': sha1(a.out)}
    json.dump(prov, open(a.out.replace('.npz', '_provenance.json'), 'w'), indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == '__main__':
    main()
