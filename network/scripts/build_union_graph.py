"""
build_union_graph.py

Builds a unioned gene-gene graph over 978 landmark genes from STRING, Reactome, and GO:BP.
Multi-hot provenance allows downstream tasks to trace the origin of each edge.
We do not normalise the three weights onto one scale; they are different quantities 
and a consumer should decide how to combine them.
"""

import numpy as np
import argparse
import json
import hashlib
import os

def sha1(filepath):
    h = hashlib.sha1()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--string_thresh', type=float, default=0.0)
    parser.add_argument('--min_term', type=int, default=5)
    parser.add_argument('--max_term', type=int, default=200)
    parser.add_argument('--out_dir', type=str, default='network/outputs/v9')
    args = parser.parse_args()

    v9_dir = args.out_dir
    data_dir = 'network/data'
    
    string_npy = 'network/outputs/v9/STRING_adj_978_v9.npy'
    m_pathway_npy = 'network/outputs/v9/M_pathway_v9.npy'
    pathway_tsv = 'network/outputs/v9/pathway_info_v9.tsv'
    landmark_tsv = 'network/outputs/v9/landmark_symbols_v9.tsv'
    go_gmt = 'network/data/GO_Biological_Process_2023.gmt'

    hashes = {
        'STRING_adj_978_v9.npy': sha1(string_npy),
        'M_pathway_v9.npy': sha1(m_pathway_npy),
        'pathway_info_v9.tsv': sha1(pathway_tsv),
        'landmark_symbols_v9.tsv': sha1(landmark_tsv),
        'GO_Biological_Process_2023.gmt': sha1(go_gmt)
    }

    # Load landmarks
    nodes = []
    with open(landmark_tsv, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        hgnc_idx = header.index('l1000_symbol')
        for line in f:
            parts = line.strip('\n').split('\t')
            nodes.append(parts[hgnc_idx])
    nodes = np.array(nodes, dtype=str)
    n_nodes = len(nodes)
    
    # Wait, the prompt says GO GMT uses HGNC symbols. We better map using HGNC symbol, but keep the nodes exactly as l1000_symbol since it's the model's preferred.
    # Let me actually read both:
    hgnc_nodes = []
    with open(landmark_tsv, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        hgnc_idx = header.index('hgnc_symbol')
        for line in f:
            parts = line.strip('\n').split('\t')
            hgnc_nodes.append(parts[hgnc_idx])
            
    sym2idx = {sym: i for i, sym in enumerate(hgnc_nodes)}
    
    print("Loaded 978 landmarks.", flush=True)

    # 1. STRING
    string_adj = np.load(string_npy)
    string_mask = (string_adj > args.string_thresh) & np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    string_i, string_j = np.where(string_mask)
    string_w = string_adj[string_i, string_j]
    
    # 2. Reactome
    M_react = np.load(m_pathway_npy)
    react_sizes = M_react.sum(axis=1)
    react_keep = (react_sizes >= args.min_term) & (react_sizes <= args.max_term)
    M_react_filt = M_react[react_keep]
    react_co = M_react_filt.T @ M_react_filt
    react_mask = (react_co > 0) & np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    react_i, react_j = np.where(react_mask)
    react_w = react_co[react_i, react_j].astype(np.float32)

    # 3. GO:BP
    go_term_count = 0
    go_term_filtered_count = 0
    M_go_list = []
    with open(go_gmt, 'r', encoding='utf-8') as f:
        for line in f:
            go_term_count += 1
            parts = line.strip('\n').split('\t')
            genes = parts[2:]
            idx_list = [sym2idx[g] for g in genes if g in sym2idx]
            if args.min_term <= len(idx_list) <= args.max_term:
                go_term_filtered_count += 1
                row = np.zeros(n_nodes, dtype=np.int8)
                row[idx_list] = 1
                M_go_list.append(row)
                
    if M_go_list:
        M_go = np.stack(M_go_list)
        go_co = M_go.T @ M_go
        go_mask = (go_co > 0) & np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
        go_i, go_j = np.where(go_mask)
        go_w = go_co[go_i, go_j].astype(np.float32)
    else:
        go_i = np.array([], dtype=np.int64)
        go_j = np.array([], dtype=np.int64)
        go_w = np.array([], dtype=np.float32)

    # Union
    prov_full = np.zeros((n_nodes, n_nodes, 3), dtype=np.uint8)
    weight_full = np.zeros((n_nodes, n_nodes, 3), dtype=np.float32)
    
    prov_full[string_i, string_j, 0] = 1
    weight_full[string_i, string_j, 0] = string_w
    prov_full[react_i, react_j, 1] = 1
    weight_full[react_i, react_j, 1] = react_w
    prov_full[go_i, go_j, 2] = 1
    weight_full[go_i, go_j, 2] = go_w
    
    union_mask = prov_full.sum(axis=2) > 0
    u_i, u_j = np.where(union_mask)
    
    edge_index = np.vstack((u_i, u_j)).astype(np.int32)
    provenance = prov_full[u_i, u_j]
    weights = weight_full[u_i, u_j]
    
    npz_path = os.path.join(args.out_dir, 'union_graph_v9.npz')
    np.savez_compressed(
        npz_path,
        edge_index=edge_index,
        provenance=provenance,
        weights=weights,
        nodes=nodes,
        n_nodes=n_nodes
    )
    print(f"Saved {npz_path}", flush=True)

    def calc_density(n_edges):
        return (n_edges * 2) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

    s_mask = prov_full[:, :, 0] > 0
    r_mask = prov_full[:, :, 1] > 0
    g_mask = prov_full[:, :, 2] > 0

    def jaccard(m1, m2):
        inter = np.sum(m1 & m2)
        union = np.sum(m1 | m2)
        return float(inter / union) if union > 0 else 0.0

    prov_counts = {}
    for c in range(1, 8):
        match = (provenance[:, 0] == (c & 1)) & \
                (provenance[:, 1] == ((c >> 1) & 1)) & \
                (provenance[:, 2] == ((c >> 2) & 1))
        # The brief asks for the seven non-empty combinations. We can name them descriptively
        name_parts = []
        if (c & 1): name_parts.append("STRING")
        if (c >> 1) & 1: name_parts.append("Reactome")
        if (c >> 2) & 1: name_parts.append("GO")
        prov_counts["_".join(name_parts)] = int(np.sum(match))

    report = {
        "parameters": {
            "string_thresh": args.string_thresh,
            "min_term": args.min_term,
            "max_term": args.max_term
        },
        "file_sha1s": hashes,
        "landmark_order_sha1": hashes['landmark_symbols_v9.tsv'],
        "sources": {
            "STRING": {
                "raw_edge_count": len(string_i), 
                "edges_contributed": len(string_i),
                "density": calc_density(len(string_i))
            },
            "Reactome": {
                "raw_term_count": int(M_react.shape[0]),
                "surviving_terms": int(np.sum(react_keep)),
                "edges_contributed": len(react_i),
                "density": calc_density(len(react_i))
            },
            "GO_BP": {
                "raw_term_count": go_term_count,
                "surviving_terms": go_term_filtered_count,
                "edges_contributed": len(go_i),
                "density": calc_density(len(go_i))
            }
        },
        "union_edges": int(len(u_i)),
        "provenance_combinations": prov_counts,
        "jaccard_overlap": {
            "STRING_Reactome": jaccard(s_mask, r_mask),
            "STRING_GO": jaccard(s_mask, g_mask),
            "Reactome_GO": jaccard(r_mask, g_mask)
        }
    }

    json_path = os.path.join(args.out_dir, 'union_graph_v9_provenance.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()
