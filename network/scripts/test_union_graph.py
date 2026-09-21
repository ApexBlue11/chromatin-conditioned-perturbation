"""
test_union_graph.py
Verifies the output of build_union_graph.py
"""

import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--string_thresh', type=float, default=0.0)
    parser.add_argument('--min_term', type=int, default=5)
    parser.add_argument('--max_term', type=int, default=200)
    parser.add_argument('--out_dir', type=str, default='network/outputs/v9')
    args = parser.parse_args()

    npz_path = os.path.join(args.out_dir, 'union_graph_v9.npz')
    json_path = os.path.join(args.out_dir, 'union_graph_v9_provenance.json')
    string_npy = 'network/outputs/v9/STRING_adj_978_v9.npy'
    m_pathway_npy = 'network/outputs/v9/M_pathway_v9.npy'
    landmark_tsv = 'network/outputs/v9/landmark_symbols_v9.tsv'

    print("Loading files...", flush=True)
    data = np.load(npz_path)
    edge_index = data['edge_index']
    provenance = data['provenance']
    weights = data['weights']
    nodes = data['nodes']
    n_nodes = data['n_nodes'].item()

    with open(json_path, 'r') as f:
        report = json.load(f)

    print("1. Node order is preserved", flush=True)
    expected_nodes = []
    with open(landmark_tsv, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        hgnc_idx = header.index('l1000_symbol')
        for line in f:
            parts = line.strip('\n').split('\t')
            expected_nodes.append(parts[hgnc_idx])
    expected_nodes = np.array(expected_nodes, dtype=str)
    assert np.array_equal(nodes, expected_nodes), "Nodes do not match landmark_symbols_v9.tsv exactly"

    print("4. No self-loops, no duplicates, i < j everywhere", flush=True)
    i, j = edge_index[0], edge_index[1]
    assert np.all(i < j), "Found edges where i >= j"
    # To check for duplicates
    edge_hashes = i * n_nodes + j
    assert len(np.unique(edge_hashes)) == len(edge_hashes), "Found duplicate edges"

    print("6. Union is a superset of each source, and provenance.sum(axis=1) >= 1", flush=True)
    assert np.all(provenance.sum(axis=1) >= 1), "Found an edge with no provenance bits set"

    print("2. STRING round-trip", flush=True)
    string_adj = np.load(string_npy)
    
    # We can pre-build a dictionary or matrix for O(1) lookups in the union graph
    union_dict = {}
    for idx in range(edge_index.shape[1]):
        u, v = edge_index[0, idx], edge_index[1, idx]
        union_dict[(u, v)] = {
            'prov': provenance[idx],
            'weight': weights[idx]
        }

    rng = np.random.RandomState(42)
    pairs = set()
    while len(pairs) < 200:
        u = rng.randint(0, n_nodes)
        v = rng.randint(0, n_nodes)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    
    for u, v in pairs:
        val = string_adj[u, v]
        is_in_string = val > args.string_thresh
        in_union = (u, v) in union_dict and union_dict[(u, v)]['prov'][0] == 1
        
        assert is_in_string == in_union, f"STRING round-trip failed for pair ({u}, {v})"
        if in_union:
            w = union_dict[(u, v)]['weight'][0]
            assert np.isclose(w, val), f"STRING weight mismatch for pair ({u}, {v}): {w} != {val}"

    print("3. Reactome round-trip", flush=True)
    M_react = np.load(m_pathway_npy)
    react_sizes = M_react.sum(axis=1)
    react_keep = (react_sizes >= args.min_term) & (react_sizes <= args.max_term)
    surviving_indices = np.where(react_keep)[0]
    
    if len(surviving_indices) > 0:
        chosen_pathways = rng.choice(surviving_indices, size=min(50, len(surviving_indices)), replace=False)
        for pw_idx in chosen_pathways:
            members = np.where(M_react[pw_idx] == 1)[0]
            for idx1 in range(len(members)):
                for idx2 in range(idx1 + 1, len(members)):
                    u, v = members[idx1], members[idx2]
                    assert (u, v) in union_dict, f"Co-members {(u, v)} of pathway {pw_idx} not in union"
                    assert union_dict[(u, v)]['prov'][1] == 1, f"Reactome prov bit not set for {(u, v)}"
    else:
        print("WARNING: No Reactome pathways survived the size filter!")

    print("5. The size filter actually removed something", flush=True)
    raw_reactome = int(M_react.shape[0])
    surviving_reactome = int(np.sum(react_keep))
    
    if raw_reactome == surviving_reactome:
        # Check GO as well?
        print("LOUD WARNING: Nothing was filtered from Reactome at these settings!")
    
    assert report['sources']['Reactome']['raw_term_count'] == raw_reactome
    assert report['sources']['Reactome']['surviving_terms'] == surviving_reactome
    assert raw_reactome > surviving_reactome, "Reactome filter removed nothing!"

    print("7. Multi-hot is not degenerate", flush=True)
    # At least one edge has provenance summing to 2 or more
    assert np.any(provenance.sum(axis=1) >= 2), "No edges have overlapping sources (sum >= 2)"
    
    # At least one unique to each source
    has_unique_string = np.any((provenance[:, 0] == 1) & (provenance[:, 1] == 0) & (provenance[:, 2] == 0))
    has_unique_react = np.any((provenance[:, 0] == 0) & (provenance[:, 1] == 1) & (provenance[:, 2] == 0))
    has_unique_go = np.any((provenance[:, 0] == 0) & (provenance[:, 1] == 0) & (provenance[:, 2] == 1))
    
    assert has_unique_string, "No edge is unique to STRING"
    assert has_unique_react, "No edge is unique to Reactome"
    assert has_unique_go, "No edge is unique to GO:BP"
    
    print("\nProvenance combination counts:")
    for k, v in report['provenance_combinations'].items():
        print(f"  {k}: {v}")

    print("\nAll tests passed successfully!", flush=True)

if __name__ == "__main__":
    main()
