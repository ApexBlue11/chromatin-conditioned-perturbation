# Task W7 Report: Unioned Gene-Gene Graph

## Key Sections of `build_union_graph.py`

**STRING Parsing and Thresholding**
```python
    string_adj = np.load(string_npy)
    string_mask = (string_adj > args.string_thresh) & np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    string_i, string_j = np.where(string_mask)
    string_w = string_adj[string_i, string_j]
```

**Reactome Filtering and Co-occurrence**
```python
    M_react = np.load(m_pathway_npy)
    react_sizes = M_react.sum(axis=1)
    react_keep = (react_sizes >= args.min_term) & (react_sizes <= args.max_term)
    M_react_filt = M_react[react_keep]
    react_co = M_react_filt.T @ M_react_filt
    react_mask = (react_co > 0) & np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    react_i, react_j = np.where(react_mask)
    react_w = react_co[react_i, react_j].astype(np.float32)
```

**GO:BP GMT Parsing and Filtering**
```python
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
```

**Union Provenance Assembly**
```python
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
```

## Key Sections of `test_union_graph.py`

**STRING Round-Trip Check**
```python
    for u, v in pairs:
        val = string_adj[u, v]
        is_in_string = val > args.string_thresh
        in_union = (u, v) in union_dict and union_dict[(u, v)]['prov'][0] == 1
        
        assert is_in_string == in_union, f"STRING round-trip failed for pair ({u}, {v})"
        if in_union:
            w = union_dict[(u, v)]['weight'][0]
            assert np.isclose(w, val), f"STRING weight mismatch for pair ({u}, {v}): {w} != {val}"
```

**Multi-hot / Super-set Check**
```python
    assert np.all(provenance.sum(axis=1) >= 1), "Found an edge with no provenance bits set"
    assert np.any(provenance.sum(axis=1) >= 2), "No edges have overlapping sources (sum >= 2)"
    
    has_unique_string = np.any((provenance[:, 0] == 1) & (provenance[:, 1] == 0) & (provenance[:, 2] == 0))
    has_unique_react = np.any((provenance[:, 0] == 0) & (provenance[:, 1] == 1) & (provenance[:, 2] == 0))
    has_unique_go = np.any((provenance[:, 0] == 0) & (provenance[:, 1] == 0) & (provenance[:, 2] == 1))
    
    assert has_unique_string, "No edge is unique to STRING"
```

## Build Script Output
```
Loaded 978 landmarks.
Saved network/outputs/v9\union_graph_v9.npz
{
  "parameters": {
    "string_thresh": 0.0,
    "min_term": 5,
    "max_term": 200
  },
  "file_sha1s": {
    "STRING_adj_978_v9.npy": "609e289227b4e96da09dcc347667deb5f3645a8a",
    "M_pathway_v9.npy": "c752c9c25a066af63cbe9595a6565e026441765c",
    "pathway_info_v9.tsv": "68c4fa6289bc5ab36757b8b355ca62a9ec995b0c",
    "landmark_symbols_v9.tsv": "e7405ff44f4838d4c43378411aa851514daf71b4",
    "GO_Biological_Process_2023.gmt": "93f49956f22f95bada11b1b80c4390ca716ecc39"
  },
  "landmark_order_sha1": "e7405ff44f4838d4c43378411aa851514daf71b4",
  "sources": {
    "STRING": {
      "raw_edge_count": 13001,
      "edges_contributed": 13001,
      "density": 0.02721280661764552
    },
    "Reactome": {
      "raw_term_count": 800,
      "surviving_terms": 784,
      "edges_contributed": 59860,
      "density": 0.1252948699432552
    },
    "GO_BP": {
      "raw_term_count": 5407,
      "surviving_terms": 1087,
      "edges_contributed": 61295,
      "density": 0.12829851408573048
    }
  },
  "union_edges": 81846,
  "provenance_combinations": {
    "STRING": 3465,
    "Reactome": 14922,
    "STRING_Reactome": 2164,
    "GO": 17902,
    "STRING_GO": 619,
    "Reactome_GO": 36021,
    "STRING_Reactome_GO": 6753
  },
  "jaccard_overlap": {
    "STRING_Reactome": 0.1394501438758914,
    "STRING_GO": 0.11015480246249477,
    "Reactome_GO": 0.5457189880200559
  }
}
```

## Test Script Output
```
Loading files...
1. Node order is preserved
4. No self-loops, no duplicates, i < j everywhere
6. Union is a superset of each source, and provenance.sum(axis=1) >= 1
2. STRING round-trip
3. Reactome round-trip
5. The size filter actually removed something
7. Multi-hot is not degenerate

Provenance combination counts:
  STRING: 3465
  Reactome: 14922
  STRING_Reactome: 2164
  GO: 17902
  STRING_GO: 619
  Reactome_GO: 36021
  STRING_Reactome_GO: 6753

All tests passed successfully!
```

## Provenance Table and Densities
- **STRING**: Density = 2.72%, 13001 edges contributed.
- **Reactome**: Density = 12.53%, 59860 edges contributed.
- **GO:BP**: Density = 12.83%, 61295 edges contributed.

Provenance Combination Counts:
- STRING only: 3,465
- Reactome only: 14,922
- STRING + Reactome: 2,164
- GO only: 17,902
- STRING + GO: 619
- Reactome + GO: 36,021
- STRING + Reactome + GO: 6,753

## What I was unsure about
- **Symbol-to-Index Mapping:** There are 978 landmarks in `landmark_symbols_v9.tsv`. I had to decide whether to match against the `l1000_symbol` or `hgnc_symbol` columns. I elected to use `hgnc_symbol` for mapping since it hits significantly more targets when crossed with standard GMT files (877 hits vs 852 hits). I mapped the `hgnc_symbol` to the row index 0-977 directly. The `nodes` array saved inside the final artefact `.npz` uses the `l1000_symbol` string representation as instructed by standard practice in the codebase (the authoritative representation is `l1000_symbol` as it's the model's preferred string).
- **GO GMT Parsing Ambiguity:** The GO GMT file has 14,698 unique genes across 5,407 terms. After mapping the GMT records against the 978 `hgnc_symbols`, 877 unique landmark genes were successfully mapped. The unmapped genes (about 13,821 unique genes in GO not present in landmarks) were simply discarded. 
- No `hgnc_symbol` mapped to more than one index; they are unique across the 978 landmarks. 
- For STRING, "terms surviving the size filter" does not exactly apply. I omitted the key from the report and used `raw_edge_count` and `edges_contributed` instead, which is logically equivalent and maintains the report's tabular integrity.
