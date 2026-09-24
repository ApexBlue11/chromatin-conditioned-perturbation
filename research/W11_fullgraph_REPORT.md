# W11 Full-Graph Target Propagation Report

## `model/v9/a1_fullgraph_pretest.py`
```python
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.stats
import json
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Full graph pretest")
    parser.add_argument('--n_boot', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_compounds', type=int, default=None)
    return parser.parse_args()

def run():
    args = parse_args()
    
    # Load graph
    graph_path = r'C:\Projects\LINCS\network\outputs\v9\string_graph_v9.npz'
    graph = np.load(graph_path)
    nodes = graph['nodes'].astype(str)
    edge_index = graph['edge_index']
    weight = graph['weight']
    landmark_idx = graph['landmark_idx']
    n_nodes = len(nodes)
    
    # Check edges
    u = edge_index[0]
    v = edge_index[1]
    df_dir = pd.DataFrame({'u': u, 'v': v})
    df_dir = df_dir[df_dir['u'] != df_dir['v']]
    swapped = pd.DataFrame({'u': df_dir['v'], 'v': df_dir['u']})
    merged = pd.merge(df_dir[df_dir['u'] > df_dir['v']], swapped, on=['u', 'v'])
    both_directions = len(merged)
    print(f"Check undirected edges: {both_directions} pairs appear in both directions (i > j and j > i).")
    
    # Build W
    df = pd.DataFrame({'u': u, 'v': v, 'w': weight})
    df['u_min'] = np.minimum(df['u'], df['v'])
    df['v_max'] = np.maximum(df['u'], df['v'])
    df = df[df['u_min'] != df['v_max']]
    df = df.groupby(['u_min', 'v_max'])['w'].max().reset_index()
    W_upper = sp.csr_matrix((df['w'], (df['u_min'], df['v_max'])), shape=(n_nodes, n_nodes))
    W = W_upper + W_upper.T
    
    # Canonical landmark order
    landmark_path = r'C:\Projects\LINCS\Data Info\pathway_landmark_genes.txt'
    with open(landmark_path, 'r') as f:
        landmarks = [line.strip() for line in f if line.strip()]
    assert np.array_equal(nodes[landmark_idx], landmarks), "Landmark order mismatch"
    
    # Targets
    dti_path = r'C:\Projects\LINCS\drug\outputs\dti\chembl_dti_edges.tsv'
    dti = pd.read_csv(dti_path, sep='\t', keep_default_na=False)
    dti = dti[(dti['direct_interaction'].astype(str) == '1') & (dti['gene_symbol'] != '')]
    
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    dti['node_idx'] = dti['gene_symbol'].map(node_to_idx)
    mapped = dti.dropna(subset=['node_idx']).copy()
    mapped['node_idx'] = mapped['node_idx'].astype(int)
    
    unmapped_symbols = dti.loc[dti['node_idx'].isna(), 'gene_symbol'].nunique()
    print(f"Unmapped target symbols: {unmapped_symbols}")
    
    comp_targets = mapped.groupby('pert_id')['node_idx'].unique().to_dict()
    print(f"Compounds with targets: {len(comp_targets)}")
    
    # Signatures
    sig_path = r'C:\Projects\LINCS\phase2_assembly\outputs\signatures_usable.tsv'
    sigs = pd.read_csv(sig_path, sep='\t')
    
    sigs_used = sigs[sigs['pert_id'].isin(comp_targets)].copy()
    
    eligible_comps = sorted(list(comp_targets.keys()))
    if args.max_compounds is not None:
        eligible_comps = eligible_comps[:args.max_compounds]
        sigs_used = sigs_used[sigs_used['pert_id'].isin(eligible_comps)]
        
    print(f"Rows with targets for eligible compounds: {len(sigs_used)}")
    
    # P = D^-1/2 W D^-1/2
    deg = np.array(W.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    P = D_inv_sqrt @ W @ D_inv_sqrt
    
    def propagate(T_block):
        S = T_block.copy()
        max_iters = 500
        for i in range(max_iters):
            S_new = args.alpha * (P @ S) + T_block
            diff = np.sum(np.abs(S_new - S), axis=0)
            if np.max(diff) < 1e-10:
                return S_new, i+1
            S = S_new
        raise RuntimeError(f"Not converged after {max_iters} iterations")
        
    n_eligible = len(eligible_comps)
    T_NONE = np.zeros((n_nodes, n_eligible), dtype=float)
    T_RANDOM = [np.zeros((n_nodes, n_eligible), dtype=float) for _ in range(5)]
    
    for i, comp in enumerate(eligible_comps):
        targets = comp_targets[comp]
        k = len(targets)
        T_NONE[targets, i] = 1.0 / k
        
        comp_rng = np.random.default_rng(args.seed + i)
        for j in range(5):
            rand_targets = comp_rng.choice(n_nodes, size=k, replace=False)
            T_RANDOM[j][rand_targets, i] = 1.0 / k
            
    print("Propagating NONE...")
    S_NONE, iter_none = propagate(T_NONE)
    S_RANDOM = []
    iters_random = []
    for j in range(5):
        print(f"Propagating RANDOM {j+1}...")
        S, it = propagate(T_RANDOM[j])
        S_RANDOM.append(S)
        iters_random.append(it)
        
    max_iters = max([iter_none] + iters_random)
    
    S_DEGREE = deg.copy()
    
    # Response
    Y_path = r'C:\Projects\LINCS\phase2_assembly\outputs\Y_target_level5_978.npy'
    Y = np.load(Y_path, mmap_mode='r')
    
    score_NONE = np.full(len(sigs_used), np.nan)
    score_RANDOM = np.full(len(sigs_used), np.nan)
    score_DEGREE = np.full(len(sigs_used), np.nan)
    
    used_rows = sigs_used['row'].values
    Y_used = np.abs(Y[used_rows])
    
    sigs_used['array_idx'] = np.arange(len(sigs_used))
    
    def corr(a, b):
        if np.all(a == a[0]):
            return np.full(b.shape[0], np.nan)
        a_rank = scipy.stats.rankdata(a)
        try:
            b_rank = scipy.stats.rankdata(b, axis=1)
        except TypeError:
            b_rank = np.apply_along_axis(scipy.stats.rankdata, 1, b)
        a_ctr = a_rank - a_rank.mean()
        b_ctr = b_rank - b_rank.mean(axis=1, keepdims=True)
        num = (b_ctr * a_ctr).sum(axis=1)
        den = np.sqrt((b_ctr**2).sum(axis=1) * (a_ctr**2).sum())
        with np.errstate(divide='ignore', invalid='ignore'):
            res = num / den
        return res

    print("Computing scores...")
    for i, comp in enumerate(eligible_comps):
        comp_mask = (sigs_used['pert_id'] == comp).values
        if not comp_mask.any(): continue
        
        targets = comp_targets[comp]
        target_landmarks = [idx for idx, l_node in enumerate(landmark_idx) if l_node in targets]
        
        mask_landmarks = np.ones(978, dtype=bool)
        mask_landmarks[target_landmarks] = False
        
        y_comp = Y_used[comp_mask][:, mask_landmarks]
        
        s_none = S_NONE[landmark_idx, i][mask_landmarks]
        s_rand = [S_R[landmark_idx, i][mask_landmarks] for S_R in S_RANDOM]
        s_deg = S_DEGREE[landmark_idx][mask_landmarks]
        
        r_none = corr(s_none, y_comp)
        
        r_rands = np.zeros(y_comp.shape[0])
        for s_r in s_rand:
            r_rands += corr(s_r, y_comp)
        r_rand = r_rands / 5.0
        
        r_deg = corr(s_deg, y_comp)
        
        arr_indices = sigs_used['array_idx'].values[comp_mask]
        score_NONE[arr_indices] = r_none
        score_RANDOM[arr_indices] = r_rand
        score_DEGREE[arr_indices] = r_deg
        
    sigs_used['score_NONE'] = score_NONE
    sigs_used['score_RANDOM'] = score_RANDOM
    sigs_used['score_DEGREE'] = score_DEGREE
    
    # Estimand
    d_c_primary = {}
    d_c_secondary = {}
    
    for cell, group in sigs_used.groupby('cell_id'):
        valid = ~group['score_NONE'].isna()
        if valid.sum() >= 50:
            d_c_primary[cell] = np.median(group.loc[valid, 'score_NONE'] - group.loc[valid, 'score_RANDOM'])
            d_c_secondary[cell] = np.median(group.loc[valid, 'score_NONE'] - group.loc[valid, 'score_DEGREE'])
            
    cells_used = list(d_c_primary.keys())
    
    if len(cells_used) > 0:
        d_c = np.array([d_c_primary[c] for c in cells_used])
        mean_d = np.mean(d_c)
        n_cells_pos = np.sum(d_c > 0)
        
        boot_rng = np.random.default_rng(args.seed)
        boot_indices = boot_rng.integers(0, len(d_c), size=(args.n_boot, len(d_c)))
        boot_means = np.mean(d_c[boot_indices], axis=1)
        ci95 = np.percentile(boot_means, [2.5, 97.5])
        
        pval_sign = scipy.stats.binomtest(n_cells_pos, len(d_c), p=0.5).pvalue
    else:
        mean_d = np.nan
        n_cells_pos = 0
        ci95 = [np.nan, np.nan]
        pval_sign = np.nan
        
    med_none = np.nanmedian(sigs_used['score_NONE'])
    med_random = np.nanmedian(sigs_used['score_RANDOM'])
    med_degree = np.nanmedian(sigs_used['score_DEGREE'])
    
    n_cells = len(cells_used)
    if n_cells > 0 and med_none > 0.01 and mean_d > 0 and ci95[0] > 0 and n_cells_pos >= 0.75 * n_cells:
        reading = "SIGNAL"
    elif n_cells > 0 and mean_d > 0 and ci95[0] > 0 and n_cells_pos >= 0.75 * n_cells and med_none <= 0.01:
        reading = "SPECIFIC_BUT_NEGLIGIBLE"
    else:
        reading = "NO_SIGNAL"
        
    out_dir = r'C:\Projects\LINCS\model\results'
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_cmp{args.max_compounds}" if args.max_compounds is not None else ""
    out_json = os.path.join(out_dir, f'a1_fullgraph_pretest{suffix}.json')
    out_npz = os.path.join(out_dir, f'a1_fullgraph_pretest{suffix}_rows.npz')
    
    res = {
        'unmapped_symbols': int(unmapped_symbols),
        'compounds_with_targets': len(comp_targets),
        'rows': len(sigs_used),
        'cells_used': n_cells,
        'iterations_to_converge': int(max_iters),
        'edge_check': int(both_directions),
        'reading_83_4': reading,
        'mean_d': float(mean_d),
        'ci95': [float(x) for x in ci95] if len(ci95) == 2 else [],
        'n_cells_positive': int(n_cells_pos),
        'pval_sign_test': float(pval_sign) if not np.isnan(pval_sign) else None,
        'median_score_NONE': float(med_none) if not np.isnan(med_none) else None,
        'median_score_RANDOM': float(med_random) if not np.isnan(med_random) else None,
        'median_score_DEGREE': float(med_degree) if not np.isnan(med_degree) else None,
    }
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
        
    np.savez(out_npz,
             row=sigs_used['row'].values,
             cell_id=sigs_used['cell_id'].values,
             pert_id=sigs_used['pert_id'].values,
             score_NONE=sigs_used['score_NONE'].values,
             score_RANDOM=sigs_used['score_RANDOM'].values,
             score_DEGREE=sigs_used['score_DEGREE'].values)

if __name__ == '__main__':
    run()
```

## `model/v9/test_a1_fullgraph.py`
```python
import numpy as np
import scipy.sparse as sp

def build_W(edge_index, weight, n_nodes):
    u = np.minimum(edge_index[0], edge_index[1])
    v = np.maximum(edge_index[0], edge_index[1])
    mask = u != v
    u = u[mask]
    v = v[mask]
    w = weight[mask]
    
    edges = np.stack([u, v], axis=1)
    unique_edges, indices = np.unique(edges, axis=0, return_index=True)
    w_unique = w[indices]
    
    W_upper = sp.csr_matrix((w_unique, (unique_edges[:, 0], unique_edges[:, 1])), shape=(n_nodes, n_nodes))
    W = W_upper + W_upper.T
    return W

def get_P(W):
    deg = np.array(W.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    return D_inv_sqrt @ W @ D_inv_sqrt

def propagate(P, T_block, alpha, max_iters=500):
    S = T_block.copy()
    for i in range(max_iters):
        S_new = alpha * (P @ S) + T_block
        diff = np.sum(np.abs(S_new - S), axis=0)
        if np.max(diff) < 1e-10:
            return S_new, i+1
        S = S_new
    raise RuntimeError(f"Not converged after {max_iters} iterations")

def test_T1_T2_T7():
    n = 5
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    weights = np.array([1.0, 2.0, 1.5, 0.5])
    edge_index = edges.T
    W = build_W(edge_index, weights, n)
    P = get_P(W)
    
    T = np.random.rand(n, 2)
    
    # T1: alpha = 0 gives S == T
    S_alpha0, _ = propagate(P, T, alpha=0.0)
    assert np.allclose(S_alpha0, T), "T1 failed"
    
    # T2: iteration matches dense solve inv(I - alpha P) @ T to 1e-8
    alpha = 0.5
    S_iter, _ = propagate(P, T, alpha=alpha)
    
    P_dense = P.toarray()
    I = np.eye(n)
    S_dense = np.linalg.inv(I - alpha * P_dense) @ T
    assert np.allclose(S_iter, S_dense, atol=1e-8), "T2 failed"
    
    # T7: non-convergence raises
    try:
        propagate(P, T, alpha=1.0, max_iters=10)
        assert False, "T7 failed: Should have raised RuntimeError"
    except RuntimeError:
        pass

def test_T3():
    n = 3
    edges1 = np.array([[0, 1], [1, 2]])
    w1 = np.array([1.0, 2.0])
    W1 = build_W(edges1.T, w1, n)
    
    edges2 = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
    w2 = np.array([1.0, 1.0, 2.0, 2.0])
    W2 = build_W(edges2.T, w2, n)
    
    assert np.allclose(W1.toarray(), W2.toarray()), "T3 failed"

def test_T4():
    n_nodes = 100
    seed = 42
    compound_index = 5
    k = 10
    
    rng1 = np.random.default_rng(seed + compound_index)
    draws1 = [rng1.choice(n_nodes, size=k, replace=False) for _ in range(5)]
    
    rng2 = np.random.default_rng(seed + compound_index)
    draws2 = [rng2.choice(n_nodes, size=k, replace=False) for _ in range(5)]
    
    for i in range(5):
        assert len(draws1[i]) == k, "T4 failed: target count"
        assert len(set(draws1[i])) == k, "T4 failed: never repeat within a draw"
        assert np.array_equal(draws1[i], draws2[i]), "T4 failed: identical across calls"

def test_T5():
    from scipy.stats import rankdata
    def corr(a, b):
        if np.all(a == a[0]):
            return np.full(b.shape[0], np.nan)
        a_rank = rankdata(a)
        try:
            b_rank = rankdata(b, axis=1)
        except TypeError:
            b_rank = np.apply_along_axis(rankdata, 1, b)
        a_ctr = a_rank - a_rank.mean()
        b_ctr = b_rank - b_rank.mean(axis=1, keepdims=True)
        num = (b_ctr * a_ctr).sum(axis=1)
        den = np.sqrt((b_ctr**2).sum(axis=1) * (a_ctr**2).sum())
        with np.errstate(divide='ignore', invalid='ignore'):
            res = num / den
        return res

    landmark_idx = np.array([10, 20, 30, 40])
    targets = [10, 50]
    target_landmarks = [i for i, l in enumerate(landmark_idx) if l in targets]
    
    mask_landmarks = np.ones(len(landmark_idx), dtype=bool)
    mask_landmarks[target_landmarks] = False
    
    s = np.array([0.5, 0.3, 0.8, 0.1])
    y_comp = np.array([[1.0, 2.0, 3.0, 4.0]])
    
    s_masked = s[mask_landmarks]
    y_masked = y_comp[:, mask_landmarks]
    
    score1 = corr(s_masked, y_masked)[0]
    
    y_comp_changed = y_comp.copy()
    y_comp_changed[0, 0] = 999.0
    y_masked_changed = y_comp_changed[:, mask_landmarks]
    
    score2 = corr(s_masked, y_masked_changed)[0]
    
    assert score1 == score2, "T5 failed"

def test_T6():
    def get_reading(med_none, d_c, mean_d, ci95, n_cells_pos):
        n_cells = len(d_c)
        if med_none > 0.01 and mean_d > 0 and ci95[0] > 0 and n_cells_pos >= 0.75 * n_cells:
            return "SIGNAL"
        elif mean_d > 0 and ci95[0] > 0 and n_cells_pos >= 0.75 * n_cells and med_none <= 0.01:
            return "SPECIFIC_BUT_NEGLIGIBLE"
        else:
            return "NO_SIGNAL"
            
    d_c = np.array([0.1, 0.2, 0.15, 0.3])
    mean_d = np.mean(d_c)
    ci95 = [0.05, 0.3]
    n_cells_pos = 4
    assert get_reading(0.05, d_c, mean_d, ci95, n_cells_pos) == "SIGNAL", "T6 failed: SIGNAL"
    assert get_reading(0.005, d_c, mean_d, ci95, n_cells_pos) == "SPECIFIC_BUT_NEGLIGIBLE", "T6 failed: SPECIFIC_BUT_NEGLIGIBLE"
    
    d_c_bad = np.array([-0.1, -0.2, 0.1, -0.3])
    mean_d_bad = np.mean(d_c_bad)
    ci95_bad = [-0.3, 0.1]
    n_cells_pos_bad = 1
    assert get_reading(0.05, d_c_bad, mean_d_bad, ci95_bad, n_cells_pos_bad) == "NO_SIGNAL", "T6 failed: NO_SIGNAL"

if __name__ == '__main__':
    test_T1_T2_T7()
    test_T3()
    test_T4()
    test_T5()
    test_T6()
    print("All tests passed.")
```

## Command Outputs

**`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_a1_fullgraph.py`**
```
All tests passed.
```

**`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_fullgraph_pretest.py --help`**
```
usage: a1_fullgraph_pretest.py [-h] [--n_boot N_BOOT] [--seed SEED]
                               [--alpha ALPHA] [--max_compounds MAX_COMPOUNDS]

Full graph pretest

options:
  -h, --help            show this help message and exit
  --n_boot N_BOOT
  --seed SEED
  --alpha ALPHA
  --max_compounds MAX_COMPOUNDS
```

**`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_fullgraph_pretest.py --max_compounds 20 --n_boot 200`**
```
Check undirected edges: 0 pairs appear in both directions (i > j and j > i).
Traceback (most recent call last):
  File "C:\Projects\LINCS\model\v9\a1_fullgraph_pretest.py", line 268, in <module>
    run()
    ~~~^^
  File "C:\Projects\LINCS\model\v9\a1_fullgraph_pretest.py", line 52, in run
    assert np.array_equal(nodes[landmark_idx], landmarks), "Landmark order mismatch"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Landmark order mismatch
```

## What I was unsure about
- The assertion `np.array_equal(nodes[landmark_idx], landmarks)` failed during the smoke test. Upon inspection, there are 34 mismatching symbols between the string graph `nodes[landmark_idx]` array and the text file `pathway_landmark_genes.txt` (e.g., `EPRS1` vs `EPRS`, `TOMM70` vs `TOMM70A`). Since the brief strongly explicitly specified "**Assert** `nodes[landmark_idx]` equals it" and instructed not to redesign it, I kept the strict equality assertion as-is, causing the smoke test to exit early.
