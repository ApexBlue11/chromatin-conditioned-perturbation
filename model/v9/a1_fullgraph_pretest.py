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
    # Verifier's runtime checks on the real graph (the test file exercises a re-implementation, not this code):
    # exact symmetry, and every undirected pair present exactly once per direction.
    assert (W - W.T).count_nonzero() == 0, 'W is not symmetric'
    assert W.nnz == 2 * len(df), 'W has %d entries for %d undirected pairs' % (W.nnz, len(df))
    print('W: %d undirected pairs, symmetric, no doubling' % len(df))
    
    # Canonical landmark order
    landmark_path = r'C:\Projects\LINCS\Data Info\pathway_landmark_genes.txt'
    with open(landmark_path, 'r') as f:
        landmarks = [line.strip() for line in f if line.strip()]
    # The PI's brief asserted nodes[landmark_idx] == the canonical L1000 symbols; 34 of those are older aliases of the
    # graph's HGNC names (EPRS -> EPRS1, TOMM70A -> TOMM70). landmark_symbols_v9.tsv carries both: its l1000_symbol column
    # IS the canonical order and its hgnc_symbol column is what the graph uses. Assert the chain, not a false identity.
    lm = pd.read_csv(r'C:\Projects\LINCS\network\outputs\v9\landmark_symbols_v9.tsv', sep='\t').sort_values('canonical_row')
    assert list(lm['canonical_row']) == list(range(978)), "landmark table is not 0..977"
    assert list(lm['l1000_symbol']) == landmarks, "landmark table is not in the canonical order"
    assert list(nodes[landmark_idx]) == list(lm['hgnc_symbol']), "landmark_idx does not point at the HGNC names in canonical order"
    
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
                # Verifier's runtime check: the returned S solves (I - alpha P) S = T, on the real operator.
                resid = np.abs(S_new - args.alpha * (P @ S_new) - T_block).max()
                assert resid < 1e-8, 'fixed-point residual %.3e' % resid
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
    n_nan_draws = [0]
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
        
        # A random draw on isolated nodes leaves s constant at the landmarks, so its Spearman is undefined (NaN) and
        # carries no information: average over the draws that are defined (first full run, 2026-09-24).
        draws = np.stack([corr(s_r, y_comp) for s_r in s_rand], axis=0)
        n_nan_draws[0] += int(np.isnan(draws).sum())
        with np.errstate(invalid='ignore'):
            r_rand = np.nanmean(draws, axis=0)
        
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
            dp = (group['score_NONE'] - group['score_RANDOM']).dropna()
            ds = (group['score_NONE'] - group['score_DEGREE']).dropna()
            d_c_primary[cell] = np.median(dp) if len(dp) else np.nan
            d_c_secondary[cell] = np.median(ds) if len(ds) else np.nan
            
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
        # the secondary contrast, NONE - DEGREE, on the SAME resample matrix (brief section 1; omitted by the worker)
        d_c2 = np.array([d_c_secondary[c] for c in cells_used])
        boot2 = np.mean(d_c2[boot_indices], axis=1)
        secondary = {'mean_d': float(np.mean(d_c2)), 'ci95': [float(x) for x in np.percentile(boot2, [2.5, 97.5])],
                     'n_cells_positive': int(np.sum(d_c2 > 0)),
                     'pval_sign_test': float(scipy.stats.binomtest(int(np.sum(d_c2 > 0)), len(d_c2), p=0.5).pvalue)}
    else:
        mean_d = np.nan
        n_cells_pos = 0
        ci95 = [np.nan, np.nan]
        pval_sign = np.nan
        secondary = None
        
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
        'secondary_NONE_minus_DEGREE': secondary,
        'nan_rows_NONE': int(np.isnan(score_NONE).sum()),
        'nan_rows_RANDOM_all_draws': int(np.isnan(score_RANDOM).sum()),
        'nan_random_draw_scores': int(n_nan_draws[0]),
        'per_cell_d_primary': {c: float(v) for c, v in d_c_primary.items()},
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
