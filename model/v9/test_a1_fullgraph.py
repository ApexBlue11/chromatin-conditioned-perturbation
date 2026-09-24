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
