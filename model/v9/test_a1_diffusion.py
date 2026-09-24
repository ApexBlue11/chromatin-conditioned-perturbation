import numpy as np
from a1_diffusion_pretest import (
    get_K, compute_score, get_mismatch_cells, 
    generate_resamples, compute_contrast_stats, get_reading,
    check_gene_order
)
import scipy.stats

def test_t1():
    W = np.array([[0, 1], [1, 0]], dtype=float)
    K = get_K(W, 0.0)
    assert np.allclose(K, np.eye(2))
    t = np.array([0.5, 0.5])
    s = K @ t
    assert np.allclose(s, t)

def test_t2():
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    a = np.array([1.0, 1.0, 1.0])
    W_own = A * np.sqrt(np.outer(a, a))
    K_own = get_K(W_own, 0.5)
    K_none = get_K(A, 0.5)
    assert np.allclose(K_own, K_none, atol=1e-12)

def test_t3():
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    alpha = 0.5
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    P = A * np.outer(d_inv_sqrt, d_inv_sqrt)
    K = get_K(A, alpha)
    
    assert np.allclose(K, K.T, atol=1e-10)
    I = np.eye(3)
    assert np.allclose((I - alpha * P) @ K, I, atol=1e-10)

def test_t4():
    A = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    K = get_K(A, 0.5)
    assert not np.any(np.isnan(K))
    assert not np.any(np.isinf(K))

def test_t5():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    z1 = np.array([1.0, -2.0, 3.0, -4.0])
    z2 = np.array([5.0, -2.0, 3.0, -4.0])
    targets = [0]
    
    score1 = compute_score(s, z1, targets)
    score2 = compute_score(s, z2, targets)
    assert score1 == score2
    
    z3 = np.array([1.0, -5.0, 3.0, -4.0])
    score3 = compute_score(s, z3, targets)
    assert score1 != score3

def test_t6():
    eligible_list = 10
    idx = 2
    seed = 42
    draw1 = get_mismatch_cells(idx, eligible_list, seed)
    assert idx not in draw1
    assert len(draw1) == len(set(draw1))
    
    draw2 = get_mismatch_cells(idx, eligible_list, seed)
    assert draw1 == draw2

def test_t7():
    nodes = np.array(["A", "B", "C"])
    genes = ["A", "B", "C"]
    check_gene_order(nodes, genes)
    
    genes_permuted = ["B", "A", "C"]
    raised = False
    try:
        check_gene_order(nodes, genes_permuted)
    except AssertionError:
        raised = True
    assert raised

def test_t8():
    d_c = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    resamples = generate_resamples(5, 200, 42)
    stats = compute_contrast_stats(d_c, resamples)
    assert np.isclose(stats["mean_d"], 0.3)
    assert np.allclose(stats["ci95"], [0.3, 0.3])

def test_t9():
    assert get_reading(0.01, [0.1, 0.2], 0.5, [0.1, 0.9], 10, 10, [0.1, 0.9], [0.1, 0.9]) == "UNINFORMATIVE"
    assert get_reading(0.05, [-0.1, 0.2], 0.5, [0.1, 0.9], 10, 10, [0.1, 0.9], [0.1, 0.9]) == "UNINFORMATIVE"
    assert get_reading(0.05, [0.1, 0.2], 0.5, [0.1, 0.9], 8, 10, [0.1, 0.9], [0.1, 0.9]) == "SUPPORTS"
    assert get_reading(0.05, [0.1, 0.2], 0.5, [-0.1, 0.9], 8, 10, [0.1, 0.9], [-0.1, 0.2]) == "GENE_LEVEL_PRIOR"
    assert get_reading(0.05, [0.1, 0.2], 0.5, [-0.1, 0.9], 8, 10, [-0.1, 0.9], [0.1, 0.9]) == "NO_SUPPORT"

if __name__ == "__main__":
    test_t1()
    test_t2()
    test_t3()
    test_t4()
    test_t5()
    test_t6()
    test_t7()
    test_t8()
    test_t9()
    print("All tests passed.")
