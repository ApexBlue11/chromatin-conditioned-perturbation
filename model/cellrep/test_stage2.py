"""Tests for stage2.py (RESULTS 103.6 / 103.7) on a small fake dataset built in memory."""
import os, sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stage2 as s2  # noqa: E402


def fake(N=1100, n_genes=40, seed=0, path=False):
    """978 landmark nodes (0..977), the last 6 isolated; genes are non-landmark nodes plus two landmark genes."""
    rng = np.random.default_rng(seed)
    D = object.__new__(s2.Data)
    D.nodes = [f'n{i}' for i in range(N)]
    D.node_of = {s: i for i, s in enumerate(D.nodes)}
    D.lm_node = np.arange(978)
    if path:
        E = np.array([[i, i + 1] for i in range(N - 1)])
    else:
        E = np.array(sorted({(min(a, b), max(a, b)) for a in range(N) for b in rng.integers(0, N, 3) if a != b}))
    E = E[~np.isin(E, np.arange(972, 978)).any(1)]                      # landmarks 972..977 isolated
    D.edges = {'real': E, 'rewired': E.copy()}
    deg = np.bincount(E.ravel(), minlength=N)
    D.lm_reach = deg[D.lm_node] > 0
    D.lm_sym = D.nodes[:978]
    D.lm_pos = {s: i for i, s in enumerate(D.lm_sym)}
    D.gvec = rng.normal(size=(N, 128)).astype(np.float32)
    genes = [f'n{i}' for i in range(978, 978 + n_genes)] + ['n5', 'n10']
    D.cell_index = {c: k for k, c in enumerate(s2.TRAIN_CELLS)}
    rows, D.idx = [], {}
    for g in genes:
        for c in s2.TRAIN_CELLS + ('HA1E',):
            D.idx[(g, c)] = len(rows)
            rows.append(rng.normal(size=978))
    D.Y = np.array(rows, np.float32)
    D.genes = sorted(genes)
    nc = len(s2.TRAIN_CELLS)
    D.cellfeat = {'CE': rng.normal(size=(nc, 978 + s2.N_PC)).astype(np.float32), 'CG': rng.normal(size=(nc, 43)).astype(np.float32),
                  'CL': rng.normal(size=(nc, 978, 16)).astype(np.float32), 'CLM': np.ones((nc, 978, 4), np.float32),
                  'CNE': rng.normal(size=(nc, N, 2)).astype(np.float32), 'CNC': rng.normal(size=(nc, N, 20)).astype(np.float32)}
    D.act_lm = {c: rng.normal(size=978) for c in s2.TRAIN_CELLS}
    D.expr_lm = {c: rng.normal(size=978) for c in s2.TRAIN_CELLS}
    return D


def test_inner_split_keeps_genes_seen_in_loco_and_removes_them_in_cold():
    rows = [(f'g{i % 20}', f'c{i // 20}', None) for i in range(200)]
    m_rows = s2.inner_split(rows, 'rows')
    m_genes = s2.inner_split(rows, 'genes')
    assert m_rows.sum() == 20
    vg = {g for (g, _, _), x in zip(rows, m_genes) if x}
    assert all(x == (g in vg) for (g, _, _), x in zip(rows, m_genes)) and len(vg) == 2


def test_phi_is_uncorrelated_with_expression_within_cell():
    D = fake()
    for c in s2.TRAIN_CELLS:
        D.act_lm[c] = D.expr_lm[c] * 2.0 + np.random.default_rng(len(c) * 7 + ord(c[0])).normal(size=978)  # tracks expression
    phi = s2.phi_matrix(D, residualise=True)
    E = np.stack([D.expr_lm[c] for c in s2.TRAIN_CELLS])
    E = E - E.mean(0)
    assert all(abs(np.corrcoef(phi[c], E[k])[0, 1]) < 0.02 for k, c in enumerate(s2.TRAIN_CELLS))
    raw = s2.phi_matrix(D, residualise=False)
    assert np.corrcoef(raw['PC3'], E[0])[0, 1] > 0.5


@pytest.mark.parametrize('shape', ['dense', 'graph'])
def test_injection_hits_its_variance_share(shape):
    D = fake()
    tr, te, _ = D.loco('PC3')
    A = s2.norm_adj_scipy(D.edges['real'], len(D.nodes))
    tr2, te2, _ = s2.inject(D, tr, te, 0.05, shape, True, A)
    T = np.stack([t for *_, t in tr])
    U = np.stack([t for *_, t in tr2]) - T
    assert U.var() / T.var() == pytest.approx(0.05, rel=1e-6)


def test_graph_shape_is_zero_beyond_three_hops():
    D = fake(N=1100, path=True)
    A = s2.norm_adj_scipy(D.edges['real'], len(D.nodes))
    v = s2.graph_shape(D, A, 'n500')
    hops = np.abs(np.arange(978) - 500)
    assert (v[hops > 3] == 0).all() and (v[hops <= 3][[0, 3, 6]] != 0).any()


def test_graphprop_moves_only_landmarks_within_K_hops_and_pools_the_isolated():
    D = fake(N=1100, path=True)
    torch.manual_seed(0)
    m = s2.build_model('M3', D, K=3)
    m.eval()
    rows = [('n500', 'PC3', np.zeros(978))]
    b = s2.batch(D, rows, 'cpu')
    with torch.no_grad():
        base = m(b).numpy()[0]
        m.start.mul_(3.0)
        moved = m(b).numpy()[0]
    hops = np.abs(np.arange(978) - 500)
    changed = ~np.isclose(base, moved)
    reach = D.lm_reach
    assert not changed[(hops > 3) & reach].any() and changed[(hops <= 2) & reach].all()
    assert changed[~reach].all()                                         # isolated landmarks read the pooled state


def test_m2_chromatin_term_starts_at_zero():
    D = fake()
    m = s2.build_model('M2', D)
    m.eval()
    b = s2.batch(D, [('n980', 'PC3', np.zeros(978))], 'cpu')
    with torch.no_grad():
        a = m(b)
        D2 = fake()
        b2 = s2.batch(D, [('n980', 'PC3', np.zeros(978))], 'cpu', swap_cell='A549')
        a2 = m(b2)
    h = m.trunk(torch.cat([b['gvec'], m.cf.CE[b['cell']], m.cf.CG[b['chrom_cell']]], 1))
    assert torch.allclose(a, m.a(h), atol=1e-6)


def test_usage_swap_changes_only_the_chromatin_cell():
    D = fake()
    b = s2.batch(D, [('n980', 'PC3', np.zeros(978)), ('n981', 'PC3', np.zeros(978))], 'cpu', swap_cell='A549')
    assert (b['cell'] == D.cell_index['PC3']).all() and (b['chrom_cell'] == D.cell_index['A549']).all()


def test_row_scores_use_reachable_landmarks_and_skip_the_own_coordinate():
    D = fake()
    t = np.random.default_rng(3).normal(size=978)
    p = t.copy()
    p[975] = 100.0                                                       # an isolated landmark, ignored by r_reachable
    r_all, r_reach = s2.row_scores(D, [('n980', 'PC3', t)], [p])
    assert r_reach[0] == pytest.approx(1.0) and r_all[0] < 0.99


@pytest.mark.parametrize('name', ['M1', 'M2', 'M3', 'M4', 'M5'])
def test_every_model_trains_one_epoch(name, monkeypatch):
    monkeypatch.setattr(s2, 'MAX_EPOCHS', 1)
    D = fake(N=1100, n_genes=12)
    tr, te, _ = D.loco('PC3')
    m = s2.build_model(name, D)
    hist = s2.train(m, D, tr[:60], tr[60:80], 'cpu', 16, 0)
    P = s2.predict(m, D, te, 'cpu', 16)
    assert len(hist) == 1 and P.shape == (len(te), 978) and np.isfinite(P).all()
