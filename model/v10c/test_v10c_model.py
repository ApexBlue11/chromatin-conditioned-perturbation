"""Tests for the v10-C model, data and trainer (RESULTS 100.4). CPU only; tiny widths."""
import json, os, sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v10c as mv  # noqa: E402
import data_v10c as dv  # noqa: E402
import train_v10c as tv  # noqa: E402

G = 978


def tiny_cfg(**kw):
    base = dict(d_model=32, n_heads=4, l_control=1, l_base=1, l_perturb=2, d_drug_hidden=32, d_chrom_hidden=16,
                d_gate_hidden=8, n_bins=16, dropout=0.0, stoch_depth=0.0)
    base.update(kw)
    return mv.V10CConfig(**base)


def priors(seed=0):
    r = np.random.default_rng(seed)
    M = (r.random((6, G)) < 0.05).astype(np.float32)
    A = (r.random((G, G)) < 0.01).astype(np.float32)
    A = np.maximum(A, A.T)
    gv = r.normal(size=(G, 8)).astype(np.float32)
    return M, A, gv


def make_batch(B=6, n_atoms=5, seed=0, same_cell=True):
    r = np.random.default_rng(seed)
    t = lambda x: torch.as_tensor(np.asarray(x, np.float32))
    xc = r.normal(size=(1 if same_cell else B, G))
    xc = np.repeat(xc, B, 0) if same_cell else xc
    F = r.random((1, G, 16)) if same_cell else r.random((B, G, 16))
    F = np.repeat(F, B, 0) if same_cell else F
    Mk = np.ones((B, G, 4), np.float32)
    ctx = np.zeros((B, 21), np.float32)
    ctx[:, 3] = 1
    return {'x_ctl': t(xc), 'x_cell': t(xc + 0.1), 'F': t(F), 'M': t(Mk), 'cell_ctx': t(ctx),
            'atoms': t(r.normal(size=(B, n_atoms, 512))), 'atom_mask': torch.ones(B, n_atoms, dtype=torch.bool),
            'u_feats': t(r.normal(size=(B, 2580))), 'dose': t(r.normal(size=B)), 'time': t(r.normal(size=B)),
            'y_delta': t(r.normal(size=(B, G))), 'plate': torch.tensor([0, 0, 0, 1, 1, 1][:B]),
            'pert': torch.arange(B)}


def built(seed=0, **kw):
    torch.manual_seed(seed)
    M, A, gv = priors()
    m = mv.LincsV10C(tiny_cfg(**kw), M, A, gv)
    r = np.random.default_rng(1)
    m.fit_bins(r.normal(size=(500, G)), r.normal(size=(500, G)))
    return m.eval()


# --- the chromatin pathways start as exact no-ops ----------------------------------------------------------------------
def test_chromatin_is_a_noop_at_init():
    m = built()
    b1 = make_batch()
    b2 = dict(b1, F=torch.rand_like(b1['F']), M=torch.zeros_like(b1['M']))
    with torch.no_grad():
        o1, o2 = m(b1), m(b2)
    assert torch.allclose(o1['delta'], o2['delta'], atol=1e-6)


@pytest.mark.parametrize('which', ['wc', 'gate', 'ppi', 'basal'])
def test_each_chromatin_pathway_carries_chromatin_once_opened(which):
    m = built()
    with torch.no_grad():
        if which == 'wc':
            for blk in m.perturb:
                blk.wc.weight.normal_(0, 0.3)
        elif which == 'gate':
            for blk in m.perturb:
                blk.gate[-1].weight.normal_(0, 0.5)
        elif which == 'ppi':
            m.ppi.w.weight.normal_(0, 0.3)
            m.ppi.gate.weight.normal_(0, 0.5)
        else:
            m.basal_c.weight.normal_(0, 0.3)
    b1 = make_batch()
    b2 = dict(b1, F=torch.rand_like(b1['F']))
    with torch.no_grad():
        d = (m(b1)['delta'] - m(b2)['delta']).abs().max()
    assert d > 1e-4


def test_gated_ppi_factorisation_matches_the_explicit_edge_weights():
    torch.manual_seed(0)
    A = (torch.rand(G, G) < 0.01).float()
    A = torch.maximum(A, A.T)
    p = mv.GatedPPI(A.numpy(), 16, 0.0).eval()
    with torch.no_grad():
        p.w.weight.normal_()
        p.gate.weight.normal_()
    h, c = torch.randn(2, G, 16), torch.randn(2, G, 16)
    with torch.no_grad():
        got, a = p(h, c)
        An = p.A
        W = An[None] * torch.sqrt(a[:, :, None] * a[:, None, :])
        want = p.w(torch.bmm(W, p.norm(h))) * p.has_edge
    assert torch.allclose(got, want, atol=1e-5)


def test_dose_scale_goes_to_zero_and_is_monotone():
    ds = mv.DoseScale(8)
    g = torch.randn(3, 8)
    lo, mid, hi = (ds(g, torch.full((3,), v)) for v in (-50.0, 0.0, 3.0))
    assert (lo < 1e-6).all() and (lo < mid).all() and (mid < hi).all()


def test_zero_dose_scale_removes_the_drug_effect():
    m = built()
    with torch.no_grad():
        m.dose.b.bias.fill_(1e6)                        # s = sigma(a (ld - 1e6)) = 0 for every row
    b1 = make_batch()
    b2 = dict(b1, u_feats=torch.randn_like(b1['u_feats']), atoms=torch.randn_like(b1['atoms']))
    with torch.no_grad():
        o1, o2 = m(b1), m(b2)
    # the cross-attention updates are zeroed; drug tokens still differ but cannot reach the genes
    assert torch.allclose(o1['delta'], o2['delta'], atol=1e-5)


def test_basal_term_is_drug_free():
    m = built()
    b1 = make_batch()
    b2 = dict(b1, u_feats=torch.randn_like(b1['u_feats']), dose=torch.randn(6), time=torch.randn(6))
    with torch.no_grad():
        o1, o2 = m(b1), m(b2)
    assert torch.allclose(o1['basal'], o2['basal'], atol=1e-6)
    assert not torch.allclose(o1['drug'], o2['drug'], atol=1e-4)
    assert torch.allclose(o1['delta'], o1['basal'] + o1['drug'], atol=1e-6)
    assert torch.allclose(o1['abs'], b1['x_ctl'] + o1['delta'])


def test_contrast_cancels_the_basal_term_for_plate_mates():
    m = built()
    with torch.no_grad():
        m.basal_h.bias.fill_(3.0)
        m.basal_c.weight.normal_()
    b = make_batch()
    with torch.no_grad():
        o = m(b)
    c_full, ok = mv.batch_contrast(o['delta'], b['plate'], b['pert'])
    c_drug, _ = mv.batch_contrast(o['drug'], b['plate'], b['pert'])
    assert ok.all() and torch.allclose(c_full, c_drug, atol=1e-5)


# --- contrast and loss --------------------------------------------------------------------------------------------------
def test_batch_contrast_by_hand():
    Y = torch.tensor([[1., 0.], [3., 0.], [5., 0.], [7., 1.], [9., 2.]])
    plate = torch.tensor([0, 0, 0, 1, 1])
    pert = torch.tensor([0, 1, 1, 2, 3])
    c, ok = mv.batch_contrast(Y, plate, pert)
    # row 0: mates rows 1, 2 (other compound) -> 1 - 4; row 1: only row 0 differs in compound -> 1 mate -> not ok
    assert ok.tolist() == [True, False, False, False, False]
    assert torch.allclose(c[0], torch.tensor([-3., 0.]))
    c1, ok1 = mv.batch_contrast(Y, plate, pert, min_mates=1)
    assert ok1.tolist() == [True, True, True, True, True] and torch.allclose(c1[3], torch.tensor([-2., -1.]))


def test_loss_finite_and_uses_contrast():
    m = built()
    m.train()
    b = make_batch()
    out = m(b)
    loss, parts = mv.v10c_loss(out, b, m.cfg)
    assert torch.isfinite(loss) and parts['n_contrast_rows'] == 6 and 'con' in parts
    loss.backward()
    assert m.drug_head.weight.grad is not None and float(m.drug_head.weight.grad.abs().sum()) > 0


# --- data -----------------------------------------------------------------------------------------------------------------
def test_leave_own_plate_out_by_hand():
    ctl = np.array([[0., 0.], [2., 2.], [4., 8.], [10., 10.]], np.float32)
    cell = np.array(['A', 'A', 'A', 'A', 'B'])
    ctl_id = np.array([0, 1, 2, 2, 3])
    x, single = dv.leave_own_plate_out(cell, ctl_id, ctl)
    assert np.allclose(x[0], [3., 5.]) and np.allclose(x[1], [2., 4.]) and np.allclose(x[2], [1., 1.])
    assert np.allclose(x[3], x[2]) and np.allclose(x[4], [10., 10.]) and single.tolist() == [0, 0, 0, 0, 1]


def test_plate_batches_keep_plates_whole_and_cover_each_row_once():
    plates = np.repeat(np.arange(30), np.random.default_rng(0).integers(1, 9, 30))
    bs = dv.plate_batches(plates, 16, np.random.default_rng(1), cap=16)
    allr = np.concatenate(bs)
    assert sorted(allr.tolist()) == list(range(len(plates)))
    for b in bs:
        assert len(b) <= 16
        for p in np.unique(plates[b]):
            assert (plates[b] == p).sum() == (plates == p).sum()
    big = dv.plate_batches(np.zeros(40, int), 16, np.random.default_rng(1), cap=5)
    assert sorted(np.concatenate(big).tolist()) == list(range(40)) and all(len(b) <= 16 for b in big)


def test_dev_carve_rule():
    rows = {'A': 600, 'B': 400, 'C': 6000, 'D': 1000, 'E': 2000, 'F': 3000, 'G': 700, 'H': 800, 'I': 900}
    axes = {c: (True, True, False) for c in rows}
    axes['D'] = (True, False, False)
    dev, pool = dv.dev_carve(list(rows), rows, axes, k=3, seed=0)
    assert pool == ['A', 'E', 'F', 'G', 'H', 'I'] and len(dev) == 3 and set(dev) <= set(pool)
    assert dv.dev_carve(list(rows), rows, axes, k=3, seed=0)[0] == dev


# --- the dataset and an end-to-end CPU run ---------------------------------------------------------------------------
def write_fixture(tmp, n_cells=8, rows_per=60):
    r = np.random.default_rng(0)
    cells = [f'C{i}' for i in range(n_cells)]
    cell, pert, plate, ctl_id, split, dose, time = [], [], [], [], [], [], []
    ctl = r.normal(size=(n_cells * 3, G)).astype(np.float32)
    for ci, c in enumerate(cells):
        for k in range(rows_per):
            cell.append(c); pert.append(f'p{k % 12}'); p = ci * 3 + (k % 3); plate.append(p); ctl_id.append(p)
            split.append('test' if ci >= n_cells - 2 else 'train'); dose.append(['0.1;0.04', '1.11', '10.0'][k % 3])
            time.append('24.0')
    n = len(cell)
    X = (ctl[np.array(ctl_id)] + r.normal(size=(n, G)) * 0.3).astype(np.float32)
    np.savez(tmp / 'v10c_full_substrate.npz', X=X, ctl=ctl, ctl_id=np.array(ctl_id), plate_id=np.array(plate),
             bench_row=np.arange(n), full_row=np.arange(n), meta_cell_iname=np.array(cell), meta_pert_id=np.array(pert),
             meta_pert_dose=np.array(dose), meta_pert_dose_idx=np.array([str([0, 4, 8][k % 3]) for k in range(n)]),
             meta_pert_time=np.array(time), split_cold_cell_1=np.array(split))
    F = r.random((n_cells, G, 16)).astype(np.float32)
    M = np.ones((n_cells, G, 4), bool)
    M[1] = False
    np.savez(tmp / 'v10c_chromatin.npz', F=F, M=M, cells=np.array(cells))
    json.dump({f'p{k}': k for k in range(12)}, open(tmp / 'drug_feature_index.json', 'w'))
    json.dump({'tissue': {c: ('lung' if i % 2 else 'breast') for i, c in enumerate(cells)}}, open(tmp / 'cell_tissue.json', 'w'))
    np.save(tmp / 'drug_descriptors.npy', r.normal(size=(12, 20)).astype(np.float32))
    np.save(tmp / 'drug_fingerprints.npy', (r.random((12, 2048)) < 0.05).astype(np.float32))
    np.save(tmp / 'drug_unimol.npy', r.normal(size=(12, 512)).astype(np.float32))
    na = r.integers(3, 8, 12)
    np.save(tmp / 'drug_atom_offsets.npy', np.concatenate([[0], np.cumsum(na)]))
    np.save(tmp / 'drug_atom_reprs.npy', r.normal(size=(int(na.sum()), 512)).astype(np.float32))
    Mp, A, gv = priors()
    np.save(tmp / 'M_pathway_v9.npy', Mp)
    np.save(tmp / 'STRING_adj_978_v9.npy', A)
    np.save(tmp / 'gene_vectors_978.npy', gv)
    return cells


def test_dataset_lookup_offswitch_and_holdout(tmp_path):
    cells = write_fixture(tmp_path)
    D = dv.V10CData([str(tmp_path)])
    assert set(D.cell[D.tr]).isdisjoint(set(D.cell[D.ev])) and set(D.cell[D.ev]) == {'C6', 'C7'}
    b = D.batch(D.tr[:4], 'cpu')
    assert torch.allclose(b['y_delta'], b['y_abs'] - b['x_ctl'])
    i = int(D.tr[0])
    assert np.allclose(b['F'][0].numpy(), np.load(tmp_path / 'v10c_chromatin.npz')['F'][cells.index(D.cell[i])])
    off = dv.V10CData([str(tmp_path)], chrom_off=True)
    assert np.allclose(off.F[0], off.F[5]) and np.allclose(off.M[1], off.M[0])
    assert not np.allclose(D.F[0], D.F[5])


def test_dev_mode_removes_dev_cells_from_training(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(dv, 'DEV_POOL_ROWS', (1, 10 ** 6))
    monkeypatch.setattr(dv, 'dev_carve', lambda tc, rpc, ax, **k: (['C0', 'C2'], sorted(set(tc))))
    D = dv.V10CData([str(tmp_path)], dev=True)
    assert set(D.cell[D.ev]) == {'C0', 'C2'} and not ({'C0', 'C2'} & set(D.cell[D.tr]))
    assert not ({'C6', 'C7'} & set(D.cell[D.tr]) | {'C6', 'C7'} & set(D.cell[D.ev]))


def test_end_to_end_cpu_training_and_scoring(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    rec = tv.main(['--roots', str(tmp_path), '--out', str(out), '--cpu_ok', '--epochs', '1', '--batch', '16',
                   '--d_model', '32'])
    s = rec['scores']
    assert s['n_rows'] == 120 and s['n_contrast_rows'] > 0 and np.isfinite(s['contrast_r_row_mean'])
    z = np.load(out / f"v10c_pred_{rec['tag']}.npz")
    assert z['delta'].shape == (120, G) and set(s['per_cell']) == {'C6', 'C7'}
    assert (out / f"V10C_COMPLETE_{rec['tag']}.json").exists()


def test_timing_mode_stops_and_records(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    rec = tv.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--timing_steps', '3',
                   '--batch', '16', '--d_model', '32'])
    assert rec['s_per_step'] > 0 and (tmp_path / 'o' / 'timing.json').exists()


def test_refuses_cpu_without_flag(tmp_path):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        pytest.skip('a GPU is present')
    with pytest.raises(SystemExit, match='no CUDA'):
        tv.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o')])


def test_heldout_scores_by_hand():
    T = np.random.default_rng(0).normal(size=(6, 20))
    s, rc, rd = tv.heldout_scores(T, T, np.array([0, 0, 0, 1, 1, 1]), np.arange(6), np.array(['A'] * 3 + ['B'] * 3))
    assert s['n_contrast_rows'] == 6 and abs(s['contrast_r_row_mean'] - 1) < 1e-9 and abs(s['delta_r_row_mean'] - 1) < 1e-9
    s2, _, _ = tv.heldout_scores(-T, T, np.array([0, 0, 0, 1, 1, 1]), np.arange(6), np.array(['A'] * 6))
    assert abs(s2['contrast_r_row_mean'] + 1) < 1e-9


# --- the GPU kernel builder ---------------------------------------------------------------------------------------------
def test_kernel_builder_pins_compile_and_layout(tmp_path):
    import py_compile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'orchestration'))
    import make_v10c_kernels as mk
    pins = mk.build_src(str(tmp_path / 'src'))
    assert set(pins) == set(mk.CODE) | set(mk.SMALL) | {mk.SUBSTRATE[0]} | set(mk.FROM_V9)
    for n in list(mk.CODE) + list(mk.SMALL):
        assert mk.sha1(str(tmp_path / 'src' / n)) == pins[n]
    d = mk.render('lincs-v10c-timing', [['t', ['--dev']]], pins, kern_dir=str(tmp_path / 'k'))
    py = os.path.join(d, 'lincs-v10c-timing.py')
    py_compile.compile(py, doraise=True)
    code = open(py, encoding='utf-8').read()
    assert repr(pins) in code and "'--dev'" in code and 'P100' in code
    meta = json.load(open(os.path.join(d, 'kernel-metadata.json')))
    assert meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape'] == 'NvidiaTeslaT4'
    assert meta['dataset_sources'] == mk.DATASETS


def test_gate_and_string_gate_start_at_their_registered_values():
    m = built()
    with torch.no_grad():
        _, aux = m(make_batch(), return_aux=True)
    assert torch.allclose(aux['gamma'], torch.ones_like(aux['gamma']))
    assert torch.allclose(aux['a_ppi'], torch.full_like(aux['a_ppi'], 0.95), atol=1e-5)


def test_rows_of_featureless_compounds_are_dropped(tmp_path):
    write_fixture(tmp_path)
    json.dump({f'p{k}': k for k in range(11)}, open(tmp_path / 'drug_feature_index.json', 'w'))   # p11 unfeaturised
    D = dv.V10CData([str(tmp_path)])
    assert 'p11' not in set(D.pert) and D.n_dropped['train'] == 30 and D.n_dropped['test'] == 10


def test_heldout_contrast_needs_two_mates():
    T = np.random.default_rng(1).normal(size=(5, 20))
    s, rc, _ = tv.heldout_scores(T, T, np.array([0, 0, 1, 1, 1]), np.arange(5), np.array(['A'] * 5))
    assert s['n_contrast_rows'] == 3 and np.isnan(rc[:2]).all()


def test_budget_guard_refuses_a_run_that_would_be_cut(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    with pytest.raises(SystemExit, match='budget'):
        tv.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--epochs', '4', '--batch', '16',
                 '--d_model', '32', '--budget_h', '1e-9'])


def test_unavailable_features_are_masked_out():
    torch.manual_seed(0)
    cs = mv.ChromatinStream(16, 8)
    F = torch.rand(2, G, 16)
    Mk = torch.ones(2, G, 4)
    Mk[:, :, 1] = 0                                        # ACT missing: its 5 window features must not matter
    F2 = F.clone()
    F2[..., 5:10] = torch.rand(2, G, 5) * 10
    with torch.no_grad():
        assert torch.allclose(cs(F, Mk), cs(F2, Mk), atol=1e-6)
    assert torch.equal(mv.mask_per_feature(Mk)[0, 0], torch.tensor([1.] * 5 + [0.] * 5 + [1.] * 5 + [1.]))


def test_dose_comes_from_the_dose_index_not_the_pooled_string(tmp_path):
    write_fixture(tmp_path)
    D = dv.V10CData([str(tmp_path)])
    ld = np.log10(np.array([0.1, 1.11, 10.0]))
    raw = ld[np.arange(len(D.X)) % 3]
    assert np.allclose(D.dose_n, (raw - raw[D.tr].mean()) / (raw[D.tr].std() + 1e-6), atol=1e-5)
    assert dv.DOSE_OF_IDX[9] == 40.0 and len(dv.DOSE_OF_IDX) == 10


def test_permuted_chromatin_kinds():
    r = np.random.default_rng(0)
    F = r.random((4, G, 16)).astype(np.float32)
    M = (r.random((4, G, 4)) > 0.3).astype(np.float32)
    Fg, Mg = tv.permuted_chromatin(F, M, [1, 3], 'gene_shuffle', np.random.default_rng(1))
    assert np.allclose(Fg[0], F[0]) and np.allclose(Fg[2], F[2]) and not np.allclose(Fg[1], F[1])
    assert np.allclose(np.sort(Fg[1].ravel()), np.sort(F[1].ravel())) and np.allclose(Mg[1].sum(), M[1].sum())
    Fs, Ms = tv.permuted_chromatin(F, M, [3, 1], 'cell_swap', np.random.default_rng(1))
    assert np.allclose(Fs[1], F[3]) and np.allclose(Fs[3], F[1]) and np.allclose(Fs[0], F[0]) and np.allclose(Ms[1], M[3])


def test_end_to_end_reports_the_permutations_and_restores_chromatin(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    rec = tv.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--epochs', '1', '--batch', '16',
                   '--d_model', '32'])
    pr = rec['reported_permutations']
    assert set(pr) == {'gene_shuffle', 'cell_swap'} and set(pr['cell_swap']['per_cell_contrast_r']) == {'C6', 'C7'}
    assert np.isfinite(pr['gene_shuffle']['contrast_r_row_mean'])
    off = tv.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o2'), '--cpu_ok', '--epochs', '1', '--batch', '16',
                   '--d_model', '32', '--chrom_off'])
    assert off['reported_permutations'] == {}


def test_heldout_alias_in_training_is_refused(tmp_path):
    """Review 088 C1: the held-out check canonicalises names, so H1975 held out with NCIH1975 in training is refused."""
    write_fixture(tmp_path)
    z = dict(np.load(tmp_path / 'v10c_full_substrate.npz'))
    z['meta_cell_iname'] = np.where(z['meta_cell_iname'] == 'C7', 'H1975', z['meta_cell_iname'])
    z['meta_cell_iname'] = np.where(z['meta_cell_iname'] == 'C0', 'NCIH1975', z['meta_cell_iname'])
    np.savez(tmp_path / 'v10c_full_substrate.npz', **z)
    ch = dict(np.load(tmp_path / 'v10c_chromatin.npz'))
    ch['cells'] = np.array(['NCIH1975' if c == 'C0' else 'H1975' if c == 'C7' else c for c in ch['cells']])
    np.savez(tmp_path / 'v10c_chromatin.npz', **ch)
    t = json.load(open(tmp_path / 'cell_tissue.json'))
    t['tissue']['H1975'] = t['tissue'].pop('C7'); t['tissue']['NCIH1975'] = t['tissue'].pop('C0')
    json.dump(t, open(tmp_path / 'cell_tissue.json', 'w'))
    with pytest.raises(SystemExit, match='under any of its names'):
        dv.V10CData([str(tmp_path)])


# --- RESULTS 100.8: the drug-path guard -------------------------------------------------------------------------------
def test_model_returns_the_dose_scale_it_applies():
    m = built()
    with torch.no_grad():
        out, aux = m(make_batch(), return_aux=True)
        out2 = m(make_batch())
    assert out['dose_scale'].shape == (6,) and torch.equal(out['dose_scale'], aux['dose_scale'])
    assert torch.equal(out['delta'], out2['delta'])


def test_drug_path_guard_rule_by_hand():
    g = tv.DrugPathGuard(start=4, every=2, window=3, floor=1e-4)
    for _ in range(2):
        g.update(0.0)
    assert not g.failed(4)                      # the window is not full
    g.update(0.0)
    assert not g.failed(3) and not g.failed(5)  # before start; not on the check cadence
    assert g.failed(4) and g.failed(6)
    for v in (0.5, 0.5):
        g.update(v)
    assert len(g.meds) == 3 and not g.failed(8)  # median 0.5
    g2 = tv.DrugPathGuard(start=1, every=1, window=3, floor=1e-4)
    for v in (0.0, 0.0, 1e-3):
        g2.update(v)
    assert g2.failed(1)                          # median 0 < floor although one batch is above it


def _tight_guard(monkeypatch):
    real = tv.DrugPathGuard
    monkeypatch.setattr(tv, 'DrugPathGuard', lambda: real(start=2, every=1, window=2, floor=1e-4))


def test_guard_stops_a_run_whose_dose_scale_is_zero(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    monkeypatch.setattr(mv.DoseScale, 'forward', lambda self, g, ld: torch.zeros_like(ld))
    _tight_guard(monkeypatch)
    with pytest.raises(SystemExit) as e:
        tv.main(['--roots', str(tmp_path), '--out', str(out), '--cpu_ok', '--epochs', '1', '--batch', '16',
                 '--d_model', '32', '--guard'])
    assert e.value.code == tv.GUARD_EXIT
    fails = list(out.glob('V10C_FAILED_*.json'))
    assert len(fails) == 1 and not list(out.glob('v10c_pred_*')) and not list(out.glob('V10C_COMPLETE_*'))
    f = json.load(open(fails[0]))
    assert f['step'] == 2 and f['guard_level'] == 0.0 and f['tag'].endswith('_s0')


def test_guard_lets_a_healthy_run_finish_and_logs_the_dose_scale(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    _tight_guard(monkeypatch)
    rec = tv.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--epochs', '1', '--batch', '16',
                   '--d_model', '32', '--guard'])
    h = rec['hist'][0]
    assert rec['guard'] and 0 < h['dose_scale_batch_median_mean'] < 1 and 0 < h['gamma_mean_last_batch'] < 2


def test_without_guard_a_dead_dose_scale_runs_to_the_end(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    monkeypatch.setattr(mv.DoseScale, 'forward', lambda self, g, ld: torch.zeros_like(ld))
    _tight_guard(monkeypatch)
    rec = tv.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--epochs', '1', '--batch', '16',
                   '--d_model', '32'])
    assert not rec['guard'] and rec['hist'][0]['dose_scale_batch_median_mean'] == 0.0


def test_next_attempt_bumps_the_seed_and_respects_the_budget():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'orchestration'))
    import make_v10c_kernels as mk
    assert mk.GUARD_EXIT == tv.GUARD_EXIT
    args = ['--seed', '1', '--guard']
    assert mk.next_attempt(args, 0, 50.0, 3.8) == args                 # the registered run always starts
    assert mk.next_attempt(args, 1, 2.0, 3.8) == ['--seed', '11', '--guard']
    assert mk.next_attempt(args, 2, 6.0, 3.8) == ['--seed', '21', '--guard']
    assert mk.next_attempt(args, 1, 7.3, 3.8) is None                  # 7.3 + 3.8 > 11
    assert mk.next_attempt(args, 3, 0.0, 3.8) is None                  # at most two retries
    assert args == ['--seed', '1', '--guard']                          # not mutated


def test_guarded_kernel_renders_and_compiles(tmp_path):
    import py_compile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'orchestration'))
    import make_v10c_kernels as mk
    pins = mk.build_src(str(tmp_path / 'src'))
    d = mk.render('lincs-v10c-tc-on-s0', [['tc_on_s0', ['--seed', '0', '--guard']]], pins, kern_dir=str(tmp_path / 'k'),
                  run_h=5.7)
    py = os.path.join(d, 'lincs-v10c-tc-on-s0.py')
    py_compile.compile(py, doraise=True)
    code = open(py, encoding='utf-8').read()
    assert 'def next_attempt(' in code and 'RUN_H = 5.7' in code and 'GUARD_EXIT = 3' in code and "'attempts'" in code


# --- review 090: C5 (the guard at the final step), C4 (--start_attempt) and the trained-run criterion --------------------
def test_guard_checks_the_final_step(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(tv, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    monkeypatch.setattr(mv.DoseScale, 'forward', lambda self, g, ld: torch.zeros_like(ld))
    real = tv.DrugPathGuard
    monkeypatch.setattr(tv, 'DrugPathGuard', lambda: real(start=10 ** 6, every=500, window=2, floor=1e-4))
    with pytest.raises(SystemExit) as e:                # no cadence check can fire in this run: only the final one
        tv.main(['--roots', str(tmp_path), '--out', str(out), '--cpu_ok', '--epochs', '1', '--batch', '16',
                 '--d_model', '32', '--guard'])
    assert e.value.code == tv.GUARD_EXIT and not list(out.glob('v10c_pred_*'))
    g = real(start=10 ** 6, every=500, window=2)
    g.update(0.0)
    assert not g.failed(7, final=True)                  # the window must still be full
    g.update(0.0)
    assert g.failed(7, final=True) and not g.failed(7)


def test_start_attempt_renders_and_is_refused_without_guard(tmp_path):
    import py_compile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'orchestration'))
    import make_v10c_kernels as mk
    pins = mk.build_src(str(tmp_path / 'src'))
    d = mk.render('lincs-v10c-tc-on-s1-r1', [['tc_on_s1', ['--seed', '1', '--guard']]], pins, kern_dir=str(tmp_path / 'k'),
                  start_attempt=1)
    py = os.path.join(d, 'lincs-v10c-tc-on-s1-r1.py')
    py_compile.compile(py, doraise=True)
    code = open(py, encoding='utf-8').read()
    assert 'START_ATTEMPT = 1' in code and "'--seed', '1'" in code      # the registered seed; the bump happens at run time
    assert 'k, attempts[tag] = (START_ATTEMPT if guard else 0), []' in code
    assert mk.next_attempt(['--seed', '1', '--guard'], 1, 0.0, 3.8) == ['--seed', '11', '--guard']
    with pytest.raises(SystemExit, match='needs --guard'):
        mk.render('x', [['t', ['--seed', '1']]], pins, kern_dir=str(tmp_path / 'k2'), start_attempt=1)
    with pytest.raises(SystemExit, match='two retries'):
        mk.render('x', [['t', ['--seed', '1', '--guard']]], pins, kern_dir=str(tmp_path / 'k3'), start_attempt=3)


def test_trained_check_verdict_and_dose_terms():
    import tb_trained_check as tc
    v = tc.verdict(np.array([0.0, 0.0, 1e-8, 5e-5, 0.3]), np.array([-120.0, -90.0, -18.4, -9.9, -0.8]))
    assert not v['drug_path_trained'] and v['n_s_exactly_0'] == 2 and v['frac_s_below_1e-6'] == 0.6
    assert v['z_min'] == -120.0
    assert tc.verdict(np.array([1e-4, 1e-4, 0.0]), np.zeros(3))['drug_path_trained']    # median == floor: trained
    m = built()
    with torch.no_grad():
        b = make_batch()
        s, z = tc.dose_terms(m, b)
        out = m(b)
    assert torch.allclose(s, torch.sigmoid(z), atol=1e-6) and torch.equal(s, out['dose_scale'])


def test_trained_check_draws_the_diagnostic_rows():
    import tb_trained_check as tc

    class _D:
        ev, tr = np.arange(1000, 2000), np.arange(5000)
    rng = np.random.default_rng(0)
    np.sort(rng.choice(_D.ev, 480, replace=False))
    assert np.array_equal(tc.training_rows(_D()), np.sort(rng.choice(_D.tr, 480, replace=False)))


# --- the rendered kernel EXECUTED with a stub trainer (100.8's retry loop; review 090 C4) --------------------------------
STUB_TRAINER = r'''
import json, os, sys
a = sys.argv
seed, out = a[a.index('--seed') + 1], a[a.index('--out') + 1]
fail = os.environ['STUB_FAIL_SEEDS'].split(',')
if seed in fail:
    json.dump({'seed': seed}, open(os.path.join(out, f'V10C_FAILED_s{seed}.json'), 'w'))
    sys.exit(int(os.environ.get('STUB_CODE', '3')))
json.dump({'seed': seed}, open(os.path.join(out, f'V10C_COMPLETE_s{seed}.json'), 'w'))
'''


def _run_kernel(tmp_path, runs, fail_seeds, start_attempt=0, run_h=1.0, code='3'):
    import subprocess, hashlib
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'orchestration'))
    import make_v10c_kernels as mk
    inp, work = tmp_path / 'input' / 'ds', tmp_path / 'working'
    inp.mkdir(parents=True)
    work.mkdir()
    names = list(mk.CODE) + list(mk.SMALL) + [mk.SUBSTRATE[0]] + list(mk.FROM_V9)
    pins = {}
    for n in names:
        body = STUB_TRAINER if n == 'train_v10c.py' else f'stub {n}'
        (inp / n).write_text(body, encoding='utf-8')
        pins[n] = hashlib.sha1((inp / n).read_bytes()).hexdigest()
    d = mk.render('lincs-v10c-x', runs, pins, kern_dir=str(tmp_path / 'k'), run_h=run_h, start_attempt=start_attempt)
    code_txt = open(os.path.join(d, 'lincs-v10c-x.py'), encoding='utf-8').read()
    probe = "'import json, torch; print(json.dumps([torch.cuda.get_device_name(i) '"
    assert code_txt.count(probe) == 1 and code_txt.count('os.symlink(') == 1 and code_txt.count("'/kaggle/input/**/'") == 1
    assert code_txt.count("'/kaggle/working/src', '/kaggle/working/data', '/kaggle/working'") == 1
    code_txt = (code_txt.replace(probe, "'import json; print(json.dumps([\"Tesla T4\", \"Tesla T4\"])) #'")
                .replace('os.symlink(', 'shutil.copyfile(')
                .replace("'/kaggle/input/**/'", repr(str(tmp_path / 'input').replace('\\', '/') + '/**/'))
                .replace("'/kaggle/working/src', '/kaggle/working/data', '/kaggle/working'",
                         ', '.join(repr(str(p).replace('\\', '/')) for p in (work / 'src', work / 'data', work))))
    py = tmp_path / 'kernel.py'
    py.write_text(code_txt, encoding='utf-8')
    env = dict(os.environ, STUB_FAIL_SEEDS=','.join(fail_seeds), STUB_CODE=code)
    r = subprocess.run([sys.executable, str(py)], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    rec = work / 'V10C_KERNEL_COMPLETE.json'
    return r, (json.load(open(rec)) if rec.exists() else None)


def test_executed_kernel_retries_a_guard_failure_with_seed_plus_10(tmp_path):
    r, rec = _run_kernel(tmp_path, [['tc_on_s0', ['--seed', '0', '--guard']]], fail_seeds=['0'])
    assert r.returncode == 0, r.stderr[-2000:]
    a = rec['attempts']['tc_on_s0']
    assert rec['done'] == ['tc_on_s0'] and [x['returncode'] for x in a] == [3, 0]
    assert a[1]['args'] == ['--seed', '10', '--guard'] and a[1]['out'].endswith('tc_on_s0_retry1')


def test_executed_kernel_stops_after_two_retries(tmp_path):
    r, rec = _run_kernel(tmp_path, [['tc_on_s0', ['--seed', '0', '--guard']]], fail_seeds=['0', '10', '20'])
    assert r.returncode == 0 and rec['done'] == [] and len(rec['attempts']['tc_on_s0']) == 3


def test_executed_kernel_start_attempt_bumps_from_the_registered_seed(tmp_path):
    r, rec = _run_kernel(tmp_path, [['tc_on_s1', ['--seed', '1', '--guard']]], fail_seeds=['11'], start_attempt=1)
    a = rec['attempts']['tc_on_s1']
    assert rec['done'] == ['tc_on_s1'] and [x['args'][1] for x in a] == ['11', '21']


def test_executed_kernel_no_retry_without_guard_and_other_exit_codes_are_fatal(tmp_path):
    r, rec = _run_kernel(tmp_path, [['tb_on_s0', ['--seed', '0']]], fail_seeds=['0'])
    assert r.returncode != 0 and 'FATAL: run tb_on_s0 exited 3' in (r.stdout + r.stderr) and rec is None
    r2, rec2 = _run_kernel(tmp_path / 'b', [['tc_on_s0', ['--seed', '0', '--guard']]], fail_seeds=['0'], code='1')
    assert r2.returncode != 0 and 'exited 1' in (r2.stdout + r2.stderr) and rec2 is None


def test_executed_kernel_retry_that_does_not_fit_the_budget_is_left_for_a_new_kernel(tmp_path):
    r, rec = _run_kernel(tmp_path, [['tc_on_s0', ['--seed', '0', '--guard']]], fail_seeds=['0'], run_h=11.5)
    assert r.returncode == 0 and rec['done'] == [] and len(rec['attempts']['tc_on_s0']) == 1
    assert 'NO ATTEMPT LEFT IN THIS KERNEL' in r.stdout
