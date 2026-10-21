"""Tests for v11 (RESULTS 102 / 102.1): the base is T-B off's, FG starts as an exact no-op, the arms switch what they
register, the target rules hold, and the trainer's recipe is train_v10c's. CPU only; tiny widths."""
import csv, hashlib, json, os, sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
import model_v10c as mv  # noqa: E402
import data_v10c as dv  # noqa: E402
import train_v10c as tv  # noqa: E402
import model_v11 as m11  # noqa: E402
import data_v11 as d11  # noqa: E402
import train_v11 as t11  # noqa: E402
from test_v10c_model import G, tiny_cfg, priors, write_fixture  # noqa: E402

N_EXTRA = 120
N = G + N_EXTRA
REPO = os.path.dirname(os.path.dirname(HERE))


# --- a v11 fixture on top of test_v10c_model's -------------------------------------------------------------------------
def write_v11_fixture(tmp, n_cells=8, rows_per=60):
    cells = write_fixture(tmp, n_cells, rows_per)
    r = np.random.default_rng(5)
    nodes = [f'L{i}' for i in range(G)] + [f'X{i}' for i in range(N_EXTRA)]
    order = r.permutation(N)                                  # landmarks scattered among the nodes
    nodes = [nodes[i] for i in order]
    lm = np.array([nodes.index(f'L{i}') for i in range(G)], np.int32)
    E = set()
    while len(E) < 3 * N:
        i, j = sorted(r.integers(0, N, 2))
        if i != j:
            E.add((i, j))
    E = np.array(sorted(E), np.int32).T
    np.savez(tmp / 'string_graph_v11_expdb.npz', nodes=np.array(nodes), edge_index=E,
             weight=r.uniform(0.4, 1.0, E.shape[1]).astype(np.float32), landmark_idx=lm)
    np.save(tmp / d11.GENE_VEC_FG, r.normal(size=(N, 12)).astype(np.float32))
    F = r.random((n_cells, N, 16)).astype(np.float16)
    M = np.ones((n_cells, N, 4), bool)
    M[1] = False
    np.savez(tmp / 'v11_chromatin.npz', F=F, M=M, cells=np.array(cells), genes=np.array(nodes))
    ccle_cells = cells[:5]
    expr = r.gamma(2.0, 2.0, size=(len(ccle_cells), N)).astype(np.float32)
    expr[:, :7] = np.nan
    np.savez(tmp / 'ccle_expr_v11.npz', cells=np.array(ccle_cells), nodes=np.array(nodes), expr_nodes=expr)
    # compounds p0..p11: p0-p5 annotated; p6 probe; p7 and p11 share a molecule; p9 human-free
    tg = {'p0': [('L3', 'INHIBITOR'), ('X5', 'INHIBITOR')], 'p1': [('X7', 'AGONIST')], 'p2': [('L9', 'MODULATOR')],
          'p3': [('L3', 'INHIBITOR'), ('L3', 'AGONIST')], 'p4': [('X9', 'BLOCKER')], 'p5': [('L20', 'OPENER')],
          'p6': [('X11', 'INHIBITOR')], 'p7': [('L30', 'INHIBITOR')], 'p11': [('X13', 'ANTAGONIST')]}
    with open(tmp / 'chembl_dti_edges.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['pert_id', 'gene_symbol', 'organism', 'action_type'])
        for p, lst in tg.items():
            for gsym, act in lst:
                w.writerow([p, gsym, 'Homo sapiens', act])
        w.writerow(['p9', 'L5', 'Bacteria', 'INHIBITOR'])
        w.writerow(['p8', 'NOT_A_NODE', 'Homo sapiens', 'INHIBITOR'])
    with open(tmp / 'drug_inchikeys.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['pert_id', 'block1'])
        for k in range(12):
            w.writerow([f'p{k}', 'MOL7' if k in (7, 11) else f'MOL{k}'])
    json.dump({'sign': {'INHIBITOR': -1, 'BLOCKER': -1, 'ANTAGONIST': -1, 'AGONIST': 1, 'OPENER': 1, 'MODULATOR': 0}},
              open(tmp / 'action_sign_map.json', 'w'))
    json.dump({'probe': ['p6']}, open(tmp / 'dti_probe_split.json', 'w'))
    return cells, nodes, lm


def v11_data(tmp, **kw):
    return d11.V11Data([str(tmp)], **kw)


def unseen_p11(tmp):
    """Make p11 a compound with no training row: its rows move to the test cells."""
    z = dict(np.load(tmp / 'v10c_full_substrate.npz'))
    pert = z['meta_pert_id'].copy()
    tr = z['split_cold_cell_1'] == 'train'
    pert[tr & (pert == 'p11')] = 'p10'
    z['meta_pert_id'] = pert
    np.savez(tmp / 'v10c_full_substrate.npz', **z)


def built_pair(tmp, seed=0, arm='full', **kw):
    D = v11_data(tmp, arm=arm)
    kw = dict(kw, n_lineage=D.L.shape[1])
    Mp, A, gv = priors()
    torch.manual_seed(seed)
    base = mv.LincsV10C(tiny_cfg(**kw), Mp, A, gv)
    torch.manual_seed(seed)
    v11 = m11.LincsV11(tiny_cfg(**kw), Mp, A, gv, D.fg_inputs())
    Xc, Xm = D.fit_rows()
    base.fit_bins(Xc, Xm)
    v11.fit_bins(Xc, Xm)
    return D, base, v11


# --- the base is T-B off's ---------------------------------------------------------------------------------------------
def test_base_init_is_v10c_under_the_same_seed_and_the_global_stream_is_unchanged(tmp_path):
    write_v11_fixture(tmp_path)
    D = v11_data(tmp_path)
    Mp, A, gv = priors()
    torch.manual_seed(3)
    base = mv.LincsV10C(tiny_cfg(), Mp, A, gv)
    after_base = torch.rand(5)
    torch.manual_seed(3)
    v11 = m11.LincsV11(tiny_cfg(), Mp, A, gv, D.fg_inputs())
    after_v11 = torch.rand(5)
    assert torch.equal(after_base, after_v11)
    sb, s11 = base.state_dict(), v11.state_dict()
    assert set(sb) <= set(s11)
    assert all(torch.equal(sb[k], s11[k]) for k in sb)
    assert all(k.startswith(('fg.', 'target_head.')) for k in set(s11) - set(sb))


def test_constructing_v11_does_not_reseed_any_cuda_generator(tmp_path, monkeypatch):
    """Review 099 C1: the FG fork must seed the CPU generator only; a CUDA re-seed would change training's dropout stream
    on GPU, which fork_rng(devices=[]) does not restore."""
    write_v11_fixture(tmp_path)
    D = v11_data(tmp_path)
    Mp, A, gv = priors()
    torch.manual_seed(3)
    calls = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda s: calls.append(('all', s)))
    monkeypatch.setattr(torch.cuda, 'manual_seed', lambda s: calls.append(('one', s)))
    m11.LincsV11(tiny_cfg(), Mp, A, gv, D.fg_inputs())
    assert calls == []


@pytest.mark.parametrize('mode', ['eval', 'train'])
def test_forward_equals_tb_off_at_init_with_identical_base_weights(tmp_path, mode):
    write_v11_fixture(tmp_path)
    D, base, v11 = built_pair(tmp_path, dropout=0.1, stoch_depth=0.1)
    v11.load_state_dict(base.state_dict(), strict=False)
    getattr(base, mode)()
    getattr(v11, mode)()
    rows = np.concatenate([D.tr[:10], D.ev[:6]])
    b = D.batch(rows, 'cpu')
    torch.manual_seed(11)
    o1 = base(b)
    torch.manual_seed(11)
    o2 = v11(b)
    for k in ('delta', 'drug', 'basal', 'dose_scale'):
        assert torch.equal(o1[k], o2[k]), k


def test_base_inputs_are_tb_offs(tmp_path):
    write_v11_fixture(tmp_path)
    off = dv.V10CData([str(tmp_path)], chrom_off=True)
    for arm in d11.ARMS:
        D = v11_data(tmp_path, arm=arm)
        b0, b1 = off.batch(off.tr[:9], 'cpu'), D.batch(D.tr[:9], 'cpu')
        for k, v in b0.items():
            assert torch.equal(v, b1[k]), (arm, k)


def test_pinned_base_files_are_tb_offs():
    """102.1 C1: data_v10c and the modules are T-B's pinned bytes; model_v10c differs from T-B's pin only by the 100.8
    output key (s is returned, not used), shown by undoing that one line and recovering T-B's sha1."""
    sha = lambda b: hashlib.sha1(b).hexdigest()
    rd = lambda p: open(os.path.join(REPO, p), 'rb').read()
    assert sha(rd('model/v10c/data_v10c.py')) == '6efd1b4a2d545090616b05b81c416a3a7ad282bd'
    assert sha(rd('model/v9/modules_v9.py')) == '4140bf702be5b1a75d1f767a53d9290d8d905d22'
    assert sha(rd('model/v7/modules_v7.py')) == '8c7517e87fe9e3a29e807253095760b6ee40de1f'
    m = rd('model/v10c/model_v10c.py')
    new = (b"        out = {'delta': delta, 'abs': x_ctl + delta, 'drug': drug, 'basal': basal, 'dose_scale': s}"
           b"   # s: 100.8's guard")
    old = b"        out = {'delta': delta, 'abs': x_ctl + delta, 'drug': drug, 'basal': basal}"
    assert m.count(new) == 1
    assert sha(m.replace(new, old)) == '8e7e509cbf05ddc7882ddba59629fd39d8909482'


def test_trainer_recipe_is_train_v10cs(tmp_path, monkeypatch):
    """With FG's readout frozen at zero and the targets off, train_v11 reproduces train_v10c (chrom_off) step for step:
    same batches, optimiser, schedule, clipping, dropout stream and scorer."""
    write_v11_fixture(tmp_path)
    cfg = lambda **kw: tiny_cfg(**dict(kw, dropout=0.1, stoch_depth=0.1))
    monkeypatch.setattr(tv, 'V10CConfig', cfg)
    monkeypatch.setattr(t11, 'V10CConfig', cfg)
    real = m11.LincsV11.__init__

    def frozen(self, *a, **k):
        real(self, *a, **k)
        self.fg.readout.weight.requires_grad_(False)
        self.fg.readout.bias.requires_grad_(False)
    monkeypatch.setattr(m11.LincsV11, '__init__', frozen)
    args = ['--roots', str(tmp_path), '--cpu_ok', '--epochs', '1', '--batch', '16', '--d_model', '32', '--seed', '4']
    r0 = tv.main(args + ['--out', str(tmp_path / 'a'), '--chrom_off'])
    r1 = t11.main(args + ['--out', str(tmp_path / 'b'), '--arm', 'no_targets'])
    p0 = np.load(tmp_path / 'a' / f"v10c_pred_{r0['tag']}.npz")
    p1 = np.load(tmp_path / 'b' / f"v11_pred_{r1['tag']}.npz")
    assert np.allclose(p0['delta'], p1['delta'], atol=1e-6, rtol=0)
    assert abs(r0['hist'][0]['loss'] - r1['hist'][0]['loss']) < 1e-6


# --- FG: no-op at init, opens, gating, the target signal --------------------------------------------------------------
def test_fg_carries_cell_and_target_information_once_opened_and_basal_stays_drug_free(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    v11.eval()
    with torch.no_grad():
        v11.fg.readout.weight.normal_(0, 0.3)
    b = D.batch(D.tr[:8], 'cpu')
    with torch.no_grad():
        o = v11(b)
        bF = dict(b, cell_row=(b['cell_row'] + 1) % 8)              # another cell's FG tables
        assert not torch.allclose(v11(bF)['delta'], o['delta'])
        k = torch.nonzero(b['tgt_known']).flatten()
        assert len(k)
        bt = dict(b, tgt_known=torch.zeros_like(b['tgt_known']))
        o2 = v11(bt)
        assert not torch.allclose(o2['delta'][k], o['delta'][k])
        assert torch.equal(o2['basal'], o['basal'])


def test_fg_gate_factorisation_matches_the_explicit_edge_weights(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    T = v11.fg.tables.get('cpu')
    with torch.no_grad():
        v11.fg.gate.weight.normal_(0, 1.0)
        uc = torch.tensor([0, 3])
        F, M = T['F'][uc].float(), T['M'][uc].float()
        chrom = torch.cat([F * mv.mask_per_feature(M), M], -1)
        ce = torch.stack([T['E'][uc].float(), T['Em'][uc].float()], -1)
        a = torch.sigmoid(v11.fg.gate(torch.cat([chrom, ce], -1))).squeeze(-1)
        X = torch.randn(2, N, 3)
        Ad = T['A'].to_dense()
        r = a.sqrt().unsqueeze(-1)
        fact = r * m11.spmm(T['A'], r * X)
        for u in range(2):
            explicit = (a[u].sqrt()[:, None] * Ad * a[u].sqrt()[None, :]) @ X[u]
            assert torch.allclose(fact[u], explicit, atol=1e-5)
    assert torch.allclose(Ad, Ad.T) and float(Ad.diagonal().abs().max()) == 0.0


def test_gate_and_rounds_start_at_their_registered_values(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    b = D.batch(D.tr[:8], 'cpu')
    with torch.no_grad():
        out, aux = v11(b, return_aux=True)
    assert torch.allclose(aux['fg_gate'], torch.full_like(aux['fg_gate'], 0.95), atol=1e-6)
    assert all(float(w.weight.abs().max()) == 0 for w in v11.fg.rounds)
    assert float(aux['fg_readout'].abs().max()) == 0.0


def test_target_signal_known_rows_scatter_and_others_take_the_head(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    T = v11.fg.tables.get('cpu')
    rows = np.array([np.flatnonzero(D.pert[D.tr] == p)[0] for p in ('p0', 'p3', 'p6', 'p9')])
    b = D.batch(D.tr[rows], 'cpu')
    nid = {n: i for i, n in enumerate(np.load(tmp_path / 'string_graph_v11_expdb.npz')['nodes'])}
    p = torch.rand(4, len(D.vocab))
    t = v11.fg.target_signal(b, p, T)
    # p0: L3 inhibitor, X5 inhibitor
    assert set(torch.nonzero(t[0, :, 0]).flatten().tolist()) == {nid['L3'], nid['X5']}
    assert t[0, nid['L3'], 1] == -1 and t[0, nid['X5'], 1] == -1
    # p3: L3 as inhibitor and agonist -> sign 0
    assert t[1, nid['L3'], 0] == 1 and t[1, nid['L3'], 1] == 0
    # p6 (probe) and p9 (no human target): the head's probabilities on the vocabulary, sign 0
    for i in (2, 3):
        assert torch.allclose(t[i, torch.as_tensor(D.vocab), 0], p[i]) and float(t[i, :, 1].abs().max()) == 0
        off = np.setdiff1d(np.arange(N), D.vocab)
        assert float(t[i, off, 0].abs().max()) == 0


def test_hops_by_hand_for_a_single_target_compound(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    with torch.no_grad():
        v11.fg.readout.weight.zero_()
        w = v11.fg.readout.weight
        w[0, m11.D_G + 2] = 1.0                            # output channel 0 reads s channel 2 (any, hop 1)
    i = np.flatnonzero(D.pert[D.tr] == 'p1')[0]
    b = D.batch(D.tr[[i]], 'cpu')
    nid = {n: k for k, n in enumerate(np.load(tmp_path / 'string_graph_v11_expdb.npz')['nodes'])}
    T = v11.fg.tables.get('cpu')
    t = torch.zeros(N)
    t[nid['X7']] = 1.0
    hop1 = 0.95 * (T['A'].to_dense() @ t)
    with torch.no_grad():
        out, _ = v11.fg(b, torch.zeros(1, len(D.vocab)))
    want = hop1[T['lm']] / T['s_scale'][2]
    assert torch.allclose(out[0, :, 0], want, atol=1e-5)


def test_targets_off_arm_has_no_signal_no_head_loss_and_no_head_gradient(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path, arm='no_targets')
    assert not D.known_input.any() and not v11.fg.targets_on
    b = D.batch(D.tr[:16], 'cpu')
    with torch.no_grad():
        v11.fg.readout.weight.normal_(0, 0.3)
    out = v11(b)
    loss, parts = m11.v11_loss(out, b, v11.cfg, 0.1, targets_on=False)
    assert 'target_bce' not in parts
    loss.backward()
    assert v11.target_head.weight.grad is None
    feat_s = v11.fg.readout.weight.grad[:, m11.D_G:m11.D_G + 6]
    assert float(feat_s.abs().max()) == 0.0


def test_head_loss_only_on_known_rows_and_its_labels(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    b = D.batch(D.tr[:40], 'cpu')
    k = D.pert_code[D.tr[:40]]
    assert torch.equal(b['tgt_known'], torch.as_tensor(D.status[k] == 0))
    assert float(b['head_y'][~b['tgt_known']].abs().max()) == 0
    out = v11(b)
    loss, parts = m11.v11_loss(out, b, v11.cfg, 0.1, True)
    base, _ = mv.v10c_loss(out, b, v11.cfg)
    ok = b['tgt_known']
    bce = torch.nn.functional.binary_cross_entropy_with_logits(out['target_logits'][ok], b['head_y'][ok])
    assert torch.allclose(loss, base + 0.1 * bce)


# --- the data rules --------------------------------------------------------------------------------------------------
def test_target_status_probe_unseen_and_duplicates(tmp_path):
    write_v11_fixture(tmp_path)
    unseen_p11(tmp_path)
    D = v11_data(tmp_path)
    uq = list(np.unique(D.pert))
    st = {p: d11.STATUS[D.status[uq.index(p)]] for p in uq}
    assert st['p0'] == st['p1'] == st['p2'] == st['p3'] == st['p4'] == st['p5'] == 'known'
    assert st['p6'] == 'probe'
    assert st['p11'] == 'unseen_annotated' and st['p7'] == 'unseen_annotated'      # p7 shares p11's molecule
    assert st['p8'] == st['p9'] == st['p10'] == 'unannotated'
    nodes = list(np.load(tmp_path / 'string_graph_v11_expdb.npz')['nodes'])
    assert nodes.index('X11') not in set(D.tgt_node[uq.index('p6')].tolist())      # probe targets never enter
    assert D.tgt_count[uq.index('p6')] == 0
    vocab = {nodes[i] for i in D.vocab}
    assert vocab == {'L3', 'X5', 'X7', 'L9', 'X9', 'L20'}                          # known compounds' targets only
    assert np.isclose(D.vocab_prior[list(D.vocab).index(nodes.index('L3'))], 2 / 6)


def test_arms_switch_only_their_tables(tmp_path):
    write_v11_fixture(tmp_path)
    full, noc, noe, not_ = (v11_data(tmp_path, arm=a) for a in ('full', 'no_chrom', 'no_ccle', 'no_targets'))
    assert not np.array_equal(full.fg_F[0], full.fg_F[2])
    assert np.array_equal(noc.fg_F[0], noc.fg_F[2]) and np.array_equal(noc.fg_M[1], noc.fg_M[0])
    w = np.bincount(full.cell_row[full.tr], minlength=8) / len(full.tr)
    assert np.allclose(noc.fg_F[0].astype(np.float64), np.tensordot(w, full.fg_F.astype(np.float64), 1), atol=2e-3)
    assert np.array_equal(noc.fg_E, full.fg_E) and np.array_equal(noe.fg_F, full.fg_F)
    assert float(np.abs(noe.fg_E).max()) == 0 and float(noe.fg_Em.max()) == 0 and float(full.fg_Em.max()) == 1
    assert np.array_equal(not_.fg_F, full.fg_F) and np.array_equal(not_.fg_E, full.fg_E)
    assert not not_.known_input.any() and full.known_input.any()


def test_ccle_ranks_sym_norm_and_signs_by_hand():
    E, m = d11.ccle_ranks(np.array([[3.0, np.nan, 1.0, 2.0], [np.nan] * 4]))
    assert np.allclose(E[0], [1.0, 0.0, 0.0, 0.5]) and m[0].tolist() == [True, False, True, True] and not m[1].any()
    A = d11.sym_norm(np.array([[0, 0], [1, 2]]), np.array([1.0, 4.0]), 4).toarray()
    deg = np.array([5.0, 1.0, 4.0, 0.0])
    assert np.isclose(A[0, 2], 4 / np.sqrt(5 * 4)) and np.isclose(A[1, 0], 1 / np.sqrt(5)) and A[3].sum() == 0
    assert np.allclose(A, A.T) and deg[3] == 0
    assert d11.combine_signs([1, 1]) == 1 and d11.combine_signs([-1]) == -1
    assert d11.combine_signs([1, -1]) == 0 and d11.combine_signs([0, -1]) == 0


def test_ccle_cells_without_data_are_masked(tmp_path):
    cells, _, _ = write_v11_fixture(tmp_path)
    D = v11_data(tmp_path)
    assert D.fg_Em[5:].max() == 0 and D.fg_Em[:5, 7:].min() == 1 and D.fg_Em[:5, :7].max() == 0


def test_device_tables_are_shared_by_dataparallel_replicas(tmp_path):
    write_v11_fixture(tmp_path)
    D, _, v11 = built_pair(tmp_path)
    fg_rep = v11.fg._replicate_for_data_parallel()                # what nn.parallel.replicate does per module
    assert fg_rep.tables is v11.fg.tables
    names = [n for n, _ in v11.named_buffers()]
    assert not any(n.startswith('fg.') for n in names)


# --- the trainer -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('arm', d11.ARMS)
def test_end_to_end_cpu_training_each_arm(tmp_path, monkeypatch, arm):
    write_v11_fixture(tmp_path)
    unseen_p11(tmp_path)
    monkeypatch.setattr(t11, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    rec = t11.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--epochs', '1',
                    '--batch', '16', '--d_model', '32', '--arm', arm, '--guard'])
    s = rec['scores']
    assert s['n_rows'] == 120 and np.isfinite(s['contrast_r_row_mean'])
    z = np.load(tmp_path / 'o' / f"v11_pred_{rec['tag']}.npz")
    assert z['delta'].shape == (120, G) and len(z['target_status']) == 120
    assert (tmp_path / 'o' / f"V11_COMPLETE_{rec['tag']}.json").exists()
    perms = rec['reported_permutations']
    assert {'ccle_cell_swap', 'chrom_cell_swap', 'chrom_gene_shuffle'} <= set(perms)
    assert ('targets_shuffled' in perms) == (arm != 'no_targets')
    ks = rec['reported_known_target_setting']
    assert ks['n_rows'] > 0
    assert ('contrast_r_unseen_annotated_known' in ks) == (arm != 'no_targets')
    assert ('target_bce' in json.dumps(rec['hist'])) and (rec['hist'][0]['target_bce_mean'] > 0) == (arm != 'no_targets')


def test_permutations_restore_the_tables(tmp_path, monkeypatch):
    write_v11_fixture(tmp_path)
    monkeypatch.setattr(t11, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    seen = {}
    real = t11.fg_permutations

    def spy(core, D, device, T, seed):
        before = {k: v.clone() for k, v in core.fg.tables.cpu.items()}
        tn = D.tgt_node.copy()
        out = real(core, D, device, T, seed)
        seen['same'] = all(torch.equal(before[k], core.fg.tables.cpu[k]) for k in before)
        seen['tgt'] = np.array_equal(tn, D.tgt_node)
        return out
    monkeypatch.setattr(t11, 'fg_permutations', spy)
    t11.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--epochs', '1', '--batch', '16',
              '--d_model', '32'])
    assert seen == {'same': True, 'tgt': True}


def test_timing_mode_records_cells_per_batch(tmp_path, monkeypatch):
    write_v11_fixture(tmp_path)
    monkeypatch.setattr(t11, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    rec = t11.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--timing_steps', '3',
                    '--batch', '16', '--d_model', '32'])
    assert rec['s_per_step'] > 0 and rec['cells_per_batch']['max'] >= 1 and (tmp_path / 'o' / 'timing.json').exists()


def test_guard_stops_a_dead_run_with_the_guard_exit(tmp_path, monkeypatch):
    write_v11_fixture(tmp_path)
    monkeypatch.setattr(t11, 'V10CConfig', lambda **kw: tiny_cfg(**kw))
    monkeypatch.setattr(t11, 'DrugPathGuard', lambda: tv.DrugPathGuard(start=2, every=1, window=2, floor=2.0))
    with pytest.raises(SystemExit) as e:
        t11.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o'), '--cpu_ok', '--epochs', '1', '--batch', '16',
                  '--d_model', '32', '--guard'])
    assert e.value.code == tv.GUARD_EXIT
    assert any(f.startswith('V11_FAILED_') for f in os.listdir(tmp_path / 'o'))


def test_refuses_cpu_without_flag(tmp_path):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        pytest.skip('a GPU is present')
    with pytest.raises(SystemExit, match='no CUDA'):
        t11.main(['--roots', str(tmp_path), '--out', str(tmp_path / 'o')])


# --- the kernels -----------------------------------------------------------------------------------------------------
def _mk():
    sys.path.insert(0, os.path.join(REPO, 'orchestration'))
    import make_v11_kernels as k11
    return k11


def test_kernel_builder_pins_compile_and_layout(tmp_path):
    import py_compile
    k11 = _mk()
    mk = k11.mk
    before = (mk.TEMPLATE, mk.CODE, mk.DATASETS)
    pins = k11.build_src(str(tmp_path / 'src'))
    assert set(pins) == set(k11.CODE) | set(k11.SMALL) | {mk.SUBSTRATE[0]} | set(mk.FROM_V9)
    for n in list(k11.CODE) + list(k11.SMALL):
        assert mk.sha1(str(tmp_path / 'src' / n)) == pins[n]
    # the base code is the bytes T-C pins (T-B off's computation, test_pinned_base_files_are_tb_offs)
    assert pins['data_v10c.py'] == '6efd1b4a2d545090616b05b81c416a3a7ad282bd'
    assert pins['model_v10c.py'] == '26a2d5dbe248a9643c6f7d08867515a3cffae0dd'
    assert pins['gene_vectors_v11_full.npy'] == mk.sha1(os.path.join(REPO, 'network/outputs/v11/gene_vectors_full.npy'))
    d = k11.render('lincs-v11-timing', [['t', ['--dev', '--arm', 'full']]], pins, kern_dir=str(tmp_path / 'k'))
    assert (mk.TEMPLATE, mk.CODE, mk.DATASETS) == before
    py = os.path.join(d, 'lincs-v11-timing.py')
    py_compile.compile(py, doraise=True)
    code = open(py, encoding='utf-8').read()
    assert "'model/v11/train_v11.py'" in code and "'V11_KERNEL_COMPLETE.json'" in code and 'train_v10c.py\')' not in code
    assert repr(pins) in code and repr(k11.CODE) in code and 'RUN_H = 4.5' in code
    meta = json.load(open(os.path.join(d, 'kernel-metadata.json')))
    assert meta['dataset_sources'] == k11.DATASETS and 'apexblue/lincs-v10c-src' not in meta['dataset_sources']
    assert meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape'] == 'NvidiaTeslaT4'


STUB_V11 = r'''import json, os, sys
a = sys.argv
seed, out = a[a.index('--seed') + 1], a[a.index('--out') + 1]
os.makedirs(out, exist_ok=True)
if seed in os.environ.get('STUB_FAIL_SEEDS', '').split(','):
    sys.exit(3)
json.dump({'seed': seed, 'arm': a[a.index('--arm') + 1]}, open(os.path.join(out, f'V11_COMPLETE_s{seed}.json'), 'w'))
'''


def test_executed_kernel_runs_train_v11_and_retries_a_guard_failure(tmp_path):
    import subprocess
    k11 = _mk()
    mk = k11.mk
    inp, work = tmp_path / 'input' / 'ds', tmp_path / 'working'
    inp.mkdir(parents=True)
    work.mkdir()
    pins = {}
    for n in list(k11.CODE) + list(k11.SMALL) + [mk.SUBSTRATE[0]] + list(mk.FROM_V9):
        (inp / n).write_text(STUB_V11 if n == 'train_v11.py' else f'stub {n}', encoding='utf-8')
        pins[n] = hashlib.sha1((inp / n).read_bytes()).hexdigest()
    d = k11.render('lincs-v11-x', [['v11_full_s0', ['--dev', '--seed', '0', '--arm', 'full', '--guard']]], pins,
                   kern_dir=str(tmp_path / 'k'), run_h=1.0)
    txt = open(os.path.join(d, 'lincs-v11-x.py'), encoding='utf-8').read()
    probe = "'import json, torch; print(json.dumps([torch.cuda.get_device_name(i) '"
    txt = (txt.replace(probe, "'import json; print(json.dumps([\"Tesla T4\", \"Tesla T4\"])) #'")
           .replace('os.symlink(', 'shutil.copyfile(')
           .replace("'/kaggle/input/**/'", repr(str(tmp_path / 'input').replace('\\', '/') + '/**/'))
           .replace("'/kaggle/working/src', '/kaggle/working/data', '/kaggle/working'",
                    ', '.join(repr(str(p).replace('\\', '/')) for p in (work / 'src', work / 'data', work))))
    (tmp_path / 'kernel.py').write_text(txt, encoding='utf-8')
    r = subprocess.run([sys.executable, str(tmp_path / 'kernel.py')], capture_output=True, text=True,
                       env=dict(os.environ, STUB_FAIL_SEEDS='0'), cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.load(open(work / 'V11_KERNEL_COMPLETE.json'))
    att = rec['attempts']['v11_full_s0']
    assert [x['returncode'] for x in att] == [3, 0] and att[1]['args'][att[1]['args'].index('--seed') + 1] == '10'
    assert (work / 'v11_full_s0_retry1' / 'V11_COMPLETE_s10.json').exists() and rec['done'] == ['v11_full_s0']
    assert (work / 'src' / 'model' / 'v10c' / 'model_v10c.py').exists()
