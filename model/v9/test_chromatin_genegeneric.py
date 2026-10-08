# -*- coding: utf-8 -*-
"""Tests for chromatin_genegeneric.py (RESULTS 93.15 / 93.15a / 93.16). Synthetic worlds only."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402
import chromatin_genegeneric as gg  # noqa: E402
from test_chromatin_funnel import world  # noqa: E402


def _fit(ctx):
    return list(ctx.enc['rank_normal']['cov_dt'])


def _covered(ctx):
    e = ctx.enc['rank_normal']
    return next(c for i, c in enumerate(ctx.cells) if e['has'][i].all())


def test_the_N1_kind_is_exactly_section_91s_N1():
    ctx = world(seed=0, n_perts=4)
    fit = _fit(ctx)
    a, b = gg.variant(ctx, 'rank_normal', fit, 'N1'), cf.feature_builder(ctx, 'rank_normal', 'N1', fit)
    for h in (None, fit[0]):
        A, B = a(h), b(h)
        assert all(np.array_equal(A[c], B[c]) for c in ctx.cells)


def test_one_permutation_per_draw_is_shared_across_marks():
    ctx = world(seed=1, n_perts=4)
    fit = _fit(ctx)
    c = _covered(ctx)
    n1 = gg.variant(ctx, 'rank_normal', fit, 'N1')(fit[0])[c]
    p0, p0b, p1 = (gg.variant(ctx, 'rank_normal', fit, 'perm', d)(fit[0])[c] for d in (0, 0, 1))
    perm = np.random.default_rng(9500).permutation(cf.G)
    assert np.array_equal(p0[:, 1:], n1[perm, 1:])                       # the same index permutation for all three marks
    assert np.array_equal(p0, p0b) and not np.array_equal(p0, p1)
    assert np.array_equal(p0[:, 0], n1[:, 0])                            # b untouched


def test_E_columns_are_three_expression_stats_quantile_matched_to_the_marks_and_E_plus_N1_appends_the_marks():
    ctx = world(seed=2, n_perts=4)
    fit = _fit(ctx)
    c = _covered(ctx)
    src = [cf.cell_index(ctx, x) for x in fit if x != fit[0]]
    bs = ctx.enc['rank_normal']['b'][src].astype(np.float64)
    stats = (bs.mean(0), bs.std(0), bs.mean(0) ** 2)
    n1 = gg.variant(ctx, 'rank_normal', fit, 'N1')(fit[0])[c]
    Ef = gg.variant(ctx, 'rank_normal', fit, 'E')(fit[0])[c]
    EN = gg.variant(ctx, 'rank_normal', fit, 'E+N1')(fit[0])[c]
    EP = gg.variant(ctx, 'rank_normal', fit, 'E+perm', 3)(fit[0])[c]
    assert Ef.shape[1] == 4 and EN.shape[1] == 7 and EP.shape[1] == 7
    for k in range(3):
        assert np.array_equal(np.sort(Ef[:, 1 + k]), np.sort(n1[:, 1 + k]))                 # marginal of mark k's mean
        assert np.all(np.diff(Ef[np.argsort(stats[k], kind='stable'), 1 + k]) >= 0)          # ordering of stat k
    assert np.array_equal(EN[:, :4], Ef) and np.array_equal(EN[:, 4:], n1[:, 1:])
    assert np.array_equal(EP[:, :4], Ef) and not np.array_equal(EP[:, 4:], n1[:, 1:])


def test_tie_encoding_ties_the_no_peak_block_and_only_it(monkeypatch):
    ctx = world(seed=3, n_perts=4)
    n = len(ctx.cells)
    rng = np.random.default_rng(0)
    E = rng.random((n, cf.G, 3)).astype(np.float32)
    monkeypatch.setattr(gg, 'TIE', {0: (int(0.3 * 1000), 1001)})        # block = values <= (300 - 0.5) / 1000
    cidx = {c: i for i, c in enumerate(ctx.cells)}
    t = gg.install_tie(ctx, E, cidx)
    i = ctx.cells.index(_covered(ctx))
    low = E[i, :, 0] <= (300 - 0.5) / 1000
    assert low.any() and (~low).any()
    assert len(np.unique(t['Ez'][i, low, 0])) == 1                                         # tied
    assert len(np.unique(t['Ez'][i, ~low, 0])) == int((~low).sum())                         # the rest keep their order
    np.testing.assert_allclose(t['Ez'][i, :, 1], cf.rank_normal(E[i, :, 1].astype(np.float64)), atol=1e-6)   # untouched mark
    assert t['b'] is ctx.enc['rank_normal']['b'] and np.array_equal(t['has'], ctx.enc['rank_normal']['has'])


def _res(g_chr, g_perm_max, x_chr, x_perm_max, x_pos=6, v9_g=0.0012):
    """Fabricated scores: the 'tie' set carries the case; the 'v9' set carries the harness value and a capacity-null case."""
    cells = list(cf.DEV_CELLS)
    flat = lambda v: {'all': v, 'per_cell': {c: v for c in cells}}      # noqa: E731
    s = {'FB': flat(0.10)}
    for f, (gc, gp, xc, xp) in (('tie', (g_chr, g_perm_max, x_chr, x_perm_max)), ('v9', (v9_g, 0.01, 0.0, 0.01))):
        s['N1_' + f], s['E_' + f], s['E_%s+N1' % f] = flat(0.10 + gc), flat(0.20), flat(0.20 + xc)
        for j, c in enumerate(cells):
            if f == 'tie' and j >= x_pos:
                s['E_%s+N1' % f]['per_cell'][c] = 0.20 - 0.001
        for d in range(20):
            s['N1perm_%s_%d' % (f, d)] = flat(0.10 + gp * d / 19)
            s['E_%s+perm_%d' % (f, d)] = flat(0.20 + xp * d / 19)
    return {'scores': s, 'n_perm': 20, 'dev_cells': cells}


def test_the_reading_of_record_is_on_the_tie_corrected_features_with_the_v9_harness():
    r = lambda *a, **k: gg.reading(_res(*a, **k), 0.0012)['reading']   # noqa: E731
    assert r(0.0012, 0.0015, 0.001, 0.0) == 'NOT DISTINGUISHABLE FROM CAPACITY'
    assert r(0.0012, 0.0005, 0.0003, 0.0004) == 'GENE-LEVEL, MATCHED BY EXPRESSION (2a)'
    assert r(0.0012, 0.0005, 0.0008, 0.0004, x_pos=3) == 'GENE-LEVEL, EXCESS NOT ESTABLISHED ACROSS CELLS (2b)'
    assert r(0.0012, 0.0005, 0.0008, 0.0004, x_pos=4) == 'CHROMATIN-SPECIFIC'
    out = gg.reading(_res(0.0012, 0.0005, 0.0008, 0.0004), 0.0012)
    assert out['reported_v9_features']['reading'] == 'NOT DISTINGUISHABLE FROM CAPACITY' and 'tie-break' in out['reported_v9_features']['artefact']
    assert gg.reading(_res(0.0030, 0.0005, 0.0008, 0.0004, v9_g=0.0012), 0.0099)['reading'] == 'HARNESS_FAULT'   # harness is v9's N1
    assert gg.reading(_res(0.0030, 0.0005, 0.0008, 0.0004, v9_g=0.0012), 0.0012)['reading'] != 'HARNESS_FAULT'   # not the tie N1


def test_run_on_a_world_and_the_marker_refusal(tmp_path):
    ctx = world(seed=4, n_perts=4)
    rng = np.random.default_rng(1)
    gg.install_tie(ctx, rng.random((len(ctx.cells), cf.G, 3)).astype(np.float32), {c: i for i, c in enumerate(ctx.cells)})
    res = gg.run(ctx, ctx.y, n_perm=2)
    want = {'FB'} | {k % f for f in ('tie', 'v9') for k in ('N1_%s', 'E_%s', 'E_%s+N1', 'N1perm_%s_0', 'N1perm_%s_1', 'E_%s+perm_0', 'E_%s+perm_1')}
    assert set(res['scores']) == want
    v9n1, cfn1 = res['scores']['N1_v9']['all'], cf.score(ctx, ctx.y, cf.run_t1(ctx, ctx.y, {'N1': (cf.feature_builder(
        ctx, 'rank_normal', 'N1', list(ctx.enc['rank_normal']['cov_dt'])), ['full'])}, list(ctx.enc['rank_normal']['cov_dt']))[0][
        ('N1', 'full')]['y_hat_dev'], cf.run_t1(ctx, ctx.y, {}, list(ctx.enc['rank_normal']['cov_dt']))[2][ctx.dev_mask] <= 2)['all']
    assert abs(v9n1 - cfn1) < 1e-9                                           # N1_v9 is §91's N1, scored the same way
    p = tmp_path / gg.OUT
    p.write_text(json.dumps(cf.jsonable(res)))
    (tmp_path / gg.MARKER).write_text(json.dumps({'complete': True, 'outputs': {gg.OUT: '0' * 40}}))
    with pytest.raises(SystemExit):
        gg.read(str(tmp_path))
    (tmp_path / gg.MARKER).write_text(json.dumps({'complete': True, 'outputs': {gg.OUT: hashlib.sha1(p.read_bytes()).hexdigest()}}))
    assert gg.read(str(tmp_path))['reading'] == 'HARNESS_FAULT'          # a synthetic world does not reproduce 91.12


def test_tie_encoding_covers_all_three_marks():
    assert set(gg.TIE) == {0, 1, 2}


def test_the_pinned_tie_blocks_describe_the_real_E_final():
    """93.16's constants are checked against the committed E_final (inputs only; no response is read)."""
    d = os.path.join(gg.REPO, 'phase2_assembly', 'outputs')
    if not os.path.exists(os.path.join(d, 'E_final.npy')):
        pytest.skip('E_final not present')
    E, Em = np.load(os.path.join(d, 'E_final.npy')), np.load(os.path.join(d, 'E_final_mask.npy'))
    for k, (Z, N) in gg.TIE.items():
        assert int(Em[:, :, k].sum()) == N and int((E[:, :, k][Em[:, :, k]] <= (Z - 0.5) / (N - 1)).sum()) == Z
