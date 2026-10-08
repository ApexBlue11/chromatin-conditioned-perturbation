# -*- coding: utf-8 -*-
"""Tests for chromatin_genegeneric.py (RESULTS 93.15). Synthetic worlds only."""
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


def test_the_N1_kind_is_exactly_section_91s_N1():
    ctx = world(seed=0, n_perts=4)
    fit = _fit(ctx)
    a, b = gg.n1_variant(ctx, 'rank_normal', fit, 'N1'), cf.feature_builder(ctx, 'rank_normal', 'N1', fit)
    for h in (None, fit[0]):
        A, B = a(h), b(h)
        assert all(np.array_equal(A[c], B[c]) for c in ctx.cells)


def test_perm_is_a_fixed_gene_permutation_of_the_mark_mean_and_expr_is_quantile_matched_to_mean_b():
    ctx = world(seed=1, n_perts=4)
    fit = _fit(ctx)
    n1, p0, p0b, p1, ex = (gg.n1_variant(ctx, 'rank_normal', fit, 'N1')(fit[0]),
                           gg.n1_variant(ctx, 'rank_normal', fit, 'perm', 0)(fit[0]),
                           gg.n1_variant(ctx, 'rank_normal', fit, 'perm', 0)(fit[0]),
                           gg.n1_variant(ctx, 'rank_normal', fit, 'perm', 1)(fit[0]),
                           gg.n1_variant(ctx, 'rank_normal', fit, 'expr')(fit[0]))
    e = ctx.enc['rank_normal']
    src = [cf.cell_index(ctx, c) for c in fit if c != fit[0]]
    mb = e['b'][src].mean(0)
    c = next(c for i, c in enumerate(ctx.cells) if e['has'][i].any())
    i = ctx.cells.index(c)
    for k in range(3):
        if not e['has'][i, k]:
            continue
        col = 1 + k
        assert np.array_equal(np.sort(n1[c][:, col]), np.sort(p0[c][:, col])) and not np.array_equal(n1[c][:, col], p0[c][:, col])
        assert np.array_equal(p0[c][:, col], p0b[c][:, col]) and not np.array_equal(p0[c][:, col], p1[c][:, col])
        assert np.array_equal(np.sort(ex[c][:, col]), np.sort(n1[c][:, col]))           # same marginal
        assert np.all(np.diff(ex[c][np.argsort(mb, kind='stable'), col]) >= 0)          # ordered by mean basal expression
    assert np.array_equal(n1[c][:, 0], p0[c][:, 0]) and np.array_equal(n1[c][:, 0], ex[c][:, 0])   # b untouched


def _res(chr_, expr, perm, cell_diff_pos=6):
    cells = list(cf.DEV_CELLS)
    fb = {'all': 0.10, 'per_cell': {c: 0.10 for c in cells}}
    mk = lambda g, pos=6: {'all': 0.10 + g, 'per_cell': {c: 0.10 + g for c in cells}}   # noqa: E731
    s = {'FB': fb, 'N1': mk(chr_), 'N1expr': mk(expr)}
    for j, c in enumerate(cells):                        # N1 - N1expr > 0 in cell_diff_pos cells only
        if j >= cell_diff_pos:
            s['N1expr']['per_cell'][c] = 0.10 + chr_ + 0.001
    for d, g in enumerate(perm):
        s['N1perm_%d' % d] = mk(g)
    return {'scores': s, 'n_perm': len(perm), 'dev_cells': cells}


def test_the_reading_and_its_harness_check():
    perm = [0.0001 * d for d in range(20)]                                   # max 0.0019
    assert gg.reading(_res(0.0012, 0.0, perm), 0.0012)['reading'] == 'NOT DISTINGUISHABLE FROM CAPACITY'
    assert gg.reading(_res(0.0030, 0.0035, perm), 0.0030)['reading'] == 'GENE-LEVEL, NOT CHROMATIN-SPECIFIC'
    assert gg.reading(_res(0.0030, 0.0010, perm, cell_diff_pos=3), 0.0030)['reading'] == 'GENE-LEVEL, NOT CHROMATIN-SPECIFIC'
    assert gg.reading(_res(0.0030, 0.0010, perm, cell_diff_pos=4), 0.0030)['reading'] == 'CHROMATIN-SPECIFIC'
    assert gg.reading(_res(0.0030, 0.0010, perm), 0.0012)['reading'] == 'HARNESS_FAULT'


def test_run_on_a_world_and_the_marker_refusal(tmp_path, monkeypatch):
    ctx = world(seed=2, n_perts=4)
    res = gg.run(ctx, ctx.y, n_perm=2)
    assert set(res['scores']) == {'FB', 'N1', 'N1expr', 'N1perm_0', 'N1perm_1'}
    p = tmp_path / gg.OUT
    p.write_text(json.dumps(cf.jsonable(res)))
    (tmp_path / gg.MARKER).write_text(json.dumps({'complete': True, 'outputs': {gg.OUT: '0' * 40}}))
    with pytest.raises(SystemExit):
        gg.read(str(tmp_path))
    (tmp_path / gg.MARKER).write_text(json.dumps({'complete': True, 'outputs': {gg.OUT: hashlib.sha1(p.read_bytes()).hexdigest()}}))
    assert gg.read(str(tmp_path))['reading'] == 'HARNESS_FAULT'          # a synthetic world does not reproduce 91.12
