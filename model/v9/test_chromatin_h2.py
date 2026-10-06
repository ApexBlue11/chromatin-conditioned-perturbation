# -*- coding: utf-8 -*-
"""Tests for chromatin_h2.py (RESULTS 93.2 item 1 / 93.5 C4). Synthetic worlds only (test_chromatin_funnel.world)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_funnel as cf  # noqa: E402
import chromatin_h2 as h2  # noqa: E402
from test_chromatin_funnel import world  # noqa: E402


def test_subsets_are_seeded_sized_and_drawn_from_the_fit_cells():
    fit = ['T%02d' % i for i in range(11)]
    a, b = h2.subsets(fit), h2.subsets(list(reversed(fit)))
    assert a == b                                            # order of the input does not matter; seeds fix the draw
    for k in h2.KS:
        assert len(a[str(k)]) == h2.N_SUB and all(len(s) == k and set(s) <= set(fit) for s in a[str(k)])
    assert a['all'] == [sorted(fit)]


def test_a_cell_specific_planted_gain_shows_in_FBC_minus_N1_and_a_subset_fit_uses_only_its_cells():
    ctx = world(plant='gain', alpha=0.4, chrom_mark=1)
    fit = ctx.enc['rank_normal']['cov_dt'][:4]
    B = {kind: cf.feature_builder(ctx, 'rank_normal', kind, fit) for kind in ('FB', 'FBC', 'N1')}
    d = h2.t1_deltas(ctx, ctx.y, B, fit)
    assert d['FBC_minus_N1'] > 0.005 and d['FBC_minus_FB'] > 0.005
    ctx0 = world(seed=3)                                     # no planted effect: nothing cell-specific to find
    B0 = {kind: cf.feature_builder(ctx0, 'rank_normal', kind, fit) for kind in ('FB', 'FBC', 'N1')}
    assert abs(h2.t1_deltas(ctx0, ctx0.y, B0, fit)['FBC_minus_N1']) < 0.003
