# -*- coding: utf-8 -*-
"""Tests for xpert_arm.encode_chromatin (RESULTS 92 E1 'clean', 92.11 E3 'tie'). The 'v9' and 'clean' paths must be the SAME
operations as the inline code they were lifted from (copied verbatim below as the reference); 'tie' must tie each cell's
no-peak block, mask channels with fewer than 10 non-tied genes, and otherwise equal 'clean'."""
import json
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xpert_arm as xa  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _old_v9(E, Em):                               # verbatim from xpert_arm.py before 92.11
    for k in range(E.shape[2]):
        has = Em[:, :, k].any(1)
        for c in np.where(has)[0]:
            v = E[c, :, k]
            E[c, :, k] = (v - v.mean()) / (v.std() + 1e-6)
    return E, Em


def _old_clean(E, Em, cidx, failed):              # verbatim from xpert_arm.py before 92.11
    Em = Em.copy()
    for cname, j in cidx.items():
        if cname in failed:
            Em[j, :, 2] = False
            E[j, :, 2] = 0.0
    for k in range(E.shape[2]):
        for c in np.where(Em[:, :, k].any(1))[0]:
            E[c, :, k] = norm.ppf((rankdata(E[c, :, k], method='average') - 0.5) / E.shape[1]).astype(np.float32)
    return E, Em


def _world(seed=0, C=6, G=40):
    rng = np.random.default_rng(seed)
    E = rng.uniform(0.3, 1.0, size=(C, G, 3)).astype(np.float32)        # all above any tie threshold used below
    Em = np.ones((C, G, 3), dtype=bool)
    Em[4, :, 1] = False                                                  # one absent (cell, mark)
    cidx = {'C%d' % i: i for i in range(C)}
    return E, Em, cidx


def test_v9_and_clean_are_the_lifted_code_exactly():
    E, Em, cidx = _world(1)
    a, am, _ = xa.encode_chromatin(E.copy(), Em.copy(), cidx, set(), 'v9')
    b, bm = _old_v9(E.copy(), Em.copy())
    assert np.array_equal(a, b) and np.array_equal(am, bm)
    failed = {'C2', 'C5'}
    a, am, info = xa.encode_chromatin(E.copy(), Em.copy(), cidx, failed, 'clean')
    b, bm = _old_clean(E.copy(), Em.copy(), cidx, failed)
    assert np.array_equal(a, b) and np.array_equal(am, bm) and info['failed'] == ['C2', 'C5']


def _tie_world():
    """Values ~ global ranks / (N - 1) as in E_final; each (cell, mark) gets its own tie block at the bottom."""
    E, Em, cidx = _world(2)
    thr_val = 0.05
    E[0, :12, 0] = np.linspace(0.0, thr_val, 12)            # C0 ATAC: 12 tied entries, distinct tie-break codes
    E[1, :35, 0] = np.linspace(0.0, thr_val, 35)            # C1 ATAC: 35 tied -> 5 non-tied < 10 -> missing
    E[3, :20, 2] = np.linspace(0.0, thr_val, 20)            # C3 H3K27me3: 20 tied
    tie = {}
    for k in range(3):
        v = E[:, :, k][Em[:, :, k]]
        N = int(Em[:, :, k].sum())
        Z = int((v <= thr_val).sum())
        tie[k] = (Z, N)
    # make (Z - 0.5) / (N - 1) land between thr_val and the smallest non-tied value (0.3): rescale the tied entries
    for k, (Z, N) in tie.items():
        hi = (Z - 0.5) / (N - 1) if Z else 0.0
        for c in range(E.shape[0]):
            m = (E[c, :, k] <= thr_val) & Em[c, :, k]
            E[c, m, k] = E[c, m, k] * (hi / thr_val) * 0.99 if hi else E[c, m, k]
    return E, Em, cidx, tie


def test_tie_ties_the_block_masks_thin_channels_and_otherwise_equals_clean():
    E, Em, cidx, tie = _tie_world()
    out, om, info = xa.encode_chromatin(E.copy(), Em.copy(), cidx, set(), 'tie', tie=tie)
    # C0 ATAC: its 12 tie-block entries now share ONE value (the tie-break order is gone), below every non-tied entry
    blk = out[0, :12, 0]
    assert np.allclose(blk, blk[0]) and blk[0] < out[0, 12:, 0].min()
    # the clean encoding keeps their distinct codes
    cl, _, _ = xa.encode_chromatin(E.copy(), Em.copy(), cidx, set(), 'clean')
    assert len(np.unique(np.round(cl[0, :12, 0], 6))) == 12
    # non-tied genes keep their order
    assert np.array_equal(np.argsort(out[0, 12:, 0], kind='stable'), np.argsort(cl[0, 12:, 0], kind='stable'))
    # C1 ATAC (5 non-tied genes) becomes missing, mask and values
    assert info['tie_masked'] == [('C1', 0)] and not om[1, :, 0].any() and not out[1, :, 0].any()
    # channels without any tied entry are identical to 'clean'
    assert np.array_equal(out[2], cl[2]) and np.array_equal(out[5], cl[5])


def test_tie_refuses_block_sizes_that_do_not_describe_E():
    E, Em, cidx, tie = _tie_world()
    bad = dict(tie)
    bad[0] = (tie[0][0] + 1, tie[0][1])
    with pytest.raises(SystemExit, match='does not describe this E_final'):
        xa.encode_chromatin(E.copy(), Em.copy(), cidx, set(), 'tie', tie=bad)


def test_unknown_encoding_raises():
    E, Em, cidx = _world()
    with pytest.raises(ValueError):
        xa.encode_chromatin(E, Em, cidx, set(), 'other')


DATA = os.path.join(REPO, 'external', 'kaggle_chromatin_src')


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, 'E_final.npy')), reason='local E_final not present')
def test_real_E_final_masks_exactly_the_registered_channels():
    """Identity level (no response): on the shipped E_final, the pinned blocks hold and 059 C8 masks HME1 and SKBR3 ATAC."""
    E = np.load(os.path.join(DATA, 'E_final.npy')).astype(np.float32)
    Em = np.load(os.path.join(DATA, 'E_final_mask.npy'))
    cidx = json.load(open(os.path.join(DATA, 'lincs_cell_index.json')))
    cidx = cidx.get('cell_id_to_row', cidx)
    failed = set(json.load(open(os.path.join(DATA, 'E_final_provenance.json')))['h3k27me3_failed_chip_downweighted'])
    _, _, info = xa.encode_chromatin(E, Em, cidx, failed, 'tie')
    assert info['tie_masked'] == xa.TIE_EXPECTED_MASKED


def test_clean_ranks_ties_by_average_like_the_lifted_code():
    """Exact duplicates inside a channel: average ranks (the lifted code), not min ranks."""
    E, Em, cidx = _world(3)
    E[2, :10, 0] = 0.5                                  # ten genes share one value
    a, _, _ = xa.encode_chromatin(E.copy(), Em.copy(), cidx, set(), 'clean')
    b, _ = _old_clean(E.copy(), Em.copy(), cidx, set())
    assert np.array_equal(a, b)


def test_the_tie_threshold_keeps_the_first_non_tied_rank_out_of_the_block():
    """E_final-like values: global ranks / (N - 1) over all present entries; ranks < Z are the tie block. The entry of rank
    exactly Z sits at Z / (N - 1), just above the (Z - 0.5) / (N - 1) threshold, and must NOT join its cell's tie."""
    C, G = 4, 30
    rng = np.random.default_rng(7)
    N = C * G
    Z = 30
    ranks = rng.permutation(N).reshape(C, G)
    E = np.zeros((C, G, 3), np.float32)
    E[:, :, 0] = ranks / (N - 1)
    E[:, :, 1:] = rng.uniform(0.3, 1.0, size=(C, G, 2))
    Em = np.ones((C, G, 3), dtype=bool)
    cidx = {'C%d' % i: i for i in range(C)}
    c, g = map(int, np.argwhere(ranks == Z)[0])
    assert int((ranks[c] < Z).sum()) >= 1 and int((ranks[c] >= Z).sum()) >= 10
    out, _, _ = xa.encode_chromatin(E.copy(), Em.copy(), cidx, set(), 'tie', tie={0: (Z, N)})
    tied_val = out[c, ranks[c] < Z, 0]
    assert np.allclose(tied_val, tied_val[0])
    assert out[c, g, 0] > tied_val[0] + 1e-6


LOG = os.path.join(REPO, 'external', 'kaggle_c93_inputs', 'E_peaks_log.txt')


@pytest.mark.skipif(not (os.path.exists(LOG) and os.path.exists(os.path.join(DATA, 'E_final.npy'))), reason='log not present')
def test_tie_blocks_match_the_peak_log_channel_by_channel():
    """Review 070 C2: the guard can only check N (E_final's covered values are an exact rank grid, so #<=thr == Z for any Z).
    What validates Z: for every present channel the log covers, the non-tied count equals the log's nonzero_genes."""
    import re as _re
    E = np.load(os.path.join(DATA, 'E_final.npy')).astype(np.float32)
    Em = np.load(os.path.join(DATA, 'E_final_mask.npy'))
    cidx = json.load(open(os.path.join(DATA, 'lincs_cell_index.json')))
    cidx = cidx.get('cell_id_to_row', cidx)
    marks = {'ATAC-seq': 0, 'H3K27ac': 1, 'H3K27me3': 2}
    n_checked, bad = 0, []
    for line in open(LOG, encoding='utf-8'):
        m = _re.match(r'(\S+)/(ATAC-seq|H3K27ac|H3K27me3) .*nonzero_genes=(\d+)', line)
        if not m or m.group(1) not in cidx:
            continue
        j, k, nz = cidx[m.group(1)], marks[m.group(2)], int(m.group(3))
        if not Em[j, :, k].any():
            continue
        Z, N = xa.TIE[k]
        nontied = int(((E[j, :, k] > (Z - 0.5) / (N - 1)) & Em[j, :, k]).sum())
        n_checked += 1
        if nontied != nz:
            bad.append((m.group(1), m.group(2), nontied, nz))
    assert not bad and n_checked == 86, (n_checked, bad[:5])
