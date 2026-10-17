"""Tests for the v10-C chromatin features (RESULTS 100.1 normalisation)."""
import os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chromatin_features as cf  # noqa: E402


def test_rank01_ties_nan_and_scale():
    r = cf.rank01(np.array([3.0, 1.0, np.nan, 1.0, 5.0]))
    assert np.isnan(r[2])
    assert r[1] == r[3] == pytest.approx(0.5 / 3) and r[0] == pytest.approx(2 / 3) and r[4] == 1.0
    assert np.isnan(cf.rank01(np.array([np.nan, 2.0]))).all()


def test_rank_is_log1p_invariant():
    v = np.random.default_rng(0).gamma(0.5, 2.0, size=50)
    assert np.array_equal(cf.rank01(v), cf.rank01(np.log1p(v)))


def _fixture(tmp, genes=('G1', 'G2', 'G3', 'G4'), entrez=('1', '2', '3', '4')):
    tss = tmp / 'tss.tsv'
    tss.write_text('symbol\tentrez\tchrom\ttss_hg38\tstrand\n' +
                   ''.join(f'{g}\t{e}\tchr1\t{1000 * (i + 1)}\t+\n' for i, (g, e) in enumerate(zip(genes, entrez))),
                   encoding='utf-8')
    w = np.array(['a', 'b'])
    s0 = np.array([[[1, 10], [2, 20], [3, 30], [4, 40]],      # T1: increasing
                   [[4, 1], [3, 2], [2, 3], [1, 4]]], float)   # T2
    s1 = np.array([[[1, 1], [1, 1], [9, 9], [np.nan, np.nan]]], float)   # T3: gene 4 missing (absent chromosome)
    np.savez(tmp / 's0.npz', srx=np.array(['T1', 'T2']), values=s0, genes=np.array(genes), windows=w)
    np.savez(tmp / 's1.npz', srx=np.array(['T3']), values=s1, genes=np.array(genes), windows=w)
    return tss, [str(tmp / 's0.npz'), str(tmp / 's1.npz')]


def test_build_features_mask_order_and_averaging(tmp_path):
    tss, paths = _fixture(tmp_path)
    model_entrez = ['3', '1', '4', '2', '99']          # another order, plus a gene with no TSS (MUC1-like)
    sel = {'C1': {'ACC': ['T1', 'T2'], 'PCG': ['T3']}, 'C2': {'ACT': ['T1']}}
    meth = {'cells': np.array(['C2']), 'genes': np.array(['G2', 'G1']), 'values': np.array([[0.25, 0.75]])}
    F, M, names, used = cf.build(paths, sel, ['C1', 'C2', 'C3'], str(tss), model_entrez, methyl=meth)
    assert F.shape == (3, 5, 7) and M.shape == (3, 5, 4)
    assert names == ['ACC:a', 'ACC:b', 'ACT:a', 'ACT:b', 'PCG:a', 'PCG:b', 'METH:tss_1kb']
    # C1 ACC = mean of T1 and T2 ranks: T1 window a ranks G1..G4 = 0, 1/3, 2/3, 1; T2 window a = 1, 2/3, 1/3, 0 -> 0.5
    g1 = model_entrez.index('1')
    assert F[0, g1, 0] == pytest.approx(0.5) and F[0, g1, 1] == 0.0
    assert M[0, :4, 0].all() and not M[0, 4, 0] and not M[0, :, 1].any()
    # C1 PCG from T3: G4 is NaN -> masked; G1, G2 tie at the bottom, G3 on top
    g4, g3 = model_entrez.index('4'), model_entrez.index('3')
    assert not M[0, g4, 2] and F[0, g4, 4] == 0 and F[0, g3, 4] == 1.0
    assert F[0, g1, 4] == pytest.approx(0.25)
    # methylation for C2 only, by symbol
    assert F[1, model_entrez.index('2'), 6] == pytest.approx(0.25) and M[1, model_entrez.index('1'), 3]
    assert not M[1, g3, 3] and not M[0, :, 3].any()
    # C3 has nothing
    assert not M[2].any() and not F[2].any()
    assert used[('C1', 'ACC')] == ['T1', 'T2']


def test_build_refuses_missing_track_and_duplicate(tmp_path):
    tss, paths = _fixture(tmp_path)
    with pytest.raises(SystemExit, match='no signal'):
        cf.build(paths, {'C1': {'ACC': ['T9']}}, ['C1'], str(tss), ['1', '2', '3', '4'])
    with pytest.raises(SystemExit, match='two shards'):
        cf.build(paths + [paths[0]], {}, ['C1'], str(tss), ['1', '2', '3', '4'])


def test_build_refuses_unmapped_tss_gene(tmp_path):
    tss, paths = _fixture(tmp_path)
    with pytest.raises(SystemExit, match='without a model gene'):
        cf.build(paths, {}, ['C1'], str(tss), ['1', '2', '3'])
