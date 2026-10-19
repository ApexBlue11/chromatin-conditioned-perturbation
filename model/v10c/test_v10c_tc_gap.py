"""Unit tests for tc_gap.py (the end-to-end reproduction of v9's references is tc_gap_repro.py, 100.11)."""
import os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tc_gap  # noqa: E402


def test_align_reorders_to_the_reference_and_refuses_other_rows(tmp_path):
    p = tmp_path / 'pred.npz'
    np.savez(p, bench_row=np.array([30, 10, 20]), sub_row=np.array([3, 1, 2]), delta=np.arange(6.).reshape(3, 2))
    d, sub = tc_gap.align(str(p), [10, 20, 30])
    assert np.array_equal(sub, [1, 2, 3]) and np.array_equal(d, [[2., 3.], [4., 5.], [0., 1.]])
    with pytest.raises(SystemExit, match='rows differ'):
        tc_gap.align(str(p), [10, 20, 31])
    with pytest.raises(SystemExit, match='rows differ'):
        tc_gap.align(str(p), [10, 20])


def test_gap_is_compound_minus_cell_per_target():
    cell = {'m': {'contrast': {'clean7': {'row_mean': 0.16}, 'all8_labelled': {'row_mean': 0.17}},
                  'delta_r': {'clean7': {'row_mean': 0.48}, 'all8_labelled': {'row_mean': 0.49}}}}
    comp = {'m': {'contrast_clean': 0.11, 'delta_r_clean': 0.64}}
    g = tc_gap.gap(cell, comp, 'm', 'm')
    assert g['clean7']['contrast'] == pytest.approx(-0.05) and g['all8_labelled']['delta_r'] == pytest.approx(0.15)


def test_summarise_primary_is_the_seven_clean_cells():
    r = np.array([0.1, 0.3, 0.9, np.nan])
    cell = np.array(['MCF7', 'HT29', 'H1975', 'MCF7'])
    s7 = tc_gap.summarise(r, cell, ('MCF7', 'HT29'))
    s8 = tc_gap.summarise(r, cell, ('MCF7', 'HT29', 'H1975'))
    assert s7['n_rows'] == 2 and s7['row_mean'] == pytest.approx(0.2) and s8['row_mean'] == pytest.approx(13 / 30)
    assert s7['per_cell'] == {'MCF7': pytest.approx(0.1), 'HT29': pytest.approx(0.3)}
