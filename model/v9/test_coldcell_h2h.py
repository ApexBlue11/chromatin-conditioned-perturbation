import os
import json
import subprocess
import numpy as np
import pytest
import h5py

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'coldcell_h2h.py')
PYTHON_EXE = r"C:\Projects\LINCS\.venv-cuda\Scripts\python.exe"
os.environ["PYTHONPATH"] = r"C:\Projects\LINCS"

def test_regression_a(tmp_path):
    out_file = tmp_path / "out.json"
    cmd = [
        PYTHON_EXE, SCRIPT_PATH,
        "--theirs", r"C:\Projects\LINCS\external\kaggle_out\cc1_v9\xpert_trained_split_cold_cell_1_final_test_profile.npy",
        "--ours", r"C:\Projects\LINCS\external\v9_mdmt_preds\v9_cc1_epi_seed0.npz",
        "--h5ad", r"C:\Projects\LINCS\external\xpert\code\XPert\processed_data\l1000_mdmt_68830_subset.h5ad",
        "--split", "split_cold_cell_1",
        "--run_record", r"C:\Projects\LINCS\external\kaggle_out\cc1_v9\run_record.json",
        "--theirs_label", "XPert trained to its published recipe on split_cold_cell_1 (O2)",
        "--n_boot", "20000",
        "--seed", "0",
        "--out", str(out_file)
    ]
    subprocess.run(cmd, check=True)
    
    with open(r"C:\Projects\LINCS\model\results\coldcell_h2h_split_cold_cell_1_O2.json") as f:
        committed = json.load(f)
        
    with open(out_file) as f:
        new_out = json.load(f)
        
    for k, v in committed.items():
        assert k in new_out, f"Missing key {k}"
        if k == 'ours':
            assert new_out[k] == v
        else:
            assert new_out[k] == v, f"Mismatch at {k}"

def test_three_copies_b(tmp_path):
    out_file = tmp_path / "out_3.json"
    cmd = [
        PYTHON_EXE, SCRIPT_PATH,
        "--theirs", r"C:\Projects\LINCS\external\kaggle_out\cc1_v9\xpert_trained_split_cold_cell_1_final_test_profile.npy",
        "--ours", 
        r"C:\Projects\LINCS\external\v9_mdmt_preds\v9_cc1_epi_seed0.npz",
        r"C:\Projects\LINCS\external\v9_mdmt_preds\v9_cc1_epi_seed0.npz",
        r"C:\Projects\LINCS\external\v9_mdmt_preds\v9_cc1_epi_seed0.npz",
        "--h5ad", r"C:\Projects\LINCS\external\xpert\code\XPert\processed_data\l1000_mdmt_68830_subset.h5ad",
        "--split", "split_cold_cell_1",
        "--run_record", r"C:\Projects\LINCS\external\kaggle_out\cc1_v9\run_record.json",
        "--theirs_label", "XPert trained to its published recipe on split_cold_cell_1 (O2)",
        "--n_boot", "20000",
        "--seed", "0",
        "--out", str(out_file)
    ]
    subprocess.run(cmd, check=True)
    
    with open(out_file) as f:
        new_out = json.load(f)
        
    with open(r"C:\Projects\LINCS\model\results\coldcell_h2h_split_cold_cell_1_O2.json") as f:
        committed = json.load(f)
        
    for k in ('cluster', 'per_cell', 'row_pooled', 'verdict'):
        assert new_out[k] == committed[k]

def test_synthetic_c_d_e(tmp_path):
    # Create toy profiles
    row_idx = np.array([0, 1, 2])
    y_true = np.array([[2.0, 4.0], [5.0, 6.0], [1.0, 2.0]])
    ctl_true = np.zeros_like(y_true)
    
    t_pred = np.array([[1.0, 3.0], [4.0, 5.0], [0.5, 1.5]])
    o1_pred = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]])
    o2_pred = np.array([[2.0, 3.0], [1.0, 0.0], [1.5, 2.5]])
    
    t_file = tmp_path / "t.npy"
    o1_file = tmp_path / "o1.npz"
    o2_file = tmp_path / "o2.npz"
    h5ad_file = tmp_path / "toy.h5ad"
    
    np.save(t_file, {'row_index': row_idx, 'y_pred': t_pred, 'y_true': y_true, 'ctl_true': ctl_true})
    np.savez(o1_file, row_index=row_idx, y_pred=o1_pred, y_true=y_true, ctl_true=ctl_true)
    np.savez(o2_file, row_index=row_idx, y_pred=o2_pred, y_true=y_true, ctl_true=ctl_true)
    
    with h5py.File(h5ad_file, 'w') as f:
        obs = f.create_group('obs')
        obs.create_dataset('cell_iname', data=np.array(['C1', 'C1', 'C2'], dtype='S'))
        
    out_file = tmp_path / "toy_out.json"
    
    cmd = [
        PYTHON_EXE, SCRIPT_PATH,
        "--theirs", str(t_file),
        "--ours", str(o1_file), str(o2_file),
        "--h5ad", str(h5ad_file),
        "--split", "toy",
        "--theirs_label", "T",
        "--out", str(out_file)
    ]
    subprocess.run(cmd, check=True)
    
    with open(out_file) as f:
        out = json.load(f)
        
    # (e) ensemble_own has no key containing d_c, paired or cluster
    ens = out['ensemble_own']
    ens_str = json.dumps(ens)
    assert 'd_c' not in ens_str
    assert 'paired' not in ens_str
    assert 'cluster' not in ens_str
    
    # (c) factor functions and test
    import sys
    sys.path.append(r"C:\Projects\LINCS")
    from model.v9.coldcell_h2h import per_row_pearson, compute_centred_pearson
    
    r_o1 = per_row_pearson(o1_pred, y_true)
    r_o2 = per_row_pearson(o2_pred, y_true)
    seed_averaged = np.mean([r_o1, r_o2], axis=0)
    
    r_avg_pred = per_row_pearson((o1_pred + o2_pred) / 2.0, y_true)
    assert not np.allclose(seed_averaged, r_avg_pred)
    
    # (d) centred score invariant
    cells = np.array(['C1', 'C1', 'C2'])
    u = np.array(['C1', 'C2'])
    base_c = compute_centred_pearson(o1_pred, y_true, cells, u)
    
    o1_mod = o1_pred.copy()
    o1_mod[cells == 'C1'] += np.array([[10.0, -5.0]])
    mod_c = compute_centred_pearson(o1_mod, y_true, cells, u)
    np.testing.assert_allclose(base_c, mod_c)



def _toy(tmp_path, n=40, g=20, seed=0):
    """A non-degenerate toy (20 genes, 4 cells) written as the script's inputs."""
    rng = np.random.default_rng(seed)
    row_idx = np.arange(n)
    y = rng.normal(size=(n, g)); ctl = rng.normal(size=(n, g)) * 0.1
    t_pred = y + rng.normal(size=(n, g)) * 1.5
    preds = [y + rng.normal(size=(n, g)) * s for s in (1.0, 1.3, 0.8)]
    t_file = tmp_path / 't.npy'
    np.save(t_file, {'row_index': row_idx, 'y_pred': t_pred, 'y_true': y, 'ctl_true': ctl})
    files = []
    for i, p in enumerate(preds):
        f = tmp_path / ('o%d.npz' % i)
        np.savez(f, row_index=row_idx, y_pred=p, y_true=y, ctl_true=ctl)
        files.append(str(f))
    h5 = tmp_path / 'toy.h5ad'
    with h5py.File(h5, 'w') as f:
        f.create_group('obs').create_dataset('cell_iname', data=np.array(['A', 'B', 'C', 'D'] * (n // 4), dtype='S'))
    return t_file, files, h5, y, ctl, preds


def _run(t_file, ours, h5, out, extra=()):
    subprocess.run([PYTHON_EXE, SCRIPT_PATH, '--theirs', str(t_file), '--ours'] + list(ours) +
                   ['--h5ad', str(h5), '--split', 'toy', '--theirs_label', 'T', '--n_boot', '200', '--out', str(out)] +
                   list(extra), check=True, capture_output=True)
    return json.load(open(out))


def test_pi_seed_averaged_score_in_output(tmp_path):
    """PI (review of W22): the script's ours_mean is the mean over rows of the per-file per-row Pearson, averaged over files."""
    import sys
    sys.path.append(r"C:\Projects\LINCS")
    from model.v9.coldcell_h2h import per_row_pearson
    t_file, files, h5, y, ctl, preds = _toy(tmp_path)
    out = _run(t_file, files, h5, tmp_path / 'o.json')
    want = np.mean([per_row_pearson(p - ctl, y - ctl) for p in preds], axis=0).mean()
    assert out['row_pooled']['ours_mean'] == round(float(want), 5)
    ens = per_row_pearson(np.mean(preds, 0) - ctl, y - ctl).mean()
    assert out['ensemble_own']['ours_mean_over_rows'] == round(float(ens), 5)
    assert abs(ens - want) > 1e-3                      # the two estimands genuinely differ on this toy


def test_pi_alt_never_changes_headline(tmp_path):
    """PI (review of W22): --ours_alt with a non-finite row must leave the headline row set and numbers unchanged."""
    t_file, files, h5, y, ctl, preds = _toy(tmp_path)
    base = _run(t_file, files, h5, tmp_path / 'b.json')
    bad = preds[0].copy(); bad[3] = ctl[3]            # pred - ctl == 0 on row 3 -> NaN Pearson for alt only
    alt = tmp_path / 'alt.npz'
    np.savez(alt, row_index=np.arange(len(y)), y_pred=bad, y_true=y, ctl_true=ctl)
    with_alt = _run(t_file, files, h5, tmp_path / 'a.json', ['--ours_alt', str(alt), '--ours_alt_label', 'V2-last'])
    for k in ('cluster', 'per_cell', 'row_pooled', 'verdict', 'centred'):
        assert with_alt[k] == base[k], k
    assert with_alt['alt']['n_rows_dropped_alt_only'] == 1
