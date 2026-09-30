import os
import sys
import tempfile
import json
import numpy as np
from unittest.mock import patch
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xpert_arm import main as xpert_arm_main
from align_dev import licence

def run_xpert_arm(args_list, work_dir):
    with patch('sys.argv', ['xpert_arm.py'] + args_list):
        with patch('xpert_arm.os.path.isdir', side_effect=lambda p: False if p == '/kaggle/working' else os.path.isdir(p)):
            with patch('xpert_arm.os.path.join', side_effect=lambda *a: work_dir if 'results' in a[-1] else os.path.join(*a)):
                xpert_arm_main()

def test_a_b():
    import io
    from contextlib import redirect_stdout
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_file = os.path.join(tmp_dir, 'x.npz')
        args = [
            '--dev_cells', '6',
            '--epochs', '1',
            '--limit_train', '300',
            '--seeds', '1',
            '--batch', '2',
            '--save_pred', pred_file,
            '--no_test_metrics',
            '--bundle', 'xpert_mdmt_splits.npz',
            '--split', 'split_cold_cell_1'
        ]
        
        f = io.StringIO()
        with redirect_stdout(f):
            # Patch sys.argv for the function call
            with patch('sys.argv', ['xpert_arm.py'] + args):
                # We need to mock WORK in xpert_arm to tmp_dir
                import xpert_arm
                original_makedirs = os.makedirs
                original_join = os.path.join
                original_isdir = os.path.isdir
                
                def mock_isdir(path):
                    if path == '/kaggle/working': return False
                    return original_isdir(path)
                
                def mock_join(*args):
                    if len(args) > 1 and args[-1] == 'results':
                        return tmp_dir
                    return original_join(*args)

                with patch('os.path.isdir', mock_isdir), \
                     patch('os.path.join', mock_join):
                    xpert_arm.main()
        
        stdout_text = f.getvalue()
        
        # Check output JSON
        # Need to find the json file generated
        import glob
        json_files = glob.glob(os.path.join(tmp_dir, '*.json'))
        assert len(json_files) == 1
        with open(json_files[0]) as jf:
            data = json.load(jf)
            for run in data['runs']:
                for k in run.keys():
                    assert 'Pearson' not in k, f"Found Pearson key in JSON runs: {k}"
        
        # Check stdout
        import re
        assert not re.search(r'Pearson_deg\s+[-0-9]', stdout_text), "Found Pearson_deg in stdout"
        assert not re.search(r'\[seed \d+\] Pearson', stdout_text), "Found [seed X] Pearson in stdout"
        
        # Check npz
        pred_base = pred_file.replace('.npz', '_dev6s0_seed0')
        npz_file = f'{pred_base}.npz'
        assert os.path.exists(npz_file), f"Expected npz file {npz_file} not found"
        with np.load(npz_file) as z:
            assert 'y_pred' in z.files
            assert 'deg_pred' in z.files
            assert 'y_true' in z.files
            assert 'ctl_true' in z.files

        # Part B: Run without flag
        args_no_flag = [
            '--dev_cells', '6',
            '--epochs', '1',
            '--limit_train', '300',
            '--seeds', '1',
            '--batch', '2',
            '--save_pred', pred_file,
            '--bundle', 'xpert_mdmt_splits.npz',
            '--split', 'split_cold_cell_1'
        ]
        
        f2 = io.StringIO()
        with redirect_stdout(f2):
            with patch('sys.argv', ['xpert_arm.py'] + args_no_flag):
                with patch('os.path.isdir', mock_isdir), \
                     patch('os.path.join', mock_join):
                    xpert_arm.main()
        
        stdout_text2 = f2.getvalue()
        assert re.search(r'\[seed \d+\] Pearson', stdout_text2), "Expected [seed X] Pearson not found in stdout without flag"
        print("Smoke tests A and B passed.")

def test_c():
    # licence(...) tests
    # per_seed_cell_increments is list of dicts {cell: float}
    
    # 8 toy cells with 7 positive -> licensed at min_cells 7
    cells = [str(i) for i in range(8)]
    inc = {c: 1.0 if i < 7 else -1.0 for i, c in enumerate(cells)}
    inc2 = {c: 1.0 if i < 7 else -1.0 for i, c in enumerate(cells)}
    inc3 = {c: 1.0 if i < 7 else -1.0 for i, c in enumerate(cells)}
    per_seed_incs = [inc, inc2, inc3]
    alignments = [0.5, 0.5, 0.5]
    prior = 0.4
    
    res = licence(per_seed_incs, alignments, prior, min_cells=7)
    assert res['cells_positive'] == 7
    assert res['in_cell_licensed'] == True
    
    # 6 positive -> not licensed at min_cells 7
    inc_6 = {c: 1.0 if i < 6 else -1.0 for i, c in enumerate(cells)}
    res_6 = licence([inc_6, inc_6, inc_6], alignments, prior, min_cells=7)
    assert res_6['cells_positive'] == 6
    assert res_6['in_cell_licensed'] == False
    
    # a seed whose cell mean is <= 0 -> not licensed
    inc_bad_seed = {c: -10.0 for c in cells}
    res_bad = licence([inc, inc, inc_bad_seed], alignments, prior, min_cells=7)
    # The mean of inc_bad_seed is < 0, so seed_means_positive should be False
    assert res_bad['seed_means_positive'] == False
    assert res_bad['in_cell_licensed'] == False
    
    # beats_prior_every_seed False when one seed's alignment <= prior
    res_prior = licence([inc, inc, inc], [0.5, 0.5, 0.3], 0.4, min_cells=7)
    assert res_prior['beats_prior_every_seed'] == False

    print("Licence toy tests C passed.")

def test_d():
    # The dev path: on the committed model/results/v9_dev_align_P2_baseline_incell_aux.json
    # read per-checkpoint increments, call licence with min_cells=5
    path = os.path.join(HERE, '..', 'results', 'v9_dev_align_P2_baseline_incell_aux.json')
    with open(path, 'r') as f:
        data = json.load(f)
    
    per = data['per_checkpoint']
    per_seed_incs = [r['in_cell_increment'] for r in per]
    alignments = [r['alignment'] for r in per]
    prior = data['references']['training_prior']
    
    res = licence(per_seed_incs, alignments, prior, min_cells=5)
    
    expected_cells_positive = data['in_cell']['cells_positive']
    expected_licensed = data['in_cell']['licensed']
    
    assert res['cells_positive'] == expected_cells_positive
    assert res['in_cell_licensed'] == expected_licensed
    
    print("Dev path test D passed.")

if __name__ == '__main__':
    test_c()
    test_d()
    test_a_b()
    print("All tests passed.")
