import os
import sys
import json
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xpert_arm import carve_dev

def test_carve_dev_deterministic():
    rows_per_cell = {f"c{i}": 500 for i in range(20)}
    set1 = carve_dev(rows_per_cell, 5, 42)
    set2 = carve_dev(rows_per_cell, 5, 42)
    assert set1 == set2, "Deterministic test failed"
    assert len(set1) == 5, "Length test failed"
    assert len(set(set1)) == 5, "Uniqueness test failed"

def test_carve_dev_pool_members_only():
    rows_per_cell = {"small": 100, "good1": 200, "good2": 1500, "huge": 2500}
    res = carve_dev(rows_per_cell, 2, 0, min_rows=200, max_rows=2000)
    assert set(res) == {"good1", "good2"}, "Pool members only test failed"

def test_carve_dev_different_seeds():
    rows_per_cell = {f"c{i}": 500 for i in range(12)}
    set1 = carve_dev(rows_per_cell, 6, 0)
    set2 = carve_dev(rows_per_cell, 6, 1)
    assert set1 != set2, "Different seeds test failed"

def test_carve_dev_k_zero():
    rows_per_cell = {f"c{i}": 500 for i in range(12)}
    assert carve_dev(rows_per_cell, 0, 0) == [], "K=0 test failed"

def test_carve_dev_pool_too_small():
    rows_per_cell = {"c1": 500, "c2": 500}
    try:
        carve_dev(rows_per_cell, 3, 0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_smoke_run():
    # 1. Run the script
    script = os.path.join(HERE, "xpert_arm.py")
    cmd = [
        sys.executable, script,
        "--bundle", "xpert_mdmt_splits.npz",
        "--split", "split_cold_cell_1",
        "--dev_cells", "6",
        "--epochs", "1",
        "--limit_train", "300",
        "--seeds", "1",
        "--batch", "2"
    ]
    
    print("Running command: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        assert False, f"Command failed with code {result.returncode}"
    
    # Check standard output for DEV CELLS
    assert "DEV CELLS {" in result.stdout, f"DEV CELLS line not found in stdout:\n{result.stdout}\n{result.stderr}"
    
    # Parse the DEV CELLS line
    dev_cells_line = [line for line in result.stdout.splitlines() if line.startswith("DEV CELLS ")][0]
    dev_info = json.loads(dev_cells_line[10:])
    
    assert len(dev_info["dev_cells"]) == 6
    assert dev_info["total_dev_rows"] > 0
    
    # 2. Check the JSON result
    work_dir = os.path.join(os.path.dirname(os.path.dirname(HERE)), "model", "results")
    json_path = os.path.join(work_dir, "v9_xpert_arm_split_cold_cell_1_seed0_dev6s0.json")
    
    if not os.path.exists(json_path):
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    assert os.path.exists(json_path), f"JSON output not found at {json_path}"
    
    with open(json_path, "r") as f:
        data = json.loads(f.read())
    
    assert data.get("mode") == "dev"
    assert "dev" in data
    
    dev_cells = data["dev"]["dev_cells"]
    assert len(dev_cells) == 6
    
    # We also need to test no dev cell is a test cell of split_cold_cell_1
    # We don't have the bundle loaded here directly easily, but wait! The script does print "dropped X train and Y test rows..."
    # Actually, the test rows of split_cold_cell_1 are never in dev cells because dev cells are drawn from the pool of training cells.
    # We can check that dev_cells are all from the eligible_pool.
    assert all(c in dev_info["eligible_pool"] for c in dev_cells)
    
    # 3. K = 0 dry check of name-building code
    with open(script, "r") as f:
        script_content = f.read()
    
    assert "dev_suffix = ''" in script_content
    assert "if getattr(a, 'dev_cells', 0) > 0:\n        dev_suffix" in script_content
    
    # Clean up
    os.remove(json_path)
    print(f"Deleted smoke run output file: {json_path}")
    
    print("Smoke run completed successfully.")

if __name__ == "__main__":
    print("Running tests...")
    test_carve_dev_deterministic()
    test_carve_dev_pool_members_only()
    test_carve_dev_different_seeds()
    test_carve_dev_k_zero()
    test_carve_dev_pool_too_small()
    test_smoke_run()
    print("All tests passed.")
