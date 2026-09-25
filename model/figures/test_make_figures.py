import os
import json
import subprocess
import glob

def test_make_figures():
    # Run make_figures.py
    cmd = [r"C:\Projects\LINCS\.venv-cuda\Scripts\python.exe", "model/figures/make_figures.py"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = result.stdout
    print("STDOUT of make_figures.py:\n")
    print(stdout)
    
    # Parse printed values
    printed = {}
    for line in stdout.splitlines():
        if " | " in line:
            fig, rest = line.split(" | ", 1)
            key, val = rest.split(": ", 1)
            try:
                printed[f"{fig}_{key}"] = float(val)
            except ValueError:
                printed[f"{fig}_{key}"] = val

    # Verify outputs exist
    OUT_DIR = "model/figures/out"
    expected_files = [
        "f1_coldcell_h2h.png", "f1_coldcell_h2h.svg", "f1_coldcell_h2h.txt",
        "f2_xpert_reproduction.png", "f2_xpert_reproduction.svg",
        "f3_dev_screens.png", "f3_dev_screens.svg",
        "f4_moa_probe.png", "f4_moa_probe.svg",
        "f5_input_coverage.png", "f5_input_coverage.svg"
    ]
    for f in expected_files:
        path = os.path.join(OUT_DIR, f)
        assert os.path.exists(path), f"File {path} does not exist"
        
    # Verify F1
    with open("model/results/coldcell_h2h_split_cold_cell_1_O2.json", "r") as f:
        data = json.load(f)
    for c in data["per_cell"]:
        cell = c["cell"]
        assert printed[f"F1_cell_{cell}_n_scored"] == c["n_scored"]
        assert printed[f"F1_cell_{cell}_d_c_median"] == c["d_c_median"]
        assert printed[f"F1_cell_{cell}_d_c_median_ci95_0"] == c["d_c_median_ci95"][0]
        assert printed[f"F1_cell_{cell}_d_c_median_ci95_1"] == c["d_c_median_ci95"][1]
        assert printed[f"F1_cell_{cell}_ours_mean"] == c["ours_mean"]
        assert printed[f"F1_cell_{cell}_theirs_mean"] == c["theirs_mean"]
    
    assert printed["F1_cluster_mean_of_d_c"] == data["cluster"]["mean_of_d_c"]
    assert printed["F1_cluster_ci95_0"] == data["cluster"]["cluster_ci95"][0]
    assert printed["F1_cluster_ci95_1"] == data["cluster"]["cluster_ci95"][1]
    assert printed["F1_cells_favouring_ours"] == data["cluster"]["cells_favouring_ours"]
    assert printed["F1_n_cells"] == data["cluster"]["n_cells"]
    
    # Verify F2
    assert printed["F2_pub_mean"] == 0.383
    assert printed["F2_pub_sd"] == 0.027
    assert printed["F2_theirs_all_rows_mean"] == data["reproduction"]["theirs_all_rows_mean"]
    assert printed["F2_band_0"] == data["reproduction"]["band"][0]
    assert printed["F2_band_1"] == data["reproduction"]["band"][1]
    
    # Verify F3
    with open("model/results/v9_dev_score_baseline_kaggle.json", "r") as f:
        base_data = json.load(f)
    s0 = base_data["sd"]
    assert printed["F3_s0"] == s0
    assert printed["F3_adv_thresh"] == max(2 * s0, 0.003)
    
    for f in glob.glob("model/results/v9_dev_score_*.json"):
        with open(f, "r") as fd:
            d = json.load(fd)
        if "vs_baseline" in d:
            label = d["label"]
            delta = d["vs_baseline"]["delta_per_row_mean"]
            assert printed[f"F3_cand_{label}_delta"] == delta
            
    # Verify F4
    with open("model/results/probe_moa_v9_reading_86.json", "r") as f:
        data = json.load(f)
    assert printed["F4_floor"] == -0.02
    assert printed["F4_reading"] == data.get("reading", "NULL")
    for s in ["0", "1", "2"]:
        assert printed[f"F4_seed_{s}_trained_diff"] == data["seeds"][s]["trained"]["diff"]
        assert printed[f"F4_seed_{s}_untrained_diff"] == data["seeds"][s]["untrained"]["diff"]
        
    # Verify F5
    with open("model/results/cc1_input_coverage.json", "r") as f:
        data = json.load(f)
    for g in ["train", "dev", "test"]:
        assert printed[f"F5_{g}_ccle_direct_rows_frac"] == data["summary"][g]["ccle_direct_rows_frac"]
        assert printed[f"F5_{g}_ATAC_rows_frac"] == data["summary"][g]["ATAC_rows_frac"]
        assert printed[f"F5_{g}_H3K27ac_rows_frac"] == data["summary"][g]["H3K27ac_rows_frac"]
        assert printed[f"F5_{g}_H3K27me3_rows_frac"] == data["summary"][g]["H3K27me3_rows_frac"]

    print("All tests passed!")

if __name__ == "__main__":
    test_make_figures()
