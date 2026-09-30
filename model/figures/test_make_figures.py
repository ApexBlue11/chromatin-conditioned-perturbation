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
                if val in ["True", "False"]:
                    printed[f"{fig}_{key}"] = val == "True"
                else:
                    printed[f"{fig}_{key}"] = float(val)
            except ValueError:
                printed[f"{fig}_{key}"] = val

    # Verify outputs exist
    OUT_DIR = "model/figures/out"
    expected_files = [
        "f1_coldcell_h2h.png", "f1_coldcell_h2h.svg",
        "f2_xpert_reproduction.png", "f2_xpert_reproduction.svg",
        "f3_dev_screens.png", "f3_dev_screens.svg",
        "f4_moa_probe.png", "f4_moa_probe.svg",
        "f5_input_coverage.png", "f5_input_coverage.svg",
        "f6_seed_ensemble.png", "f6_seed_ensemble.svg",
        "f7_pathway_alignment.png", "f7_pathway_alignment.svg"
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
    
    rows = [
        ("C1_noatoms", "v9_dev_score_C1_noatoms.json"),
        ("C2_lctl0", "v9_dev_score_C2_lctl0.json"),
        ("C3_listnet", "v9_dev_score_C3_listnet.json"),
        ("C4_degk50", "v9_dev_score_C4_degk50.json"),
        ("C6_signhead_3seed", "v9_dev_score_C6_signhead_3seed.json"),
        ("C7_chromedges_3seed", "v9_dev_score_C7_chromedges_3seed.json"),
        ("C8b_postpath", "v9_dev_score_C8b_postpath.json")
    ]
    import numpy as np
    for key, fname in rows:
        with open("model/results/" + fname, "r") as fd:
            d = json.load(fd)
        
        vb = d["vs_baseline"]
        delta = vb["delta_per_row_mean"]
        assert abs(printed[f"F3_{key}_delta_per_row_mean"] - delta) < 1e-9
        
        if vb.get("n_seeds_variant", 1) == 3:
            sd = d.get("sd", 0.0)
            err = 2 * np.sqrt((s0**2)/3 + (sd**2)/3)
            assert abs(printed[f"F3_{key}_err"] - err) < 1e-9
            
        if "delta_centred" in vb:
            assert abs(printed[f"F3_{key}_delta_centred"] - vb["delta_centred"]) < 1e-9

    # Verify F4
    with open("model/results/probe_moa_v9_reading_86.json", "r") as f:
        data = json.load(f)
    assert printed["F4_floor"] == -0.02
    assert printed["F4_reading"] == data.get("reading", "NULL")
    for s in ["0", "1", "2"]:
        assert printed[f"F4_seed_{s}_trained_diff"] == data["seeds"][s]["trained"]["diff"]
        assert printed[f"F4_seed_{s}_untrained_diff"] == data["seeds"][s]["untrained"]["diff"]
        assert printed[f"F4_seed_{s}_trained_p"] == data["seeds"][s]["trained"]["p"]
        assert printed[f"F4_seed_{s}_untrained_p"] == data["seeds"][s]["untrained"]["p"]
        
    # Verify F5
    with open("model/results/cc1_input_coverage.json", "r") as f:
        data = json.load(f)
    for g in ["train", "dev", "test"]:
        assert printed[f"F5_{g}_ccle_direct_rows_frac"] == data["summary"][g]["ccle_direct_rows_frac"]
        assert printed[f"F5_{g}_ATAC_rows_frac"] == data["summary"][g]["ATAC_rows_frac"]
        assert printed[f"F5_{g}_H3K27ac_rows_frac"] == data["summary"][g]["H3K27ac_rows_frac"]
        assert printed[f"F5_{g}_H3K27me3_rows_frac"] == data["summary"][g]["H3K27me3_rows_frac"]

    # Verify F6
    assert abs(printed["F6_P2_raw_K3_0"] - 0.4662) <= 5e-5
    assert abs(printed["F6_P2_centred_K3_0"] - 0.5018) <= 5e-5
    
    # Verify F7
    with open("model/results/v9_dev_align_P2_baseline_aux.json", "r") as f:
        p2_aux = json.load(f)
    assert printed["F7_training_prior"] == p2_aux["references"]["training_prior"]
    assert printed["F7_loco_prior"] == p2_aux["references"]["loco_prior"]
    assert printed["F7_mean_null_mean"] == p2_aux["mean_null_mean"]
    
    for chk in p2_aux["per_checkpoint"]:
        s = chk["seed"]
        assert printed[f"F7_seed_{s}_alignment"] == chk["alignment"]
        assert printed[f"F7_seed_{s}_cell_shuffle_mean"] == chk["cell_shuffle_mean"]
        assert printed[f"F7_seed_{s}_cell_shuffle_sd"] == chk["cell_shuffle_sd"]
        
    with open("model/results/v9_dev_align_P2_baseline_incell_aux.json", "r") as f:
        p2_incell = json.load(f)
    with open("model/results/v9_dev_align_C6_signhead_aux.json", "r") as f:
        c6_incell = json.load(f)
        
    for c in p2_incell["in_cell"]["per_cell_3seed_mean"]:
        assert printed[f"F7_P2_incell_{c}"] == p2_incell["in_cell"]["per_cell_3seed_mean"][c]
        assert printed[f"F7_C6_incell_{c}"] == c6_incell["in_cell"]["per_cell_3seed_mean"][c]
        
    assert printed["F7_P2_licensed"] == p2_incell["in_cell"]["licensed"]
    assert printed["F7_P2_cells_positive"] == p2_incell["in_cell"]["cells_positive"]

    print("All tests passed!")

if __name__ == "__main__":
    test_make_figures()
