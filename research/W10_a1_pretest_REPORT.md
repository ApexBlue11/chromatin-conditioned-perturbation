# W10 A1 Pretest Report

## `model/v9/a1_diffusion_pretest.py`
```python
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.stats
import csv
import os

def get_K(W, alpha):
    deg = W.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    
    P = W * np.outer(d_inv_sqrt, d_inv_sqrt)
    I = np.eye(W.shape[0])
    K = np.linalg.solve(I - alpha * P, I)
    return K

def compute_score(s, z, target_indices):
    mask = np.ones(s.shape[0], dtype=bool)
    mask[target_indices] = False
    s_sub = s[mask]
    z_sub = z[mask]
    
    if np.all(s_sub == s_sub[0]) or np.all(z_sub == z_sub[0]):
        return np.nan
        
    return float(scipy.stats.spearmanr(s_sub, np.abs(z_sub))[0])

def get_mismatch_cells(c_idx_in_eligible, num_eligible, seed):
    rng = np.random.default_rng(seed + c_idx_in_eligible)
    others = [i for i in range(num_eligible) if i != c_idx_in_eligible]
    if len(others) <= 5:
        return others
    return rng.choice(others, size=5, replace=False).tolist()

def generate_resamples(n_cells, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_cells, size=(n_boot, n_cells))

def compute_contrast_stats(d_c, resamples):
    mean_d = np.nanmean(d_c)
    
    valid_mask = ~np.isnan(d_c)
    n_valid = int(np.sum(valid_mask))
    
    if n_valid == 0:
        return {
            "mean_d": np.nan,
            "ci95": [np.nan, np.nan],
            "n_cells_positive": 0,
            "p_value": np.nan
        }

    with np.errstate(invalid='ignore'):
        boot_means = np.nanmean(d_c[resamples], axis=1)
        
    valid_boot_means = boot_means[~np.isnan(boot_means)]
    if len(valid_boot_means) > 0:
        ci95 = np.percentile(valid_boot_means, [2.5, 97.5]).tolist()
    else:
        ci95 = [np.nan, np.nan]
        
    n_cells_positive = int(np.sum(d_c[valid_mask] > 0))
    p_val = float(scipy.stats.binomtest(n_cells_positive, n_valid, 0.5, alternative='two-sided').pvalue)
    
    return {
        "mean_d": float(mean_d),
        "ci95": ci95,
        "n_cells_positive": n_cells_positive,
        "p_value": p_val
    }

def get_reading(median_none, ci95_none_degree, 
                mean_own_mismatch, ci95_own_mismatch, n_cells_positive, n_cells, 
                ci95_own_none, ci95_own_mean):
    if median_none <= 0.01 or (ci95_none_degree[0] <= 0 and ci95_none_degree[1] >= 0):
        return "UNINFORMATIVE"
    elif mean_own_mismatch > 0 and ci95_own_mismatch[0] > 0 and n_cells_positive >= 0.75 * n_cells:
        return "SUPPORTS"
    elif ci95_own_none[0] > 0 and (ci95_own_mean[0] <= 0 and ci95_own_mean[1] >= 0):
        return "GENE_LEVEL_PRIOR"
    else:
        return "NO_SUPPORT"

def check_gene_order(nodes, landmark_genes):
    assert list(nodes) == landmark_genes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--n_boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    root = Path("C:/Projects/LINCS")
    
    print("Loading graph...")
    union_graph = np.load(root / "network/outputs/v9/union_graph_v9.npz")
    edge_index = union_graph["edge_index"]
    nodes = union_graph["nodes"]
    A = np.zeros((978, 978), dtype=np.float64)
    A[edge_index[0], edge_index[1]] = 1.0
    A[edge_index[1], edge_index[0]] = 1.0
    np.fill_diagonal(A, 0.0)

    print("Verifying gene order...")
    with open(root / "Data Info/pathway_landmark_genes.txt", "r") as f:
        landmark_genes = [line.strip() for line in f if line.strip()]
    check_gene_order(nodes, landmark_genes)

    print("Loading targets...")
    dti_reference = {}
    with open(root / "drug/outputs/dti/dti_reference.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pert_id = row["pert_id"]
            gene_symbol = row["gene_symbol"]
            gene_idx = int(row["gene_idx"])
            assert nodes[gene_idx] == gene_symbol
            if pert_id not in dti_reference:
                dti_reference[pert_id] = set()
            dti_reference[pert_id].add(gene_idx)

    print("Loading response Y...")
    Y = np.load(root / "phase2_assembly/outputs/Y_target_level5_978.npy")

    print("Loading signatures...")
    signatures = []
    with open(root / "phase2_assembly/outputs/signatures_usable.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            signatures.append({
                "row": int(row["row"]),
                "cell_id": row["cell_id"],
                "pert_id": row["pert_id"]
            })

    print("Loading chromatin data...")
    E_final = np.load(root / "phase2_assembly/outputs/E_final.npy")
    E_final_mask = np.load(root / "phase2_assembly/outputs/E_final_mask.npy")
    with open(root / "epigenetics/outputs/epigenetics_cell_index.json", "r", encoding="utf-8-sig") as f:
        cell_id_to_row = json.load(f)["cell_id_to_row"]

    print("Filtering eligible cells...")
    channel = args.channel
    eligible_cells = []
    for cell_id, c in cell_id_to_row.items():
        c = int(c)
        if E_final_mask[c, :, channel].mean() >= 0.9:
            eligible_cells.append(cell_id)

    eligible_cells.sort()
    if args.max_cells is not None:
        eligible_cells = eligible_cells[:args.max_cells]

    print(f"Found {len(eligible_cells)} eligible cells.")
    
    cell_vectors = {}
    for cell_id in eligible_cells:
        c = int(cell_id_to_row[cell_id])
        vec = E_final[c, :, channel].copy()
        mask = E_final_mask[c, :, channel]
        if not np.all(mask):
            median_val = np.median(vec[mask])
            vec[~mask] = median_val
        cell_vectors[cell_id] = vec

    valid_rows = []
    cell_row_counts = {cell_id: 0 for cell_id in eligible_cells}
    for sig in signatures:
        cid = sig["cell_id"]
        pid = sig["pert_id"]
        if cid in eligible_cells and pid in dti_reference and len(dti_reference[pid]) >= 1:
            valid_rows.append(sig)
            cell_row_counts[cid] += 1
            
    print(f"Found {len(valid_rows)} valid rows.")

    print("Precomputing operators...")
    K_none = get_K(A, args.alpha)
    s_degree = A.sum(axis=1)
    
    if len(eligible_cells) > 0:
        a_mean = np.mean([cell_vectors[cid] for cid in eligible_cells], axis=0)
        K_mean = get_K(A * np.sqrt(np.outer(a_mean, a_mean)), args.alpha)
    else:
        K_mean = K_none
    
    K_cell = {}
    for cid in eligible_cells:
        a_c = cell_vectors[cid]
        K_cell[cid] = get_K(A * np.sqrt(np.outer(a_c, a_c)), args.alpha)

    print("Processing scores...")
    results_per_row = {
        "row_index": [],
        "cell_id_index": [],
        "score_own": [],
        "score_mismatch": [],
        "score_mean": [],
        "score_none": [],
        "score_degree": []
    }

    for cid_idx, cid in enumerate(eligible_cells):
        mismatch_indices = get_mismatch_cells(cid_idx, len(eligible_cells), args.seed)
        mismatch_cids = [eligible_cells[i] for i in mismatch_indices]
        
        rows_for_cell = [r for r in valid_rows if r["cell_id"] == cid]
        for r in rows_for_cell:
            pert_id = r["pert_id"]
            targets = list(dti_reference[pert_id])
            t_d = np.zeros(978, dtype=np.float64)
            t_d[targets] = 1.0 / len(targets)
            
            s_own = K_cell[cid] @ t_d
            score_own = compute_score(s_own, Y[r["row"]], targets)
            
            score_mismatches = []
            for m_cid in mismatch_cids:
                s_m = K_cell[m_cid] @ t_d
                score_mismatches.append(compute_score(s_m, Y[r["row"]], targets))
            with np.errstate(invalid='ignore'):
                score_mismatch = float(np.nanmean(score_mismatches))
            
            s_mean = K_mean @ t_d
            score_mean = compute_score(s_mean, Y[r["row"]], targets)
            
            s_none = K_none @ t_d
            score_none = compute_score(s_none, Y[r["row"]], targets)
            
            score_degree = compute_score(s_degree, Y[r["row"]], targets)
            
            results_per_row["row_index"].append(r["row"])
            results_per_row["cell_id_index"].append(cid_idx)
            results_per_row["score_own"].append(score_own)
            results_per_row["score_mismatch"].append(score_mismatch)
            results_per_row["score_mean"].append(score_mean)
            results_per_row["score_none"].append(score_none)
            results_per_row["score_degree"].append(score_degree)

    for k in results_per_row:
        results_per_row[k] = np.array(results_per_row[k], dtype=np.float64)
        if k in ["row_index", "cell_id_index"]:
            results_per_row[k] = results_per_row[k].astype(int)

    print("Computing statistics...")
    d_c_dict = {
        "OWN-MISMATCH": np.full(len(eligible_cells), np.nan),
        "OWN-MEAN": np.full(len(eligible_cells), np.nan),
        "OWN-NONE": np.full(len(eligible_cells), np.nan),
        "NONE-DEGREE": np.full(len(eligible_cells), np.nan)
    }

    for cid_idx in range(len(eligible_cells)):
        mask = results_per_row["cell_id_index"] == cid_idx
        if not np.any(mask):
            continue
            
        def get_median_diff(key1, key2):
            diff = results_per_row[key1][mask] - results_per_row[key2][mask]
            diff = diff[~np.isnan(diff)]
            return np.median(diff) if len(diff) > 0 else np.nan
            
        d_c_dict["OWN-MISMATCH"][cid_idx] = get_median_diff("score_own", "score_mismatch")
        d_c_dict["OWN-MEAN"][cid_idx] = get_median_diff("score_own", "score_mean")
        d_c_dict["OWN-NONE"][cid_idx] = get_median_diff("score_own", "score_none")
        d_c_dict["NONE-DEGREE"][cid_idx] = get_median_diff("score_none", "score_degree")

    resamples = generate_resamples(len(eligible_cells), args.n_boot, args.seed)
    
    stats_dict = {}
    for contrast, d_c in d_c_dict.items():
        stats_dict[contrast] = compute_contrast_stats(d_c, resamples)

    with np.errstate(invalid='ignore'):
        condition_medians = {
            "OWN": float(np.nanmedian(results_per_row["score_own"])) if len(results_per_row["score_own"]) > 0 else np.nan,
            "MISMATCH": float(np.nanmedian(results_per_row["score_mismatch"])) if len(results_per_row["score_mismatch"]) > 0 else np.nan,
            "MEAN": float(np.nanmedian(results_per_row["score_mean"])) if len(results_per_row["score_mean"]) > 0 else np.nan,
            "NONE": float(np.nanmedian(results_per_row["score_none"])) if len(results_per_row["score_none"]) > 0 else np.nan,
            "DEGREE": float(np.nanmedian(results_per_row["score_degree"])) if len(results_per_row["score_degree"]) > 0 else np.nan
        }

    valid_cells_om = np.sum(~np.isnan(d_c_dict["OWN-MISMATCH"]))
    reading = get_reading(
        condition_medians["NONE"],
        stats_dict["NONE-DEGREE"]["ci95"],
        stats_dict["OWN-MISMATCH"]["mean_d"],
        stats_dict["OWN-MISMATCH"]["ci95"],
        stats_dict["OWN-MISMATCH"]["n_cells_positive"],
        valid_cells_om,
        stats_dict["OWN-NONE"]["ci95"],
        stats_dict["OWN-MEAN"]["ci95"]
    )

    print(f"Reading: {reading}")
    print("Saving results...")
    
    os.makedirs(root / "model/results", exist_ok=True)
    suffix = f"_ch{args.channel}"
    if args.max_cells is not None:
        suffix += f"_cells{args.max_cells}"

    json_path = root / f"model/results/a1_diffusion_pretest{suffix}.json"
    npz_path = root / f"model/results/a1_diffusion_pretest{suffix}_rows.npz"

    with open(json_path, "w") as f:
        json.dump({
            "reading_82_5": reading,
            "condition_medians": condition_medians,
            "contrasts": stats_dict,
            "cell_row_counts": cell_row_counts,
            "args": vars(args)
        }, f, indent=2)

    np.savez(npz_path, **results_per_row)
    print(f"Saved {json_path.name} and {npz_path.name}")

if __name__ == "__main__":
    main()
```

## `model/v9/test_a1_diffusion.py`
```python
import numpy as np
from a1_diffusion_pretest import (
    get_K, compute_score, get_mismatch_cells, 
    generate_resamples, compute_contrast_stats, get_reading,
    check_gene_order
)
import scipy.stats

def test_t1():
    W = np.array([[0, 1], [1, 0]], dtype=float)
    K = get_K(W, 0.0)
    assert np.allclose(K, np.eye(2))
    t = np.array([0.5, 0.5])
    s = K @ t
    assert np.allclose(s, t)

def test_t2():
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    a = np.array([1.0, 1.0, 1.0])
    W_own = A * np.sqrt(np.outer(a, a))
    K_own = get_K(W_own, 0.5)
    K_none = get_K(A, 0.5)
    assert np.allclose(K_own, K_none, atol=1e-12)

def test_t3():
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    alpha = 0.5
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    P = A * np.outer(d_inv_sqrt, d_inv_sqrt)
    K = get_K(A, alpha)
    
    assert np.allclose(K, K.T, atol=1e-10)
    I = np.eye(3)
    assert np.allclose((I - alpha * P) @ K, I, atol=1e-10)

def test_t4():
    A = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    K = get_K(A, 0.5)
    assert not np.any(np.isnan(K))
    assert not np.any(np.isinf(K))

def test_t5():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    z1 = np.array([1.0, -2.0, 3.0, -4.0])
    z2 = np.array([5.0, -2.0, 3.0, -4.0])
    targets = [0]
    
    score1 = compute_score(s, z1, targets)
    score2 = compute_score(s, z2, targets)
    assert score1 == score2
    
    z3 = np.array([1.0, -5.0, 3.0, -4.0])
    score3 = compute_score(s, z3, targets)
    assert score1 != score3

def test_t6():
    eligible_list = 10
    idx = 2
    seed = 42
    draw1 = get_mismatch_cells(idx, eligible_list, seed)
    assert idx not in draw1
    assert len(draw1) == len(set(draw1))
    
    draw2 = get_mismatch_cells(idx, eligible_list, seed)
    assert draw1 == draw2

def test_t7():
    nodes = np.array(["A", "B", "C"])
    genes = ["A", "B", "C"]
    check_gene_order(nodes, genes)
    
    genes_permuted = ["B", "A", "C"]
    raised = False
    try:
        check_gene_order(nodes, genes_permuted)
    except AssertionError:
        raised = True
    assert raised

def test_t8():
    d_c = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    resamples = generate_resamples(5, 200, 42)
    stats = compute_contrast_stats(d_c, resamples)
    assert np.isclose(stats["mean_d"], 0.3)
    assert np.allclose(stats["ci95"], [0.3, 0.3])

def test_t9():
    assert get_reading(0.01, [0.1, 0.2], 0.5, [0.1, 0.9], 10, 10, [0.1, 0.9], [0.1, 0.9]) == "UNINFORMATIVE"
    assert get_reading(0.05, [-0.1, 0.2], 0.5, [0.1, 0.9], 10, 10, [0.1, 0.9], [0.1, 0.9]) == "UNINFORMATIVE"
    assert get_reading(0.05, [0.1, 0.2], 0.5, [0.1, 0.9], 8, 10, [0.1, 0.9], [0.1, 0.9]) == "SUPPORTS"
    assert get_reading(0.05, [0.1, 0.2], 0.5, [-0.1, 0.9], 8, 10, [0.1, 0.9], [-0.1, 0.2]) == "GENE_LEVEL_PRIOR"
    assert get_reading(0.05, [0.1, 0.2], 0.5, [-0.1, 0.9], 8, 10, [-0.1, 0.9], [0.1, 0.9]) == "NO_SUPPORT"

if __name__ == "__main__":
    test_t1()
    test_t2()
    test_t3()
    test_t4()
    test_t5()
    test_t6()
    test_t7()
    test_t8()
    test_t9()
    print("All tests passed.")
```

## Verifications Output

### `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe test_a1_diffusion.py`
```
All tests passed.
```

### `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_diffusion_pretest.py --help`
```
usage: a1_diffusion_pretest.py [-h] [--channel {0,1}] [--max_cells MAX_CELLS]
                               [--n_boot N_BOOT] [--seed SEED] [--alpha ALPHA]

options:
  -h, --help            show this help message and exit
  --channel {0,1}
  --max_cells MAX_CELLS
  --n_boot N_BOOT
  --seed SEED
  --alpha ALPHA
```

### `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe a1_diffusion_pretest.py --max_cells 3 --n_boot 200`
```
Loading graph...
Verifying gene order...
Loading targets...
Loading response Y...
Loading signatures...
Loading chromatin data...
Filtering eligible cells...
Found 3 eligible cells.
Found 6988 valid rows.
Precomputing operators...
Processing scores...
Computing statistics...
Reading: UNINFORMATIVE
Saving results...
Saved a1_diffusion_pretest_ch0_cells3.json and a1_diffusion_pretest_ch0_cells3_rows.npz
```

## What I was unsure about
- **Dealing with missing values when extracting JSON parameters**: `epigenetics_cell_index.json` contained a top level dictionary wrapping `cell_id_to_row` rather than directly holding it. This was initially causing indexing issues because I didn't see the exact file structure, but it was resolved after inspecting the JSON format directly.
- **`scipy.stats.binomtest` Compatibility:** The `binomtest` function was added in SciPy `1.7.0` (replacing `binom_test`). Considering the instructions indicated to just use `numpy`/`scipy` efficiently, I assumed the `scipy` version was at least `1.7.0`. The tests indeed verified it succeeded properly.
- **Handling NaN predictions during Resampling**: In `compute_contrast_stats`, I utilized `np.nanmean` during resampling in the case where bootstrapping may somehow choose resampled items containing full NaNs.
- **NaN score vs Valid Score Rows:** The instructions noted that predictions that were identical over kept genes are assigned a `np.nan` score. If a cell contains valid rows, we assign those row values into the overall difference computations.
