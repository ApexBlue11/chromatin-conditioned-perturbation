# -*- coding: utf-8 -*-
"""Tests for P10 arm M adapter, M+dt script builder, and Kaggle CPU kernel builder (RESULTS §98.2, §98.6)."""
import ast
import difflib
import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
import types

import anndata as ad
import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "model", "v9"))
sys.path.insert(0, os.path.join(REPO_ROOT, "orchestration"))

import make_p10_m_kernels  # noqa: E402
import p10_mlp  # noqa: E402

TRAIN_MLP_SRC = os.path.join(
    REPO_ROOT,
    "external",
    "bai_benchmark_1d750cd",
    "experiments",
    "_shared",
    "training",
    "train_mlp.py",
)


@pytest.fixture
def synthetic_bundle_data():
    """Create a synthetic dataset of 400 rows and 978 genes."""
    rng = np.random.default_rng(9810)
    n_rows = 400
    n_genes = 978

    row_index = np.arange(n_rows, dtype=np.int64)
    # Row-specific pattern D[r]
    d_pattern = np.zeros((n_rows, n_genes), dtype=np.float32)
    for r in range(n_rows):
        d_pattern[r, :] = (float(r) + 1.0) * 0.02 + np.sin(
            np.arange(n_genes, dtype=np.float32) * 0.1
        )

    x_ctl = rng.normal(size=(n_rows, n_genes)).astype(np.float32)
    x = x_ctl + d_pattern

    # 30 perts, 10 cells
    meta_pert_id = np.array([f"pert_{i % 30}" for i in range(n_rows)])
    meta_pert_idx = np.array([i % 30 for i in range(n_rows)], dtype=np.int64)
    meta_cell = np.array([f"cell_{i % 10}" for i in range(n_rows)])
    meta_cell_idx = np.array([i % 10 for i in range(n_rows)], dtype=np.int64)

    meta_dose = np.linspace(0.1, 10.0, n_rows).astype(np.float64)
    meta_time = np.where(np.arange(n_rows) % 2 == 0, 24.0, 48.0).astype(np.float64)

    # 300 train rows, 100 test rows
    split = np.array(["train"] * 300 + ["test"] * 100)

    # Carve: 40 rows from training (rows 0..39)
    val_rows = np.arange(0, 40, dtype=np.int64)
    val_sha1 = p10_mlp.sha1_sorted_rows(val_rows)

    carves_dict = {
        "validation_carve": {
            "val_row_index": val_rows.tolist(),
            "val_rows_sha1": val_sha1,
        }
    }
    carves_bytes = json.dumps(carves_dict).encode("utf-8")
    carves_sha1 = hashlib.sha1(carves_bytes).hexdigest()

    # Fit train rows: rows 40..299 (260 rows)
    fit_rows = np.arange(40, 300, dtype=np.int64)
    train_sha1 = p10_mlp.sha1_sorted_rows(fit_rows)

    # Test rows: rows 300..399 (100 rows)
    test_rows = np.arange(300, 400, dtype=np.int64)
    test_sha1 = p10_mlp.sha1_sorted_rows(test_rows)

    bundle_dict = {
        "X": x,
        "X_ctl": x_ctl,
        "row_index": row_index,
        "meta_pert_id": meta_pert_id,
        "meta_pert_idx": meta_pert_idx,
        "meta_cell": meta_cell,
        "meta_cell_idx": meta_cell_idx,
        "meta_dose": meta_dose,
        "meta_time": meta_time,
        "split_split_cold_drug_1": split,
    }

    constants = {
        "CARVES_SHA1": carves_sha1,
        "VAL_SHA1": val_sha1,
        "EXPECT_N_TRAIN": 260,
        "EXPECT_N_VAL": 40,
        "EXPECT_N_TEST": 100,
        "EXPECT_TRAIN_SHA1": train_sha1,
        "EXPECT_TEST_SHA1": test_sha1,
    }

    return {
        "bundle": bundle_dict,
        "carves_dict": carves_dict,
        "carves_bytes": carves_bytes,
        "constants": constants,
        "d_pattern": d_pattern,
        "val_rows": val_rows,
        "fit_rows": fit_rows,
        "test_rows": test_rows,
    }


def patch_constants(monkeypatch, constants):
    for k, v in constants.items():
        monkeypatch.setattr(p10_mlp, k, v)


# =========================================================================
# Test 1: Synthetic bundle & build_h5ad
# =========================================================================
def test_synthetic_bundle_build_h5ad(monkeypatch, synthetic_bundle_data, tmp_path):
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data["constants"])

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, **data["bundle"])

    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(data["carves_bytes"])

    out_h5ad = str(tmp_path / "out.h5ad")
    p10_mlp.build_h5ad(bundle_path, carves_path, out_h5ad, with_dt=False)

    adata = ad.read_h5ad(out_h5ad)
    assert adata.shape == (400, 978)
    assert adata.X.dtype == np.float32
    assert "X_ctl" in adata.obsm
    assert adata.obsm["X_ctl"].dtype == np.float32

    obs = adata.obs
    cols = [
        "row_index",
        "pert_id",
        "cell",
        "dose",
        "time",
        "pert_idx",
        "cell_idx",
        "p10_split",
    ]
    for c in cols:
        assert c in obs.columns

    # Verify split assignments by row_index
    val_set = set(data["val_rows"])
    fit_set = set(data["fit_rows"])
    test_set = set(data["test_rows"])

    for _, row in obs.iterrows():
        ri = row["row_index"]
        sp = row["p10_split"]
        if ri in val_set:
            assert sp == "valid"
        elif ri in fit_set:
            assert sp == "train"
        elif ri in test_set:
            assert sp == "test"
        else:
            pytest.fail(f"unexpected row_index {ri}")

    assert int(np.sum(obs["p10_split"] == "train")) == 260
    assert int(np.sum(obs["p10_split"] == "valid")) == 40
    assert int(np.sum(obs["p10_split"] == "test")) == 100


# =========================================================================
# Test 2: Guards (a)-(e) each isolated
# =========================================================================
def test_guard_a_isolated(monkeypatch, synthetic_bundle_data, tmp_path):
    """Guard (a): Carves file with one byte changed fails (a) only."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data["constants"])

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, **data["bundle"])

    # Corrupt one byte of carves json
    bad_bytes = data["carves_bytes"] + b" "
    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(bad_bytes)

    out_h5ad = str(tmp_path / "out.h5ad")
    with pytest.raises(SystemExit) as exc:
        p10_mlp.build_h5ad(bundle_path, carves_path, out_h5ad)

    msg = str(exc.value)
    assert "FATAL" in msg
    assert "guard (a)" in msg
    assert "carves_json sha1" in msg


def test_guard_b_isolated(monkeypatch, synthetic_bundle_data, tmp_path):
    """Guard (b): A carve row that is a test row fails (b)."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data["constants"])

    # In bundle, mark carve row 0 as a 'test' row
    bundle_dict = dict(data["bundle"])
    split_copy = bundle_dict["split_split_cold_drug_1"].copy()
    split_copy[0] = "test"
    bundle_dict["split_split_cold_drug_1"] = split_copy

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, **bundle_dict)

    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(data["carves_bytes"])

    out_h5ad = str(tmp_path / "out.h5ad")
    with pytest.raises(SystemExit) as exc:
        p10_mlp.build_h5ad(bundle_path, carves_path, out_h5ad)

    msg = str(exc.value)
    assert "FATAL" in msg
    assert "guard (b)" in msg


def test_guard_c_isolated(monkeypatch, synthetic_bundle_data, tmp_path):
    """Guard (c): A dropped row fails (c)."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data["constants"])

    # Drop fit row 299 from the bundle (row 299 is a training row, not in carve)
    bundle_dict = {
        k: v[:399] if isinstance(v, np.ndarray) and len(v) == 400 else v
        for k, v in data["bundle"].items()
    }
    # row 399 was test; let's instead drop row 299 so fit row count drops
    keep_indices = [i for i in range(400) if i != 299]
    bundle_dict = {
        k: v[keep_indices] if isinstance(v, np.ndarray) and len(v) == 400 else v
        for k, v in data["bundle"].items()
    }

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, **bundle_dict)

    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(data["carves_bytes"])

    out_h5ad = str(tmp_path / "out.h5ad")
    with pytest.raises(SystemExit) as exc:
        p10_mlp.build_h5ad(bundle_path, carves_path, out_h5ad)

    msg = str(exc.value)
    assert "FATAL" in msg
    assert "guard (c)" in msg
    assert "split counts" in msg


def test_guard_d_isolated(monkeypatch, synthetic_bundle_data, tmp_path):
    """Guard (d): A swapped row fails (d) with (c) intact."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data["constants"])

    # Swap row_index of fit row 40 with test row 300
    bundle_dict = dict(data["bundle"])
    ri_copy = bundle_dict["row_index"].copy()
    ri_copy[40], ri_copy[300] = ri_copy[300], ri_copy[40]
    bundle_dict["row_index"] = ri_copy

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, **bundle_dict)

    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(data["carves_bytes"])

    out_h5ad = str(tmp_path / "out.h5ad")
    with pytest.raises(SystemExit) as exc:
        p10_mlp.build_h5ad(bundle_path, carves_path, out_h5ad)

    msg = str(exc.value)
    assert "FATAL" in msg
    assert "guard (d)" in msg


def test_guard_e_isolated(monkeypatch, synthetic_bundle_data, tmp_path):
    """Guard (e): Duplicated row_index fails (e)."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data["constants"])

    bundle_dict = dict(data["bundle"])
    ri_copy = bundle_dict["row_index"].copy()
    ri_copy[100] = ri_copy[101]  # duplicate row_index
    bundle_dict["row_index"] = ri_copy

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, **bundle_dict)

    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(data["carves_bytes"])

    out_h5ad = str(tmp_path / "out.h5ad")
    with pytest.raises(SystemExit) as exc:
        p10_mlp.build_h5ad(bundle_path, carves_path, out_h5ad)

    msg = str(exc.value)
    assert "FATAL" in msg
    assert "guard (e)" in msg
    assert "row_index is not unique" in msg


# =========================================================================
# Test 3: cond calculation and fit-only statistics
# =========================================================================
def test_cond_fit_only_statistics(monkeypatch, synthetic_bundle_data, tmp_path):
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data["constants"])

    bundle_path = str(tmp_path / "bundle.npz")
    np.savez(bundle_path, **data["bundle"])

    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(data["carves_bytes"])

    out_h5ad = str(tmp_path / "out_dt.h5ad")
    p10_mlp.build_h5ad(bundle_path, carves_path, out_h5ad, with_dt=True)

    adata = ad.read_h5ad(out_h5ad)
    cond = adata.obsm["cond"]
    assert cond.shape == (400, 2)
    assert cond.dtype == np.float32

    # Hand computation on fit rows
    dose = data["bundle"]["meta_dose"]
    time = data["bundle"]["meta_time"]
    fit_mask = adata.obs["p10_split"] == "train"

    ld = np.log10(np.clip(dose, 1e-4, None))
    mu_d = ld[fit_mask].mean()
    sigma_d = ld[fit_mask].std()
    exp_dose_n = ((ld - mu_d) / (sigma_d + 1e-6)).astype(np.float32)

    mu_t = time[fit_mask].mean()
    sigma_t = time[fit_mask].std()
    exp_time_n = ((time - mu_t) / (sigma_t + 1e-6)).astype(np.float32)

    assert np.allclose(cond[:, 0], exp_dose_n, atol=1e-6)
    assert np.allclose(cond[:, 1], exp_time_n, atol=1e-6)

    # Modify dose of a valid row (row 0) and a test row (row 350)
    bundle_mod = dict(data["bundle"])
    dose_mod = dose.copy()
    dose_mod[0] = 999.0
    dose_mod[350] = 777.0
    bundle_mod["meta_dose"] = dose_mod

    bundle_mod_path = str(tmp_path / "bundle_mod.npz")
    np.savez(bundle_mod_path, **bundle_mod)

    out_mod_h5ad = str(tmp_path / "out_mod_dt.h5ad")
    p10_mlp.build_h5ad(bundle_mod_path, carves_path, out_mod_h5ad, with_dt=True)

    adata_mod = ad.read_h5ad(out_mod_h5ad)
    cond_mod = adata_mod.obsm["cond"]

    # Statistics on fit rows must be unchanged!
    assert np.allclose(cond[fit_mask], cond_mod[fit_mask], atol=1e-6)


# =========================================================================
# Test 4: make_dt_script substitutions, difflib, and behaviour
# =========================================================================
def test_make_dt_script_difflib_and_behaviour(tmp_path):
    dst = str(tmp_path / "p10_train_mlp_dt.py")
    p10_mlp.make_dt_script(TRAIN_MLP_SRC, dst)

    with open(TRAIN_MLP_SRC, "r", encoding="utf-8") as f:
        src_lines = f.read().splitlines(keepends=True)
    with open(dst, "r", encoding="utf-8") as f:
        dst_lines = f.read().splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(src_lines, dst_lines, fromfile="src", tofile="dst")
    )
    plus_lines = [l for l in diff if l.startswith("+") and not l.startswith("+++")]
    minus_lines = [l for l in diff if l.startswith("-") and not l.startswith("---")]

    # Exact line diff count assertion
    assert len(plus_lines) == 16
    assert len(minus_lines) == 13
    assert len(diff) == 84

    # Header check
    assert (
        dst_lines[0].strip()
        == f"# Source sha1 {p10_mlp.sha1_file(TRAIN_MLP_SRC)}; RESULTS 98.6 C5, M+dt, reported only"
    )

    # Missing old pattern raises SystemExit
    fake_src = str(tmp_path / "fake_train.py")
    with open(fake_src, "w", encoding="utf-8") as f:
        f.write("# dummy code without target patterns\n")

    fake_dst = str(tmp_path / "fake_out.py")
    with pytest.raises(SystemExit) as exc:
        p10_mlp.make_dt_script(fake_src, fake_dst)
    assert "FATAL: make_dt_script" in str(exc.value)

    # Behaviour test of PureMLP
    ns = {}
    with open(dst, "r", encoding="utf-8") as f:
        code_str = f.read()
    exec(compile(ast.parse(code_str), dst, "exec"), ns)
    PureMLP = ns["PureMLP"]
    model = PureMLP(num_layers=3, mlp_hidden=2048)

    # First Linear layer has in_features == 980
    first_linear = model.layers[0][0]
    assert first_linear.in_features == 980
    assert model.output_fc.out_features == 978

    # Changing c changes output
    x1 = torch.randn(2, 978)
    c1 = torch.zeros(2, 2)
    c2 = torch.ones(2, 2)
    out1 = model(x1, c1)
    out2 = model(x1, c2)
    assert (out1 - out2).abs().max().item() > 0.0


# =========================================================================
# Test 5: End-to-end run on synthetic h5ad with shuffle and row-order test
# =========================================================================
def test_end_to_end_m_and_mdt_with_shuffled_row_order(
    monkeypatch, synthetic_bundle_data, tmp_path
):
    """End-to-end run of train_mlp.py and M+dt script on synthetic h5ad.

    Verifies that bundle shuffling preserves correct test row ordering in convert_outputs.
    """
    data = synthetic_bundle_data
    rng = np.random.default_rng(42)
    n_rows = 400

    # Shuffle the bundle row order
    perm = rng.permutation(n_rows)
    shuffled_bundle = {
        k: v[perm] if isinstance(v, np.ndarray) and len(v) == n_rows else v
        for k, v in data["bundle"].items()
    }

    # Re-calculate constants for shuffled bundle
    patch_constants(monkeypatch, data["constants"])

    bundle_path = str(tmp_path / "bundle_shuffled.npz")
    np.savez(bundle_path, **shuffled_bundle)

    carves_path = str(tmp_path / "carves.json")
    with open(carves_path, "wb") as f:
        f.write(data["carves_bytes"])

    # Build h5ad for M and M+dt
    h5ad_m = str(tmp_path / "m.h5ad")
    h5ad_mdt = str(tmp_path / "mdt.h5ad")
    p10_mlp.build_h5ad(bundle_path, carves_path, h5ad_m, with_dt=False)
    p10_mlp.build_h5ad(bundle_path, carves_path, h5ad_mdt, with_dt=True)

    dt_script_path = str(tmp_path / "train_mlp_dt.py")
    p10_mlp.make_dt_script(TRAIN_MLP_SRC, dt_script_path)

    # Monkeypatch os.uname and force CPU
    uname_obj = types.SimpleNamespace(nodename="test-host")
    monkeypatch.setattr(os, "uname", lambda: uname_obj, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    d_pattern = data["d_pattern"]

    # 1. Run M
    run_dir_m = str(tmp_path / "run_m")
    sys.argv = [
        "train_mlp.py",
        "--h5ad",
        h5ad_m,
        "--split_col",
        "p10_split",
        "--output_dir",
        run_dir_m,
        "--epochs",
        "2",
        "--seed",
        "131419",
    ]
    try:
        runpy.run_path(TRAIN_MLP_SRC, run_name="__main__")
    except Exception as exc:
        if "DataLoader worker" in str(exc) or "multiprocessing" in str(exc):
            pytest.skip(
                f"DataLoader with num_workers=2 failed on Windows: {exc}"
            )
        raise

    out_npz_m = str(tmp_path / "p10_m_seed0.npz")
    p10_mlp.convert_outputs(h5ad_m, run_dir_m, 131419, "m", out_npz_m)

    with np.load(out_npz_m) as z_m:
        ri_m = z_m["row_index"]
        pred_m = z_m["delta_pred"]
        assert len(ri_m) == 100
        assert pred_m.shape == (100, 978)
        assert np.isfinite(pred_m).all()
        assert int(z_m["seed"]) == 131419
        assert str(z_m["arm"]) == "m"

    # Row-order verification for M:
    gt_m = np.load(os.path.join(run_dir_m, "predictions", "test_ground_truth.npy"))
    for i in range(100):
        expected_row = ri_m[i]
        expected_truth = d_pattern[expected_row]
        assert np.allclose(gt_m[i], expected_truth, atol=1e-5)

    # 2. Run M+dt
    run_dir_mdt = str(tmp_path / "run_mdt")
    sys.argv = [
        "train_mlp_dt.py",
        "--h5ad",
        h5ad_mdt,
        "--split_col",
        "p10_split",
        "--output_dir",
        run_dir_mdt,
        "--epochs",
        "2",
        "--seed",
        "131419",
    ]
    try:
        runpy.run_path(dt_script_path, run_name="__main__")
    except Exception as exc:
        if "DataLoader worker" in str(exc) or "multiprocessing" in str(exc):
            pytest.skip(
                f"DataLoader with num_workers=2 failed on Windows: {exc}"
            )
        raise

    out_npz_mdt = str(tmp_path / "p10_mdt_seed0.npz")
    p10_mlp.convert_outputs(h5ad_mdt, run_dir_mdt, 131419, "mdt", out_npz_mdt)

    with np.load(out_npz_mdt) as z_mdt:
        ri_mdt = z_mdt["row_index"]
        pred_mdt = z_mdt["delta_pred"]
        assert len(ri_mdt) == 100
        assert pred_mdt.shape == (100, 978)
        assert np.isfinite(pred_mdt).all()
        assert int(z_mdt["seed"]) == 131419
        assert str(z_mdt["arm"]) == "mdt"

    # Row-order verification for M+dt:
    gt_mdt = np.load(
        os.path.join(run_dir_mdt, "predictions", "test_ground_truth.npy")
    )
    for i in range(100):
        expected_row = ri_mdt[i]
        expected_truth = d_pattern[expected_row]
        assert np.allclose(gt_mdt[i], expected_truth, atol=1e-5)


# =========================================================================
# Test 6: Kernel builder, metadata, AST parse, static blinding check
# =========================================================================
def test_kernel_builder_and_blinding(tmp_path):
    out_root = str(tmp_path / "kernels")
    make_p10_m_kernels.build_kernels(out_root)

    v9p9_meta_path = os.path.join(
        REPO_ROOT,
        "external",
        "kaggle_kernels",
        "kern_v9p9",
        "kernel-metadata.json",
    )
    with open(v9p9_meta_path, "r", encoding="utf-8") as f:
        v9p9_docker = json.load(f)["docker_image"]

    expected_folders = [
        ("m", 0, 131419),
        ("m", 1, 131420),
        ("m", 2, 131421),
        ("mdt", 0, 131419),
        ("mdt", 1, 131420),
        ("mdt", 2, 131421),
    ]

    for arm, k, seed in expected_folders:
        folder_name = f"kern_p10_{arm}_s{k}"
        folder_path = os.path.join(out_root, folder_name)
        assert os.path.isdir(folder_path)

        meta_path = os.path.join(folder_path, "kernel-metadata.json")
        assert os.path.isfile(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["id"] == f"apexblue/lincs-p10-{arm}-s{k}"
        assert meta["is_private"] is True
        assert meta["enable_gpu"] is False
        assert meta["enable_internet"] is False
        assert meta["docker_image"] == v9p9_docker
        assert meta["dataset_sources"] == [
            "apexblue/xpert-mdmt-benchmark",
            "apexblue/p10-wheels",
            "apexblue/lincs-p10-src",
        ]
        assert meta["code_file"] == f"lincs-p10-{arm}-s{k}.py"

        script_path = os.path.join(folder_path, meta["code_file"])
        assert os.path.isfile(script_path)
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        # AST parse check
        ast.parse(code)

        # Seed check
        assert f"SEED = {seed}" in code

        # Static blinding check:
        # 1. Never writes per_drug_pcc or test_ground_truth to /kaggle/working
        assert "per_drug_pcc" not in code
        assert "test_ground_truth" not in code

        # 2. Never passes trainer stdout directly, only prints allowed line prefixes
        assert "stdout=f_log" in code
        assert "stderr=subprocess.STDOUT" in code
        assert (
            'line.startswith("Epoch ") or line.startswith("Early stopping") or line.startswith("Best epoch")'
            in code
        )


# =========================================================================
# PI additions after the mutation check (15 of 27 mutants survived W36's suite). Each test isolates ONE decision.
# =========================================================================
def _write_inputs(tmp_path, bundle, carves_bytes, name='b'):
    bp = str(tmp_path / (name + '.npz'))
    np.savez(bp, **bundle)
    cp = str(tmp_path / (name + '_carves.json'))
    with open(cp, 'wb') as f:
        f.write(carves_bytes)
    return bp, cp


def test_pi_guard_b_sha1_isolated(monkeypatch, synthetic_bundle_data, tmp_path):
    """Guard (b)'s sha1 half alone: a different carve (still training rows, and guard (a) re-pinned to the new file) fails (b)."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data['constants'])
    carves = json.loads(data['carves_bytes'].decode('utf-8'))
    carves['validation_carve']['val_row_index'] = list(range(1, 41))        # rows 1..40: all training rows, a different set
    cb = json.dumps(carves).encode('utf-8')
    monkeypatch.setattr(p10_mlp, 'CARVES_SHA1', hashlib.sha1(cb).hexdigest())  # (a) passes
    bp, cp = _write_inputs(tmp_path, data['bundle'], cb)
    with pytest.raises(SystemExit) as exc:
        p10_mlp.build_h5ad(bp, cp, str(tmp_path / 'o.h5ad'))
    assert 'guard (b): carve sorted row_index sha1' in str(exc.value)


@pytest.mark.parametrize('which, const', [('train', 'EXPECT_TRAIN_SHA1'), ('test', 'EXPECT_TEST_SHA1')])
def test_pi_guard_d_each_half_isolated(monkeypatch, synthetic_bundle_data, tmp_path, which, const):
    """Guard (d)'s train and test halves each alone: only that expected sha1 is wrong; counts and the other half are right."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data['constants'])
    monkeypatch.setattr(p10_mlp, const, '0' * 40)
    bp, cp = _write_inputs(tmp_path, data['bundle'], data['carves_bytes'])
    with pytest.raises(SystemExit) as exc:
        p10_mlp.build_h5ad(bp, cp, str(tmp_path / 'o.h5ad'))
    assert 'guard (d): %s sorted row_index sha1' % which in str(exc.value)


def test_pi_carves_must_be_a_file(monkeypatch, synthetic_bundle_data, tmp_path):
    """Guard (a) cannot be bypassed: the carves must be a file path (the dict branch was removed by the PI)."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data['constants'])
    bp, _ = _write_inputs(tmp_path, data['bundle'], data['carves_bytes'])
    with pytest.raises(SystemExit):
        p10_mlp.build_h5ad(bp, data['carves_dict'], str(tmp_path / 'o.h5ad'))


def test_pi_cond_dose_clip(monkeypatch, synthetic_bundle_data, tmp_path):
    """The 1e-4 dose clip (xpert_arm.py's convention) matters: fit rows with dose 0 and 5e-4 must be clipped at 1e-4 exactly."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data['constants'])
    b = dict(data['bundle'])
    dose = b['meta_dose'].copy()
    dose[50], dose[51] = 0.0, 5e-4                                          # fit rows (40..299)
    b['meta_dose'] = dose
    bp, cp = _write_inputs(tmp_path, b, data['carves_bytes'])
    out = str(tmp_path / 'o.h5ad')
    p10_mlp.build_h5ad(bp, cp, out, with_dt=True)
    cond = ad.read_h5ad(out).obsm['cond']
    fit = np.zeros(400, bool)
    fit[40:300] = True
    ld = np.log10(np.maximum(dose, 1e-4))
    want = (ld - ld[fit].mean()) / (ld[fit].std() + 1e-6)
    assert np.allclose(cond[:, 0], want, atol=1e-5)
    assert not np.allclose(cond[51, 0], cond[50, 0])                        # 5e-4 > 1e-4 is not clipped


def _fake_run(tmp_path, n, bad=None):
    run = tmp_path / 'run'
    (run / 'predictions').mkdir(parents=True)
    (run / 'logs').mkdir(parents=True)
    pred = np.zeros((n, 978), np.float32)
    if bad == 'nan':
        pred[3, 7] = np.nan
    np.save(run / 'predictions' / 'test_predictions.npy', pred)
    with open(run / 'logs' / 'resource_log.json', 'w') as f:
        json.dump({'best_epoch': 1, 'n_epochs_run': 2, 'n_test': n}, f)
    return str(run)


@pytest.mark.parametrize('bad, n, msg', [('nan', 100, 'non-finite'), (None, 99, 'prediction count')])
def test_pi_convert_guards_isolated(monkeypatch, synthetic_bundle_data, tmp_path, bad, n, msg):
    """convert_outputs' finite guard and count guard, each alone."""
    data = synthetic_bundle_data
    patch_constants(monkeypatch, data['constants'])
    bp, cp = _write_inputs(tmp_path, data['bundle'], data['carves_bytes'])
    h = str(tmp_path / 'o.h5ad')
    p10_mlp.build_h5ad(bp, cp, h)
    run = _fake_run(tmp_path, n, bad)
    with pytest.raises(SystemExit) as exc:
        p10_mlp.convert_outputs(h, run, 131419, 'm', str(tmp_path / 'out.npz'))
    assert msg in str(exc.value)


def test_pi_dt_uses_c_in_eval_mode(tmp_path):
    """The M+dt network reads c: in eval mode (dropout off) changing c changes the output, and equal c gives equal output.
    W36's version ran in train mode, where dropout alone made the outputs differ."""
    dst = str(tmp_path / 'dt.py')
    p10_mlp.make_dt_script(TRAIN_MLP_SRC, dst)
    ns = {}
    exec(compile(open(dst, encoding='utf-8').read(), dst, 'exec'), ns)
    torch.manual_seed(0)
    model = ns['PureMLP'](num_layers=3, mlp_hidden=64).eval()
    x1 = torch.randn(4, 978)
    with torch.no_grad():
        a = model(x1, torch.zeros(4, 2))
        b = model(x1, torch.zeros(4, 2))
        c = model(x1, torch.ones(4, 2))
    assert torch.equal(a, b)
    assert (a - c).abs().max().item() > 1e-6


def test_pi_dt_refuses_a_doubled_pattern(tmp_path):
    """Each substitution must match EXACTLY once: a source with one pattern duplicated is refused."""
    src = open(TRAIN_MLP_SRC, encoding='utf-8').read()
    pat = "    def forward(self, x1):\n        h = x1"
    assert src.count(pat) == 1
    bad = str(tmp_path / 'doubled.py')
    with open(bad, 'w', encoding='utf-8', newline='\n') as f:
        f.write(src + '\n\nclass Extra:\n' + pat + '\n        return h\n')
    with pytest.raises(SystemExit) as exc:
        p10_mlp.make_dt_script(bad, str(tmp_path / 'o.py'))
    assert 'found 2 occurrences' in str(exc.value)


def test_pi_kernel_template_pins(tmp_path):
    """The generated kernels' load-bearing lines, pinned exactly (the template is not run here; each check kills one mutant)."""
    out_root = str(tmp_path / 'k')
    make_p10_m_kernels.build_kernels(out_root)
    for arm in ('m', 'mdt'):
        for k in range(3):
            code = open(os.path.join(out_root, 'kern_p10_%s_s%d' % (arm, k), 'lincs-p10-%s-s%d.py' % (arm, k)),
                        encoding='utf-8').read()
            tree = ast.parse(code)
            assert "ARM = '%s'" % arm in code and 'K = %d' % k in code
            assert '"--epochs", "500",' in code                                  # the published 500-epoch cap
            assert '    trainer_script = dt_script_path\n' in code               # M+dt trains with the generated script
            assert '    trainer_script = train_mlp_path\n' in code               # M with theirs, unchanged
            assert 'with_dt=(ARM == "mdt"))' in code                             # cond exists exactly for M+dt
            assert 'if got_dt != DT_SCRIPT_SHA1:' in code
            for name, var in (('bundle', 'bundle_path, BUNDLE_SHA1'), ('train_mlp', 'train_mlp_path, TRAIN_MLP_SHA1'),
                              ('p10_mlp', 'p10_mlp_path, P10_MLP_SHA1'), ('carves', 'carves_path, CARVES_SHA1')):
                assert '("%s", %s),' % (name, var) in code                      # every mounted file is guarded
            assert code.count('line.startswith(') == 3                           # exactly the three allowed prefixes
            assert 'f"p10_{ARM}_seed{K}.npz"' in code and 'f"p10_{ARM}_seed{K}_record.json"' in code
            assert '/kaggle/working' in code and code.count('working_dir, ') == 4   # npz, ckpt, rlog, record: nothing else
            lit = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            assert len([v for v in lit if len(v) == 40 and all(ch in '0123456789abcdef' for ch in v)]) == 5   # five sha1 pins


def test_pi_kernel_guard3_trainer_split_counts(tmp_path):
    """Review 075 C1: before anything is echoed, each kernel refuses unless the TRAINER's own log shows the registered split
    counts and no fallback to the test set (their main() validates on test when 'valid' is empty)."""
    out_root = str(tmp_path / 'k')
    make_p10_m_kernels.build_kernels(out_root)
    for arm in ('m', 'mdt'):
        code = open(os.path.join(out_root, 'kern_p10_%s_s0' % arm, 'lincs-p10-%s-s0.py' % arm), encoding='utf-8').read()
        for need in ('"  train: 49846 samples"', '"  valid: 5539 samples"', '"  test: 13445 samples"', '"No valid set found"'):
            assert need in code
        g, e = code.index('GUARD 3 (review 075 C1)'), code.index('=== Validation curve (blinded test) ===')
        assert g < e                                                         # the guard runs before the echo
    # their trainer prints exactly these lines (main(): print(f"  {split_value}: {len(self.x1)} samples"))
    src = open(TRAIN_MLP_SRC, encoding='utf-8').read()
    assert 'print(f"  {split_value}: {len(self.x1)} samples")' in src and 'No valid set found' in src
