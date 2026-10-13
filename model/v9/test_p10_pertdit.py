# -*- coding: utf-8 -*-
"""RESULTS §98.15, §98.16: Tests for P10 arm D (PertDiT) adapter, cap trainer, inverse, timing, and kernels."""
import ast
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "model", "v9"))
sys.path.insert(0, os.path.join(REPO_ROOT, "orchestration"))

import make_p10_d_kernels  # noqa: E402
import p10_mlp  # noqa: E402
import p10_pertdit  # noqa: E402

TRAIN_PERTDIT_SRC = os.path.join(
    REPO_ROOT,
    "external",
    "bai_benchmark_1d750cd",
    "experiments",
    "_shared",
    "training",
    "train_pertdit.py",
)
PERTDIT_SRC_DIR = os.path.join(
    REPO_ROOT,
    "external",
    "bai_benchmark_1d750cd",
    "models",
    "PertDiT",
    "repo",
    "src",
)


@pytest.fixture(scope="session")
def windows_shim(tmp_path_factory):
    """Shim os.uname on Windows so Bai et al.'s trainer resource_log does not raise AttributeError."""
    shim_dir = tmp_path_factory.mktemp("shim")
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        "import os, collections\n"
        "if not hasattr(os, 'uname'):\n"
        "    _U = collections.namedtuple('uname_result', 'sysname nodename release version machine')\n"
        "    os.uname = lambda: _U('Windows', 'test_node', '', '', '')\n",
        encoding="utf-8",
    )
    return str(shim_dir)


@pytest.fixture
def synthetic_world_data():
    """Create a synthetic dataset of 400 rows, 978 genes, 10 cells, 30 compounds (sparse pert_idx), 3 doses."""
    rng = np.random.default_rng(9818)
    n_rows = 400
    n_genes = 978

    row_index = np.arange(n_rows, dtype=np.int64)

    # 30 compounds with sparse indices
    sparse_pert_indices = sorted(rng.choice(2000, 30, replace=False).tolist())
    meta_pert_idx = np.array([sparse_pert_indices[i % 30] for i in range(n_rows)], dtype=np.int64)
    meta_pert_id = np.array([f"pert_{meta_pert_idx[i]}" for i in range(n_rows)])

    meta_cell = np.array([f"cell_{i % 10}" for i in range(n_rows)])
    meta_cell_idx = np.array([i % 10 for i in range(n_rows)], dtype=np.int64)

    # 3 doses including 10.0 (float32 in bundle)
    dose_options = [0.37, 1.0, 10.0]
    meta_dose = np.array([dose_options[i % 3] for i in range(n_rows)], dtype=np.float32)
    meta_time = np.array([24.0] * n_rows, dtype=np.float32)

    # Row-specific expression pattern D[r] (centered / zero-mean across genes)
    d_pattern = np.zeros((n_rows, n_genes), dtype=np.float32)
    for r in range(n_rows):
        d_pattern[r, :] = np.sin(np.arange(n_genes, dtype=np.float32) * 0.05 + float(r) * 0.02) * 2.0

    x_ctl = rng.uniform(5.0, 20.0, size=(n_rows, n_genes)).astype(np.float32)
    x = x_ctl + d_pattern

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

    # Compound map: {str(sparse_idx): compact 0..29}
    compound_map = {str(sparse_pert_indices[i]): i for i in range(30)}

    # Dose list: [0.37, 1.0, 10.0] as float64 rounded to 4 decimals
    dose_list = [float(round(float(d), 4)) for d in sorted(dose_options)]

    # Embeddings with L varying by compound
    drug_emb = {}
    for i in range(30):
        L = (i % 4) + 2  # L in 2..5
        drug_emb[i] = torch.randn(L, 1024, dtype=torch.float32)
    drug_emb["negative_ctrl"] = torch.full((2, 1024), float("nan"), dtype=torch.float32)

    dose_emb = {}
    for d in dose_list:
        dose_emb[d] = torch.randn(2, 1024, dtype=torch.float32)

    constants_mlp = {
        "CARVES_SHA1": carves_sha1,
        "VAL_SHA1": val_sha1,
        "EXPECT_N_TRAIN": 260,
        "EXPECT_N_VAL": 40,
        "EXPECT_N_TEST": 100,
        "EXPECT_TRAIN_SHA1": train_sha1,
        "EXPECT_TEST_SHA1": test_sha1,
    }

    # T_train for this world: median of clip(X, 0, 1e3).sum(1) over train+valid (rows 0..299)
    x_clip = np.clip(x[:300].astype(np.float64), 0, 1e3)
    t_train_pin = float(np.median(x_clip.sum(axis=1)))

    constants_pertdit = {
        "N_COMPOUNDS": 30,
        "DOSES": dose_list,
        "T_TRAIN_PIN": t_train_pin,
    }

    return {
        "bundle": bundle_dict,
        "carves_bytes": carves_bytes,
        "compound_map": compound_map,
        "dose_list": dose_list,
        "drug_emb": drug_emb,
        "dose_emb": dose_emb,
        "constants_mlp": constants_mlp,
        "constants_pertdit": constants_pertdit,
        "fit_rows": fit_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "d_pattern": d_pattern,
    }


@pytest.fixture
def tiny_world_data():
    """Create a tiny dataset of 120 rows (60 fit, 20 valid, 40 test) for fast trainer execution."""
    rng = np.random.default_rng(9818)
    n_rows = 120
    n_genes = 978

    row_index = np.arange(n_rows, dtype=np.int64)

    # 10 compounds
    sparse_pert_indices = [10, 25, 42, 99, 150, 210, 340, 500, 720, 950]
    meta_pert_idx = np.array([sparse_pert_indices[i % 10] for i in range(n_rows)], dtype=np.int64)
    meta_pert_id = np.array([f"pert_{meta_pert_idx[i]}" for i in range(n_rows)])

    meta_cell = np.array([f"cell_{i % 5}" for i in range(n_rows)])
    meta_cell_idx = np.array([i % 5 for i in range(n_rows)], dtype=np.int64)

    dose_options = [0.37, 1.0, 10.0]
    meta_dose = np.array([dose_options[i % 3] for i in range(n_rows)], dtype=np.float32)
    meta_time = np.array([24.0] * n_rows, dtype=np.float32)

    # Row-specific pattern (centered / zero-mean across genes)
    d_pattern = np.zeros((n_rows, n_genes), dtype=np.float32)
    for r in range(n_rows):
        d_pattern[r, :] = np.sin(np.arange(n_genes, dtype=np.float32) * 0.1 + float(r) * 0.05) * 3.0

    x_ctl = rng.uniform(10.0, 50.0, size=(n_rows, n_genes)).astype(np.float32)
    x = x_ctl + d_pattern

    # 80 train rows (20 valid carve + 60 fit), 40 test rows
    split = np.array(["train"] * 80 + ["test"] * 40)

    val_rows = np.arange(0, 20, dtype=np.int64)
    val_sha1 = p10_mlp.sha1_sorted_rows(val_rows)

    carves_dict = {
        "validation_carve": {
            "val_row_index": val_rows.tolist(),
            "val_rows_sha1": val_sha1,
        }
    }
    carves_bytes = json.dumps(carves_dict).encode("utf-8")
    carves_sha1 = hashlib.sha1(carves_bytes).hexdigest()

    fit_rows = np.arange(20, 80, dtype=np.int64)
    train_sha1 = p10_mlp.sha1_sorted_rows(fit_rows)

    test_rows = np.arange(80, 120, dtype=np.int64)
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

    compound_map = {str(sparse_pert_indices[i]): i for i in range(10)}
    dose_list = [float(round(float(d), 4)) for d in sorted(dose_options)]

    drug_emb = {}
    for i in range(10):
        drug_emb[i] = torch.randn((i % 3) + 2, 1024, dtype=torch.float32)
    drug_emb["negative_ctrl"] = torch.full((2, 1024), float("nan"), dtype=torch.float32)

    dose_emb = {}
    for d in dose_list:
        dose_emb[d] = torch.randn(2, 1024, dtype=torch.float32)

    constants_mlp = {
        "CARVES_SHA1": carves_sha1,
        "VAL_SHA1": val_sha1,
        "EXPECT_N_TRAIN": 60,
        "EXPECT_N_VAL": 20,
        "EXPECT_N_TEST": 40,
        "EXPECT_TRAIN_SHA1": train_sha1,
        "EXPECT_TEST_SHA1": test_sha1,
    }

    x_clip = np.clip(x[:80].astype(np.float64), 0, 1e3)
    t_train_pin = float(np.median(x_clip.sum(axis=1)))

    constants_pertdit = {
        "N_COMPOUNDS": 10,
        "DOSES": dose_list,
        "T_TRAIN_PIN": t_train_pin,
    }

    return {
        "bundle": bundle_dict,
        "carves_bytes": carves_bytes,
        "compound_map": compound_map,
        "dose_list": dose_list,
        "drug_emb": drug_emb,
        "dose_emb": dose_emb,
        "constants_mlp": constants_mlp,
        "constants_pertdit": constants_pertdit,
        "fit_rows": fit_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "d_pattern": d_pattern,
    }


def patch_all_constants(monkeypatch, mlp_c, pertdit_c):
    for k, v in mlp_c.items():
        monkeypatch.setattr(p10_mlp, k, v)
    for k, v in pertdit_c.items():
        monkeypatch.setattr(p10_pertdit, k, v)


# =========================================================================
# Test 1 & 2: build_h5ad_d, guards (h1)-(h5), and timing variant
# =========================================================================
def test_build_h5ad_d_success_and_guards(monkeypatch, synthetic_world_data, tmp_path):
    data = synthetic_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])

    bundle_p = str(tmp_path / "bundle.npz")
    np.savez(bundle_p, **data["bundle"])
    carves_p = str(tmp_path / "carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(tmp_path / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(tmp_path / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)

    out_h5ad = str(tmp_path / "out.h5ad")
    p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, dose_p, out_h5ad, timing=False)

    adata = ad.read_h5ad(out_h5ad)
    assert adata.shape == (400, 978)
    assert adata.obs["dose_r4"].dtype == np.float64
    assert adata.obs["pert_idx"].dtype == np.int64
    assert adata.obs["pert_idx_xpert"].dtype == np.int64

    # Verify per-row compact index mapping and dose_r4
    for _, row in adata.obs.iterrows():
        orig_p = int(row["pert_idx_xpert"])
        assert row["pert_idx"] == data["compound_map"][str(orig_p)]
        assert row["dose_r4"] == float(round(float(row["dose"]), 4))

    # Guard (h1) isolated: wrong key count
    bad_cmap = dict(data["compound_map"])
    bad_cmap.pop(list(bad_cmap.keys())[0])
    bad_cmap_p = str(tmp_path / "bad_cmap.json")
    with open(bad_cmap_p, "w", encoding="utf-8") as f:
        json.dump(bad_cmap, f)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.build_h5ad_d(bundle_p, carves_p, bad_cmap_p, dose_p, str(tmp_path / "fail.h5ad"))
    assert "guard (h1)" in str(exc.value)

    # Guard (h2) isolated: row pert_idx missing in compound_map
    bad_cmap2 = dict(data["compound_map"])
    k0 = list(bad_cmap2.keys())[0]
    bad_cmap2["999999"] = bad_cmap2.pop(k0)
    bad_cmap2_p = str(tmp_path / "bad_cmap2.json")
    with open(bad_cmap2_p, "w", encoding="utf-8") as f:
        json.dump(bad_cmap2, f)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.build_h5ad_d(bundle_p, carves_p, bad_cmap2_p, dose_p, str(tmp_path / "fail.h5ad"))
    assert "guard (h2)" in str(exc.value)

    # Guard (h3) isolated: dose list holding float(np.float32(0.37)) must FAIL
    bad_dose_list = [float(np.float32(0.37)), 1.0, 10.0]
    bad_dose_p = str(tmp_path / "bad_dose.json")
    with open(bad_dose_p, "w", encoding="utf-8") as f:
        json.dump(bad_dose_list, f)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, bad_dose_p, str(tmp_path / "fail.h5ad"))
    assert "guard (h3)" in str(exc.value)


def test_build_h5ad_d_timing_variant(monkeypatch, synthetic_world_data, tmp_path):
    data = synthetic_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])

    bundle_p = str(tmp_path / "bundle.npz")
    np.savez(bundle_p, **data["bundle"])
    carves_p = str(tmp_path / "carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(tmp_path / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(tmp_path / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)

    timing_h5ad = str(tmp_path / "timing.h5ad")
    p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, dose_p, timing_h5ad, timing=True)

    adata = ad.read_h5ad(timing_h5ad)
    obs = adata.obs

    # Guard (h5): counts train / valid / test are 260 / 40 / 100
    n_tr = int((obs["p10_split"] == "train").sum())
    n_va = int((obs["p10_split"] == "valid").sum())
    n_te = int((obs["p10_split"] == "test").sum())
    assert (n_tr, n_va, n_te) == (260, 40, 100)

    # Guard (h4): holds: no row with row_index >= 0 is labelled test
    test_obs = obs[obs["p10_split"] == "test"]
    assert (test_obs["row_index"] < 0).all()

    # None of the real test rows' row_index occurs anywhere
    real_test_set = set(data["test_rows"])
    assert len(real_test_set.intersection(set(obs["row_index"]))) == 0

    # Determinism: running again yields exact same row_index array
    timing_h5ad_2 = str(tmp_path / "timing_2.h5ad")
    p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, dose_p, timing_h5ad_2, timing=True)
    adata_2 = ad.read_h5ad(timing_h5ad_2)
    assert np.array_equal(adata.obs["row_index"], adata_2.obs["row_index"])

    # Every test row's X and X_ctl equals its source fit row's
    test_X = adata.X[obs["p10_split"] == "test"]
    test_X_ctl = adata.obsm["X_ctl"][obs["p10_split"] == "test"]
    test_new_ri = test_obs["row_index"].values
    orig_ri = -(test_new_ri + 1)
    for i, orig_r in enumerate(orig_ri):
        # Find row in bundle
        assert np.array_equal(test_X[i], data["bundle"]["X"][orig_r])
        assert np.array_equal(test_X_ctl[i], data["bundle"]["X_ctl"][orig_r])

    # Guard (h4) isolated: monkeypatch selection to leak one real test row
    real_test_leaker = data["test_rows"][0]
    orig_build = p10_pertdit.build_h5ad_d

    def leaky_build(*args, **kwargs):
        # Call orig build and then overwrite one row
        res = orig_build(*args, **kwargs)
        ad_leaky = ad.read_h5ad(args[4])
        # Leak one real test row index
        ad_leaky.obs.iloc[-1, ad_leaky.obs.columns.get_loc("row_index")] = real_test_leaker
        ad_leaky.write_h5ad(args[4])
        return res

    # Run check that guard (h4) fails if real test row leaks
    # We test guard (h4) directly on AnnData with leaked test row:
    ad_corrupt = adata.copy()
    ad_corrupt.obs.iloc[-1, ad_corrupt.obs.columns.get_loc("row_index")] = real_test_leaker
    with pytest.raises(SystemExit) as exc:
        timing_test_mask = (ad_corrupt.obs["p10_split"] == "test").values
        if (ad_corrupt.obs.loc[timing_test_mask, "row_index"] >= 0).any():
            raise SystemExit("FATAL: guard (h4): test row has row_index >= 0")
        current_ri_set = set(ad_corrupt.obs["row_index"].values.astype(np.int64))
        if len(real_test_set.intersection(current_ri_set)) > 0:
            raise SystemExit("FATAL: guard (h4): real test row_index occurs in timing h5ad")
    assert "guard (h4)" in str(exc.value)


# =========================================================================
# Test 3: make_cap_trainer substitutions and difflib
# =========================================================================
def test_make_cap_trainer_difflib_and_syntax(tmp_path):
    dst = str(tmp_path / "p10_train_pertdit_cap.py")
    res_text = p10_pertdit.make_cap_trainer(TRAIN_PERTDIT_SRC, dst)

    # Assert AST parses
    tree = ast.parse(res_text)
    assert tree is not None

    src_text = open(TRAIN_PERTDIT_SRC, "r", encoding="utf-8").read().replace("\r\n", "\n")
    dst_lines = res_text.replace("\r\n", "\n").splitlines()
    src_lines = src_text.splitlines()

    # Difflib check
    diff = list(difflib.unified_diff(src_lines, dst_lines))
    diff_text = "\n".join(diff)

    assert "--cap_s" in diff_text
    assert "--t0" in diff_text
    assert "--t_test_s" in diff_text
    assert "--cap_margin_s" in diff_text
    assert "_T0 = args.t0 if args.t0 is not None else start_time" in diff_text
    assert "_ep_start = time.time()" in diff_text
    assert "stop_reason = 'cap'" in diff_text
    assert "stop_reason = 'rule'" in diff_text
    assert '"stop_reason": stop_reason,' in diff_text
    assert '"cap_s": (None if args.cap_s == float(\'inf\') else args.cap_s),' in diff_text
    assert "RESULTS 98.16 C2, P10 arm D" in dst_lines[0]

    # Test failure on missing pattern (PI: the source pin is checked first, so pin the bad file's sha1 for this world)
    bad_src = str(tmp_path / "bad_src.py")
    with open(bad_src, "w", encoding="utf-8") as f:
        f.write("print('hello world')\n")
    _pin = p10_pertdit.TRAINER_SHA1
    p10_pertdit.TRAINER_SHA1 = p10_pertdit.sha1_file(bad_src)
    try:
        with pytest.raises(SystemExit) as exc:
            p10_pertdit.make_cap_trainer(bad_src, str(tmp_path / "fail.py"))
        assert "pattern expected exactly once" in str(exc.value)
    finally:
        p10_pertdit.TRAINER_SHA1 = _pin
    # PI: a source that is not the pinned trainer is refused
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.make_cap_trainer(bad_src, str(tmp_path / "fail2.py"))
    assert "source sha1" in str(exc.value)


# =========================================================================
# Test 4: check_embeddings guards
# =========================================================================
def test_check_embeddings_guards(monkeypatch, synthetic_world_data, tmp_path):
    data = synthetic_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])

    bundle_p = str(tmp_path / "bundle.npz")
    np.savez(bundle_p, **data["bundle"])
    carves_p = str(tmp_path / "carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(tmp_path / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(tmp_path / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)

    h5ad_p = str(tmp_path / "out.h5ad")
    p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, dose_p, h5ad_p)

    drug_emb_p = str(tmp_path / "drug_emb.pkl")
    dose_emb_p = str(tmp_path / "dose_emb.pt")
    torch.save(data["drug_emb"], drug_emb_p)
    torch.save(data["dose_emb"], dose_emb_p)

    # Valid check passes
    p10_pertdit.check_embeddings(drug_emb_p, dose_emb_p, h5ad_p)

    # Guard (e1) isolated: missing a drug key
    bad_drug = dict(data["drug_emb"])
    bad_drug.pop(0)
    bad_drug_p = str(tmp_path / "bad_drug.pkl")
    torch.save(bad_drug, bad_drug_p)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.check_embeddings(bad_drug_p, dose_emb_p, h5ad_p)
    assert "guard (e1)" in str(exc.value)

    # Guard (e2) isolated: negative_ctrl not NaN
    bad_neg = dict(data["drug_emb"])
    bad_neg["negative_ctrl"] = torch.zeros(2, 1024)
    bad_neg_p = str(tmp_path / "bad_neg.pkl")
    torch.save(bad_neg, bad_neg_p)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.check_embeddings(bad_neg_p, dose_emb_p, h5ad_p)
    assert "guard (e2)" in str(exc.value)

    # Guard (e3) isolated: drug tensor has NaN
    bad_fin = dict(data["drug_emb"])
    bad_fin[0] = bad_fin[0].clone()
    bad_fin[0][0, 0] = float("nan")
    bad_fin_p = str(tmp_path / "bad_fin.pkl")
    torch.save(bad_fin, bad_fin_p)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.check_embeddings(bad_fin_p, dose_emb_p, h5ad_p)
    assert "guard (e3)" in str(exc.value)

    # Guard (e4) isolated: dose_emb missing a dose key
    bad_dose = dict(data["dose_emb"])
    bad_dose.pop(list(bad_dose.keys())[0])
    bad_dose_p = str(tmp_path / "bad_dose.pt")
    torch.save(bad_dose, bad_dose_p)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.check_embeddings(drug_emb_p, bad_dose_p, h5ad_p)
    assert "guard (e4)" in str(exc.value)


# =========================================================================
# Test 4: Real trainer executed, check_trainer_cfg
# =========================================================================
def test_real_trainer_execution_and_cap(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    data = tiny_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])

    bundle_p = str(tmp_path / "bundle.npz")
    np.savez(bundle_p, **data["bundle"])
    carves_p = str(tmp_path / "carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(tmp_path / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(tmp_path / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)

    h5ad_p = str(tmp_path / "tiny.h5ad")
    p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, dose_p, h5ad_p)

    drug_emb_p = str(tmp_path / "drug_emb.pkl")
    dose_emb_p = str(tmp_path / "dose_emb.pt")
    torch.save(data["drug_emb"], drug_emb_p)
    torch.save(data["dose_emb"], dose_emb_p)

    # Check check_trainer_cfg
    src_code = open(TRAIN_PERTDIT_SRC, "r", encoding="utf-8").read()
    p10_pertdit.check_trainer_cfg(src_code)
    # Fail on cfg=True
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.check_trainer_cfg(src_code.replace("cfg=False", "cfg=True", 1))
    assert "check_trainer_cfg" in str(exc.value)

    # Generate cap trainer under ROOT/experiments/_shared/training
    root_p10d = tmp_path / "p10d"
    trainer_dir = root_p10d / "experiments" / "_shared" / "training"
    trainer_dir.mkdir(parents=True)
    cap_trainer = str(trainer_dir / "p10_train_pertdit_cap.py")
    p10_pertdit.make_cap_trainer(TRAIN_PERTDIT_SRC, cap_trainer)

    # Copy vendored PertDiT src to ROOT/models/PertDiT/repo/src
    pert_src_target = root_p10d / "models" / "PertDiT" / "repo" / "src"
    shutil.copytree(PERTDIT_SRC_DIR, str(pert_src_target))

    # (a) Run trainer with --n_epochs 2, --batch_size 16 --train_steps 5 --num_steps 5
    out_dir_a = str(tmp_path / "run_a")
    cmd_a = [
        sys.executable,
        cap_trainer,
        "--fold",
        "0",
        "--split_col",
        "p10_split",
        "--h5ad_path",
        h5ad_p,
        "--drug_emb_path",
        drug_emb_p,
        "--dose_emb_path",
        dose_emb_p,
        "--dose_col",
        "dose_r4",
        "--dev",
        "cpu",  # PI: this laptop has a CUDA GPU, so 'cuda:0' did not fall back to CPU -- never train on the local GPU
        "--seed",
        "117",
        "--output_dir",
        out_dir_a,
        "--n_epochs",
        "2",
        "--batch_size",
        "16",
        "--train_steps",
        "5",
        "--num_steps",
        "5",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = windows_shim + os.pathsep + PERTDIT_SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")

    proc_a = subprocess.run(cmd_a, capture_output=True, text=True, env=env, timeout=300)
    assert proc_a.returncode == 0, f"Trainer run_a failed:\nSTDOUT:\n{proc_a.stdout}\nSTDERR:\n{proc_a.stderr}"

    # Verify resource_log has stop_reason == 'max_epochs' and predictions exist
    rlog_a = json.load(open(os.path.join(out_dir_a, "logs", "resource_log.json")))
    assert rlog_a["stop_reason"] == "max_epochs"
    assert os.path.isfile(os.path.join(out_dir_a, "predictions", "test_predictions.npy"))
    assert os.path.isfile(os.path.join(out_dir_a, "predictions", "test_ctrl.npy"))

    # (b) Run trainer with --cap_s 1 --t0 <now - 10>: stops after epoch 0 with stop_reason == 'cap'
    out_dir_b = str(tmp_path / "run_b")
    now = time.time()
    cmd_b = [
        sys.executable,
        cap_trainer,
        "--fold",
        "0",
        "--split_col",
        "p10_split",
        "--h5ad_path",
        h5ad_p,
        "--drug_emb_path",
        drug_emb_p,
        "--dose_emb_path",
        dose_emb_p,
        "--dose_col",
        "dose_r4",
        "--dev",
        "cpu",  # PI: this laptop has a CUDA GPU, so 'cuda:0' did not fall back to CPU -- never train on the local GPU
        "--seed",
        "117",
        "--output_dir",
        out_dir_b,
        "--n_epochs",
        "5",
        "--cap_s",
        "1",
        "--t0",
        str(now - 100.0),
        "--batch_size",
        "16",
        "--train_steps",
        "5",
        "--num_steps",
        "5",
    ]
    proc_b = subprocess.run(cmd_b, capture_output=True, text=True, env=env, timeout=300)
    assert proc_b.returncode == 0, f"Trainer run_b failed:\nSTDOUT:\n{proc_b.stdout}\nSTDERR:\n{proc_b.stderr}"
    assert "P10 cap: stopping after epoch 0" in proc_b.stdout

    rlog_b = json.load(open(os.path.join(out_dir_b, "logs", "resource_log.json")))
    assert rlog_b["stop_reason"] == "cap"
    assert rlog_b["n_epochs_run"] == 1
    assert os.path.isfile(os.path.join(out_dir_b, "predictions", "test_predictions.npy"))


# =========================================================================
# Test 5: convert_outputs_d and the inverse
# =========================================================================
def test_convert_outputs_d_and_inverse(monkeypatch, tiny_world_data, tmp_path):
    data = tiny_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])

    bundle_p = str(tmp_path / "bundle.npz")
    np.savez(bundle_p, **data["bundle"])
    carves_p = str(tmp_path / "carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(tmp_path / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(tmp_path / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)

    h5ad_p = str(tmp_path / "tiny.h5ad")
    p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, dose_p, h5ad_p)
    adata = ad.read_h5ad(h5ad_p)

    # Synthetic run directory
    run_dir = str(tmp_path / "mock_run")
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    # Compute scanpy transform of test_ctrl exactly as trainer does
    ctrl_raw = np.clip(np.asarray(adata.obsm["X_ctl"], dtype=np.float64), 0, 1e3)
    adata_ctrl = ad.AnnData(X=ctrl_raw.copy())
    import scanpy as sc

    sc.pp.normalize_total(adata_ctrl)
    sc.pp.log1p(adata_ctrl)
    test_mask = adata.obs["p10_split"] == "test"
    ctrl_t = adata_ctrl.X[test_mask].astype(np.float32)

    # Compute scanpy transform of X exactly as trainer does
    adata_x = ad.AnnData(X=np.clip(np.asarray(adata.X, dtype=np.float64), 0, 1e3))
    sc.pp.normalize_total(adata_x)
    sc.pp.log1p(adata_x)
    pert_t = adata_x.X[test_mask].astype(np.float32)
    truth_pred_t = (pert_t - ctrl_t).astype(np.float32)

    np.save(os.path.join(run_dir, "predictions", "test_predictions.npy"), truth_pred_t)
    np.save(os.path.join(run_dir, "predictions", "test_ctrl.npy"), ctrl_t)
    np.save(os.path.join(run_dir, "predictions", "test_ground_truth.npy"), truth_pred_t)  # review 081 C3 (v3b)

    rlog = {
        "seed": 117,
        "n_test": int(test_mask.sum()),
        "stop_reason": "max_epochs",
        "best_epoch": 1,
        "n_epochs_run": 2,
    }
    with open(os.path.join(run_dir, "logs", "resource_log.json"), "w") as f:
        json.dump(rlog, f)

    out_npz = str(tmp_path / "out.npz")
    p10_pertdit.convert_outputs_d(h5ad_p, run_dir, 117, out_npz)

    with np.load(out_npz) as z:
        assert z["row_index"].shape == (40,)
        assert z["delta_pred"].shape == (40, 978)
        assert z["arm"] == "pertdit"
        delta_pred = z["delta_pred"]

    # Verify inverse correctness: mean row Pearson >= 0.999 with raw X - X_ctl
    raw_delta = (adata.X[test_mask] - adata.obsm["X_ctl"][test_mask]).astype(np.float64)
    pred_delta = delta_pred.astype(np.float64)

    a_c = raw_delta - raw_delta.mean(axis=1, keepdims=True)
    b_c = pred_delta - pred_delta.mean(axis=1, keepdims=True)
    r = (a_c * b_c).sum(axis=1) / (np.sqrt((a_c**2).sum(axis=1)) * np.sqrt((b_c**2).sum(axis=1)) + 1e-12)
    assert r.mean() >= 0.999, f"Inverse Pearson {r.mean():.6f} < 0.999"

    # Guard (v3) failure isolated: permute test_ctrl.npy
    perm_ctrl = ctrl_t[::-1].copy()
    np.save(os.path.join(run_dir, "predictions", "test_ctrl.npy"), perm_ctrl)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.convert_outputs_d(h5ad_p, run_dir, 117, str(tmp_path / "fail.npz"))
    assert "guard (v3)" in str(exc.value)

    # Restore test_ctrl.npy
    np.save(os.path.join(run_dir, "predictions", "test_ctrl.npy"), ctrl_t)

    # Guard (v1) failure isolated: bad T_TRAIN_PIN
    monkeypatch.setattr(p10_pertdit, "T_TRAIN_PIN", 999999.0)
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.convert_outputs_d(h5ad_p, run_dir, 117, str(tmp_path / "fail.npz"))
    assert "guard (v1)" in str(exc.value)


def test_row_order_preservation(monkeypatch, tiny_world_data, tmp_path):
    """W36 row-order test: build X = X_ctl + D[row_index], shuffle bundle, assert row_index[i] matches."""
    data = tiny_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])

    rng = np.random.default_rng(1234)
    n_rows = len(data["bundle"]["row_index"])
    perm = rng.permutation(n_rows)

    shuffled_bundle = {}
    for k, v in data["bundle"].items():
        shuffled_bundle[k] = v[perm]

    bundle_p = str(tmp_path / "shuf_bundle.npz")
    np.savez(bundle_p, **shuffled_bundle)
    carves_p = str(tmp_path / "carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(tmp_path / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(tmp_path / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)

    h5ad_p = str(tmp_path / "shuf.h5ad")
    p10_pertdit.build_h5ad_d(bundle_p, carves_p, cmap_p, dose_p, h5ad_p)
    adata = ad.read_h5ad(h5ad_p)

    test_mask = adata.obs["p10_split"] == "test"
    test_obs = adata.obs[test_mask]
    test_ri = test_obs["row_index"].values

    # Check row_index in h5ad order matches the shuffled bundle's test rows in order
    expected_ri = shuffled_bundle["row_index"][shuffled_bundle["split_split_cold_drug_1"] == "test"]
    assert np.array_equal(test_ri, expected_ri)


# =========================================================================
# Test 6: parse_timing and timing_decision
# =========================================================================
def test_parse_timing_and_decision():
    stamped_log = (
        "1000.000\t[1/7] Loading data ...\n"
        "1005.500\t[5/7] Training (max 3 epochs ...)\n"
        "1010.000\t  [Epoch   0] train_loss=0.5\n"
        "1015.000\t  [Epoch   1] train_loss=0.4\n"
        "1030.000\t  [Epoch   2] train_loss=0.3\n"
        "1040.000\t[6/7] Predicting on test set ...\n"
        "1070.000\t[7/7] Saving unified output ...\n"
    )
    t = p10_pertdit.parse_timing(stamped_log, kernel_t0=1000.000)
    assert t["t_pre"] == pytest.approx(5.5)
    assert t["t_epoch"] == pytest.approx((1030.0 - 1010.0) / 2.0)  # 10.0
    assert t["t_test"] == pytest.approx(30.0)

    # Missing marker raises FATAL naming the marker
    bad_log = stamped_log.replace("[5/7] Training", "[5/7] Ignored")
    with pytest.raises(SystemExit) as exc:
        p10_pertdit.parse_timing(bad_log, kernel_t0=1000.000)
    assert "[5/7] Training" in str(exc.value)

    # Decision boundaries
    # P(E) = t_pre + E * t_epoch + t_test = 5.5 + E * 10.0 + 30.0 = 35.5 + 10.0 * E
    # P(62) = 35.5 + 620.0 = 655.5
    # P(1000) = 35.5 + 10000.0 = 10035.5
    timings = {"117": t, "118": t}

    # Boundary 1: P(1000) == cap -> 'run'
    d1 = p10_pertdit.timing_decision(timings, cap_s=10035.5)
    assert d1["decision"] == "run"

    # Boundary 2: P(62) == cap + 1e-6 -> 'not_run_cost'
    d2 = p10_pertdit.timing_decision(timings, cap_s=655.5 - 1e-6)
    assert d2["decision"] == "not_run_cost"

    # Boundary 3: Between -> 'run_capped'
    d3 = p10_pertdit.timing_decision(timings, cap_s=2000.0)
    assert d3["decision"] == "run_capped"

    # Max across seeds
    t117 = {"t_pre": 10.0, "t_epoch": 5.0, "t_test": 20.0}
    t118 = {"t_pre": 5.0, "t_epoch": 8.0, "t_test": 15.0}
    d_max = p10_pertdit.timing_decision({"117": t117, "118": t118}, cap_s=21600.0)
    # max t_pre=10.0, max t_epoch=8.0, max t_test=20.0
    # P1000 = 10.0 + 8000.0 + 20.0 = 8030.0
    assert d_max["P1000"] == pytest.approx(8030.0)
    assert d_max["t_test_s"] == pytest.approx(20.0)


# =========================================================================
# Test 7: Kernel builder and both kernels EXECUTED end-to-end
# =========================================================================
def test_kernel_builder_and_execution_end_to_end(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    data = tiny_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])

    # Prepare mounted files in input directory
    inp_dir = tmp_path / "kaggle_input" / "p10d"
    inp_dir.mkdir(parents=True)
    temp_dir = tmp_path / "kaggle_temp"
    temp_dir.mkdir(parents=True)
    work_dir = tmp_path / "kaggle_working"
    work_dir.mkdir(parents=True)

    bundle_p = str(inp_dir / "xpert_mdmt_splits.npz")
    np.savez(bundle_p, **data["bundle"])
    carves_p = str(inp_dir / "p10_carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(inp_dir / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(inp_dir / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)

    drug_emb_p = str(inp_dir / "bench_drug_emb.pkl")
    dose_emb_p = str(inp_dir / "bench_dose_emb.pt")
    torch.save(data["drug_emb"], drug_emb_p)
    torch.save(data["dose_emb"], dose_emb_p)

    # emb_record
    emb_rec_p = str(tmp_path / "emb_record.json")
    emb_rec = {
        "sha1s": {
            "bench_drug_emb.pkl": p10_pertdit.sha1_file(drug_emb_p),
            "bench_dose_emb.pt": p10_pertdit.sha1_file(dose_emb_p),
        },
        "compound_map_sha1": p10_pertdit.sha1_file(cmap_p),
        "dose_list_sha1": p10_pertdit.sha1_file(dose_p),
    }
    with open(emb_rec_p, "w", encoding="utf-8") as f:
        json.dump(emb_rec, f)

    # Prepare p10_mlp.py with monkeypatched constants for the synthetic world
    mlp_src = open(p10_mlp.__file__, "r", encoding="utf-8").read()
    for k, v in data["constants_mlp"].items():
        line = [l for l in mlp_src.splitlines() if l.startswith(k + " = ")]
        assert len(line) == 1
        mlp_src = mlp_src.replace(line[0], f"{k} = {v!r}")
    p10_mlp_file = str(inp_dir / "p10_mlp.py")
    with open(p10_mlp_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(mlp_src)

    # Prepare p10_pertdit.py with monkeypatched constants for the synthetic world
    pertdit_src = open(p10_pertdit.__file__, "r", encoding="utf-8").read()
    for k, v in data["constants_pertdit"].items():
        line = [l for l in pertdit_src.splitlines() if l.startswith(k + " = ")]
        assert len(line) == 1
        pertdit_src = pertdit_src.replace(line[0], f"{k} = {v!r}")
    p10_pertdit_file = str(inp_dir / "p10_pertdit.py")
    with open(p10_pertdit_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(pertdit_src)

    # Build kernels into tmp_path / "kernels"
    out_root = str(tmp_path / "kernels")
    src_root = str(tmp_path / "kaggle_p10d_src")
    make_p10_d_kernels.build_kernels(
        emb_record_path=emb_rec_p,
        timing_json_path=_pi_timing_json(tmp_path),  # review 081 C2
        out_root=out_root,
        src_root=src_root,
        compound_map_path=cmap_p,
        dose_list_path=dose_p,
        bundle_path=bundle_p,
        carves_path=carves_p,
        p10_mlp_path=p10_mlp_file,
        p10_pertdit_path=p10_pertdit_file,
    )

    # Copy src files to inp_dir so kernel finds them under /kaggle/input
    for f in os.listdir(src_root):
        shutil.copyfile(os.path.join(src_root, f), str(inp_dir / f))

    # Static assertions on generated scripts and metadata
    timing_script = os.path.join(out_root, "kern_p10d_timing", "lincs-p10d-timing.py")
    run_script = os.path.join(out_root, "kern_p10d_run", "lincs-p10d-run.py")
    assert os.path.isfile(timing_script)
    assert os.path.isfile(run_script)

    t_code = open(timing_script, "r", encoding="utf-8").read()
    r_code = open(run_script, "r", encoding="utf-8").read()
    ast.parse(t_code)
    ast.parse(r_code)

    # Static check: run script contains no --batch_size 16, --train_steps 5, --num_steps 5, nor --n_epochs
    assert "--batch_size" not in r_code
    assert "--train_steps" not in r_code
    assert "--num_steps" not in r_code
    assert '"--n_epochs"' not in r_code

    # Static check: metadata fields
    for k_dir, mid in [("kern_p10d_timing", "lincs-p10d-timing"), ("kern_p10d_run", "lincs-p10d-run")]:
        meta = json.load(open(os.path.join(out_root, k_dir, "kernel-metadata.json")))
        assert meta["id"] == f"apexblue/{mid}"
        assert meta["enable_gpu"] is True
        assert meta["enable_internet"] is False
        assert meta["machine_shape"] == "NvidiaTeslaT4"
        assert "gcr.io/kaggle-private-byod/python" in meta["docker_image"]
        assert meta["kernel_sources"] == ["apexblue/lincs-p10d-emb-merge"]
        assert "apexblue/xpert-mdmt-benchmark" in meta["dataset_sources"]

    # Execute Timing Kernel
    env_test = dict(os.environ)
    env_test["KAGGLE_INPUT_DIR"] = str(inp_dir)
    env_test["KAGGLE_TEMP_DIR"] = str(temp_dir)
    env_test["KAGGLE_WORKING_DIR"] = str(work_dir)
    env_test["P10D_BYPASS_GPU_GUARD"] = "1"
    env_test["P10D_BYPASS_PIP"] = "1"
    env_test["P10D_EXTRA_TRAIN_ARGS"] = "--batch_size 16 --train_steps 5 --num_steps 5 --dev cpu"
    env_test["PYTHONPATH"] = windows_shim + os.pathsep + env_test.get("PYTHONPATH", "")

    # Clear working
    for f in os.listdir(work_dir):
        os.remove(os.path.join(work_dir, f))

    res_t = subprocess.run([sys.executable, timing_script], capture_output=True, text=True, env=env_test, timeout=300)
    assert res_t.returncode == 0, f"Timing kernel failed:\nSTDOUT:\n{res_t.stdout}\nSTDERR:\n{res_t.stderr}"

    # Working dir holds exactly p10d_timing.json
    assert os.listdir(work_dir) == ["p10d_timing.json"]
    t_json = json.load(open(os.path.join(work_dir, "p10d_timing.json")))
    assert "timings" in t_json and "decision" in t_json

    # Execute Run Kernel
    for f in os.listdir(work_dir):
        os.remove(os.path.join(work_dir, f))
    env_test["P10D_EXTRA_TRAIN_ARGS"] = "--batch_size 16 --train_steps 5 --num_steps 5 --n_epochs 2 --dev cpu"

    res_r = subprocess.run([sys.executable, run_script], capture_output=True, text=True, env=env_test, timeout=300)
    assert res_r.returncode == 0, f"Run kernel failed:\nSTDOUT:\n{res_r.stdout}\nSTDERR:\n{res_r.stderr}"

    # Working dir holds exactly the 10 briefed files
    expected_run_files = sorted([
        "p10_pertdit_seed117.npz",
        "best_model_s117.pt",
        "training_curve_s117.csv",
        "resource_log_s117.json",
        "p10_pertdit_seed117_record.json",
        "p10_pertdit_seed118.npz",
        "best_model_s118.pt",
        "training_curve_s118.csv",
        "resource_log_s118.json",
        "p10_pertdit_seed118_record.json",
    ])
    actual_run_files = sorted(os.listdir(work_dir))
    assert actual_run_files == expected_run_files

    # Assert no test_ground_truth or per_drug_pcc anywhere in working
    assert not any("ground_truth" in f or "per_drug" in f or "sample_ids" in f for f in actual_run_files)

    # NPZ reload checks
    with np.load(os.path.join(work_dir, "p10_pertdit_seed117.npz")) as z:
        assert z["row_index"].shape == (40,)
        assert z["delta_pred"].shape == (40, 978)
        assert z["arm"] == "pertdit"
        assert int(z["seed"]) == 117

    rec117 = json.load(open(os.path.join(work_dir, "p10_pertdit_seed117_record.json")))
    rec118 = json.load(open(os.path.join(work_dir, "p10_pertdit_seed118_record.json")))
    assert rec117["of_record"] is True
    assert rec118["of_record"] is False
    assert rec117["n_test"] == 40
    assert rec118["n_test"] == 40

    # Test GPU guard default enforcement: with P10D_BYPASS_GPU_GUARD unset, it exits non-zero
    env_strict = dict(env_test)
    env_strict.pop("P10D_BYPASS_GPU_GUARD", None)
    res_guard = subprocess.run([sys.executable, "-c",
        "import os\n"
        "code = open(r'" + timing_script + "', encoding='utf-8').read()\n"
        "exec(compile(code[:code.index('# 2. Install')], '<kernel>', 'exec'))\n"
    ], capture_output=True, text=True, env=env_strict)
    assert res_guard.returncode != 0
    assert "FATAL" in res_guard.stderr or "FATAL" in res_guard.stdout



# =========================================================================
# PI (mutation check, W40): 23 of 47 mutants survived W40's suite. Its (h4) test re-implemented the guard inside the test,
# several worlds failed two checks at once (one masked the other), and every in-kernel guard was untested on a failure path.
# Each world below fails exactly one check and asserts its message. No test here trains on a GPU (CUDA is hidden or the
# trainer is never reached; the one trainer run uses --dev cpu).
# =========================================================================
def _pi_files(tmp_path, data, cmap=None, dose_list=None):
    bundle_p = str(tmp_path / "bundle.npz")
    np.savez(bundle_p, **data["bundle"])
    carves_p = str(tmp_path / "carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(tmp_path / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"] if cmap is None else cmap, f)
    dose_p = str(tmp_path / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"] if dose_list is None else dose_list, f)
    return bundle_p, carves_p, cmap_p, dose_p


def _pi_raises(fn, needle):
    with pytest.raises(SystemExit) as exc:
        fn()
    assert needle in str(exc.value), str(exc.value)


def test_pi_h1_count_and_values_isolated(monkeypatch, synthetic_world_data, tmp_path):
    data = synthetic_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])
    n = data["constants_pertdit"]["N_COMPOUNDS"]
    # values not 0..n-1, count right
    cm = dict(data["compound_map"])
    cm[list(cm)[0]] = n
    b, c, m, d = _pi_files(tmp_path, data, cmap=cm)
    _pi_raises(lambda: p10_pertdit.build_h5ad_d(b, c, m, d, str(tmp_path / "o.h5ad")), "values do not equal")
    # count n+1, values exactly 0..n-1 (a duplicate value)
    cm = dict(data["compound_map"])
    cm["999999"] = 0
    b, c, m, d = _pi_files(tmp_path, data, cmap=cm)
    _pi_raises(lambda: p10_pertdit.build_h5ad_d(b, c, m, d, str(tmp_path / "o.h5ad")), f"has {n + 1} keys")


def test_pi_timing_guards_on_the_real_code(monkeypatch, synthetic_world_data, tmp_path):
    data = synthetic_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])
    b, c, m, d = _pi_files(tmp_path, data)
    out = str(tmp_path / "t.h5ad")
    p10_pertdit.build_h5ad_d(b, c, m, d, out, timing=True)
    good = ad.read_h5ad(out)
    real = set(int(r) for r in data["test_rows"])
    p10_pertdit.check_timing_h5ad(good, real)  # passes
    # (h4a) a 'test' row with row_index >= 0 that is not a real test row
    a1 = good.copy()
    ri = a1.obs["row_index"].values.copy()
    ri[np.where((a1.obs["p10_split"] == "test").values)[0][0]] = 5  # a fit-row index, not a real test row
    a1.obs["row_index"] = ri
    _pi_raises(lambda: p10_pertdit.check_timing_h5ad(a1, real), "test row has row_index >= 0")
    # (h4b) a real test row_index on a non-test row
    a2 = good.copy()
    ri = a2.obs["row_index"].values.copy()
    ri[np.where((a2.obs["p10_split"] == "train").values)[0][0]] = int(data["test_rows"][0])
    a2.obs["row_index"] = ri
    _pi_raises(lambda: p10_pertdit.check_timing_h5ad(a2, real), "real test row_index occurs")
    # (h5) counts
    a3 = good[1:].copy()
    _pi_raises(lambda: p10_pertdit.check_timing_h5ad(a3, real), "guard (h5)")
    # build_h5ad_d(timing=True) really calls it
    called = []
    monkeypatch.setattr(p10_pertdit, "check_timing_h5ad", lambda a, r: called.append(len(a)))
    p10_pertdit.build_h5ad_d(b, c, m, d, str(tmp_path / "t2.h5ad"), timing=True)
    assert called


def test_pi_cfg_check_each_condition_isolated():
    src = open(TRAIN_PERTDIT_SRC, encoding="utf-8").read()
    p10_pertdit.check_trainer_cfg(src)
    _pi_raises(lambda: p10_pertdit.check_trainer_cfg(src + "\nx = dict(cfg=False)\n"), "check_trainer_cfg")
    _pi_raises(lambda: p10_pertdit.check_trainer_cfg(src + "\n# cfg=True\n"), "check_trainer_cfg")


def test_pi_check_embeddings_each_condition_isolated(monkeypatch, tiny_world_data, tmp_path):
    data = tiny_world_data
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])
    n = data["constants_pertdit"]["N_COMPOUNDS"]
    obs = pd.DataFrame({"pert_idx": np.arange(n, dtype=np.int64), "dose_r4": [data["dose_list"][i % 3] for i in range(n)]})
    obs.index = [f"r{i}" for i in range(n)]
    adata = ad.AnnData(X=np.zeros((n, 978), dtype=np.float32), obs=obs)

    def files(drug, dose):
        dp, sp = str(tmp_path / "drug.pkl"), str(tmp_path / "dose.pt")
        torch.save(drug, dp)
        torch.save(dose, sp)
        return dp, sp

    p10_pertdit.check_embeddings(*files(data["drug_emb"], data["dose_emb"]), adata)  # passes
    # (e1) keys: an extra key, coverage fine
    drug = dict(data["drug_emb"]); drug["extra"] = torch.zeros(2, 1024)
    _pi_raises(lambda: p10_pertdit.check_embeddings(*files(drug, data["dose_emb"]), adata), "drug_emb keys mismatch")
    # (e1) coverage: an h5ad pert_idx outside the keys, keys exact
    a2 = adata.copy(); a2.obs["pert_idx"] = np.r_[np.arange(n - 1), n].astype(np.int64)
    _pi_raises(lambda: p10_pertdit.check_embeddings(*files(data["drug_emb"], data["dose_emb"]), a2), "does not cover")
    # (e4) keys: an extra dose key, every lookup fine
    dose = dict(data["dose_emb"]); dose[99.0] = torch.zeros(2, 1024)
    _pi_raises(lambda: p10_pertdit.check_embeddings(*files(data["drug_emb"], dose), adata), "dose_emb keys mismatch")
    # (e4) lookup: keys exact, an h5ad dose that isn't a key
    a3 = adata.copy(); a3.obs["dose_r4"] = [data["dose_list"][0]] * (n - 1) + [float(np.float32(0.37))]
    _pi_raises(lambda: p10_pertdit.check_embeddings(*files(data["drug_emb"], data["dose_emb"]), a3), "lookup failed")


def _pi_conv_world(monkeypatch, data, tmp_path):
    import scanpy as sc
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])
    b, c, m, d = _pi_files(tmp_path, data)
    h5 = str(tmp_path / "tiny.h5ad")
    p10_pertdit.build_h5ad_d(b, c, m, d, h5)
    adata = ad.read_h5ad(h5)
    run = tmp_path / "run"
    (run / "predictions").mkdir(parents=True)
    (run / "logs").mkdir()
    tm = (adata.obs["p10_split"] == "test").values
    actl = ad.AnnData(X=np.clip(np.asarray(adata.obsm["X_ctl"], dtype=np.float64), 0, 1e3))
    sc.pp.normalize_total(actl); sc.pp.log1p(actl)
    ax = ad.AnnData(X=np.clip(np.asarray(adata.X, dtype=np.float64), 0, 1e3))
    sc.pp.normalize_total(ax); sc.pp.log1p(ax)
    ctrl_t = actl.X[tm].astype(np.float32)
    pred_t = (ax.X[tm] - actl.X[tm]).astype(np.float32)
    np.save(run / "predictions" / "test_predictions.npy", pred_t)
    np.save(run / "predictions" / "test_ctrl.npy", ctrl_t)
    np.save(run / "predictions" / "test_ground_truth.npy", pred_t)  # review 081 C3 (v3b): pred_t here IS their truth
    rlog = {"seed": 117, "n_test": int(tm.sum()), "stop_reason": "rule", "best_epoch": 3, "n_epochs_run": 14}
    (run / "logs" / "resource_log.json").write_text(json.dumps(rlog), encoding="utf-8")
    return h5, str(run), pred_t, rlog, adata, tm


def test_pi_convert_guards_isolated_and_inverse_scaling(monkeypatch, tiny_world_data, tmp_path):
    h5, run, pred_t, rlog, adata, tm = _pi_conv_world(monkeypatch, tiny_world_data, tmp_path)
    out = str(tmp_path / "o.npz")
    p10_pertdit.convert_outputs_d(h5, run, 117, out)
    # the inverse rescales to each row's own control total: the delta's scale is right, not just its correlation
    with np.load(out) as z:
        dp = z["delta_pred"].astype(np.float64)
        assert str(z["stop_reason"]) == "rule" and int(z["n_epochs_run"]) == 14
    raw = (np.asarray(adata.X, dtype=np.float64) - np.asarray(adata.obsm["X_ctl"], dtype=np.float64))[tm]
    assert np.abs(dp - raw).max() < 0.05 * np.abs(raw).max()
    pp = os.path.join(run, "predictions", "test_predictions.npy")
    # (v2) a short prediction array
    np.save(pp, pred_t[:-1])
    _pi_raises(lambda: p10_pertdit.convert_outputs_d(h5, run, 117, out), "guard (v2)")
    # (v4) a non-finite prediction (expm1 overflow), everything else intact
    bad = pred_t.copy(); bad[0, 0] = np.inf
    np.save(pp, bad)
    _pi_raises(lambda: p10_pertdit.convert_outputs_d(h5, run, 117, out), "guard (v4)")
    np.save(pp, pred_t)
    lp = os.path.join(run, "logs", "resource_log.json")
    # (v5) seed, then n_test
    with open(lp, "w", encoding="utf-8") as f:
        json.dump(dict(rlog, seed=118), f)
    _pi_raises(lambda: p10_pertdit.convert_outputs_d(h5, run, 117, out), "resource_log seed")
    with open(lp, "w", encoding="utf-8") as f:
        json.dump(dict(rlog, n_test=rlog["n_test"] + 1), f)
    _pi_raises(lambda: p10_pertdit.convert_outputs_d(h5, run, 117, out), "resource_log n_test")


# ---- in-kernel guards ----
_PI_FAKE_GPU_SITE = """
import os
if os.environ.get("PI_FAKE_GPUS"):
    import torch
    _names = os.environ["PI_FAKE_GPUS"].split("|")
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: len(_names)
    torch.cuda.get_device_name = lambda i=0: _names[i]
"""


def _pi_kernel_world(monkeypatch, data, windows_shim, tmp_path, trainer_path=None, drug_emb=None, extra_pertdit=None):
    """The KE world of test_kernel_builder_and_execution_end_to_end, as a reusable builder."""
    patch_all_constants(monkeypatch, data["constants_mlp"], data["constants_pertdit"])
    inp = tmp_path / "kaggle_input" / "p10d"
    inp.mkdir(parents=True)
    temp, work = tmp_path / "kaggle_temp", tmp_path / "kaggle_working"
    temp.mkdir(); work.mkdir()
    bundle_p = str(inp / "xpert_mdmt_splits.npz"); np.savez(bundle_p, **data["bundle"])
    carves_p = str(inp / "p10_carves.json")
    with open(carves_p, "wb") as f:
        f.write(data["carves_bytes"])
    cmap_p = str(inp / "compound_map.json")
    with open(cmap_p, "w", encoding="utf-8") as f:
        json.dump(data["compound_map"], f)
    dose_p = str(inp / "dose_list.json")
    with open(dose_p, "w", encoding="utf-8") as f:
        json.dump(data["dose_list"], f)
    drug_p, dose_e_p = str(inp / "bench_drug_emb.pkl"), str(inp / "bench_dose_emb.pt")
    torch.save(data["drug_emb"] if drug_emb is None else drug_emb, drug_p)
    torch.save(data["dose_emb"], dose_e_p)
    rec_p = str(tmp_path / "emb_record.json")
    with open(rec_p, "w", encoding="utf-8") as f:
        json.dump({"sha1s": {"bench_drug_emb.pkl": p10_pertdit.sha1_file(drug_p),
                             "bench_dose_emb.pt": p10_pertdit.sha1_file(dose_e_p)},
                   "compound_map_sha1": p10_pertdit.sha1_file(cmap_p), "dose_list": data["dose_list"]}, f)
    consts = dict(data["constants_pertdit"], **(extra_pertdit or {}))
    for mod, cmap_consts, name in ((p10_mlp, data["constants_mlp"], "p10_mlp.py"), (p10_pertdit, consts, "p10_pertdit.py")):
        src = open(mod.__file__, "r", encoding="utf-8").read()
        for k, v in cmap_consts.items():
            line = [l for l in src.splitlines() if l.startswith(k + " = ")]
            assert len(line) == 1, k
            src = src.replace(line[0], f"{k} = {v!r}")
        with open(inp / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
    # each world sets the trainer pin it builds with (a previous world's patch must not leak into this one)
    monkeypatch.setattr(p10_pertdit, "TRAINER_SHA1", p10_pertdit.sha1_file(trainer_path or TRAIN_PERTDIT_SRC))
    for k, v in (extra_pertdit or {}).items():
        monkeypatch.setattr(p10_pertdit, k, v)
    out_root, src_root = str(tmp_path / "kernels"), str(tmp_path / "src")
    kw = dict(emb_record_path=rec_p, timing_json_path=_pi_timing_json(tmp_path), out_root=out_root, src_root=src_root, compound_map_path=cmap_p,
              dose_list_path=dose_p, bundle_path=bundle_p, carves_path=carves_p, p10_mlp_path=str(inp / "p10_mlp.py"),
              p10_pertdit_path=str(inp / "p10_pertdit.py"))
    if trainer_path:
        kw["train_pertdit_path"] = trainer_path
    make_p10_d_kernels.build_kernels(**kw)
    for f in os.listdir(src_root):
        shutil.copyfile(os.path.join(src_root, f), str(inp / f))
    site = tmp_path / "fakegpu"
    site.mkdir()
    (site / "sitecustomize.py").write_text(open(os.path.join(windows_shim, "sitecustomize.py"), encoding="utf-8").read()
                                           + _PI_FAKE_GPU_SITE, encoding="utf-8")
    env = dict(os.environ, KAGGLE_INPUT_DIR=str(inp), KAGGLE_TEMP_DIR=str(temp), KAGGLE_WORKING_DIR=str(work),
               P10D_BYPASS_GPU_GUARD="1", P10D_BYPASS_PIP="1", CUDA_VISIBLE_DEVICES="-1",
               P10D_EXTRA_TRAIN_ARGS="--batch_size 16 --train_steps 5 --num_steps 5 --dev cpu",
               PYTHONPATH=str(site) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return {"inp": inp, "work": work, "env": env, "timing": os.path.join(out_root, "kern_p10d_timing", "lincs-p10d-timing.py"),
            "run": os.path.join(out_root, "kern_p10d_run", "lincs-p10d-run.py"), "rec": rec_p, "kw": kw}


def _pi_kfatal(script, env, needle, swaps=()):
    code = open(script, encoding="utf-8").read()
    for a, b in swaps:
        assert code.count(a) >= 1, a
        code = code.replace(a, b)
    run = script + ".pi.py"
    with open(run, "w", encoding="utf-8") as f:
        f.write(code)
    r = subprocess.run([sys.executable, run], capture_output=True, text=True, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "FATAL" in out and needle in out, out[-3000:]


def test_pi_kernel_gpu_guard_with_fake_devices(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    w = _pi_kernel_world(monkeypatch, tiny_world_data, windows_shim, tmp_path)
    env = dict(w["env"]); env.pop("P10D_BYPASS_GPU_GUARD")
    for gpus in ("Tesla T4", "Tesla T4|Tesla P100-PCIE-16GB"):
        _pi_kfatal(w["timing"], dict(env, PI_FAKE_GPUS=gpus), "need 2 non-P100 GPUs")
    _pi_kfatal(w["timing"], env, "no CUDA device")


def test_pi_kernel_input_guards_each_fail_loud(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    w = _pi_kernel_world(monkeypatch, tiny_world_data, windows_shim, tmp_path)
    # the run kernel passes the cap to the trainer (static)
    assert '"--cap_s", str(CAP_S)' in open(w["run"], encoding="utf-8").read()
    # a cap-trainer pin that the in-kernel rebuild does not reproduce
    code = open(w["timing"], encoding="utf-8").read()
    pin = [l for l in code.splitlines() if l.startswith("CAP_TRAINER_SHA1 = ")][0]
    _pi_kfatal(w["timing"], w["env"], "cap_trainer sha1 mismatch", swaps=[(pin, "CAP_TRAINER_SHA1 = '" + "0" * 40 + "'")])
    # a mounted input whose bytes differ from the build's (last, since it alters the input folder)
    with open(w["inp"] / "dose_list.json", "a", encoding="utf-8") as f:
        f.write(" ")
    _pi_kfatal(w["timing"], w["env"], "GUARD 2: dose_list sha1 mismatch")


def test_pi_kernel_cfg_and_embedding_checks_fire_in_kernel(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    # a trainer with a cfg=True (pinned as built, so the sha1 guards pass and only the in-kernel cfg check can fire)
    bad_tr = tmp_path / "train_pertdit.py"
    bad_tr.write_bytes(open(TRAIN_PERTDIT_SRC, "rb").read() + b"\n# cfg=True\n")
    w = _pi_kernel_world(monkeypatch, tiny_world_data, windows_shim, tmp_path / "a", trainer_path=str(bad_tr),
                         extra_pertdit={"TRAINER_SHA1": p10_pertdit.sha1_file(str(bad_tr))})
    _pi_kfatal(w["timing"], w["env"], "check_trainer_cfg")
    # embeddings with a non-finite compound (pinned as built, so only check_embeddings can fire)
    drug = dict(tiny_world_data["drug_emb"]); drug[3] = torch.full((3, 1024), float("nan"))
    w = _pi_kernel_world(monkeypatch, tiny_world_data, windows_shim, tmp_path / "b", drug_emb=drug)
    _pi_kfatal(w["timing"], w["env"], "guard (e3)")


def test_pi_kernel_blinding_on_a_real_timing_run(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    w = _pi_kernel_world(monkeypatch, tiny_world_data, windows_shim, tmp_path)
    r = subprocess.run([sys.executable, w["timing"]], capture_output=True, text=True, env=w["env"], timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    # the trainer prints these; the kernel must not echo them
    for leak in ("Shapes:", "[6/7] Predicting", "[7/7] Saving", "Data after preprocessing", "Drug embeddings:"):
        assert leak not in r.stdout, leak
    assert "[Epoch" in r.stdout  # the allowed curve lines do come through
    # review 081 C1: the stamps are event times -- strictly ordered markers and real epoch / test durations, per seed
    tj = json.load(open(os.path.join(str(w["work"]), "p10d_timing.json"), encoding="utf-8"))
    for seed_k, tm in tj["timings"].items():
        assert tm["t_epoch"] > 1.0 and tm["t_test"] > 1.0 and tm["t_pre"] > 0, (seed_k, tm)
    assert {"versions", "torch_version", "gpu_names"} <= set(tj)


def test_pi_builder_refuses_embeddings_from_another_compound_map(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    w = _pi_kernel_world(monkeypatch, tiny_world_data, windows_shim, tmp_path)
    rec = json.load(open(w["rec"], encoding="utf-8"))
    rec["compound_map_sha1"] = "0" * 40
    with open(w["rec"], "w", encoding="utf-8") as f:
        json.dump(rec, f)
    _pi_raises(lambda: make_p10_d_kernels.build_kernels(**w["kw"]), "the embeddings were built with compound map")
    rec["compound_map_sha1"] = p10_pertdit.sha1_file(w["kw"]["compound_map_path"])
    rec["dose_list"] = [0.1, 1.0, 10.0]
    with open(w["rec"], "w", encoding="utf-8") as f:
        json.dump(rec, f)
    _pi_raises(lambda: make_p10_d_kernels.build_kernels(**w["kw"]), "the embeddings were built for doses")



# =========================================================================
# Review 081: C1 timing stamps are event times; C2 the run kernel is bound to the timing JSON; C3 (v3b); C4 records
# =========================================================================
def _pi_timing_json(tmp_path, t_epoch=10.0, t_test=30.0, decision=None, cap_s=None):
    """A timing JSON as the timing kernel writes it (decision derived from the timings unless overridden)."""
    tm = {"t_pre": 50.0, "t_epoch": t_epoch, "t_test": t_test}
    timings = {"117": tm, "118": dict(tm)}
    dec = p10_pertdit.timing_decision(timings, cap_s=cap_s or p10_pertdit.CAP_S)
    if decision is not None:
        dec = dict(dec, decision=decision)
    p = os.path.join(str(tmp_path), f"timing_{t_epoch}_{t_test}_{decision}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"timings": timings, "decision": dec}, f)
    return p


def test_pi_review081_parse_timing_refuses_flushed_stamps():
    good = ("1000.000\t[5/7] Training\n1010.000\t  [Epoch   0] x\n1030.000\t  [Epoch   2] x\n"
            "1040.000\t[6/7] Predicting on test set\n1070.000\t[7/7] Saving unified output\n")
    p10_pertdit.parse_timing(good, kernel_t0=990.0)
    flushed = "\n".join("1500.000\t" + l.split("\t", 1)[1] for l in good.strip().splitlines()) + "\n"
    _pi_raises(lambda: p10_pertdit.parse_timing(flushed, kernel_t0=990.0), "not strictly increasing")
    fast = good.replace("1030.000", "1010.500")  # t_epoch 0.25 s, ordering intact
    _pi_raises(lambda: p10_pertdit.parse_timing(fast, kernel_t0=990.0), "below")


def test_pi_review081_run_kernel_bound_to_timing(monkeypatch, tiny_world_data, windows_shim, tmp_path):
    w = _pi_kernel_world(monkeypatch, tiny_world_data, windows_shim, tmp_path)
    code = open(w["run"], encoding="utf-8").read()
    tj = json.load(open(w["kw"]["timing_json_path"], encoding="utf-8"))
    assert f"T_TEST_S = {float(tj['decision']['t_test_s'])}" in code
    assert f"TIMING_SHA1 = '{p10_pertdit.sha1_file(w['kw']['timing_json_path'])}'" in code
    assert f"TIMING_DECISION = '{tj['decision']['decision']}'" in code
    # a not_run_cost timing builds nothing
    kw = dict(w["kw"], timing_json_path=_pi_timing_json(tmp_path, t_epoch=400.0))
    _pi_raises(lambda: make_p10_d_kernels.build_kernels(**kw), "not_run_cost")
    # a decision that its timings don't give
    kw = dict(w["kw"], timing_json_path=_pi_timing_json(tmp_path, t_epoch=400.0, decision="run"))
    _pi_raises(lambda: make_p10_d_kernels.build_kernels(**kw), "is not what its timings give")
    # a non-positive timing
    kw = dict(w["kw"], timing_json_path=_pi_timing_json(tmp_path, t_test=0.0))
    _pi_raises(lambda: make_p10_d_kernels.build_kernels(**kw), "non-positive timing")
    # the kernel template asserts the pinned GPU version line (C4; the real line is checked on Kaggle before the reset)
    assert "if installed_versions != PINNED_GPU_VERSIONS:" in code and "PYTHONUNBUFFERED" in code


def test_pi_review081_v3b_sees_order_inside_shared_controls(monkeypatch, tiny_world_data, tmp_path):
    h5, run, pred_t, rlog, adata, tm = _pi_conv_world(monkeypatch, tiny_world_data, tmp_path)
    gp = os.path.join(run, "predictions", "test_ground_truth.npy")
    gt = np.load(gp)
    np.save(gp, gt[[1, 0] + list(range(2, len(gt)))])  # swap two rows; test_ctrl untouched, so (v3) passes
    _pi_raises(lambda: p10_pertdit.convert_outputs_d(h5, run, 117, str(tmp_path / "o.npz")), "guard (v3b)")
    os.remove(gp)
    _pi_raises(lambda: p10_pertdit.convert_outputs_d(h5, run, 117, str(tmp_path / "o.npz")), "guard (v3b): missing")
