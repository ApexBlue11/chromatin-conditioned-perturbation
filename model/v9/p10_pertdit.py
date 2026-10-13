# -*- coding: utf-8 -*-
"""RESULTS §98.15, §98.16: P10 arm D (PertDiT) adapter, cap trainer patch, inverse, and timing.

Module 1: Adapter and tools for arm D (PertDiT / CrossDiT via Bai et al.'s trainer).
- build_h5ad_d: reads bundle, carves, compound_map, and dose_list; validates guards (h1)-(h5);
  and writes an AnnData h5ad. With timing=True, drops test rows and appends 13,445 sampled train rows.
- make_cap_trainer: generates p10_train_pertdit_cap.py from Bai et al.'s train_pertdit.py via
  asserted exact substitutions for session cap loop control.
- check_trainer_cfg: verifies cfg=False occurs exactly 3 times and cfg=True 0 times.
- check_embeddings: validates drug and dose embedding shapes, types, negative_ctrl, and coverage.
- convert_outputs_d: applies the output inverse inverting log1p(normalize_total) back to raw expression space,
  verifying alignment and T_train guards.
- parse_timing / timing_decision: parses stamped trainer logs and computes P62, P1000 and timing decision.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "model", "v9"))
import p10_mlp  # noqa: E402

# Module constants (monkeypatchable for synthetic worlds)
TRAINER_SHA1 = "a431f53ba18d43348161ee19b97b01148229f1ac"
T_TRAIN_PIN = 8242.431170
N_COMPOUNDS = 1970
DOSES = [0.1, 0.37, 0.62, 0.77, 1.11, 2.31, 3.33, 5.0, 10.0, 40.0]
TIMING_RNG = 9818
CAP_S = 21600.0


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_h5ad_d(bundle_npz, carves_json, compound_map_json, dose_list_json, out_h5ad, timing=False):
    """Build AnnData h5ad for arm D with guards (h1)-(h5)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_h5ad = os.path.join(tmp_dir, "base.h5ad")
        # Reuse p10_mlp.build_h5ad (its guards (a)-(e) run there)
        p10_mlp.build_h5ad(bundle_npz, carves_json, tmp_h5ad, with_dt=False)
        adata = ad.read_h5ad(tmp_h5ad)

    # Load compound_map_json
    if isinstance(compound_map_json, (str, os.PathLike)):
        with open(compound_map_json, "r", encoding="utf-8") as f:
            compound_map = json.load(f)
    else:
        compound_map = compound_map_json

    # Guard (h1): compound_map_json maps exactly N_COMPOUNDS compounds onto 0..N_COMPOUNDS-1 (a bijection)
    if len(compound_map) != N_COMPOUNDS:
        raise SystemExit(
            f"FATAL: guard (h1): compound_map has {len(compound_map)} keys, expected {N_COMPOUNDS}"
        )
    val_set = set(compound_map.values())
    if val_set != set(range(N_COMPOUNDS)):
        raise SystemExit(
            f"FATAL: guard (h1): compound_map values do not equal 0..{N_COMPOUNDS - 1}"
        )
    try:
        cmap_int_keys = {int(k): int(v) for k, v in compound_map.items()}
    except (ValueError, TypeError):
        raise SystemExit("FATAL: guard (h1): compound_map keys must be integer strings")
    if len(cmap_int_keys) != N_COMPOUNDS:
        raise SystemExit("FATAL: guard (h1): compound_map int keys are not unique")

    # Modify obs: pert_idx_xpert, pert_idx, dose_r4
    orig_pert_idx = np.asarray(adata.obs["pert_idx"], dtype=np.int64)
    adata.obs["pert_idx_xpert"] = orig_pert_idx

    # Guard (h2): every row's pert_idx is in the map
    for p in orig_pert_idx:
        if p not in cmap_int_keys:
            raise SystemExit(f"FATAL: guard (h2): row pert_idx {p} not found in compound_map")

    adata.obs["pert_idx"] = np.array([cmap_int_keys[p] for p in orig_pert_idx], dtype=np.int64)
    adata.obs["dose_r4"] = np.array(
        [float(round(float(d), 4)) for d in adata.obs["dose"]], dtype=np.float64
    )

    # Load dose_list_json
    if isinstance(dose_list_json, (str, os.PathLike)):
        with open(dose_list_json, "r", encoding="utf-8") as f:
            dose_list = json.load(f)
    else:
        dose_list = dose_list_json

    # Guard (h3): the set of dose_r4 values equals dose_list_json's list exactly (float ==)
    obs_doses = set(adata.obs["dose_r4"].values)
    target_doses = set(dose_list)
    if obs_doses != target_doses:
        raise SystemExit(
            f"FATAL: guard (h3): dose_r4 set does not equal dose_list: {obs_doses} != {target_doses}"
        )

    if timing:
        # 98.16 C3: timing h5ad holds fit and validation rows, plus copies of 13,445 fit rows labelled test
        real_test_mask = (adata.obs["p10_split"] == "test").values
        real_test_ri = set(adata.obs.loc[real_test_mask, "row_index"].values.astype(np.int64))

        # Drop every p10_split == 'test' row
        adata_fit_val = adata[~real_test_mask].copy()

        train_mask = (adata_fit_val.obs["p10_split"] == "train").values
        train_indices = np.where(train_mask)[0]
        n_train = len(train_indices)
        n_draw = p10_mlp.EXPECT_N_TEST

        rng = np.random.default_rng(TIMING_RNG)
        chosen_sub = rng.choice(n_train, n_draw, replace=False)
        chosen_sub.sort()
        chosen_indices = train_indices[chosen_sub]

        # Extract copies of those rows
        test_X = adata_fit_val.X[chosen_indices].copy()
        test_X_ctl = adata_fit_val.obsm["X_ctl"][chosen_indices].copy()
        test_obs = adata_fit_val.obs.iloc[chosen_indices].copy()

        orig_ri = test_obs["row_index"].values.astype(np.int64)
        new_ri = -(1 + orig_ri)
        test_obs["row_index"] = new_ri
        test_obs["p10_split"] = "test"
        test_obs.index = [f"row_{r}" for r in new_ri]

        new_X = np.vstack([adata_fit_val.X, test_X])
        new_X_ctl = np.vstack([adata_fit_val.obsm["X_ctl"], test_X_ctl])
        new_obs = pd.concat([adata_fit_val.obs, test_obs], axis=0)

        adata = ad.AnnData(X=new_X, obs=new_obs, obsm={"X_ctl": new_X_ctl})
        check_timing_h5ad(adata, real_test_ri)

    adata.write_h5ad(out_h5ad)


def check_timing_h5ad(adata, real_test_ri):
    """Guards (h4) and (h5) on a timing AnnData (98.16 C3). PI: a function of its own so the guards are tested on the real
    code, not re-implemented in a test."""
    # Guard (h4): no row with row_index >= 0 is labelled test, AND none of the real test rows' row_index occurs anywhere
    timing_test_mask = (adata.obs["p10_split"] == "test").values
    if (adata.obs.loc[timing_test_mask, "row_index"] >= 0).any():
        raise SystemExit("FATAL: guard (h4): test row has row_index >= 0")
    current_ri_set = set(adata.obs["row_index"].values.astype(np.int64))
    if len(set(real_test_ri).intersection(current_ri_set)) > 0:
        raise SystemExit("FATAL: guard (h4): real test row_index occurs in timing h5ad")

    # Guard (h5): the counts train / valid / test are 49,846 / 5,539 / 13,445
    n_tr = int((adata.obs["p10_split"] == "train").sum())
    n_va = int((adata.obs["p10_split"] == "valid").sum())
    n_te = int((adata.obs["p10_split"] == "test").sum())
    if (n_tr, n_va, n_te) != (p10_mlp.EXPECT_N_TRAIN, p10_mlp.EXPECT_N_VAL, p10_mlp.EXPECT_N_TEST):
        raise SystemExit(
            f"FATAL: guard (h5): split counts (train={n_tr}, valid={n_va}, test={n_te}) != "
            f"expected ({p10_mlp.EXPECT_N_TRAIN}, {p10_mlp.EXPECT_N_VAL}, {p10_mlp.EXPECT_N_TEST})"
        )


def make_cap_trainer(src, dst):
    """Generate p10_train_pertdit_cap.py from Bai et al.'s train_pertdit.py via asserted substitutions."""
    with open(src, "rb") as f:
        src_bytes = f.read()
    src_sha1 = hashlib.sha1(src_bytes).hexdigest()
    # PI (W40 review): the source is the pinned trainer, so the header's sha1 is a checked fact
    if src_sha1 != TRAINER_SHA1:
        raise SystemExit(f"FATAL: make_cap_trainer: source sha1 {src_sha1} != TRAINER_SHA1 {TRAINER_SHA1}")
    text = src_bytes.decode("utf-8")

    is_crlf = "\r\n" in text
    if is_crlf:
        text = text.replace("\r\n", "\n")

    substitutions = [
        # (c1) directly before return p.parse_args(), four new arguments
        (
            "    return p.parse_args()",
            (
                "    p.add_argument(\"--cap_s\", type=float, default=float(\"inf\"))\n"
                "    p.add_argument(\"--t0\", type=float, default=None)\n"
                "    p.add_argument(\"--t_test_s\", type=float, default=0.0)\n"
                "    p.add_argument(\"--cap_margin_s\", type=float, default=900.0)\n\n"
                "    return p.parse_args()"
            ),
        ),
        # (c2) directly after start_time = time.time()
        (
            "    start_time = time.time()",
            (
                "    start_time = time.time()\n"
                "    _T0 = args.t0 if args.t0 is not None else start_time\n"
                "    stop_reason = 'max_epochs'"
            ),
        ),
        # (c3) directly after for epoch in range(args.n_epochs):
        (
            "    for epoch in range(args.n_epochs):",
            (
                "    for epoch in range(args.n_epochs):\n"
                "        _ep_start = time.time()"
            ),
        ),
        # (c4) in the early-stopping block, set stop_reason = 'rule' and add loop cap check
        (
            (
                "        if (epoch - best_epoch >= args.early_stop_patience\n"
                "                and epoch > args.min_epochs):\n"
                "            print(f\"  Early stopping at epoch {epoch} \"\n"
                "                  f\"(no improvement since epoch {best_epoch})\")\n"
                "            break"
            ),
            (
                "        if (epoch - best_epoch >= args.early_stop_patience\n"
                "                and epoch > args.min_epochs):\n"
                "            print(f\"  Early stopping at epoch {epoch} \"\n"
                "                  f\"(no improvement since epoch {best_epoch})\")\n"
                "            stop_reason = 'rule'\n"
                "            break\n\n"
                "        # P10 (RESULTS 98.16 C2): loop control only -- stop before the session cap; their best-checkpoint test prediction follows\n"
                "        _t_ep = time.time() - _ep_start\n"
                "        if time.time() - _T0 + _t_ep + args.t_test_s + args.cap_margin_s > args.cap_s:\n"
                "            stop_reason = 'cap'\n"
                "            print(f\"  P10 cap: stopping after epoch {epoch} (elapsed {time.time() - _T0:.0f}s of cap {args.cap_s:.0f}s)\")\n"
                "            break"
            ),
        ),
        # (c5) in resource_log, directly after "best_epoch": int(best_epoch),
        (
            "        \"best_epoch\": int(best_epoch),",
            (
                "        \"best_epoch\": int(best_epoch),\n"
                "        \"stop_reason\": stop_reason,\n"
                "        \"cap_s\": (None if args.cap_s == float('inf') else args.cap_s),"
            ),
        ),
    ]

    for old, new in substitutions:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"FATAL: make_cap_trainer: pattern expected exactly once, found {count}:\n{old}"
            )
        text = text.replace(old, new, 1)

    # (c6) header comment line naming source sha1 and "RESULTS 98.16 C2, P10 arm D"
    header = f"# Source sha1 {src_sha1}; RESULTS 98.16 C2, P10 arm D\n"
    res = header + text
    if is_crlf:
        res = res.replace("\n", "\r\n")

    ast.parse(res)

    if dst is not None:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="" if is_crlf else "\n") as f:
            f.write(res)
    return res


def check_trainer_cfg(src_text):
    """Assert cfg=False occurs exactly 3 times and cfg=True occurs 0 times as word tokens."""
    matches_false = re.findall(r"\bcfg=False\b", src_text)
    matches_true = re.findall(r"\bcfg=True\b", src_text)
    if len(matches_false) != 3 or len(matches_true) != 0:
        raise SystemExit(
            f"FATAL: check_trainer_cfg: expected \\bcfg=False\\b exactly 3 times (found {len(matches_false)}) "
            f"and \\bcfg=True\\b exactly 0 times (found {len(matches_true)})"
        )


def check_embeddings(drug_emb_pkl, dose_emb_pt, h5ad):
    """Validate drug and dose embeddings before any GPU work (98.16 C5, Ask 6)."""
    drug_emb = torch.load(drug_emb_pkl, map_location="cpu")
    dose_emb = torch.load(dose_emb_pt, map_location="cpu")

    if isinstance(h5ad, (str, os.PathLike)):
        adata = ad.read_h5ad(h5ad)
    else:
        adata = h5ad

    # (e1) drug keys equal set(range(N_COMPOUNDS)) | {'negative_ctrl'}, and cover every obs['pert_idx'] in all 3 splits
    expected_drug_keys = set(range(N_COMPOUNDS)) | {"negative_ctrl"}
    actual_drug_keys = set(drug_emb.keys())
    if actual_drug_keys != expected_drug_keys:
        raise SystemExit(
            f"FATAL: guard (e1): drug_emb keys mismatch: "
            f"missing={len(expected_drug_keys - actual_drug_keys)}, extra={len(actual_drug_keys - expected_drug_keys)}"
        )
    h5ad_pert_indices = set(adata.obs["pert_idx"].values.astype(int))
    if not h5ad_pert_indices.issubset(actual_drug_keys):
        missing = h5ad_pert_indices - actual_drug_keys
        raise SystemExit(f"FATAL: guard (e1): drug_emb does not cover all h5ad pert_idx: {missing}")

    # (e2) negative_ctrl is all-NaN, (2, 1024)
    neg = drug_emb["negative_ctrl"]
    if not isinstance(neg, torch.Tensor) or neg.shape != (2, 1024):
        raise SystemExit(
            f"FATAL: guard (e2): negative_ctrl shape is not (2, 1024): {getattr(neg, 'shape', None)}"
        )
    if not torch.isnan(neg).all().item():
        raise SystemExit("FATAL: guard (e2): negative_ctrl is not all-NaN")

    # (e3) every drug tensor is 2-D (L, 1024) and finite
    for k in range(N_COMPOUNDS):
        t = drug_emb[k]
        if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[1] != 1024:
            raise SystemExit(
                f"FATAL: guard (e3): drug tensor {k} shape is not 2-D (L, 1024): {getattr(t, 'shape', None)}"
            )
        if not torch.isfinite(t).all().item():
            raise SystemExit(f"FATAL: guard (e3): drug tensor {k} contains non-finite values")

    # (e4) dose dict keys equal DOSES exactly, and every obs['dose_r4'] value is a key
    if not isinstance(dose_emb, dict):
        raise SystemExit("FATAL: guard (e4): dose_emb must be a dict")
    dose_keys = set(dose_emb.keys())
    expected_doses = set(DOSES)
    if dose_keys != expected_doses:
        raise SystemExit(
            f"FATAL: guard (e4): dose_emb keys mismatch: {dose_keys} != {expected_doses}"
        )
    # Check exact lookup trainer does: dose_emb[v] for each v in obs['dose_r4'].values.astype(float)
    for v in adata.obs["dose_r4"].values.astype(float):
        try:
            _ = dose_emb[v]
        except KeyError:
            raise SystemExit(f"FATAL: guard (e4): dose_r4 value {v} lookup failed in dose_emb")


def convert_outputs_d(h5ad, run_dir, seed, out_npz):
    """Apply output inverse to PertDiT predictions and persist standard npz (98.15 ruling 1, 98.16 Ask 6)."""
    if isinstance(h5ad, (str, os.PathLike)):
        adata = ad.read_h5ad(h5ad)
    else:
        adata = h5ad

    test_mask = (adata.obs["p10_split"] == "test").values
    test_obs = adata.obs[test_mask]
    row_index = np.asarray(test_obs["row_index"], dtype=np.int64)
    n_test = len(row_index)

    # Raw inputs
    Xc = np.clip(np.asarray(adata.obsm["X_ctl"], dtype=np.float64), 0, 1e3)
    Sc_all = Xc.sum(axis=1)
    S_ctl = Sc_all[test_mask]
    Xc_test = Xc[test_mask]

    # T_train = median of clip(X, 0, 1e3).sum(1) over train + valid rows
    train_valid_mask = adata.obs["p10_split"].isin(["train", "valid"]).values
    X_raw = np.clip(np.asarray(adata.X, dtype=np.float64), 0, 1e3)
    T_train = float(np.median(X_raw[train_valid_mask].sum(axis=1)))

    # Guard (v1): |T_train - T_TRAIN_PIN| / T_TRAIN_PIN <= 1e-6
    rel_diff = abs(T_train - T_TRAIN_PIN) / T_TRAIN_PIN
    if rel_diff > 1e-6:
        raise SystemExit(
            f"FATAL: guard (v1): T_train relative error {rel_diff:.6e} > 1e-6 ({T_train} vs {T_TRAIN_PIN})"
        )

    # Load predictions
    pred_path = os.path.join(run_dir, "predictions", "test_predictions.npy")
    ctrl_path = os.path.join(run_dir, "predictions", "test_ctrl.npy")
    if not os.path.isfile(pred_path):
        raise SystemExit(f"FATAL: missing predictions file: {pred_path}")
    if not os.path.isfile(ctrl_path):
        raise SystemExit(f"FATAL: missing control file: {ctrl_path}")

    pred_t = np.load(pred_path)
    ctrl_t = np.load(ctrl_path)

    # Guard (v2): both are (n_test, 978)
    if pred_t.shape != (n_test, 978) or ctrl_t.shape != (n_test, 978):
        raise SystemExit(
            f"FATAL: guard (v2): prediction shapes {pred_t.shape}, {ctrl_t.shape} != ({n_test}, 978)"
        )

    # Guard (v3): alignment guard
    # Recompute ctrl_t_ref = log1p(Xc * Tc / Sc_all) over ALL h5ad rows, with Tc = median(Sc_all[Sc_all > 0])
    Tc = np.median(Sc_all[Sc_all > 0])
    ctrl_t_ref_all = np.log1p(Xc * (Tc / Sc_all[:, None]))
    ctrl_t_ref = ctrl_t_ref_all[test_mask]

    max_align_diff = float(np.max(np.abs(ctrl_t_ref - ctrl_t)))
    if max_align_diff > 1e-4:
        raise SystemExit(
            f"FATAL: guard (v3): alignment guard failed: max |ctrl_t_ref - ctrl_t| = {max_align_diff:.6e} > 1e-4"
        )

    # The inverse: X_hat = expm1(pred_t + ctrl_t) * (S_ctl / T_train)[:, None]
    X_hat = np.expm1(pred_t.astype(np.float64) + ctrl_t.astype(np.float64)) * (S_ctl / T_train)[:, None]
    delta_pred = (X_hat - Xc_test).astype(np.float32)

    # Guard (v4): all finite
    if not np.isfinite(delta_pred).all():
        raise SystemExit("FATAL: guard (v4): non-finite values in delta_pred")

    # Load resource_log
    log_path = os.path.join(run_dir, "logs", "resource_log.json")
    if not os.path.isfile(log_path):
        raise SystemExit(f"FATAL: missing resource_log at {log_path}")
    with open(log_path, "r", encoding="utf-8") as f:
        rlog = json.load(f)

    # Guard (v5): seed and n_test match
    if int(rlog.get("seed", -1)) != int(seed):
        raise SystemExit(f"FATAL: guard (v5): resource_log seed {rlog.get('seed')} != {seed}")
    if int(rlog.get("n_test", -1)) != n_test:
        raise SystemExit(f"FATAL: guard (v5): resource_log n_test {rlog.get('n_test')} != {n_test}")

    stop_reason = rlog.get("stop_reason", "")
    best_epoch = int(rlog.get("best_epoch", 0))
    n_epochs_run = int(rlog.get("n_epochs_run", 0))

    # Stage write: under temp ending in .npz, copy -> .part -> os.replace
    out_dir = os.path.dirname(os.path.abspath(out_npz))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=out_dir, delete=False) as tmp_f:
        tmp_npz_path = tmp_f.name

    np.savez(
        tmp_npz_path,
        row_index=row_index,
        delta_pred=delta_pred,
        delta_pred_t=pred_t.astype(np.float32),
        ctrl_t=ctrl_t.astype(np.float32),
        seed=np.int64(seed),
        arm=np.str_("pertdit"),
        stop_reason=np.str_(stop_reason),
        best_epoch=np.int64(best_epoch),
        n_epochs_run=np.int64(n_epochs_run),
        t_train=np.float64(T_train),
    )

    part_path = out_npz + ".part"
    shutil.copyfile(tmp_npz_path, part_path)
    os.replace(part_path, out_npz)
    try:
        os.remove(tmp_npz_path)
    except OSError:
        pass


def parse_timing(stamped_log, kernel_t0):
    """Parse stamped log to extract t_pre, t_epoch, t_test."""
    if isinstance(stamped_log, (str, os.PathLike)) and os.path.isfile(stamped_log):
        with open(stamped_log, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(stamped_log, str):
        lines = stamped_log.splitlines()
    else:
        lines = list(stamped_log)

    ts_step5 = None
    ts_ep0 = None
    ts_ep2 = None
    ts_step6 = None
    ts_step7 = None

    for line in lines:
        line_str = line.strip()
        if not line_str or "\t" not in line:
            continue
        parts = line.split("\t", 1)
        try:
            ts = float(parts[0])
        except ValueError:
            continue
        msg = parts[1]

        if ts_step5 is None and "[5/7] Training" in msg:
            ts_step5 = ts
        if ts_ep0 is None and msg.startswith("  [Epoch   0]"):
            ts_ep0 = ts
        if ts_ep2 is None and msg.startswith("  [Epoch   2]"):
            ts_ep2 = ts
        if ts_step6 is None and "[6/7] Predicting on test set" in msg:
            ts_step6 = ts
        if ts_step7 is None and "[7/7] Saving unified output" in msg:
            ts_step7 = ts

    if ts_step5 is None:
        raise SystemExit("FATAL: parse_timing missing marker: '[5/7] Training'")
    if ts_ep0 is None:
        raise SystemExit("FATAL: parse_timing missing marker: '  [Epoch   0]'")
    if ts_ep2 is None:
        raise SystemExit("FATAL: parse_timing missing marker: '  [Epoch   2]'")
    if ts_step6 is None:
        raise SystemExit("FATAL: parse_timing missing marker: '[6/7] Predicting on test set'")
    if ts_step7 is None:
        raise SystemExit("FATAL: parse_timing missing marker: '[7/7] Saving unified output'")

    t_pre = ts_step5 - kernel_t0
    t_epoch = (ts_ep2 - ts_ep0) / 2.0
    t_test = ts_step7 - ts_step6

    return {"t_pre": float(t_pre), "t_epoch": float(t_epoch), "t_test": float(t_test)}


def timing_decision(timings, cap_s=CAP_S):
    """Compute projection and timing decision (RESULTS §98.15 ruling 5)."""
    if isinstance(timings, dict):
        timing_list = list(timings.values())
    else:
        timing_list = list(timings)

    t_pre = max(t["t_pre"] for t in timing_list)
    t_epoch = max(t["t_epoch"] for t in timing_list)
    t_test = max(t["t_test"] for t in timing_list)

    P62 = t_pre + 62.0 * t_epoch + t_test
    P1000 = t_pre + 1000.0 * t_epoch + t_test

    if P1000 <= cap_s:
        decision = "run"
    elif P62 > cap_s:
        decision = "not_run_cost"
    else:
        decision = "run_capped"

    return {
        "P62": float(P62),
        "P1000": float(P1000),
        "decision": decision,
        "cap_s": float(cap_s),
        "t_test_s": float(t_test),
    }


def main():
    parser = argparse.ArgumentParser(description="P10 arm D (PertDiT) adapter and tools")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # build
    p_build = subparsers.add_parser("build")
    p_build.add_argument("--bundle", required=True, help="Path to bundle npz")
    p_build.add_argument("--carves", required=True, help="Path to carves json")
    p_build.add_argument("--compound_map", required=True, help="Path to compound map json")
    p_build.add_argument("--dose_list", required=True, help="Path to dose list json")
    p_build.add_argument("--out", required=True, help="Path to output h5ad")
    p_build.add_argument("--timing", action="store_true", default=False, help="Build timing h5ad")

    # captrainer
    p_cap = subparsers.add_parser("captrainer")
    p_cap.add_argument("--src", required=True, help="Source train_pertdit.py")
    p_cap.add_argument("--dst", required=True, help="Destination cap trainer script")

    # convert
    p_conv = subparsers.add_parser("convert")
    p_conv.add_argument("--h5ad", required=True, help="Path to h5ad")
    p_conv.add_argument("--run_dir", required=True, help="Path to run output directory")
    p_conv.add_argument("--seed", type=int, required=True, help="Random seed")
    p_conv.add_argument("--out", required=True, help="Path to output npz")

    # decide
    p_dec = subparsers.add_parser("decide")
    p_dec.add_argument("--timing_json", required=True, help="Path to timing json")

    args = parser.parse_args()
    if args.cmd == "build":
        build_h5ad_d(
            args.bundle,
            args.carves,
            args.compound_map,
            args.dose_list,
            args.out,
            timing=args.timing,
        )
    elif args.cmd == "captrainer":
        make_cap_trainer(args.src, args.dst)
    elif args.cmd == "convert":
        convert_outputs_d(args.h5ad, args.run_dir, args.seed, args.out)
    elif args.cmd == "decide":
        with open(args.timing_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        dec = timing_decision(data.get("timings", data))
        print(json.dumps(dec, indent=2))


if __name__ == "__main__":
    main()
