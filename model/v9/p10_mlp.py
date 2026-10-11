# -*- coding: utf-8 -*-
"""RESULTS §98.2, §98.6: P10 arm M (Bai et al.'s drug-free MLP on split_cold_drug_1).

Module 1: Adapter and tools for arm M and the M+dt variant.
- build_h5ad: reads xpert_mdmt_splits.npz and p10_carves.json, validates guards,
  and writes an AnnData h5ad with p10_split ('train', 'valid', 'test').
- make_dt_script: generates M+dt trainer from Bai et al.'s train_mlp.py via asserted
  exact substitutions.
- convert_outputs: packages test predictions and run metadata into standard npz format.
"""
import argparse
import hashlib
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

# Binding constants for the real bundle and carve (RESULTS 98.2, 98.6)
CARVES_SHA1 = "d6fc78447228b87641cbd5af8ff0143a714edeb8"
VAL_SHA1 = "ba6996e6c15eddd57e3364be1371a0285c2880af"
EXPECT_N_TRAIN = 49846
EXPECT_N_VAL = 5539
EXPECT_N_TEST = 13445
EXPECT_TRAIN_SHA1 = "d10bd1265e6c0ddb973ca7b418792d7c4e1c41e4"
EXPECT_TEST_SHA1 = "1d2b49a0947da6787c818afa187f125782c5c9de"


def sha1_file(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def sha1_sorted_rows(arr):
    return hashlib.sha1(np.sort(np.asarray(arr, dtype=np.int64)).tobytes()).hexdigest()


def build_h5ad(bundle_npz, carves_json, out_h5ad, with_dt=False):
    """Build AnnData h5ad from bundle npz and carve json with guards (a)-(e)."""
    # Guard (a): carves_json sha1
    if isinstance(carves_json, (str, os.PathLike)):
        with open(carves_json, "rb") as f:
            carves_bytes = f.read()
        got_carves_sha1 = hashlib.sha1(carves_bytes).hexdigest()
        if got_carves_sha1 != CARVES_SHA1:
            raise SystemExit(
                f"FATAL: guard (a): carves_json sha1 mismatch: {got_carves_sha1} != {CARVES_SHA1}"
            )
        carves_data = json.loads(carves_bytes.decode("utf-8"))
    else:
        # PI (mutation check): no dict path -- guard (a) must never be bypassable
        raise SystemExit("FATAL: carves_json must be a file path (guard (a) pins its sha1)")

    # Load bundle
    if isinstance(bundle_npz, (str, os.PathLike)):
        bundle = np.load(bundle_npz)
    else:
        bundle = bundle_npz

    row_index = np.asarray(bundle["row_index"], dtype=np.int64)

    # Guard (e): row_index uniqueness
    if len(np.unique(row_index)) != len(row_index):
        raise SystemExit("FATAL: guard (e): row_index is not unique")

    # Guard (b): carve sorted row_index sha1 and carve rows are training rows
    val_row_index = np.asarray(
        carves_data["validation_carve"]["val_row_index"], dtype=np.int64
    )
    got_val_sha1 = sha1_sorted_rows(val_row_index)
    if got_val_sha1 != VAL_SHA1:
        raise SystemExit(
            f"FATAL: guard (b): carve sorted row_index sha1 mismatch: {got_val_sha1} != {VAL_SHA1}"
        )

    split_col = np.asarray(bundle["split_split_cold_drug_1"])
    bundle_train_ri = set(row_index[split_col == "train"])
    val_ri_set = set(val_row_index)
    if not val_ri_set.issubset(bundle_train_ri):
        raise SystemExit(
            "FATAL: guard (b): carve rows are not all training rows in split_split_cold_drug_1"
        )

    # Form p10_split
    p10_split = np.empty(len(row_index), dtype=object)
    val_mask = np.isin(row_index, val_row_index)
    p10_split[val_mask] = "valid"
    train_mask = (split_col == "train") & (~val_mask)
    p10_split[train_mask] = "train"
    test_mask = split_col == "test"
    p10_split[test_mask] = "test"

    if np.any(p10_split == None):  # noqa: E711
        raise SystemExit("FATAL: unassigned rows found in p10_split")

    # Guard (c): counts train / valid / test
    n_train = int(np.sum(p10_split == "train"))
    n_val = int(np.sum(p10_split == "valid"))
    n_test = int(np.sum(p10_split == "test"))
    if (n_train, n_val, n_test) != (EXPECT_N_TRAIN, EXPECT_N_VAL, EXPECT_N_TEST):
        raise SystemExit(
            f"FATAL: guard (c): split counts (train={n_train}, valid={n_val}, test={n_test}) != "
            f"expected ({EXPECT_N_TRAIN}, {EXPECT_N_VAL}, {EXPECT_N_TEST})"
        )

    # Guard (d): sorted sha1s for train and test
    train_ri = row_index[p10_split == "train"]
    got_train_sha1 = sha1_sorted_rows(train_ri)
    if got_train_sha1 != EXPECT_TRAIN_SHA1:
        raise SystemExit(
            f"FATAL: guard (d): train sorted row_index sha1 mismatch: {got_train_sha1} != {EXPECT_TRAIN_SHA1}"
        )

    test_ri = row_index[p10_split == "test"]
    got_test_sha1 = sha1_sorted_rows(test_ri)
    if got_test_sha1 != EXPECT_TEST_SHA1:
        raise SystemExit(
            f"FATAL: guard (d): test sorted row_index sha1 mismatch: {got_test_sha1} != {EXPECT_TEST_SHA1}"
        )

    # AnnData creation
    obs = pd.DataFrame(
        {
            "row_index": row_index,
            "pert_id": bundle["meta_pert_id"],
            "cell": bundle["meta_cell"],
            "dose": bundle["meta_dose"],
            "time": bundle["meta_time"],
            "pert_idx": bundle["meta_pert_idx"],
            "cell_idx": bundle["meta_cell_idx"],
            "p10_split": p10_split,
        }
    )
    obs.index = [f"row_{r}" for r in row_index]
    obs.index.name = None

    obsm = {
        "X_ctl": np.asarray(bundle["X_ctl"], dtype=np.float32),
    }

    if with_dt:
        dose = np.asarray(bundle["meta_dose"], dtype=np.float64)
        time = np.asarray(bundle["meta_time"], dtype=np.float64)
        fit_mask = p10_split == "train"

        ld = np.log10(np.clip(dose, 1e-4, None))
        mu_d = ld[fit_mask].mean()
        sigma_d = ld[fit_mask].std()
        dose_n = ((ld - mu_d) / (sigma_d + 1e-6)).astype(np.float32)

        mu_t = time[fit_mask].mean()
        sigma_t = time[fit_mask].std()
        time_n = ((time - mu_t) / (sigma_t + 1e-6)).astype(np.float32)

        obsm["cond"] = np.column_stack([dose_n, time_n]).astype(np.float32)

    adata = ad.AnnData(
        X=np.asarray(bundle["X"], dtype=np.float32),
        obs=obs,
        obsm=obsm,
    )
    adata.write_h5ad(out_h5ad)


def make_dt_script(src, dst):
    """Generate M+dt trainer from Bai et al.'s train_mlp.py via asserted exact substitutions."""
    with open(src, "rb") as f:
        src_bytes = f.read()
    src_sha1 = hashlib.sha1(src_bytes).hexdigest()
    text = src_bytes.decode("utf-8")

    substitutions = [
        (
            "        for i in range(num_layers):\n            in_dim = self.X1_DIM if i == 0 else mlp_hidden",
            "        for i in range(num_layers):\n            in_dim = (self.X1_DIM + 2) if i == 0 else mlp_hidden",
        ),
        (
            "    def forward(self, x1):\n        h = x1",
            "    def forward(self, x1, c):\n        h = torch.cat([x1, c], 1)",
        ),
        (
            "        self.x1 = adata.obsm['X_ctl'][mask].astype(np.float32)\n        self.x2 = adata.X[mask].astype(np.float32)",
            "        self.x1 = adata.obsm['X_ctl'][mask].astype(np.float32)\n        self.x2 = adata.X[mask].astype(np.float32)\n        self.c = adata.obsm['cond'][mask].astype(np.float32)",
        ),
        (
            "    def __getitem__(self, idx):\n        return (torch.from_numpy(self.x1[idx]),\n                torch.from_numpy(self.x2[idx]))",
            "    def __getitem__(self, idx):\n        return (torch.from_numpy(self.x1[idx]),\n                torch.from_numpy(self.x2[idx]),\n                torch.from_numpy(self.c[idx]))",
        ),
        (
            "def train_epoch(model, optimizer, loader, device):\n    model.train()\n    total_loss, total_mse, total_deg, n = 0, 0, 0, 0\n    for x1, x2 in loader:\n        x1, x2 = x1.to(device), x2.to(device)\n        delta = model(x1)",
            "def train_epoch(model, optimizer, loader, device):\n    model.train()\n    total_loss, total_mse, total_deg, n = 0, 0, 0, 0\n    for x1, x2, c in loader:\n        x1, x2, c = x1.to(device), x2.to(device), c.to(device)\n        delta = model(x1, c)",
        ),
        (
            "def eval_epoch(model, loader, device):\n    model.eval()\n    total_mse, total_deg, n = 0, 0, 0\n    for x1, x2 in loader:\n        x1, x2 = x1.to(device), x2.to(device)\n        delta = model(x1)",
            "def eval_epoch(model, loader, device):\n    model.eval()\n    total_mse, total_deg, n = 0, 0, 0\n    for x1, x2, c in loader:\n        x1, x2, c = x1.to(device), x2.to(device), c.to(device)\n        delta = model(x1, c)",
        ),
        (
            "    all_x1, all_x2, all_pred = [], [], []\n    with torch.no_grad():\n        for x1, x2 in test_loader:\n            x1 = x1.to(device)\n            delta = model(x1)",
            "    all_x1, all_x2, all_pred = [], [], []\n    with torch.no_grad():\n        for x1, x2, c in test_loader:\n            x1, c = x1.to(device), c.to(device)\n            delta = model(x1, c)",
        ),
    ]

    is_crlf = "\r\n" in text
    if is_crlf:
        text = text.replace("\r\n", "\n")

    for old, new in substitutions:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"FATAL: make_dt_script: pattern expected once, found {count} occurrences:\n{old}"
            )
        text = text.replace(old, new, 1)

    header = f"# Source sha1 {src_sha1}; RESULTS 98.6 C5, M+dt, reported only\n"
    res = header + text
    if is_crlf:
        res = res.replace("\n", "\r\n")

    with open(dst, "w", encoding="utf-8", newline="" if is_crlf else "\n") as f:
        f.write(res)


def convert_outputs(h5ad, run_dir, seed, arm, out_npz):
    """Convert test_predictions.npy and run metadata into standard npz format."""
    pred_path = os.path.join(run_dir, "predictions", "test_predictions.npy")
    if not os.path.isfile(pred_path):
        raise SystemExit(f"FATAL: missing test predictions file: {pred_path}")

    log_path = os.path.join(run_dir, "logs", "resource_log.json")
    if not os.path.isfile(log_path):
        raise SystemExit(f"FATAL: missing resource log file: {log_path}")

    deg_pred = np.load(pred_path)
    with open(log_path, "r", encoding="utf-8") as f:
        rlog = json.load(f)

    best_epoch = int(rlog["best_epoch"])
    n_epochs_run = int(rlog["n_epochs_run"])

    adata = ad.read_h5ad(h5ad)
    test_mask = adata.obs["p10_split"] == "test"
    test_obs = adata.obs[test_mask]
    row_index = np.asarray(test_obs["row_index"], dtype=np.int64)

    if len(deg_pred) != len(row_index):
        raise SystemExit(
            f"FATAL: prediction count {len(deg_pred)} != test count {len(row_index)}"
        )

    if not np.isfinite(deg_pred).all():
        raise SystemExit("FATAL: non-finite values in predictions")

    if deg_pred.ndim != 2 or deg_pred.shape[1] != 978:
        raise SystemExit(
            f"FATAL: invalid prediction shape {deg_pred.shape}, expected (N, 978)"
        )

    if arm not in ("m", "mdt"):
        raise SystemExit(f"FATAL: unknown arm {arm}, expected 'm' or 'mdt'")

    np.savez(
        out_npz,
        row_index=row_index,
        delta_pred=deg_pred.astype(np.float32),
        seed=np.int64(seed),
        arm=np.str_(arm),
        best_epoch=np.int64(best_epoch),
        n_epochs_run=np.int64(n_epochs_run),
    )


def main():
    parser = argparse.ArgumentParser(description="P10 MLP adapter and tools")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # build
    p_build = subparsers.add_parser("build")
    p_build.add_argument("--bundle", required=True, help="Path to bundle npz")
    p_build.add_argument("--carves", required=True, help="Path to carves json")
    p_build.add_argument("--out", required=True, help="Path to output h5ad")
    p_build.add_argument(
        "--with_dt",
        action="store_true",
        default=False,
        help="Include dose and time conditioning",
    )

    # dtscript
    p_dt = subparsers.add_parser("dtscript")
    p_dt.add_argument("--src", required=True, help="Source train_mlp.py")
    p_dt.add_argument("--dst", required=True, help="Destination M+dt script")

    # convert
    p_conv = subparsers.add_parser("convert")
    p_conv.add_argument("--h5ad", required=True, help="Path to h5ad")
    p_conv.add_argument(
        "--run_dir", required=True, help="Path to run output directory"
    )
    p_conv.add_argument("--seed", type=int, required=True, help="Random seed")
    p_conv.add_argument(
        "--arm", required=True, choices=["m", "mdt"], help="Arm (m or mdt)"
    )
    p_conv.add_argument("--out", required=True, help="Path to output npz")

    args = parser.parse_args()
    if args.cmd == "build":
        build_h5ad(args.bundle, args.carves, args.out, with_dt=args.with_dt)
    elif args.cmd == "dtscript":
        make_dt_script(args.src, args.dst)
    elif args.cmd == "convert":
        convert_outputs(args.h5ad, args.run_dir, args.seed, args.arm, args.out)


if __name__ == "__main__":
    main()
