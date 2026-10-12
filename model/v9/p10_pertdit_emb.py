# -*- coding: utf-8 -*-
"""RESULTS §98.15, §98.16: P10 arm D (PertDiT) drug and dose embeddings precompute.

Module 1: Adapter and precompute utilities for PertDiT embeddings:
- compound_map: maps bundle meta_pert_idx to compact 0..n-1 indices.
- dose_values: extracts unique sorted dose values with rounding guards.
- build_inputs: produces input dataset files (idx2smi_all, slices, placeholder, dose_list, compound_map, manifest).
- make_emb_script: generates prepare_pertdit_embeddings_p10.py via asserted exact substitutions.
- merge_caches: merges caption caches across slices with consistency guards.
- check_outputs: validates final embedding shapes, types, negative_ctrl, and dose embeddings.
"""
import argparse
import ast
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import torch

SRC_SHA1 = "2bea30573a174d039b1d6de2569ad562f1aee4e5"
IDX2SMI_FULL_SHA1 = "682fd4afbe5f0f5f4eb7af3437c056797f28f544"
EXPECT_N_COMPOUNDS = 1970
EXPECT_DOSES = [0.1, 0.37, 0.62, 0.77, 1.11, 2.31, 3.33, 5.0, 10.0, 40.0]
N_SLICES = 5


def fatal(msg):
    raise SystemExit("FATAL: " + msg)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compound_map(bundle_npz):
    """Return {int(xpert_idx): compact} mapping for bundle compounds.

    Reads ONLY meta_pert_idx via np.load(..., allow_pickle=False).
    Never reads X or X_ctl.
    """
    if isinstance(bundle_npz, (str, os.PathLike)):
        with np.load(bundle_npz, allow_pickle=False) as npz:
            pert_idx = npz["meta_pert_idx"]
    else:
        pert_idx = bundle_npz["meta_pert_idx"]

    unique_perts = sorted({int(x) for x in pert_idx})
    return {int(idx): rank for rank, idx in enumerate(unique_perts)}


def dose_values(bundle_npz):
    """Return sorted unique dose values rounded to 4 decimals as floats.

    Reads ONLY meta_dose.
    Guards:
    - f-string f"The dosage is {10.0} micromoles." matches their exact literal.
    - list equals EXPECT_DOSES.
    """
    if isinstance(bundle_npz, (str, os.PathLike)):
        with np.load(bundle_npz, allow_pickle=False) as npz:
            dose_arr = npz["meta_dose"]
    else:
        dose_arr = bundle_npz["meta_dose"]

    f_str = f"The dosage is {10.0} micromoles."
    literal_str = "The dosage is 10.0 micromoles."
    if f_str != literal_str:
        fatal(f"dose template string mismatch: {f_str!r} != {literal_str!r}")

    doses = sorted({float(round(float(d), 4)) for d in np.unique(dose_arr)})
    if doses != EXPECT_DOSES:
        fatal(f"dose_values guard: dose list mismatch: {doses} != {EXPECT_DOSES}")
    return doses


def build_inputs(bundle_npz, idx2smi_npy, out_dir):
    """Build input files for PertDiT embedding precompute in out_dir."""
    # PI (W39 review): no in-memory path -- guard (g1) must never be bypassable, and the bundle's sha1 is recorded
    if not isinstance(idx2smi_npy, (str, os.PathLike)) or not isinstance(bundle_npz, (str, os.PathLike)):
        fatal("build_inputs: bundle_npz and idx2smi_npy must be file paths (guard (g1) pins the table's sha1)")
    os.makedirs(out_dir, exist_ok=True)

    # Guard (g1): idx2smi sha1
    got_idx2smi_sha1 = sha1_file(idx2smi_npy)
    if got_idx2smi_sha1 != IDX2SMI_FULL_SHA1:
        fatal(f"guard (g1): idx2smi_npy sha1 mismatch: {got_idx2smi_sha1} != {IDX2SMI_FULL_SHA1}")
    idx2smi_table = np.load(idx2smi_npy, allow_pickle=True).item()

    cmap = compound_map(bundle_npz)

    # Guard (g2): every bundle xpert_idx is a key of idx2smi_table
    missing = [idx for idx in cmap.keys() if idx not in idx2smi_table]
    if missing:
        fatal(f"guard (g2): bundle compound indices missing from idx2smi table: {missing[:5]}")

    # Guard (g3): compounds map 1-to-1 onto SMILES and count equals EXPECT_N_COMPOUNDS
    bundle_smiles = [idx2smi_table[idx] for idx in cmap.keys()]
    unique_smiles = set(bundle_smiles)
    n_compounds = len(cmap)
    if len(unique_smiles) != n_compounds or n_compounds != EXPECT_N_COMPOUNDS:
        fatal(
            f"guard (g3): compounds do not map 1-to-1 onto SMILES: "
            f"{len(unique_smiles)} unique SMILES for {n_compounds} compounds (expected {EXPECT_N_COMPOUNDS})"
        )

    # 1. idx2smi_all.npy: {compact: smiles}
    idx2smi_all = {compact: idx2smi_table[xpert_idx] for xpert_idx, compact in cmap.items()}
    all_path = os.path.join(out_dir, "idx2smi_all.npy")
    np.save(all_path, idx2smi_all, allow_pickle=True)

    # 2. Slices: idx2smi_slice{k}.npy
    sorted_all_smiles = sorted(unique_smiles, key=lambda s: (-len(s), s))
    slices_smiles = [[] for _ in range(N_SLICES)]
    for idx, s in enumerate(sorted_all_smiles):
        slices_smiles[idx % N_SLICES].append(s)

    # Disjoint and union assertions
    seen_smiles = set()
    for k in range(N_SLICES):
        sk_set = set(slices_smiles[k])
        assert not (sk_set & seen_smiles), f"Slice {k} overlaps with previous slices"
        seen_smiles.update(sk_set)
    assert seen_smiles == unique_smiles, "Union of slices does not equal all unique SMILES"

    slice_paths = []
    for k in range(N_SLICES):
        slice_sorted = sorted(slices_smiles[k])
        slice_dict = {i: s for i, s in enumerate(slice_sorted)}
        sp = os.path.join(out_dir, f"idx2smi_slice{k}.npy")
        np.save(sp, slice_dict, allow_pickle=True)
        slice_paths.append(sp)

    # 3. Placeholder: pert_smiles_emb_placeholder.pkl
    placeholder_path = os.path.join(out_dir, "pert_smiles_emb_placeholder.pkl")
    torch.save({"negative_ctrl": torch.full((2, 1024), float("nan"))}, placeholder_path)

    # 4. Dose list: dose_list.json
    doses = dose_values(bundle_npz)
    dose_list_path = os.path.join(out_dir, "dose_list.json")
    with open(dose_list_path, "w", encoding="utf-8") as f:
        json.dump(doses, f, indent=2)

    # 5. Compound map: compound_map.json ({str(xpert_idx): compact}, sorted numerically)
    cmap_json = {str(k): cmap[k] for k in sorted(cmap.keys(), key=int)}
    cmap_path = os.path.join(out_dir, "compound_map.json")
    with open(cmap_path, "w", encoding="utf-8") as f:
        json.dump(cmap_json, f, indent=2)

    # 6. Manifest: inputs_manifest.json
    bundle_sha1 = sha1_file(bundle_npz)
    files_to_hash = [
        "idx2smi_all.npy",
        *[f"idx2smi_slice{k}.npy" for k in range(N_SLICES)],
        "pert_smiles_emb_placeholder.pkl",
        "dose_list.json",
        "compound_map.json",
    ]
    file_sha1s = {fname: sha1_file(os.path.join(out_dir, fname)) for fname in files_to_hash}

    manifest = {
        "n_compounds": n_compounds,
        "slice_sizes": [len(slices_smiles[k]) for k in range(N_SLICES)],
        "source_sha1s": {
            "idx2smi": got_idx2smi_sha1,
            "bundle": bundle_sha1,
        },
        "file_sha1s": file_sha1s,
        "sha1s": file_sha1s,
    }
    manifest_path = os.path.join(out_dir, "inputs_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def make_emb_script(src, dst):
    """Generate prepare_pertdit_embeddings_p10.py from Bai et al. source via asserted substitutions."""
    # PI (W39 review): the source is a file whose sha1 is pinned, so the header's sha1 claim is checked, not asserted
    if not isinstance(src, (str, os.PathLike)) or not os.path.isfile(src):
        fatal("make_emb_script: src must be a file path (its sha1 is pinned)")
    got_src_sha1 = sha1_file(src)
    if got_src_sha1 != SRC_SHA1:
        fatal(f"make_emb_script: source sha1 {got_src_sha1} != SRC_SHA1 {SRC_SHA1}")
    with open(src, "r", encoding="utf-8") as f:
        code = f.read()

    # (iii) Header comment line naming source sha1 and "RESULTS 98.15 / 98.16, P10 arm D"
    header = f"# Source sha1: {SRC_SHA1}; RESULTS 98.15 / 98.16, P10 arm D\n"

    # (ii) Add DOSE_LIST_PATH directly after OUT_DIR = ...
    old_out_dir = 'OUT_DIR = os.path.join(XPERT_ROOT, "data", "PertDiT", "bench_drug_emb")'
    new_out_dir = (
        'OUT_DIR = os.path.join(XPERT_ROOT, "data", "PertDiT", "bench_drug_emb")\n'
        'DOSE_LIST_PATH = os.path.join(XPERT_ROOT, "data", "PertDiT", "p10_dose_list.json")'
    )
    if code.count(old_out_dir) != 1:
        fatal(f"make_emb_script: OUT_DIR target found {code.count(old_out_dir)} times, expected exactly 1")

    # (i) CUDA gate
    old_cuda_gate = (
        "        # Generating captions requires CUDA — fail fast on CPU-only nodes.\n"
        "        if not torch.cuda.is_available():\n"
        "            sys.exit(\n"
        '                f"ERROR: {len(remaining)} SMILES still need MolT5 captions, "\n'
        '                "which require a CUDA GPU. Either run this script on a GPU "\n'
        '                "node, or copy a complete _caption_cache.pkl into "\n'
        '                f"{OUT_DIR}/ from a previous GPU run."\n'
        "            )"
    )
    new_cuda_gate = '        print(f"P10 (RESULTS 98.15 ruling 4): MolT5 captions on {device}")'
    if code.count(old_cuda_gate) != 1:
        fatal(f"make_emb_script: CUDA gate target found {code.count(old_cuda_gate)} times, expected exactly 1")

    # (ii) Dose loop
    old_dose_block = (
        '    dose_prompt = "The dosage is 10.0 micromoles."\n'
        '    with torch.no_grad():\n'
        '        inputs = tokenizer_bert(dose_prompt, return_tensors="pt").to(bert_device)\n'
        '        outputs = model_bert(**inputs)\n'
        '        dose_emb = outputs.last_hidden_state.detach().squeeze(0).cpu()\n'
        '    print(f"Dose embedding shape: {dose_emb.shape} for prompt: \\"{dose_prompt}\\"")'
    )
    new_dose_block = (
        '    import json\n'
        '    with open(DOSE_LIST_PATH, "r", encoding="utf-8") as f:\n'
        '        p10_doses = json.load(f)\n'
        '    dose_emb = {}\n'
        '    with torch.no_grad():\n'
        '        for d in p10_doses:\n'
        '            d = float(d)\n'
        '            dose_prompt = f"The dosage is {d} micromoles."\n'
        '            inputs = tokenizer_bert(dose_prompt, return_tensors="pt").to(bert_device)\n'
        '            outputs = model_bert(**inputs)\n'
        '            dose_emb[d] = outputs.last_hidden_state.detach().squeeze(0).cpu()\n'
        '            print(f"Dose embedding shape: {dose_emb[d].shape} for prompt: \\"{dose_prompt}\\"")'
    )
    if code.count(old_dose_block) != 1:
        fatal(f"make_emb_script: dose block target found {code.count(old_dose_block)} times, expected exactly 1")

    out_code = header + code
    out_code = out_code.replace(old_out_dir, new_out_dir, 1)
    out_code = out_code.replace(old_cuda_gate, new_cuda_gate, 1)
    out_code = out_code.replace(old_dose_block, new_dose_block, 1)

    # Validate ast.parse
    ast.parse(out_code)

    if isinstance(dst, (str, os.PathLike)):
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_code)
    return out_code


def merge_caches(cache_paths, idx2smi_all_npy):
    """Merge caption caches and validate guards (m1)-(m3)."""
    if isinstance(idx2smi_all_npy, (str, os.PathLike)):
        idx2smi_all = np.load(idx2smi_all_npy, allow_pickle=True).item()
    else:
        idx2smi_all = idx2smi_all_npy

    all_smiles = set(idx2smi_all.values())

    merged = {}
    for p in cache_paths:
        if isinstance(p, (str, os.PathLike)):
            with open(p, "rb") as f:
                c = pickle.load(f)
        else:
            c = p
        for smi, cap in c.items():
            # (m3) a caption is empty, or not a str
            if not isinstance(cap, str) or len(cap.strip()) == 0:
                fatal(f"guard (m3): caption for SMILES {smi!r} is empty or not a str: {cap!r}")
            # (m1) two caches give the same SMILES different captions
            if smi in merged and merged[smi] != cap:
                fatal(
                    f"guard (m1): two caches give the same SMILES different captions: "
                    f"{smi!r} has {merged[smi]!r} vs {cap!r}"
                )
            merged[smi] = cap

    # (m2) the union's key set is not exactly the set of SMILES in idx2smi_all
    if set(merged.keys()) != all_smiles:
        missing = all_smiles - set(merged.keys())
        extra = set(merged.keys()) - all_smiles
        fatal(
            f"guard (m2): merged caption keys do not match idx2smi_all SMILES: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    return merged


def check_outputs(out_dir, idx2smi_all_npy, dose_list_json):
    """Validate output files in out_dir against guards (o1)-(o4) and return sha1s."""
    drug_emb_path = os.path.join(out_dir, "bench_drug_emb.pkl")
    dose_emb_path = os.path.join(out_dir, "bench_dose_emb.pt")

    if not os.path.isfile(drug_emb_path):
        fatal(f"missing bench_drug_emb.pkl at {drug_emb_path}")
    if not os.path.isfile(dose_emb_path):
        fatal(f"missing bench_dose_emb.pt at {dose_emb_path}")

    if isinstance(idx2smi_all_npy, (str, os.PathLike)):
        idx2smi_all = np.load(idx2smi_all_npy, allow_pickle=True).item()
    else:
        idx2smi_all = idx2smi_all_npy
    n = len(idx2smi_all)

    # (o1) bench_drug_emb.pkl's keys are exactly set(range(n)) | {'negative_ctrl'}
    bench_drug_emb = torch.load(drug_emb_path, map_location="cpu")
    expected_keys = set(range(n)) | {"negative_ctrl"}
    if set(bench_drug_emb.keys()) != expected_keys:
        fatal(
            f"guard (o1): bench_drug_emb.pkl keys mismatch: "
            f"missing={expected_keys - set(bench_drug_emb.keys())}, "
            f"extra={set(bench_drug_emb.keys()) - expected_keys}"
        )

    # (o2) each drug tensor is 2-D (L, 1024) with L >= 1, float32 and finite
    for i in range(n):
        t = bench_drug_emb[i]
        if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[0] < 1 or t.shape[1] != 1024:
            fatal(
                f"guard (o2): drug tensor {i} shape is not 2-D (L, 1024) with L >= 1: {getattr(t, 'shape', None)}"
            )
        if t.dtype != torch.float32:
            fatal(f"guard (o2): drug tensor {i} dtype is not float32: {t.dtype}")
        if not torch.isfinite(t).all().item():
            fatal(f"guard (o2): drug tensor {i} contains non-finite values")

    # (o3) negative_ctrl is (2, 1024) and all-NaN
    neg = bench_drug_emb["negative_ctrl"]
    if not isinstance(neg, torch.Tensor) or neg.shape != (2, 1024):
        fatal(f"guard (o3): negative_ctrl shape is not (2, 1024): {getattr(neg, 'shape', None)}")
    if not torch.isnan(neg).all().item():
        fatal("guard (o3): negative_ctrl is not all-NaN")

    # (o4) bench_dose_emb.pt is a dict whose keys equal the dose list exactly (float ==), each tensor (L, 1024) and finite
    bench_dose_emb = torch.load(dose_emb_path, map_location="cpu")
    if not isinstance(bench_dose_emb, dict):
        fatal("guard (o4): bench_dose_emb.pt is not a dict")

    if isinstance(dose_list_json, (str, os.PathLike)):
        with open(dose_list_json, "r", encoding="utf-8") as f:
            dose_list = json.load(f)
    else:
        dose_list = dose_list_json

    expected_doses = [float(d) for d in dose_list]
    if len(bench_dose_emb) != len(expected_doses):
        fatal(
            f"guard (o4): bench_dose_emb.pt key count mismatch: {len(bench_dose_emb)} != {len(expected_doses)}"
        )
    for ed in expected_doses:
        found = False
        for k in bench_dose_emb.keys():
            if type(k) in (float, np.float64, int) and k == ed:
                found = True
                break
        if not found:
            fatal(f"guard (o4): dose {ed} not found in bench_dose_emb keys: {list(bench_dose_emb.keys())}")

    for k in bench_dose_emb.keys():
        if not any(k == ed for ed in expected_doses):
            fatal(f"guard (o4): unexpected dose key {k} in bench_dose_emb")

    for k, t in bench_dose_emb.items():
        if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[0] < 1 or t.shape[1] != 1024:
            fatal(f"guard (o4): dose tensor for {k} shape is not (L, 1024): {getattr(t, 'shape', None)}")
        if not torch.isfinite(t).all().item():
            fatal(f"guard (o4): dose tensor for {k} contains non-finite values")

    sha1s = {
        "bench_drug_emb": sha1_file(drug_emb_path),
        "bench_dose_emb": sha1_file(dose_emb_path),
    }
    caption_cache = os.path.join(out_dir, "_caption_cache.pkl")
    if os.path.isfile(caption_cache):
        sha1s["_caption_cache"] = sha1_file(caption_cache)
    return sha1s


def main():
    parser = argparse.ArgumentParser(description="P10 PertDiT embedding precompute tools")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_inputs = subparsers.add_parser("inputs", help="Build input dataset files")
    p_inputs.add_argument("--bundle", required=True, help="Path to bundle npz")
    p_inputs.add_argument("--idx2smi", required=True, help="Path to full idx2smi table")
    p_inputs.add_argument("--out", required=True, help="Output directory")

    p_script = subparsers.add_parser("script", help="Generate prepare_pertdit_embeddings_p10.py")
    p_script.add_argument("--src", required=True, help="Path to Bai et al. script")
    p_script.add_argument("--dst", required=True, help="Destination script path")

    args = parser.parse_args()
    if args.subcommand == "inputs":
        build_inputs(args.bundle, args.idx2smi, args.out)
    elif args.subcommand == "script":
        make_emb_script(args.src, args.dst)


if __name__ == "__main__":
    main()
