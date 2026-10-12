# -*- coding: utf-8 -*-
"""RESULTS 98.15, 98.16: P10 arm D PertDiT embeddings merge and dose embeddings.
Kaggle CPU kernel, internet on (SMILES only, no expression data).
"""
import glob
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import numpy as np

IS_CAPTION_KERNEL = False

PREPARE_SCRIPT_SHA1 = 'dd41c66a31b92ee17e4f5ba3907cce5fdb9943af'
P10_EMB_SHA1 = 'db1d85462ef6b75bf2a561ba1c044d630bca6d68'
IDX2SMI_ALL_SHA1 = '86532513d3e52031bd79a2f8f08a06d3456e1be8'
PLACEHOLDER_SHA1 = 'f412d992ff92e03e18e53f43231497e8c71fdfa6'
DOSE_LIST_SHA1 = '7941bcd3155ae73fbdcca2f11ce6af647300d2ce'
COMPOUND_MAP_SHA1 = '918e08f61352e8f122c3451d71a800e616de4f7c'
MANIFEST_SHA1 = '372a9d97478e0f31ad95d68b8b7f98e80c917980'

PIN_MOLT5_REV = 'e307e7a1f959d4c9ba6553ce656440cd5d3b4660'
PIN_BERT_REV = '1eb6d81c5fc1c42d3a43c71956b0e526558ae053'
PIN_BERT_WEIGHTS_SHA256 = 'fed75e5716547b54198d4dd123e7a3f3c64a82e1172b3492a11deebd6ab4cd4d'


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


print("=== Starting P10 arm D merge kernel ===", flush=True)

# 1. Install
if os.environ.get("P10D_SKIP_PIP_INSTALL") != "1":
    cmd_pip = [
        sys.executable, "-m", "pip", "install",
        "transformers==4.39.1",
        "tokenizers==0.15.2",
        "sentencepiece==0.2.0",
        "huggingface-hub==0.24.6",
    ]
    res_pip = subprocess.run(cmd_pip)
    if res_pip.returncode != 0:
        fatal(f"pip install exited {res_pip.returncode}")

res_v = subprocess.run(
    [
        sys.executable, "-c",
        "import transformers, tokenizers, sentencepiece, huggingface_hub; "
        "print(f'transformers=={transformers.__version__} "
        "tokenizers=={tokenizers.__version__} "
        "sentencepiece=={sentencepiece.__version__} "
        "huggingface-hub=={huggingface_hub.__version__}')",
    ],
    capture_output=True,
    text=True,
)
if res_v.returncode != 0:
    fatal(f"reading versions failed: {res_v.stderr}")
installed_versions = res_v.stdout.strip()
print(f"Installed versions: {installed_versions}", flush=True)

# 2. Revision guard in subprocess
code_rev = f"""
import sys, huggingface_hub
api = huggingface_hub.HfApi()
s_molt5 = api.model_info("laituan245/molt5-large-smiles2caption").sha
s_bert = api.model_info("michiyasunaga/BioLinkBERT-large").sha
if s_molt5 != "{PIN_MOLT5_REV}":
    print(f"FATAL: MolT5 revision mismatch: {{s_molt5}} != {PIN_MOLT5_REV}", file=sys.stderr)
    sys.exit(1)
if s_bert != "{PIN_BERT_REV}":
    print(f"FATAL: BioLinkBERT revision mismatch: {{s_bert}} != {PIN_BERT_REV}", file=sys.stderr)
    sys.exit(1)
"""
res_rev = subprocess.run([sys.executable, "-c", code_rev], capture_output=True, text=True)
if res_rev.returncode != 0:
    fatal(f"HF revision check failed: {res_rev.stderr.strip() or res_rev.stdout.strip()}")
print("GUARD 2: HF revisions verified against pins", flush=True)

# 3. Mounted-file guards
KAGGLE_INPUT = os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input")
KAGGLE_TEMP = os.environ.get("KAGGLE_TEMP_DIR", "/kaggle/temp")
KAGGLE_WORKING = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")


def find_one(pattern):
    hits = glob.glob(os.path.join(KAGGLE_INPUT, "**", pattern), recursive=True)
    if len(hits) != 1:
        fatal(f"expected exactly 1 match for {pattern}, found {len(hits)}: {hits}")
    return hits[0]


script_in = find_one("prepare_pertdit_embeddings_p10.py")
p10_emb_in = find_one("p10_pertdit_emb.py")
idx2smi_all_in = find_one("idx2smi_all.npy")
placeholder_in = find_one("pert_smiles_emb_placeholder.pkl")
dose_list_in = find_one("dose_list.json")
compound_map_in = find_one("compound_map.json")
manifest_in = find_one("inputs_manifest.json")

verified_sha1s = {}
for name, p, expected in [
    ("prepare_pertdit_embeddings_p10.py", script_in, PREPARE_SCRIPT_SHA1),
    ("p10_pertdit_emb.py", p10_emb_in, P10_EMB_SHA1),
    ("idx2smi_all.npy", idx2smi_all_in, IDX2SMI_ALL_SHA1),
    ("pert_smiles_emb_placeholder.pkl", placeholder_in, PLACEHOLDER_SHA1),
    ("dose_list.json", dose_list_in, DOSE_LIST_SHA1),
    ("compound_map.json", compound_map_in, COMPOUND_MAP_SHA1),
    ("inputs_manifest.json", manifest_in, MANIFEST_SHA1),
]:
    got = sha1_file(p)
    if got != expected:
        fatal(f"mounted file {name} sha1 mismatch: {got} != {expected}")
    verified_sha1s[name] = got

sys.path.insert(0, os.path.dirname(p10_emb_in))
import p10_pertdit_emb

# 4. Layout
root_dir = os.path.join(KAGGLE_TEMP, "p10d")
script_dst = os.path.join(root_dir, "experiments", "main_benchmark", "data_prep", "prepare_pertdit_embeddings_p10.py")
idx2smi_dst = os.path.join(root_dir, "data", "XPert", "processed_data", "all_drugs_idx2smi_8981.npy")
placeholder_dst = os.path.join(root_dir, "data", "PertDiT", "extracted", "pert_smiles_emb.pkl")
dose_list_dst = os.path.join(root_dir, "data", "PertDiT", "p10_dose_list.json")

for target, src in [
    (script_dst, script_in),
    (idx2smi_dst, idx2smi_all_in),
    (placeholder_dst, placeholder_in),
    (dose_list_dst, dose_list_in),
]:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(src, target)

# 5. Merge caches
cache_paths = []
slice_cache_sha1s = {}
for i in range(5):
    c_path = find_one(f"_caption_cache_s{i}.pkl")
    r_path = find_one(f"cap_record_s{i}.json")
    with open(r_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    got_c_sha1 = sha1_file(c_path)
    if got_c_sha1 != rec["cache_sha1"]:
        fatal(f"caption cache s{i} sha1 mismatch: {got_c_sha1} != {rec['cache_sha1']}")
    cache_paths.append(c_path)
    slice_cache_sha1s[f"s{i}"] = got_c_sha1

merged_captions = p10_pertdit_emb.merge_caches(cache_paths, idx2smi_all_in)
merged_cache_path = os.path.join(root_dir, "data", "PertDiT", "bench_drug_emb", "_caption_cache.pkl")
os.makedirs(os.path.dirname(merged_cache_path), exist_ok=True)
with open(merged_cache_path, "wb") as f:
    pickle.dump(merged_captions, f)

# 6. Run script as subprocess
hf_home = os.path.join(KAGGLE_TEMP, "hf")
os.makedirs(hf_home, exist_ok=True)
log_path = os.path.join(KAGGLE_TEMP, "emb.log")

env = os.environ.copy()
env["HF_HOME"] = hf_home
env.pop("OMP_NUM_THREADS", None)
env.pop("MKL_NUM_THREADS", None)

import torch
n_threads = torch.get_num_threads()

cmd = [sys.executable, script_dst]
t0 = time.time()
with open(log_path, "w", encoding="utf-8") as f_log:
    try:
        proc = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, env=env, timeout=41400)
    except subprocess.TimeoutExpired:
        fatal("precompute script timed out after 41400 s")
wall_s = int(time.time() - t0)

with open(log_path, "r", encoding="utf-8", errors="replace") as f_log:
    log_text = f_log.read()

if proc.returncode != 0:
    lines = log_text.splitlines()
    last_lines = "\n".join(lines[-60:])
    fatal(f"precompute script exited with code {proc.returncode}. Last 60 lines:\n{last_lines}")

idx2smi_data = np.load(idx2smi_all_in, allow_pickle=True).item()
n_compounds = len(idx2smi_data)
expected_skip = f"All {n_compounds} captions already cached; skipping MolT5"
if expected_skip not in log_text:
    fatal(f"merge kernel log missing required line: {expected_skip!r}")

# 7. Weight guard (BioLinkBERT only)
patt = os.path.join(hf_home, "**", "snapshots", PIN_BERT_REV, "pytorch_model.bin")
hits = glob.glob(patt, recursive=True)
if len(hits) != 1:
    fatal(f"expected 1 weights file for BioLinkBERT at {PIN_BERT_REV}, found {len(hits)}: {hits}")
bert_bin = hits[0]
got_b_sha256 = sha256_file(bert_bin)
if got_b_sha256 != PIN_BERT_WEIGHTS_SHA256:
    fatal(f"BioLinkBERT weights sha256 mismatch: {got_b_sha256} != {PIN_BERT_WEIGHTS_SHA256}")
verified_sha256s = {"bert_weights": got_b_sha256}

# 8. Check outputs and persist to /kaggle/working
out_dir = os.path.join(root_dir, "data", "PertDiT", "bench_drug_emb")
check_sha1s = p10_pertdit_emb.check_outputs(out_dir, idx2smi_all_in, dose_list_in)

for fname in ["bench_drug_emb.pkl", "bench_dose_emb.pt", "_caption_cache.pkl"]:
    src_f = os.path.join(out_dir, fname)
    dst_f = os.path.join(KAGGLE_WORKING, fname)
    shutil.copyfile(src_f, dst_f + ".part")
    os.replace(dst_f + ".part", dst_f)

with open(dose_list_in, "r", encoding="utf-8") as f:
    dose_list_data = json.load(f)

emb_record = {
    "sha1s": check_sha1s,
    "compound_map_sha1": verified_sha1s["compound_map.json"],
    "dose_list": dose_list_data,
    "versions": installed_versions,
    "verified_sha256s": verified_sha256s,
    "caption_cache_sha1": sha1_file(os.path.join(out_dir, "_caption_cache.pkl")),
    "slice_cache_sha1s": slice_cache_sha1s,
    "wall_s": wall_s,
}
rec_path = os.path.join(KAGGLE_WORKING, "emb_record.json")
temp_rec = os.path.join(KAGGLE_TEMP, "emb_record.json")
with open(temp_rec, "w", encoding="utf-8") as f:
    json.dump(emb_record, f, indent=2)
shutil.copyfile(temp_rec, rec_path + ".part")
os.replace(rec_path + ".part", rec_path)

print("=== Completed P10 arm D merge kernel ===", flush=True)
