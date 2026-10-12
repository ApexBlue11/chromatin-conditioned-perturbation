# -*- coding: utf-8 -*-
"""RESULTS 98.15, 98.16: P10 arm D MolT5 caption generation slice 4.
Kaggle CPU kernel, internet on (SMILES only, no expression data).
"""
import glob
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import numpy as np

K = 4
IS_CAPTION_KERNEL = True

PREPARE_SCRIPT_SHA1 = 'dd41c66a31b92ee17e4f5ba3907cce5fdb9943af'
P10_EMB_SHA1 = 'db1d85462ef6b75bf2a561ba1c044d630bca6d68'
SLICE_NPY_SHA1 = 'f4a842fb543d18b2969e4e0639d7f089de84cfd5'
SLICE_NPY_NAME = 'idx2smi_slice4.npy'
PLACEHOLDER_SHA1 = 'f412d992ff92e03e18e53f43231497e8c71fdfa6'
DOSE_LIST_SHA1 = '7941bcd3155ae73fbdcca2f11ce6af647300d2ce'
COMPOUND_MAP_SHA1 = '918e08f61352e8f122c3451d71a800e616de4f7c'
MANIFEST_SHA1 = '372a9d97478e0f31ad95d68b8b7f98e80c917980'

PIN_MOLT5_REV = 'e307e7a1f959d4c9ba6553ce656440cd5d3b4660'
PIN_BERT_REV = '1eb6d81c5fc1c42d3a43c71956b0e526558ae053'
PIN_MOLT5_WEIGHTS_SHA256 = 'eb5e5759049ba398aa8ee7560edd14d024108a0e3fd3e5ea26dc5f2097044a89'
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


def cpu_model():
    """Review 080 C2: beam ties depend on the CPU, so the record names it."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or "unknown"


print(f"=== Starting P10 arm D caption kernel slice {K} ===", flush=True)

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
PINNED_VERSIONS = "transformers==4.39.1 tokenizers==0.15.2 sentencepiece==0.2.0 huggingface-hub==0.24.6"
if installed_versions != PINNED_VERSIONS:
    fatal(f"installed versions {installed_versions!r} != pinned {PINNED_VERSIONS!r} (review 080 C3)")
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
slice_in = find_one(SLICE_NPY_NAME)
placeholder_in = find_one("pert_smiles_emb_placeholder.pkl")
dose_list_in = find_one("dose_list.json")
compound_map_in = find_one("compound_map.json")
manifest_in = find_one("inputs_manifest.json")

verified_sha1s = {}
for name, p, expected in [
    ("prepare_pertdit_embeddings_p10.py", script_in, PREPARE_SCRIPT_SHA1),
    ("p10_pertdit_emb.py", p10_emb_in, P10_EMB_SHA1),
    (SLICE_NPY_NAME, slice_in, SLICE_NPY_SHA1),
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
slice_dst = os.path.join(root_dir, "data", "XPert", "processed_data", "all_drugs_idx2smi_8981.npy")
placeholder_dst = os.path.join(root_dir, "data", "PertDiT", "extracted", "pert_smiles_emb.pkl")
dose_list_dst = os.path.join(root_dir, "data", "PertDiT", "p10_dose_list.json")

for target, src in [
    (script_dst, script_in),
    (slice_dst, slice_in),
    (placeholder_dst, placeholder_in),
    (dose_list_dst, dose_list_in),
]:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(src, target)

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

# 7. Weight guard
def find_weights(repo_id, pin_sha):
    patt = os.path.join(hf_home, "**", "snapshots", pin_sha, "pytorch_model.bin")
    hits = glob.glob(patt, recursive=True)
    if len(hits) != 1:
        fatal(f"expected 1 weights file for {repo_id} at {pin_sha}, found {len(hits)}: {hits}")
    return hits[0]

verified_sha256s = {}
molt5_bin = find_weights("laituan245/molt5-large-smiles2caption", PIN_MOLT5_REV)
got_m_sha256 = sha256_file(molt5_bin)
if got_m_sha256 != PIN_MOLT5_WEIGHTS_SHA256:
    fatal(f"MolT5 weights sha256 mismatch: {got_m_sha256} != {PIN_MOLT5_WEIGHTS_SHA256}")
verified_sha256s["molt5_weights"] = got_m_sha256

bert_bin = find_weights("michiyasunaga/BioLinkBERT-large", PIN_BERT_REV)
got_b_sha256 = sha256_file(bert_bin)
if got_b_sha256 != PIN_BERT_WEIGHTS_SHA256:
    fatal(f"BioLinkBERT weights sha256 mismatch: {got_b_sha256} != {PIN_BERT_WEIGHTS_SHA256}")
verified_sha256s["bert_weights"] = got_b_sha256

# 8. Persist to /kaggle/working
out_cache_src = os.path.join(root_dir, "data", "PertDiT", "bench_drug_emb", "_caption_cache.pkl")
slice_dict = np.load(slice_in, allow_pickle=True).item()
expected_smiles = set(slice_dict.values())
with open(out_cache_src, "rb") as f:
    cache_data = pickle.load(f)
if set(cache_data.keys()) != expected_smiles:
    fatal(f"caption cache keys mismatch with slice SMILES: {len(cache_data)} != {len(expected_smiles)}")

working_cache = os.path.join(KAGGLE_WORKING, f"_caption_cache_s{K}.pkl")
temp_cache = os.path.join(KAGGLE_TEMP, f"_caption_cache_s{K}.pkl")
shutil.copyfile(out_cache_src, temp_cache)
shutil.copyfile(temp_cache, working_cache + ".part")
os.replace(working_cache + ".part", working_cache)

match = re.search(r"Caption generation done in .*?\(([\d\.]+)s/drug\)", log_text)
s_per_smiles = float(match.group(1)) if match else 0.0

cap_record = {
    "slice": K,
    "n_smiles": len(expected_smiles),
    "seconds_per_smiles": s_per_smiles,
    "torch_num_threads": n_threads,
    "torch_version": torch.__version__,
    "cpu_model": cpu_model(),
    "versions": installed_versions,
    "verified_sha1s": verified_sha1s,
    "verified_sha256s": verified_sha256s,
    "cache_sha1": sha1_file(temp_cache),
    "wall_s": wall_s,
}
working_rec = os.path.join(KAGGLE_WORKING, f"cap_record_s{K}.json")
temp_rec = os.path.join(KAGGLE_TEMP, f"cap_record_s{K}.json")
with open(temp_rec, "w", encoding="utf-8") as f:
    json.dump(cap_record, f, indent=2)
shutil.copyfile(temp_rec, working_rec + ".part")
os.replace(working_rec + ".part", working_rec)

print(f"=== Completed P10 arm D caption kernel slice {K} ===", flush=True)
