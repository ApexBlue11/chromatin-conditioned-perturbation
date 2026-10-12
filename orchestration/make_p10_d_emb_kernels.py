# -*- coding: utf-8 -*-
"""RESULTS §98.15, §98.16: Build the five caption kernels and one merge kernel for P10 arm D.

Kernels:
- kern_p10d_cap_s{k} (k < 5): MolT5 captions for slice k on free Kaggle CPU.
- kern_p10d_emb_merge: merges the 5 caption caches, runs BioLinkBERT on all captions and dose prompts.
"""
import argparse
import ast
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "model", "v9"))
import p10_pertdit_emb  # noqa: E402


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def fatal(msg):
    # PI (W39 review): the builder called fatal() without defining it at module level (a NameError on any builder failure)
    raise SystemExit("FATAL: " + msg)


PIN_MOLT5_REV = "e307e7a1f959d4c9ba6553ce656440cd5d3b4660"
PIN_BERT_REV = "1eb6d81c5fc1c42d3a43c71956b0e526558ae053"
PIN_MOLT5_WEIGHTS_SHA256 = "eb5e5759049ba398aa8ee7560edd14d024108a0e3fd3e5ea26dc5f2097044a89"
PIN_BERT_WEIGHTS_SHA256 = "fed75e5716547b54198d4dd123e7a3f3c64a82e1172b3492a11deebd6ab4cd4d"

DEFAULT_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)

CAPTION_KERNEL_TEMPLATE = '''# -*- coding: utf-8 -*-
"""RESULTS 98.15, 98.16: P10 arm D MolT5 caption generation slice __K__.
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

K = __K__
IS_CAPTION_KERNEL = True

PREPARE_SCRIPT_SHA1 = '__PREPARE_SCRIPT_SHA1__'
P10_EMB_SHA1 = '__P10_EMB_SHA1__'
SLICE_NPY_SHA1 = '__SLICE_NPY_SHA1__'
SLICE_NPY_NAME = '__SLICE_NPY_NAME__'
PLACEHOLDER_SHA1 = '__PLACEHOLDER_SHA1__'
DOSE_LIST_SHA1 = '__DOSE_LIST_SHA1__'
COMPOUND_MAP_SHA1 = '__COMPOUND_MAP_SHA1__'
MANIFEST_SHA1 = '__MANIFEST_SHA1__'

PIN_MOLT5_REV = '__PIN_MOLT5_REV__'
PIN_BERT_REV = '__PIN_BERT_REV__'
PIN_MOLT5_WEIGHTS_SHA256 = '__PIN_MOLT5_WEIGHTS_SHA256__'
PIN_BERT_WEIGHTS_SHA256 = '__PIN_BERT_WEIGHTS_SHA256__'


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
    last_lines = "\\n".join(lines[-60:])
    fatal(f"precompute script exited with code {proc.returncode}. Last 60 lines:\\n{last_lines}")

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

match = re.search(r"Caption generation done in .*?\\(([\\d\\.]+)s/drug\\)", log_text)
s_per_smiles = float(match.group(1)) if match else 0.0

cap_record = {
    "slice": K,
    "n_smiles": len(expected_smiles),
    "seconds_per_smiles": s_per_smiles,
    "torch_num_threads": n_threads,
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
'''

MERGE_KERNEL_TEMPLATE = '''# -*- coding: utf-8 -*-
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

PREPARE_SCRIPT_SHA1 = '__PREPARE_SCRIPT_SHA1__'
P10_EMB_SHA1 = '__P10_EMB_SHA1__'
IDX2SMI_ALL_SHA1 = '__IDX2SMI_ALL_SHA1__'
PLACEHOLDER_SHA1 = '__PLACEHOLDER_SHA1__'
DOSE_LIST_SHA1 = '__DOSE_LIST_SHA1__'
COMPOUND_MAP_SHA1 = '__COMPOUND_MAP_SHA1__'
MANIFEST_SHA1 = '__MANIFEST_SHA1__'

PIN_MOLT5_REV = '__PIN_MOLT5_REV__'
PIN_BERT_REV = '__PIN_BERT_REV__'
PIN_BERT_WEIGHTS_SHA256 = '__PIN_BERT_WEIGHTS_SHA256__'


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
    last_lines = "\\n".join(lines[-60:])
    fatal(f"precompute script exited with code {proc.returncode}. Last 60 lines:\\n{last_lines}")

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
'''


def build_kernels(out_root, inputs_dir=None):
    """Build the 5 caption kernel directories and 1 merge kernel directory under out_root."""
    if inputs_dir is None:
        inputs_dir = os.path.join(REPO_ROOT, "model", "results", "p10", "p10d_inputs")

    if not os.path.isdir(inputs_dir):
        fatal(f"inputs_dir does not exist: {inputs_dir}")

    # Read mounted file hashes
    # PI (W39 review): no fallbacks -- every pinned file must be the one in the folder that becomes the dataset
    script_path = os.path.join(inputs_dir, "prepare_pertdit_embeddings_p10.py")
    p10_emb_path = os.path.join(inputs_dir, "p10_pertdit_emb.py")
    for need in (script_path, p10_emb_path):
        if not os.path.isfile(need):
            fatal(f"missing {need} (the inputs folder must hold every file the kernels pin)")

    prepare_script_sha1 = sha1_file(script_path)
    p10_emb_sha1 = sha1_file(p10_emb_path)
    placeholder_sha1 = sha1_file(os.path.join(inputs_dir, "pert_smiles_emb_placeholder.pkl"))
    dose_list_sha1 = sha1_file(os.path.join(inputs_dir, "dose_list.json"))
    compound_map_sha1 = sha1_file(os.path.join(inputs_dir, "compound_map.json"))
    manifest_sha1 = sha1_file(os.path.join(inputs_dir, "inputs_manifest.json"))
    idx2smi_all_sha1 = sha1_file(os.path.join(inputs_dir, "idx2smi_all.npy"))

    slice_sha1s = [sha1_file(os.path.join(inputs_dir, f"idx2smi_slice{k}.npy")) for k in range(5)]

    # Read docker image from kern_p10_m_s0 if available
    docker_image = DEFAULT_DOCKER_IMAGE
    p10m_meta_path = os.path.join(
        REPO_ROOT,
        "external",
        "kaggle_kernels",
        "kern_p10_m_s0",
        "kernel-metadata.json",
    )
    if os.path.exists(p10m_meta_path):
        with open(p10m_meta_path, "r", encoding="utf-8") as f:
            docker_image = json.load(f)["docker_image"]

    os.makedirs(out_root, exist_ok=True)

    # 1. Build caption kernels
    for k in range(5):
        folder_name = f"kern_p10d_cap_s{k}"
        k_dir = os.path.join(out_root, folder_name)
        os.makedirs(k_dir, exist_ok=True)

        code_filename = f"lincs-p10d-cap-s{k}.py"
        script_code = (
            CAPTION_KERNEL_TEMPLATE.replace("__K__", str(k))
            .replace("__PREPARE_SCRIPT_SHA1__", prepare_script_sha1)
            .replace("__P10_EMB_SHA1__", p10_emb_sha1)
            .replace("__SLICE_NPY_SHA1__", slice_sha1s[k])
            .replace("__SLICE_NPY_NAME__", f"idx2smi_slice{k}.npy")
            .replace("__PLACEHOLDER_SHA1__", placeholder_sha1)
            .replace("__DOSE_LIST_SHA1__", dose_list_sha1)
            .replace("__COMPOUND_MAP_SHA1__", compound_map_sha1)
            .replace("__MANIFEST_SHA1__", manifest_sha1)
            .replace("__PIN_MOLT5_REV__", PIN_MOLT5_REV)
            .replace("__PIN_BERT_REV__", PIN_BERT_REV)
            .replace("__PIN_MOLT5_WEIGHTS_SHA256__", PIN_MOLT5_WEIGHTS_SHA256)
            .replace("__PIN_BERT_WEIGHTS_SHA256__", PIN_BERT_WEIGHTS_SHA256)
        )
        ast.parse(script_code)

        script_path_out = os.path.join(k_dir, code_filename)
        with open(script_path_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(script_code)

        metadata = {
            "id": f"apexblue/lincs-p10d-cap-s{k}",
            "title": f"lincs-p10d-cap-s{k}",
            "code_file": code_filename,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": ["apexblue/p10d-emb-inputs"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "docker_image": docker_image,
        }
        meta_path = os.path.join(k_dir, "kernel-metadata.json")
        with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(metadata, f, indent=1)
            f.write("\n")

    # 2. Build merge kernel
    merge_folder = "kern_p10d_emb_merge"
    m_dir = os.path.join(out_root, merge_folder)
    os.makedirs(m_dir, exist_ok=True)

    merge_code_filename = "lincs-p10d-emb-merge.py"
    merge_script_code = (
        MERGE_KERNEL_TEMPLATE.replace("__PREPARE_SCRIPT_SHA1__", prepare_script_sha1)
        .replace("__P10_EMB_SHA1__", p10_emb_sha1)
        .replace("__IDX2SMI_ALL_SHA1__", idx2smi_all_sha1)
        .replace("__PLACEHOLDER_SHA1__", placeholder_sha1)
        .replace("__DOSE_LIST_SHA1__", dose_list_sha1)
        .replace("__COMPOUND_MAP_SHA1__", compound_map_sha1)
        .replace("__MANIFEST_SHA1__", manifest_sha1)
        .replace("__PIN_MOLT5_REV__", PIN_MOLT5_REV)
        .replace("__PIN_BERT_REV__", PIN_BERT_REV)
        .replace("__PIN_BERT_WEIGHTS_SHA256__", PIN_BERT_WEIGHTS_SHA256)
    )
    ast.parse(merge_script_code)

    merge_script_path = os.path.join(m_dir, merge_code_filename)
    with open(merge_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(merge_script_code)

    merge_metadata = {
        "id": "apexblue/lincs-p10d-emb-merge",
        "title": "lincs-p10d-emb-merge",
        "code_file": merge_code_filename,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": ["apexblue/p10d-emb-inputs"],
        "kernel_sources": [f"apexblue/lincs-p10d-cap-s{k}" for k in range(5)],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": docker_image,
    }
    merge_meta_path = os.path.join(m_dir, "kernel-metadata.json")
    with open(merge_meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(merge_metadata, f, indent=1)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate P10 PertDiT embedding Kaggle kernels")
    parser.add_argument(
        "--out_root",
        default=os.path.join(REPO_ROOT, "external", "kaggle_kernels"),
        help="Target directory for kernel folders",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Alias for --out_root",
    )
    parser.add_argument(
        "--inputs_dir",
        default=None,
        help="Directory holding inputs dataset files",
    )
    args = parser.parse_args()
    out_root = args.out_dir if args.out_dir is not None else args.out_root
    build_kernels(out_root, args.inputs_dir)


if __name__ == "__main__":
    main()
