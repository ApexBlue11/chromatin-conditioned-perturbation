# -*- coding: utf-8 -*-
"""RESULTS §98.15, §98.16: Build Kaggle T4x2 kernels for P10 arm D (PertDiT) timing and run.

Usage:
  python orchestration/make_p10_d_kernels.py --emb_record R [--t_test_s T] [--out_root DIR]
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "model", "v9"))
import p10_mlp  # noqa: E402
import p10_pertdit  # noqa: E402


def fatal(msg):
    raise SystemExit("FATAL: " + msg)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def make_deterministic_zip(src_dir, zip_path):
    """Create a byte-deterministic zip archive with fixed timestamps and permissions."""
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_dir):
            dirs.sort()
            for f in sorted(files):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, src_dir).replace("\\", "/")
                zinfo = zipfile.ZipInfo(rel_path, date_time=(2026, 1, 1, 0, 0, 0))
                zinfo.external_attr = 0o644 << 16
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                with open(full_path, "rb") as fp:
                    zf.writestr(zinfo, fp.read())


KERNEL_TEMPLATE = '''import time
KERNEL_T0 = time.time()
# -*- coding: utf-8 -*-
"""RESULTS 98.15, 98.16: P10 arm D PertDiT (__MODE_UPPER__).
Kaggle T4x2 GPU kernel, internet off, blinded.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import numpy as np
import torch

MODE = '__MODE__'
PINNED_GPU_VERSIONS = "scanpy=1.12.4 anndata=0.13.3.post0 diffusers=0.30.2 einops=0.8.0 huggingface_hub=0.24.6 rdkit=2023.09.1"
TIMING_SHA1 = '__TIMING_SHA1__'
TIMING_DECISION = '__TIMING_DECISION__'
T_TEST_S = __T_TEST_S__
CAP_S = __CAP_S__

BUNDLE_SHA1 = '__BUNDLE_SHA1__'
CARVES_SHA1 = '__CARVES_SHA1__'
P10_MLP_SHA1 = '__P10_MLP_SHA1__'
P10_PERTDIT_SHA1 = '__P10_PERTDIT_SHA1__'
TRAINER_SHA1 = '__TRAINER_SHA1__'
PERTDIT_SRC_SHA1 = '__PERTDIT_SRC_SHA1__'
COMPOUND_MAP_SHA1 = '__COMPOUND_MAP_SHA1__'
DOSE_LIST_SHA1 = '__DOSE_LIST_SHA1__'
DRUG_EMB_SHA1 = '__DRUG_EMB_SHA1__'
DOSE_EMB_SHA1 = '__DOSE_EMB_SHA1__'
CAP_TRAINER_SHA1 = '__CAP_TRAINER_SHA1__'


def fatal(msg):
    raise SystemExit("FATAL: " + msg)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_one(pattern):
    hits = glob.glob(pattern, recursive=True)
    if len(hits) != 1:
        fatal(f"expected exactly 1 match for {pattern}, found {len(hits)}: {hits}")
    return hits[0]


print(f"=== Starting P10 arm D {MODE} ===", flush=True)

# 1. GPU guard: two non-P100 GPUs (make_p9 lines)
if os.environ.get("P10D_BYPASS_GPU_GUARD", "0") != "1":
    if not torch.cuda.is_available():
        fatal("no CUDA device. NOT falling back to CPU.")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if any("P100" in n for n in names) or len(names) < 2:
        fatal(f"need 2 non-P100 GPUs, got {names!r}")
    print(f"GPUs: {names}", flush=True)

# 2. Install wheels (no network)
KAGGLE_INPUT = os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input")
KAGGLE_TEMP = os.environ.get("KAGGLE_TEMP_DIR", "/kaggle/temp")
KAGGLE_WORKING = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
os.makedirs(KAGGLE_TEMP, exist_ok=True)
os.makedirs(KAGGLE_WORKING, exist_ok=True)

if os.environ.get("P10D_BYPASS_PIP", "0") != "1":
    whl_scanpy = find_one(os.path.join(KAGGLE_INPUT, "**/scanpy-1.12.4-py3-none-any.whl"))
    p10_wheels_dir = os.path.dirname(whl_scanpy)
    whl_diffusers = find_one(os.path.join(KAGGLE_INPUT, "**/diffusers-0.30.2-py3-none-any.whl"))
    p10d_wheels_dir = os.path.dirname(whl_diffusers)

    print(f"GUARD 1: installing scanpy and anndata wheels from {p10_wheels_dir}", flush=True)
    res1 = subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", p10_wheels_dir,
        "scanpy==1.12.4", "anndata==0.13.3.post0",
    ])
    if res1.returncode != 0:
        fatal(f"pip install scanpy/anndata exited {res1.returncode}")

    print(f"GUARD 1: installing PertDiT wheels from {p10d_wheels_dir}", flush=True)
    res2 = subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--find-links", p10d_wheels_dir,
        "diffusers==0.30.2", "einops==0.8.0", "huggingface-hub==0.24.6", "rdkit==2023.9.1",
    ])
    if res2.returncode != 0:
        fatal(f"pip install diffusers/rdkit exited {res2.returncode}")

    chk_cmd = [
        sys.executable, "-c",
        "import scanpy, anndata, diffusers, einops, huggingface_hub, rdkit; "
        "print(f'scanpy={scanpy.__version__} anndata={anndata.__version__} diffusers={diffusers.__version__} "
        "einops={einops.__version__} huggingface_hub={huggingface_hub.__version__} rdkit={rdkit.__version__}')"
    ]
    res_chk = subprocess.run(chk_cmd, capture_output=True, text=True)
    if res_chk.returncode != 0:
        fatal(f"version check failed: {res_chk.stderr}")
    installed_versions = res_chk.stdout.strip()
    print(f"GUARD 1: installed package versions: {installed_versions}", flush=True)
    if installed_versions != PINNED_GPU_VERSIONS:
        fatal(f"GUARD 1: installed versions {installed_versions!r} != pinned {PINNED_GPU_VERSIONS!r} (review 081 C4)")
else:
    installed_versions = "pip bypassed (tests only)"

# 3. Mounted-file guards
bundle_path = find_one(os.path.join(KAGGLE_INPUT, "**/xpert_mdmt_splits.npz"))
carves_path = find_one(os.path.join(KAGGLE_INPUT, "**/p10_carves.json"))
p10_mlp_path = find_one(os.path.join(KAGGLE_INPUT, "**/p10_mlp.py"))
p10_pertdit_path = find_one(os.path.join(KAGGLE_INPUT, "**/p10_pertdit.py"))
train_pertdit_path = find_one(os.path.join(KAGGLE_INPUT, "**/train_pertdit.py"))
pertdit_src_zip = find_one(os.path.join(KAGGLE_INPUT, "**/pertdit_src.zip"))
compound_map_path = find_one(os.path.join(KAGGLE_INPUT, "**/compound_map.json"))
dose_list_path = find_one(os.path.join(KAGGLE_INPUT, "**/dose_list.json"))
drug_emb_path = find_one(os.path.join(KAGGLE_INPUT, "**/bench_drug_emb.pkl"))
dose_emb_path = find_one(os.path.join(KAGGLE_INPUT, "**/bench_dose_emb.pt"))

verified_sha1s = {}
for name, p, expected in [
    ("bundle", bundle_path, BUNDLE_SHA1),
    ("carves", carves_path, CARVES_SHA1),
    ("p10_mlp", p10_mlp_path, P10_MLP_SHA1),
    ("p10_pertdit", p10_pertdit_path, P10_PERTDIT_SHA1),
    ("train_pertdit", train_pertdit_path, TRAINER_SHA1),
    ("pertdit_src_zip", pertdit_src_zip, PERTDIT_SRC_SHA1),
    ("compound_map", compound_map_path, COMPOUND_MAP_SHA1),
    ("dose_list", dose_list_path, DOSE_LIST_SHA1),
    ("bench_drug_emb", drug_emb_path, DRUG_EMB_SHA1),
    ("bench_dose_emb", dose_emb_path, DOSE_EMB_SHA1),
]:
    got = sha1_file(p)
    print(f"GUARD 2: {name} sha1 {got}", flush=True)
    if got != expected:
        fatal(f"GUARD 2: {name} sha1 mismatch: {got} != {expected}")
    verified_sha1s[name] = got

sys.path.insert(0, os.path.dirname(p10_pertdit_path))
import p10_pertdit

cap_trainer_tmp = os.path.join(KAGGLE_TEMP, "p10_train_pertdit_cap.py")
p10_pertdit.make_cap_trainer(train_pertdit_path, cap_trainer_tmp)
got_cap = sha1_file(cap_trainer_tmp)
print(f"GUARD 2: cap_trainer sha1 {got_cap}", flush=True)
if got_cap != CAP_TRAINER_SHA1:
    fatal(f"GUARD 2: cap_trainer sha1 mismatch: {got_cap} != {CAP_TRAINER_SHA1}")
verified_sha1s["cap_trainer"] = got_cap

# 4. Layout under ROOT = /kaggle/temp/p10d
ROOT = os.path.join(KAGGLE_TEMP, "p10d")
trainer_target_dir = os.path.join(ROOT, "experiments", "_shared", "training")
os.makedirs(trainer_target_dir, exist_ok=True)
cap_trainer_path = os.path.join(trainer_target_dir, "p10_train_pertdit_cap.py")
shutil.copyfile(cap_trainer_tmp, cap_trainer_path)

import zipfile
target_src_dir = os.path.join(ROOT, "models", "PertDiT", "repo", "src")
os.makedirs(target_src_dir, exist_ok=True)
with zipfile.ZipFile(pertdit_src_zip, "r") as zf:
    zf.extractall(target_src_dir)

with open(cap_trainer_path, "r", encoding="utf-8") as f:
    cap_text = f.read()
p10_pertdit.check_trainer_cfg(cap_text)
print("GUARD 2: trainer cfg checks passed", flush=True)

# 5. Build h5ad and check embeddings before any GPU work
h5ad_path = os.path.join(KAGGLE_TEMP, f"p10_d_{MODE}.h5ad")
print(f"Building h5ad to {h5ad_path} (timing={MODE == 'timing'})", flush=True)
p10_pertdit.build_h5ad_d(
    bundle_path, carves_path, compound_map_path, dose_list_path,
    h5ad_path, timing=(MODE == "timing")
)
print("GUARD 2: checking embeddings", flush=True)
p10_pertdit.check_embeddings(drug_emb_path, dose_emb_path, h5ad_path)
print("GUARD 2: embeddings verified", flush=True)

# 6. Two trainer subprocesses, started concurrently
seeds = [117, 118]
devs = ["cuda:0", "cuda:1"]
procs = []
allowed_prefixes = (
    "[Epoch", "Early stopping", "Best epoch", "P10 cap", "Wall time", "train=", "Batch caps", "Parameters"
)

extra_flags = os.environ.get("P10D_EXTRA_TRAIN_ARGS", "").strip().split() if os.environ.get("P10D_EXTRA_TRAIN_ARGS", "").strip() else []

for s, dev in zip(seeds, devs):
    out_dir_s = os.path.join(KAGGLE_TEMP, f"run_s{s}")
    os.makedirs(out_dir_s, exist_ok=True)
    cmd = [
        sys.executable, cap_trainer_path,
        "--fold", "0",
        "--split_col", "p10_split",
        "--h5ad_path", h5ad_path,
        "--drug_emb_path", drug_emb_path,
        "--dose_emb_path", dose_emb_path,
        "--dose_col", "dose_r4",
        "--dev", dev,
        "--seed", str(s),
        "--output_dir", out_dir_s,
    ]
    cmd += __MODE_SPECIFIC_ARGS__

    if extra_flags:
        cmd += extra_flags

    log_path_s = os.path.join(KAGGLE_TEMP, f"train_s{s}.log")
    print(f"Starting trainer for seed {s} on {dev}", flush=True)
    # Review 081 C1: unbuffered, or the stamps are flush times (a 3-epoch run's stdout arrives all at once at exit)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
    procs.append((proc, s, log_path_s))


def log_reader(proc, seed, log_path):
    with open(log_path, "w", encoding="utf-8") as f_log:
        for line in iter(proc.stdout.readline, ""):
            stamped = f"{time.time():.3f}\\t{line}"
            f_log.write(stamped)
            f_log.flush()
            if any(pfx in line for pfx in allowed_prefixes):
                print(f"[s{seed}] {line}", end="", flush=True)
    proc.stdout.close()


threads = []
for proc, s, log_path_s in procs:
    t = threading.Thread(target=log_reader, args=(proc, s, log_path_s))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

for proc, s, log_path_s in procs:
    ret = proc.wait()
    if ret != 0:
        with open(log_path_s, "r", encoding="utf-8", errors="replace") as f_log:
            lines = f_log.readlines()
        last30 = "".join(lines[-30:])
        fatal(f"seed {s} trainer exited with code {ret}.\\nLast 30 lines:\\n{last30}")

# 7. Timing kernel: parse_timing and timing_decision
if MODE == "timing":
    timings = {}
    for s in seeds:
        log_path_s = os.path.join(KAGGLE_TEMP, f"train_s{s}.log")
        timings[str(s)] = p10_pertdit.parse_timing(log_path_s, KERNEL_T0)
    decision = p10_pertdit.timing_decision(timings, cap_s=CAP_S)
    timing_record = {
        "timings": timings,
        "decision": decision,
        "verified_sha1s": verified_sha1s,
        "kernel_t0": KERNEL_T0,
        "versions": installed_versions,
        "torch_version": torch.__version__,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
    tmp_out = os.path.join(KAGGLE_TEMP, "p10d_timing.json")
    final_out = os.path.join(KAGGLE_WORKING, "p10d_timing.json")
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(timing_record, f, indent=2)
    shutil.copyfile(tmp_out, final_out + ".part")
    os.replace(final_out + ".part", final_out)
    print("=== Completed P10 arm D timing ===", flush=True)

# 8. Run kernel: convert outputs and persist briefed files for both seeds
if MODE == "run":
    for s in seeds:
        run_dir_s = os.path.join(KAGGLE_TEMP, f"run_s{s}")
        tmp_npz = os.path.join(KAGGLE_TEMP, f"p10_pertdit_seed{s}.npz")
        final_npz = os.path.join(KAGGLE_WORKING, f"p10_pertdit_seed{s}.npz")
        p10_pertdit.convert_outputs_d(h5ad_path, run_dir_s, s, tmp_npz)
        shutil.copyfile(tmp_npz, final_npz + ".part")
        os.replace(final_npz + ".part", final_npz)

        src_ckpt = os.path.join(run_dir_s, "checkpoints", "best_model.pt")
        dst_ckpt = os.path.join(KAGGLE_WORKING, f"best_model_s{s}.pt")
        shutil.copyfile(src_ckpt, dst_ckpt + ".part")
        os.replace(dst_ckpt + ".part", dst_ckpt)

        src_curve = os.path.join(run_dir_s, "logs", "training_curve.csv")
        dst_curve = os.path.join(KAGGLE_WORKING, f"training_curve_s{s}.csv")
        shutil.copyfile(src_curve, dst_curve + ".part")
        os.replace(dst_curve + ".part", dst_curve)

        src_rlog = os.path.join(run_dir_s, "logs", "resource_log.json")
        dst_rlog = os.path.join(KAGGLE_WORKING, f"resource_log_s{s}.json")
        shutil.copyfile(src_rlog, dst_rlog + ".part")
        os.replace(dst_rlog + ".part", dst_rlog)

        with open(src_rlog, "r", encoding="utf-8") as f:
            rdata = json.load(f)

        with np.load(final_npz) as z_loaded:
            t_train_val = float(z_loaded["t_train"])

        record = {
            "seed": s,
            "sha1s": verified_sha1s,
            "stop_reason": rdata["stop_reason"],
            "best_epoch": int(rdata["best_epoch"]),
            "n_epochs_run": int(rdata["n_epochs_run"]),
            "wall_s": int(rdata.get("wall_time_s", 0)),
            "n_test": int(rdata["n_test"]),
            "t_train": t_train_val,
            "of_record": (s == 117),
            "versions": installed_versions,
            "torch_version": torch.__version__,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "timing_sha1": TIMING_SHA1,
            "timing_decision": TIMING_DECISION,
        }
        tmp_rec = os.path.join(KAGGLE_TEMP, f"p10_pertdit_seed{s}_record.json")
        dst_rec = os.path.join(KAGGLE_WORKING, f"p10_pertdit_seed{s}_record.json")
        with open(tmp_rec, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        shutil.copyfile(tmp_rec, dst_rec + ".part")
        os.replace(dst_rec + ".part", dst_rec)

    print("=== Completed P10 arm D run ===", flush=True)
'''


def build_kernels(
    emb_record_path,
    timing_json_path=None,
    out_root=None,
    src_root=None,
    compound_map_path=None,
    dose_list_path=None,
    bundle_path=None,
    carves_path=None,
    p10_mlp_path=None,
    p10_pertdit_path=None,
    train_pertdit_path=None,
    pertdit_src_dir=None,
):
    """Build the Kaggle kernel directories and apexblue/lincs-p10d-src folder."""
    if out_root is None:
        out_root = os.path.join(REPO_ROOT, "external", "kaggle_kernels")
    default_out_root = os.path.join(REPO_ROOT, "external", "kaggle_kernels")

    if src_root is None:
        if os.path.abspath(out_root) == os.path.abspath(default_out_root):
            src_root = os.path.join(REPO_ROOT, "external", "kaggle_p10d_src")
        else:
            src_root = os.path.join(out_root, "kaggle_p10d_src")

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(src_root, exist_ok=True)

    # 1. Resolve source files
    if bundle_path is None:
        bundle_path = os.path.join(REPO_ROOT, "external", "xpert_split_bundle", "xpert_mdmt_splits.npz")
    if carves_path is None:
        carves_path = os.path.join(REPO_ROOT, "model", "results", "p10", "p10_carves.json")
    if p10_mlp_path is None:
        p10_mlp_path = os.path.join(REPO_ROOT, "model", "v9", "p10_mlp.py")
    if p10_pertdit_path is None:
        p10_pertdit_path = os.path.join(REPO_ROOT, "model", "v9", "p10_pertdit.py")
    if train_pertdit_path is None:
        train_pertdit_path = os.path.join(
            REPO_ROOT, "external", "bai_benchmark_1d750cd", "experiments", "_shared", "training", "train_pertdit.py"
        )
    if pertdit_src_dir is None:
        pertdit_src_dir = os.path.join(
            REPO_ROOT, "external", "bai_benchmark_1d750cd", "models", "PertDiT", "repo", "src"
        )

    # 2. Load emb_record
    with open(emb_record_path, "r", encoding="utf-8") as f:
        emb_record = json.load(f)

    sha1s_rec = emb_record.get("sha1s", emb_record)
    drug_emb_sha1 = sha1s_rec.get("bench_drug_emb.pkl") or sha1s_rec.get("bench_drug_emb")
    dose_emb_sha1 = sha1s_rec.get("bench_dose_emb.pt") or sha1s_rec.get("bench_dose_emb")

    if not drug_emb_sha1:
        fatal("emb_record missing key 'bench_drug_emb.pkl'")
    if not dose_emb_sha1:
        fatal("emb_record missing key 'bench_dose_emb.pt'")

    # 3. PI (W40 review): the h5ad's compound map and dose list are the very files the embeddings were built from. No
    # fallbacks; a mismatch would give rows another compound's embedding, silently, so it is refused here.
    inputs_default = os.path.join(REPO_ROOT, "model", "results", "p10", "p10d_inputs")
    if compound_map_path is None:
        compound_map_path = os.path.join(inputs_default, "compound_map.json")
    if dose_list_path is None:
        dose_list_path = os.path.join(inputs_default, "dose_list.json")
    for need in (compound_map_path, dose_list_path):
        if not os.path.isfile(need):
            fatal(f"missing {need}")
    compound_map_sha1 = sha1_file(compound_map_path)
    dose_list_sha1 = sha1_file(dose_list_path)
    if emb_record.get("compound_map_sha1") != compound_map_sha1:
        fatal(f"the embeddings were built with compound map {emb_record.get('compound_map_sha1')}, not {compound_map_sha1}")
    with open(dose_list_path, "r", encoding="utf-8") as f:
        dose_list_data = json.load(f)
    if "dose_list" in emb_record:
        if [float(d) for d in emb_record["dose_list"]] != [float(d) for d in dose_list_data]:
            fatal(f"the embeddings were built for doses {emb_record['dose_list']}, not {dose_list_data}")
    elif emb_record.get("dose_list_sha1") != dose_list_sha1:
        fatal(f"the embeddings were built with dose list {emb_record.get('dose_list_sha1')}, not {dose_list_sha1}")

    # 4. Populate apexblue/lincs-p10d-src
    src_train_pertdit = os.path.join(src_root, "train_pertdit.py")
    shutil.copyfile(train_pertdit_path, src_train_pertdit)

    src_cap_trainer = os.path.join(src_root, "p10_train_pertdit_cap.py")
    p10_pertdit.make_cap_trainer(train_pertdit_path, src_cap_trainer)

    src_pertdit_zip = os.path.join(src_root, "pertdit_src.zip")
    make_deterministic_zip(pertdit_src_dir, src_pertdit_zip)

    src_p10_mlp = os.path.join(src_root, "p10_mlp.py")
    shutil.copyfile(p10_mlp_path, src_p10_mlp)

    src_p10_pertdit = os.path.join(src_root, "p10_pertdit.py")
    shutil.copyfile(p10_pertdit_path, src_p10_pertdit)

    src_p10_carves = os.path.join(src_root, "p10_carves.json")
    shutil.copyfile(carves_path, src_p10_carves)

    # Compute pins
    bundle_sha1 = sha1_file(bundle_path)
    carves_sha1 = sha1_file(src_p10_carves)
    p10_mlp_sha1 = sha1_file(src_p10_mlp)
    p10_pertdit_sha1 = sha1_file(src_p10_pertdit)
    trainer_sha1 = sha1_file(src_train_pertdit)
    pertdit_src_sha1 = sha1_file(src_pertdit_zip)
    cap_trainer_sha1 = sha1_file(src_cap_trainer)

    # 5. Metadata template
    def make_metadata(kernel_name):
        return {
            "id": f"apexblue/{kernel_name}",
            "title": kernel_name,
            "code_file": f"{kernel_name}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["gpu"],
            "dataset_sources": [
                "apexblue/xpert-mdmt-benchmark",
                "apexblue/p10-wheels",
                "apexblue/p10d-wheels",
                "apexblue/p10d-emb-inputs",
                "apexblue/lincs-p10d-src",
            ],
            "kernel_sources": ["apexblue/lincs-p10d-emb-merge"],
            "competition_sources": [],
            "model_sources": [],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
            "machine_shape": "NvidiaTeslaT4",
        }

    # Review 081 C2: the run kernel is bound to the timing result -- its t_test_s, sha1 and decision come from the downloaded
    # p10d_timing.json, the decision is re-derived from its timings, and "not_run_cost" builds nothing.
    timing_sha1, timing_decision_label, t_test_s = "none", "none", None
    if timing_json_path is not None:
        with open(timing_json_path, "r", encoding="utf-8") as f:
            tj = json.load(f)
        for seed_k, tm in tj["timings"].items():
            if not all(float(tm[k]) > 0 for k in ("t_pre", "t_epoch", "t_test")):
                fatal(f"timing JSON seed {seed_k} has a non-positive timing: {tm}")
        redo = p10_pertdit.timing_decision(tj["timings"], cap_s=p10_pertdit.CAP_S)
        if redo["decision"] != tj["decision"]["decision"] or abs(redo["t_test_s"] - float(tj["decision"]["t_test_s"])) > 1e-9:
            fatal(f"timing JSON's decision {tj['decision']} is not what its timings give: {redo}")
        if redo["decision"] == "not_run_cost":
            fatal("timing decision is 'not_run_cost': D is recorded as not run (cost); no run kernel is built (98.15 ruling 5)")
        timing_sha1, timing_decision_label, t_test_s = sha1_file(timing_json_path), redo["decision"], redo["t_test_s"]

    # Helper to generate kernel script
    def render_kernel(mode, t_test):
        if mode == "timing":
            mode_args = '["--n_epochs", "3"]'
        else:
            mode_args = '["--cap_s", str(CAP_S), "--t0", repr(KERNEL_T0), "--t_test_s", str(T_TEST_S)]'

        code = (
            KERNEL_TEMPLATE.replace("__MODE__", mode)
            .replace("__MODE_UPPER__", mode.upper())
            .replace("__MODE_SPECIFIC_ARGS__", mode_args)
            .replace("__T_TEST_S__", str(t_test if t_test is not None else 0.0))
            .replace("__CAP_S__", str(p10_pertdit.CAP_S))
            .replace("__BUNDLE_SHA1__", bundle_sha1)
            .replace("__CARVES_SHA1__", carves_sha1)
            .replace("__P10_MLP_SHA1__", p10_mlp_sha1)
            .replace("__P10_PERTDIT_SHA1__", p10_pertdit_sha1)
            .replace("__TRAINER_SHA1__", trainer_sha1)
            .replace("__PERTDIT_SRC_SHA1__", pertdit_src_sha1)
            .replace("__COMPOUND_MAP_SHA1__", compound_map_sha1)
            .replace("__DOSE_LIST_SHA1__", dose_list_sha1)
            .replace("__DRUG_EMB_SHA1__", drug_emb_sha1)
            .replace("__DOSE_EMB_SHA1__", dose_emb_sha1)
            .replace("__CAP_TRAINER_SHA1__", cap_trainer_sha1)
            .replace("__TIMING_SHA1__", timing_sha1 if mode == "run" else "none")
            .replace("__TIMING_DECISION__", timing_decision_label if mode == "run" else "none")
        )
        ast.parse(code)
        return code

    # 6. Build timing kernel: external/kaggle_kernels/kern_p10d_timing
    timing_dir = os.path.join(out_root, "kern_p10d_timing")
    os.makedirs(timing_dir, exist_ok=True)
    timing_code = render_kernel("timing", 0.0)
    with open(os.path.join(timing_dir, "lincs-p10d-timing.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(timing_code)
    with open(os.path.join(timing_dir, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(make_metadata("lincs-p10d-timing"), f, indent=1)
        f.write("\n")

    # 7. Build run kernel (only from a timing JSON): external/kaggle_kernels/kern_p10d_run
    if t_test_s is not None:
        run_dir = os.path.join(out_root, "kern_p10d_run")
        os.makedirs(run_dir, exist_ok=True)
        run_code = render_kernel("run", float(t_test_s))
        with open(os.path.join(run_dir, "lincs-p10d-run.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(run_code)
        with open(os.path.join(run_dir, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(make_metadata("lincs-p10d-run"), f, indent=1)
            f.write("\n")


def main():
    p = argparse.ArgumentParser(description="Make Kaggle kernels for P10 arm D (PertDiT)")
    p.add_argument("--emb_record", required=True, help="Path to emb_record.json")
    p.add_argument("--timing_json", type=str, default=None, help="Downloaded p10d_timing.json (enables the run kernel build)")
    p.add_argument("--out_root", type=str, default=None, help="Root directory for kernels")
    p.add_argument("--src_root", type=str, default=None, help="Root directory for lincs-p10d-src")
    p.add_argument("--compound_map", type=str, default=None, help="Path to compound_map.json")
    p.add_argument("--dose_list", type=str, default=None, help="Path to dose_list.json")
    p.add_argument("--bundle", type=str, default=None, help="Path to bundle npz")
    p.add_argument("--carves", type=str, default=None, help="Path to carves json")

    args = p.parse_args()
    build_kernels(
        emb_record_path=args.emb_record,
        timing_json_path=args.timing_json,
        out_root=args.out_root,
        src_root=args.src_root,
        compound_map_path=args.compound_map,
        dose_list_path=args.dose_list,
        bundle_path=args.bundle,
        carves_path=args.carves,
    )


if __name__ == "__main__":
    main()
