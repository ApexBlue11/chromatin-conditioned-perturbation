import time
KERNEL_T0 = time.time()
# -*- coding: utf-8 -*-
"""RESULTS 98.15, 98.16: P10 arm D PertDiT (TIMING).
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

MODE = 'timing'
PINNED_GPU_VERSIONS = "scanpy=1.12.4 anndata=0.13.3.post0 diffusers=0.30.2 einops=0.8.0 huggingface_hub=0.24.6 rdkit=2023.09.1"
TIMING_SHA1 = 'none'
TIMING_DECISION = 'none'
T_TEST_S = 0.0
CAP_S = 21600.0

BUNDLE_SHA1 = '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464'
CARVES_SHA1 = 'd6fc78447228b87641cbd5af8ff0143a714edeb8'
P10_MLP_SHA1 = '97f84874df4609c463289fbc9bf387de3ff8775a'
P10_PERTDIT_SHA1 = '929ce1fe6d9e9c87c89eccb153f5c7d8c8926b12'
TRAINER_SHA1 = 'a431f53ba18d43348161ee19b97b01148229f1ac'
PERTDIT_SRC_SHA1 = '06bf66bdb2b21f53ddf72d6f1459636e96bca7ad'
COMPOUND_MAP_SHA1 = '918e08f61352e8f122c3451d71a800e616de4f7c'
DOSE_LIST_SHA1 = '7941bcd3155ae73fbdcca2f11ce6af647300d2ce'
DRUG_EMB_SHA1 = 'eba75fdc2ef761fbd844dc7a2a967d4e8a3b0a0c'
DOSE_EMB_SHA1 = 'c42bce8eeca3cf47c4ff25e16608c6e00474f1a0'
CAP_TRAINER_SHA1 = 'e910a855bb4405a7bb1fb56c84572470be9c6316'


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
    cmd += ["--n_epochs", "3"]

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
            stamped = f"{time.time():.3f}\t{line}"
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
        fatal(f"seed {s} trainer exited with code {ret}.\nLast 30 lines:\n{last30}")

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
