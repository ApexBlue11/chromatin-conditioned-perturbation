# -*- coding: utf-8 -*-
"""RESULTS §98.2, §98.6: Build the six Kaggle CPU kernels for P10 arm M and M+dt.

Arms:
- m: Bai et al.'s drug-free MLP, inputs basal profile X_ctl only.
- mdt: M with standardized log-dose and time appended (M+dt).

Seeds: 131419 (s0), 131420 (s1), 131421 (s2).
"""
import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "model", "v9"))
import p10_mlp  # noqa: E402


def sha1_file(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


KERNEL_TEMPLATE = '''# -*- coding: utf-8 -*-
"""RESULTS 98.2, 98.6: P10 arm __ARM_UPPER__ seed __SEED__ (__ARM_LABEL__).
Kaggle CPU kernel, internet off, blinded.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ARM = '__ARM__'
SEED = __SEED__
K = __K__

BUNDLE_SHA1 = '__BUNDLE_SHA1__'
TRAIN_MLP_SHA1 = '__TRAIN_MLP_SHA1__'
P10_MLP_SHA1 = '__P10_MLP_SHA1__'
CARVES_SHA1 = '__CARVES_SHA1__'
DT_SCRIPT_SHA1 = '__DT_SCRIPT_SHA1__'


def fatal(msg):
    raise SystemExit("FATAL: " + msg)


def sha1_file(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def find_one(pattern):
    hits = glob.glob(pattern, recursive=True)
    if len(hits) != 1:
        fatal(f"expected exactly 1 match for {pattern}, found {len(hits)}: {hits}")
    return hits[0]


print(f"=== Starting P10 arm {ARM} seed {SEED} ===", flush=True)

# 1. Install wheels
whl_file = find_one("/kaggle/input/**/scanpy-1.12.4-py3-none-any.whl")
wheels_dir = os.path.dirname(whl_file)
print(f"GUARD 1: installing scanpy and anndata wheels from {wheels_dir}", flush=True)
res = subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index",
    "--find-links", wheels_dir,
    "scanpy==1.12.4", "anndata==0.13.3.post0",
])
if res.returncode != 0:
    fatal(f"pip install exited {res.returncode}")

# 2. Mounted-file guards
bundle_path = find_one("/kaggle/input/**/xpert_mdmt_splits.npz")
train_mlp_path = find_one("/kaggle/input/**/train_mlp.py")
p10_mlp_path = find_one("/kaggle/input/**/p10_mlp.py")
carves_path = find_one("/kaggle/input/**/p10_carves.json")

verified_sha1s = {}
for name, p, expected in [
    ("bundle", bundle_path, BUNDLE_SHA1),
    ("train_mlp", train_mlp_path, TRAIN_MLP_SHA1),
    ("p10_mlp", p10_mlp_path, P10_MLP_SHA1),
    ("carves", carves_path, CARVES_SHA1),
]:
    got = sha1_file(p)
    print(f"GUARD 2: {name} sha1 {got}", flush=True)
    if got != expected:
        fatal(f"GUARD 2: {name} sha1 mismatch: {got} != {expected}")
    verified_sha1s[name] = got

# Add p10_mlp directory to sys.path
sys.path.insert(0, os.path.dirname(p10_mlp_path))
import p10_mlp

temp_dir = "/kaggle/temp" if os.path.exists("/kaggle/temp") else "/tmp"
os.makedirs(temp_dir, exist_ok=True)

if ARM == "mdt":
    dt_script_path = os.path.join(temp_dir, "p10_train_mlp_dt.py")
    p10_mlp.make_dt_script(train_mlp_path, dt_script_path)
    got_dt = sha1_file(dt_script_path)
    print(f"GUARD 2: generated dt_script sha1 {got_dt}", flush=True)
    if got_dt != DT_SCRIPT_SHA1:
        fatal(f"GUARD 2: dt_script sha1 mismatch: {got_dt} != {DT_SCRIPT_SHA1}")
    verified_sha1s["dt_script"] = got_dt
    trainer_script = dt_script_path
else:
    trainer_script = train_mlp_path

# 3. Build h5ad
h5ad_path = os.path.join(temp_dir, f"p10_{ARM}.h5ad")
print(f"Building h5ad to {h5ad_path} (with_dt={ARM == 'mdt'})", flush=True)
p10_mlp.build_h5ad(bundle_path, carves_path, h5ad_path, with_dt=(ARM == "mdt"))

# 4. Run trainer as subprocess
run_dir = os.path.join(temp_dir, "run")
os.makedirs(run_dir, exist_ok=True)
log_file = os.path.join(temp_dir, "trainer.log")

cmd = [
    sys.executable, trainer_script,
    "--h5ad", h5ad_path,
    "--split_col", "p10_split",
    "--output_dir", run_dir,
    "--seed", str(SEED),
    "--epochs", "500",
]
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "4"
env["MKL_NUM_THREADS"] = "4"

print(f"Running trainer: {' '.join(cmd)}", flush=True)
t0 = time.time()
with open(log_file, "w", encoding="utf-8") as f_log:
    proc = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, env=env)
wall_s = int(time.time() - t0)

if proc.returncode != 0:
    fatal(f"trainer exited with code {proc.returncode}")

# 5. Blinding: print only allowed lines
print("=== Validation curve (blinded test) ===", flush=True)
with open(log_file, "r", encoding="utf-8", errors="replace") as f_log:
    for line in f_log:
        if line.startswith("Epoch ") or line.startswith("Early stopping") or line.startswith("Best epoch"):
            print(line, end="", flush=True)

# 6. Convert outputs and stage to /kaggle/working
working_dir = "/kaggle/working"
out_npz = os.path.join(working_dir, f"p10_{ARM}_seed{K}.npz")
out_npz_part = out_npz + ".part"
p10_mlp.convert_outputs(h5ad_path, run_dir, SEED, ARM, out_npz_part)
os.replace(out_npz_part, out_npz)

# Copy checkpoints/best_model.pt and logs/resource_log.json
src_ckpt = os.path.join(run_dir, "checkpoints", "best_model.pt")
dst_ckpt = os.path.join(working_dir, "best_model.pt")
shutil.copyfile(src_ckpt, dst_ckpt + ".part")
os.replace(dst_ckpt + ".part", dst_ckpt)

src_rlog = os.path.join(run_dir, "logs", "resource_log.json")
dst_rlog = os.path.join(working_dir, "resource_log.json")
shutil.copyfile(src_rlog, dst_rlog + ".part")
os.replace(dst_rlog + ".part", dst_rlog)

with open(src_rlog, "r", encoding="utf-8") as f:
    rlog_data = json.load(f)

best_epoch = int(rlog_data["best_epoch"])
n_epochs_run = int(rlog_data["n_epochs_run"])
n_test = int(rlog_data["n_test"])

record = {
    "seed": SEED,
    "arm": ARM,
    "sha1s": verified_sha1s,
    "best_epoch": best_epoch,
    "n_epochs_run": n_epochs_run,
    "wall_s": wall_s,
    "n_test": n_test,
}
record_path = os.path.join(working_dir, f"p10_{ARM}_seed{K}_record.json")
with open(record_path + ".part", "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2)
os.replace(record_path + ".part", record_path)

print(f"=== Completed P10 arm {ARM} seed {SEED} ===", flush=True)
'''


def build_kernels(out_root):
    """Build the six Kaggle kernel directories under out_root."""
    bundle_path = os.path.join(
        REPO_ROOT, "external", "xpert_split_bundle", "xpert_mdmt_splits.npz"
    )
    train_mlp_path = os.path.join(
        REPO_ROOT,
        "external",
        "bai_benchmark_1d750cd",
        "experiments",
        "_shared",
        "training",
        "train_mlp.py",
    )
    p10_mlp_path = os.path.join(REPO_ROOT, "model", "v9", "p10_mlp.py")
    p10_carves_path = os.path.join(
        REPO_ROOT, "model", "results", "p10", "p10_carves.json"
    )

    bundle_sha1 = sha1_file(bundle_path)
    train_mlp_sha1 = sha1_file(train_mlp_path)
    p10_mlp_sha1 = sha1_file(p10_mlp_path)
    carves_sha1 = sha1_file(p10_carves_path)

    # Compute M+dt trainer sha1
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
        temp_dt = tf.name
    try:
        p10_mlp.make_dt_script(train_mlp_path, temp_dt)
        dt_script_sha1 = sha1_file(temp_dt)
    finally:
        if os.path.exists(temp_dt):
            os.remove(temp_dt)

    # Read docker image from kern_v9p9
    v9p9_meta_path = os.path.join(
        REPO_ROOT,
        "external",
        "kaggle_kernels",
        "kern_v9p9",
        "kernel-metadata.json",
    )
    with open(v9p9_meta_path, "r", encoding="utf-8") as f:
        v9p9_meta = json.load(f)
    docker_image = v9p9_meta["docker_image"]

    configs = [
        ("m", "M, drug-free MLP as published", 0, 131419),
        ("m", "M, drug-free MLP as published", 1, 131420),
        ("m", "M, drug-free MLP as published", 2, 131421),
        ("mdt", "M+dt, with dose and time", 0, 131419),
        ("mdt", "M+dt, with dose and time", 1, 131420),
        ("mdt", "M+dt, with dose and time", 2, 131421),
    ]

    for arm, label, k, seed in configs:
        folder_name = f"kern_p10_{arm}_s{k}"
        kernel_dir = os.path.join(out_root, folder_name)
        os.makedirs(kernel_dir, exist_ok=True)

        code_filename = f"lincs-p10-{arm}-s{k}.py"
        script_code = (
            KERNEL_TEMPLATE.replace("__ARM__", arm)
            .replace("__ARM_UPPER__", arm.upper())
            .replace("__ARM_LABEL__", label)
            .replace("__SEED__", str(seed))
            .replace("__K__", str(k))
            .replace("__BUNDLE_SHA1__", bundle_sha1)
            .replace("__TRAIN_MLP_SHA1__", train_mlp_sha1)
            .replace("__P10_MLP_SHA1__", p10_mlp_sha1)
            .replace("__CARVES_SHA1__", carves_sha1)
            .replace("__DT_SCRIPT_SHA1__", dt_script_sha1)
        )

        # Validate ast.parse
        ast.parse(script_code)

        script_path = os.path.join(kernel_dir, code_filename)
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script_code)

        metadata = {
            "id": f"apexblue/lincs-p10-{arm}-s{k}",
            "title": f"lincs-p10-{arm}-s{k}",
            "code_file": code_filename,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": [],
            "dataset_sources": [
                "apexblue/xpert-mdmt-benchmark",
                "apexblue/p10-wheels",
                "apexblue/lincs-p10-src",
            ],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "docker_image": docker_image,
        }

        meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
        with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(metadata, f, indent=1)
            f.write("\n")

        print(f"Wrote {folder_name} (seed {seed})", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate P10 Kaggle CPU kernels"
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(REPO_ROOT, "external", "kaggle_kernels"),
        help="Target directory for kernel folders",
    )
    args = parser.parse_args()
    build_kernels(args.out_dir)


if __name__ == "__main__":
    main()
