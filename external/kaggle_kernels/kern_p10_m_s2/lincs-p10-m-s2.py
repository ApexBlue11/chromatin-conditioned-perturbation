# -*- coding: utf-8 -*-
"""RESULTS 98.2, 98.6: P10 arm M seed 131421 (M, drug-free MLP as published).
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

ARM = 'm'
SEED = 131421
K = 2

BUNDLE_SHA1 = '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464'
TRAIN_MLP_SHA1 = 'ddc0725ee0a7d73bae0ac478d92e5f8a879b13bd'
P10_MLP_SHA1 = '97f84874df4609c463289fbc9bf387de3ff8775a'
CARVES_SHA1 = 'd6fc78447228b87641cbd5af8ff0143a714edeb8'
DT_SCRIPT_SHA1 = '09bcf0b2d0c1afebdd87324043f21dc65f460644'


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

# GUARD 3 (review 075 C1): what the TRAINER itself saw. Their main() falls back to the TEST set for validation when the
# 'valid' mask is empty; its echoed val_deg lines would then be test PCC and its checkpoint test-selected. Refuse unless
# the trainer's own split counts are the registered ones and it printed no fallback -- before anything is echoed.
with open(log_file, "r", encoding="utf-8", errors="replace") as f_log:
    trainer_text = f_log.read()
for need in ("  train: 49846 samples", "  valid: 5539 samples", "  test: 13445 samples"):
    if need not in trainer_text:
        fatal(f"GUARD 3: trainer log lacks {need.strip()!r} (review 075 C1)")
if "No valid set found" in trainer_text:
    fatal("GUARD 3: the trainer fell back to the test set for validation (review 075 C1)")
print("GUARD 3: trainer saw train 49846 / valid 5539 / test 13445, no test fallback", flush=True)

# 5. Blinding: print only allowed lines
print("=== Validation curve (blinded test) ===", flush=True)
with open(log_file, "r", encoding="utf-8", errors="replace") as f_log:
    for line in f_log:
        if line.startswith("Epoch ") or line.startswith("Early stopping") or line.startswith("Best epoch"):
            print(line, end="", flush=True)

# 6. Convert outputs and stage to /kaggle/working
working_dir = "/kaggle/working"
out_npz = os.path.join(working_dir, f"p10_{ARM}_seed{K}.npz")
# RESULTS 98.8: np.savez appends '.npz' to a path that lacks it, so convert into /kaggle/temp under the final name, then
# copy -> .part -> os.replace (the v1 kernels wrote '<name>.npz.part.npz' and crashed on the rename after training).
tmp_npz = os.path.join(temp_dir, f"p10_{ARM}_seed{K}.npz")
p10_mlp.convert_outputs(h5ad_path, run_dir, SEED, ARM, tmp_npz)
if not os.path.isfile(tmp_npz):
    fatal(f"convert_outputs did not write {tmp_npz}")
shutil.copyfile(tmp_npz, out_npz + ".part")
os.replace(out_npz + ".part", out_npz)

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
