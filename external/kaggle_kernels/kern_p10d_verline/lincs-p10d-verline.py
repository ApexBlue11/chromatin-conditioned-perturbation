# RESULTS 98.20 (review 081 C4): on the pinned image, internet off, install exactly the D GPU kernel's wheel set the way that
# kernel does and print its version line, so the kernel's PINNED_GPU_VERSIONS assertion is checked before any GPU session.
import glob, os, subprocess, sys
PINNED = "scanpy=1.12.4 anndata=0.13.3.post0 diffusers=0.30.2 einops=0.8.0 huggingface_hub=0.24.6 rdkit=2023.09.1"
d1 = os.path.dirname(glob.glob("/kaggle/input/**/scanpy-1.12.4-py3-none-any.whl", recursive=True)[0])
d2 = os.path.dirname(glob.glob("/kaggle/input/**/diffusers-0.30.2-py3-none-any.whl", recursive=True)[0])
r1 = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", d1, "scanpy==1.12.4", "anndata==0.13.3.post0"])
r2 = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", d2,
                     "diffusers==0.30.2", "einops==0.8.0", "huggingface-hub==0.24.6", "rdkit==2023.9.1"])
print("pip rc", r1.returncode, r2.returncode, flush=True)
chk = subprocess.run([sys.executable, "-c",
    "import scanpy, anndata, diffusers, einops, huggingface_hub, rdkit; "
    "print(f'scanpy={scanpy.__version__} anndata={anndata.__version__} diffusers={diffusers.__version__} "
    "einops={einops.__version__} huggingface_hub={huggingface_hub.__version__} rdkit={rdkit.__version__}')"],
    capture_output=True, text=True)
line = chk.stdout.strip()
print("VERSION LINE", repr(line), flush=True)
print("MATCHES PIN", line == PINNED, flush=True)
print("stderr tail", chk.stderr[-1500:], flush=True)
# and the trainer's own imports under that stack (diffusers schedulers, rdkit logger, scanpy transform)
chk2 = subprocess.run([sys.executable, "-c",
    "from rdkit import RDLogger; from diffusers import DDIMScheduler, DDPMScheduler; import einops, scanpy as sc, anndata, numpy as np;"
    "a = anndata.AnnData(X=np.random.rand(5, 978).astype('float32')); sc.pp.normalize_total(a); sc.pp.log1p(a); print('trainer imports ok')"],
    capture_output=True, text=True)
print(chk2.stdout.strip(), chk2.returncode, chk2.stderr[-800:], flush=True)
