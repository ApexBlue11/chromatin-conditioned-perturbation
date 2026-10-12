# P10 D stack probe (RESULTS 98.16): on the pinned image with PertDiT's environment pins (diffusers 0.30.2, einops 0.8.0,
# huggingface-hub 0.24.6) and each candidate rdkit, do their trainer's imports work under the image's torch 2.10 / numpy 2.0.2,
# and does PertDiT's Crossformer (vendored at Bai et al. 1d750cd, fetched read-only) train one step and DDIM-sample on CPU?
# Random tensors only; no LINCS data.
import os, subprocess, sys, textwrap, urllib.request
SHA = '1d750cd098f0d32187a9cb28244e8846bc40b0bc'
BASE = f'https://raw.githubusercontent.com/baijinming97/drug-perturbation-benchmark/{SHA}/models/PertDiT/repo/src/'
FILES = ['model/__init__.py', 'model/CrossDit.py', 'model/common.py', 'sampler/__init__.py', 'sampler/Sampler.py',
         'trainer/__init__.py', 'trainer/optimizer.py', 'utils/__init__.py', 'utils/seed_everything.py']
ROOT = '/kaggle/temp/pertdit_src'
for f in FILES:
    os.makedirs(os.path.dirname(os.path.join(ROOT, f)), exist_ok=True)
    urllib.request.urlretrieve(BASE + f, os.path.join(ROOT, f))
print('fetched', len(FILES), flush=True)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'diffusers==0.30.2', 'einops==0.8.0',
                    'huggingface-hub==0.24.6'], capture_output=True, text=True)
print('pip pins rc', r.returncode, flush=True)
TEST = textwrap.dedent(r'''
import sys, time, numpy, torch
sys.path.insert(0, %r)
from rdkit import RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')
import diffusers, einops, huggingface_hub
print('versions rdkit', rdBase.rdkitVersion, 'diffusers', diffusers.__version__, 'einops', einops.__version__,
      'hub', huggingface_hub.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__, flush=True)
from model.CrossDit import Crossformer
from sampler.Sampler import Diffusion_Sampler
from trainer.optimizer import get_optimizer_scheduler
from utils.seed_everything import seed_everything
seed_everything(117)
m = Crossformer(num_steps=50, num_layers=2, d_model=256, d_cond=1024, d_pre=978, mlp_hidden_dim=256, num_heads=4,
                dropout=0.0, nonorm=False)
m.init_weights()
s = Diffusion_Sampler(sampler_type='DDPM', num_train_timesteps=50, timesteps=50, start=0.002, end=0.4,
                      beta_schedule='scaled_linear', device=torch.device('cpu'), guidance_scale=3.0, uncond=None)
opt, sch = get_optimizer_scheduler(m.parameters(), lr_max=1e-3, warmup_n_steps=5000, lr_start=5e-4, T_max=1000000, lr_min=5e-5)
x = torch.rand(64, 978) * 5; c = torch.rand(64, 978) * 5; d = torch.randn(64, 40, 1024); mask = [40] * 64
t = time.time()
for _ in range(5):
    loss = s.diffusion_loss_fn(m, x, c, d, mask); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
print('params', sum(p.numel() for p in m.parameters()), 'cpu s/step %%.3f' %% ((time.time() - t) / 5), 'loss', float(loss), flush=True)
m.eval(); t = time.time()
with torch.no_grad():
    y = s.ddim_sample(m, x.shape, c, d, mask)
print('ddim sample', tuple(y.shape), 'cpu s/batch %%.2f' %% (time.time() - t), 'finite', bool(torch.isfinite(y).all()), flush=True)
''' % ROOT)
for rd in ['rdkit==2023.9.1', 'rdkit==2024.9.6']:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', rd], capture_output=True, text=True)
    print('==', rd, 'pip rc', r.returncode, flush=True)
    r = subprocess.run([sys.executable, '-c', TEST], capture_output=True, text=True)
    print(r.stdout[-3000:], '\nrc', r.returncode, r.stderr[-2500:], flush=True)
