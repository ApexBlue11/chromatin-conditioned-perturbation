# Environments

Three environments, deliberately split (rdkit and pybigtools have no Python 3.14 Windows wheels; torch does).

| File | Env | Python | Purpose |
|---|---|---|---|
| `requirements-system.txt` | system interpreter | 3.14.3 | model code, 45 unit tests, all CPU analyses |
| `requirements-drug.txt` | `drug/.venv-drug` | 3.12.13 | SMILES → descriptors/fingerprints, scaffold split |
| `requirements-epi.txt` | `phase2_assembly/.venv-epi` | 3.12.13 | bigWig/narrowPeak epigenetics extraction |

Recreate with [uv](https://github.com/astral-sh/uv):

    uv venv --python 3.12 drug/.venv-drug
    uv pip install --python drug/.venv-drug/Scripts/python.exe -r env/requirements-drug.txt

## Accelerated compute (Kaggle)
Local machine has no GPU. Training and molecular embeddings ran on Kaggle:
- **GPU:** T4×2 (`--accelerator NvidiaTeslaT4`). **Never P100** — Kaggle's torch ships sm_70+ kernels only,
  P100 is sm_60 and fails at the first matmul with "no kernel image is available".
- **TPU:** v3-8, `torch_xla` 2.8.0, 8 cores (`train_tpu.py`).
- **CPU kernels are free** and were used for every inference-only analysis.
- Kaggle base image pins its own torch; our code targets torch ≥ 2.8 and is version-agnostic apart from
  `torch.amp` API usage (`torch.amp.GradScaler("cuda")`, not the deprecated `torch.cuda.amp.*`).
