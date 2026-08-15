# -*- coding: utf-8 -*-
"""v7 configuration. Inherits v6's data config; every new value carries the reason it exists.
Design rationale: ../V7_PLAN.md. v6's measurements that motivate each change: ../results/CLAIMS.md."""
import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v6"))
from config_v6 import V6DataConfig, resolve_paths        # noqa: F401  (re-exported for the trainer)


@dataclass
class V7Config:
    # ---- unchanged from v6 (frozen by the data, or measured-good) ----
    n_genes: int = 978
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    l_base: int = 2
    l_perturb: int = 4
    max_atoms: int = 96
    dropout: float = 0.1
    d_atom: int = 512
    d_epi: int = 3
    d_global: int = 512 + 20 + 2048       # UniMol CLS + descriptors + ECFP4 (ChemBERTa dropped, dR2 +0.001)
    d_cell_ctx: int = 16
    d_pathway: int = 64
    pathway_epi_gate: bool = True
    d_epi_hidden: int = 16
    huber_delta: float = 1.0

    # ---- v7: make the priors load-bearing -------------------------------------------------------
    use_aux: bool = True          # deep supervision of the pathway + chromatin branches. v6 asked nothing
                                  # of either and measured both at ~0 while they were live.
    use_ppi: bool = True          # STRING message passing. 12,665 edges v6 never even loaded.
    stoch_depth: float = 0.1      # max residual-drop rate, scaled linearly with depth. Regularises AND
                                  # removes the option to ignore a branch (residual "loafing"), which is
                                  # v6's pathway layer exactly: live but worth +0.0003 [4.13].
    aux_pathway_w: float = 1.0    # initial scale only -- UncertaintyWeighting learns the real balance
    aux_epi_w: float = 1.0


@dataclass
class V7TrainConfig:
    """The modern recipe. Expect a few points from this block, not a transformation -- v6's gap is the
    drug x cell interaction (V7_PLAN §2), which no optimiser fixes."""
    epochs: int = 12
    batch: int = 48
    lr: float = 4e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"      # "adamw" | "muon"; Muon/SOAP beat AdamW at scale, smaller edge at ours
    schedule: str = "wsd"         # warmup -> stable -> decay; standard in current large-scale recipes
    warmup_frac: float = 0.03
    decay_frac: float = 0.20      # final fraction of steps spent annealing
    ema_decay: float = 0.999      # EMA of weights: documented gains in ROBUSTNESS TO NOISY LABELS, and
                                  # ~75% of LINCS is inert noise -- the best-matched item in the recipe
    grad_clip: float = 1.0
    budget_h: float = 8.5
    fold: int = 0
    workers: int = 2
