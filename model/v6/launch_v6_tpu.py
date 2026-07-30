# -*- coding: utf-8 -*-
"""
Launcher for v6 TPU training. EXISTS SOLELY TO SET THE ENVIRONMENT BEFORE torch_xla IS IMPORTED.

TPU_NOTES.md issue #4: torch_xla reads its configuration at import time. Setting XLA_USE_BF16 (or any
XLA_* flag) inside the training function -- after `import torch_xla` at module scope -- silently does
nothing, and you keep paying fp32 on bf16-native hardware. So the env must be set here, first.
"""
import os, sys

os.environ.setdefault("XLA_USE_BF16", "1")           # MXU is bf16-native
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_v6_tpu                                   # torch_xla imported HERE, after the env is set

if __name__ == "__main__":
    train_v6_tpu.main()
