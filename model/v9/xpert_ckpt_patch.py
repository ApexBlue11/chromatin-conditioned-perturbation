# -*- coding: utf-8 -*-
"""Activation checkpointing for XPert's encoder layers, applied at RUNTIME so their source files stay verbatim.
[RESULTS 77, packet 011]

Their published recipe (batch 128, 978 gene tokens) needs ~0.117 GiB of stored activations per sample in training --
~14.95 GiB at batch 128 -- against a T4's 14.56 GiB. Checkpointing stores only layer-boundary activations and recomputes
the rest in the backward pass: same batch, same loss, same gradients (model/v9/prove_checkpoint_exact.py: loss equal to
8 decimals, gradient difference 5.9e-05 inside the 8.7e-05 run-to-run noise floor), ~3.7 GiB at batch 128.

Active only in training with grad enabled, so inference and evaluation are untouched. use_reentrant=False, which
preserves the RNG state for dropout replay and the autocast state for recomputation.
"""
import sys

import torch
import torch.utils.checkpoint as cp


def apply(model_utils=None):
    """Patch models.model_utils.Encoder and .crossEncoder in place. Returns the list of patched class names."""
    MU = model_utils or sys.modules['models.model_utils']
    patched = []
    for cls in (MU.Encoder, MU.crossEncoder):
        if getattr(cls.forward, '_lincs_checkpointed', False):
            continue
        orig = cls.forward

        def make(orig):
            def fwd(self, *a, **k):
                if self.training and torch.is_grad_enabled():
                    return cp.checkpoint(orig, self, *a, use_reentrant=False, **k)
                return orig(self, *a, **k)
            fwd._lincs_checkpointed = True
            return fwd

        cls.forward = make(orig)
        patched.append(cls.__name__)
    return patched
