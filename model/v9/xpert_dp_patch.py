# -*- coding: utf-8 -*-
"""DataParallel for XPert over every visible GPU, applied at RUNTIME inside XPertNet.forward so their files stay
verbatim. [RESULTS 78, review 012]

Why inside forward rather than wrapping the model in nn.DataParallel: the object their train_xpert.py holds stays an
XPertNet. So state_dict() keys carry NO 'module.' prefix -- review 012 C1 showed their --resume_from filter would match
no key of a prefixed checkpoint and silently restart from random weights while logging success -- and every attribute
their code touches keeps working.

Why it is exact for their recipe, where gradient accumulation was not [RESULTS 77.2]: torch.nn.parallel.data_parallel
scatters the batch, runs one replica per GPU, and GATHERS the outputs to GPU 0 before their train() computes the loss,
so batch_weighted_loss's sqrt(loss / num_samples) terms see all 128 samples exactly as on one GPU. Gradients from the
replicas are summed into the original parameters. XPert has LayerNorm only, no BatchNorm, so no statistic depends on
the per-replica batch. Dropout masks differ per replica: i.i.d. draws of the same Bernoulli, a different realisation,
as a different seed is [review 012 ask 3].

Two things in their forward are pinned to one device and are localised on each replica:
  * `self.device`, which forward uses to move every input (model_XPert.py:188-198);
  * `self.drug_HG_embed`, a plain tensor attribute created on `device` (model_XPert.py:135). DataParallel replicates
    parameters and buffers only; a plain tensor stays on GPU 0. It is constant (never trained), so a cached copy per
    device is exact. The localiser is generic over every plain tensor attribute of every submodule, and
    `plain_tensor_attributes()` lists them so a kernel guard can assert the list is exactly the one expected.

Active only in training with grad enabled and at least one row per GPU, so validation, prediction and a ragged last
batch smaller than the GPU count run unchanged on one device.
"""
import sys

import torch
from torch.nn.parallel import data_parallel

# (id(source), device) -> (source, copy). The source is kept and checked by IDENTITY, so a reused id() can never
# serve a stale copy [RESULTS 80.5, Amendment F].
_CACHE = {}
# Only these plain tensor attributes are moved to a replica's device; any other tensor attribute found off-device
# is an error, not something to copy. On a replica the parameter copies are plain attributes too (replicate()
# sets them that way), already on the device -- they must never be moved.
LOCALISE = frozenset(['drug_HG_embed'])
# Set by the probe: True asserts autocast is ON inside every replica forward, False asserts it is OFF, None skips.
EXPECT_AUTOCAST = None
REPLICA_CALLS = {'n': 0, 'autocast_on': 0}


def plain_tensor_attributes(model):
    """Every tensor held as a plain attribute (not a parameter or buffer), as 'module.path.attr'."""
    out = []
    for mname, m in model.named_modules():
        for name, val in vars(m).items():
            if torch.is_tensor(val):
                out.append((mname + '.' if mname else '') + name)
    return sorted(out)


def _localise(replica, dev):
    for m in replica.modules():
        for name, val in list(vars(m).items()):
            if not torch.is_tensor(val) or val.device == dev:
                continue
            if name not in LOCALISE:
                raise RuntimeError('xpert_dp_patch: tensor attribute %r is on %s, not the replica device %s, and is not '
                                   'in the localise set %s' % (name, val.device, dev, sorted(LOCALISE)))
            key = (id(val), dev)
            hit = _CACHE.get(key)
            if hit is None or hit[0] is not val:
                hit = (val, val.to(dev))
                _CACHE[key] = hit
            setattr(m, name, hit[1])      # replicas hold a shallow copy of __dict__: the original is untouched
    replica.device = dev


def apply(model_XPert=None, device_ids=None):
    """Patch models.model_XPert.XPertNet.forward in place. Returns the device ids it will use."""
    MX = model_XPert or sys.modules['models.model_XPert']
    cls = MX.XPertNet
    ids = list(device_ids) if device_ids is not None else list(range(torch.cuda.device_count()))
    if getattr(cls.forward, '_lincs_dp', False):
        return ids
    orig = cls.forward

    def fwd(self, data, *a, **k):
        if getattr(self, '_is_replica', False):          # set by torch.nn.parallel.replicate
            # NOT next(self.parameters()).device: replicate() turns a replica's parameters into plain non-leaf
            # attributes, so parameters() is EMPTY on a replica (StopIteration -- caught by the local smoke test,
            # 2026-09-24, before any GPU spend). parallel_apply runs each replica under
            # torch.cuda.device(<its device>), so the current device is the replica's.
            _localise(self, torch.device('cuda', torch.cuda.current_device()))
            on = torch.is_autocast_enabled()
            REPLICA_CALLS['n'] += 1
            REPLICA_CALLS['autocast_on'] += int(on)
            if EXPECT_AUTOCAST is not None and on != EXPECT_AUTOCAST:
                raise RuntimeError('xpert_dp_patch: autocast is %s inside a replica, expected %s [RESULTS 81.5]'
                                   % (on, EXPECT_AUTOCAST))
            return orig(self, data, *a, **k)
        if len(ids) > 1 and self.training and torch.is_grad_enabled() and data[0].shape[0] >= len(ids):
            if a:
                raise TypeError('xpert_dp_patch: positional arguments after data are not scattered; pass mode= by name')
            return data_parallel(self, (data,), device_ids=ids, output_device=ids[0], module_kwargs=k or None)
        return orig(self, data, *a, **k)

    fwd._lincs_dp = True
    cls.forward = fwd
    return ids
