# -*- coding: utf-8 -*-
"""
Interpretability utilities that ENFORCE this project's method rules in code, so a future run cannot quietly
break them.

Rule 2 -- ablate to the MEAN, never 0 or 1, and always report |dY|max. Ablating the pathway conductance to
1 once produced a +0.103 "contribution" that was a 30x scale artefact; the true effect was -0.003/+0.006.
`ablate_to_mean` takes the replacement value from the batch itself and returns |dY|max alongside the score
change, so a true null (effect ~0, |dY|max >> 0) stays distinguishable from a branch that never fired
(effect ~0, |dY|max == 0).

Rule 6 -- a readout's chance level is MEASURED, not assumed. On pathway-level target alignment 0.5 is not
chance: an untrained model scores 0.218 against a permutation null of 0.229. `permutation_null` produces
that null for any readout scored by any function.
"""
import numpy as np
import torch


@torch.no_grad()
def ablate_to_mean(model, batch, key, score_fn, n_ref=None):
    """Replace `batch[key]` with its BATCH MEAN (broadcast), not with zeros.

    Returns (delta_score, dY_max) where dY_max = max |prediction change| over every element. A component
    whose delta_score is ~0 AND whose dY_max is ~0 did not fire at all -- that is a broken ablation, not a
    null result, and the two must never be reported as the same thing.
    """
    was = model.training
    model.eval()
    out0 = model(batch)
    y0 = out0['delta'] if isinstance(out0, dict) else out0
    ref = batch[key]
    mean = ref.mean(dim=0, keepdim=True).expand_as(ref) if n_ref is None else n_ref
    patched = dict(batch)
    patched[key] = mean
    out1 = model(patched)
    y1 = out1['delta'] if isinstance(out1, dict) else out1
    model.train(was)
    return float(score_fn(y0) - score_fn(y1)), float((y0 - y1).abs().max())


@torch.no_grad()
def ablate_module_to_mean(model, batch, module, score_fn):
    """Same rule, for a MODULE rather than an input: the module's output is replaced by its own mean over
    the gene axis, so its per-gene structure is removed while its scale is preserved."""
    was = model.training
    model.eval()
    out0 = model(batch)
    y0 = out0['delta'] if isinstance(out0, dict) else out0
    store = {}

    def hook(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        store['shape'] = t.shape
        m = t.mean(dim=1, keepdim=True).expand_as(t)
        return (m,) + o[1:] if isinstance(o, tuple) else m

    h = module.register_forward_hook(hook)
    out1 = model(batch)
    h.remove()
    y1 = out1['delta'] if isinstance(out1, dict) else out1
    model.train(was)
    return float(score_fn(y0) - score_fn(y1)), float((y0 - y1).abs().max())


def permutation_null(scores, labels, score_fn, n_perm=200, seed=0):
    """The MEASURED chance level for a readout.

    `score_fn(scores, labels) -> float` is evaluated once on the real pairing and `n_perm` times on shuffled
    labels. Returns (observed, null_mean, null_sd, p). Report all four: a readout that beats 0.5 but not its
    own permutation null has shown nothing.
    """
    rng = np.random.default_rng(seed)
    obs = float(score_fn(scores, labels))
    null = np.array([float(score_fn(scores, rng.permutation(labels))) for _ in range(n_perm)])
    p = float(((null >= obs).sum() + 1) / (n_perm + 1))
    return obs, float(null.mean()), float(null.std()), p


def pathway_alignment(activations, delta, M):
    """How well the named pathway activations rank the pathways that actually moved.

    activations [B, P], delta [B, G], M [P, G]. The target is mean|delta| over each term's member genes.
    Returned as a mean Spearman rank correlation across rows -- and it must be reported against
    `permutation_null`, never against 0.5.
    """
    A = np.asarray(activations, np.float64)
    if A.ndim == 3:
        A = A.mean(-1)
    Mn = np.asarray(M, np.float64)
    Mn = Mn / np.maximum(Mn.sum(1, keepdims=True), 1)
    T = np.abs(np.asarray(delta, np.float64)) @ Mn.T                     # [B, P]
    rs = []
    for a, t in zip(A, T):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rt = np.argsort(np.argsort(t)).astype(np.float64)
        ra -= ra.mean(); rt -= rt.mean()
        d = np.sqrt((ra ** 2).sum() * (rt ** 2).sum())
        rs.append((ra * rt).sum() / d if d > 0 else 0.0)
    return float(np.mean(rs))
