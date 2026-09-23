# -*- coding: utf-8 -*-
"""
A CPU-runnable, numerically faithful stand-in for `flash_attn.flash_attn_interface.flash_attn_func`.

Why this is needed and why it is legitimate.

XPert's `models/model_utils.py` imports flash_attn at module scope, and -- this is the part that matters --
its attention branches the wrong way round from what the name suggests:

    if output_attention:      # dense path: builds the full score matrix, and APPLIES attention_mask
        ...
    else:                     # DEFAULT path: flash_attn_func(q, k, v, dropout_p), NO mask argument
        ...

So the default forward, the one the released checkpoint was trained and evaluated under, is the flash
path. Running on CPU by flipping `output_attention=True` would NOT be equivalent: the dense branch adds
`attention_mask` to the scores, and the drug self-attention is called WITH a padding mask, so the two
branches genuinely differ for the drug branch. Reproducing their numbers therefore requires reproducing
the flash path's semantics -- including the fact that it ignores the mask -- not the dense one.

FlashAttention is an EXACT algorithm: it tiles the softmax to avoid materialising the score matrix, and
returns the same tensor standard attention would, up to floating-point associativity. So computing the
same function densely is faithful, not an approximation. This shim does exactly that via PyTorch's
scaled_dot_product_attention, matching flash_attn_func's signature and (batch, seqlen, nheads, headdim)
layout, its default scale of 1/sqrt(headdim), and its non-causal, unmasked default.

`model/v9/test_xpert_compare.py` checks this against a hand-written reference on random tensors.

DELIBERATELY MORE PERMISSIVE THAN THE REAL KERNEL [review 009 C4]. The real flash_attn_func accepts only
fp16/bf16; this shim accepts fp32, because the CPU inference harness depends on that. The consequence is
that the shim does NOT reproduce flash_attn's dtype rejection -- and that rejection is what would have
caught a training command missing --use_gradscaler True (their train_xpert.py:84-85 enters autocast only
when a GradScaler exists, so without the flag the model runs in fp32). Under the real kernel that command
crashes on the first attention call; under this shim it trains to completion at the wrong precision.
So any training run through this shim must check its executed arguments independently -- the XPert
cold-cell kernel does, as GUARD E. Asserting fp16 here instead would break the CPU harness.

This module is placed on sys.path ONLY when the real flash_attn is absent (see xpert_native_eval.py), so
on a CUDA machine with flash_attn installed the genuine kernel is used and this file is inert.
"""
import math

import torch
import torch.nn.functional as F

__all__ = ['flash_attn_func', 'flash_attn_qkvpacked_func']

# ---------------------------------------------------------------------------------------------------------
# OPTIONAL KEY-PADDING MASK, for one diagnostic only [RESULTS 70, review 007 C2/C4]. Default None, and while
# it is None this shim is exactly the unmasked flash_attn_func it has always been -- which is what XPert's
# executed path uses and what reproduced 0.6933. It exists to run the experiment review 007 pre-committed:
# score the RELEASED checkpoint once as coded (unmasked) and once with the drug padding correctly masked.
#
# It is set per batch by xpert_native_eval.py --mask_drug_keys, as a bool tensor (batch, seqlen_k) with
# True = padded key. It is applied ONLY when its shape equals (k.batch, k.seqlen), which selects exactly the
# drug-keyed calls -- drug self-attention and gene->drug cross-attention, both seqlen_k = 124
# ([dose, time, HG, atom_1..atom_121]) -- and leaves the 978-key gene attention untouched. KEY_PAD_APPLIED
# counts applications, so a run in which the shape never matched cannot pass itself off as a masked run.
# ---------------------------------------------------------------------------------------------------------
KEY_PAD_MASK = None
KEY_PAD_APPLIED = 0


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                    return_attn_probs=False):
    """q, k, v: (batch, seqlen, nheads, headdim). Returns (batch, seqlen, nheads, headdim).

    Mirrors flash_attn_func's defaults: scale = 1/sqrt(headdim), no mask, non-causal. Anything this shim
    does not implement raises instead of silently returning a different quantity."""
    if alibi_slopes is not None:
        raise NotImplementedError('flash_attn shim: alibi_slopes not supported')
    if window_size != (-1, -1):
        raise NotImplementedError('flash_attn shim: sliding-window attention not supported')
    if return_attn_probs:
        raise NotImplementedError('flash_attn shim: return_attn_probs not supported')
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError('flash_attn shim expects (batch, seqlen, nheads, headdim) tensors, got %s'
                         % (tuple(q.shape),))

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    # (B, S, H, D) -> (B, H, S, D), the layout SDPA wants
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    global KEY_PAD_APPLIED
    attn_mask = None
    if KEY_PAD_MASK is not None and tuple(KEY_PAD_MASK.shape) == (k.shape[0], k.shape[1]):
        attn_mask = (~KEY_PAD_MASK.to(device=k.device, dtype=torch.bool))[:, None, None, :]
        KEY_PAD_APPLIED += 1
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=causal, scale=scale)
    return out.transpose(1, 2).contiguous()


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, **kw)
