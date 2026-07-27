# -*- coding: utf-8 -*-
"""
Losses + metrics. Per the reference warnings we report R2/magnitude (not just correlation) and always
compare against Mean / Meancell / Meandrug baselines (esp. cold-cell). DTI never enters the loss.
"""
import numpy as np
import torch
import torch.nn.functional as F


def weighted_huber(yhat, y, w=None, delta=1.0):
    """yhat,y:[B,G]; w:[B] reliability weights (optional). Per-example mean over genes, weighted mean over batch."""
    per = F.huber_loss(yhat, y, delta=delta, reduction="none").mean(dim=1)   # [B]
    if w is None:
        return per.mean()
    return (w * per).sum() / (w.sum() + 1e-8)


def correlation_loss(yhat, y, w=None):
    """1 - Pearson r per signature (across the 978 genes), reliability-weighted mean over the batch.
    MAGNITUDE-INVARIANT: it optimises the signed PATTERN (incl. the drug x cell interaction), so it is
    immune to the MSE shrinkage that suppresses cell-specificity (diagnosed: interaction expressed at only
    ~0.49 magnitude / corr 0.42 on strong sigs). w (reliability) MUST down-weight inert sigs so we do not
    fit the pattern of NOISE on non-responding perturbations. Pair with Huber (keeps magnitude calibrated)."""
    yh = yhat - yhat.mean(1, keepdim=True)
    yy = y - y.mean(1, keepdim=True)
    num = (yh * yy).sum(1)
    den = torch.sqrt((yh ** 2).sum(1) * (yy ** 2).sum(1) + 1e-8)
    loss = 1.0 - num / den                         # [B]; 0 = perfect pattern match
    if w is None:
        return loss.mean()
    return (w * loss).sum() / (w.sum() + 1e-8)


def r2_per_gene(yhat, y):
    """1 - SS_res/SS_tot per gene (can be negative). yhat,y:[N,G] -> [G]."""
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdim=True)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def r2_overall(yhat, y):
    """Single R2 over all elements (variance-weighted across genes)."""
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return (1.0 - ss_res / (ss_tot + 1e-8)).item()


def pearson_per_row(yhat, y):
    """Per-signature Pearson r across the 978 genes. yhat,y:[N,G] -> [N]."""
    yh = yhat - yhat.mean(1, keepdim=True)
    yy = y - y.mean(1, keepdim=True)
    num = (yh * yy).sum(1)
    den = torch.sqrt((yh ** 2).sum(1) * (yy ** 2).sum(1) + 1e-8)
    return num / den


def naive_baselines(Y_train, cell_train, drug_train, Y_eval, cell_eval, drug_eval):
    """MSE of the three trivial predictors on the eval set (numpy). Cell/drug are integer id arrays.
    Meancell/Meandrug fall back to the global mean for ids unseen in train (cold split)."""
    gmean = Y_train.mean(0)                                            # [G]

    def group_means(Y, ids):
        out = {}
        for u in np.unique(ids):
            out[u] = Y[ids == u].mean(0)
        return out

    cmean = group_means(Y_train, cell_train)
    dmean = group_means(Y_train, drug_train)

    def mse(pred):
        return float(((Y_eval - pred) ** 2).mean())

    pred_mean = np.broadcast_to(gmean, Y_eval.shape)
    pred_cell = np.stack([cmean.get(c, gmean) for c in cell_eval])
    pred_drug = np.stack([dmean.get(d, gmean) for d in drug_eval])
    return {"Mean": mse(pred_mean), "Meancell": mse(pred_cell), "Meandrug": mse(pred_drug)}
