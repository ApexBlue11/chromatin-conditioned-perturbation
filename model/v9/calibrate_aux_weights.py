# -*- coding: utf-8 -*-
"""
§85.7 "Auxiliary weights by rule": compute w3, w6 as 0.1 * ||grad L_delta|| / ||grad L_aux||.

Builds the baseline configuration exactly as xpert_arm.py does (dev mode, --dev_cells 6 --dev_seed 0,
seed 0, fresh init, fp32, CPU, no training step taken), iterates the first N batches of seed 0's training
order on the dev training rows, and for each auxiliary term (C3 ListNet at unit weight; C6 sign BCE at
unit weight) computes the gradient-norm ratio, averages over batches, and writes the result.

    python model/v9/calibrate_aux_weights.py             # full run: 20 batches
    python model/v9/calibrate_aux_weights.py --n_batches 2  # smoke test

Do NOT run the full 20-batch version without the PI's say-so.
"""
import os, sys, json, argparse, copy

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config
from model_v9 import LincsV9, v9_loss, listnet_loss


def grad_norm(model):
    """L2 norm of the full gradient vector."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().pow(2).sum().item()
    return total ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_batches', type=int, default=20)
    ap.add_argument('--batch', type=int, default=48)
    a = ap.parse_args()

    # Import XPertData and helpers from xpert_arm
    from xpert_arm import XPertData, find, BUNDLE

    roots = ['/kaggle/input', os.path.join(r'C:\Projects\LINCS'), os.path.join(r'C:\Projects\LINCS',
                                                                              'external')]
    if os.environ.get('LINCS_DATA_ROOT'):
        roots.insert(0, os.environ['LINCS_DATA_ROOT'])
    roots = [r for r in roots if os.path.isdir(r)]

    # Replicate xpert_arm's dev configuration
    class DevArgs:
        dev_cells = 6
        dev_seed = 0
        dev_min_rows = 200
        dev_max_rows = 2000

    npz = find('xpert_mdmt_splits.npz', roots)
    D = XPertData(npz, roots, 'split_cold_cell_1', dev_args=DevArgs())

    M = np.load(find('M_pathway_v9.npy', roots))
    ppi = np.load(find('STRING_adj_978_v9.npy', roots))
    gv = np.load(find('gene_vectors_978.npy', roots))
    Mn = torch.as_tensor(M, dtype=torch.float32)
    Mn = Mn / Mn.sum(1, keepdim=True).clamp(min=1)

    # Seed 0, fresh init, fp32, CPU
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = V9Config()
    cfg.n_pathways = M.shape[0]
    cfg.predict_l5 = False
    # Baseline config — all candidate flags at defaults
    model = LincsV9(cfg, M, ppi, gv)

    if cfg.expr_encoder == 'binned':
        Xfit = D.C[D.tr]
        if not np.isfinite(Xfit).all():
            raise SystemExit('FATAL: non-finite values in the quantiser fitting sample.')
        model.fit_bins(Xfit)
        if not model.bins_fitted:
            raise SystemExit('FATAL: quantiser did not fit.')

    model.train()
    device = 'cpu'

    # Training order: seed 0
    np.random.seed(0)
    order = np.random.permutation(D.tr)

    # For each batch: compute ||grad L_delta|| and ||grad L_aux|| for ListNet (C3) and sign BCE (C6)
    ratios_listnet = []
    ratios_sign = []

    # Create a config copy with sign_head for gradient computation
    cfg_sign = copy.deepcopy(cfg)
    cfg_sign.sign_head_w = 1.0

    torch.manual_seed(0)
    np.random.seed(0)
    model_sign = LincsV9(cfg_sign, M, ppi, gv)
    if cfg_sign.expr_encoder == 'binned':
        model_sign.fit_bins(Xfit)
    model_sign.train()

    np.random.seed(0)
    order = np.random.permutation(D.tr)

    for it in range(min(a.n_batches, len(order) // a.batch)):
        idx = order[it * a.batch:(it + 1) * a.batch]
        b = D.batch(idx, device)

        # --- Gradient norm for L_delta (the weighted delta Huber term only) ---
        model.zero_grad(set_to_none=True)
        out, aux = model(b, return_aux=True)
        w_abs, w_delta, w_l5, w_pcc = cfg.task_w
        hub = nn.functional.huber_loss
        L_delta = hub(out['delta'], b['y_delta'], delta=cfg.huber_delta)
        (w_delta * L_delta).backward()
        gn_delta = grad_norm(model)

        # --- Gradient norm for ListNet at unit weight ---
        model.zero_grad(set_to_none=True)
        out2, _ = model(b, return_aux=True)
        L_listnet = listnet_loss(out2['delta'], b['y_delta'])
        L_listnet.backward()
        gn_listnet = grad_norm(model)

        if gn_listnet > 0:
            ratios_listnet.append(gn_delta / gn_listnet)

        # --- Gradient norm for sign BCE at unit weight ---
        model_sign.zero_grad(set_to_none=True)
        out_s, aux_s = model_sign(b, return_aux=True)
        y_d = b['y_delta']
        topk = min(50, y_d.shape[1])
        topk_idx = y_d.abs().topk(topk, dim=1).indices
        logits = aux_s['sign_logits'].gather(1, topk_idx)
        target_sign = (y_d.gather(1, topk_idx) > 0).float()
        L_sign = nn.functional.binary_cross_entropy_with_logits(logits, target_sign)
        L_sign.backward()
        gn_sign = grad_norm(model_sign)

        # For the sign ratio, use the same gn_delta from the baseline model
        if gn_sign > 0:
            ratios_sign.append(gn_delta / gn_sign)

        print(f'batch {it}: gn_delta={gn_delta:.4f}  gn_listnet={gn_listnet:.4f}  '
              f'gn_sign={gn_sign:.4f}  '
              f'ratio_listnet={ratios_listnet[-1]:.4f}  ratio_sign={ratios_sign[-1]:.4f}',
              flush=True)

    avg_ratio_listnet = sum(ratios_listnet) / len(ratios_listnet) if ratios_listnet else float('nan')
    avg_ratio_sign = sum(ratios_sign) / len(ratios_sign) if ratios_sign else float('nan')

    w3 = 0.1 * avg_ratio_listnet
    w6 = 0.1 * avg_ratio_sign

    result = {
        'w3_listnet': round(w3, 6),
        'w6_sign': round(w6, 6),
        'avg_ratio_listnet': round(avg_ratio_listnet, 6),
        'avg_ratio_sign': round(avg_ratio_sign, 6),
        'n_batches': len(ratios_listnet),
        'batch_size': a.batch,
        'ratios_listnet': [round(r, 6) for r in ratios_listnet],
        'ratios_sign': [round(r, 6) for r in ratios_sign],
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'model', 'results',
                            'v9_aux_weight_calibration.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, 'w'), indent=2)
    print(f'\nw3 = {w3:.6f}  (0.1 * {avg_ratio_listnet:.4f})')
    print(f'w6 = {w6:.6f}  (0.1 * {avg_ratio_sign:.4f})')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
