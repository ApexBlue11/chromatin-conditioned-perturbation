import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config
from model_v9 import LincsV9
from modules_v7 import StochasticDepth
from xpert_arm import XPertData, find

def set_mc_mode(model, arm):
    model.eval()
    counts = {'Dropout': 0, 'StochasticDepth': 0}
    if arm == 'det':
        pass
    elif arm == 'drop':
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                counts['Dropout'] += 1
    elif arm == 'full':
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                counts['Dropout'] += 1
            elif isinstance(m, StochasticDepth):
                m.train()
                counts['StochasticDepth'] += 1
    return counts

def predict_rows(model, D, idx, dev):
    P, T = [], []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            batch_idx = idx[s:s+64]
            b = D.batch(batch_idx, dev)
            o = model(b)
            P.append(o['abs'].float().cpu().numpy())
            T.append(o['delta'].float().cpu().numpy())
    return np.concatenate(P), np.concatenate(T)

def mc_predict(model, D, idx, dev, arm, K, seed):
    counts = set_mc_mode(model, arm)
    
    if arm == 'det':
        return predict_rows(model, D, idx, dev), counts
    
    P_all, T_all = [], []
    for k in range(K):
        torch.manual_seed(1000 * seed + k)
        torch.cuda.manual_seed_all(1000 * seed + k)
        pa, pd = predict_rows(model, D, idx, dev)
        P_all.append(pa)
        T_all.append(pd)
        
    return (np.mean(P_all, axis=0), np.mean(T_all, axis=0)), counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', choices=['det', 'full', 'drop'], required=True)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bundle', default='xpert_mdmt_splits.npz')
    ap.add_argument('--split', required=True)
    ap.add_argument('--dev_cells', type=int, default=6)
    ap.add_argument('--dev_seed', type=int, default=0)
    ap.add_argument('--identity_check', default=None)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    
    t0 = time.time()
    
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    assert ck['split'] == a.split
    seed = ck['seed']
    
    roots = ['/kaggle/input', os.path.join(r'C:\Projects\LINCS'), os.path.join(r'C:\Projects\LINCS', 'external')]
    npz = find(a.bundle, roots)
    
    dev_args = argparse.Namespace(dev_cells=a.dev_cells, dev_seed=a.dev_seed, dev_min_rows=200, dev_max_rows=2000)
    D = XPertData(npz, roots, a.split, ablate_epi=ck.get('ablate_epi', False), dev_args=dev_args)
    
    dev_idx = D.te
    if a.limit > 0:
        dev_idx = dev_idx[:a.limit]
    
    ri = D.row_index[dev_idx].astype(np.int64)
    if not a.limit:
        sha = hashlib.sha1(np.sort(ri).tobytes()).hexdigest()
        assert sha == '51e7e4ab8b9c3c3709d43da7fa4a8c80b77d5980', f'wrong dev rows: {sha}'
        
    cfg = V9Config()
    for k, v in ck['cfg'].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
            
    M = np.load(find('M_pathway_v9.npy', roots))
    ppi = np.load(find('STRING_adj_978_v9.npy', roots))
    gv = np.load(find('gene_vectors_978.npy', roots))
    
    model = LincsV9(cfg, M, ppi, gv)
    model.load_state_dict(ck['model'], strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    (y_pred, deg_pred), counts = mc_predict(model, D, dev_idx, device, a.arm, a.K, seed)
    
    y_true = D.X[dev_idx]
    ctl_true = D.C[dev_idx]
    
    np.savez_compressed(a.out,
                        y_pred=y_pred.astype(np.float32),
                        deg_pred=deg_pred.astype(np.float32),
                        y_true=y_true.astype(np.float32),
                        ctl_true=ctl_true.astype(np.float32),
                        row_index=ri)
                        
    try:
        with open(a.ckpt, 'rb') as f:
            ckpt_sha1 = hashlib.sha1(f.read()).hexdigest()
    except Exception:
        ckpt_sha1 = 'unknown'
        
    meta = {
        'arm': a.arm,
        'K': a.K,
        'seed': seed,
        'switched_module_counts': counts,
        'ckpt_sha1': ckpt_sha1,
        'n_rows': len(dev_idx),
        'wall_seconds': time.time() - t0
    }
    with open(a.out.replace('.npz', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    if a.identity_check:
        assert a.arm == 'det', "identity_check only valid for det arm"
        z = np.load(a.identity_check)
        saved_deg_pred = z['deg_pred'][:a.limit] if a.limit else z['deg_pred']
        
        # Calculate per-row Pearson
        a_val = deg_pred - deg_pred.mean(1, keepdims=True)
        b_val = saved_deg_pred - saved_deg_pred.mean(1, keepdims=True)
        r = (a_val * b_val).sum(1) / np.sqrt((a_val * a_val).sum(1) * (b_val * b_val).sum(1))
        min_pearson = r.min()
        print(f"Minimum per-row Pearson between recomputed and saved deg_pred: {min_pearson}")
        
        if min_pearson < 0.9999:
            raise SystemExit(f"Identity check failed: min Pearson {min_pearson} < 0.9999")
            
        def row_pearson(a_arr, b_arr):
            a_arr = a_arr - a_arr.mean(1, keepdims=True)
            b_arr = b_arr - b_arr.mean(1, keepdims=True)
            return (a_arr * b_arr).sum(1) / np.sqrt((a_arr * a_arr).sum(1) * (b_arr * b_arr).sum(1))
            
        r_recomp = row_pearson(deg_pred, y_true - ctl_true)
        r_saved = row_pearson(saved_deg_pred, z['y_true'][:a.limit] - z['ctl_true'][:a.limit] if a.limit else z['y_true'] - z['ctl_true'])
        
        mean_recomp = float(np.nanmean(r_recomp))
        mean_saved = float(np.nanmean(r_saved))
        print(f"Dev per-row mean Pearson: recomputed={mean_recomp:.6f}, saved={mean_saved:.6f}")
        
        if abs(mean_recomp - mean_saved) > 5e-5:
            raise SystemExit(f"Identity check failed: mean Pearson delta {abs(mean_recomp - mean_saved)} > 5e-5")

if __name__ == '__main__':
    main()
