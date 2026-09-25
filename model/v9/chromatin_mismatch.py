import torch
import numpy as np

def score_mismatched_chromatin(model, D, dev_row_indices, donor_mapping, out_npz, device='cuda'):
    model.eval()
    
    cell_to_E = {}
    cell_to_Em = {}
    for i in dev_row_indices:
        c = D.cell[i]
        if c not in cell_to_E:
            cell_to_E[c] = D.E[i].copy()
            cell_to_Em[c] = D.Em[i].copy()
            
    donor_track_means = {}
    for c, E_val in cell_to_E.items():
        Em_val = cell_to_Em[c]
        means = np.zeros(3, dtype=np.float32)
        for t in range(3):
            obs = Em_val[:, t]
            if obs.any():
                means[t] = E_val[obs, t].mean()
            else:
                means[t] = 0.0
        donor_track_means[c] = means

    own_preds = []
    mismatch_preds = {i: [] for i in range(5)}
    targets = []
    
    with torch.no_grad():
        for i in dev_row_indices:
            c = D.cell[i]
            b = D.batch([i], device)
            
            out = model(b)
            own_p = out['delta'].cpu().numpy()[0]
            own_preds.append(own_p)
            targets.append(b['y_delta'].cpu().numpy()[0])
            
            donors = donor_mapping[c]
            for donor_idx, donor in enumerate(donors):
                E_eval = cell_to_E[c]
                Em_eval = cell_to_Em[c]
                
                E_donor = cell_to_E[donor]
                Em_donor = cell_to_Em[donor]
                donor_mean = donor_track_means[donor]
                
                E_new = E_eval.copy()
                for t in range(3):
                    both_obs = Em_eval[:, t] & Em_donor[:, t]
                    only_eval_obs = Em_eval[:, t] & ~Em_donor[:, t]
                    
                    E_new[both_obs, t] = E_donor[both_obs, t]
                    E_new[only_eval_obs, t] = donor_mean[t]

                b_mismatch = D.batch([i], device)
                b_mismatch['E'] = torch.as_tensor(E_new).unsqueeze(0).to(device)
                
                out_mismatch = model(b_mismatch)
                mismatch_preds[donor_idx].append(out_mismatch['delta'].cpu().numpy()[0])

    own_preds = np.array(own_preds)
    targets = np.array(targets)
    mismatch_arrs = {f'mismatch_preds_{k}': np.array(v) for k, v in mismatch_preds.items()}
    
    np.savez(out_npz, own_preds=own_preds, targets=targets, **mismatch_arrs)
    
    return own_preds, mismatch_preds, targets
