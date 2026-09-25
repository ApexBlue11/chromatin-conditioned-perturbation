import os, sys, csv, json, argparse, hashlib
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from data_v9 import LincsV9Dataset, build_splits, collate_v9
from model_v9 import LincsV9
from train_v9_gpu import resolve_v9
from probe_pathways_v6 import spearman, rank

def load_chembl_targets(path):
    ref = defaultdict(set)
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row.get('organism') != 'Homo sapiens': continue
            if row.get('target_type') != 'SINGLE PROTEIN': continue
            if row.get('direct_interaction') != '1': continue
            sym = row.get('gene_symbol')
            if sym:
                ref[row['pert_id']].add(sym)
    return ref

def parse_gmt_full_sets(reactome_path, go_path):
    sets = {}
    with open(reactome_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                sets[parts[1]] = set(parts[2:])
    with open(go_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                name_id = parts[0]
                if '(' in name_id and name_id.endswith(')'):
                    tid = name_id[name_id.rfind('(')+1:-1]
                    if len(parts) >= 3:
                        sets[tid] = set(parts[2:])
    return sets

def importance(model, batch, obj='sq'):
    out, aux = model(batch, return_aux=True)
    a = aux['pathway_activations'] # [B, P, d_pathway]
    if obj == 'sq':
        o = (out['delta'] ** 2).sum()
    else:
        o = out['delta'].sum()
    g, = torch.autograd.grad(o, a)
    return (a.detach() * g).sum(-1) # [B, P]

def project_diff(diff, M_norm):
    return np.abs(diff @ M_norm.T)

def score(scores, allpos, sizes, rng_seed, n_perm, n_size):
    rng = np.random.default_rng(rng_seed)
    D = len(scores)
    P = scores.shape[1]
    if D == 0:
        return None
        
    pct = np.empty_like(scores)
    for i in range(D):
        order = np.argsort(-scores[i])
        pct[i, order] = np.arange(P) / P
        
    obs = [float(np.median(pct[i][allpos[i]])) for i in range(D)]
    S = float(np.median(obs))
    
    perm = np.empty(n_perm)
    for t in range(n_perm):
        sh = rng.permutation(D)
        perm[t] = np.median([np.median(pct[i][allpos[sh[i]]]) for i in range(D)])
        
    null1_mean = float(perm.mean())
    null1_sd = float(perm.std())
    diff = S - null1_mean
    p = float((perm <= S).mean())
    
    order_by_size = np.argsort(sizes)
    rank_in_size = np.empty(P, int)
    rank_in_size[order_by_size] = np.arange(P)
    size_null = np.empty(min(n_size, 200))
    for t in range(len(size_null)):
        vals = []
        for i, pos in enumerate(allpos):
            val_medians = []
            for p_idx in pos:
                k = rank_in_size[p_idx]
                lo = np.clip(k - 15, 0, P - 1)
                hi = np.clip(k + 15, 1, P)
                picked = order_by_size[rng.integers(lo, hi)]
                val_medians.append(pct[i][picked])
            vals.append(np.median(val_medians))
        size_null[t] = np.median(vals)
        
    null2_mean = float(size_null.mean())
    
    return {
        'S': S,
        'null1_mean': null1_mean,
        'null1_sd': null1_sd,
        'diff': diff,
        'p': p,
        'null2_mean': null2_mean,
        'n': D
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--untrained', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_rows_per_drug', type=int, default=4)
    ap.add_argument('--n_perm', type=int, default=1000)
    ap.add_argument('--n_size', type=int, default=200)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    if a.out is None:
        if a.untrained:
            stem = f"untrained_s{a.seed}"
        else:
            if not a.ckpt:
                raise SystemExit('give --ckpt or --untrained')
            stem = os.path.splitext(os.path.basename(a.ckpt))[0]
        a.out = (f"/kaggle/working/probe_moa_v9_{stem}.json" if os.path.isdir("/kaggle/working")
                 else f"model/results/probe_moa_v9_{stem}.json")

    ck_name = f"untrained seed {a.seed}" if a.untrained else os.path.basename(a.ckpt)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if a.untrained:
        ck = {'cfg': vars(V9Config()), 'epoch': -1}
    else:
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = V9Config(**{k: v for k, v in ck['cfg'].items() if k in V9Config.__dataclass_fields__})
    dc = resolve_v9(V9DataConfig())
    dc.cache_in_ram = False
    def R(p):
        """Repo-relative path; on Kaggle (inputs mounted flat) fall back to the file's basename under /kaggle/input."""
        q = p if os.path.isabs(p) else os.path.join(dc.root, p)
        if os.path.exists(q) or not os.path.isdir('/kaggle/input'):
            return q
        import glob
        hit = glob.glob('/kaggle/input/**/' + os.path.basename(p), recursive=True)
        if not hit:
            raise SystemExit('FATAL: %s not found locally or under /kaggle/input' % p)
        return hit[0]

    M = np.load(R(dc.m_pathway_path))
    ppi = np.load(R(dc.ppi_v9_path))
    gv = np.load(R(dc.gene_vec_path))
    info = list(csv.DictReader(open(R(dc.pathway_info_v9_path), encoding='utf-8'), delimiter='\t'))
    P = len(info)

    torch.manual_seed(a.seed)
    model = LincsV9(cfg, M, ppi, gv)
    if not a.untrained:
        model.load_state_dict(ck['model'])
    model = model.to(dev).eval()

    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)

    if a.untrained and cfg.expr_encoder == 'binned':
        tr = sp['train'][ds.has_l3[sp['train']]]
        rws = np.sort(ds.ds_to_l3[tr])
        model.fit_bins(np.asarray(ds.Xctl[rws[np.linspace(0, len(rws) - 1, 20000).astype(int)]], np.float32))
        model = model.to(dev)

    # 1. Rows Selection
    valid_splits = ['test_coldcell', 'test_colddrug', 'test_coldboth']
    idx = np.sort(np.concatenate([sp[k] for k in valid_splits]))
    idx = idx[(ds.strength[idx] >= dc.eval_min_strength) & ds.has_l3[idx]]

    dindex = json.load(open(R(dc.drug_index_path)))
    row_to_pert = {r: pid for pid, r in dindex.items()}
    drug_sigs = defaultdict(list)
    for j in idx:
        pid = row_to_pert.get(int(ds.drug_row[j]))
        if pid:
            drug_sigs[pid].append(int(j))

    selected_rows = []
    compounds = []
    for pid in sorted(drug_sigs.keys()):
        rows = sorted(drug_sigs[pid])[:a.max_rows_per_drug]
        selected_rows.extend(rows)
        compounds.append(pid)
    selected_rows = sorted(selected_rows)

    row_sha1 = hashlib.sha1(np.array(selected_rows, dtype=np.int64).tobytes()).hexdigest()

    # 2. Positive Sets
    chembl_targets = load_chembl_targets(R('drug/outputs/dti/chembl_dti_edges.tsv'))
    gmt_sets = parse_gmt_full_sets(R('network/data/ReactomePathways.gmt'), R('network/data/GO_Biological_Process_2023.gmt'))
    
    matched_nodes = 0
    unmatched_nodes = []
    node_full_sets = []
    for i, row in enumerate(info):
        tid = row['term_id']
        if tid in gmt_sets:
            matched_nodes += 1
            node_full_sets.append(gmt_sets[tid])
        else:
            unmatched_nodes.append(tid)
            node_full_sets.append(set())
            
    positives = {}
    for pid in compounds:
        tgs = chembl_targets.get(pid, set())
        pos = set()
        for i, fset in enumerate(node_full_sets):
            if tgs.intersection(fset):
                pos.add(i)
        positives[pid] = pos

    # Mean-drug baseline
    scored_compounds = [pid for pid, pos in positives.items() if 0 < len(pos) < P]
    if len(scored_compounds) < 10:
        print("Too few compounds"); return
        
    pos_sizes = [len(positives[pid]) for pid in scored_compounds]

    u_list, atom_list = [], []
    for pid in scored_compounds:
        d = dindex[pid]
        u_list.append(ds.u_feats[d])
        a0, a1 = int(ds.atom_off[d]), int(ds.atom_off[d+1])
        atom_list.append(ds.atom_reprs[a0:a1])
        
    u_mean = torch.from_numpy(np.array(u_list).mean(0)).to(dev)
    k_med = int(np.median([len(a) for a in atom_list]))
    atom_mean = torch.zeros(cfg.d_atom)
    atom_count = 0
    for a_rep in atom_list:
        at = torch.from_numpy(np.array(a_rep, np.float32))
        atom_mean += at.sum(0)
        atom_count += at.shape[0]
    atom_mean = (atom_mean / max(atom_count, 1)).to(dev)

    # Computations
    raw_imp, delta_imp, yhats, yhats0, deltas = [], [], [], [], []
    with torch.enable_grad():
        for i, pid in enumerate(scored_compounds):
            rows = sorted(drug_sigs[pid])[:a.max_rows_per_drug]
            b = collate_v9([ds[j] for j in rows])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            b['x_cell'] = b.get('x_cell', b['x_ctl'])
            
            # Normal inference
            b['atoms'].requires_grad_(True)
            imp = importance(model, b, obj='sq')
            raw_imp.append(np.median(imp.cpu().numpy(), axis=0))
            
            with torch.no_grad():
                out = model(b)
                yhats.append(out['delta'].cpu().numpy())
                deltas.append(b['y_delta'].cpu().numpy())

            # Mean inference
            b0 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            b0['u_feats'] = u_mean.unsqueeze(0).expand_as(b0['u_feats']).clone()
            
            # Atom tokens = k_med copies of atom_mean
            atoms = atom_mean.unsqueeze(0).unsqueeze(0).expand(len(rows), k_med, -1).clone()
            atom_mask = torch.ones(len(rows), k_med, dtype=torch.bool, device=dev)
            b0['atoms'] = atoms
            b0['atom_mask'] = atom_mask
            
            imp0 = importance(model, b0, obj='sq')
            delta_imp.append(np.median((imp - imp0).cpu().numpy(), axis=0))
            
            with torch.no_grad():
                out0 = model(b0)
                yhats0.append(out0['delta'].cpu().numpy())
                
    raw_imp = np.stack(raw_imp) # [D, P]
    delta_imp = np.stack(delta_imp) # [D, P]
    delta_imp_abs = np.abs(delta_imp)
    
    scale = float(np.abs(raw_imp).max())
    degen = scale < 1e-12

    # Gate
    rng = np.random.default_rng(a.seed)
    pairs = [(i, j) for i in range(len(scored_compounds)) for j in rng.choice(len(scored_compounds), 3) if i != j][:20000]
    nanmed = lambda v: float(np.median([x for x in v if np.isfinite(x)])) if any(np.isfinite(v)) else float("nan")
    rho_raw = nanmed([spearman(raw_imp[i], raw_imp[j]) for i, j in pairs])
    rho_del = nanmed([spearman(delta_imp_abs[i], delta_imp_abs[j]) for i, j in pairs])
    
    allpos = [list(positives[pid]) for pid in scored_compounds]
    sizes = np.array([len(s) for s in node_full_sets])

    # Precompute cell means in scored set
    cell_deltas = defaultdict(list)
    for i, pid in enumerate(scored_compounds):
        rows = sorted(drug_sigs[pid])[:a.max_rows_per_drug]
        for idx_in_batch, j in enumerate(rows):
            cell_deltas[int(ds.cell_row[j])].append(deltas[i][idx_in_batch])
    cell_means = {c: np.mean(v, axis=0) for c, v in cell_deltas.items()}
    
    M_norm = model.pathway.M_norm.cpu().numpy()
    
    out_proj = np.empty_like(delta_imp)
    data_proj = np.empty_like(delta_imp)
    for i, pid in enumerate(scored_compounds):
        rows = sorted(drug_sigs[pid])[:a.max_rows_per_drug]
        c_means = np.stack([cell_means[int(ds.cell_row[j])] for j in rows])
        out_proj[i] = np.median(project_diff(yhats[i] - yhats0[i], M_norm), axis=0)
        data_proj[i] = np.median(project_diff(deltas[i] - c_means, M_norm), axis=0)
        
    res_grad = score(delta_imp_abs, allpos, sizes, a.seed, a.n_perm, a.n_size)
    res_out = score(out_proj, allpos, sizes, a.seed, a.n_perm, a.n_size)
    res_data = score(data_proj, allpos, sizes, a.seed, a.n_perm, a.n_size)
    
    # Strata
    # Seen vs unseen
    tr_drugs = set(ds.drug_row[sp['train']])
    seen_mask = np.array([dindex[pid] in tr_drugs for pid in scored_compounds])
    
    res_seen = score(delta_imp_abs[seen_mask], [allpos[i] for i in range(len(allpos)) if seen_mask[i]], sizes, a.seed, a.n_perm, a.n_size) if seen_mask.sum() > 0 else None
    res_unseen = score(delta_imp_abs[~seen_mask], [allpos[i] for i in range(len(allpos)) if not seen_mask[i]], sizes, a.seed, a.n_perm, a.n_size) if (~seen_mask).sum() > 0 else None
    
    # Responsive targets
    landmark_symbols = []
    with open(R('network/outputs/v9/landmark_symbols_v9.tsv'), encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            landmark_symbols.append(row['hgnc_symbol'])

    resp_mask = []
    for i, pid in enumerate(scored_compounds):
        mean_y_delta = np.mean(np.abs(deltas[i]), axis=0)
        top50 = set(np.argsort(-mean_y_delta)[:50])
        tgs = chembl_targets.get(pid, set())
        target_landmark_indices = [idx for idx, sym in enumerate(landmark_symbols) if sym in tgs]
        is_resp = any(idx in top50 for idx in target_landmark_indices)
        resp_mask.append(is_resp)
        
    resp_mask = np.array(resp_mask)
    res_resp = score(delta_imp_abs[resp_mask], [allpos[i] for i in range(len(allpos)) if resp_mask[i]], sizes, a.seed, a.n_perm, a.n_size) if resp_mask.sum() > 0 else None
    res_not_resp = score(delta_imp_abs[~resp_mask], [allpos[i] for i in range(len(allpos)) if not resp_mask[i]], sizes, a.seed, a.n_perm, a.n_size) if (~resp_mask).sum() > 0 else None
            
    # Landmark-only tier
    pos_landmark = []
    for pid in scored_compounds:
        tgs = chembl_targets.get(pid, set())
        target_landmark_indices = [idx for idx, sym in enumerate(landmark_symbols) if sym in tgs]
        pos = []
        for p_idx in range(P):
            if any(M[p_idx, tgt_idx] != 0 for tgt_idx in target_landmark_indices):
                pos.append(p_idx)
        pos_landmark.append(pos)
        
    landmark_compounds = []
    landmark_scores = []
    landmark_allpos = []
    for i, pid in enumerate(scored_compounds):
        if 0 < len(pos_landmark[i]) < P:
            landmark_compounds.append(pid)
            landmark_scores.append(delta_imp_abs[i])
            landmark_allpos.append(pos_landmark[i])
            
    if landmark_scores:
        landmark_scores = np.stack(landmark_scores)
        res_landmark = score(landmark_scores, landmark_allpos, sizes, a.seed, a.n_perm, a.n_size)
    else:
        res_landmark = None
            
    out_dict = {
        'checkpoint': ck_name,
        'n_compounds': len(scored_compounds),
        'n_stratum': {
            'seen': int(seen_mask.sum()), 
            'unseen': int((~seen_mask).sum()), 
            'target_responsive': int(resp_mask.sum()), 
            'target_not_responsive': int((~resp_mask).sum())
        },
        'n_gate_pairs': len(pairs),
        'k_med': k_med,
        'pos_sizes_summary': [int(np.min(pos_sizes)), float(np.median(pos_sizes)), int(np.max(pos_sizes))] if pos_sizes else [],
        'pos_sizes': pos_sizes,
        'row_sha1': row_sha1,
        'matched_nodes': matched_nodes,
        'unmatched_nodes': unmatched_nodes,
        'degeneracy': degen,
        'rho_raw': rho_raw,
        'rho_del': rho_del,
        'scores': {
            'gradient_readout': res_grad,
            'output_projection': res_out,
            'data_projection': res_data,
            'stratum_seen': res_seen,
            'stratum_unseen': res_unseen,
            'stratum_target_responsive': res_resp,
            'stratum_target_not_responsive': res_not_resp,
            'landmark_tier': res_landmark
        }
    }
    
    with open(a.out, 'w') as f:
        json.dump(out_dict, f, indent=2)

if __name__ == '__main__':
    main()

