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
from probe_pathways_v6 import spearman

from probe_moa_v9 import load_chembl_targets, parse_gmt_full_sets, project_diff, score


def score_within_strata(scores, allpos, strata, rng_seed, n_perm):
    rng = np.random.default_rng(rng_seed)
    D = len(scores)
    P = scores.shape[1]
    if D == 0:
        return None
        
    pct = np.empty_like(scores)
    for i in range(D):
        order = np.argsort(-scores[i])
        pct[i, order] = np.arange(P) / P
        
    obs = [float(np.median(pct[i][allpos[i]])) for i in range(D) if len(allpos[i])]
    S = float(np.median(obs)) if obs else 0.0
    
    stratum_indices = defaultdict(list)
    for i, st in enumerate(strata):
        stratum_indices[st].append(i)
        
    perm = np.empty(n_perm)
    for t in range(n_perm):
        shuffled_allpos = [None] * D
        for st, idxs in stratum_indices.items():
            shuffled_idxs = rng.permutation(idxs)
            for original_i, shuffled_i in zip(idxs, shuffled_idxs):
                shuffled_allpos[original_i] = allpos[shuffled_i]
        
        vals = [np.median(pct[i][shuffled_allpos[i]]) for i in range(D) if len(shuffled_allpos[i])]
        perm[t] = np.median(vals) if vals else 0.0
        
    null1_mean = float(perm.mean())
    null1_sd = float(perm.std())
    diff_s = S - null1_mean
    p_s = float((perm <= S).mean())
    
    return {
        'S': S,
        'null1s_mean': null1_mean,
        'null1s_sd': null1_sd,
        'diff_s': diff_s,
        'p_s': p_s
    }

def strength_quintiles(deltas):
    """RESULTS 88.6 item 4: per compound, median over its rows of the row mean over genes of the MEASURED |y_delta|;
    quintile edges by np.quantile over the D compounds; returns (strength, stratum 0..4, sha1 of the assignment)."""
    strength = np.array([np.median(np.mean(np.abs(d), axis=1)) for d in deltas])
    edges = np.quantile(strength, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(strength, edges)
    return strength, strata, hashlib.sha1(np.asarray(strata, np.int64).tobytes()).hexdigest()


def responsiveness_groups(deltas, target_idx):
    """RESULTS 88.6 item 5: per compound, genes ranked by mean |y_delta| over its rows (percentile, 0 = most
    responsive); value = mean percentile of its landmark targets; no landmark target -> its own group; the rest split
    at their median (<= median = responsive). Returns (value, no_landmark, responsive, not_responsive)."""
    vals, no_lm = [], []
    for d, tix in zip(deltas, target_idx):
        m = np.mean(np.abs(d), axis=0)
        pct = np.argsort(np.argsort(-m)) / float(len(m))
        vals.append(float(np.mean(pct[tix])) if len(tix) else np.nan)
        no_lm.append(len(tix) == 0)
    vals, no_lm = np.array(vals), np.array(no_lm)
    med = np.nanmedian(vals) if (~no_lm).any() else np.nan
    resp = (~no_lm) & (vals <= med)
    return vals, no_lm, resp, (~no_lm) & ~resp


def check_arch_match(model_sd, ref_sd):
    """RESULTS 88.6 item 2: the untrained model's state_dict keys and shapes must equal the reference checkpoint's."""
    ka, kb = set(model_sd), set(ref_sd)
    if ka != kb:
        raise SystemExit('FATAL: untrained state_dict keys differ from --cfg_from: only-model %s, only-ckpt %s'
                         % (sorted(ka - kb)[:5], sorted(kb - ka)[:5]))
    bad = [k for k in ka if tuple(model_sd[k].shape) != tuple(ref_sd[k].shape)]
    if bad:
        raise SystemExit('FATAL: untrained shapes differ from --cfg_from at %s' % sorted(bad)[:5])


def post_delta(model, b, b0):
    """RESULTS 88.1 item 2: per row and node, ||a_post(d) - a_post(mean drug)||_2 over the d_pathway channels. No
    gradients; the model must be in eval mode. Returns ([B, P] norms, out, out0)."""
    out, aux = model(b, return_aux=True)
    out0, aux0 = model(b0, return_aux=True)
    return torch.norm(aux['post_pathway_activations'] - aux0['post_pathway_activations'], p=2, dim=2), out, out0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--untrained', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cfg_from', default=None)
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
        a.out = (f"/kaggle/working/probe_moa_88_{stem}.json" if os.path.isdir("/kaggle/working")
                 else f"model/results/probe_moa_88_{stem}.json")

    ck_name = f"untrained seed {a.seed}" if a.untrained else os.path.basename(a.ckpt)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sha1_of = lambda f: hashlib.sha1(open(f, 'rb').read()).hexdigest() if f and os.path.exists(f) else None
    prov = ({'init_seed': a.seed, 'cfg_from_sha1': sha1_of(a.cfg_from)} if a.untrained
            else {'ckpt_sha1': sha1_of(a.ckpt)})         # review 028 C3: the chain reaches the JSON

    if a.untrained:
        if not a.cfg_from:
            raise SystemExit('give --cfg_from with --untrained')
        ck = torch.load(a.cfg_from, map_location='cpu', weights_only=False)
        if (ck.get('tcfg') or {}).get('fold') != 0:
            raise SystemExit('FATAL: --cfg_from must be a fold-0 C8b checkpoint (tcfg.fold == 0), RESULTS 88.6 item 2')
    else:
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        if ck.get('epoch') != 11:
            raise SystemExit('FATAL: trained checkpoint last epoch %r != 11 (budget-cut run), RESULTS 88.6 item 1'
                             % ck.get('epoch'))
    
    cfg = V9Config(**{k: v for k, v in ck['cfg'].items() if k in V9Config.__dataclass_fields__})
    
    if not cfg.post_pathway:
        raise SystemExit('FATAL: cfg lacks post_pathway=True')

    dc = resolve_v9(V9DataConfig())
    dc.cache_in_ram = False
    def R(p):
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
    if a.untrained:
        check_arch_match(model.state_dict(), ck['model'])
    else:
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
    assert row_sha1 == '3e59a7ba7832775596f032dbab06c2c36a7723b4', f"row_sha1 {row_sha1} != 3e59a7ba7832775596f032dbab06c2c36a7723b4"

    chembl_targets = load_chembl_targets(R('drug/outputs/dti/chembl_dti_edges.tsv'))
    gmt_sets = parse_gmt_full_sets(R('network/data/ReactomePathways.gmt'), R('network/data/GO_Biological_Process_2023.gmt'))
    
    node_full_sets = []
    for row in info:
        tid = row['term_id']
        node_full_sets.append(gmt_sets.get(tid, set()))
            
    positives = {}
    for pid in compounds:
        tgs = chembl_targets.get(pid, set())
        pos = set()
        for i, fset in enumerate(node_full_sets):
            if tgs.intersection(fset):
                pos.add(i)
        positives[pid] = pos

    scored_compounds = [pid for pid, pos in positives.items() if 0 < len(pos) < P]
    if len(scored_compounds) < 10:
        raise SystemExit("Too few compounds")
        
    u_list, atom_list = [], []
    for pid in scored_compounds:
        d = dindex[pid]
        u_list.append(ds.u_feats[d])
        a0, a1 = int(ds.atom_off[d]), int(ds.atom_off[d+1])
        atom_list.append(ds.atom_reprs[a0:a1])
        
    u_mean = torch.from_numpy(np.array(u_list).mean(0)).to(dev)
    k_raw = float(np.median([len(a) for a in atom_list]))
    k_med = int(round(k_raw))                      # RESULTS 88.2: int(round(median atom count)) (review 028 C2)
    atom_mean = torch.zeros(cfg.d_atom)
    atom_count = 0
    for a_rep in atom_list:
        at = torch.from_numpy(np.array(a_rep, np.float32))
        atom_mean += at.sum(0)
        atom_count += at.shape[0]
    atom_mean = (atom_mean / max(atom_count, 1)).to(dev)

    delta_a, yhats, yhats0, deltas = [], [], [], []
    with torch.no_grad():
        for i, pid in enumerate(scored_compounds):
            rows = sorted(drug_sigs[pid])[:a.max_rows_per_drug]
            b = collate_v9([ds[j] for j in rows])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            b['x_cell'] = b.get('x_cell', b['x_ctl'])
            
            b0 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            b0['u_feats'] = u_mean.unsqueeze(0).expand_as(b0['u_feats']).clone()
            
            atoms = atom_mean.unsqueeze(0).unsqueeze(0).expand(len(rows), k_med, -1).clone()
            atom_mask = torch.ones(len(rows), k_med, dtype=torch.bool, device=dev)
            b0['atoms'] = atoms
            b0['atom_mask'] = atom_mask
            
            diff, out, out0 = post_delta(model, b, b0)
            yhats.append(out['delta'].cpu().numpy())
            yhats0.append(out0['delta'].cpu().numpy())
            deltas.append(b['y_delta'].cpu().numpy())
            delta_a.append(np.median(diff.cpu().numpy(), axis=0))

    delta_a = np.stack(delta_a)
    max_da = float(np.max(delta_a))
    if max_da <= 1e-12:
        out_dict = {'void': True, 'reason': 'degeneracy', 'checkpoint': ck_name, 'row_sha1': row_sha1, 'max_da': max_da,
                    'epoch': None if a.untrained else ck.get('epoch'), 'untrained': bool(a.untrained),
                    'cfg_from': os.path.basename(a.cfg_from) if a.cfg_from else None, **prov}
        with open(a.out, 'w') as f:
            json.dump(out_dict, f, indent=2)
        return

    rng = np.random.default_rng(a.seed)
    pairs = [(i, j) for i in range(len(scored_compounds)) for j in rng.choice(len(scored_compounds), 3) if i != j][:20000]
    nanmed = lambda v: float(np.median([x for x in v if np.isfinite(x)])) if any(np.isfinite(v)) else float("nan")
    rho_post = nanmed([spearman(delta_a[i], delta_a[j]) for i, j in pairs])
    gate_pass = bool(rho_post < 0.95)

    allpos = [list(positives[pid]) for pid in scored_compounds]
    sizes = np.array([len(s) for s in node_full_sets])

    # Strength strata
    comp_strength, strength_strata, quintile_sha1 = strength_quintiles(deltas)

    res_post_score = score(delta_a, allpos, sizes, a.seed, a.n_perm, a.n_size)
    res_post_s1 = score_within_strata(delta_a, allpos, strength_strata, a.seed, a.n_perm)

    cell_deltas = defaultdict(list)
    for i, pid in enumerate(scored_compounds):
        rows = sorted(drug_sigs[pid])[:a.max_rows_per_drug]
        for idx_in_batch, j in enumerate(rows):
            cell_deltas[int(ds.cell_row[j])].append(deltas[i][idx_in_batch])
    cell_means = {c: np.mean(v, axis=0) for c, v in cell_deltas.items()}
    
    M_norm = model.pathway.M_norm.cpu().numpy()
    
    out_proj = np.empty_like(delta_a)
    data_proj = np.empty_like(delta_a)
    for i, pid in enumerate(scored_compounds):
        rows = sorted(drug_sigs[pid])[:a.max_rows_per_drug]
        c_means = np.stack([cell_means[int(ds.cell_row[j])] for j in rows])
        out_proj[i] = np.median(project_diff(yhats[i] - yhats0[i], M_norm), axis=0)
        data_proj[i] = np.median(project_diff(deltas[i] - c_means, M_norm), axis=0)

    res_out_score = score(out_proj, allpos, sizes, a.seed, a.n_perm, a.n_size)
    res_out_s1 = score_within_strata(out_proj, allpos, strength_strata, a.seed, a.n_perm)
    
    res_data_score = score(data_proj, allpos, sizes, a.seed, a.n_perm, a.n_size)
    res_data_s1 = score_within_strata(data_proj, allpos, strength_strata, a.seed, a.n_perm)

    # Strata: Seen vs Unseen
    tr_drugs = set(ds.drug_row[sp['train']])
    seen_mask = np.array([dindex[pid] in tr_drugs for pid in scored_compounds])

    def score_and_s1(mask):
        if mask.sum() > 0:
            s = score(delta_a[mask], [allpos[i] for i in range(len(allpos)) if mask[i]], sizes, a.seed, a.n_perm, a.n_size)
            s1 = score_within_strata(delta_a[mask], [allpos[i] for i in range(len(allpos)) if mask[i]], strength_strata[mask], a.seed, a.n_perm)
            return {'score': s, 'null1s': s1, 'n': int(mask.sum())}
        return None

    res_seen = score_and_s1(seen_mask)
    res_unseen = score_and_s1(~seen_mask)

    # Strata: Target Responsiveness
    landmark_symbols = []
    with open(R('network/outputs/v9/landmark_symbols_v9.tsv'), encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            landmark_symbols.append(row['hgnc_symbol'])

    target_idx = [[k for k, sym in enumerate(landmark_symbols) if sym in chembl_targets.get(pid, set())]
                  for pid in scored_compounds]
    landmark_ranks, no_landmark_mask, resp_mask, not_resp_mask = responsiveness_groups(deltas, target_idx)

    res_no_lm = score_and_s1(no_landmark_mask)
    res_resp = score_and_s1(resp_mask)
    res_not_resp = score_and_s1(not_resp_mask)

    out_dict = {
        'checkpoint': ck_name,
        'row_sha1': row_sha1,
        'k_med': k_med,
        'k_median_raw': k_raw,
        **prov,
        'n_compounds': len(scored_compounds),
        'flags': vars(cfg),
        'void': False,
        'max_da': max_da,
        'epoch': None if a.untrained else ck.get('epoch'),
        'untrained': bool(a.untrained),
        'cfg_from': os.path.basename(a.cfg_from) if a.cfg_from else None,
        'quintile_sha1': quintile_sha1,
        'quintile_counts': np.bincount(strength_strata, minlength=5).tolist(),
        'gate_pass': gate_pass,
        'rho_post': rho_post,
        'primary': {
            'score': res_post_score,
            'null1s': res_post_s1
        },
        'output_projection': {
            'score': res_out_score,
            'null1s': res_out_s1
        },
        'data_projection': {
            'score': res_data_score,
            'null1s': res_data_s1
        },
        'strata': {
            'seen': res_seen,
            'unseen': res_unseen,
            'no_landmark_target': res_no_lm,
            'target_responsive': res_resp,
            'target_not_responsive': res_not_resp
        }
    }

    with open(a.out, 'w') as f:
        json.dump(out_dict, f, indent=2)

if __name__ == '__main__':
    main()
