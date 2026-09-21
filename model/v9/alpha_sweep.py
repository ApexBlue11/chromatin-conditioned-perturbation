import argparse
import io
import json
import os
import sys
import numpy as np
import scipy.stats
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
os.chdir(HERE)

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9
from data_v9 import LincsV9Dataset, collate_v9, build_splits
from train_v9_gpu import resolve_v9

def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)

def make_atom_ablated_batch(batch, key='atoms'):
    ref = batch[key]
    patched = dict(batch)
    patched[key] = ref.mean(dim=0, keepdim=True).expand_as(ref)
    return patched

def check_diagonal_available(model):
    cfg = getattr(model, 'cfg', None)
    has_cfg = bool(getattr(cfg, 'drug_self_attn', False)) if cfg is not None else False
    perturb_blocks = getattr(model, 'perturb', [])
    n_drug_sa = sum(1 for blk in perturb_blocks if getattr(blk, 'drug_sa', None) is not None)
    if not has_cfg or n_drug_sa == 0:
        return False, (
            "diagonal mode unavailable for this checkpoint: "
            f"cfg.drug_self_attn={has_cfg}, perturb blocks with drug_sa={n_drug_sa}/{len(perturb_blocks)}"
        )
    return True, ""

def evaluate_chunks_alpha(model, chunks, alpha, key='atoms'):
    r_full_list, r_ablated_list = [], []
    dymax_atom = 0.0

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for c in chunks:
            tgt = c['y_delta'].float().cpu().numpy()
            c_ablated = make_atom_ablated_batch(c, key=key)
            
            c_alpha = dict(c)
            c_alpha['drug_alpha'] = alpha
            c_ablated_alpha = dict(c_ablated)
            c_ablated_alpha['drug_alpha'] = alpha

            out_full = model(c_alpha)
            y_full = out_full['delta'] if isinstance(out_full, dict) else out_full

            out_ablated = model(c_ablated_alpha)
            y_ablated = out_ablated['delta'] if isinstance(out_ablated, dict) else out_ablated

            dy_a = float((y_full - y_ablated).abs().max().item())
            dymax_atom = max(dymax_atom, dy_a)

            r_full_list.append(pearson_rows(y_full.float().cpu().numpy(), tgt))
            r_ablated_list.append(pearson_rows(y_ablated.float().cpu().numpy(), tgt))

    model.train(was_training)

    r_full = np.concatenate(r_full_list) if r_full_list else np.empty(0, dtype=np.float32)
    r_ablated = np.concatenate(r_ablated_list) if r_ablated_list else np.empty(0, dtype=np.float32)

    return r_full, r_ablated, dymax_atom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--alphas', default='0,0.25,0.5,0.75,1.0')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--n_eval', type=int, default=1500)
    ap.add_argument('--n_boot', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--key', default='atoms')
    a = ap.parse_args()

    alphas = [float(x.strip()) for x in a.alphas.split(',')]

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = a.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(os.getcwd(), ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_fn = os.path.basename(ckpt_path)
    ckpt_stem = os.path.splitext(ckpt_fn)[0]
    key_tag = '' if a.key == 'atoms' else f'_key-{a.key}'
    dst_json = os.path.join(ROOT, 'model', 'results', f'v9_alpha_sweep_{ckpt_stem}{key_tag}.json')
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)

    print(f"=== v9 Alpha Sweep [TASK W6] ===", flush=True)
    print(f"Checkpoint: {ckpt_fn} | Device: {dev}", flush=True)
    print(f"Alphas: {alphas}", flush=True)
    
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_raw = ck.get('cfg', {})
    cfg_kwargs = {k: v for k, v in cfg_raw.items() if k in V9Config.__dataclass_fields__ and v is not None}
    cfg = V9Config(**cfg_kwargs)

    dc = resolve_v9(V9DataConfig())
    dc.cache_in_ram = False
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_pathway_path))
    ppi = np.load(R(dc.ppi_v9_path))
    gv = np.load(R(dc.gene_vec_path))

    torch.manual_seed(0)
    model = LincsV9(cfg, M, ppi, gv)
    if 'model' in ck:
        model.load_state_dict(ck['model'])
    model = model.to(dev).eval()

    avail, reason = check_diagonal_available(model)
    if not avail:
        print(f"[REFUSED] {reason}", flush=True)
        out = {
            'ckpt': ckpt_fn,
            'error': reason
        }
        with io.open(dst_json, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(json.dumps(out, indent=2) + '\n')
        sys.exit(0)

    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)

    if not model.bins_fitted and cfg.expr_encoder == 'binned':
        tr = sp['train'][ds.has_l3[sp['train']]]
        rows = np.sort(ds.ds_to_l3[tr])
        sample_rows = rows[np.linspace(0, len(rows) - 1, min(20000, len(rows))).astype(int)]
        model.fit_bins(np.asarray(ds.Xctl[sample_rows], np.float32))
        model = model.to(dev)

    out = {
        'ckpt': ckpt_fn,
        'batch': a.batch,
        'n_eval': a.n_eval,
        'n_boot': a.n_boot,
        'seed': a.seed,
        'alphas': alphas,
        'key': a.key,
        'splits': {}
    }

    # Two generators, deliberately. The delegated version drew BOTH the row sample and every alpha's
    # bootstrap from one generator, with the bootstrap inside the alpha loop. Two consequences, and the
    # second is the serious one:
    #   * each alpha got an independent resample, so the CIs along the curve were not comparable
    #     draw-for-draw -- and the SHAPE of this curve is the entire inference [review 005 C3];
    #   * the row sample for splits 2 and 3 depended on HOW MANY ALPHAS ran for split 1, because the
    #     bootstrap consumed generator state. `--alphas 0,1` and `--alphas 0,0.25,0.5,0.75,1` would have
    #     scored DIFFERENT ROWS. A result must not depend on an argument that does not name it -- the same
    #     defect class as an output filename that ignores --key.
    # `row_rng` is advanced only by row selection, in the same order as interaction_2x2.py, so the rows
    # here are identical to the 2x2's and the two sets of numbers are directly comparable.
    row_rng = np.random.default_rng(a.seed)
    
    for name, key in [('unseen_cell', 'test_coldcell'),
                      ('unseen_compound', 'test_colddrug'),
                      ('unseen_both', 'test_coldboth')]:
        idx = sp[key][ds.strength[sp[key]] >= dc.eval_min_strength]
        idx = idx[ds.has_l3[idx]]
        n_eligible = int(len(idx))
        if n_eligible == 0:
            continue

        n_eval_bound = bool(a.n_eval < n_eligible)
        n_sample = min(a.n_eval, n_eligible)
        idx_eval = np.sort(row_rng.choice(idx, n_sample, replace=False))

        chunks = [collate_v9([ds[i] for i in idx_eval[s:s + a.batch]])
                  for s in range(0, len(idx_eval), a.batch)]
        chunks = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in c.items()} for c in chunks]

        split_results = {
            'n_eligible': n_eligible,
            'n_eval_requested': a.n_eval,
            'n_eval_bound': n_eval_bound,
            # The claim that this sweep scores the SAME rows as interaction_2x2.py has to be checkable
            # from the artefact, not taken on trust. Record them. (interaction_2x2.py dumps its rows to
            # npz; a set comparison between the two is then a one-liner.)
            'rows_sha': __import__('hashlib').sha1(idx_eval.tobytes()).hexdigest()[:16],
            'rows_first5': [int(v) for v in idx_eval[:5]],
            'rows_n': int(len(idx_eval)),
            'results_by_alpha': {}
        }
        
        print(f"\n--- {name} ---", flush=True)

        # One resample matrix per split, from its own generator, SHARED by every alpha, so
        # differences along the curve are not contaminated by bootstrap noise that differs per point.
        boot_rng = np.random.default_rng(a.seed)
        boot_idx_split = None

        for alpha in alphas:
            r_full, r_ablated, dymax_atom = evaluate_chunks_alpha(model, chunks, alpha, key=a.key)
            ok = np.isfinite(r_full) & np.isfinite(r_ablated)
            r_f, r_a = r_full[ok], r_ablated[ok]
            
            n_rows = len(r_f)
            if n_rows == 0:
                continue
            
            s_full = float(np.median(r_f))
            s_ablated = float(np.median(r_a))
            atom_effect = s_full - s_ablated
            
            diff = r_f - r_a
            atom_effect_median_per_row = float(np.median(diff))
            
            non_zero = diff[diff != 0]
            if len(non_zero) > 0:
                n_pos = np.sum(non_zero > 0)
                sign_test = scipy.stats.binomtest(n_pos, len(non_zero), p=0.5, alternative='two-sided')
                sign_test_p = sign_test.pvalue
            else:
                sign_test_p = 1.0
                
            if boot_idx_split is None or boot_idx_split.shape[1] != n_rows:
                boot_idx_split = boot_rng.integers(0, n_rows, size=(a.n_boot, n_rows))
            boot_idx = boot_idx_split
            boot_f = np.median(r_f[boot_idx], axis=1)
            boot_a = np.median(r_a[boot_idx], axis=1)
            boot_effect = boot_f - boot_a
            ci95 = [float(np.percentile(boot_effect, 2.5)), float(np.percentile(boot_effect, 97.5))]
            
            split_results['results_by_alpha'][str(alpha)] = {
                'score_full': s_full,
                'score_ablated': s_ablated,
                'atom_effect': atom_effect,
                'atom_effect_median_per_row': atom_effect_median_per_row,
                'sign_test_p': sign_test_p,
                'atom_effect_ci95': ci95,
                'dY_max': dymax_atom
            }
            
            print(f"Alpha {alpha:4.2f} | Effect: {atom_effect:+.5f} (CI [{ci95[0]:+.5f}, {ci95[1]:+.5f}]) | MedPerRow: {atom_effect_median_per_row:+.5f} | p={sign_test_p:.2e} | dYmax={dymax_atom:.4f}", flush=True)

        out['splits'][name] = split_results

    with io.open(dst_json, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(out, indent=2) + '\n')
    print(f"\nWrote results to {dst_json}", flush=True)

if __name__ == '__main__':
    main()
