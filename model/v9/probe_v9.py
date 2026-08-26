# -*- coding: utf-8 -*-
"""
The v9 interpretability report — every readout with its MEASURED null, every ablation to the MEAN with
|dY|max beside it.

This is the deliverable. Accuracy on this benchmark is a means; the project's claim is mechanistic
association, and that claim is only defensible if each readout is scored against the chance level it
actually has. On pathway-level target alignment 0.5 is NOT chance: an untrained model scores 0.218 against
a permutation null of 0.229 [handoff §D.5]. So nothing here is reported against an assumed baseline.

Rules enforced in code, not by discipline:
  * ablate to the batch MEAN, never 0 or 1 (ablating pathway conductance to 1 once produced a +0.103
    "contribution" that was a 30x scale artefact; the true effect was -0.003/+0.006);
  * report |dY|max next to every effect, so a TRUE NULL (effect ~0, |dY|max >> 0) stays distinguishable
    from a branch that never fired (effect ~0, |dY|max == 0);
  * every readout is scored against a permutation null with its own p-value;
  * results are per split, because chromatin's one positive result was +0.0061 on unseen COMPOUND and
    ~0 on the two unseen-CELL splits -- an aggregate number would have hidden it.

    python model/v9/probe_v9.py --ckpt ckpt_v9_fold0_seed0.pt
"""
import os, sys, csv, json, argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9
from data_v9 import LincsV9Dataset, build_splits, collate_v9
from train_v9_gpu import resolve_v9
from interp_v9 import ablate_to_mean, ablate_module_to_mean, permutation_null, pathway_alignment


def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None,
                    help="omit with --untrained to MEASURE the readout's chance level")
    ap.add_argument('--untrained', action='store_true',
                    help="fresh weights: this is how the readout's chance level is measured "
                         "rather than assumed. An untrained v6 scored 0.218 against a "
                         "permutation null of 0.229 -- i.e. 0.5 is NOT chance here.")
    ap.add_argument('--skip_ablations', action='store_true')
    ap.add_argument('--n_eval', type=int, default=1500)
    ap.add_argument('--n_perm', type=int, default=200)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if a.untrained:
        ck = {'cfg': vars(V9Config()), 'epoch': -1}
    else:
        if not a.ckpt:
            raise SystemExit('give --ckpt, or --untrained to measure the readout chance level')
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = V9Config(**{k: v for k, v in ck['cfg'].items() if k in V9Config.__dataclass_fields__})
    dc = resolve_v9(V9DataConfig())
    dc.cache_in_ram = False
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_pathway_path))
    ppi = np.load(R(dc.ppi_v9_path))
    gv = np.load(R(dc.gene_vec_path))
    info = list(csv.DictReader(open(R(dc.pathway_info_v9_path), encoding='utf-8'), delimiter='\t'))

    torch.manual_seed(a.seed)
    model = LincsV9(cfg, M, ppi, gv)
    if not a.untrained:
        model.load_state_dict(ck['model'])
    model = model.to(dev).eval()
    print(f'{"UNTRAINED (chance-level measurement)" if a.untrained else a.ckpt} '
          f'(epoch {ck.get("epoch")}) | encoder {cfg.expr_encoder} | {M.shape[0]} named nodes', flush=True)

    shared = LincsV9Dataset.load_shared_v9(dc)
    ds = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(ds, dc)
    if a.untrained and cfg.expr_encoder == 'binned':
        # a checkpoint restores the quantiser's buffers; a fresh model has none, and an unfitted quantiser
        # would send every value to bin 0 and silently delete the expression input. TRAINING rows only.
        rows = ds.ds_to_l3[sp['train']]
        rows = np.sort(rows[rows >= 0])
        model.fit_bins(np.asarray(ds.Xctl[rows[np.linspace(0, len(rows) - 1, 20000).astype(int)]],
                                  np.float32))
        model = model.to(dev)
        if not model.bins_fitted:
            raise SystemExit('FATAL: quantiser did not fit.')
    out = {'ckpt': 'untrained' if a.untrained else os.path.basename(a.ckpt),
           'epoch': int(ck.get('epoch', -1)), 'splits': {}}

    for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                      ('unseen_both', 'test_coldboth')]:
        idx = sp[key][ds.strength[sp[key]] >= dc.eval_min_strength]
        idx = idx[ds.has_l3[idx]]
        if len(idx) < 300:
            continue
        idx = np.sort(rng.choice(idx, min(a.n_eval, len(idx)), replace=False))
        rec = {'n': int(len(idx))}
        print(f'\n=== {name} (n={len(idx)}) ===', flush=True)

        # one big batch per chunk, kept for the ablations so every ablation sees identical rows
        chunks = [collate_v9([ds[i] for i in idx[s:s + a.batch]])
                  for s in range(0, len(idx), a.batch)]
        chunks = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in c.items()} for c in chunks]

        with torch.no_grad():
            preds = [model(c) for c in chunks]
        yd = torch.cat([p['delta'] for p in preds]).float().cpu().numpy()
        td = torch.cat([c['y_delta'] for c in chunks]).float().cpu().numpy()
        rec['pearson_delta'] = round(float(np.nanmedian(pearson_rows(yd, td))), 4)
        ctl = torch.cat([c['x_ctl'] for c in chunks]).float().cpu().numpy()
        ta = torch.cat([c['y_abs'] for c in chunks]).float().cpu().numpy()
        ya = torch.cat([p['abs'] for p in preds]).float().cpu().numpy()
        rec['pearson_abs'] = round(float(np.nanmedian(pearson_rows(ya, ta))), 4)
        rec['null_copy_ctl_abs'] = round(float(np.nanmedian(pearson_rows(ctl, ta))), 4)
        rec['abs_value_added_over_doing_nothing'] = round(rec['pearson_abs'] - rec['null_copy_ctl_abs'], 4)
        print(f'  delta {rec["pearson_delta"]:.4f} | abs {rec["pearson_abs"]:.4f} vs copy-the-control '
              f'{rec["null_copy_ctl_abs"]:.4f} -> value added {rec["abs_value_added_over_doing_nothing"]:+.4f}')

        # ---- ablations: to the MEAN, with |dY|max ----
        score = lambda y: float(np.nanmedian(pearson_rows(y.float().cpu().numpy(), td[:len(y)])))
        abl = {}
        for label, kind, target in ([] if a.skip_ablations else [('matched_control', 'input', 'x_ctl'),
                                    ('cell_control', 'input', 'x_cell'),
                                    ('chromatin', 'input', 'E'),
                                    ('drug_global', 'input', 'u_feats'),
                                    ('atom_tokens', 'input', 'atoms'),
                                    ('lineage', 'input', 'cell_ctx'),
                                    ('string_mp', 'module', 'ppi'),
                                    ('pathway_readout', 'module', 'pathway'),
                                    ('gene_vectors', 'module', 'gene_repr')]):
            d_tot, m_tot = 0.0, 0.0
            for c in chunks:
                if kind == 'input':
                    if target not in c:
                        break
                    d, m = ablate_to_mean(model, c, target, lambda y: float(y.abs().mean()))
                else:
                    mod = getattr(model, target, None)
                    if mod is None:
                        break
                    d, m = ablate_module_to_mean(model, c, mod, lambda y: float(y.abs().mean()))
                d_tot += d * len(c['x_ctl']); m_tot = max(m_tot, m)
            abl[label] = {'d_score': round(d_tot / max(len(idx), 1), 5), 'dY_max': round(m_tot, 4),
                          'fired': bool(m_tot > 1e-5)}
            print(f'    ablate {label:18s} dScore {abl[label]["d_score"]:+.5f}  |dY|max '
                  f'{abl[label]["dY_max"]:.4f}  {"" if abl[label]["fired"] else "<-- NEVER FIRED"}',
                  flush=True)
        rec['ablations'] = abl

        # ---- the named pathway readout, against its permutation null ----
        with torch.no_grad():
            acts = torch.cat([model(c, return_aux=True)[1]['pathway_activations'].mean(-1)
                              for c in chunks]).float().cpu().numpy()
        obs = pathway_alignment(acts, td, M)
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        tgt = np.abs(td) @ Mn.T
        nulls = []
        for _ in range(a.n_perm):
            perm = rng.permutation(acts.shape[1])
            nulls.append(pathway_alignment(acts[:, perm], td, M))
        nulls = np.array(nulls)
        p = float(((nulls >= obs).sum() + 1) / (a.n_perm + 1))
        rec['pathway_alignment'] = {'observed': round(float(obs), 4),
                                    'permutation_null_mean': round(float(nulls.mean()), 4),
                                    'permutation_null_sd': round(float(nulls.std()), 4),
                                    'p': round(p, 4),
                                    'beats_null': bool(obs > nulls.mean() + 2 * nulls.std())}
        print(f'  pathway alignment {obs:+.4f} vs permutation null {nulls.mean():+.4f} '
              f'+/- {nulls.std():.4f}  p={p:.4f}  '
              f'{"BEATS NULL" if rec["pathway_alignment"]["beats_null"] else "does NOT beat its null"}')

        # ---- which named nodes does it actually light up ----
        top = np.argsort(-acts.mean(0))[:12]
        rec['top_named_nodes'] = [{'source': info[i]['source'], 'term': info[i]['term_name'],
                                   'n_landmarks': int(info[i]['n_landmark_genes']),
                                   'mean_activation': round(float(acts[:, i].mean()), 4)} for i in top]
        print('  top named nodes: ' + '; '.join(f'{info[i]["term_name"][:44]}' for i in top[:5]))
        out['splits'][name] = rec

    WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(
        os.path.dirname(os.path.dirname(HERE)), 'model', 'results')
    os.makedirs(WORK, exist_ok=True)
    tag = 'untrained' if a.untrained else os.path.basename(a.ckpt).replace('.pt', '')
    dst = os.path.join(WORK, f'v9_probe_{tag}.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
