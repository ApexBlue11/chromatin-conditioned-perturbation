# -*- coding: utf-8 -*-
"""
Run XPert's OWN released checkpoint, with XPert's OWN code, on rows of our choosing.

Why. Every comparison this project has made against XPert so far has been between two numbers measured on
different rows, and RESULTS 39/40 shows that reading such a pair as a model difference goes wrong in both
directions. Their released `y_pred.npy` covers one figure subset (3,439 HDACi conditions). To compare on
their MAIN benchmark -- the 68,830-condition mdmt corpus with its warm / cold-cell / cold-drug splits --
their model has to be run, not read off.

This driver imports their modules rather than reimplementing them:
    models.model_XPert.XPertNet     the architecture
    datasets.MyDataset.MyDataset    the exact input tensors, including their expression binning
    metrics.get_metrics_new         the exact metric, mean of per-row Pearson (NOT median)
and reproduces their prediction convention from train_xpert.py:225 --
    y_pred = deg_output + ctl_raw_data
i.e. the delta head added to the TRUE control, the same anchoring v9 uses.

TWO THINGS THAT WOULD HAVE SILENTLY CORRUPTED THIS, both caught by inspecting the checkpoint:
  * the released weights contain `cls_token` and `class_fc`, which XPertNet only builds when
    `--include_cell_idx True` -- a NON-DEFAULT flag. Running with argparse defaults would have built a
    different forward pass. `--strict` load is therefore mandatory here and is on by default.
  * `pert_time_idx` is absent from obs and is derived by their dataset; `pert_dose_idx` is present and is
    used as-is. Recomputing either would change the dose/time embeddings.

The checkpoint is `l1000_mdmt_warm_split.pth`: warm split only. Which of the five warm folds it was
trained on is not recorded, so `--diagnose` scores every fold and reports the spread -- a checkpoint
scores far higher on folds whose test rows were in its own training set, and the honest number is the
lowest one.

    python model/v9/xpert_native_eval.py --nfold split_1 --diagnose
    python model/v9/xpert_native_eval.py --nfold split_1 --out xpert_split_1_profile.npy
"""
import os, sys, json, time, types, argparse, logging

import numpy as np

XPERT = r'C:\Projects\LINCS\external\xpert\code\XPert'
CKPT = r'C:\Projects\LINCS\external\xpert\saved_model_extracted\saved_model\l1000_mdmt_warm_split.pth'
UNIMOL = os.path.join(XPERT, 'processed_data', 'unimol_mdmt_1970.npz')
H5AD = os.path.join(XPERT, 'processed_data', 'l1000_mdmt_68830_subset.h5ad')


def their_args(**kw):
    """The argparse namespace train_xpert.py would have built, with the flags the checkpoint implies."""
    a = types.SimpleNamespace(
        mode='test', nfold='split_1', drug_feat='unimol', device='cpu', model='XPert',
        config='config_l1000', seed=2024, dataset='l1000_mdmt', pretrained_mode='global',
        include_cell_idx=True,          # the checkpoint HAS cls_token + class_fc; see the docstring
        wo_HG=False, wo_atom=False, wo_atom_HG=False, wo_unimol=False, wo_ppi=False,
        use_gene_pos_emed=False, use_gradscaler=False, lr_scheduler=True,
        resume_from=None, saved_model_path=None, saved_model=None, pretrained_model=None,
        output_profile=True, output_attention=False, output_cls_embed=False,
        weighted_loss=False, kl_loss=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def load_everything(device, quiet=True):
    """Import their code with their working directory, so the relative paths in their config resolve."""
    sys.path.insert(0, XPERT)
    os.chdir(XPERT)
    # Their model_utils.py imports flash_attn at module scope and takes the FLASH branch by default
    # (`if output_attention: dense else: flash`). Without CUDA the real package cannot be installed, so a
    # numerically faithful CPU stand-in is put on the path -- but ONLY if the genuine one is absent.
    try:
        import flash_attn                                            # noqa: F401
        print('using the real flash_attn', flush=True)
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '_shims'))
        import flash_attn                                            # noqa: F401
        print('flash_attn absent; using the exact dense stand-in in model/v9/_shims', flush=True)
    import yaml
    import torch
    import anndata as ad
    from models.model_XPert import XPertNet
    from datasets.MyDataset import MyDataset
    import metrics as their_metrics

    logger = logging.getLogger('xpert')
    logger.addHandler(logging.NullHandler() if quiet else logging.StreamHandler())
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    cfg = yaml.safe_load(open(os.path.join(XPERT, 'configs', 'config_l1000.yaml')))
    return dict(torch=torch, ad=ad, XPertNet=XPertNet, MyDataset=MyDataset, M=their_metrics,
                cfg=cfg, logger=logger)


def drug_feat_dict(path=None, pad_mode='asis', seed=0):
    """{pert_idx: (122, 514) float32}. Their MyDataset indexes drug_feat[pert_idx], so a dict is a
    drop-in for the full (8981, 122, 514) array without holding 4.5 GB.

    `pad_mode` exists to settle review 007 C4 by measurement [RESULTS 69.6]. Their executed attention
    path takes no mask (`model_utils.py:226`, verified), and 55.8 % of the 122 atom slots are padding
    whose features are exactly zero -- but `key`/`value` are `nn.Linear(..., bias=True)`, so each padded
    slot still emits one identical learned constant and takes softmax mass. The open question is whether
    the released checkpoint LEARNED to suppress those slots. If it did, masking would change nothing and
    the unmasked path is fine; if it did not, padding is load-bearing.

    This tests it directly instead of plumbing a mask through their code: perturb ONLY the 512 feature
    channels of the padded slots and see whether the released weights notice.

      asis  - exactly as released (padded features zero). The reference arm.
      noise - padded feature channels filled with Gaussian noise matched to the valid-atom scale.
      ones  - padded feature channels set to 1.0, a large coherent perturbation.

    Channel 0 (validity) and channel 1 (atom symbol) are NEVER touched: channel 0 is what their mask is
    built from and channel 1 indexes an embedding, so perturbing either would change a different thing
    and confound the measurement.
    """
    path = path or UNIMOL
    if not os.path.exists(path):
        raise SystemExit('FATAL: %s missing -- run model/v9/fetch_xpert_unimol.py first' % path)
    z = np.load(path)
    out = {int(i): f for i, f in zip(z['idx'], z['feat'])}
    if pad_mode == 'asis':
        return out
    rng = np.random.default_rng(seed)
    scale = float(np.abs(z['feat'][:, :, 2:][z['feat'][:, :, 0] == 1]).std())
    n_touched = 0
    for k in out:
        f = out[k].copy()
        pad = f[:, 0] == 0
        if pad.any():
            if pad_mode == 'noise':
                f[pad, 2:] = rng.normal(0.0, scale, size=(int(pad.sum()), f.shape[1] - 2)).astype(f.dtype)
            elif pad_mode == 'ones':
                f[pad, 2:] = 1.0
            else:
                raise SystemExit('FATAL: unknown pad_mode %r' % pad_mode)
            n_touched += int(pad.sum())
        out[k] = f
    print('pad_mode=%s: perturbed %d padded slots across %d compounds (valid-atom sd %.4f)'
          % (pad_mode, n_touched, len(out), scale), flush=True)
    return out


def build_model(E, args, device, ckpt=CKPT):
    torch = E['torch']
    model = E['XPertNet'](args, E['cfg'], device, E['logger'])
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(sd, strict=True)     # strict: a silently different architecture is the hazard
    model = model.to(device).eval()
    return model, sd


def subset(E, adata, mask):
    """Rows of the AnnData, with obs re-indexed 0..n-1 so their positional obs lookups are unambiguous
    under pandas 2 (their MyDataset does `obs['col'][idx]` with an integer idx)."""
    sub = adata[mask].to_memory() if getattr(adata, 'isbacked', False) else adata[mask].copy()
    sub.obs.index = range(sub.n_obs)
    return sub


def predict(E, model, ds, batch=64, device='cpu', log_every=20):
    torch = E['torch']
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, drop_last=False)
    y_true, y_pred, ctl_true = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(dl):
            out = model(data)
            trt_output, ctl_output, deg_output, trt_raw, ctl_raw = out[0], out[1], out[2], out[3], out[4]
            y_true.append(trt_raw.cpu().numpy())
            y_pred.append((deg_output + ctl_raw).cpu().numpy())      # train_xpert.py:215
            ctl_true.append(ctl_raw.cpu().numpy())
            if log_every and (i + 1) % log_every == 0:
                done = (i + 1) * batch
                print('    %d/%d rows  %.0fs' % (done, len(ds), time.time() - t0), flush=True)
    return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(ctl_true))


def score(E, y, f, ctl):
    """Their metric, plus the null their metric never reports."""
    M = E['M']
    out = {
        'Pearson': float(M.pearson(y, f)),
        'Pearson_deg': float(M.pearson(y - ctl, f - ctl)),
        'Spearman': float(M.spearman(y, f)),
        'Spearman_deg': float(M.spearman(y - ctl, f - ctl)),
        'MSE': float(M.mse(y, f)),
        'MSE_deg': float(M.mse(y - ctl, f - ctl)),
        'copy_ctl_Pearson': float(M.pearson(y, ctl)),
        'n': int(len(y)),
    }
    out['abs_value_added'] = out['Pearson'] - out['copy_ctl_Pearson']
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nfold', default='split_1')
    ap.add_argument('--rows', default='test', choices=['test', 'train', 'all'])
    ap.add_argument('--h5ad', default=H5AD)
    ap.add_argument('--unimol', default=None)
    ap.add_argument('--ckpt', default=CKPT,
                    help='which released checkpoint; the warm-split one is the default, but the HDACi '
                         'figure predictions do not come from it -- see RESULTS')
    ap.add_argument('--row_file', default=None,
                    help='.npy of explicit row indices, overriding --rows; for targeted diagnostics')
    ap.add_argument('--max_rows', type=int, default=0, help='0 = all')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='write a *_predict_profile.npy in THEIR format')
    ap.add_argument('--pad_mode', default='asis', choices=['asis', 'noise', 'ones'],
                    help='perturb the PADDED atom slots, to test whether the released '
                         'checkpoint learned to ignore them [RESULTS 69.6]')
    ap.add_argument('--diagnose', action='store_true',
                    help='score every warm fold, to find which one the checkpoint was trained on')
    a = ap.parse_args()

    E = load_everything(a.device)
    torch = E['torch']
    print('loading %s' % a.h5ad, flush=True)
    adata = E['ad'].read_h5ad(a.h5ad)
    feats = drug_feat_dict(a.unimol, pad_mode=a.pad_mode, seed=a.seed)
    print('drug features for %d compounds' % len(feats), flush=True)

    rng = np.random.default_rng(a.seed)
    folds = ['split_%d' % k for k in range(1, 6)] if a.diagnose else [a.nfold]
    results = {}
    for nf in folds:
        args = their_args(nfold=nf, device=a.device)
        if a.row_file:
            idx = np.load(a.row_file).astype(np.int64)
        elif a.rows == 'all':
            idx = np.arange(adata.n_obs)
        else:
            lab = np.asarray(adata.obs[nf]).astype(str)
            idx = np.flatnonzero(lab == a.rows)
        if a.max_rows and len(idx) > a.max_rows:
            idx = np.sort(rng.choice(idx, a.max_rows, replace=False))
        missing = {int(p) for p in np.asarray(adata.obs['pert_idx'])[idx]} - set(feats)
        if missing:
            raise SystemExit('FATAL: %d pert_idx have no unimol row, e.g. %s'
                             % (len(missing), sorted(missing)[:5]))
        print('\n=== %s / %s : %d rows ===' % (nf, a.rows, len(idx)), flush=True)
        sub = subset(E, adata, idx)
        ds = E['MyDataset'](sub, feats, args=args, config=E['cfg'], logger=E['logger'],
                            max_value=E['cfg']['dataset']['max_value'],
                            min_value=E['cfg']['dataset']['min_value'])
        model, _ = build_model(E, args, a.device, a.ckpt)
        y, f, c = predict(E, model, ds, a.batch, a.device)
        m = score(E, y, f, c)
        results[nf] = m
        print('  Pearson %.4f  Pearson_deg %.4f  (copy-the-control %.4f)'
              % (m['Pearson'], m['Pearson_deg'], m['copy_ctl_Pearson']), flush=True)
        if a.out:
            np.save(a.out if len(folds) == 1 else a.out.replace('.npy', '_%s.npy' % nf),
                    {'y_true': y, 'y_pred': f, 'ctl_true': c, 'deg_true': y - c, 'deg_pred': f - c,
                     'row_index': idx})

    if a.diagnose:
        print('\n' + '=' * 88)
        print('WHICH WARM FOLD WAS THE RELEASED CHECKPOINT TRAINED ON?')
        print('A fold whose test rows were in its training set scores inflated; the LOWEST is the honest one.')
        print('=' * 88)
        for nf, m in sorted(results.items(), key=lambda kv: kv[1]['Pearson_deg']):
            print('  %-10s n=%5d  Pearson %.4f  Pearson_deg %.4f' % (nf, m['n'], m['Pearson'],
                                                                     m['Pearson_deg']))
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results',
                       'xpert_native_%s_%s.json' % (a.nfold if not a.diagnose else 'diagnose', a.rows))
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(results, open(dst, 'w'), indent=2)
    print('\nwrote %s' % dst)


if __name__ == '__main__':
    main()
