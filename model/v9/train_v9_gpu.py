# -*- coding: utf-8 -*-
"""
v9 training on Kaggle T4 x2.

The guard set is carried over verbatim from v6/v7 because each guard caught a real, expensive failure:
  * no CPU fallback  -- a GPU kernel silently on CPU still burns the whole session
  * P100 probe       -- Kaggle torch has no sm_60 kernels; a 1s matmul beats an 8h waste
  * both-GPUs-active -- DataParallel degrades to one GPU silently if the batch is too small
  * check_inputs_v9  -- data.py degrades to neutral defaults when a file is missing, and a rebuilt Kaggle
                        dataset once dropped the compound holdout AND reliability weighting with no log line
  * resumable ckpt + time budget -- a non-clean exit discards /kaggle/working

v9 adds two of its own, both for failure modes measured during the v9 build:
  * the quantiser must be FITTED before training -- an unfitted one sends every value to bin 0 and the
    expression input vanishes while the run looks completely normal (caught by test_v9.py);
  * bins are fitted on TRAINING ROWS ONLY, asserted here rather than trusted.

Run 3 seeds. Seed sd on this project reaches 0.0232, so a single run is not a result.
    python model/v9/train_v9_gpu.py --seed 0
"""
import os, sys, json, time, math, argparse, glob

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v7'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9TrainConfig, V9DataConfig
from model_v9 import LincsV9, v9_loss
from data_v9 import LincsV9Dataset, build_splits, collate_v9, check_inputs_v9
from train_v7_gpu import probe_gpu, assert_both_gpus_used, wsd_lambda, EMA


def resolve_v9(dc):
    """Kaggle mounts inputs flat under /kaggle/input/**; find each file rather than assuming a layout."""
    if not os.path.isdir('/kaggle/input'):
        return dc
    hit = lambda n: next(iter(glob.glob(f'/kaggle/input/**/{n}', recursive=True)), None)
    from config_v6 import resolve_paths
    dc = resolve_paths(dc)
    for attr, name in [('x_trt_path', 'X_trt_l3.npy'), ('x_ctl_path', 'X_ctl_l3.npy'),
                       ('l3_covered_path', 'l3_covered.npy'), ('l3_yrow_path', 'l3_yrow.npy'),
                       ('m_pathway_path', 'M_pathway_v9.npy'), ('ppi_v9_path', 'STRING_adj_978_v9.npy'),
                       ('gene_vec_path', 'gene_vectors_978.npy'),
                       ('pathway_info_v9_path', 'pathway_info_v9.tsv'),
                       ('landmark_symbols_path', 'landmark_symbols_v9.tsv')]:
        p = hit(name)
        if p is None:
            raise SystemExit(f'FATAL: {name} not found under /kaggle/input. The v9 dataset is incomplete; '
                             f'refusing to train on a partial substrate.')
        setattr(dc, attr, p)
    return dc


@torch.no_grad()
def evaluate(model, ds, idx, device, batch=64, max_n=3000, seed=0):
    """Every metric this project has ever reported is on the reproducible stratum, and v9 reports THREE
    conventions side by side so nothing is hidden by the choice of one:
      delta  -- what v9 trains on and what the field reports as Pearson_deg
      abs    -- the absolute convention, which self-agrees at 0.93 before any model is involved
      l5     -- the Level-5 z-score, the only target v3-v7 ever used, kept for comparability
    `copy_ctl` is the do-nothing baseline in the absolute convention: XPert's own released predictions beat
    it by only +0.060 [RESULTS 23], so an absolute number without it beside it is uninterpretable.
    """
    if len(idx) == 0:
        return {}
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
    was = model.training
    model.eval()
    P = {k: [] for k in ['delta', 'abs', 'l5']}
    T = {k: [] for k in ['delta', 'abs', 'l5']}
    ctls, keep3 = [], []
    for s in range(0, len(idx), batch):
        b = collate_v9([ds[i] for i in idx[s:s + batch]])
        bd = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(bd)
        for k in P:
            P[k].append(out[k].float().cpu())
        T['delta'].append(b['y_delta']); T['abs'].append(b['y_abs']); T['l5'].append(b['y_l5'])
        ctls.append(b['x_ctl']); keep3.append(b['m_l3'])
    model.train(was)
    m3 = torch.cat(keep3)
    ctl = torch.cat(ctls)

    def pear(a, b):
        a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
        return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8))

    res = {'n': int(len(idx))}
    for k in ['delta', 'abs', 'l5']:
        p, t = torch.cat(P[k]), torch.cat(T[k])
        if k in ('delta', 'abs'):
            p, t = p[m3], t[m3]
        if len(p):
            res[k] = round(float(pear(p, t).median()), 4)
    if int(m3.sum()):
        res['copy_ctl_abs'] = round(float(pear(ctl[m3], torch.cat(T['abs'])[m3]).median()), 4)
        res['zero_delta'] = 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    tc = V9TrainConfig()
    for f, t in [('epochs', int), ('batch', int), ('lr', float), ('budget_h', float), ('fold', int),
                 ('workers', int), ('ema_decay', float)]:
        ap.add_argument(f'--{f}', type=t, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--expr_encoder', default=None, choices=['raw', 'binned'])
    ap.add_argument('--no_aux', action='store_true')
    ap.add_argument('--no_ppi', action='store_true')
    ap.add_argument('--no_gene_vectors', action='store_true')
    ap.add_argument('--no_epi_embedding', action='store_true')
    ap.add_argument('--no_cell_ctl', action='store_true')
    ap.add_argument('--use_ccle', action='store_true')
    ap.add_argument('--gpus', type=int, default=2)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    cfg = V9Config()
    for f in ['epochs', 'batch', 'lr', 'budget_h', 'fold', 'workers', 'ema_decay']:
        if getattr(a, f, None) is not None:
            setattr(tc, f, getattr(a, f))
    tc.seed = a.seed
    if a.expr_encoder:
        cfg.expr_encoder = a.expr_encoder
    cfg.use_aux = not a.no_aux
    cfg.use_ppi = not a.no_ppi
    cfg.use_gene_vectors = not a.no_gene_vectors
    cfg.epi_as_gene_embedding = not a.no_epi_embedding

    names = probe_gpu(require_n=a.gpus)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda'
    n_gpu = torch.cuda.device_count()
    print(f'GPUs: {names} | seed={a.seed} encoder={cfg.expr_encoder} aux={cfg.use_aux} ppi={cfg.use_ppi} '
          f'gene_vec={cfg.use_gene_vectors} epi_emb={cfg.epi_as_gene_embedding}', flush=True)

    dc = resolve_v9(V9DataConfig())
    dc.cell_fold = tc.fold
    dc.cache_in_ram = True
    dc.use_ccle = a.use_ccle
    dc.use_cell_ctl = not a.no_cell_ctl
    cfg.use_cell_ctl = dc.use_cell_ctl
    WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
    CKPT = f'{WORK}/ckpt_v9_fold{tc.fold}_seed{a.seed}.pt'
    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)

    M = np.load(R(dc.m_pathway_path))
    ppi = np.load(R(dc.ppi_v9_path)) if cfg.use_ppi else None
    gv = np.load(R(dc.gene_vec_path)) if cfg.use_gene_vectors else None
    cfg.n_pathways = M.shape[0]
    print(f'pathway nodes {M.shape[0]} ({int((M.sum(0) > 0).sum())}/{M.shape[1]} genes covered) | '
          f'STRING {int((np.asarray(ppi) != 0).sum() // 2) if ppi is not None else 0} edges | '
          f'gene vectors {None if gv is None else gv.shape}', flush=True)

    shared = LincsV9Dataset.load_shared_v9(dc)
    full = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(full, dc)
    print('splits ' + str({k: len(v) for k, v in sp.items() if not k.startswith('_')}), flush=True)
    check_inputs_v9(full, sp, dc)

    core = LincsV9(cfg, M, ppi, gv)
    if cfg.expr_encoder == 'binned':
        rows = full.ds_to_l3[sp['train']]
        rows = np.sort(rows[rows >= 0])
        take = rows[np.linspace(0, len(rows) - 1, min(40000, len(rows))).astype(int)]
        core.fit_bins(np.asarray(full.Xctl[take], np.float32))
        if not core.bins_fitted:
            raise SystemExit('FATAL: quantiser did not fit; the expression input would be ignored.')
        test_rows = set(np.asarray(sp['test_coldcell']).tolist())
        if test_rows & set(np.asarray(sp['train']).tolist()):
            raise SystemExit('FATAL: train and test rows overlap.')
        print(f'quantiser fitted on {len(take)} TRAINING rows, {cfg.n_bins} bins, mode={cfg.bin_mode}',
              flush=True)
    core = core.to(device)
    print(f'params {sum(p.numel() for p in core.parameters()) / 1e6:.2f}M', flush=True)
    if tc.batch % n_gpu:
        raise SystemExit(f'FATAL: batch {tc.batch} is not divisible by {n_gpu} GPUs.')
    model = torch.nn.DataParallel(core) if n_gpu > 1 else core

    train_ds = LincsV9Dataset(dc, indices=sp['train'], _shared=shared)
    dl = DataLoader(train_ds, batch_size=tc.batch, shuffle=True, drop_last=True, num_workers=tc.workers,
                    persistent_workers=tc.workers > 0, pin_memory=True, prefetch_factor=6,
                    collate_fn=collate_v9)

    Mn = torch.as_tensor(M, dtype=torch.float32)
    Mn = (Mn / Mn.sum(1, keepdim=True).clamp(min=1)).to(device)

    decay, no_decay = [], []
    for n, p in core.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or 'gene_repr.emb' in n or 'type_' in n) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': tc.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=tc.lr, betas=(0.9, 0.95))
    total = tc.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, wsd_lambda(total, tc.warmup_frac, tc.decay_frac))
    scaler = torch.amp.GradScaler('cuda')
    ema = EMA(core, tc.ema_decay)

    rep = {k: sp[k][full.strength[sp[k]] >= dc.eval_min_strength]
           for k in ['test_coldcell', 'test_colddrug', 'test_coldboth']}
    hist, t0 = [], time.time()
    model.train()
    for epoch in range(tc.epochs):
        te = time.time(); run = n = 0; last = {}
        for it, b in enumerate(dl):
            bd = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out, aux = model(bd, return_aux=True)
                loss, parts = v9_loss(out, bd, cfg, Mn, aux)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(core.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(core)
            run += float(loss); n += 1; last = parts
            if epoch == 0 and it == 0 and n_gpu > 1:
                assert_both_gpus_used(' after first step')
            if it % 200 == 0:
                print(f'  e{epoch} it{it}/{len(dl)} loss {float(loss):.4f} '
                      f'lr {sched.get_last_lr()[0]:.2e} {last}', flush=True)
        rec = {'epoch': epoch, 'train_loss': run / max(n, 1), 'parts': last,
               'sec': round(time.time() - te, 1), 'elapsed_h': round((time.time() - t0) / 3600, 3)}
        for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                          ('unseen_both', 'test_coldboth')]:
            rec[name] = evaluate(model, full, rep[key], device, seed=a.seed)
        hist.append(rec)
        print('EPOCH ' + json.dumps(rec), flush=True)
        torch.save({'model': core.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'epoch': epoch, 'hist': hist,
                    'cfg': vars(cfg), 'tcfg': vars(tc)}, CKPT)
        json.dump(hist, open(f'{WORK}/metrics_v9_fold{tc.fold}_seed{a.seed}.json', 'w'), indent=2)
        if (time.time() - t0) / 3600 > tc.budget_h:
            print('time budget reached -> clean checkpoint + stop (resumable)', flush=True)
            break
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
