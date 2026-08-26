# -*- coding: utf-8 -*-
"""
GATE 5: does binning-and-embedding the expression input actually beat a linear layer on the raw scalar?

V9_HANDOFF §D.2 says to move to a 128-bin embedded encoding because that is what the field does
(`n_bins: 128`, `exp_vocab_size` in XPert's config), and §D gates it: "ridge/simple A/B before committing
to a full train". Six architectures in this project produced no difference distinguishable from seed noise,
so an unmeasured architectural change is exactly the kind of thing that has never paid here.

This runs the REAL v9 model at reduced width and depth, changing ONE thing: `expr_encoder`. A ridge cannot
answer the question -- an embedding lookup has no linear analogue -- and a throwaway proxy model would be
answering about the proxy. Everything else is identical, including the data, the split, the schedule, and
the seed sequence.

Reported as mean +/- range over >=3 seeds, per method rule 3, because seed sd on this project reaches
0.0232 and a 2-sd band is +/-0.046. A difference smaller than the seed range is reported as NO DIFFERENCE.

    python model/v9/ab_encoder.py --seeds 3 --epochs 2 --train_n 24000

On CPU this is hours; it is meant for a GPU. `--smoke` runs a 2-minute shape/plumbing check instead.
"""
import os, sys, json, time, argparse, math

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9, v9_loss
from data_v9 import LincsV9Dataset, build_splits, collate_v9, check_inputs_v9
from train_v9_gpu import resolve_v9


def pearson_rows(a, b):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8))


@torch.no_grad()
def evaluate(model, ds, idx, device, batch=64, max_n=2500, seed=0):
    if len(idx) == 0:
        return {}
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
    was = model.training
    model.eval()
    acc = {k: [] for k in ['delta', 'l5']}
    tgt = {k: [] for k in ['delta', 'l5']}
    keep = []
    for s in range(0, len(idx), batch):
        b = collate_v9([ds[i] for i in idx[s:s + batch]])
        bd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(bd)
        acc['delta'].append(out['delta'].float().cpu())
        acc['l5'].append(out['l5'].float().cpu())
        tgt['delta'].append(b['y_delta'])
        tgt['l5'].append(b['y_l5'])
        keep.append(b['m_l3'])
    model.train(was)
    m = torch.cat(keep)
    res = {}
    for k in ['delta', 'l5']:
        p, t = torch.cat(acc[k]), torch.cat(tgt[k])
        if k == 'delta':
            p, t = p[m], t[m]
        if len(p) == 0:
            continue
        res[k] = float(pearson_rows(p, t).median())
    res['n'] = int(len(idx))
    return res


def run_one(arm, seed, a, dc, base_cfg, shared, sp, M, ppi, gv, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = V9Config(**{**vars(base_cfg), 'expr_encoder': arm})
    cfg.d_model, cfg.d_ff = a.d_model, 2 * a.d_model
    cfg.l_control, cfg.l_base, cfg.l_perturb = 1, 1, a.l_perturb

    full = LincsV9Dataset(dc, _shared=shared)
    tr = sp['train']
    if len(tr) > a.train_n:
        tr = np.sort(np.random.default_rng(seed).choice(tr, a.train_n, replace=False))
    train_ds = LincsV9Dataset(dc, indices=tr, _shared=shared)

    core = LincsV9(cfg, M, ppi, gv)
    if cfg.expr_encoder == 'binned':
        # bins fitted on TRAINING ROWS ONLY -- evaluation must not reach the quantiser
        rows = full.ds_to_l3[tr]
        rows = rows[rows >= 0][:20000]
        core.fit_bins(np.asarray(full.Xctl[np.sort(rows)], np.float32))
        if not core.bins_fitted:
            raise SystemExit('FATAL: quantiser did not fit; the expression input would be ignored.')
    core = core.to(device)
    n_par = sum(p.numel() for p in core.parameters())
    n_gpu = torch.cuda.device_count() if device == 'cuda' else 0
    # both arms get the SAME treatment, including data parallelism -- an A/B where one arm uses two GPUs
    # and the other uses one is not an A/B
    model = torch.nn.DataParallel(core) if n_gpu > 1 else core

    dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, collate_fn=collate_v9)
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    total = max(1, a.epochs * len(dl))
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.0, 1 - math.sqrt(max(0, s - 0.8 * total) /
                                                                        max(1, 0.2 * total))))
    Mn = torch.as_tensor(M, dtype=torch.float32)
    Mn = (Mn / Mn.sum(1, keepdim=True).clamp(min=1)).to(device)
    amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        for it, b in enumerate(dl):
            bd = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                out, aux = model(bd, return_aux=True)
                loss, _ = v9_loss(out, bd, cfg, Mn, aux)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if it % 100 == 0:
                print(f'    {arm} seed{seed} e{ep} it{it}/{len(dl)} loss {float(loss):.4f} '
                      f'{time.time() - t0:.0f}s', flush=True)
    out = {'params': n_par, 'seconds': round(time.time() - t0, 1)}
    for name, key in [('unseen_cell', 'test_coldcell'), ('unseen_compound', 'test_colddrug'),
                      ('unseen_both', 'test_coldboth')]:
        idx = sp[key][full.strength[sp[key]] >= dc.eval_min_strength]
        out[name] = evaluate(model, full, idx, device, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--train_n', type=int, default=24000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--d_model', type=int, default=96)
    ap.add_argument('--l_perturb', type=int, default=2)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='2-minute plumbing check, not a result')
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.epochs, a.train_n, a.d_model, a.l_perturb = 1, 1, 600, 48, 1

    dc = resolve_v9(V9DataConfig())      # on Kaggle the inputs are mounted flat under /kaggle/input
    dc.cache_in_ram = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}  seeds={a.seeds} epochs={a.epochs} train_n={a.train_n} d_model={a.d_model}',
          flush=True)

    R = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    M = np.load(R(dc.m_pathway_path))
    ppi = np.load(R(dc.ppi_v9_path))
    gv = np.load(R(dc.gene_vec_path))
    shared = LincsV9Dataset.load_shared_v9(dc)
    full = LincsV9Dataset(dc, _shared=shared)
    sp = build_splits(full, dc)
    if not a.smoke:
        check_inputs_v9(full, sp, dc)
    print('splits ' + str({k: len(v) for k, v in sp.items() if not k.startswith('_')}), flush=True)

    base = V9Config()
    res = {}
    for arm in ['raw', 'binned']:
        res[arm] = []
        for s in range(a.seeds):
            r = run_one(arm, s, a, dc, base, shared, sp, M, ppi, gv, device)
            res[arm].append(r)
            print(f'  [{arm} seed {s}] ' + '  '.join(
                f"{k} d={r[k].get('delta', float('nan')):.4f}/l5={r[k].get('l5', float('nan')):.4f}"
                for k in ['unseen_cell', 'unseen_compound', 'unseen_both'] if k in r), flush=True)

    print('\n' + '=' * 96)
    print('EXPRESSION ENCODER A/B -- mean [min, max] over %d seeds; a gap inside the seed range is NO '
          'DIFFERENCE' % a.seeds)
    print('=' * 96)
    summary = {}
    for tgt in ['delta', 'l5']:
        print(f'\n--- target: {tgt} ---')
        print('  %-16s %26s %26s %12s' % ('split', 'raw', 'binned', 'binned-raw'))
        for name in ['unseen_cell', 'unseen_compound', 'unseen_both']:
            v = {}
            for arm in ['raw', 'binned']:
                xs = [r[name][tgt] for r in res[arm] if name in r and tgt in r[name]]
                v[arm] = (float(np.mean(xs)), float(np.min(xs)), float(np.max(xs))) if xs else None
            if not (v['raw'] and v['binned']):
                continue
            d = v['binned'][0] - v['raw'][0]
            rng = max(v['raw'][2] - v['raw'][1], v['binned'][2] - v['binned'][1])
            verdict = 'inside seed range' if abs(d) <= rng else ''
            print('  %-16s %10.4f [%.4f, %.4f] %10.4f [%.4f, %.4f] %+11.4f  %s'
                  % (name, *v['raw'], *v['binned'], d, verdict))
            summary[f'{tgt}/{name}'] = dict(raw=v['raw'], binned=v['binned'], diff=round(d, 4),
                                            max_seed_range=round(rng, 4),
                                            distinguishable=bool(abs(d) > rng))
    WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(
        os.path.dirname(os.path.dirname(HERE)), 'model', 'results')
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, 'v9_encoder_ab.json')
    json.dump({'config': vars(a), 'runs': res, 'summary': summary}, open(out, 'w'), indent=2, default=str)
    print('\nwrote ' + out)


if __name__ == '__main__':
    main()
