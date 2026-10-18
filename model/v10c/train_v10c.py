"""v10-C training and one-shot scoring (RESULTS 100.4). Kaggle T4 x2 (DataParallel), never CPU in a GPU kernel.

    python model/v10c/train_v10c.py --split cold_cell_1 --dev --seed 0 [--chrom_off] [--d_model 384] --out DIR

Scores, on the held-out rows (dev cells with --dev, else the split's test rows), each read by the registered rule only:
  contrast r  per row: Pearson of (pred_i - mean pred of other-compound held-out plate-mates) with the same for the truth,
              rows with >= 2 such mates (the C1 definition, RESULTS 99.6)
  delta r     per row: Pearson of predicted and true delta (XPert's metric, their mean)
Predictions are saved (row ids, delta, the drug part) so every reading can be recomputed from the file.
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_v10c import V10CConfig, LincsV10C, v10c_loss, batch_contrast, row_pcc  # noqa: E402
from data_v10c import V10CData, plate_batches, find  # noqa: E402

ID_KEYS = ('sub_row', 'bench_row')
GUARD_EXIT = 3      # RESULTS 100.8: the exit code of a run stopped by the drug-path guard


def wsd(step, total, warm=0.03, decay=0.20):
    w, d = max(1, int(total * warm)), max(1, int(total * decay))
    if step < w:
        return (step + 1) / w
    if step >= total - d:
        return max(0.0, (total - step) / d)
    return 1.0


class DrugPathGuard:
    """RESULTS 100.8: stop a run whose dose scale s has saturated at zero, so its drug path cannot train (100.7). Reads
    training batches only. From step `start`, every `every` steps: failed iff the median of the last `window` per-batch
    medians of s is below `floor`."""

    def __init__(self, start=1000, every=500, window=200, floor=1e-4):
        self.start, self.every, self.window, self.floor = start, every, window, floor
        self.meds = []

    def update(self, batch_median):
        self.meds.append(float(batch_median))
        if len(self.meds) > self.window:
            del self.meds[0]

    def level(self):
        return float(np.median(self.meds)) if self.meds else float('nan')

    def failed(self, step):
        if step < self.start or step % self.every or len(self.meds) < self.window:
            return False
        return self.level() < self.floor


def heldout_scores(pred, true, plate, pert, cell, min_mates=2):
    """Per-row contrast r and delta r over the held-out set, plus per-cell means. numpy in, dict out."""
    P, T = torch.as_tensor(pred, dtype=torch.float64), torch.as_tensor(true, dtype=torch.float64)
    pl, pe = torch.as_tensor(plate), torch.as_tensor(pert)
    cp, ok = batch_contrast(P, pl, pe, min_mates)
    ct, _ = batch_contrast(T, pl, pe, min_mates)
    r_con = np.full(len(P), np.nan)
    if bool(ok.any()):
        r_con[ok.numpy()] = row_pcc(cp[ok], ct[ok]).numpy()
    r_del = row_pcc(P, T).numpy()
    out = {'n_rows': int(len(P)), 'n_contrast_rows': int(ok.sum()),
           'contrast_r_row_mean': float(np.nanmean(r_con)), 'delta_r_row_mean': float(np.mean(r_del))}
    per = {}
    for c in np.unique(cell):
        m = cell == c
        per[str(c)] = {'n': int(m.sum()), 'contrast_r': float(np.nanmean(r_con[m])) if np.isfinite(r_con[m]).any() else None,
                       'delta_r': float(np.mean(r_del[m]))}
    out['per_cell'] = per
    vals = [v['contrast_r'] for v in per.values() if v['contrast_r'] is not None]
    out['contrast_r_cell_mean'] = float(np.mean(vals)) if vals else None
    out['delta_r_cell_mean'] = float(np.mean([v['delta_r'] for v in per.values()]))
    return out, r_con, r_del


def predict_by_plate(model, D, rows, device, batch):
    """Predictions in plate-grouped batches (BatchNorm-free model; the order does not change the outputs)."""
    model.eval()
    pd, pg = np.zeros((len(rows), 978), np.float32), np.zeros((len(rows), 978), np.float32)
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            idx = rows[s:s + batch]
            with torch.amp.autocast('cuda', enabled=device == 'cuda'):
                out = model(D.batch(idx, device))
            pd[s:s + len(idx)] = out['delta'].float().cpu().numpy()
            pg[s:s + len(idx)] = out['drug'].float().cpu().numpy()
    model.train()
    return pd, pg


def permuted_chromatin(F, M, held_cells, kind, rng):
    """Review 086 C4 (reported, never read): chromatin for the held-out cells with
    'gene_shuffle': each cell's genes permuted (F and M together), keeping the cell's summary, breaking gene identity;
    'cell_swap':    each held-out cell given the next held-out cell's chromatin (a cyclic derangement of the sorted cells)."""
    F2, M2 = F.copy(), M.copy()
    cells = np.sort(np.asarray(held_cells))
    if kind == 'gene_shuffle':
        for c in cells:
            p = rng.permutation(F.shape[1])
            F2[c], M2[c] = F[c][p], M[c][p]
    elif kind == 'cell_swap':
        for c, d in zip(cells, np.roll(cells, 1)):
            F2[c], M2[c] = F[d], M[d]
    else:
        raise ValueError(kind)
    return F2, M2


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='cold_cell_1')
    ap.add_argument('--dev', action='store_true')
    ap.add_argument('--chrom_off', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--plate_cap', type=int, default=16)
    ap.add_argument('--budget_h', type=float, default=11.0)
    ap.add_argument('--timing_steps', type=int, default=0, help='>0: time this many steps, write timing.json, stop')
    ap.add_argument('--roots', nargs='+', default=['/kaggle/input'])
    ap.add_argument('--out', default='/kaggle/working')
    ap.add_argument('--cpu_ok', action='store_true', help='tests only')
    ap.add_argument('--limit_rows', type=int, default=0, help='tests only')
    ap.add_argument('--guard', action='store_true', help='RESULTS 100.8: stop if the dose scale saturates at zero')
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    t0 = time.time()
    if a.plate_cap > a.batch:
        raise SystemExit(f'FATAL: plate_cap {a.plate_cap} > batch {a.batch} would make batches larger than --batch')
    # review 088 C2: decide by device_count (CUDA_VISIBLE_DEVICES="" can leave is_available() True with 0 devices)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0 and not a.cpu_ok:
        raise SystemExit('FATAL: no CUDA device; not falling back to CPU')
    device = 'cuda' if n_gpu > 0 else 'cpu'
    if device == 'cuda':
        names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        if any('P100' in n for n in names) or n_gpu < 2:
            raise SystemExit(f'FATAL: need 2 non-P100 GPUs, got {names}')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    D = V10CData(a.roots, split=a.split, dev=a.dev, chrom_off=a.chrom_off)
    if a.limit_rows:
        rng0 = np.random.default_rng(0)
        D.tr = np.sort(rng0.choice(D.tr, min(a.limit_rows, len(D.tr)), replace=False))
        D.ev = np.sort(rng0.choice(D.ev, min(a.limit_rows, len(D.ev)), replace=False))
    print(f'split {a.split} dev={a.dev} dev_cells={D.dev_cells} chrom_off={a.chrom_off} train {len(D.tr)} '
          f'held-out {len(D.ev)} dropped {D.n_dropped} single-control rows {int(D.single_ctl.sum())}', flush=True)
    cfg = V10CConfig(d_model=a.d_model, n_lineage=D.L.shape[1])
    roots = a.roots
    M = np.load(find('M_pathway_v9.npy', roots))
    ppi = np.load(find('STRING_adj_978_v9.npy', roots))
    gv = np.load(find('gene_vectors_978.npy', roots))
    core = LincsV10C(cfg, M, ppi, gv)
    Xc, Xm = D.fit_rows()
    core.fit_bins(Xc, Xm)
    if not core.bins_fitted:
        raise SystemExit('FATAL: the quantiser did not fit')
    core = core.to(device)
    n_par = sum(p.numel() for p in core.parameters())
    print(f'params {n_par / 1e6:.2f}M', flush=True)
    model = torch.nn.DataParallel(core) if n_gpu > 1 else core
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    rng = np.random.default_rng(a.seed)
    n_batches = len(plate_batches(D.plate[D.tr], a.batch, np.random.default_rng(a.seed), a.plate_cap))
    total = a.epochs * n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: wsd(s, total))
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    step, hist = 0, []
    tag = f"{a.split}_{'dev' if a.dev else 'test'}_{'off' if a.chrom_off else 'on'}_d{a.d_model}_s{a.seed}"
    guard = DrugPathGuard() if a.guard else None
    recent = []
    t_train = time.time()
    for ep in range(a.epochs):
        batches = plate_batches(D.plate[D.tr], a.batch, rng, a.plate_cap)
        run, n_ok, s_sum = 0.0, 0, 0.0
        for it, bi in enumerate(batches):
            b = D.batch(D.tr[bi], device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device == 'cuda'):
                out = model(b)
                loss, parts = v10c_loss(out, b, cfg)
            s_med = float(out['dose_scale'].detach().float().median())
            s_sum += s_med
            if guard is not None:
                guard.update(s_med)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += float(loss.detach())
            n_ok += parts['n_contrast_rows']
            if not np.isfinite(float(loss.detach())):
                raise SystemExit(f'FATAL: non-finite loss at epoch {ep} step {it}')
            recent = (recent + [{'step': step, 'loss': float(loss.detach()), 'dose_scale_median': s_med, **parts}])[-20:]
            if guard is not None and guard.failed(step):
                fail = {'tag': tag, 'args': vars(a), 'status': 'drug path did not train (RESULTS 100.8)', 'step': step,
                        'epoch': ep, 'guard_level': guard.level(), 'guard': vars(guard) | {'meds': guard.meds[-50:]},
                        'recent': recent, 'hist': hist, 'wall_h': (time.time() - t0) / 3600}
                json.dump(fail, open(os.path.join(a.out, f'V10C_FAILED_{tag}.json'), 'w'), indent=1)
                print('FAILED ' + json.dumps({k: fail[k] for k in ('tag', 'step', 'guard_level')}), flush=True)
                raise SystemExit(GUARD_EXIT)
            if a.timing_steps and step == a.timing_steps:
                sec = (time.time() - t_train) / step
                rec = {'s_per_step': sec, 'steps_per_epoch': n_batches, 'projected_h': sec * total / 3600,
                       'gpus': [torch.cuda.get_device_name(i) for i in range(n_gpu)] if n_gpu else [], 'params': n_par,
                       'd_model': a.d_model, 'batch': a.batch, 'train_rows': int(len(D.tr))}
                json.dump(rec, open(os.path.join(a.out, 'timing.json'), 'w'), indent=1)
                print('TIMING ' + json.dumps(rec), flush=True)
                return rec
            if step == 50:
                proj = (time.time() - t_train) / 50 * total / 3600
                print(f'projected training {proj:.2f} h', flush=True)
                if proj > a.budget_h:
                    raise SystemExit(f'FATAL: projected {proj:.2f} h > budget {a.budget_h} h; refusing a run that '
                                     f'would be cut (a cut run is not read)')
            if it % 500 == 0:
                print(f'  e{ep} it{it}/{len(batches)} loss {float(loss):.4f} {parts}', flush=True)
        core.eval()                                                  # 100.8, reported only: no RNG is consumed
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device == 'cuda'):
            _, aux = core(b, return_aux=True)
        core.train()
        hist.append({'epoch': ep, 'loss': run / max(1, len(batches)), 'contrast_rows_per_step': n_ok / max(1, len(batches)),
                     'elapsed_h': (time.time() - t0) / 3600, 'dose_scale_batch_median_mean': s_sum / max(1, len(batches)),
                     'gamma_mean_last_batch': float(aux['gamma'].float().mean())})
        print('EPOCH ' + json.dumps(hist[-1]), flush=True)

    pd, pg = predict_by_plate(core, D, D.ev, device, 64)
    T = D.X[D.ev] - D.ctl[D.ctl_id[D.ev]]
    scores, r_con, r_del = heldout_scores(pd, T, D.plate[D.ev], D.pert_code[D.ev], D.cell[D.ev])
    permutations = {}
    if not a.chrom_off and len(np.unique(D.cell_row[D.ev])) > 1:
        F0, M0 = D.F, D.M
        try:
            for kind in ('gene_shuffle', 'cell_swap'):
                D.F, D.M = permuted_chromatin(F0, M0, np.unique(D.cell_row[D.ev]), kind, np.random.default_rng(1000 + a.seed))
                pk, _ = predict_by_plate(core, D, D.ev, device, 64)
                sk, _, _ = heldout_scores(pk, T, D.plate[D.ev], D.pert_code[D.ev], D.cell[D.ev])
                permutations[kind] = {k: sk[k] for k in ('contrast_r_row_mean', 'contrast_r_cell_mean', 'delta_r_row_mean')}
                permutations[kind]['per_cell_contrast_r'] = {c: v['contrast_r'] for c, v in sk['per_cell'].items()}
        finally:
            D.F, D.M = F0, M0
    pred_path = os.path.join(a.out, f'v10c_pred_{tag}.npz')
    np.savez_compressed(pred_path, sub_row=D.sub_row[D.ev], bench_row=D.bench_row[D.ev], delta=pd, drug=pg,
                        r_contrast=r_con, r_delta=r_del)
    h = hashlib.sha1(open(pred_path, 'rb').read()).hexdigest()
    rec = {'tag': tag, 'args': vars(a), 'dev_cells': D.dev_cells, 'n_train': int(len(D.tr)), 'n_heldout': int(len(D.ev)),
           'dropped': D.n_dropped, 'params': n_par, 'hist': hist, 'scores': scores, 'pred_sha1': h,
           'reported_permutations': permutations, 'single_control_heldout_rows': int(D.single_ctl[D.ev].sum()),
           'guard': bool(a.guard), 'wall_h': (time.time() - t0) / 3600}
    json.dump(rec, open(os.path.join(a.out, f'V10C_COMPLETE_{tag}.json'), 'w'), indent=1)
    torch.save({'model': core.state_dict(), 'cfg': {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')
                                                     and not callable(getattr(cfg, k))}},
               os.path.join(a.out, f'v10c_{tag}.pt'))
    print('DONE ' + json.dumps({k: rec['scores'][k] for k in ('contrast_r_row_mean', 'delta_r_row_mean')}), flush=True)
    return rec


if __name__ == '__main__':
    main()
