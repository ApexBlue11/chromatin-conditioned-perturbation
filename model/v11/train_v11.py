"""v11 training and one-shot scoring (RESULTS 102 / 102.1). Kaggle T4 x2 (DataParallel), never CPU in a GPU kernel.

    python model/v11/train_v11.py --split cold_cell_1 --dev --seed 0 --arm full --guard --out DIR

The loop is train_v10c's (same recipe, optimiser, schedule, plate batches, guard, scorer), with V11Data / LincsV11 and
the structure head's BCE (lambda 0.1, off in the -targets arm). Scores as train_v10c (contrast r, delta r, per cell).
Also written, reported and never read by R1 / R2: the held-out rows of unseen annotated compounds scored with their
known targets as input (102 item 2's second setting), and the permutations of 102 item 5 (CCLE cell-swap, chromatin
gene-shuffle and cell-swap, targets shuffled across compounds).
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
from model_v10c import V10CConfig  # noqa: E402
from data_v10c import find, plate_batches  # noqa: E402
from train_v10c import wsd, DrugPathGuard, GUARD_EXIT, heldout_scores, predict_by_plate  # noqa: E402
from model_v11 import LincsV11, v11_loss  # noqa: E402
from data_v11 import V11Data, ARMS  # noqa: E402


def guard_stop(a, tag, guard, step, ep, recent, hist, t0):
    """RESULTS 100.8: record a run stopped by the drug-path guard and exit with GUARD_EXIT (nothing scored)."""
    fail = {'tag': tag, 'args': vars(a), 'status': 'drug path did not train (RESULTS 100.8)', 'step': step, 'epoch': ep,
            'guard_level': guard.level(), 'guard': vars(guard) | {'meds': guard.meds[-50:]}, 'recent': recent,
            'hist': hist, 'wall_h': (time.time() - t0) / 3600}
    json.dump(fail, open(os.path.join(a.out, f'V11_FAILED_{tag}.json'), 'w'), indent=1)
    print('FAILED ' + json.dumps({k: fail[k] for k in ('tag', 'step', 'guard_level')}), flush=True)
    raise SystemExit(GUARD_EXIT)


def derange(n, rng):
    """A permutation with no fixed point (n >= 2)."""
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def fg_permutations(core, D, device, T, seed):
    """102 item 5, reported only: FG's inputs permuted for the held-out cells / compounds, the trained model unchanged."""
    out = {}
    tab = core.fg.tables
    held = np.unique(D.cell_row[D.ev])
    rng = np.random.default_rng(2000 + seed)
    F0, M0, E0, Em0 = D.fg_F, D.fg_M, D.fg_E, D.fg_Em
    kinds = []
    if len(held) > 1:
        kinds += [('ccle_cell_swap', 'ccle'), ('chrom_cell_swap', 'chrom')]
    kinds += [('chrom_gene_shuffle', 'chrom')]
    try:
        for kind, what in kinds:
            F, M, E, Em = F0.copy(), M0.copy(), E0.copy(), Em0.copy()
            cells = np.sort(held)
            if kind.endswith('cell_swap'):
                for c, d in zip(cells, np.roll(cells, 1)):
                    if what == 'ccle':
                        E[c], Em[c] = E0[d], Em0[d]
                    else:
                        F[c], M[c] = F0[d], M0[d]
            else:
                for c in cells:
                    p = rng.permutation(F.shape[1])
                    F[c], M[c] = F0[c][p], M0[c][p]
            tab.replace(F=F, M=M, E=E, Em=Em)
            pk, _ = predict_by_plate(core, D, D.ev, device, 64)
            sk, _, _ = heldout_scores(pk, T, D.plate[D.ev], D.pert_code[D.ev], D.cell[D.ev])
            out[kind] = {k: sk[k] for k in ('contrast_r_row_mean', 'contrast_r_cell_mean', 'delta_r_row_mean')}
    finally:
        tab.replace(F=F0, M=M0, E=E0, Em=Em0)
    if core.fg.targets_on:
        saved = (D.tgt_node, D.tgt_sign, D.tgt_count, D.known_input)
        try:
            k = np.flatnonzero(D.known_input)
            if len(k) > 1:
                p = k[derange(len(k), rng)]
                D.tgt_node, D.tgt_sign, D.tgt_count = D.tgt_node.copy(), D.tgt_sign.copy(), D.tgt_count.copy()
                D.tgt_node[k], D.tgt_sign[k], D.tgt_count[k] = saved[0][p], saved[1][p], saved[2][p]
                pk, _ = predict_by_plate(core, D, D.ev, device, 64)
                sk, _, _ = heldout_scores(pk, T, D.plate[D.ev], D.pert_code[D.ev], D.cell[D.ev])
                out['targets_shuffled'] = {kk: sk[kk] for kk in ('contrast_r_row_mean', 'contrast_r_cell_mean',
                                                                 'delta_r_row_mean')}
        finally:
            D.tgt_node, D.tgt_sign, D.tgt_count, D.known_input = saved
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='cold_cell_1')
    ap.add_argument('--dev', action='store_true')
    ap.add_argument('--arm', default='full', choices=ARMS)
    ap.add_argument('--lam', type=float, default=0.1)
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

    D = V11Data(a.roots, split=a.split, dev=a.dev, arm=a.arm)
    if a.limit_rows:
        rng0 = np.random.default_rng(0)
        D.tr = np.sort(rng0.choice(D.tr, min(a.limit_rows, len(D.tr)), replace=False))
        D.ev = np.sort(rng0.choice(D.ev, min(a.limit_rows, len(D.ev)), replace=False))
    targets_on = a.arm != 'no_targets'
    st = {s: int((D.status[D.pert_code[D.tr]] == i).sum()) for i, s in enumerate(('known', 'probe', 'unseen_annotated',
                                                                                     'unannotated'))}
    print(f'split {a.split} dev={a.dev} dev_cells={D.dev_cells} arm={a.arm} train {len(D.tr)} held-out {len(D.ev)} '
          f'dropped {D.n_dropped} vocab {len(D.vocab)} train rows by target status {st}', flush=True)
    cfg = V10CConfig(d_model=a.d_model, n_lineage=D.L.shape[1])
    roots = a.roots
    M = np.load(find('M_pathway_v9.npy', roots))
    ppi = np.load(find('STRING_adj_978_v9.npy', roots))
    gv = np.load(find('gene_vectors_978.npy', roots))
    core = LincsV11(cfg, M, ppi, gv, D.fg_inputs())
    Xc, Xm = D.fit_rows()
    core.fit_bins(Xc, Xm)
    if not core.bins_fitted:
        raise SystemExit('FATAL: the quantiser did not fit')
    core = core.to(device)
    n_par = sum(p.numel() for p in core.parameters())
    n_fg = sum(p.numel() for p in core.fg.parameters()) + sum(p.numel() for p in core.target_head.parameters())
    print(f'params {n_par / 1e6:.2f}M (FG + head {n_fg / 1e6:.3f}M)', flush=True)
    model = torch.nn.DataParallel(core) if n_gpu > 1 else core
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    rng = np.random.default_rng(a.seed)
    n_batches = len(plate_batches(D.plate[D.tr], a.batch, np.random.default_rng(a.seed), a.plate_cap))
    total = a.epochs * n_batches
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: wsd(s, total))
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    step, hist = 0, []
    tag = f"{a.split}_{'dev' if a.dev else 'test'}_v11{a.arm}_d{a.d_model}_s{a.seed}"
    guard = DrugPathGuard() if a.guard else None
    recent, cells_per_batch, cells_per_replica = [], [], []
    t_train = time.time()
    for ep in range(a.epochs):
        batches = plate_batches(D.plate[D.tr], a.batch, rng, a.plate_cap)
        run, n_ok, s_sum, bce_sum, n_bce, gate_sum = 0.0, 0, 0.0, 0.0, 0, 0.0
        for it, bi in enumerate(batches):
            rows = D.tr[bi]
            cr = D.cell_row[rows]
            cells_per_batch.append(len(np.unique(cr)))
            if n_gpu > 1:
                cells_per_replica.append(max(len(np.unique(c)) for c in np.array_split(cr, n_gpu) if len(c)))
            b = D.batch(rows, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device == 'cuda'):
                out = model(b)
                loss, parts = v11_loss(out, b, cfg, a.lam, targets_on)
            s_med = float(out['dose_scale'].detach().float().median())
            s_sum += s_med
            gate_sum += float(out['fg_gate_mean'].detach().float().mean())
            if 'target_bce' in parts:
                bce_sum += parts['target_bce']
                n_bce += 1
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
                guard_stop(a, tag, guard, step, ep, recent, hist, t0)
            if a.timing_steps and step == a.timing_steps:
                sec = (time.time() - t_train) / step
                cb = np.array(cells_per_batch)
                rec = {'s_per_step': sec, 'steps_per_epoch': n_batches, 'projected_h': sec * total / 3600,
                       'gpus': [torch.cuda.get_device_name(i) for i in range(n_gpu)] if n_gpu else [], 'params': n_par,
                       'params_fg_and_head': n_fg, 'd_model': a.d_model, 'batch': a.batch, 'train_rows': int(len(D.tr)),
                       'arm': a.arm, 'cells_per_batch': {'mean': float(cb.mean()), 'max': int(cb.max()),
                                                         'p95': float(np.percentile(cb, 95))},
                       'cells_per_replica_max': int(max(cells_per_replica)) if cells_per_replica else None,
                       'max_mem_gb': [torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(n_gpu)]}
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
        core.eval()                                                  # reported only: no RNG is consumed
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device == 'cuda'):
            _, aux = core(b, return_aux=True)
        core.train()
        nb = max(1, len(batches))
        hist.append({'epoch': ep, 'loss': run / nb, 'contrast_rows_per_step': n_ok / nb,
                     'elapsed_h': (time.time() - t0) / 3600, 'dose_scale_batch_median_mean': s_sum / nb,
                     'gamma_mean_last_batch': float(aux['gamma'].float().mean()), 'fg_gate_mean': gate_sum / nb,
                     'target_bce_mean': bce_sum / max(1, n_bce),
                     'fg_readout_rms_last_batch': float(aux['fg_readout'].float().pow(2).mean().sqrt()),
                     'cells_per_batch_mean': float(np.mean(cells_per_batch[-nb:]))})
        print('EPOCH ' + json.dumps(hist[-1]), flush=True)

    if guard is not None and guard.failed(step, final=True):
        guard_stop(a, tag, guard, step, a.epochs - 1, recent, hist, t0)
    pd, pg = predict_by_plate(core, D, D.ev, device, 64)
    T = D.X[D.ev] - D.ctl[D.ctl_id[D.ev]]
    scores, r_con, r_del = heldout_scores(pd, T, D.plate[D.ev], D.pert_code[D.ev], D.cell[D.ev])
    # 102 item 2's second setting (reported): unseen annotated compounds with their known targets as input
    status_ev = D.status[D.pert_code[D.ev]]
    known_setting = {'n_rows': int((status_ev == 2).sum())}
    pd2 = None
    if targets_on and known_setting['n_rows']:
        saved = D.known_input
        try:
            D.known_input = saved | (D.status == 2)
            pd2, _ = predict_by_plate(core, D, D.ev, device, 64)
        finally:
            D.known_input = saved
        s2, r_con2, r_del2 = heldout_scores(pd2, T, D.plate[D.ev], D.pert_code[D.ev], D.cell[D.ev])
        m = status_ev == 2
        known_setting.update({'contrast_r_unseen_annotated_predicted': float(np.nanmean(r_con[m])),
                              'contrast_r_unseen_annotated_known': float(np.nanmean(r_con2[m])),
                              'delta_r_unseen_annotated_predicted': float(np.mean(r_del[m])),
                              'delta_r_unseen_annotated_known': float(np.mean(r_del2[m]))})
    permutations = fg_permutations(core, D, device, T, a.seed)
    pred_path = os.path.join(a.out, f'v11_pred_{tag}.npz')
    np.savez_compressed(pred_path, sub_row=D.sub_row[D.ev], bench_row=D.bench_row[D.ev], delta=pd, drug=pg,
                        r_contrast=r_con, r_delta=r_del, target_status=status_ev,
                        **({'delta_known_setting': pd2} if pd2 is not None else {}))
    h = hashlib.sha1(open(pred_path, 'rb').read()).hexdigest()
    rec = {'tag': tag, 'args': vars(a), 'dev_cells': D.dev_cells, 'n_train': int(len(D.tr)), 'n_heldout': int(len(D.ev)),
           'dropped': D.n_dropped, 'params': n_par, 'params_fg_and_head': n_fg, 'hist': hist, 'scores': scores,
           'pred_sha1': h, 'reported_permutations': permutations, 'reported_known_target_setting': known_setting,
           'target_status_train_rows': st, 'vocab': int(len(D.vocab)), 's_scale': D.s_scale.tolist(),
           'single_control_heldout_rows': int(D.single_ctl[D.ev].sum()), 'guard': bool(a.guard),
           'wall_h': (time.time() - t0) / 3600}
    json.dump(rec, open(os.path.join(a.out, f'V11_COMPLETE_{tag}.json'), 'w'), indent=1)
    torch.save({'model': core.state_dict(), 'cfg': {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')
                                                     and not callable(getattr(cfg, k))}},
               os.path.join(a.out, f'v11_{tag}.pt'))
    print('DONE ' + json.dumps({k: rec['scores'][k] for k in ('contrast_r_row_mean', 'delta_r_row_mean')}), flush=True)
    return rec


if __name__ == '__main__':
    main()
