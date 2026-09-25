# -*- coding: utf-8 -*-
"""
THE SOTA ARM THE GATE ADMITS: v9 trained on XPert's own data, their split, their metric.

V9_HANDOFF §D: "any SOTA comparison | same data level + convention + split, else report non-comparability."
Nothing else in this project satisfies that. Their reported numbers cannot be set against ours directly --
§28 shows their splits leave 100 % of test cells and 89.4 % of test (cell, compound) pairs in the training
set, so the tasks differ by more than any model does. Running OUR model on THEIR benchmark removes every
difference except the model.

They report two numbers, and both are reproduced here:
    Pearson      on absolute Level-3 expression      (their `metrics['Pearson']`,     reported 0.9804)
    Pearson_deg  on the delta                        (their `metrics['Pearson_deg']`, reported 0.8440)
plus the two nulls without which neither is interpretable: copy-the-control (0.9200 on their own released
predictions) and the drug mean.

Honest limits, stated rather than buried:
  * 5.4 % of the rows their splits touch use compounds we have no UniMol/ECFP4 features for and are dropped;
  * 143 of their 217 cell lines have no chromatin or lineage in our data, so those rows carry the UNKNOWN
    lineage and zero chromatin reliability -- v9 handles that by construction, but it is less input than
    their model gets;
  * their numbers come from their trained weights on their own preprocessing; ours come from this code.

    python model/v9/xpert_arm.py --split split_lung_1 --seeds 3
"""
import os, sys, json, time, math, argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v6'))
sys.path.insert(0, os.path.dirname(HERE))

from config_v9 import V9Config, V9DataConfig
from model_v9 import LincsV9, v9_loss

BUNDLE = 'xpert_splits.npz'


def find(name, roots):
    import glob
    for r in roots:
        hit = glob.glob(os.path.join(r, '**', name), recursive=True)
        if hit:
            return hit[0]
    raise SystemExit(f'FATAL: {name} not found under {roots}')


def their_pearson(a, b):
    """XPert's metric (metrics.py:pearson) is the MEAN of per-row Pearson, not the median. Reporting our
    median against their mean would flatter us: the median discards the left tail of badly-predicted rows,
    and on these data the two differ by ~0.02-0.05. Both are reported; the MEAN is the comparable one."""
    return float(np.nanmean(pearson_rows(a, b)))


def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    n = (a * b).sum(1)
    d = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)



def carve_dev(rows_per_cell, K, seed, min_rows=200, max_rows=2000):
    pool = [c for c, count in rows_per_cell.items() if min_rows <= count <= max_rows]
    pool.sort()
    if len(pool) < K:
        raise ValueError(f"Pool size {len(pool)} is less than K {K}")
    if K == 0:
        return []
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(pool), K, replace=False)
    return sorted([pool[i] for i in chosen])

def seed_devices(seed, n_gpu, mode):
    """RESULTS 85.5 (review 021 C1). `torch.manual_seed` seeds EVERY CUDA device alike, and this loop only ever builds
    full batches, so under DataParallel the replicas' dropout and stochastic-depth masks would be identical for the
    whole run (rows j and j + batch/2 share them). 'distinct' reseeds device k >= 1 with seed + 1000 * k; device 0 keeps
    `seed`, so a one-GPU run is unchanged. 'lockstep' is the behaviour before 85.5. Returns the seed of each device."""
    seeds = [seed] * n_gpu
    if mode == 'distinct':
        for k in range(1, n_gpu):
            seeds[k] = seed + 1000 * k
            torch.cuda.default_generators[k].manual_seed(seeds[k])
    return seeds


def cuda_states_equal(n_gpu):
    """True iff every device's CUDA generator state is byte-equal to device 0's (None on fewer than 2 devices)."""
    if n_gpu < 2:
        return None
    s0 = torch.cuda.get_rng_state(0)
    return all(torch.equal(s0, torch.cuda.get_rng_state(k)) for k in range(1, n_gpu))


class XPertData:
    """Their rows, presented in the v9 batch format."""

    def __init__(self, npz, roots, split, ablate_epi=False, dev_args=None):
        z = np.load(npz, allow_pickle=True)
        lab = z[f'split_{split}']
        self.tr = np.flatnonzero(lab == 'train')
        self.te = np.flatnonzero(lab == 'test')
        if len(self.te) == 0:
            raise SystemExit(f'FATAL: split {split} has no test rows in the bundle')
        keep = np.concatenate([self.tr, self.te])
        self.X = np.asarray(z['X'][keep], np.float32)
        self.C = np.asarray(z['X_ctl'][keep], np.float32)
        self.pert = z['meta_pert_id'][keep]
        self.cell = z['meta_cell'][keep]
        self.dose = z['meta_dose'][keep].astype(np.float32)
        self.time = z['meta_time'][keep].astype(np.float32)
        # carried so a saved prediction can be aligned back to THEIR h5ad row; comparing two 13,766-row
        # arrays that were built in different orders would silently pair row i with row j
        self.row_index = (z['row_index'][keep] if 'row_index' in z.files
                          else np.asarray(keep, np.int64))
        n_tr = len(self.tr)
        self.tr = np.arange(n_tr)
        self.te = np.arange(n_tr, len(keep))

        # --- our drug features, keyed by their pert_id ---
        di = json.load(open(find('drug_feature_index.json', roots)))
        # Rows whose compound we cannot featurise are DROPPED, counted and reported -- never silently
        # imputed. The tissue bundle was pre-filtered upstream; this one is not, because dropping rows
        # before the split would change which rows their checkpoint is scored on.
        have = np.array([p in di for p in self.pert])
        if not have.all():
            keep2 = np.flatnonzero(have)
            remap = -np.ones(len(have), np.int64)
            remap[keep2] = np.arange(len(keep2))
            n_tr0, n_te0 = len(self.tr), len(self.te)
            self.tr = remap[self.tr[have[self.tr]]]
            self.te = remap[self.te[have[self.te]]]
            for attr in ['X', 'C', 'pert', 'cell', 'dose', 'time', 'row_index']:
                setattr(self, attr, getattr(self, attr)[keep2])
            self.n_dropped_train = n_tr0 - len(self.tr)
            self.n_dropped_test = n_te0 - len(self.te)
            print('dropped %d train and %d test rows whose compound we cannot featurise '
                  '(%.2f%% of test); the comparison runs on the rows BOTH models can score'
                  % (self.n_dropped_train, self.n_dropped_test,
                     100 * self.n_dropped_test / max(1, n_te0)), flush=True)
        else:
            self.n_dropped_train = self.n_dropped_test = 0

        self.original_test_row_indices = set(self.row_index[self.te].tolist())
        
        self.dev_info = None
        if dev_args and dev_args.dev_cells > 0:
            import hashlib
            cells_tr = self.cell[self.tr]
            unique_c, counts = np.unique(cells_tr, return_counts=True)
            rows_per_cell = dict(zip(unique_c, counts))
            dev_cells = carve_dev(rows_per_cell, dev_args.dev_cells, dev_args.dev_seed,
                                  min_rows=dev_args.dev_min_rows, max_rows=dev_args.dev_max_rows)
            
            dev_cells_set = set(dev_cells)
            is_dev = np.array([c in dev_cells_set for c in cells_tr])
            new_tr = self.tr[~is_dev]
            new_te = self.tr[is_dev]
            
            # eligible pool for printing
            pool = sorted([c for c, cnt in rows_per_cell.items() 
                           if dev_args.dev_min_rows <= cnt <= dev_args.dev_max_rows])
            pool_dict = {c: int(rows_per_cell[c]) for c in pool}
            
            dev_row_indices = self.row_index[new_te].astype(np.int64).copy()
            dev_row_indices.sort()
            sha = hashlib.sha1(dev_row_indices.tobytes()).hexdigest()
            
            self.dev_info = {
                "eligible_pool": pool_dict,
                "dev_cells": dev_cells,
                "rows_per_dev_cell": {c: int(rows_per_cell[c]) for c in dev_cells},
                "total_dev_rows": len(new_te),
                "total_remaining_training_rows": len(new_tr),
                "dev_row_index_sha1": sha
            }
            print(f"DEV CELLS {json.dumps(self.dev_info)}", flush=True)
            
            # keep only new_tr and new_te
            keep_indices = np.concatenate([new_tr, new_te])
            for attr in ['X', 'C', 'pert', 'cell', 'dose', 'time', 'row_index']:
                setattr(self, attr, getattr(self, attr)[keep_indices])
                
            self.tr = np.arange(len(new_tr))
            self.te = np.arange(len(new_tr), len(keep_indices))
            self.test_rows_dropped = True
            self.dev_row_indices = set(self.row_index[self.te].tolist())
        else:
            self.test_rows_dropped = False


        rows = np.array([di[p] for p in self.pert])
        desc = np.load(find('drug_descriptors.npy', roots)).astype(np.float32)
        desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)
        fp = np.load(find('drug_fingerprints.npy', roots)).astype(np.float32)
        ucls = np.load(find('drug_unimol.npy', roots)).astype(np.float32)
        self.u = np.concatenate([ucls[rows], desc[rows], fp[rows]], 1)
        self.atom_reprs = np.load(find('drug_atom_reprs.npy', roots), mmap_mode='r')
        self.atom_off = np.load(find('drug_atom_offsets.npy', roots))
        self.drug_row = rows

        # --- chromatin / lineage where we have the cell line, neutral where we do not ---
        cidx = json.load(open(find('lincs_cell_index.json', roots)))
        cidx = cidx.get('cell_id_to_row', cidx)
        E = np.load(find('E_final.npy', roots)).astype(np.float32)
        Em = np.load(find('E_final_mask.npy', roots))
        for k in range(E.shape[2]):
            has = Em[:, :, k].any(1)
            for c in np.where(has)[0]:
                v = E[c, :, k]
                E[c, :, k] = (v - v.mean()) / (v.std() + 1e-6)
        lin = np.load(find('cell_lineage.npy', roots)).astype(np.float32)
        G = self.X.shape[1]
        self.E = np.zeros((len(self.X), G, E.shape[2]), np.float32)
        self.r = np.zeros((len(self.X), G), np.float32)
        self.Em = np.zeros((len(self.X), G, Em.shape[2]), bool)
        self.ctx = np.zeros((len(self.X), lin.shape[1]), np.float32)
        self.ctx[:, 0] = 1.0
        known = 0
        for i, c in enumerate(self.cell):
            j = cidx.get(str(c))
            if j is None:
                continue
            known += 1
            self.E[i] = E[j]
            self.r[i] = Em[j].any(-1).astype(np.float32)
            self.Em[i] = Em[j]
            self.ctx[i] = lin[j]
        self.known_cell_frac = known / len(self.cell)

        if ablate_epi:
            # Mean-ablate BOTH the chromatin values and the track-availability mask. Ablating only E
            # would leave r varying by cell, which still carries cell identity -- the ablation has to
            # remove the information, not just the numbers.
            m_tr = np.zeros(len(self.cell), bool)
            m_tr[self.tr] = True
            self.E[:] = self.E[m_tr].mean(0)
            self.r[:] = self.r[m_tr].mean(0)
            print('CHROMATIN MEAN-ABLATED: every row now carries the training-mean chromatin and track '
                  'mask, so the branch is intact but cell-specific chromatin information is gone. '
                  'Lineage (cell_ctx) is untouched -- this isolates chromatin, not cell identity.',
                  flush=True)


        if getattr(self, 'test_rows_dropped', False):
            tr_ri = set(self.row_index[self.tr].tolist())
            assert not tr_ri.intersection(self.original_test_row_indices)
            assert not tr_ri.intersection(self.dev_row_indices)
        ld = np.log10(np.clip(self.dose, 1e-4, None))
        self.dose_n = ((ld - ld[self.tr].mean()) / (ld[self.tr].std() + 1e-6)).astype(np.float32)
        self.time_n = ((self.time - self.time[self.tr].mean()) /
                       (self.time[self.tr].std() + 1e-6)).astype(np.float32)

        # per-cell aggregate control, from TRAINING rows only
        self.cell_mean = {}
        for c in np.unique(self.cell):
            m = (self.cell == c)
            m_tr = np.zeros(len(self.cell), bool)
            m_tr[self.tr] = True
            sel = m & m_tr
            self.cell_mean[c] = self.C[sel].mean(0) if sel.any() else self.C[m].mean(0)
        self.x_cell = np.stack([self.cell_mean[c] for c in self.cell])

    def batch(self, idx, device, max_atoms=96):
        d = self.drug_row[idx]
        n = min(max_atoms, max(1, int(max(self.atom_off[k + 1] - self.atom_off[k] for k in d))))
        atoms = np.zeros((len(idx), n, 512), np.float32)
        amask = np.zeros((len(idx), n), bool)
        for b, k in enumerate(d):
            a0, a1 = int(self.atom_off[k]), int(self.atom_off[k + 1])
            m = min(n, a1 - a0)
            atoms[b, :m] = np.asarray(self.atom_reprs[a0:a0 + m], np.float32)
            amask[b, :m] = True
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(device)
        return {'x_ctl': t(self.C[idx]), 'x_cell': t(self.x_cell[idx]),
                'y_abs': t(self.X[idx]), 'y_delta': t(self.X[idx] - self.C[idx]),
                'y_l5': t(np.zeros((len(idx), self.X.shape[1]), np.float32)),
                'm_l3': torch.ones(len(idx), dtype=torch.bool, device=device),
                'm_l5': torch.zeros(len(idx), dtype=torch.bool, device=device),
                'E': t(self.E[idx]), 'r': t(self.r[idx]), 'E_mask': t(self.Em[idx].astype(np.float32)), 'cell_ctx': t(self.ctx[idx]),
                'atoms': t(atoms), 'atom_mask': torch.as_tensor(amask).to(device),
                'u_feats': t(self.u[idx]), 'dose': t(self.dose_n[idx]), 'time': t(self.time_n[idx])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='split_lung_1')
    ap.add_argument('--bundle', default=BUNDLE,
                    help='xpert_splits.npz (tissue splits) or xpert_mdmt_splits.npz (their main benchmark)')
    ap.add_argument('--ablate_epi', action='store_true',
                    help='replace the chromatin input with its TRAINING mean for every row, so the '
                         'architecture, parameter count and reliability channel are unchanged and only '
                         'the CELL-SPECIFIC chromatin information is removed. This is the '
                         'ablate-to-the-mean convention of this project, applied at training time.')
    ap.add_argument('--save_ckpt', default=None,
                    help='write the trained weights, so an inference-time ablation can be run later')
    ap.add_argument('--save_pred', default=None,
                    help='write per-row predictions, so a PAIRED comparison against their checkpoint is '
                         'possible instead of two independently-computed summary numbers')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed_start', type=int, default=0,
                    help='first seed index; lets one seed per Kaggle session so a 9 h limit cannot '
                         'discard a completed seed along with an unfinished one')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--expr_encoder', default='binned')
    ap.add_argument('--limit_train', type=int, default=0,
                    help='cap training rows -- for a plumbing check, NOT a result')
    ap.add_argument('--dev_cells', type=int, default=0)
    ap.add_argument('--dp_seed_mode', choices=['distinct', 'lockstep'], default='distinct',
                    help='RESULTS 85.5: per-device CUDA seeds under DataParallel; lockstep = the pre-85.5 behaviour')
    ap.add_argument('--dev_seed', type=int, default=0)
    ap.add_argument('--dev_min_rows', type=int, default=200)
    ap.add_argument('--dev_max_rows', type=int, default=2000)
    # --- §85.7 candidate flags ---
    ap.add_argument('--no_atoms', action='store_true',
                    help='C1: drug sequence is [global] only; atom tokens reach nothing')
    ap.add_argument('--l_control', type=int, default=2,
                    help='C2: number of _GeneBlock layers in ControlEncoder (default 2; 0 = no blocks)')
    ap.add_argument('--listnet_w', type=float, default=0.0,
                    help='C3: symmetric ListNet ranking loss weight (0 = off)')
    ap.add_argument('--deg_adapt_k', type=int, default=0,
                    help='C4: adaptive DE weighting, top-K genes by |y_delta| (0 = off)')
    ap.add_argument('--sign_head_w', type=float, default=0.0,
                    help='C6: sign-prediction BCE head weight (0 = off)')
    ap.add_argument('--post_pathway', action='store_true',
                    help='C8b: second NamedPathwayReadout after last perturb block')
    ap.add_argument('--chromatin_edges', action='store_true',
                    help='C7: Chromatin-gated union-graph edges')
    ap.add_argument('--union_edges', action='store_true',
                    help='C7u: Ungated union-graph edges')
    a = ap.parse_args()

    roots = ['/kaggle/input', os.path.join(r'C:\Projects\LINCS'), os.path.join(r'C:\Projects\LINCS',
                                                                              'external')]
    if os.environ.get('LINCS_DATA_ROOT'):                       # Lightning (RESULTS 85.5)
        roots.insert(0, os.environ['LINCS_DATA_ROOT'])
    roots = [r for r in roots if os.path.isdir(r)]
    npz = find(a.bundle, roots)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # RESULTS 85.5 rule 4: record the precision class on every platform; on request (Lightning) assert it.
    platform = {'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'torch': torch.__version__, 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'NVIDIA_TF32_OVERRIDE': os.environ.get('NVIDIA_TF32_OVERRIDE')}
    if dev == 'cuda':
        platform.update(flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                        capability=list(torch.cuda.get_device_capability(0)))
    print('PLATFORM ' + json.dumps(platform), flush=True)
    if os.environ.get('LINCS_ASSERT_NO_TF32'):
        if platform['matmul_allow_tf32'] or platform['float32_matmul_precision'] != 'highest' or \
                platform['NVIDIA_TF32_OVERRIDE'] is not None:
            raise SystemExit('FATAL: TF32 or a non-highest fp32 matmul precision is active (RESULTS 85.5 rule 4).')
    if dev == 'cuda' and any('P100' in torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())):
        raise SystemExit('FATAL: P100 assigned; Kaggle torch has no sm_60 kernels.')
    D = XPertData(npz, roots, a.split, ablate_epi=a.ablate_epi, dev_args=a)
    if a.limit_train:
        D.tr = D.tr[:a.limit_train]
        D.te = D.te[:min(len(D.te), 400)]
        print(f'LIMITED to {len(D.tr)} train / {len(D.te)} test rows -- plumbing check, not a result')
    print(f'{a.split}: train {len(D.tr)} test {len(D.te)} | rows with a cell line we know: '
          f'{100 * D.known_cell_frac:.1f}% | device {dev}', flush=True)

    M = np.load(find('M_pathway_v9.npy', roots))
    ppi = np.load(find('STRING_adj_978_v9.npy', roots))
    gv = np.load(find('gene_vectors_978.npy', roots))

    # nulls first: an absolute number is uninterpretable without them
    Xte, Cte = D.X[D.te], D.C[D.te]
    nulls = {'copy_ctl_abs': round(their_pearson(Cte, Xte), 4),
             'copy_ctl_abs_median': round(float(np.nanmedian(pearson_rows(Cte, Xte))), 4),
             'zero_delta': 0.0}
    dtr = D.X[D.tr] - D.C[D.tr]
    gm = dtr.mean(0)
    per = {}
    for p in np.unique(D.pert[D.tr]):
        per[p] = dtr[D.pert[D.tr] == p].mean(0)
    pred = np.stack([per.get(p, gm) for p in D.pert[D.te]])
    nulls['mean_drug_delta'] = round(their_pearson(pred, Xte - Cte), 4)
    print(f'nulls on THEIR test rows: {nulls}', flush=True)

    runs = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_gpu = torch.cuda.device_count() if dev == 'cuda' else 0
        device_seeds = seed_devices(seed, n_gpu, a.dp_seed_mode)
        rng_equal_by_epoch = []
        cfg = V9Config()
        cfg.d_model, cfg.d_ff, cfg.expr_encoder = a.d_model, 4 * a.d_model, a.expr_encoder
        cfg.n_pathways = M.shape[0]
        cfg.predict_l5 = False
        # §85.7 candidate flags
        cfg.no_atoms = a.no_atoms
        cfg.l_control = a.l_control
        cfg.listnet_w = a.listnet_w
        cfg.deg_adapt_k = a.deg_adapt_k
        cfg.sign_head_w = a.sign_head_w
        cfg.post_pathway = a.post_pathway
        cfg.chromatin_edges = a.chromatin_edges
        cfg.union_edges = a.union_edges
        assert not (cfg.chromatin_edges and cfg.union_edges), "Cannot use both --chromatin_edges and --union_edges"
        core = LincsV9(cfg, M, ppi, gv)
        if cfg.expr_encoder == 'binned':
            Xfit = D.C[D.tr]                  # THEIR training rows only
            if getattr(D, 'test_rows_dropped', False):
                tr_ri = set(D.row_index[D.tr].tolist())
                assert not tr_ri.intersection(D.original_test_row_indices)
                assert not tr_ri.intersection(D.dev_row_indices)
            if not np.isfinite(Xfit).all():
                raise SystemExit('FATAL: non-finite values in the quantiser fitting sample.')
            core.fit_bins(Xfit)
            if not core.bins_fitted:
                raise SystemExit('FATAL: quantiser did not fit.')
        core = core.to(dev)
        model = torch.nn.DataParallel(core) if torch.cuda.device_count() > 1 else core
        opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.95))
        steps = a.epochs * max(1, len(D.tr) // a.batch)
        warm = max(1, int(0.03 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm else max(0.0, 1 - math.sqrt(
                max(0, s - 0.8 * steps) / max(1, 0.2 * steps))))
        Mn = torch.as_tensor(M, dtype=torch.float32)
        Mn = (Mn / Mn.sum(1, keepdim=True).clamp(min=1)).to(dev)
        scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda')
        t0 = time.time()
        model.train()
        for ep in range(a.epochs):
            order = np.random.permutation(D.tr)
            for it in range(len(order) // a.batch):
                idx = order[it * a.batch:(it + 1) * a.batch]
                b = D.batch(idx, dev)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=dev == 'cuda'):
                    out, aux = model(b, return_aux=True)
                    loss, _ = v9_loss(out, b, cfg, Mn, aux)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                if it % 200 == 0:
                    print(f'  seed{seed} e{ep} it{it}/{len(order) // a.batch} loss {float(loss):.4f} '
                          f'{time.time() - t0:.0f}s', flush=True)
            rng_equal_by_epoch.append(cuda_states_equal(n_gpu))
            if ep == 0:
                print(f'  seed{seed} DP seeding {a.dp_seed_mode} {device_seeds}: device CUDA states equal after '
                      f'epoch 0 = {rng_equal_by_epoch[-1]}', flush=True)
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for s in range(0, len(D.te), 64):
                idx = D.te[s:s + 64]
                if getattr(D, 'test_rows_dropped', False):
                    eval_ri = set(D.row_index[idx].tolist())
                    assert not eval_ri.intersection(D.original_test_row_indices)
                b = D.batch(idx, dev)
                o = model(b)
                P.append(o['abs'].float().cpu().numpy())
                T.append(o['delta'].float().cpu().numpy())
        pa, pd = np.concatenate(P), np.concatenate(T)
        rec = {'seed': seed,
               'Pearson': round(their_pearson(pa, Xte), 4),                    # their convention: MEAN
               'Pearson_deg': round(their_pearson(pd, Xte - Cte), 4),
               'Pearson_median': round(float(np.nanmedian(pearson_rows(pa, Xte))), 4),
               'Pearson_deg_median': round(float(np.nanmedian(pearson_rows(pd, Xte - Cte))), 4),
               'seconds': round(time.time() - t0, 1),
               'n_gpu': n_gpu, 'dp_seed_mode': a.dp_seed_mode, 'device_seeds': device_seeds,
               'cuda_rng_states_equal_by_epoch': rng_equal_by_epoch}
        if a.save_ckpt:
            ckpt_name = a.save_ckpt
            if getattr(a, 'dev_cells', 0) > 0:
                ckpt_name = ckpt_name.replace('.pt', f'_dev{a.dev_cells}s{a.dev_seed}.pt')
            torch.save({'model': core.state_dict(), 'cfg': vars(cfg), 'split': a.split, 'seed': seed,
                        'ablate_epi': bool(a.ablate_epi), 'epochs': a.epochs},
                       ckpt_name.replace('.pt', '_seed%d.pt' % seed))
        if a.save_pred:
            pred_name = a.save_pred
            if getattr(a, 'dev_cells', 0) > 0:
                pred_name = pred_name.replace('.npz', f'_dev{a.dev_cells}s{a.dev_seed}.npz')
            np.savez_compressed(pred_name.replace('.npz', '_seed%d.npz' % seed),
                                y_pred=pa.astype(np.float32), deg_pred=pd.astype(np.float32),
                                y_true=Xte.astype(np.float32), ctl_true=Cte.astype(np.float32),
                                row_index=D.row_index[D.te] if hasattr(D, 'row_index') else D.te)
        runs.append(rec)
        print(f'  [seed {seed}] Pearson {rec["Pearson"]:.4f}  Pearson_deg {rec["Pearson_deg"]:.4f}',
              flush=True)

    def mmr(k):
        v = [r[k] for r in runs]
        return f'{np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]'

    print('\n' + '=' * 92)
    print(f'v9 ON XPERT\'S OWN BENCHMARK ({a.split}), mean [min, max] over {a.seeds} seeds')
    print('=' * 92)
    print(f'  Pearson      (absolute) : {mmr("Pearson")}   | copy-the-control {nulls["copy_ctl_abs"]:.4f}')
    print(f'  Pearson_deg  (delta)    : {mmr("Pearson_deg")}   | mean-drug {nulls["mean_drug_delta"]:.4f}')
    print('  (mean of per-row Pearson, THEIR convention. Their published 0.9804 / 0.8440 is')
    print('   the HDACi figure subset, NOT this benchmark, so it is not printed as a target --')
    print('   the like-for-like number is their checkpoint run on these same rows by')
    print('   model/v9/xpert_native_eval.py.)')
    print('')
    print(f'  {100 * (1 - D.known_cell_frac):.1f}% of rows use a cell line we have no chromatin or'
          f' lineage for; {D.n_dropped_test} test and {D.n_dropped_train} train rows were dropped'
          f' for lack of drug features.')
    WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(
        os.path.dirname(os.path.dirname(HERE)), 'model', 'results')
    os.makedirs(WORK, exist_ok=True)
    # the arm belongs in the filename: the ablated run overwrote the full run's json once, and only
    # the saved predictions made the comparison recoverable
    tag = a.split if a.seeds == 3 and a.seed_start == 0 else f'{a.split}_seed{a.seed_start}'
    if a.ablate_epi:
        tag += '_noepi'
    
    dev_suffix = ''
    if getattr(a, 'dev_cells', 0) > 0:
        dev_suffix = f'_dev{a.dev_cells}s{a.dev_seed}'
    
    # §85.7: output-name tags appended only when non-default
    cand_suffix = ''
    if a.no_atoms:
        cand_suffix += '_noatoms'
    if a.l_control != 2:
        cand_suffix += '_lctl0'
    if a.listnet_w > 0:
        cand_suffix += '_listnet'
    if a.deg_adapt_k > 0:
        cand_suffix += '_degk50'
    if a.sign_head_w > 0:
        cand_suffix += '_signhead'
    if a.post_pathway:
        cand_suffix += '_postpath'
    if a.chromatin_edges:
        cand_suffix += '_chromedges'
    if a.union_edges:
        cand_suffix += '_unionedges'

    out = os.path.join(WORK, f'v9_xpert_arm_{tag}{dev_suffix}{cand_suffix}.json')
    
    candidate_flags = {
        'no_atoms': a.no_atoms,
        'l_control': a.l_control,
        'listnet_w': a.listnet_w,
        'deg_adapt_k': a.deg_adapt_k,
        'sign_head_w': a.sign_head_w,
        'post_pathway': a.post_pathway,
        'chromatin_edges': a.chromatin_edges,
        'union_edges': a.union_edges,
    }

    json_data = {'split': a.split, 'bundle': os.path.basename(npz), 'runs': runs, 'nulls': nulls,
               'platform': platform,
               'n_dropped_test_unfeaturisable': int(getattr(D, 'n_dropped_test', 0)),
               'ablate_epi': bool(a.ablate_epi),
               'metric': 'mean of per-row Pearson (XPert metrics.py convention)',
               'known_cell_frac': round(D.known_cell_frac, 4),
               'n_train': int(len(D.tr)), 'n_test': int(len(D.te)),
               'candidate_flags': candidate_flags}
               
    if getattr(a, 'dev_cells', 0) > 0:
        json_data['mode'] = 'dev'
        json_data['dev'] = D.dev_info
    json.dump(json_data, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
