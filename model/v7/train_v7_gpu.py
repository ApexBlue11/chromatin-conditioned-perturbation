# -*- coding: utf-8 -*-
"""
v7 training on Kaggle T4 x2. The modern recipe (EMA + WSD + optional Muon + grad clipping) around a model
whose biological priors are now SUPERVISED. Rationale: ../V7_PLAN.md.

Guards carried over from v6 unchanged, because each one caught a real failure:
  * no CPU fallback  -- a GPU kernel silently on CPU still burns the whole session
  * P100 probe       -- Kaggle torch has no sm_60 kernels; a 1s matmul beats an 8h waste
  * check_inputs     -- data.py degrades to neutral defaults when a file is missing, and a rebuilt Kaggle
                        dataset once silently dropped the compound holdout AND reliability weighting
  * resumable ckpt + time budget -- a non-clean exit discards /kaggle/working
"""
import os, sys, json, time, math, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v6"))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_v7 import V7Config, V7TrainConfig, V6DataConfig, resolve_paths
from model_v7 import LincsV7, aux_targets
from data import LincsDataset, build_splits, collate
import losses as L


# ---------------------------------------------------------------------------------------------------
def probe_gpu(require_n=2):
    """NO FALLBACK ANYWHERE. Anything other than exactly `require_n` usable T4s is a hard crash, because
    every silent degradation this project has hit cost a full session: a GPU kernel quietly running on CPU
    still burns the whole session, a P100 dies on the first op with no sm_60 kernels, and a run that
    silently uses ONE of two GPUs looks completely normal in the logs while taking twice as long."""
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: no CUDA device. NOT falling back to CPU.")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    if any("P100" in x for x in names):
        raise SystemExit(f"FATAL: P100 assigned ({names}); Kaggle torch has no sm_60 kernels.")
    if n != require_n:
        raise SystemExit(f"FATAL: expected exactly {require_n} GPUs (T4 x2), found {n}: {names}. "
                         f"NOT falling back to a single GPU -- provision machine_shape NvidiaTeslaT4.")
    for i in range(n):                       # every device must actually execute, not just enumerate
        try:
            _ = (torch.randn(8, 8, device=f"cuda:{i}") @ torch.randn(8, 8, device=f"cuda:{i}")).sum().item()
        except RuntimeError as e:
            raise SystemExit(f"FATAL: cuda:{i} ({names[i]}) present but unusable: {e}")
    return names


def assert_both_gpus_used(tag=""):
    """Verify BOTH devices actually hold work. DataParallel silently degrades to one GPU if the batch is
    smaller than the device count, and nothing in the log would say so."""
    used = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
    if not all(u > 0 for u in used):
        raise SystemExit(f"FATAL{tag}: not every GPU is holding tensors: "
                         f"{[f'cuda:{i}={u/1e6:.0f}MB' for i, u in enumerate(used)]}. "
                         f"The run would use one GPU while billing for two.")
    print(f"both GPUs active{tag}: " + "  ".join(f"cuda:{i}={u/1e6:.0f}MB" for i, u in enumerate(used)),
          flush=True)


def check_inputs(ds, sp):
    """Refuse to train on a silently degraded setup -- see the v6 trainer for the incident this prevents."""
    problems = []
    for k in ("test_colddrug", "test_coldboth"):
        if len(sp[k]) == 0:
            problems.append(f"{k} EMPTY -> scaffold_split.json missing; no compound holdout (leakage)")
    if float(ds.strength.min()) == 1.0 and float(ds.strength.max()) == 1.0:
        problems.append("strength identically 1.0 -> sig_strength.npy missing; reliability weighting OFF "
                        "and the reproducible-stratum filter passes everything")
    if problems:
        raise SystemExit("FATAL: inputs missing, data.py fell back to neutral defaults:\n  - "
                         + "\n  - ".join(problems) + "\nRefusing to train.")
    print(f"input check OK: strength {ds.strength.min():.2f}-{ds.strength.max():.2f}, "
          f"weight {ds.weight.min():.3f}-{ds.weight.max():.3f}, compound holdout present", flush=True)


def wsd_lambda(total, warmup_frac, decay_frac):
    """Warmup -> Stable -> Decay, with the minus-square-root anneal used in current large-scale recipes.
    Unlike cosine, the stable phase means an early stop still leaves a usable model."""
    w = max(1, int(total * warmup_frac)); d = max(1, int(total * decay_frac))

    def f(step):
        if step < w:
            return step / w
        if step < total - d:
            return 1.0
        return max(0.0, 1.0 - math.sqrt((step - (total - d)) / d))
    return f


class EMA:
    """Exponential moving average of weights. Documented gains in generalisation AND robustness to NOISY
    LABELS -- ~75 % of LINCS perturbations are inert, so this is the best-matched item in the recipe.
    Evaluated separately from the raw weights so we can see whether it actually helped."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow


def newton_schulz(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz orthogonalisation of a matrix (the core of Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalised update for 2D weights. Reported to beat AdamW at scale; the edge is
    smaller at ours, so it is an option rather than the default. Non-2D params must go to AdamW."""
    def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.setdefault("m", torch.zeros_like(p.grad))
                buf.mul_(g["momentum"]).add_(p.grad)
                d = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                upd = newton_schulz(d, g["ns_steps"])
                p.add_(upd, alpha=-g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


def build_optimizer(model, tc):
    decay, no_decay, muon_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "gene_emb" in n or "log_var" in n:
            no_decay.append(p)
        elif tc.optimizer == "muon":
            muon_p.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    adam = torch.optim.AdamW([g for g in groups if g["params"]], lr=tc.lr, betas=(0.9, 0.95))
    return (adam, Muon(muon_p, lr=tc.lr * 50) if muon_p else None)


@torch.no_grad()
def quick_eval(model, ds, idx, cfg, device, batch=64, max_n=3000):
    if len(idx) == 0:
        return {}
    if len(idx) > max_n:
        idx = np.random.default_rng(0).choice(idx, max_n, replace=False)
    was = model.training; model.eval()
    yh, yt = [], []
    for s in range(0, len(idx), batch):
        b = collate([ds[i] for i in idx[s:s + batch]], cfg.max_atoms)
        bd = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        yh.append(model(bd).float().cpu()); yt.append(b["Y"])
    model.train(was)
    yh, yt = torch.cat(yh), torch.cat(yt)
    return {"pearson_median": float(L.pearson_per_row(yh, yt).median()),
            "r2_overall": L.r2_overall(yh, yt), "n": int(len(yt))}


def main():
    ap = argparse.ArgumentParser()
    tc = V7TrainConfig()
    for f, t in [("epochs", int), ("batch", int), ("lr", float), ("budget_h", float), ("fold", int),
                 ("workers", int), ("ema_decay", float), ("stoch_depth", float)]:
        ap.add_argument(f"--{f}", type=t, default=None)
    ap.add_argument("--optimizer", default=None, choices=["adamw", "muon"])
    ap.add_argument("--no_aux", action="store_true"); ap.add_argument("--no_ppi", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="run 3 of these; v6-vs-v5 came in at +/-0.007 and "
                                                        "we cannot call that a result without seed variance")
    ap.add_argument("--m_reactome", default=None,
                    help="override the pathway matrix, e.g. M_reactome_ms5.npy (765 finer nodes)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    cfg = V7Config()
    for f in ["epochs", "batch", "lr", "budget_h", "fold", "workers", "ema_decay", "optimizer"]:
        if getattr(a, f, None) is not None:
            setattr(tc, f, getattr(a, f))
    if a.stoch_depth is not None:
        cfg.stoch_depth = a.stoch_depth
    cfg.use_aux = not a.no_aux; cfg.use_ppi = not a.no_ppi

    names = probe_gpu(require_n=2)
    torch.backends.cudnn.benchmark = True                     # static gene-token length -> autotuned kernels
    torch.backends.cuda.matmul.allow_tf32 = True              # free on Turing+ for the fp32 residue
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"; n_gpu = torch.cuda.device_count()
    print(f"GPUs: {names} | seed={a.seed} aux={cfg.use_aux} ppi={cfg.use_ppi} sd={cfg.stoch_depth} "
          f"opt={tc.optimizer} sched={tc.schedule} ema={tc.ema_decay}", flush=True)

    dc = resolve_paths(V6DataConfig()); dc.cell_fold = tc.fold
    dc.cache_in_ram = True                                    # ~2.7 GB; kills the per-sample mmap faults
    WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    CKPT = f"{WORK}/ckpt_v7_fold{tc.fold}_seed{a.seed}.pt"
    Rp = lambda p: p if os.path.isabs(p) else os.path.join(dc.root, p)
    if a.m_reactome:
        import glob as _g
        hit = next(iter(_g.glob(f"/kaggle/input/**/{a.m_reactome}", recursive=True)), None)
        dc.m_reactome_path = hit or os.path.join(dc.root, "network/outputs", a.m_reactome)
    M = np.load(Rp(dc.m_reactome_path))
    ppi = np.load(Rp(dc.ppi_path)) if cfg.use_ppi else None
    print(f"pathway matrix {os.path.basename(dc.m_reactome_path)}: {M.shape[0]} nodes, "
          f"{int((M.sum(0) > 0).sum())}/{M.shape[1]} genes covered | "
          f"STRING: {int((np.asarray(ppi) > 0).sum() // 2) if ppi is not None else 0} edges", flush=True)

    core = LincsV7(cfg, M, ppi).to(device)
    print(f"params {sum(p.numel() for p in core.parameters())/1e6:.2f}M", flush=True)
    if tc.batch % n_gpu:
        raise SystemExit(f"FATAL: batch {tc.batch} is not divisible by {n_gpu} GPUs -- DataParallel would "
                         f"give the devices unequal work.")
    model = torch.nn.DataParallel(core)
    print(f"DataParallel across {n_gpu} GPUs (per-GPU batch {tc.batch // n_gpu})", flush=True)

    shared = LincsDataset.load_shared(dc)
    full = LincsDataset(dc, _shared=shared)
    sp = build_splits(full, dc)
    print(f"splits={ {k: len(v) for k, v in sp.items() if not k.startswith('_')} }", flush=True)
    check_inputs(full, sp)
    train_ds = LincsDataset(dc, indices=sp["train"], _shared=shared)
    dl = DataLoader(train_ds, batch_size=tc.batch, shuffle=True, drop_last=True, num_workers=tc.workers,
                    persistent_workers=tc.workers > 0, pin_memory=True, prefetch_factor=6,
                    collate_fn=lambda s: collate(s, cfg.max_atoms))

    Mn = torch.as_tensor(M, dtype=torch.float32)
    if Mn.shape[0] > Mn.shape[1]:
        Mn = Mn.t()
    Mn = (Mn / Mn.sum(1, keepdim=True).clamp(min=1)).to(device)      # [P,G] row-normalised

    opt, muon = build_optimizer(core, tc)
    total = tc.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, wsd_lambda(total, tc.warmup_frac, tc.decay_frac))
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(core, tc.ema_decay)

    rep = sp["test_coldcell"][full.strength[sp["test_coldcell"]] >= dc.eval_min_strength]
    hist, t0 = [], time.time()
    model.train()
    for epoch in range(tc.epochs):
        te = time.time(); run = n = 0; wlog = {}
        for it, b in enumerate(dl):
            bd = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            if muon:
                muon.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                if cfg.use_aux:
                    yhat, aux = model(bd, return_aux=True)
                    t_path, t_epi = aux_targets(bd["Y"], Mn)
                    l_main = L.weighted_huber(yhat, bd["Y"], w=bd.get("w"), delta=cfg.huber_delta)
                    l_path = torch.nn.functional.huber_loss(aux["pathway_pred"].float(), t_path)
                    l_epi = torch.nn.functional.huber_loss(aux["epi_pred"].float(), t_epi)
                    loss, wlog = core.task_weights([l_main, l_path, l_epi])
                else:
                    loss = L.weighted_huber(model(bd), bd["Y"], w=bd.get("w"), delta=cfg.huber_delta)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(core.parameters(), tc.grad_clip)
            scaler.step(opt)
            if muon:
                muon.step()
            scaler.update(); sched.step(); ema.update(core)
            run += float(loss); n += 1
            if epoch == 0 and it == 0:
                assert_both_gpus_used(" after first step")
            if it % 200 == 0:
                print(f"  e{epoch} it{it}/{len(dl)} loss {float(loss):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {wlog}", flush=True)
        rec = {"epoch": epoch, "train_loss": run / max(n, 1), "sec": round(time.time() - te, 1),
               "elapsed_h": round((time.time() - t0) / 3600, 3), "task_weights": wlog,
               "cold_reproducible": quick_eval(model, full, rep, cfg, device)}
        hist.append(rec); print("EPOCH", json.dumps(rec), flush=True)
        torch.save({"model": core.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "hist": hist,
                    "cfg": vars(cfg), "tcfg": vars(tc)}, CKPT)
        json.dump(hist, open(f"{WORK}/metrics_v7_fold{tc.fold}.json", "w"), indent=2)
        if (time.time() - t0) / 3600 > tc.budget_h:
            print("time budget reached -> clean checkpoint + stop (resumable)", flush=True); break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
