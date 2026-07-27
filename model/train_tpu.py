# -*- coding: utf-8 -*-
"""
TPU (torch_xla) training for LincsCrossAttn. GPU quota is exhausted; TPU is the only accelerator left.

WHY THIS IS NOT A TRIVIAL PORT (measured, see results/RESULTS.md §15):
  - Per CORE the TPU is SLOWER than a T4 for our 978x978 attention (39.4 ms vs ~30 ms). The ONLY way TPU
    wins is using ALL 8 cores (~3.1x aggregate). So single-core mode exists purely as a smoke test.
  - XLA recompiles on every new tensor SHAPE. Our collate padded atoms to the batch max => a new shape
    almost every step => permanent recompilation. Fixed by `collate(fixed_pad=True)` (padding to a static
    max_atoms), which is PROVEN not to leak into attention (3 tests, 0.00e+00 deviation).
  - Any `.item()` / python-float read inside the loop forces a graph sync and kills throughput, so the
    running loss is accumulated ON DEVICE and only read once per epoch.

MODES
  python train_tpu.py --smoke     : 1 core, a few steps, verifies correctness + measures step time
  python train_tpu.py             : 8 cores via xmp.spawn (real training)
"""
import os, sys, glob, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import ModelConfig, DataConfig
from data import LincsDataset, build_splits, collate
from model import LincsCrossAttn
import losses as L
from train import make_dataconfig, build_probe, interaction_probe


def _collate_static(samples, max_atoms):
    """XLA needs STATIC shapes -> always pad to max_atoms (safe: padded slots are masked out)."""
    return collate(samples, max_atoms, fixed_pad=True)


def _run(rank, args):
    device = xm.xla_device()
    is_master = xm.is_master_ordinal()
    mc = ModelConfig(); dc = make_dataconfig()
    if args.fold is not None:
        dc.cell_fold = args.fold
    R = lambda p: os.path.join(dc.root, p)

    model = LincsCrossAttn(mc, np.load(R(dc.cop_path)), np.load(R(dc.ppi_path))).to(device)
    shared = LincsDataset.load_shared(dc)
    full = LincsDataset(dc, _shared=shared)
    sp = build_splits(full, dc)
    train_ds = LincsDataset(dc, indices=sp["train"], _shared=shared)
    if is_master:
        print(f"cores={xm.xrt_world_size()} fold={dc.cell_fold} "
              f"splits={ {k: len(v) for k, v in sp.items() if not k.startswith('_')} }", flush=True)

    sampler = DistributedSampler(train_ds, num_replicas=xm.xrt_world_size(),
                                 rank=xm.get_ordinal(), shuffle=True, drop_last=True)
    dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=2, drop_last=True,
                    collate_fn=lambda s: _collate_static(s, mc.max_atoms))
    mp_dl = pl.MpDeviceLoader(dl, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    if args.smoke:
        model.train(); t0 = time.time(); losses = []
        for it, b in enumerate(mp_dl):
            opt.zero_grad(set_to_none=True)
            loss = L.weighted_huber(model(b), b["Y"], w=b.get("w"), delta=mc.huber_delta)
            loss.backward()
            xm.optimizer_step(opt)                      # all-reduces grads across cores
            losses.append(loss)
            if it + 1 >= args.smoke_steps:
                break
        vals = [float(x) for x in losses]               # single sync at the end
        dt = time.time() - t0
        print(f"SMOKE OK: {len(vals)} steps in {dt:.1f}s = {dt/max(len(vals),1)*1000:.0f} ms/step "
              f"(batch {args.batch}/core)", flush=True)
        print(f"  loss first {vals[0]:.4f} -> last {vals[-1]:.4f}  (finite={all(np.isfinite(vals))})", flush=True)
        return

    WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    hist = []; t0 = time.time()
    probe = build_probe(full) if is_master else None
    for epoch in range(args.epochs):
        model.train(); sampler.set_epoch(epoch)
        run = torch.zeros((), device=device); n = 0
        for b in mp_dl:
            opt.zero_grad(set_to_none=True)
            loss = L.weighted_huber(model(b), b["Y"], w=b.get("w"), delta=mc.huber_delta)
            loss.backward()
            xm.optimizer_step(opt); sched.step()
            run += loss.detach(); n += 1               # stays on device: no per-step sync
        train_loss = float(run) / max(n, 1)            # ONE sync per epoch
        if is_master:
            rec = {"epoch": epoch, "train_loss": train_loss,
                   "elapsed_h": round((time.time() - t0) / 3600, 3)}
            cpu_model = model.to("cpu")
            if probe:
                rec["interaction_probe"] = interaction_probe(cpu_model, full, "cpu", probe, mc.max_atoms)
            model.to(device)
            hist.append(rec); print("EPOCH", json.dumps(rec), flush=True)
            xm.save(cpu_model.state_dict(), f"{WORK}/ckpt_tpu_fold{dc.cell_fold}.pt")
            json.dump(hist, open(f"{WORK}/metrics_tpu_fold{dc.cell_fold}.json", "w"), indent=2)
        if (time.time() - t0) / 3600 > args.budget_h:
            if is_master:
                print("time budget reached -> stop (resumable)", flush=True)
            break
    if is_master:
        print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_steps", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)     # PER CORE (8 cores -> effective 128)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--budget_h", type=float, default=8.0)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--cores", type=int, default=8)
    a = ap.parse_args()
    if a.smoke:
        _run(0, a)                                        # single core, in-process
    else:
        xmp.spawn(_run, args=(a,), nprocs=a.cores)


if __name__ == "__main__":
    main()
