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
import torch_xla.runtime as xr
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
    if getattr(args, "bf16", False):
        os.environ["XLA_USE_BF16"] = "1"      # TPU MXU is bf16-native; fp32 wastes most of the chip
    device = torch_xla.device()
    is_master = xr.global_ordinal() == 0
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
        print(f"cores={xr.world_size()} fold={dc.cell_fold} "
              f"splits={ {k: len(v) for k, v in sp.items() if not k.startswith('_')} }", flush=True)

    sampler = DistributedSampler(train_ds, num_replicas=xr.world_size(),
                                 rank=xr.global_ordinal(), shuffle=True, drop_last=True)
    dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=4, drop_last=True,
                    persistent_workers=True, prefetch_factor=4,
                    collate_fn=lambda s: _collate_static(s, mc.max_atoms))
    mp_dl = pl.MpDeviceLoader(dl, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    if args.smoke:
        # TPU-CORRECT TIMING: the first step(s) include XLA COMPILATION (tens of seconds). Timing them
        # together with steady-state steps is the classic TPU benchmarking error -- it made our first run
        # look 12x slower than a T4. Warm up first, sync, THEN time.
        model.train()
        warm = 3
        for it, b in enumerate(mp_dl):
            opt.zero_grad(set_to_none=True)
            loss = L.weighted_huber(model(b), b["Y"], w=b.get("w"), delta=mc.huber_delta)
            loss.backward(); xm.optimizer_step(opt)
            if it + 1 >= warm:
                break
        torch_xla.sync(); _ = float(loss)                       # force compile + finish
        t0 = time.time(); losses = []; n = 0
        for b in mp_dl:
            opt.zero_grad(set_to_none=True)
            loss = L.weighted_huber(model(b), b["Y"], w=b.get("w"), delta=mc.huber_delta)
            loss.backward(); xm.optimizer_step(opt)
            losses.append(loss.detach()); n += 1
            if n >= args.smoke_steps:
                break
        torch_xla.sync()
        vals = [float(x) for x in losses]
        dt = time.time() - t0
        per_step = dt / max(n, 1)
        print(f"SMOKE OK (post-warmup): {n} steps in {dt:.1f}s = {per_step*1000:.0f} ms/step "
              f"| batch {args.batch}/core -> {per_step/args.batch*1000:.1f} ms/sample/core", flush=True)
        print(f"  8-core projection: {per_step/args.batch/8*1000:.1f} ms/sample "
              f"(T4 reference ~19.4 ms/sample at batch 64)", flush=True)
        print(f"  loss {vals[0]:.4f} -> {vals[-1]:.4f} (finite={all(np.isfinite(vals))})", flush=True)
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
    ap.add_argument("--batch", type=int, default=32)     # PER CORE (8 cores -> effective 256)
    ap.add_argument("--bf16", action="store_true", help="TPU-native bfloat16 (MXU fast path)")
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
