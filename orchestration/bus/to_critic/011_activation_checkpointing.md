# PACKET 011 — activation checkpointing of XPert's encoders, for go / no-go
packet_id: 011
created: 2026-09-23
repo_commit: 50f49c7
type: **PRE-SPEND, DEVIATION REVIEW.** GPU hours this session: 6.74. This packet commits none.

v4 of the kernel you approved in 009 passed every guard and then ran out of GPU memory on the first training forward.
The fix changes how their model's forward executes, though not what it computes, so I am asking before relaunching.

## WHAT HAPPENED
All seven guards passed. Host MemAvailable minimum 18.5 GB over the run (60 s samples, in `run_record.json`). Trainer
traceback, `models/model_utils.py:26`, inside `selfEncoders` of the first training forward:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 124.00 MiB. GPU 0 has a total capacity of 14.56 GiB of
which 82.81 MiB is free. Including non-PyTorch memory, this process has 14.48 GiB memory in use. Of the allocated memory
14.17 GiB is allocated by PyTorch, and 188.85 MiB is reserved by PyTorch but unallocated.
```
GUARD F in v4 ran its forward pass under `torch.no_grad()`, so it held no training activations.

## MEASUREMENT
`model/v9/xpert_gpu_mem.py`: their `XPertNet` via our harness (`include_cell_idx=True` architecture, released weights),
`.train()`, one step = `torch.autocast('cuda', float16)` forward + backward of a proxy loss (sum of the means of the
first three outputs), at batch 2, 4 and 8 from `split_cold_cell_1` training rows; peak allocated above resident weights;
linear extrapolation. RTX 3050 laptop (sm_86), with `enable_flash_sdp(False)` to approximate a T4 (sm_75):

| SDPA setting | per sample | projected at 128 |
|---|---|---|
| flash off, PyTorch's choice | 0.114 GiB | 14.6 GiB |
| math only | 0.691 GiB | 88.5 GiB |
| memory-efficient only | 0.117 GiB | 15.0 GiB |

## THEIR LOSS
`train_xpert.py:104-106` (the `--use_gradscaler` path), used for `epoch < init_epoch` (70):
```python
batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + (loss2/num_samples) * b
                    + torch.sqrt(loss3/num_samples) * c + (loss4/num_samples) * d
```
`model_XPert.py:193` `drug_feat = drug_feat.to(self.device)`; `:135` `self.drug_HG_embed = torch.tensor(..., device=device)`
(a plain attribute, not a buffer or parameter).

## THE PROPOSED CHANGE
`model/v9/xpert_ckpt_patch.py`: replaces `Encoder.forward` and `crossEncoder.forward` (`models/model_utils.py`) with a
wrapper that calls `torch.utils.checkpoint.checkpoint(orig, self, *a, use_reentrant=False, **k)` when
`self.training and torch.is_grad_enabled()`, and the original otherwise. Applied at runtime by
`run_train_ckpt.py`, which sets `sys.argv`, imports `models.model_utils`, applies the patch, then calls
`train_xpert.main()`. Their files are not edited. `train_xpert.py:733` guards `main()` with `__name__ == '__main__'`.

## THE PROOF
`model/v9/prove_checkpoint_exact.py` imports and applies `xpert_ckpt_patch` itself. Batch 8, same weights, same
inputs, `torch.manual_seed(0)` before each step, autocast fp16:
```
patched classes: ['Encoder', 'crossEncoder']
parameters with gradients: 176
loss  unpatched run 1 7.56921387 | run 2 7.56921387 | checkpointed 7.56921387
max |grad| difference: unpatched vs unpatched (noise floor) 4.630e-05 | unpatched vs checkpointed 4.353e-05
checkpointing within the noise floor: True
unpatched     step peak {2: 0.237, 4: 0.473, 8: 0.938} GiB | per sample 0.1168 GiB | projected at batch 128: 14.95 GiB
checkpointed  step peak {2: 0.069, 4: 0.127, 8: 0.244} GiB | per sample 0.0291 GiB | projected at batch 128: 3.73 GiB
PROOF PASSED
```
An earlier run of the same proof with an inline copy of the wrapper gave floor 8.69e−05, difference 5.94e−05.

## KERNEL CHANGES FOR v5
- The patch source is embedded as `CKPT_PATCH_SRC`, generated from the repo file; compared as an AST literal against
  `model/v9/xpert_ckpt_patch.py`: identical. Written to the staged copy with `run_train_ckpt.py`.
- The trainer command runs `run_train_ckpt.py` with the same arguments; GUARD E still reads their printed args.
- GUARD F now: applies the patch, `model.train()`, `torch.cuda.amp.autocast()` forward + backward on one batch from
  their `load_dataloader`, asserts the batch is 128, records peak GPU memory, and refuses above 13.0 GiB.
- Fifth entry in `RECORD['deviations']`.

## ASKS
1. **Go or no-go.** Is runtime activation checkpointing, with gradients inside the run-to-run noise floor, a deviation
   that bears on "XPert as published"?
2. The proof uses a proxy loss, not their `weighted_loss` / `batch_weighted_loss`. Checkpointing wraps the layers, not
   the loss, so the loss form should not matter to exactness — is that reasoning sound, or should the proof use their
   loss?
3. Is 13.0 GiB the right GUARD F ceiling, given the 3.73 GiB projection, the optimizer state, and that the kernel's
   training process and GUARD F's probe run sequentially, not concurrently?
4. Anything else, including whether gradient accumulation or DataParallel should be preferred after all.
