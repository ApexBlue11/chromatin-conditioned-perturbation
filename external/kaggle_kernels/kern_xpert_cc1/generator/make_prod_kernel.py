# -*- coding: utf-8 -*-
"""Generate the O2 PRODUCTION kernel (session k) from the v7 kernel [RESULTS 84; review 015]. File-based (6d).

    python make_prod_kernel.py SESSION [PREV_JSON]

PREV_JSON (session k > 1) is session k-1's handoff: {"sha1": {file: sha1}, "final_epoch": e, "torch": v, "cuda": v}.
It is pasted into the kernel as a LITERAL, so the chain of custody runs through git (015 C3)."""
import ast
import io
import json
import sys

SP = r'C:\Users\Surya\AppData\Local\Temp\claude\C--Projects-LINCS-LINCS-project-scope-review-1b6bd9\1a2a87ef-e957-4ed6-bb10-ec2e7f9e1823\scratchpad'
V9 = r'C:\Projects\LINCS\model\v9'
args_list = sys.argv[1:]
platform = 'kaggle'
if '--platform' in args_list:
    idx = args_list.index('--platform')
    platform = args_list[idx + 1]
    args_list = args_list[:idx] + args_list[idx+2:]

first_lightning_session = None
if '--first-lightning-session' in args_list:
    idx = args_list.index('--first-lightning-session')
    first_lightning_session = int(args_list[idx + 1])
    args_list = args_list[:idx] + args_list[idx+2:]

out_path_arg = None
if '--out' in args_list:
    idx = args_list.index('--out')
    out_path_arg = args_list[idx + 1]
    args_list = args_list[:idx] + args_list[idx+2:]

SESSION = int(args_list[0])
PREV = json.load(open(args_list[1])) if len(args_list) > 1 else None
assert (SESSION == 1) == (PREV is None), 'session 1 has no PREV; every later session must have one'
if platform == 'lightning':
    assert first_lightning_session is not None, '--first-lightning-session is required for lightning'

s = io.open(SP + r'\lincs-xpert-cc1.py', encoding='utf-8').read()          # the v7 kernel

# ---- regenerate every embedded source from the repo ----------------------------------------------------------------
src = {n: io.open(V9 + '\\' + f, encoding='utf-8').read() for n, f in
       (('DP_PATCH_SRC', 'xpert_dp_patch.py'), ('DP_PROBE_SRC', 'xpert_dp_probe.py'),
        ('CKPT_PATCH_SRC', 'xpert_ckpt_patch.py'), ('RESUME_PATCH_SRC', 'xpert_resume_patch.py'))}
lines = s.split('\n')
for name in src:
    hits = [i for i, l in enumerate(lines) if l.startswith(name + ' = ')]
    assert len(hits) == 1, name
    lines[hits[0]] = name + ' = ' + repr(src[name])
s = '\n'.join(lines)


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new, 1)


# ---- header ----------------------------------------------------------------------------------------------------------
rep("MEASURE_ONLY = True\n",
    "MEASURE_ONLY = False\n"
    "# O2 PRODUCTION [RESULTS 84, review 015]. GUARD G / H (RESULTS 81) were proved in v7 [81.7] and are not rerun.\n"
    "SESSION = %d\n"
    "HORIZON_EPOCHS = 297          # 84.1: FINAL at 297 completed epochs; the only other final end is their early stop\n"
    "PREV = %s\n"
    "# PREV = session k-1's handoff, pasted as a literal and committed to git before this push (015 C3).\n"
    % (SESSION, repr(PREV)))
rep("# v7: MEASURE_ONLY runs GUARD G (the RESULTS 81 DataParallel proof) and GUARD H (the 81.3 resume tests), then stops.",
    "# Production session %d. The v7 proof and resume tests are in the v7 kernel (git da3c5ac)." % SESSION)

# ---- drop GUARD G and GUARD H; keep FROZEN at top level ------------------------------------------------------------
g0 = s.index("# GUARD G (v7) -- is DataParallel the recipe's computation?")
g0 = s.rindex('# ----', 0, g0)
h1 = s.index("log('GUARD H (81.3):'")
h1 = s.index('\n', h1) + 1
s = s[:g0] + '''import torch
FROZEN = ['attnEncoder_trt.crossEncoders.0.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.0.LayerNorm.gamma',
          'attnEncoder_trt.crossEncoders.1.LayerNorm.beta', 'attnEncoder_trt.crossEncoders.1.LayerNorm.gamma',
          'cell_emb.linear.bias', 'cell_emb.linear.weight', 'ctl_fc.0.bias', 'ctl_fc.0.weight',
          'ctl_fc.3.bias', 'ctl_fc.3.weight']       # RESULTS 80.5 Amendment A; the grad-None set in every v7 mode
''' + s[h1:]

rep("# model/v9/xpert_dp_probe.py. Used ONLY by GUARD G in this version: the trainer command is unchanged.",
    "# model/v9/xpert_dp_probe.py. The probe is kept for provenance; production does not run it (proved in v7, 81.7).")
rep("# The trainer wrapper for DataParallel + full-state resume. In v7 it is used ONLY by GUARD H's tests.",
    "# The trainer wrapper for DataParallel + full-state resume: the PRODUCTION trainer [RESULTS 84].")
# ---- deviations: the three O2 additions (015 ask 1) ------------------------------------------------------------------
rep("                        'all_drugs_unimol_arr.npy rebuilt from the released npz (config names it, never released)',\n",
    "                        'all_drugs_unimol_arr.npy rebuilt from the released npz (config names it, never released)',\n"
    "                        'DataParallel over both T4s, a runtime patch inside XPertNet.forward: loss on the gathered '\n"
    "                        'batch of 128; float64 gradients equal to one GPU to 5e-16 [RESULTS 81.7]',\n"
    "                        'the ten parameters their loss never uses are frozen (requires_grad=False): a no-op on one '\n"
    "                        'GPU, and it stops DataParallel zero-gradients from letting weight decay move them [80.5]',\n"
    "                        'full-state checkpoint and resume hooks across Kaggle sessions (model, Adam, GradScaler, '\n"
    "                        'LambdaLR, stopper, RNG), exact bitwise [81.7]; their lossy --resume_from sets start_epoch only',\n")

# ---- the production tail: everything from the training block to the end ------------------------------------------
t0 = s.index("RECORD['train_cmd'] = ' '.join(cmd)")
tail = io.open(SP + r'\prod_tail.py', encoding='utf-8').read()
s = s[:t0] + tail
if platform == 'lightning':
    rep("W = '/kaggle/working'", "W = os.environ['LINCS_WORKDIR']")
    rep("'/kaggle/input/**/XPert/train_xpert.py'", "os.environ['LINCS_INPUT_ROOT'] + '/**/XPert/train_xpert.py'")
    rep("'/kaggle/input/**/xpert_native_eval.py'", "os.environ['LINCS_INPUT_ROOT'] + '/**/xpert_native_eval.py'")
    rep("'/kaggle/input/**/full_state.pt'", "os.environ['LINCS_INPUT_ROOT'] + '/**/full_state.pt'")
    
    rep("The staged copy in /kaggle/working gets:", "The staged copy in os.environ['LINCS_WORKDIR'] gets:")
    rep("Under /kaggle/input that is read-only", "Under os.environ['LINCS_INPUT_ROOT'] that is read-only")
    rep("whose kill discards /kaggle/working", "whose kill discards os.environ['LINCS_WORKDIR']")
    
    rep("BUDGET_H = 8.3", "BUDGET_H = float(os.environ['LINCS_BUDGET_H'])")
    
    rep("if any('P100' in n for n in names):\n    fatal('P100 assigned; this image has no sm_60 kernels.')\n",
        "if len(names) != 1:\n    fatal('expected exactly ONE visible GPU on Lightning, got %d' % len(names))\n"
        "if not any(x in names[0] for x in ('A100', 'L40S', 'H100', 'H200')):\n    fatal('GPU %s is not one of the approved Lightning models' % names[0])\n"
        "mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n"
        "if mem_gib < 40.0:\n    fatal('GPU %s has %.1f GiB memory, expected >= 40' % (names[0], mem_gib))\n"
        "RECORD['platform'] = {'gpu_name': names[0], 'mem_gib': round(mem_gib, 2)}\n"
        "if torch.backends.cuda.matmul.allow_tf32 is not False:\n    fatal('allow_tf32 is not False')\n"
        "if torch.get_float32_matmul_precision() != 'highest':\n    fatal('float32_matmul_precision is not highest')\n"
        "if 'NVIDIA_TF32_OVERRIDE' in os.environ:\n    fatal('NVIDIA_TF32_OVERRIDE is in os.environ')\n"
        "RECORD['platform'].update({\n"
        "    'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,\n"
        "    'device_capability': torch.cuda.get_device_capability(0),\n"
        "    'flash_sdp_enabled': torch.backends.cuda.flash_sdp_enabled(),\n"
        "    'mem_efficient_sdp_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),\n"
        "    'math_sdp_enabled': torch.backends.cuda.math_sdp_enabled()\n"
        "})\n"
    )
    
    rep("'import xpert_ckpt_patch\\n'\n         'xpert_ckpt_patch.apply()\\n'", 
        "'import xpert_dp_patch\\n'\n         'xpert_dp_patch.apply()\\n'")
        
    rep("'assert peak < 13.0, \"training step peak %%.2f GiB leaves no headroom on a T4\" %% peak\\n'",
        "'mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)\\n'\n"
        "         'assert peak < 0.85 * mem_gib, \"training step peak %%.2f GiB leaves no headroom on %%.1f GiB GPU\" %% (peak, mem_gib)\\n'")
        
    rep("if SESSION == 1:", f"if SESSION == {first_lightning_session}:")
    rep("if SESSION == 1 and len(res['epoch_s']) >= 2:\n    first2 = sum(res['epoch_s'][:2]) / 2\n    sess['amendment_E'] = {'first_two_mean_s': round(first2, 1), 'projection_s': 482.2,\n                           'reprice_before_session_2': first2 > 1.25 * 482.2}",
        f"if SESSION == {first_lightning_session} and len(res['epoch_s']) >= 2:\n"
        "    first2 = sum(res['epoch_s'][:2]) / 2\n"
        "    probe_s = float(os.environ['LINCS_EPOCH_S_PROBE'])\n"
        "    sess['amendment_E'] = {'first_two_mean_s': round(first2, 1), 'projection_s': probe_s,\n"
        "                           'reprice': first2 > 1.25 * probe_s}"
    )

    dev_text = "sessions 1-2 on 2xT4 DataParallel; sessions >= 3 on one %s as published; different GPU architecture and SDPA backend (rounding-level); CUDA RNG of the second device not carried across"
    rep("                        'the ten parameters their loss never uses are frozen (requires_grad=False): a no-op on one '\n"
        "                        'GPU, and it stops DataParallel zero-gradients from letting weight decay move them [80.5]',\n"
        "                        'full-state checkpoint and resume hooks across Kaggle sessions (model, Adam, GradScaler, '\n"
        "                        'LambdaLR, stopper, RNG), exact bitwise [81.7]; their lossy --resume_from sets start_epoch only',\n",
        "                        'the ten parameters their loss never uses are frozen (requires_grad=False): a no-op on one '\n"
        "                        'GPU, and it stops DataParallel zero-gradients from letting weight decay move them [80.5]',\n"
        "                        'full-state checkpoint and resume hooks across Kaggle sessions (model, Adam, GradScaler, '\n"
        "                        'LambdaLR, stopper, RNG), exact bitwise [81.7]; their lossy --resume_from sets start_epoch only',\n"
        "                        %r %% names[0],\n" % dev_text)

ast.parse(s)
for node in ast.parse(s).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in src:
        assert ast.literal_eval(node.value) == src[node.targets[0].id], node.targets[0].id
if out_path_arg:
    out = out_path_arg
else:
    if platform == 'lightning':
        import os
        os.makedirs(r'C:\Projects\LINCS\external\lightning\o2', exist_ok=True)
        out = r'C:\Projects\LINCS\external\lightning\o2\lincs-xpert-cc1_lightning_session%d.py' % SESSION
    else:
        out = r'C:\Projects\LINCS\external\kaggle_kernels\kern_xpert_cc1\lincs-xpert-cc1.py'

io.open(out, 'w', encoding='utf-8').write(s)
if platform == 'kaggle':
    io.open(SP + r'\lincs-xpert-cc1_session%d.py' % SESSION, 'w', encoding='utf-8').write(s)
print('production kernel, session %d: generated, parses, embedded sources byte-identical' % SESSION)
