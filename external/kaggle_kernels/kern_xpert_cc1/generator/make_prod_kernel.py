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
SESSION = int(sys.argv[1])
PREV = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
assert (SESSION == 1) == (PREV is None), 'session 1 has no PREV; every later session must have one'

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
ast.parse(s)
for node in ast.parse(s).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in src:
        assert ast.literal_eval(node.value) == src[node.targets[0].id], node.targets[0].id
out = r'C:\Projects\LINCS\external\kaggle_kernels\kern_xpert_cc1\lincs-xpert-cc1.py'
io.open(out, 'w', encoding='utf-8').write(s)
io.open(SP + r'\lincs-xpert-cc1_session%d.py' % SESSION, 'w', encoding='utf-8').write(s)
print('production kernel, session %d: generated, parses, embedded sources byte-identical' % SESSION)
