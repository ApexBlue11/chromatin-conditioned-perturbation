# RESULTS 90.3 (pre-registered): V1, MC averaging at inference, on one C8b dev checkpoint. CPU only (0 GPU-h).
import glob, hashlib, json, os, subprocess, sys, time
hit = glob.glob('/kaggle/input/**/mc_infer_dev.py', recursive=True)
if len(hit) != 1:
    raise SystemExit('FATAL: expected one mc_infer_dev.py, found %d' % len(hit))
SRC = os.path.dirname(hit[0])
src = open(hit[0], encoding='utf-8').read()
for k in ('def set_mc_mode', 'def mc_predict', 'identity_check', "torch.manual_seed(1000 * seed + k)"):
    if k not in src:
        raise SystemExit('FATAL: mounted mc_infer_dev.py lacks %r' % k)
S = 2
ck = glob.glob('/kaggle/input/**/v9dev_c8b_dev6s0_seed%d.pt' % S, recursive=True)
sv = glob.glob('/kaggle/input/**/v9dev_c8b_dev6s0_seed%d.npz' % S, recursive=True)
if len(ck) != 1 or len(sv) != 1:
    raise SystemExit('FATAL: checkpoint / saved predictions not found: %r %r' % (ck, sv))
h = hashlib.sha1(open(ck[0], 'rb').read()).hexdigest()
print('checkpoint', ck[0], h, flush=True)
if h != '3c46faf9bf19a3856af93175b08c7a6081b49273':
    raise SystemExit('FATAL: checkpoint sha1 mismatch')
for arm, extra in (('det', ['--identity_check', sv[0]]), ('drop', []), ('full', [])):
    t0 = time.time()
    out = '/kaggle/working/v1_%s_dev6s0_seed%d.npz' % (arm, S)
    r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'mc_infer_dev.py'), '--ckpt', ck[0], '--arm', arm, '--K', '8',
                        '--split', 'split_cold_cell_1', '--dev_cells', '6', '--dev_seed', '0', '--out', out] + extra, cwd=SRC)
    print('arm', arm, 'exit', r.returncode, '%.0f s' % (time.time() - t0), flush=True)
    if r.returncode != 0:
        raise SystemExit('FATAL: arm %s failed (%d)' % (arm, r.returncode))
    meta = json.load(open(out.replace('.npz', '.json')))
    print(meta, flush=True)
    c = meta['switched_module_counts']
    # review 030 C1(b): an MC arm that switched nothing would equal det and read as a silent null
    if arm == 'drop' and not (c['Dropout'] > 0 and c['StochasticDepth'] == 0):
        raise SystemExit('FATAL: drop arm switched %r' % c)
    if arm == 'full' and not (c['Dropout'] > 0 and c['StochasticDepth'] > 0):
        raise SystemExit('FATAL: full arm switched %r' % c)
    if arm != 'det':
        import numpy as np
        dmax = float(np.abs(np.load(out)['deg_pred'] - np.load(out.replace(arm, 'det'))['deg_pred']).max())
        print('arm', arm, 'max |deg_pred - det| = %.3e' % dmax, flush=True)
        if not dmax > 0:
            raise SystemExit('FATAL: arm %s equals det exactly' % arm)
print(sorted(glob.glob('/kaggle/working/v1_*')))
