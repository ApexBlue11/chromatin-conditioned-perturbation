# RESULTS 85.7 "Auxiliary weights by rule": w = 0.1 * ||grad L_delta|| / ||grad L_aux|| at a fresh seed-0 init of the
# baseline configuration, first 20 batches of seed 0's order on the dev training rows, fp32, CPU. Run once; frozen.
import glob, os, shutil, subprocess, sys
hit = glob.glob('/kaggle/input/**/calibrate_aux_weights.py', recursive=True)
if len(hit) != 1:
    raise SystemExit('FATAL: expected one calibrate_aux_weights.py, found %d' % len(hit))
W = '/kaggle/working/w/code'
shutil.copytree(os.path.dirname(hit[0]), W)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
r = subprocess.run([sys.executable, '-u', 'calibrate_aux_weights.py', '--n_batches', '20'], cwd=W, env=env)
print('exit', r.returncode, flush=True)
out = '/kaggle/working/model/results/v9_aux_weight_calibration.json'
print(open(out).read() if os.path.exists(out) else 'NO OUTPUT')
if r.returncode != 0:
    raise SystemExit(r.returncode)
