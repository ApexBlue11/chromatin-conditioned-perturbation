# RESULTS 85.12 (pre-registered): P7, the final v9 -- stack P2 + C6 -- on all 32 training cells of split_cold_cell_1,
# seeds 0-2, BLINDED: no model-based score on the test rows is computed, printed or saved here. coldcell_h2h.py scores the
# predictions once, locally. Every output is staged in /tmp and moved to /kaggle/working only after every guard passes.
import sys, os, glob, re, json, shutil, hashlib, subprocess, torch, numpy as np
# LOCAL SMOKE ONLY (env LINCS_P7_SMOKE_DIR): a 1-seed, 1-epoch, 300-row run on the DEV carve, to exercise this glue before the
# GPU spend. It never touches the test cells. On Kaggle the variable is unset and every line below takes the P7 path.
SMOKE = os.environ.get('LINCS_P7_SMOKE_DIR')
if SMOKE:
    SRC = os.environ['LINCS_P7_SRC']
else:
    print('/kaggle/input contains:', sorted(os.listdir('/kaggle/input')), flush=True)
    hit = glob.glob('/kaggle/input/**/xpert_arm.py', recursive=True)
    if not hit:
        raise SystemExit('FATAL: xpert_arm.py not found under /kaggle/input.')
    SRC = os.path.dirname(hit[0])
sys.path.insert(0, SRC)
WORKDIR = SMOKE or '/kaggle/working'
ARMJSON = os.path.join(os.path.dirname(os.path.dirname(SRC)), 'model', 'results') if SMOKE else '/kaggle/working'
STAGE = os.path.join(SMOKE, 'stage') if SMOKE else '/tmp/p7stage'
N_SEEDS = 1 if SMOKE else 3
print('source dir:', SRC, flush=True)
if not torch.cuda.is_available():
    raise SystemExit('FATAL: no CUDA device. NOT falling back to CPU.')
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if any('P100' in n for n in names) or (len(names) < 2 and not SMOKE):
    raise SystemExit('FATAL: need 2 non-P100 GPUs, got %r' % names)
print('GPUs:', names, flush=True)

# GUARD 1: the NaN-quantiser fix (RESULTS 32).
from modules_v9 import BinnedExpression
_q = BinnedExpression(8, n_bins=128, n_genes=978, mode='global')
_x = np.random.default_rng(0).uniform(2, 15, size=(2000, 978)).astype('float32'); _x[3] = np.nan
try:
    _q.fit(_x)
    raise SystemExit('FATAL: mounted code is PRE-FIX -- fit() accepted a NaN row. Refusing.')
except ValueError:
    pass
_q.fit(_x[np.isfinite(_x).all(1)])
if not _q.discriminates(torch.as_tensor(_x[:64][np.isfinite(_x[:64]).all(1)])):
    raise SystemExit('FATAL: quantiser does not discriminate after a clean fit.')

# GUARD 2-3: the mounted arm is the one reviewed (their metric, dev-mode era, C6, V2, and the 85.12 blinding flag).
import inspect, xpert_arm
_src = inspect.getsource(xpert_arm)
need = ('their_pearson', 'seed_start', 'n_dropped_test', 'ablate_epi', 'seed_devices', 'cuda_rng_states_equal_by_epoch',
        'sign_head_w', 'snapshot_cycles', 'def wsd_factor', 'no_test_metrics')
miss = [k for k in need if k not in _src]
if miss:
    raise SystemExit('FATAL: mounted xpert_arm.py lacks %r. Refusing.' % miss)
_dp = glob.glob(os.path.join(SRC, 'dp_seeding.py'))
if not _dp or 'def seed_devices' not in open(_dp[0], encoding='utf-8').read():
    raise SystemExit('FATAL: mounted dp_seeding.py missing or lacks seed_devices. Refusing.')
print('mounted code verified', flush=True)

shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE)
argv = ['--bundle', 'xpert_mdmt_splits.npz', '--split', 'split_cold_cell_1', '--dp_seed_mode', 'distinct',
        '--seeds', '3', '--seed_start', '0', '--epochs', '12', '--d_model', '256'] + ['--sign_head_w', '0.492066'] + [
        '--no_test_metrics', '--save_pred', STAGE + '/v9p7.npz', '--save_ckpt', STAGE + '/v9p7.pt']
if SMOKE:
    argv += ['--dev_cells', '6', '--limit_train', '300', '--epochs', '1', '--seeds', '1', '--batch', '2']   # last wins
print('P7 argv:', ' '.join(argv), flush=True)
LEAK = re.compile(r'pearson[^\n]{0,40}?[-+]?\d\.\d', re.I)


def fail(msg):
    shutil.rmtree(STAGE, ignore_errors=True)       # a failed run leaves no predictions (85.12 item 3)
    for f in set(glob.glob(os.path.join(ARMJSON, 'v9_xpert_arm_split_cold_cell_1*.json'))) - before:
        os.remove(f)                                 # only the file this run created
    raise SystemExit('FATAL: ' + msg + ' -- staging deleted')


JSON_GLOB = os.path.join(ARMJSON, 'v9_xpert_arm_split_cold_cell_1*.json')
before = set(glob.glob(JSON_GLOB))                  # the arm's result JSON is the one file that appears during the run
log = open(STAGE + '/p7_arm.log', 'w', encoding='utf-8')
proc = subprocess.Popen([sys.executable, '-u', os.path.join(SRC, 'xpert_arm.py')] + argv, cwd=SRC,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
n_leak = 0
for line in proc.stdout:
    log.write(line)
    if LEAK.search(line):
        n_leak += 1                                  # never echoed to the kernel log
    else:
        print(line, end='', flush=True)
rc = proc.wait(); log.close()
print('arm exit', rc, flush=True)
if rc != 0:
    fail('the arm exited %d' % rc)
if n_leak:
    fail('%d captured line(s) carried a Pearson value despite --no_test_metrics' % n_leak)

# the arm's result JSON (written to /kaggle/working): no model score, blinding recorded, distinct seeding (GUARD 5)
js = sorted(set(glob.glob(JSON_GLOB)) - before)
if len(js) != 1:
    fail('expected one arm JSON, found %r' % js)
rec = json.load(open(js[0]))
if SMOKE:
    shutil.move(js[0], os.path.join(STAGE, os.path.basename(js[0])))   # never leave it in model/results
if not rec.get('candidate_flags', {}).get('no_test_metrics'):
    fail('arm JSON does not record no_test_metrics')
if any('pearson' in k.lower() for r in rec['runs'] for k in r) or 'metric' in rec:
    fail('arm JSON carries a score field')
for r in rec['runs']:
    eq = r.get('cuda_rng_states_equal_by_epoch') or []
    print('GUARD 5 seed', r['seed'], r.get('device_seeds'), 'equal-by-epoch', eq, flush=True)
    if r.get('n_gpu', 0) > 1 and (not eq or any(eq)):
        fail('distinct seeding did not desynchronise the devices')

# GUARD 6 (85.12 item 4): every prediction file holds exactly 87's 21,151 scored rows
WANT = 'be276e2385330240c2e6237e5b67eb32185d1dde'
preds = sorted(glob.glob(STAGE + '/v9p7*_seed*.npz'))
main = [p for p in preds if re.search(r'_seed\d\.npz$', p)]
if len(main) != N_SEEDS:
    fail('expected %d seed prediction files, found %r' % (N_SEEDS, preds))
for p in preds:
    with np.load(p, allow_pickle=True) as z:          # closed before the move (Windows refuses to move an open file)
        ri = np.asarray(z['row_index']).astype(np.int64)
    got = hashlib.sha1(np.sort(ri).tobytes()).hexdigest()
    print('GUARD 6', os.path.basename(p), len(ri), got, flush=True)
    if not SMOKE and (len(ri) != 21151 or got != WANT):
        fail('GUARD 6: %s is not 87\'s row set' % os.path.basename(p))

for f in sorted(glob.glob(STAGE + '/*')):
    shutil.move(f, os.path.join(WORKDIR, os.path.basename(f)))
for f in sorted(f for f in glob.glob(os.path.join(WORKDIR, '*')) if os.path.isfile(f)):
    print('%9.2f MB  %s  sha1 %s' % (os.path.getsize(f) / 1e6, f, hashlib.sha1(open(f, 'rb').read()).hexdigest()), flush=True)
print('P7 complete: predictions staged out; score them once with coldcell_h2h.py (RESULTS 85.12 item 5)', flush=True)
