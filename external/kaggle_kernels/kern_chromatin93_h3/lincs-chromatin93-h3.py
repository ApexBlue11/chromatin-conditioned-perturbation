# RESULTS 93 Stage 1a (pre-registered, amended by review 050): h3, run once. CPU only.
import glob, hashlib, json, os, shutil, subprocess, sys, time
PINS = {'chromatin_funnel.py': 'c56bb6f153596e4234fa74e3372606aba55f2ee4', 'chromatin_power.py': 'c501cba28b1dfe0c435bd68ab8cc6f4787956424', 'score_dev.py': 'a9002bceff97ab5552790caf12b6819e7949965e', 'chromatin_h2.py': 'e75d128189cb81208beae6b15bd20296b9c2e5d6', 'chromatin_gbm.py': '3f3ee97d135f1b82fb29b2b067b0bf7c96af6fd8', 'E_final.npy': '20e9a3e4ce56f0d8a938c185dc5ca9e016590aac', 'E_final_mask.npy': '6186bbdd9edae48d0382b5f269c600ff455fdb2f', 'lincs_cell_index.json': '5f7bc6b42fe45c56f62a27d06c1925a077056e6d', 'cell_lineage.npy': '1efd3dea686816db00b10ca426ed2e54d533c75d', 'E_final_provenance.json': '22c249d1e6c8208809e6fa088664d93b1de18c8f'}
SPLITS_SHA1 = '1444b253d48c5adf066bf7a3cc0ba8ea5fb71464'
hit = glob.glob('/kaggle/input/**/chromatin_funnel.py', recursive=True)
spl = glob.glob('/kaggle/input/**/xpert_mdmt_splits.npz', recursive=True)
if not hit or not spl:
    raise SystemExit('FATAL: inputs not mounted: %r %r' % (hit, spl))
SRC = os.path.dirname(hit[0])
sha = lambda p: hashlib.sha1(open(p, 'rb').read()).hexdigest()
for f, h in PINS.items():
    got = sha(os.path.join(SRC, f))
    if got != h:
        raise SystemExit('FATAL: %s sha1 %s != pinned %s' % (f, got, h))
if sha(spl[0]) != SPLITS_SHA1:
    raise SystemExit('FATAL: xpert_mdmt_splits.npz is not the reviewed bundle')
import lightgbm
print('inputs verified: %d pinned files + the split bundle; lightgbm %s' % (len(PINS), lightgbm.__version__), flush=True)
DATA = '/tmp/c93_data'
os.makedirs(DATA, exist_ok=True)
for f in ('E_final.npy', 'E_final_mask.npy', 'lincs_cell_index.json', 'cell_lineage.npy'):
    shutil.copyfile(os.path.join(SRC, f), os.path.join(DATA, f))
os.symlink(spl[0], os.path.join(DATA, 'xpert_mdmt_splits.npz'))
OUT = '/kaggle/working/chromatin_gbm_93.json'
env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
t = time.time()
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'chromatin_gbm.py'), '--data_dir', DATA,
                    '--provenance', os.path.join(SRC, 'E_final_provenance.json'), '--out', OUT] + ['--draws', '3'], env=env)
print('chromatin_gbm.py exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: chromatin_gbm.py failed')
json.dump({'pins': PINS, 'splits_sha1': SPLITS_SHA1, 'lightgbm': lightgbm.__version__,
           'outputs': {'chromatin_gbm_93.json': sha(OUT)}, 'complete': True}, open('/kaggle/working/CHROMATIN93_H3_COMPLETE.json', 'w'))
print('done', flush=True)
