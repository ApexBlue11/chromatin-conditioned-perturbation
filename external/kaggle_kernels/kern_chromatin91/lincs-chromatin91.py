# RESULTS 91 (pre-registered; 91.8, 91.9): the chromatin funnel and its planted-effect power calibration, run once. CPU only.
import glob, hashlib, json, os, shutil, subprocess, sys, time
PINS = {'chromatin_funnel.py': '81bb6c17dcf3d8b5d26eeba906e901eeae4dccbe', 'chromatin_power.py': '153c270271f9d54d8b492ff0eefbf178ad932d0d', 'score_dev.py': 'a9002bceff97ab5552790caf12b6819e7949965e', 'E_final.npy': '20e9a3e4ce56f0d8a938c185dc5ca9e016590aac', 'E_final_mask.npy': '6186bbdd9edae48d0382b5f269c600ff455fdb2f', 'lincs_cell_index.json': '5f7bc6b42fe45c56f62a27d06c1925a077056e6d', 'cell_lineage.npy': '1efd3dea686816db00b10ca426ed2e54d533c75d', 'E_final_provenance.json': '22c249d1e6c8208809e6fa088664d93b1de18c8f', 'chembl_dti_edges.tsv': '28b9e02f6a7f23fac27fd452cb0c745e4838c15d'}
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
print('inputs verified: %d pinned files + the split bundle' % len(PINS), flush=True)
DATA = '/tmp/funnel_data'
os.makedirs(DATA, exist_ok=True)
for f in ('E_final.npy', 'E_final_mask.npy', 'lincs_cell_index.json', 'cell_lineage.npy'):
    shutil.copyfile(os.path.join(SRC, f), os.path.join(DATA, f))
os.symlink(spl[0], os.path.join(DATA, 'xpert_mdmt_splits.npz'))
common = ['--data_dir', DATA, '--provenance', os.path.join(SRC, 'E_final_provenance.json')]
env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
t = time.time()
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'chromatin_funnel.py')] + common +
                   ['--dti', os.path.join(SRC, 'chembl_dti_edges.tsv'), '--out', '/kaggle/working/chromatin_funnel_91.json'], env=env)
print('funnel exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: the funnel failed')
env1 = dict(env, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
t = time.time()
r = subprocess.run([sys.executable, '-u', os.path.join(SRC, 'chromatin_power.py')] + common +
                   ['--out', '/kaggle/working/chromatin_power_91.json', '--draws', '5', '--workers', '4'], env=env1,
                   stdout=subprocess.DEVNULL)
print('power exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: the power calibration failed')
json.dump({'pins': PINS, 'splits_sha1': SPLITS_SHA1, 'complete': True}, open('/kaggle/working/CHROMATIN91_COMPLETE.json', 'w'))
print('done', flush=True)
