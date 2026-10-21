
import glob, hashlib, json, os, subprocess, sys, time
T0 = time.time()
PINS = {'selection.json': '70052ec4116ad4acc9df03a40a7cd084d87c53e4', 'full_cells.json': '71932be11415a12412128eaa9fd88dc5584eaace', 'tss_hg38.tsv': '67ac22c649396ef2a57de9f9f596badfe720aec1', 'chromatin_signal.py': 'e1d883454e13c61b8dbcf0aa1eb3244b6dcca65f'}

def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def find_one(name):
    hits = sorted(set(glob.glob('/kaggle/input/**/' + name, recursive=True)))
    if len(hits) != 1:
        raise SystemExit(f'FATAL: expected exactly 1 {name}, found {hits}')
    if sha1(hits[0]) != PINS[name]:
        raise SystemExit(f'FATAL: {name} sha1 {sha1(hits[0])} != pinned {PINS[name]}')
    return hits[0]

paths = {n: find_one(n) for n in PINS}
sys.path.insert(0, os.path.dirname(paths['chromatin_signal.py']))
import chromatin_signal as cs
GENES = cs.load_tss(paths['tss_hg38.tsv'])
assert len(GENES) == 18658, len(GENES)  # v11: STRING genes with a MANE Select TSS
for p in glob.glob('/kaggle/input/**/*', recursive=True):
    if any(t in os.path.basename(p).lower() for t in ('.h5ad', 'xpert_mdmt', 'level3', '.gctx')):
        raise SystemExit(f'FATAL: expression data mounted in an outcome-free kernel: {p}')

import concurrent.futures as cf, tempfile, urllib.request
import numpy as np
SHARD, N_SHARD, BUDGET_S, BW_URL = 1, 4, 37800.0, 'https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/{srx}.bw'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyBigWig==0.3.26'])
import pyBigWig
sel = json.load(open(paths['selection.json'], encoding='utf-8'))
tracks = sorted({r['srx'] for r in sel['records']})
mine = tracks[SHARD::N_SHARD]
print(f'shard {SHARD}/{N_SHARD}: {len(mine)} of {len(tracks)} tracks', flush=True)
TMP = tempfile.mkdtemp()

def fetch(url, path, tries=4):
    last = None
    for k in range(tries):
        try:
            h = hashlib.sha1(); n = 0
            with urllib.request.urlopen(url, timeout=120) as r, open(path, 'wb') as f:
                want = int(r.headers.get('Content-Length', -1))
                for b in iter(lambda: r.read(1 << 22), b''):
                    f.write(b); h.update(b); n += len(b)
            if want >= 0 and n != want:
                raise IOError(f'short read {n} of {want}')
            return h.hexdigest(), n
        except Exception as e:
            last = e
            time.sleep(10 * (k + 1))
    raise IOError(f'{url}: {last}')

def work(srx):
    if time.time() - T0 > BUDGET_S:
        return srx, None, {'status': 'not_started_budget'}
    url, path = BW_URL.format(srx=srx), os.path.join(TMP, srx + '.bw')
    t = time.time()
    try:
        digest, size = fetch(url, path)
        bw = pyBigWig.open(path)
        try:
            vals, info = cs.window_means(bw, GENES)
        finally:
            bw.close()
    except Exception as e:
        return srx, None, {'status': 'failed', 'error': repr(e)[:500], 'url': url}
    finally:
        if os.path.exists(path):
            os.remove(path)
    return srx, vals, dict(info, status='ok', url=url, sha1=digest, bytes=size, seconds=round(time.time() - t, 1))

done, vals, manifest = [], [], {}
with cf.ThreadPoolExecutor(2) as ex:
    for i, (srx, v, info) in enumerate(ex.map(work, mine)):
        manifest[srx] = info
        if v is not None:
            done.append(srx); vals.append(v)
        print(i + 1, srx, info.get('status'), info.get('bytes'), info.get('seconds'), flush=True)
arr = np.asarray(vals, dtype=np.float32).reshape(len(vals), len(GENES), len(cs.WINDOWS))
out = f'/kaggle/working/v11_signal_s{SHARD}.npz'
np.savez_compressed(out, srx=np.array(done), values=arr, genes=np.array([g[0] for g in GENES]),
                    windows=np.array([w[0] for w in cs.WINDOWS]))
json.dump({'shard': SHARD, 'n_shard': N_SHARD, 'pins': PINS, 'pybigwig': pyBigWig.__version__ if hasattr(pyBigWig, '__version__') else None,
           'n_assigned': len(mine), 'n_ok': len(done), 'tracks': manifest, 'npz_sha1': sha1(out),
           'wall_s': round(time.time() - T0, 1)},
          open(f'/kaggle/working/V11_SIGNAL_COMPLETE_s{SHARD}.json', 'w'), indent=1)
print('ok', len(done), 'of', len(mine), flush=True)
