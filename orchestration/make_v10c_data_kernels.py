"""Build the v10-C L2/L3 data kernels (RESULTS 100.1): N_SHARD ChIP-Atlas bigWig signal kernels and one CCLE methylation
kernel, all CPU with internet, all outcome-free (no expression data mounted), plus the private source dataset
apexblue/lincs-v10c-chromsrc.

  python orchestration/make_v10c_data_kernels.py --selection <selection.json>
"""
import argparse, hashlib, json, os, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, 'external', 'kaggle_v10c_chromsrc')
KERN_DIR = os.path.join(REPO, 'external', 'kaggle_kernels')
DATASET = 'apexblue/lincs-v10c-chromsrc'
N_SHARD = 4
PYBIGWIG = 'pyBigWig==0.3.26'
BW_URL = 'https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/{srx}.bw'
CCLE_URL = 'https://data.broadinstitute.org/ccle/CCLE_RRBS_TSS1kb_20181022.txt.gz'
BUDGET_S = 10.5 * 3600  # stop starting downloads here: a 12 h timeout saves nothing


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


COMMON = r'''
import glob, hashlib, json, os, subprocess, sys, time
T0 = time.time()
PINS = __PINS__

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
assert len(GENES) == 977, len(GENES)  # 978 landmarks; MUC1 has no MANE Select TSS
for p in glob.glob('/kaggle/input/**/*', recursive=True):
    if any(t in os.path.basename(p).lower() for t in ('.h5ad', 'xpert_mdmt', 'level3', '.gctx')):
        raise SystemExit(f'FATAL: expression data mounted in an outcome-free kernel: {p}')
'''

SIGNAL = r'''
import concurrent.futures as cf, tempfile, urllib.request
import numpy as np
SHARD, N_SHARD, BUDGET_S, BW_URL = __SHARD__, __N_SHARD__, __BUDGET_S__, __BW_URL__
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', __PYBIGWIG__])
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
out = f'/kaggle/working/v10c_signal_s{SHARD}.npz'
np.savez_compressed(out, srx=np.array(done), values=arr, genes=np.array([g[0] for g in GENES]),
                    windows=np.array([w[0] for w in cs.WINDOWS]))
json.dump({'shard': SHARD, 'n_shard': N_SHARD, 'pins': PINS, 'pybigwig': pyBigWig.__version__ if hasattr(pyBigWig, '__version__') else None,
           'n_assigned': len(mine), 'n_ok': len(done), 'tracks': manifest, 'npz_sha1': sha1(out),
           'wall_s': round(time.time() - T0, 1)},
          open(f'/kaggle/working/V10C_SIGNAL_COMPLETE_s{SHARD}.json', 'w'), indent=1)
print('ok', len(done), 'of', len(mine), flush=True)
'''

METHYL = r'''
import urllib.request
import numpy as np
cells = json.load(open(paths['full_cells.json'], encoding='utf-8'))['cells']
path = '/kaggle/working/ccle_rrbs.txt.gz'
urllib.request.urlretrieve(__CCLE_URL__, path)
digest = sha1(path)
text = cs.read_gz_text(path)
print('header', text.split('\n', 1)[0][:400], flush=True)
syms = [g[0] for g in GENES]
m, info = cs.ccle_methyl(text, syms, list(cells))
names = sorted(m)
arr = np.asarray([m[c] for c in names], dtype=np.float32).reshape(len(names), len(syms))
np.savez_compressed('/kaggle/working/v10c_methyl.npz', cells=np.array(names), values=arr, genes=np.array(syms))
os.remove(path)
json.dump(dict(info, url=__CCLE_URL__, sha1=digest, pins=PINS, n_cells=len(names),
               genes_with_any=int(np.isfinite(arr).any(0).sum()), npz_sha1=sha1('/kaggle/working/v10c_methyl.npz')),
          open('/kaggle/working/V10C_METHYL_COMPLETE.json', 'w'), indent=1)
print('ok', len(names), 'cells', flush=True)
'''


def metadata(name):
    return {'id': f'apexblue/{name}', 'title': name, 'code_file': f'{name}.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
            'enable_internet': True, 'dataset_sources': [DATASET], 'kernel_sources': [],
            'competition_sources': [], 'model_sources': []}


def build(selection, cells_json, tss=None, signal_py=None, src_dir=SRC_DIR, kern_dir=KERN_DIR):
    tss = tss or os.path.join(REPO, 'phase2_assembly', 'outputs', 'tss_hg38.tsv')
    signal_py = signal_py or os.path.join(REPO, 'model', 'v10c', 'chromatin_signal.py')
    os.makedirs(src_dir, exist_ok=True)
    files = {'selection.json': selection, 'full_cells.json': cells_json, 'tss_hg38.tsv': tss,
             'chromatin_signal.py': signal_py}
    for n, p in files.items():
        shutil.copyfile(p, os.path.join(src_dir, n))
    with open(os.path.join(src_dir, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'title': DATASET.split('/')[1], 'id': DATASET, 'licenses': [{'name': 'other'}]}, f, indent=1)
    pins = {n: sha1(os.path.join(src_dir, n)) for n in files}
    common = COMMON.replace('__PINS__', repr(pins))
    built = {}
    jobs = [(f'lincs-v10c-signal-s{k}', SIGNAL.replace('__SHARD__', str(k))) for k in range(N_SHARD)]
    jobs.append(('lincs-v10c-methyl', METHYL))
    for name, body in jobs:
        body = (body.replace('__N_SHARD__', str(N_SHARD)).replace('__BUDGET_S__', repr(BUDGET_S))
                .replace('__BW_URL__', repr(BW_URL)).replace('__PYBIGWIG__', repr(PYBIGWIG))
                .replace('__CCLE_URL__', repr(CCLE_URL)))
        assert '__' not in body.replace('__name__', '').replace('__version__', ''), name
        d = os.path.join(kern_dir, 'kern_' + name.replace('lincs-', '').replace('-', '_'))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name + '.py'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(common + body)
        with open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata(name), f, indent=1)
        built[name] = d
    return pins, built


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', required=True)
    ap.add_argument('--cells', required=True)
    a = ap.parse_args()
    pins, built = build(a.selection, a.cells)
    print(json.dumps(pins, indent=1))
    for n, d in built.items():
        print(n, d)


if __name__ == '__main__':
    main()
