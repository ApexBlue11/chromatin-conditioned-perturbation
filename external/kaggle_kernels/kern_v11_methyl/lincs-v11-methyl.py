
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

import urllib.request
import numpy as np
cells = json.load(open(paths['full_cells.json'], encoding='utf-8'))['cells']
path = '/kaggle/working/ccle_rrbs.txt.gz'
urllib.request.urlretrieve('https://data.broadinstitute.org/ccle/CCLE_RRBS_TSS1kb_20181022.txt.gz', path)
digest = sha1(path)
text = cs.read_gz_text(path)
print('header', text.split('\n', 1)[0][:400], flush=True)
syms = [g[0] for g in GENES]
m, info = cs.ccle_methyl(text, syms, list(cells))
names = sorted(m)
arr = np.asarray([m[c] for c in names], dtype=np.float32).reshape(len(names), len(syms))
np.savez_compressed('/kaggle/working/v11_methyl.npz', cells=np.array(names), values=arr, genes=np.array(syms))
os.remove(path)
json.dump(dict(info, url='https://data.broadinstitute.org/ccle/CCLE_RRBS_TSS1kb_20181022.txt.gz', sha1=digest, pins=PINS, n_cells=len(names),
               genes_with_any=int(np.isfinite(arr).any(0).sum()), npz_sha1=sha1('/kaggle/working/v11_methyl.npz')),
          open('/kaggle/working/V11_METHYL_COMPLETE.json', 'w'), indent=1)
print('ok', len(names), 'cells', flush=True)
