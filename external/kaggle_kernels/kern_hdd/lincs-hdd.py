# lincs-hdd: RESULTS 100.15, H-DD, read once. Kaggle CPU, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys
PINS = {'hdd_test.py': '96921253c7f2ac0b4247b6063836e0d0587c5a69', 'compound_classes.json': 'b79ac3ffcbb68103d21217cd97ec2571cf97548e', 'hdd_identity.npz': 'a4ddb7be4bf0d5972ed88b4918cb003df63d872f', 'v10c_cold_cell_1_test_on_d256_s20.pt': '54336c5f8207c58b128e80093591a60caa5fc5b9', 'v10c_cold_cell_1_test_on_d256_s11.pt': 'ebcd0879be61ff0ebf80752a3b71dcac1f14d96b', 'v10c_cold_cell_1_test_on_d256_s2.pt': '93294ab74fc28614a4d143632b9a16f21d22c354', 'v10c_cold_cell_1_test_off_d256_s0.pt': '38148319efbf9efbeb10ec9e428ba19f2aa57587', 'v10c_cold_cell_1_test_off_d256_s1.pt': '22d70ae368b749f17edccbe77fd483bcc8d1e832', 'v10c_cold_cell_1_test_off_d256_s2.pt': 'fdf05686a81e739bb8c775147e69d581a8255a0f', 'model_v10c.py': '26a2d5dbe248a9643c6f7d08867515a3cffae0dd', 'data_v10c.py': '6efd1b4a2d545090616b05b81c416a3a7ad282bd', 'train_v10c.py': 'c589ab80a022750cb92a8f4b994eefe13cd271f8', 'modules_v9.py': '4140bf702be5b1a75d1f767a53d9290d8d905d22', 'modules_v7.py': '8c7517e87fe9e3a29e807253095760b6ee40de1f', 'v10c_chromatin.npz': '00977d4080b45b98d436f99444fa20f763a6390d', 'cell_tissue.json': '978239b309ebd8bc95b088610be8bb74918dee3f', 'v10c_full_substrate.npz': 'a276eda863d052214aa8d78afee1ab7c26a95c09', 'drug_feature_index.json': '0b812337aa4dfce1e00d5d86e2dba3cb3f9c0a32', 'drug_descriptors.npy': 'a2f9c17d23d23568f25b27d92e6f74b975e36ebd', 'drug_fingerprints.npy': '8ba2da5982d67d8424b90e854086bad6b786d221', 'drug_unimol.npy': '750385b1f2aa6c65167fe3edd10aa1af919f9e7b', 'drug_atom_reprs.npy': '2f4985edf82abe1b8432bc551363423d8dee872a', 'drug_atom_offsets.npy': 'b93654fc21a633a4ab2e757e9e9fe25d3762c374', 'M_pathway_v9.npy': 'c752c9c25a066af63cbe9595a6565e026441765c', 'STRING_adj_978_v9.npy': '609e289227b4e96da09dcc347667deb5f3645a8a', 'gene_vectors_978.npy': 'f13cc890deb0f8684602d7776828a9509d87f218'}
CODE = {'hdd_test.py': 'model/v10c', 'model_v10c.py': 'model/v10c', 'data_v10c.py': 'model/v10c', 'train_v10c.py': 'model/v10c', 'modules_v9.py': 'model/v9', 'modules_v7.py': 'model/v7'}
CKPTS = {'v10c_cold_cell_1_test_on_d256_s20.pt': ('tc_on_s0', 'tc_on_s0_retry2'), 'v10c_cold_cell_1_test_on_d256_s11.pt': ('tc_on_s1', 'tc_on_s1_retry1'), 'v10c_cold_cell_1_test_on_d256_s2.pt': ('tc_on_s2', 'tc_on_s2'), 'v10c_cold_cell_1_test_off_d256_s0.pt': ('tc_off_s0', 'tc_off_s0'), 'v10c_cold_cell_1_test_off_d256_s1.pt': ('tc_off_s1', 'tc_off_s1'), 'v10c_cold_cell_1_test_off_d256_s2.pt': ('tc_off_s2', 'tc_off_s2')}

def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def find_one(name):
    hits = sorted(set(glob.glob('/kaggle/input/**/' + name, recursive=True)))
    if len(hits) != 1:
        raise SystemExit(f'FATAL: expected exactly 1 {name}, found {hits}')
    if sha1(hits[0]) != PINS[name]:
        raise SystemExit(f'FATAL: {name} sha1 differs from its pin')
    return hits[0]

paths = {n: find_one(n) for n in PINS}
SRC, DATA, RUNS, OUT = '/kaggle/working/src', '/kaggle/working/data', '/kaggle/working/tc_runs', '/kaggle/working'
for n, sub in CODE.items():
    os.makedirs(os.path.join(SRC, sub), exist_ok=True)
    shutil.copyfile(paths[n], os.path.join(SRC, sub, n))
os.makedirs(DATA, exist_ok=True)
for n in PINS:
    if n not in CODE and n not in CKPTS:
        os.symlink(paths[n], os.path.join(DATA, n))
for n, (slot, sub) in CKPTS.items():
    os.makedirs(os.path.join(RUNS, slot, sub), exist_ok=True)
    os.symlink(paths[n], os.path.join(RUNS, slot, sub, n))
cmd = [sys.executable, os.path.join(SRC, 'model/v10c/hdd_test.py'), '--tc_runs', RUNS, '--roots', DATA,
       '--classes', os.path.join(DATA, 'compound_classes.json'), '--identity', os.path.join(DATA, 'hdd_identity.npz'),
       '--out', os.path.join(OUT, 'hdd_reading.json')]
print('RUN', ' '.join(cmd), flush=True)
r = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
json.dump({'pins': PINS, 'returncode': r.returncode}, open(os.path.join(OUT, 'HDD_KERNEL_COMPLETE.json'), 'w'), indent=1)
if r.returncode:
    raise SystemExit(f'FATAL: hdd_test exited {r.returncode}')
print('KERNEL DONE', flush=True)
