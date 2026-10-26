"""Build H-DD's one Kaggle CPU kernel (RESULTS 100.15; review 101's practical note): the scorer runs on a free CPU kernel,
internet off, every input sha1-pinned. Inputs: apexblue/lincs-hdd-inputs (the six T-C checkpoints, the scorer, the classes,
the identity targets) beside the datasets T-C ran on (lincs-v10c-src, lincs-v10c-full, lincs-v9-bundle).

    python orchestration/make_hdd_kernel.py
"""
import glob, hashlib, json, os, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'model', 'v10c'))
sys.path.insert(0, os.path.join(REPO, 'orchestration'))
import hdd_test  # noqa: E402
import make_v10c_kernels as mk  # noqa: E402

DS = os.path.join(REPO, 'external', 'kaggle_hdd_inputs')
KERN = os.path.join(REPO, 'external', 'kaggle_kernels', 'kern_hdd')
TC_RUNS = os.path.join(REPO, 'external', 'kaggle_out', 'v10c_tc')
CODE = {'hdd_test.py': 'model/v10c', 'model_v10c.py': 'model/v10c', 'data_v10c.py': 'model/v10c',
        'train_v10c.py': 'model/v10c', 'modules_v9.py': 'model/v9', 'modules_v7.py': 'model/v7'}

TEMPLATE = r'''# lincs-hdd: RESULTS 100.15, H-DD, read once. Kaggle CPU, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys
PINS = __PINS__
CODE = __CODE__
CKPTS = __CKPTS__

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
'''


def main():
    os.makedirs(DS, exist_ok=True)
    ident = os.path.join(DS, 'hdd_identity.npz')
    if not os.path.exists(ident):
        hdd_test.make_identity(TC_RUNS, ident, ledger=os.path.join(REPO, 'model/results/v10c/tc_ledger.jsonl'),
                               substrate=os.path.join(REPO, 'external/v10c_full/v10c_full_substrate.npz'))
    ckpts = {}
    for arm, slots in hdd_test.CKPTS.items():
        for slot, sub, seed in slots:
            (p,) = glob.glob(os.path.join(TC_RUNS, slot, sub, f'v10c_cold_cell_1_test_{arm}_d*_s{seed}.pt'))
            n = os.path.basename(p)
            if not os.path.exists(os.path.join(DS, n)):
                shutil.copyfile(p, os.path.join(DS, n))
            ckpts[n] = (slot, sub)
    shutil.copyfile(os.path.join(REPO, 'model/v10c/hdd_test.py'), os.path.join(DS, 'hdd_test.py'))
    shutil.copyfile(os.path.join(REPO, 'model/results/v10c/compound_classes.json'), os.path.join(DS, 'compound_classes.json'))
    json.dump({'title': 'lincs-hdd-inputs', 'id': 'apexblue/lincs-hdd-inputs', 'licenses': [{'name': 'other'}]},
              open(os.path.join(DS, 'dataset-metadata.json'), 'w'), indent=1)
    pins = {n: mk.sha1(os.path.join(DS, n)) for n in ['hdd_test.py', 'compound_classes.json', 'hdd_identity.npz'] + list(ckpts)}
    # the code and data T-C ran on: lincs-v10c-src, lincs-v10c-full, lincs-v9-bundle, with T-C's own pins
    tc_kernel = open(os.path.join(REPO, 'external/kaggle_kernels/kern_v10c_tc_on_s0/lincs-v10c-tc-on-s0.py'),
                     encoding='utf-8').read()
    import ast, re
    tc_pins = ast.literal_eval(re.search(r'^PINS = (.*)$', tc_kernel, re.M).group(1))
    for n in ('model_v10c.py', 'data_v10c.py', 'train_v10c.py', 'modules_v9.py', 'modules_v7.py', 'v10c_chromatin.npz',
              'cell_tissue.json', 'v10c_full_substrate.npz', 'drug_feature_index.json', 'drug_descriptors.npy',
              'drug_fingerprints.npy', 'drug_unimol.npy', 'drug_atom_reprs.npy', 'drug_atom_offsets.npy', 'M_pathway_v9.npy',
              'STRING_adj_978_v9.npy', 'gene_vectors_978.npy'):
        pins[n] = tc_pins[n]
    os.makedirs(KERN, exist_ok=True)
    code = (TEMPLATE.replace('__PINS__', repr(pins)).replace('__CODE__', repr(CODE)).replace('__CKPTS__', repr(ckpts)))
    with open(os.path.join(KERN, 'lincs-hdd.py'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(code)
    meta = {'id': 'apexblue/lincs-hdd', 'title': 'lincs-hdd', 'code_file': 'lincs-hdd.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
            'keywords': [], 'dataset_sources': ['apexblue/lincs-hdd-inputs', 'apexblue/lincs-v10c-src',
                                                'apexblue/lincs-v10c-full', 'apexblue/lincs-v9-bundle'],
            'kernel_sources': [], 'competition_sources': [], 'model_sources': [], 'docker_image': mk.DOCKER}
    json.dump(meta, open(os.path.join(KERN, 'kernel-metadata.json'), 'w'), indent=1)
    print(json.dumps({n: pins[n][:8] for n in pins}, indent=0))
    print(KERN, DS)


if __name__ == '__main__':
    main()
