# lincs-v11-no-targets-s1 : RESULTS 102 / 102.1, v11. Kaggle 2 x T4, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys, time
T0 = time.time()
PINS = {'model_v11.py': '524da72f468b68376bbdeb40fbc9af9d23e4f2c0', 'data_v11.py': '73f797c4ebffd31dc155e7e942a58a489a06d023', 'train_v11.py': '2356732fcec6fcb1f27ef65ab2047402bc91cbcc', 'model_v10c.py': '26a2d5dbe248a9643c6f7d08867515a3cffae0dd', 'data_v10c.py': '6efd1b4a2d545090616b05b81c416a3a7ad282bd', 'train_v10c.py': 'c589ab80a022750cb92a8f4b994eefe13cd271f8', 'modules_v9.py': '4140bf702be5b1a75d1f767a53d9290d8d905d22', 'modules_v7.py': '8c7517e87fe9e3a29e807253095760b6ee40de1f', 'v10c_chromatin.npz': '00977d4080b45b98d436f99444fa20f763a6390d', 'cell_tissue.json': '978239b309ebd8bc95b088610be8bb74918dee3f', 'string_graph_v11_expdb.npz': 'b9b80d5717ab2698313867442eae6757eaf329a7', 'gene_vectors_v11_full.npy': 'e7a4cf169fd878af88e52a2131f14343ea61cf89', 'v11_chromatin.npz': '6f4dcb7bb8d638bc5c8109593a2ef193b2ee762c', 'ccle_expr_v11.npz': '46f686870bb6587bfeee4a9fc9f99a1c140ebe4c', 'chembl_dti_edges.tsv': '28b9e02f6a7f23fac27fd452cb0c745e4838c15d', 'drug_inchikeys.tsv': 'f31a4cafa994607eeab9cac56e6b697b486fde01', 'action_sign_map.json': '21599e466b84ca9db880cc4534270dd99942d717', 'dti_probe_split.json': 'a6d89d5b68d60c51e86ac05fca53ade1a016e72e', 'v10c_full_substrate.npz': 'a276eda863d052214aa8d78afee1ab7c26a95c09', 'drug_feature_index.json': '0b812337aa4dfce1e00d5d86e2dba3cb3f9c0a32', 'drug_descriptors.npy': 'a2f9c17d23d23568f25b27d92e6f74b975e36ebd', 'drug_fingerprints.npy': '8ba2da5982d67d8424b90e854086bad6b786d221', 'drug_unimol.npy': '750385b1f2aa6c65167fe3edd10aa1af919f9e7b', 'drug_atom_reprs.npy': '2f4985edf82abe1b8432bc551363423d8dee872a', 'drug_atom_offsets.npy': 'b93654fc21a633a4ab2e757e9e9fe25d3762c374', 'M_pathway_v9.npy': 'c752c9c25a066af63cbe9595a6565e026441765c', 'STRING_adj_978_v9.npy': '609e289227b4e96da09dcc347667deb5f3645a8a', 'gene_vectors_978.npy': 'f13cc890deb0f8684602d7776828a9509d87f218'}
CODE = {'model_v11.py': 'model/v11', 'data_v11.py': 'model/v11', 'train_v11.py': 'model/v11', 'model_v10c.py': 'model/v10c', 'data_v10c.py': 'model/v10c', 'train_v10c.py': 'model/v10c', 'modules_v9.py': 'model/v9', 'modules_v7.py': 'model/v7'}
RUNS = [['v11_no_targets_s1', ['--dev', '--seed', '1', '--epochs', '4', '--d_model', '256', '--arm', 'no_targets', '--guard']]]
RUN_H = 4.5
START_ATTEMPT = 0
GUARD_EXIT = 3

def next_attempt(args, attempt, elapsed_h, run_h, budget_h=11.0, bumps=(0, 10, 20)):
    """RESULTS 100.8: the arguments of attempt `attempt` (0 = the registered run) with the seed bumped by 0, 10, 20, or None
    when no retry is left or the retry would not fit this session's budget (it then runs in a new kernel)."""
    if attempt >= len(bumps) or (attempt > 0 and elapsed_h + run_h > budget_h):
        return None
    a = list(args)
    i = a.index('--seed')
    a[i + 1] = str(int(a[i + 1]) + bumps[attempt])
    return a


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
    got = sha1(hits[0])
    if got != PINS[name]:
        raise SystemExit(f'FATAL: {name} sha1 {got} != pinned {PINS[name]}')
    return hits[0]

paths = {n: find_one(n) for n in PINS}
SRC, DATA, OUT = '/kaggle/working/src', '/kaggle/working/data', '/kaggle/working'
for n, sub in CODE.items():
    os.makedirs(os.path.join(SRC, sub), exist_ok=True)
    shutil.copyfile(paths[n], os.path.join(SRC, sub, n))
os.makedirs(DATA, exist_ok=True)
for n in PINS:
    if n not in CODE:
        os.symlink(paths[n], os.path.join(DATA, n))
probe = subprocess.run([sys.executable, '-c', 'import json, torch; print(json.dumps([torch.cuda.get_device_name(i) '
                        'for i in range(torch.cuda.device_count())]))'], capture_output=True, text=True)
names = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else []
if len(names) < 2 or any('P100' in n for n in names):
    raise SystemExit(f'FATAL: need 2 non-P100 GPUs, got {names} {probe.stderr[-500:]}')
print('GPUs', names, flush=True)
done, attempts = [], {}
for tag, args in RUNS:
    guard = '--guard' in args
    k, attempts[tag] = (START_ATTEMPT if guard else 0), []
    while True:
        a2 = next_attempt(args, k, (time.time() - T0) / 3600, RUN_H) if guard else (args if k == 0 else None)
        if a2 is None:
            print('NO ATTEMPT LEFT IN THIS KERNEL', tag, flush=True)
            break
        out = os.path.join(OUT, tag if k == 0 else f'{tag}_retry{k}')
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(SRC, 'model/v11/train_v11.py'), '--roots', DATA, '--out', out] + a2
        print('RUN', tag, k, ' '.join(cmd), flush=True)
        r = subprocess.run(cmd)
        attempts[tag].append({'attempt': k, 'args': a2, 'out': out, 'returncode': r.returncode,
                              'elapsed_h': round((time.time() - T0) / 3600, 3)})
        if r.returncode == 0:
            done.append(tag)
            break
        if not (guard and r.returncode == GUARD_EXIT):
            raise SystemExit(f'FATAL: run {tag} exited {r.returncode}')
        k += 1
json.dump({'pins': PINS, 'runs': RUNS, 'done': done, 'attempts': attempts, 'gpus': names,
           'wall_s': round(time.time() - T0, 1)},
          open(os.path.join(OUT, 'V11_KERNEL_COMPLETE.json'), 'w'), indent=1)
print('KERNEL DONE', done, flush=True)
