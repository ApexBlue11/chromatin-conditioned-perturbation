"""Build the v10-C GPU kernels (RESULTS 100.4): 2 x T4, internet off, every input sha1-pinned and mounted once.

  python orchestration/make_v10c_kernels.py timing                 -> kern_v10c_timing (d 256 and d 384, 300 steps each)
  python orchestration/make_v10c_kernels.py run --tag tb_on_s0 --args "--dev --seed 0"
  python orchestration/make_v10c_kernels.py run --tag tc_on_s1 --args "--seed 1 --guard" --start_attempt 1
      (RESULTS 100.8 / review 090 C4: a retry in a new kernel; the seed is bumped from the REGISTERED seed)

The source dataset apexblue/lincs-v10c-src carries the code and the small data files; the substrate is
apexblue/lincs-v10c-full; drug and network files come from apexblue/lincs-v9-bundle with the audit99 probe's pins.
"""
import argparse, hashlib, inspect, json, os, shlex, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, 'external', 'kaggle_v10c_src')
KERN_DIR = os.path.join(REPO, 'external', 'kaggle_kernels')
PROBE = os.path.join(REPO, 'external', 'kaggle_out', 'audit99_probe', 'audit99_data_pins.json')
CODE = {'model_v10c.py': 'model/v10c', 'data_v10c.py': 'model/v10c', 'train_v10c.py': 'model/v10c',
        'modules_v9.py': 'model/v9', 'modules_v7.py': 'model/v7'}
SMALL = {'v10c_chromatin.npz': 'external/v10c_chromatin.npz', 'cell_tissue.json': 'model/results/v10c/cell_tissue.json'}
SUBSTRATE = ('v10c_full_substrate.npz', 'external/v10c_full/v10c_full_substrate.npz')
FROM_V9 = ['drug_feature_index.json', 'drug_descriptors.npy', 'drug_fingerprints.npy', 'drug_unimol.npy',
           'drug_atom_reprs.npy', 'drug_atom_offsets.npy', 'M_pathway_v9.npy', 'STRING_adj_978_v9.npy',
           'gene_vectors_978.npy']
DATASETS = ['apexblue/lincs-v10c-src', 'apexblue/lincs-v10c-full', 'apexblue/lincs-v9-bundle']
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


GUARD_EXIT = 3          # train_v10c.GUARD_EXIT (asserted equal in the tests)


def next_attempt(args, attempt, elapsed_h, run_h, budget_h=11.0, bumps=(0, 10, 20)):
    """RESULTS 100.8: the arguments of attempt `attempt` (0 = the registered run) with the seed bumped by 0, 10, 20, or None
    when no retry is left or the retry would not fit this session's budget (it then runs in a new kernel)."""
    if attempt >= len(bumps) or (attempt > 0 and elapsed_h + run_h > budget_h):
        return None
    a = list(args)
    i = a.index('--seed')
    a[i + 1] = str(int(a[i + 1]) + bumps[attempt])
    return a


TEMPLATE = r'''# __NAME__ : RESULTS 100.4, v10-C. Kaggle 2 x T4, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys, time
T0 = time.time()
PINS = __PINS__
CODE = __CODE__
RUNS = __RUNS__
RUN_H = __RUN_H__
START_ATTEMPT = __START_ATTEMPT__
GUARD_EXIT = __GUARD_EXIT__

__NEXT_ATTEMPT__

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
        cmd = [sys.executable, os.path.join(SRC, 'model/v10c/train_v10c.py'), '--roots', DATA, '--out', out] + a2
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
          open(os.path.join(OUT, 'V10C_KERNEL_COMPLETE.json'), 'w'), indent=1)
print('KERNEL DONE', done, flush=True)
'''


def build_src(src_dir=SRC_DIR):
    os.makedirs(src_dir, exist_ok=True)
    for n, sub in CODE.items():
        shutil.copyfile(os.path.join(REPO, sub, n), os.path.join(src_dir, n))
    for n, rel in SMALL.items():
        shutil.copyfile(os.path.join(REPO, rel), os.path.join(src_dir, n))
    with open(os.path.join(src_dir, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'title': 'lincs-v10c-src', 'id': 'apexblue/lincs-v10c-src', 'licenses': [{'name': 'other'}]}, f, indent=1)
    pins = {n: sha1(os.path.join(src_dir, n)) for n in list(CODE) + list(SMALL)}
    pins[SUBSTRATE[0]] = sha1(os.path.join(REPO, SUBSTRATE[1]))
    probe = json.load(open(PROBE, encoding='utf-8'))
    for n in FROM_V9:
        if probe[n]['copies'] != 1:
            raise SystemExit(f'FATAL: {n} has {probe[n]["copies"]} mounted copies in the probe')
        pins[n] = probe[n]['sha1'][0]
    return pins


def render(name, runs, pins, kern_dir=KERN_DIR, run_h=3.8, start_attempt=0):
    if start_attempt not in (0, 1, 2):
        raise SystemExit(f'FATAL: start_attempt {start_attempt} outside 100.8\'s two retries')
    if start_attempt and not all('--guard' in args for _, args in runs):
        raise SystemExit('FATAL: --start_attempt is a 100.8 retry and needs --guard in every run')
    d = os.path.join(kern_dir, 'kern_' + name.replace('lincs-', '').replace('-', '_'))
    os.makedirs(d, exist_ok=True)
    code = (TEMPLATE.replace('__NAME__', name).replace('__PINS__', repr(pins)).replace('__CODE__', repr(CODE))
            .replace('__RUNS__', repr(runs)).replace('__RUN_H__', repr(float(run_h)))
            .replace('__START_ATTEMPT__', repr(int(start_attempt)))
            .replace('__GUARD_EXIT__', repr(GUARD_EXIT)).replace('__NEXT_ATTEMPT__', inspect.getsource(next_attempt)))
    with open(os.path.join(d, name + '.py'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(code)
    meta = {'id': f'apexblue/{name}', 'title': name, 'code_file': f'{name}.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': False, 'keywords': ['gpu'], 'dataset_sources': DATASETS, 'kernel_sources': [],
            'competition_sources': [], 'model_sources': [], 'docker_image': DOCKER, 'machine_shape': 'NvidiaTeslaT4'}
    with open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['timing', 'run'])
    ap.add_argument('--tag')
    ap.add_argument('--args', default='')
    ap.add_argument('--run_h', type=float, default=3.8, help='projected hours of one run (100.8 retry budget)')
    ap.add_argument('--start_attempt', type=int, default=0, help='100.8 retry in a new kernel (review 090 C4)')
    a = ap.parse_args(argv)
    pins = build_src()
    if a.mode == 'timing':
        runs = [['timing_d256', ['--dev', '--timing_steps', '300', '--d_model', '256']],
                ['timing_d384', ['--dev', '--timing_steps', '300', '--d_model', '384']]]
        d = render('lincs-v10c-timing', runs, pins)
    else:
        if not a.tag:
            raise SystemExit('--tag is required for a run kernel')
        d = render('lincs-v10c-' + a.tag.replace('_', '-'), [[a.tag, shlex.split(a.args)]], pins, run_h=a.run_h,
                   start_attempt=a.start_attempt)
    print(json.dumps(pins, indent=1))
    print(d)


if __name__ == '__main__':
    main()
