# lincs-audit99-s0 : RESULTS 99 Phase 1, C3 (drug swap, P9) and C2b (x_cell, P7), seed 0. Kaggle CPU, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys
SEED = 0
PINS = {
 "audit99_infer.py": "14fa3c749ab141249bd7fd358514229c90627eee",
 "config_v6.py": "c3e4a18a767244074b7f454c0930735fb83308c1",
 "config_v9.py": "4b4fccca684dec3346a1d9abf09728dc8f3fc997",
 "dp_seeding.py": "69f1dad31500d48f111d9f05b0a13cd113d6562a",
 "model_v9.py": "f53c8b6a88457de7badd8dc3c6563e17de43151e",
 "modules.py": "0dd789a0b152bf5113298840505d81b0643082fd",
 "modules_v7.py": "8c7517e87fe9e3a29e807253095760b6ee40de1f",
 "modules_v9.py": "4140bf702be5b1a75d1f767a53d9290d8d905d22",
 "v9p7_seed0.pt": "9e94007ad60d5cd183ad56e14f2790451f475239",
 "v9p7_seed0_ident.npz": "3fab4a8ef20dd9e10d04cad2fbdd12b2efcfd7e3",
 "v9p9_seed0.pt": "d7fc87bfcfb0db457f1db6218d3583cfaca06a9d",
 "v9p9_seed0_ident.npz": "dc04548646b80ddd8f6a90f88b5325de891a3c3d",
 "xpert_arm.py": "c55ce25700044977d957611183377d0845055159"
}
RUNS = (('swap', 'v9p9'), ('xcell', 'v9p7'))
SRC, OUT = '/kaggle/working/src', '/kaggle/working/out'


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find_one(name):
    hits = glob.glob('/kaggle/input/**/' + name, recursive=True)
    if len(hits) != 1:
        raise SystemExit('FATAL: %s: expected one mounted copy, found %d' % (name, len(hits)))
    return hits[0]


os.makedirs(SRC, exist_ok=True)
paths = {}
for name, h in PINS.items():
    p = find_one(name)
    if sha1(p) != h:
        raise SystemExit('FATAL: %s sha1 %s != pinned %s' % (name, sha1(p), h))
    paths[name] = p
    if name.endswith('.py'):
        shutil.copy(p, os.path.join(SRC, name))
print('all %d pinned files verified' % len(PINS), flush=True)
for mode, run in RUNS:
    cmd = [sys.executable, '-u', os.path.join(SRC, 'audit99_infer.py'), '--mode', mode,
           '--ckpt', paths['%s_seed%d.pt' % (run, SEED)], '--identity', paths['%s_seed%d_ident.npz' % (run, SEED)],
           '--out', OUT, '--roots', '/kaggle/input']
    r = subprocess.run(cmd, cwd=SRC, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONUNBUFFERED='1'))
    if r.returncode:
        raise SystemExit('FATAL: %s exited %d' % (mode, r.returncode))
files = {f: sha1(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))}
want = {'%s_seed%d.%s' % (m, SEED, e) for m, _ in RUNS for e in ('npz', 'json')}
if set(files) != want:
    raise SystemExit('FATAL: outputs %s != expected %s' % (sorted(files), sorted(want)))
with open(os.path.join(OUT, 'AUDIT99_COMPLETE_s%d.json' % SEED), 'w') as f:
    json.dump({'complete': True, 'seed': SEED, 'files': files, 'pins': PINS}, f, indent=1)
shutil.rmtree(SRC)
print('AUDIT99 complete, seed %d: %d files' % (SEED, len(files)), flush=True)
