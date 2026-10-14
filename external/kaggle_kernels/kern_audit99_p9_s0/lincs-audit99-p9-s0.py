# lincs-audit99-p9-s0 : RESULTS 99 Phase 1, C3 drug swap + the seen-cell x_cell bound on P9, seed 0. Kaggle CPU, internet off.
import glob, hashlib, json, os, shutil, subprocess, sys
SEED, TAG = 0, 'p9'
MODE, RUN = 'swap', 'v9p9'
PINS = {
 "E_final.npy": "20e9a3e4ce56f0d8a938c185dc5ca9e016590aac",
 "E_final_mask.npy": "6186bbdd9edae48d0382b5f269c600ff455fdb2f",
 "M_pathway_v9.npy": "c752c9c25a066af63cbe9595a6565e026441765c",
 "STRING_adj_978_v9.npy": "609e289227b4e96da09dcc347667deb5f3645a8a",
 "audit99_infer.py": "aca6e4cf83680f29d96b2a4d1f959e0eb90ea47d",
 "cell_lineage.npy": "1efd3dea686816db00b10ca426ed2e54d533c75d",
 "config_v6.py": "c3e4a18a767244074b7f454c0930735fb83308c1",
 "config_v9.py": "4b4fccca684dec3346a1d9abf09728dc8f3fc997",
 "dp_seeding.py": "69f1dad31500d48f111d9f05b0a13cd113d6562a",
 "drug_atom_offsets.npy": "b93654fc21a633a4ab2e757e9e9fe25d3762c374",
 "drug_atom_reprs.npy": "2f4985edf82abe1b8432bc551363423d8dee872a",
 "drug_descriptors.npy": "a2f9c17d23d23568f25b27d92e6f74b975e36ebd",
 "drug_feature_index.json": "0b812337aa4dfce1e00d5d86e2dba3cb3f9c0a32",
 "drug_fingerprints.npy": "8ba2da5982d67d8424b90e854086bad6b786d221",
 "drug_unimol.npy": "750385b1f2aa6c65167fe3edd10aa1af919f9e7b",
 "gene_vectors_978.npy": "f13cc890deb0f8684602d7776828a9509d87f218",
 "lincs_cell_index.json": "5f7bc6b42fe45c56f62a27d06c1925a077056e6d",
 "model_v9.py": "f53c8b6a88457de7badd8dc3c6563e17de43151e",
 "modules.py": "0dd789a0b152bf5113298840505d81b0643082fd",
 "modules_v7.py": "8c7517e87fe9e3a29e807253095760b6ee40de1f",
 "modules_v9.py": "4140bf702be5b1a75d1f767a53d9290d8d905d22",
 "v9p9_seed0.pt": "d7fc87bfcfb0db457f1db6218d3583cfaca06a9d",
 "v9p9_seed0_ident.npz": "dc04548646b80ddd8f6a90f88b5325de891a3c3d",
 "xpert_arm.py": "c55ce25700044977d957611183377d0845055159",
 "xpert_mdmt_splits.npz": "1444b253d48c5adf066bf7a3cc0ba8ea5fb71464"
}
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
print('all %d pinned files verified, each the only mounted copy' % len(PINS), flush=True)
cmd = [sys.executable, '-u', os.path.join(SRC, 'audit99_infer.py'), '--mode', MODE,
       '--ckpt', paths['%s_seed%d.pt' % (RUN, SEED)], '--identity', paths['%s_seed%d_ident.npz' % (RUN, SEED)],
       '--out', OUT, '--roots', '/kaggle/input']
r = subprocess.run(cmd, cwd=SRC, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONUNBUFFERED='1'))
if r.returncode:
    raise SystemExit('FATAL: %s exited %d' % (MODE, r.returncode))
files = {f: sha1(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))}
want = {'%s_seed%d.%s' % (MODE, SEED, e) for e in ('npz', 'json')}
if set(files) != want:
    raise SystemExit('FATAL: outputs %s != expected %s' % (sorted(files), sorted(want)))
with open(os.path.join(OUT, 'AUDIT99_COMPLETE_%s_s%d.json' % (TAG, SEED)), 'w') as f:
    json.dump({'complete': True, 'seed': SEED, 'tag': TAG, 'mode': MODE, 'files': files, 'pins': PINS}, f, indent=1)
shutil.rmtree(SRC)
print('AUDIT99 complete, %s seed %d: %d files' % (TAG, SEED, len(files)), flush=True)
