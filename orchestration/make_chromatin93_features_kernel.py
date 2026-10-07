# -*- coding: utf-8 -*-
"""RESULTS 93 Stage 1b data (93.3, 93.5 C1/C5, 93.8): one free Kaggle CPU kernel WITH internet, outcome-free (no LINCS
response is read). PI glue.

    python orchestration/make_chromatin93_features_kernel.py

Pins every input's sha1: the seven step10 inputs (dataset `apexblue/lincs-c93-inputs`) and the code files (dataset
`apexblue/lincs-chromatin-funnel`, a version that adds chromatin_features93.py). Pins the package versions the probe kernel
`lincs-c93-probe` verified, and that probe's docker image. Runs `selftest()` first and aborts on any failure; then the module's
`main`; then writes C93_FEATURES_RUN_COMPLETE.json with every output's sha1.
"""
import hashlib
import io
import json
import os

ROOT = r'C:\Projects\LINCS'
SRC = os.path.join(ROOT, 'external', 'kaggle_chromatin_src')
INP = os.path.join(ROOT, 'external', 'kaggle_c93_inputs')
CODE_FILES = {'chromatin_features93.py': r'model\v9', 'chromatin_funnel.py': r'model\v9', 'score_dev.py': r'model\v9'}
INPUT_FILES = {'coverage_report_phase2.tsv': r'phase2_assembly\outputs', 'cistrome_human_samples.json': r'epigenetics\data',
               'E_peaks_log.txt': r'phase2_assembly\outputs', 'tss_hg38.tsv': r'phase2_assembly\outputs',
               'pathway_landmark_genes.txt': r'baseline\Network Data', 'chip_atlas_human_epi.tab': r'phase2_assembly\outputs',
               'lincs_cell_index.json': r'baseline\outputs\ccle_baseline_lincs_v5'}
PKGS = ['MOODS-python==1.9.4.1', 'py2bit==1.0.1', 'pyjaspar==4.0.0', 'pyranges==0.1.4', 'decoupler==2.2.0']
IMAGE = 'gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a'
TWOBIT = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit'
SLUG = 'lincs-chromatin93-features'

CODE = r'''# RESULTS 93 Stage 1b data (pre-registered 93.3, amended 93.5, 93.8): outcome-free; CPU + internet. Run once.
import glob, hashlib, json, os, subprocess, sys, time
CODE_PINS = __CODE_PINS__
INPUT_PINS = __INPUT_PINS__
PKGS = __PKGS__
t0 = time.time()
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + PKGS, capture_output=True, text=True)
print('pip rc', r.returncode, r.stderr[-800:], flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: pip install failed')
sha = lambda p: hashlib.sha1(open(p, 'rb').read()).hexdigest()
def find(name, ds):
    hit = glob.glob('/kaggle/input/**/%s/**/%s' % (ds, name), recursive=True) or           glob.glob('/kaggle/input/%s/%s' % (ds, name))
    hit = sorted(set(hit))
    if len(hit) != 1:
        raise SystemExit('FATAL: %s found %d times in %s' % (name, len(hit), ds))
    return hit[0]
paths = {}
for f, h, ds in [(f, h, 'lincs-chromatin-funnel') for f, h in CODE_PINS.items()] +                 [(f, h, 'lincs-c93-inputs') for f, h in INPUT_PINS.items()]:
    paths[f] = find(f, ds)
    if sha(paths[f]) != h:
        raise SystemExit('FATAL: %s sha1 %s != pinned %s' % (f, sha(paths[f]), h))
print('inputs verified: %d code + %d data files' % (len(CODE_PINS), len(INPUT_PINS)), flush=True)
SRC = os.path.dirname(paths['chromatin_features93.py'])
sys.path.insert(0, SRC)
import importlib.metadata as md
versions = {p.split('==')[0]: md.version(p.split('==')[0]) for p in PKGS}
assert all(versions[p.split('==')[0]] == p.split('==')[1] for p in PKGS), versions
import chromatin_features93 as c93
c93.selftest()
print('selftest passed', flush=True)
OUT = '/kaggle/working/c93'
CACHE = '/tmp/c93_cache'
args = ['--cov_tsv', paths['coverage_report_phase2.tsv'], '--cistrome_json', paths['cistrome_human_samples.json'],
        '--peaks_log', paths['E_peaks_log.txt'], '--tss', paths['tss_hg38.tsv'], '--genes', paths['pathway_landmark_genes.txt'],
        '--cell_index', paths['lincs_cell_index.json'], '--chipatlas_list', paths['chip_atlas_human_epi.tab'],
        '--twobit', '__TWOBIT__', '--cache', CACHE, '--out_dir', OUT, '--threads', '4']
env = dict(os.environ, PYTHONUTF8='1')
r = subprocess.run([sys.executable, '-u', paths['chromatin_features93.py']] + args, env=env)
print('chromatin_features93.py exit %d after %.0f s' % (r.returncode, time.time() - t0), flush=True)
if r.returncode != 0:
    raise SystemExit('FATAL: chromatin_features93.py failed')
outs = {f: sha(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))}
json.dump({'code_pins': CODE_PINS, 'input_pins': INPUT_PINS, 'packages': versions, 'python': sys.version,
           'outputs': outs, 'complete': True}, open('/kaggle/working/C93_FEATURES_RUN_COMPLETE.json', 'w'), indent=1)
print('done', flush=True)
'''


def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def same(a, b):
    return open(a, 'rb').read().replace(b'\r\n', b'\n') == open(b, 'rb').read().replace(b'\r\n', b'\n')


def main():
    code_pins, input_pins = {}, {}
    for f, d in CODE_FILES.items():
        p = os.path.join(SRC, f)
        assert same(p, os.path.join(ROOT, d, f)), 'staged %s differs from the repo' % f
        code_pins[f] = sha1(p)
    for f, d in INPUT_FILES.items():
        p = os.path.join(INP, f)
        assert sha1(p) == sha1(os.path.join(ROOT, d, f)), 'staged %s differs from the repo copy' % f
        input_pins[f] = sha1(p)
    kdir = os.path.join(ROOT, 'external', 'kaggle_kernels', 'kern_chromatin93_features')
    os.makedirs(kdir, exist_ok=True)
    code = (CODE.replace('__CODE_PINS__', repr(code_pins)).replace('__INPUT_PINS__', repr(input_pins))
            .replace('__PKGS__', repr(PKGS)).replace('__TWOBIT__', TWOBIT))
    io.open(os.path.join(kdir, SLUG + '.py'), 'w', encoding='utf-8', newline='\n').write(code)
    meta = {'id': 'apexblue/' + SLUG, 'title': 'lincs chromatin93 features', 'code_file': SLUG + '.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': True,
            'keywords': [], 'dataset_sources': ['apexblue/lincs-chromatin-funnel', 'apexblue/lincs-c93-inputs'],
            'kernel_sources': [], 'competition_sources': [], 'model_sources': [], 'docker_image': IMAGE}
    io.open(os.path.join(kdir, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n').write(json.dumps(meta, indent=1) + '\n')
    print('wrote', kdir)
    print('code pins', code_pins)
    print('input pins', input_pins)


if __name__ == '__main__':
    main()
