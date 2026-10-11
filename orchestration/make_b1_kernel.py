# -*- coding: utf-8 -*-
"""RESULTS 98.2 B1 / 98.6 C4: the B1 kernel -- v9 retrained DRUG-BLIND (P9's command plus --drug_blind) on split_cold_drug_1,
seeds 0-2, blinded. PI glue: P9's kernel template is taken from make_p9_kernel.py AS COMMITTED and changed only by asserted
exact substitutions (the flag, B1's output and manifest names, its source dataset, and two B1 guards).

    python orchestration/make_b1_kernel.py            # P7's stack c6_v2, exactly as P9
"""
import hashlib
import io
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P9_BUILDER = os.path.join(REPO, 'orchestration', 'make_p9_kernel.py')
ROOT = os.path.join(REPO, 'external', 'kaggle_kernels')
UP = os.path.join(REPO, 'external', 'kaggle_v9_src_b1')
SLUG, SRC_DS = 'lincs-v9p10b1', 'apexblue/lincs-v9-src-b1'

src = io.open(P9_BUILDER, encoding='utf-8').read()
i = src.index("CODE = r'''") + len("CODE = r'''")
j = src.index("'''\n\nstack = sys.argv[1]")
code = src[i:j]
p9 = src
IMG = p9.split("IMG = '")[1].split("'")[0]
FLAGS = ['--sign_head_w', '0.492066', '--snapshot_cycles', '3']          # P9's stack c6_v2 (make_p9_kernel.STACKS)
assert "'c6_v2': ['--sign_head_w', '0.492066', '--snapshot_cycles', '3']" in p9


def rep(old, new, count=1):
    global code
    n = code.count(old)
    assert n == count, (n, old[:80])
    code = code.replace(old, new)


rep('# RESULTS 96.2 (pre-registered): P9, v9 -- stack __STACK__ -- on all training rows of split_cold_drug_1 (P7\'s recipe),',
    '# RESULTS 98.2 B1 (pre-registered): v9 DRUG-BLIND (--drug_blind) -- stack __STACK__ -- on all training rows of '
    'split_cold_drug_1 (P9\'s command),')
rep("        'sign_head_w', 'snapshot_cycles', 'def wsd_factor', 'no_test_metrics')",
    "        'sign_head_w', 'snapshot_cycles', 'def wsd_factor', 'no_test_metrics', 'drug_blind', 'def assert_drug_constant',\n"
    "        'def drug_permutation_check')")
rep("'--no_test_metrics', '--save_pred', STAGE + '/v9p9.npz', '--save_ckpt', STAGE + '/v9p9.pt']",
    "'--no_test_metrics', '--drug_blind', '--save_pred', STAGE + '/v9p10b1.npz', '--save_ckpt', STAGE + '/v9p10b1.pt']")
rep("print('P9 argv:', ' '.join(argv), flush=True)", "print('B1 argv:', ' '.join(argv), flush=True)")
rep("log = open(STAGE + '/p9_arm.log', 'w', encoding='utf-8')", "log = open(STAGE + '/b1_arm.log', 'w', encoding='utf-8')")
rep("""if n_leak:
    fail('%d captured line(s) carried a Pearson value despite --no_test_metrics' % n_leak)
""", """if n_leak:
    fail('%d captured line(s) carried a Pearson value despite --no_test_metrics' % n_leak)
# GUARD B1a (98.6 C4b): the arm ran the permutation check on its TRAINED model after every seed, and each passed
_chk = [l for l in open(STAGE + '/b1_arm.log', encoding='utf-8') if l.startswith('DRUGBLIND PERMUTATION CHECK seed ')]
print('GUARD B1a permutation checks:', len(_chk), flush=True)
if len(_chk) != N_SEEDS or not all(l.rstrip().endswith(' OK') for l in _chk):
    fail('GUARD B1a: expected %d passing DRUGBLIND PERMUTATION CHECK lines, found %r' % (N_SEEDS, _chk))
""")
rep("""if not rec.get('candidate_flags', {}).get('no_test_metrics'):
    fail('arm JSON does not record no_test_metrics')
""", """if not rec.get('candidate_flags', {}).get('no_test_metrics'):
    fail('arm JSON does not record no_test_metrics')
# GUARD B1b (98.2 B1): the arm JSON records the drug-blind run and what the constant atom pooled
if rec['candidate_flags'].get('drug_blind') is not True or not rec['candidate_flags'].get('drug_blind_info'):
    fail('GUARD B1b: arm JSON does not record drug_blind with its info')
print('GUARD B1b drug_blind_info', rec['candidate_flags']['drug_blind_info'], flush=True)
""")
rep("preds = sorted(glob.glob(STAGE + '/v9p9*_seed*.npz'))", "preds = sorted(glob.glob(STAGE + '/v9p10b1*_seed*.npz'))")
rep("done = os.path.join(WORKDIR, 'P9_COMPLETE.json')", "done = os.path.join(WORKDIR, 'P10B1_COMPLETE.json')")
rep("print('P9 complete: %d files + P9_COMPLETE.json; scored once locally per RESULTS 96.3' % len(manifest),",
    "print('B1 complete: %d files + P10B1_COMPLETE.json; scored once locally by score_p10.py (RESULTS 98.6)' % len(manifest),")
assert 'v9p9' not in code.replace('v9p10b1', ''), [l for l in code.splitlines() if 'v9p9' in l.replace('v9p10b1', '')]

# the mounted bytes, pinned (review 035 C3); each must equal the repo file up to line endings
REPO_FILES = {'xpert_arm.py': r'model\v9', 'model_v9.py': r'model\v9', 'modules_v9.py': r'model\v9', 'config_v9.py': r'model\v9',
              'dp_seeding.py': r'model\v9', 'modules_v7.py': r'model\v7'}
pins = {}
for f, sub in REPO_FILES.items():
    up = open(os.path.join(UP, f), 'rb').read()
    rp = open(os.path.join(REPO, sub, f), 'rb').read()
    assert up.replace(b'\r\n', b'\n') == rp.replace(b'\r\n', b'\n'), 'upload of %s differs from the repo' % f
    pins[f] = hashlib.sha1(up).hexdigest()
code = code.replace('__STACK__', 'P2 + C6 + V2 (P9) + DRUG-BLIND').replace('__FLAGS__', repr(FLAGS)).replace('__PINS__', repr(pins))
compile(code, SLUG, 'exec')
d = os.path.join(ROOT, 'kern_v9p10b1')
os.makedirs(d, exist_ok=True)
io.open(os.path.join(d, SLUG + '.py'), 'w', encoding='utf-8', newline='\n').write(code)
meta = {'id': 'apexblue/' + SLUG, 'title': 'lincs v9p10b1', 'code_file': SLUG + '.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
        'keywords': ['gpu'], 'dataset_sources': ['apexblue/lincs-v9-bundle', SRC_DS, 'apexblue/xpert-mdmt-benchmark'],
        'kernel_sources': [], 'competition_sources': [], 'model_sources': [], 'docker_image': IMG, 'machine_shape': 'NvidiaTeslaT4'}
io.open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n').write(json.dumps(meta, indent=1) + '\n')
print('wrote', d, 'pins', {k: v[:8] for k, v in pins.items()})
