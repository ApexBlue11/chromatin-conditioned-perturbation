# -*- coding: utf-8 -*-
"""RESULTS 97.2 / 97.6: the O9 kernel -- XPert trained to its published recipe on split_cold_drug_1 -- built from the O2
session-1 kernel AS RUN (git f252b12) by asserted exact substitutions of the fold constants only. PI glue.

    python orchestration/make_o9_kernel.py SESSION [PREV_JSON]

PREV_JSON (session k > 1) is session k-1's handoff {"sha1": {...}, "final_epoch": e, "torch": v, "cuda": v}, pasted into the
kernel as a LITERAL and committed before the push (O2's chain of custody, §84.2 C3). Session k > 1 also mounts the private
state dataset apexblue/xpert-cd1-state.
"""
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_REV, BASE_PATH = 'f252b12', 'external/kaggle_kernels/kern_xpert_cc1/lincs-xpert-cc1.py'
BASE_SHA1 = 'e9d212832dd9'                       # sha1 prefix of the O2 session-1 kernel blob as git shows it
OUT_DIR = os.path.join(REPO, 'external', 'kaggle_kernels', 'kern_xpert_cd1')
SLUG, STATE_DS = 'lincs-xpert-cd1', 'apexblue/xpert-cd1-state'

SESSION = int(sys.argv[1])
PREV = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
assert (SESSION == 1) == (PREV is None), 'session 1 has no PREV; every later session must have one'
if PREV is not None:
    assert set(PREV) == {'sha1', 'final_epoch', 'torch', 'cuda'}, sorted(PREV)

s = subprocess.run(['git', 'show', '%s:%s' % (BASE_REV, BASE_PATH)], cwd=REPO, capture_output=True, check=True).stdout
assert hashlib.sha1(s).hexdigest().startswith(BASE_SHA1), 'the O2 session-1 base is not the kernel that ran'
s = s.decode('utf-8')


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new, 1)


rep('Train XPert on split_cold_cell_1 AS PUBLISHED, then predict its test rows with our row-indexed harness.',
    'O9 [RESULTS 97]: train XPert on split_cold_drug_1 AS PUBLISHED, then predict its test rows with our row-indexed harness.\n'
    "Built from O2's session-1 kernel (git f252b12) by orchestration/make_o9_kernel.py: fold constants only.")
rep("Predictions are scored later by head_to_head_mdmt.py against v9's split_cold_cell_1 predictions.",
    "Predictions are scored later by model/v9/score_p9.py --ref xpert_o9 against P9's split_cold_drug_1 predictions [97.4].")
rep('# O2 PRODUCTION [RESULTS 84, review 015]. GUARD G / H (RESULTS 81) were proved in v7 [81.7] and are not rerun.',
    "# O9 PRODUCTION [RESULTS 97], on O2's machinery [RESULTS 84, review 015]. GUARD G / H (RESULTS 81) were proved in v7\n"
    '# [81.7] and are not rerun.')
rep('SESSION = 1\n', 'SESSION = %d\n' % SESSION)
rep('PREV = None\n', 'PREV = %r\n' % (PREV,))
rep('# Production session 1. The v7 proof', '# Production session %d. The v7 proof' % SESSION)
rep("FOLD = 'split_cold_cell_1'", "FOLD = 'split_cold_drug_1'")
rep("if counts != {'train': 47509, 'test': 21321}:\n    fatal('%s levels are %s, expected {train: 47509, test: 21321}.' % (FOLD, counts))",
    "if counts != {'train': 55385, 'test': 13445}:\n    fatal('%s levels are %s, expected {train: 55385, test: 13445}.' % (FOLD, counts))")
rep('''# Only the fold list is reduced, to split_cold_cell_1: their loop builds a fresh model, init_weights() and
# optimizer PER FOLD (train_xpert.py:425-455), so the folds are independent. Their seed is set once at
# :402, before that loop, so our fold starts from a fresh seed-2024 state rather than the state their
# split_cold_drug_1 run left -- a seed difference, not a recipe difference. Disclosed.''',
    '''# Only the fold list is reduced, to split_cold_drug_1 -- the FIRST fold of their published list: their loop
# builds a fresh model, init_weights() and optimizer PER FOLD (train_xpert.py:425-455), and their seed is set
# once at :402, before that loop, so this fold starts from the same seed-2024 state as their own run of it
# (RESULTS 97.1): no seed difference at initialisation. The trajectory is not a replication (O2's disclosed
# deviations: DataParallel over two T4s, ten frozen parameters, full-state resumes, cuDNN, GPU type).''')
rep("RECORD['seed_note'] = ('seed 2024 set once at train_xpert.py:402 before the fold loop, so this fold starts '\n"
    "                       'from a fresh seed-2024 state, not the state their 3-fold sequential run would reach')",
    "RECORD['seed_note'] = ('seed 2024 set once at train_xpert.py:402 before the fold loop; split_cold_drug_1 is the first '\n"
    "                       'fold of their published list (scripts/train.sh), so this fold starts from the same seed-2024 '\n"
    "                       'state as theirs: no seed difference at initialisation; the trajectory is not a replication')")
rep('# O2 PRODUCTION SESSION [RESULTS 84; review 015]', "# O9 PRODUCTION SESSION [RESULTS 97; O2's machinery, RESULTS 84; review 015]")
rep("if 'row_index' not in prof or len(prof['row_index']) != 21321:", "if 'row_index' not in prof or len(prof['row_index']) != 13445:")
rep("log('C4 chain test PASSED: marker -> termination -> counter_at_end == patience -> 21321-row prediction')",
    "log('C4 chain test PASSED: marker -> termination -> counter_at_end == patience -> 13445-row prediction')")
rep("sess['amendment_E'] = {'first_two_mean_s': round(first2, 1), 'projection_s': 482.2,\n"
    "                           'reprice_before_session_2': first2 > 1.25 * 482.2}",
    "sess['amendment_E'] = {'first_two_mean_s': round(first2, 1), 'projection_s': 562.1,   # 97.2; an upper bound, 97.6\n"
    "                           'reprice_before_session_2': first2 > 1.25 * 562.1}")
rep("RECORD['framing'] = ('XPert trained to its published recipe on %s. NOT a reproduction of their cold-cell '\n"
    "                         'run: an independent draw of the recipe.' % FOLD)",
    "RECORD['framing'] = ('XPert trained to its published recipe on %s, from their seed-2024 state for this fold; '\n"
    "                         'not a replication of their trajectory (O2 deviations, RESULTS 97.1/97.6).' % FOLD)")

# 97.6 item 1: no fold literal of O2 may survive outside the allow-list
ALLOW = ['#          --nfold split_cold_drug_1,split_cold_cell_1,split_1 --dataset l1000_mdmt']   # their published command
BAD = re.compile(r'cold[_ -]cell|cc1|47509|21321|21,321|482\.2', re.I)
left = [(i + 1, l) for i, l in enumerate(s.split('\n')) if BAD.search(l) and l.strip() not in [a.strip() for a in ALLOW]]
assert not left, 'O2 literals left in the O9 kernel: %r' % left[:5]
print('allow-listed historical lines kept:', ALLOW)
ast.parse(s)

os.makedirs(OUT_DIR, exist_ok=True)
io.open(os.path.join(OUT_DIR, SLUG + '.py'), 'w', encoding='utf-8', newline='\n').write(s)
meta = {'id': 'apexblue/' + SLUG, 'title': 'lincs xpert cd1', 'code_file': SLUG + '.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'keywords': ['gpu'], 'dataset_sources': ['apexblue/xpert-train-src'] + ([STATE_DS] if SESSION > 1 else []),
        'kernel_sources': [], 'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}
json.dump(meta, open(os.path.join(OUT_DIR, 'kernel-metadata.json'), 'w'), indent=1)
print('O9 kernel, session %d: %s (%d lines), parses; metadata with %s'
      % (SESSION, os.path.join(OUT_DIR, SLUG + '.py'), s.count('\n'), meta['dataset_sources']))
