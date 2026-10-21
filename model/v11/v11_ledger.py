"""The v11 ledger (RESULTS 102.1 Ask 6, as T-C's 100.11): one line per attempt of a downloaded v11 kernel, logged at
download, so the reader refuses a record or prediction file that changed afterwards. tc_ledger's rules with v11's file
names; a non-guard kernel death is logged with its reason and re-run at the same seed and attempt (kernel {slot}_d<k>).

    python model/v11/v11_ledger.py add external/kaggle_out/v11/v11_full_s0
    python model/v11/v11_ledger.py death v11_full_s0 --attempt 1 --reason "..." --kernel_dir external/kaggle_out/v11/v11_full_s0
"""
import argparse, datetime, glob, hashlib, json, os

LEDGER = 'model/results/v11/v11_ledger.jsonl'
KERNEL_DONE, COMPLETE, FAILED = 'V11_KERNEL_COMPLETE.json', 'V11_COMPLETE_*.json', 'V11_FAILED_*.json'


def _now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).isoformat(timespec='seconds')


def sha1_file(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def attempt_entries(kernel_dir):
    kc = json.load(open(os.path.join(kernel_dir, KERNEL_DONE), encoding='utf-8'))
    out = []
    for slot, atts in kc['attempts'].items():
        for a in atts:
            sub = os.path.join(kernel_dir, os.path.basename(a['out'].rstrip('/')))
            seed = int(a['args'][a['args'].index('--seed') + 1]) if '--seed' in a.get('args', []) else None
            e = {'slot': slot, 'kernel_dir': os.path.basename(kernel_dir.rstrip('/\\')), 'attempt': a['attempt'],
                 'returncode': a['returncode'], 'seed': seed}
            if a['returncode'] == 0:
                recs = glob.glob(os.path.join(sub, COMPLETE))
                if len(recs) != 1:
                    raise SystemExit(f'FATAL: attempt {a["attempt"]} of {slot} has {len(recs)} COMPLETE records')
                r = json.load(open(recs[0], encoding='utf-8'))
                e.update({'record': os.path.relpath(recs[0], kernel_dir).replace('\\', '/'), 'record_sha1': sha1_file(recs[0]),
                          'pred_sha1': r.get('pred_sha1'), 'tag': r.get('tag'),
                          'end_dose_scale_batch_median_mean': (r.get('hist') or [{}])[-1].get('dose_scale_batch_median_mean')})
            else:
                fails = glob.glob(os.path.join(sub, FAILED))
                if fails:
                    f = json.load(open(fails[0], encoding='utf-8'))
                    e.update({'failed_record_sha1': sha1_file(fails[0]), 'guard_stop_step': f.get('step'),
                              'guard_level': f.get('guard_level')})
            out.append(e)
    return out


def _rows(ledger):
    return [json.loads(line) for line in open(ledger, encoding='utf-8')] if os.path.exists(ledger) else []


def add(kernel_dir, ledger=LEDGER):
    have = {(e.get('slot'), e.get('attempt')) for e in _rows(ledger) if e.get('kind') == 'attempt'}
    new = [e for e in attempt_entries(kernel_dir) if (e['slot'], e['attempt']) not in have]
    os.makedirs(os.path.dirname(os.path.abspath(ledger)), exist_ok=True)
    with open(ledger, 'a', encoding='utf-8') as L:
        for e in new:
            L.write(json.dumps({'logged': _now(), 'kind': 'attempt', 'protocol': 'RESULTS 102.1 / 100.8 / 100.11', **e}) + '\n')
    return new


def death(slot, attempt, reason, kernel_dir=None, ledger=LEDGER):
    entries = []
    kd_name = os.path.basename(kernel_dir.rstrip('/\\')) if kernel_dir else None
    if kernel_dir:
        for k in range(attempt):
            sub = os.path.join(kernel_dir, slot if k == 0 else f'{slot}_retry{k}')
            fails = glob.glob(os.path.join(sub, FAILED))
            if not fails:
                continue
            f = json.load(open(fails[0], encoding='utf-8'))
            entries.append({'logged': _now(), 'kind': 'attempt', 'protocol': 'RESULTS 100.11 (from a dead kernel, 092 C5)',
                            'slot': slot, 'kernel_dir': kd_name, 'attempt': k, 'returncode': 3,
                            'seed': (f.get('args') or {}).get('seed'), 'failed_record_sha1': sha1_file(fails[0]),
                            'guard_stop_step': f.get('step'), 'guard_level': f.get('guard_level')})
    entries.append({'logged': _now(), 'kind': 'non_guard_death', 'slot': slot, 'kernel_dir': kd_name, 'attempt': attempt,
                    'reason': reason, 'action': 're-run at the same seed and attempt (100.11), kernel {slot}_d<k>'})
    os.makedirs(os.path.dirname(os.path.abspath(ledger)), exist_ok=True)
    with open(ledger, 'a', encoding='utf-8') as L:
        for e in entries:
            L.write(json.dumps(e) + '\n')
    return entries


def dead_kernel_attempts(slot, kernel_dir_name, ledger=LEDGER):
    rows = _rows(ledger)
    if not any(e.get('kind') == 'non_guard_death' and e.get('slot') == slot and e.get('kernel_dir') == kernel_dir_name
               for e in rows):
        return None
    return [(e['attempt'], e['returncode'], None) for e in rows
            if e.get('kind') == 'attempt' and e.get('slot') == slot and e.get('kernel_dir') == kernel_dir_name]


def completed_sha1s(ledger=LEDGER):
    """(slot, attempt) -> (record_sha1, pred_sha1) for every completed attempt."""
    return {(e['slot'], e['attempt']): (e['record_sha1'], e.get('pred_sha1')) for e in _rows(ledger)
            if e.get('kind') == 'attempt' and e.get('returncode') == 0}


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    a1 = sp.add_parser('add')
    a1.add_argument('kernel_dir')
    a2 = sp.add_parser('death')
    a2.add_argument('slot')
    a2.add_argument('--attempt', type=int, required=True)
    a2.add_argument('--reason', required=True)
    a2.add_argument('--kernel_dir')
    a = ap.parse_args()
    if a.cmd == 'add':
        for e in add(a.kernel_dir):
            print(json.dumps({k: e.get(k) for k in ('slot', 'attempt', 'returncode', 'seed', 'record_sha1', 'guard_level')}))
    else:
        death(a.slot, a.attempt, a.reason, kernel_dir=a.kernel_dir)


if __name__ == '__main__':
    main()
