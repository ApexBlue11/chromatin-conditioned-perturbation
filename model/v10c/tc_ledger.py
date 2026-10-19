"""The T-C ledger (RESULTS 100.11, review 091 Ask 4): one line per attempt of a downloaded T-C / gap kernel, logged at
download, so the readers can refuse a record that changed afterwards. A non-guard kernel death is logged with its reason
and re-run at the same seed and attempt; it never counts as a guard failure.

    python model/v10c/tc_ledger.py add external/kaggle_out/v10c_tc/tc_on_s0
    python model/v10c/tc_ledger.py death tc_on_s0 --attempt 1 --reason "OOM at step 1234 (log)" \
        --kernel_dir external/kaggle_out/v10c_tc/tc_on_s0
A dead kernel writes no V10C_KERNEL_COMPLETE.json, so `death` also logs that kernel's earlier guard-failed attempts from
their FAILED files (review 092 C5); the readers then take the dead kernel's attempts from the ledger. The re-run is a new
kernel named {slot}_d<k> with --start_attempt at the dead attempt, at the same seed.
"""
import argparse, datetime, glob, hashlib, json, os

LEDGER = 'model/results/v10c/tc_ledger.jsonl'


def _now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).isoformat(timespec='seconds')


def sha1_file(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def attempt_entries(kernel_dir):
    """One dict per attempt recorded in the kernel's V10C_KERNEL_COMPLETE.json."""
    kc = json.load(open(os.path.join(kernel_dir, 'V10C_KERNEL_COMPLETE.json'), encoding='utf-8'))
    out = []
    for slot, atts in kc['attempts'].items():
        for a in atts:
            sub = os.path.join(kernel_dir, os.path.basename(a['out'].rstrip('/')))
            seed = int(a['args'][a['args'].index('--seed') + 1]) if 'args' in a and '--seed' in a['args'] else None
            e = {'slot': slot, 'kernel_dir': os.path.basename(kernel_dir.rstrip('/\\')), 'attempt': a['attempt'],
                 'returncode': a['returncode'], 'seed': seed}
            if a['returncode'] == 0:
                recs = glob.glob(os.path.join(sub, 'V10C_COMPLETE_*.json'))
                if len(recs) != 1:
                    raise SystemExit(f'FATAL: attempt {a["attempt"]} of {slot} has {len(recs)} COMPLETE records')
                r = json.load(open(recs[0], encoding='utf-8'))
                e.update({'record': os.path.relpath(recs[0], kernel_dir).replace('\\', '/'), 'record_sha1': sha1_file(recs[0]),
                          'pred_sha1': r.get('pred_sha1'), 'tag': r.get('tag'),
                          'end_dose_scale_batch_median_mean': (r.get('hist') or [{}])[-1].get('dose_scale_batch_median_mean')})
            else:
                fails = glob.glob(os.path.join(sub, 'V10C_FAILED_*.json'))
                if fails:
                    f = json.load(open(fails[0], encoding='utf-8'))
                    e.update({'failed_record_sha1': sha1_file(fails[0]), 'guard_stop_step': f.get('step'),
                              'guard_level': f.get('guard_level')})
            out.append(e)
    return out


def add(kernel_dir, ledger=LEDGER):
    have = set()
    if os.path.exists(ledger):
        for line in open(ledger, encoding='utf-8'):
            e = json.loads(line)
            have.add((e.get('slot'), e.get('attempt'), e.get('kind', 'attempt')))
    new = [e for e in attempt_entries(kernel_dir) if (e['slot'], e['attempt'], 'attempt') not in have]
    with open(ledger, 'a', encoding='utf-8') as L:
        for e in new:
            L.write(json.dumps({'logged': _now(), 'kind': 'attempt', 'protocol': 'RESULTS 100.5 / 100.8 / 100.11', **e}) + '\n')
    return new


def death(slot, attempt, reason, kernel_dir=None, ledger=LEDGER):
    """Log a non-guard death of `attempt`; with kernel_dir, also log that kernel's earlier guard-failed attempts."""
    entries = []
    kd_name = os.path.basename(kernel_dir.rstrip('/\\')) if kernel_dir else None
    if kernel_dir:
        for k in range(attempt):
            sub = os.path.join(kernel_dir, slot if k == 0 else f'{slot}_retry{k}')
            fails = glob.glob(os.path.join(sub, 'V10C_FAILED_*.json'))
            if not fails:
                continue
            f = json.load(open(fails[0], encoding='utf-8'))
            seed = (f.get('args') or {}).get('seed')
            entries.append({'logged': _now(), 'kind': 'attempt', 'protocol': 'RESULTS 100.11 (from a dead kernel, 092 C5)',
                            'slot': slot, 'kernel_dir': kd_name, 'attempt': k, 'returncode': 3, 'seed': seed,
                            'failed_record_sha1': sha1_file(fails[0]), 'guard_stop_step': f.get('step'),
                            'guard_level': f.get('guard_level')})
    entries.append({'logged': _now(), 'kind': 'non_guard_death', 'slot': slot, 'kernel_dir': kd_name, 'attempt': attempt,
                    'reason': reason, 'action': 're-run at the same seed and attempt (100.11), kernel {slot}_d<k>'})
    with open(ledger, 'a', encoding='utf-8') as L:
        for e in entries:
            L.write(json.dumps(e) + '\n')
    return entries


def dead_kernel_attempts(slot, kernel_dir_name, ledger=LEDGER):
    """For a kernel logged as dead: its attempts from the ledger, as (attempt, returncode, None); None if not logged."""
    if not os.path.exists(ledger):
        return None
    rows = [json.loads(line) for line in open(ledger, encoding='utf-8')]
    if not any(e.get('kind') == 'non_guard_death' and e.get('slot') == slot and e.get('kernel_dir') == kernel_dir_name
               for e in rows):
        return None
    return [(e['attempt'], e['returncode'], None) for e in rows
            if e.get('kind') == 'attempt' and e.get('slot') == slot and e.get('kernel_dir') == kernel_dir_name]


def pred_sha1s(ledger=LEDGER):
    """(slot, attempt) -> pred_sha1 for every completed attempt in the ledger."""
    out = {}
    if os.path.exists(ledger):
        for line in open(ledger, encoding='utf-8'):
            e = json.loads(line)
            if e.get('kind') == 'attempt' and e.get('returncode') == 0:
                out[(e['slot'], e['attempt'])] = e.get('pred_sha1')
    return out


def ledger_sha1s(ledger=LEDGER):
    """(slot, attempt) -> record_sha1 for every completed attempt in the ledger."""
    out = {}
    if os.path.exists(ledger):
        for line in open(ledger, encoding='utf-8'):
            e = json.loads(line)
            if e.get('kind') == 'attempt' and e.get('returncode') == 0:
                out[(e['slot'], e['attempt'])] = e['record_sha1']
    return out


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
