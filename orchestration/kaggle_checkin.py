"""The two-account Kaggle check-in: status of every queued kernel; each newly COMPLETE kernel is downloaded and logged in
its ledger at download (tc_ledger for tc_ / gap_, v11_ledger for v11_); an ERROR kernel's log tail is shown. Prints
return codes and guard levels only, never a score (nothing is read before its protocol says so). Never prints a key.

    .venv-cuda/Scripts/python.exe orchestration/kaggle_checkin.py
"""
import glob, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, '.venv-cuda', 'Scripts', 'python.exe')
ACCOUNTS = {'main': {'config': None, 'root': 'external/kaggle_kernels', 'list': 'external/reset_push_list.txt',
                     'log': 'external/v10c_tb_sequencer.log'},
            'alt': {'config': r'C:\Users\Surya\.kaggle_alt', 'root': 'external/kaggle_kernels_alt',
                    'list': 'external/alt_push_list.txt', 'log': 'external/alt_sequencer.log'},
            'alt2': {'config': 'C:/Users/Surya/.kaggle_alt2', 'root': 'external/kaggle_kernels_alt2',
                     'list': 'external/alt2_push_list.txt', 'log': 'external/alt2_sequencer.log'}}


def env_for(config):
    e = dict(os.environ)
    e.pop('KAGGLE_CONFIG_DIR', None)
    if config:
        e['KAGGLE_CONFIG_DIR'] = config
    return e


def kaggle(args, config, timeout=600):
    r = subprocess.run(['kaggle'] + args, capture_output=True, text=True, env=env_for(config), timeout=timeout)
    return (r.stdout + r.stderr).strip()


def status(kid, config):
    out = kaggle(['kernels', 'status', kid], config, 90).splitlines()
    last = out[-1] if out else ''
    if 'KernelWorkerStatus.' in last:
        return last.split('KernelWorkerStatus.')[1].strip('"\' ')
    return 'NOT_PUSHED' if ('404' in last or 'denied' in last) else 'UNKNOWN: ' + last[:120]


def slot_of(k):
    return k.replace('kern_v10c_', '').replace('kern_', '')


def out_dir(slot):
    return os.path.join(REPO, 'external', 'kaggle_out', 'v11' if slot.startswith('v11_') else 'v10c_tc', slot)


def done_file(slot):
    return 'V11_KERNEL_COMPLETE.json' if slot.startswith('v11_') else 'V10C_KERNEL_COMPLETE.json'


def complete_download(slot, d):
    """The kernel summary exists and every completed attempt's folder has its COMPLETE record (a download can stop early
    and leave only the summary, as v11_no_targets_s0's first one did)."""
    f = os.path.join(d, done_file(slot))
    if not os.path.exists(f):
        return False
    rec = 'V11_COMPLETE_*.json' if slot.startswith('v11_') else 'V10C_COMPLETE_*.json'
    for atts in json.load(open(f, encoding='utf-8'))['attempts'].values():
        for x in atts:
            if x['returncode'] == 0 and not glob.glob(os.path.join(d, os.path.basename(x['out'].rstrip('/')), rec)):
                return False
    return True


def ledger_add(slot, d):
    tool = 'model/v11/v11_ledger.py' if slot.startswith('v11_') else 'model/v10c/tc_ledger.py'
    r = subprocess.run([PY, tool, 'add', d], capture_output=True, text=True, cwd=REPO)
    return (r.stdout + r.stderr).strip()


def log_tail(d, n=15):
    logs = glob.glob(os.path.join(d, '*.log'))
    if not logs:
        return '(no log)'
    try:
        txt = ''.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        txt = open(logs[0], encoding='utf-8', errors='replace').read()
    return '\n'.join(txt.splitlines()[-n:])


def main():
    report, changed = [], []
    for acct, a in ACCOUNTS.items():
        kernels = [x.strip() for x in open(os.path.join(REPO, a['list']), encoding='utf-8') if x.strip()]
        counts = {}
        for k in kernels:
            kid = json.load(open(os.path.join(REPO, a['root'], k, 'kernel-metadata.json')))['id']
            st = status(kid, a['config'])
            counts[st] = counts.get(st, 0) + 1
            slot = slot_of(k)
            d = out_dir(slot)
            if st == 'COMPLETE' and not complete_download(slot, d):
                os.makedirs(d, exist_ok=True)
                for _ in range(3):                                   # a download can stop early; retry it
                    kaggle(['kernels', 'output', kid, '-p', d], a['config'], 1800)
                    if complete_download(slot, d):
                        break
                if complete_download(slot, d):
                    out = ledger_add(slot, d)
                    lines = [ln for ln in out.splitlines() if ln.startswith('{')]
                    report.append(f'{acct} {slot}: downloaded + ledgered: ' + '; '.join(
                        f"attempt {e.get('attempt')} rc {e.get('returncode')} seed {e.get('seed')} guard {e.get('guard_level')}"
                        for e in map(json.loads, lines)) if lines else f'{acct} {slot}: downloaded; ledger said: {out[-300:]}')
                    changed.append(slot)
                elif os.path.exists(os.path.join(d, done_file(slot))):
                    report.append(f'{acct} {slot}: download still incomplete after 3 tries; retry next check-in')
                else:
                    report.append(f'{acct} {slot}: COMPLETE but no {done_file(slot)} (kernel died?) -- log tail:\n'
                                  + log_tail(d))
            elif st in ('ERROR', 'CANCEL_ACKNOWLEDGED') and not os.path.exists(os.path.join(d, '.error_seen')):
                os.makedirs(d, exist_ok=True)
                kaggle(['kernels', 'output', kid, '-p', d], a['config'], 1800)
                open(os.path.join(d, '.error_seen'), 'w').close()
                report.append(f'{acct} {slot}: {st} -- log tail:\n' + log_tail(d))
        seq = open(os.path.join(REPO, a['log']), encoding='utf-8').read().splitlines()[-1:] if os.path.exists(
            os.path.join(REPO, a['log'])) else []
        report.insert(0, f'{acct}: ' + ', '.join(f'{v} {s}' for s, v in sorted(counts.items())) +
                      (f" | sequencer: {seq[0][:140]}" if seq else ''))
    print('\n'.join(report))
    print('CHANGED ' + ' '.join(changed))


if __name__ == '__main__':
    main()
