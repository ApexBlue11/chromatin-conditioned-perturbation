"""T-C, read once when every T-C slot has a completed attempt or has exhausted its retries (RESULTS 100.5's rule; 100.8's
first-completed-attempt; review 090 C2's wording and counts). Written before any T-C run exists.

  per clean test cell c (BJAB, CD34, HS578T, HT29, MCF7, MDAMB231, THP1): the 3-seed mean of the per-row mean contrast r;
  delta_c = on - off.
  "Cell-specific chromatin contributes on unseen cells (among runs whose drug path trained, 100.8)" iff
      >= 5 of the 7 delta_c > 0  AND  the cell-cluster bootstrap 95 % CI of mean delta_c (the 7 cells resampled, 20,000
      draws, np.random.default_rng(0)) excludes 0 above.
  The mirror ("chromatin hurts on unseen cells"): >= 5 of 7 delta_c < 0 and the CI excludes 0 below. Otherwise no claim.
  A slot with no completed attempt -> no claim (100.8).
  Reported, never read: delta r on the same rows; the row-pooled means; H1975 apart; the permutations; the single-control
  count; every attempt per arm and per seed (collapse counts).

    python model/v10c/tc_read.py --runs_dir external/kaggle_out/v10c_tc --out model/results/v10c/tc_reading.json
"""
import argparse, glob, hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tc_ledger  # noqa: E402

CLEAN = ('BJAB', 'CD34', 'HS578T', 'HT29', 'MCF7', 'MDAMB231', 'THP1')
APART = ('H1975',)
ARMS, SEEDS = ('on', 'off'), (0, 1, 2)
N_BOOT, MIN_FAV = 20000, 5


def slot_attempts(runs_dir, arm, seed, prefix='tc'):
    """Every attempt of slot (arm, seed) over its kernels ({prefix}_{arm}_s{seed}, and any new-kernel retries
    {prefix}_{arm}_s{seed}_r{k}), ordered by attempt index. Each item: (attempt, returncode, record path or None)."""
    out = []
    slot = f'{prefix}_{arm}_s{seed}'
    for kd in slot_kernel_dirs(runs_dir, slot):
        kc = os.path.join(kd, 'V10C_KERNEL_COMPLETE.json')
        if not os.path.exists(kc):
            raise SystemExit(f'FATAL: {kc} missing (download the kernel output first)')
        for a in json.load(open(kc, encoding='utf-8'))['attempts'][slot]:
            sub = os.path.join(kd, os.path.basename(a['out'].rstrip('/')))
            recs = glob.glob(os.path.join(sub, 'V10C_COMPLETE_*.json'))
            out.append((a['attempt'], a['returncode'], recs[0] if a['returncode'] == 0 and len(recs) == 1 else None))
    out.sort(key=lambda x: x[0])
    if [x[0] for x in out] != list(range(len(out))):
        raise SystemExit(f'FATAL: slot {arm} s{seed} attempts are not 0..k: {[x[0] for x in out]}')
    return out


def slot_kernel_dirs(runs_dir, slot):
    return sorted(glob.glob(os.path.join(runs_dir, slot)) + glob.glob(os.path.join(runs_dir, f'{slot}_r[12]')))


def attempt_report(runs_dir, trained_flags):
    """Review 091 Ask 2: per (arm, seed slot, attempt) the seed used, the guard level (at a stop; or the last epoch's mean
    batch-median s at completion), the 100.9 check, and whether the other arm's attempt at the same seed trained."""
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            for kd in slot_kernel_dirs(runs_dir, f'tc_{arm}_s{s}'):
                for e in tc_ledger.attempt_entries(kd):
                    e = dict(e, arm=arm, slot_seed=s)
                    if e['returncode'] == 0 and e.get('tag'):
                        e['check_100_9_trained'] = trained_flags.get(e['tag'])
                    rows.append(e)
    done = {(r['arm'], r['seed']) for r in rows if r['returncode'] == 0}
    for r in rows:
        other = 'off' if r['arm'] == 'on' else 'on'
        r['other_arm_same_seed_completed'] = (other, r['seed']) in done
    return rows


def first_completed(attempts):
    for att, rc, rec in attempts:
        if rc == 0:
            if rec is None:
                raise SystemExit('FATAL: a completed attempt has no single V10C_COMPLETE record')
            return att, rec
    return None, None


def cluster_ci(deltas, n_boot=N_BOOT, seed=0):
    d = np.asarray(deltas, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rule(per_cell_on, per_cell_off, cells=CLEAN, min_fav=MIN_FAV, n_boot=N_BOOT):
    """per_cell_*: list over seeds of {cell: per-row mean contrast r}. The 100.5 rule."""
    on = {c: float(np.mean([r[c] for r in per_cell_on])) for c in cells}
    off = {c: float(np.mean([r[c] for r in per_cell_off])) for c in cells}
    d = {c: on[c] - off[c] for c in cells}
    lo, hi = cluster_ci(list(d.values()), n_boot)
    n_pos, n_neg = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    if n_pos >= min_fav and lo > 0:
        verdict = 'cell-specific chromatin contributes on unseen cells (among runs whose drug path trained, 100.8)'
    elif n_neg >= min_fav and hi < 0:
        verdict = 'chromatin hurts on unseen cells (among runs whose drug path trained, 100.8)'
    else:
        verdict = 'no claim'
    return {'per_cell_on': on, 'per_cell_off': off, 'delta_c': d, 'mean_delta': float(np.mean(list(d.values()))),
            'ci95': [lo, hi], 'n_cells_on_better': n_pos, 'n_cells_off_better': n_neg, 'verdict': verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ledger', default=tc_ledger.LEDGER)
    ap.add_argument('--trained', nargs='*', default=[], help='100.9 checks of the T-C checkpoints (reported only)')
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- T-C is read once')
    slots, counts = {}, {arm: {'attempts': 0, 'failed_guard': 0, 'by_seed': {}} for arm in ARMS}
    sha = tc_ledger.ledger_sha1s(a.ledger)
    flags = {}
    for p in a.trained:
        for r in json.load(open(p, encoding='utf-8'))['runs']:
            flags[r['checkpoint'].replace('v10c_', '').replace('.pt', '')] = r['drug_path_trained']
    for arm in ARMS:
        for s in SEEDS:
            att = slot_attempts(a.runs_dir, arm, s)
            k, rec = first_completed(att)
            n_fail = sum(rc == 3 for _, rc, _ in att)
            counts[arm]['attempts'] += len(att)
            counts[arm]['failed_guard'] += n_fail
            counts[arm]['by_seed'][str(s)] = {'attempts': len(att), 'failed_guard': n_fail, 'used_attempt': k}
            if rec is not None:
                want = sha.get((f'tc_{arm}_s{s}', k))
                if want is None or hashlib.sha1(open(rec, 'rb').read()).hexdigest() != want:
                    raise SystemExit(f'FATAL: tc_{arm}_s{s} attempt {k}: the record is not the one in the ledger (100.11)')
            slots[(arm, s)] = None if rec is None else json.load(open(rec, encoding='utf-8'))
    missing = [f'{arm}_s{s}' for (arm, s), r in slots.items() if r is None]
    rec = {'rule': 'RESULTS 100.5 (+100.8, 100.11)', 'attempt_counts': counts, 'missing_slots': missing,
           'attempts': attempt_report(a.runs_dir, flags)}
    if missing:
        rec['verdict'] = 'no claim: a slot has no completed attempt (100.8)'
    else:
        pc = {k: {c: v['contrast_r'] for c, v in r['scores']['per_cell'].items()} for k, r in slots.items()}
        dr = {k: {c: v['delta_r'] for c, v in r['scores']['per_cell'].items()} for k, r in slots.items()}
        for k, r in pc.items():
            absent = [c for c in CLEAN + APART if c not in r or r[c] is None]
            if absent:
                raise SystemExit(f'FATAL: run {k} has no contrast score for {absent}')
        res = rule([pc[('on', s)] for s in SEEDS], [pc[('off', s)] for s in SEEDS])
        rec.update(res)
        rec['reported'] = {
            'delta_r_per_cell': {arm: {c: float(np.mean([dr[(arm, s)][c] for s in SEEDS])) for c in CLEAN + APART} for arm in ARMS},
            'row_pooled_contrast_r': {arm: float(np.mean([slots[(arm, s)]['scores']['contrast_r_row_mean'] for s in SEEDS])) for arm in ARMS},
            'row_pooled_delta_r': {arm: float(np.mean([slots[(arm, s)]['scores']['delta_r_row_mean'] for s in SEEDS])) for arm in ARMS},
            'H1975_apart': {arm: float(np.mean([pc[(arm, s)]['H1975'] for s in SEEDS])) for arm in ARMS},
            'permutations_on': {f's{s}': slots[('on', s)]['reported_permutations'] for s in SEEDS},
            'single_control_heldout_rows': {f'{arm}_s{s}': slots[(arm, s)]['single_control_heldout_rows'] for arm in ARMS for s in SEEDS},
            'seeds_used': {f'{arm}_s{s}': slots[(arm, s)]['args']['seed'] for arm in ARMS for s in SEEDS},
            'pred_sha1': {f'{arm}_s{s}': slots[(arm, s)]['pred_sha1'] for arm in ARMS for s in SEEDS},
            'x_cell': 'leave-own-plate-out, input-transductive across a cell\'s plates (086 C3)'}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rec.get(k) for k in ('verdict', 'mean_delta', 'ci95', 'n_cells_on_better', 'missing_slots')}, indent=1))


if __name__ == '__main__':
    main()
