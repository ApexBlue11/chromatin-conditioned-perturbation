"""v11, read once when every slot of every arm is resolved (RESULTS 102 item 5, 102.1 C2 / C3 / Ask 6). Written before any
v11 run exists. Nothing here selects anything.

  R1   v11-full against T-B's off arm (its 3 existing runs, checked against tb_ledger.jsonl), by 100.4's rule on dev
       contrast r (tb_read.rule): the 3-seed mean delta >= max(0.003, 2 sqrt(s_a^2/3 + s_b^2/3)), the mean of per-cell
       means > 0, and >= 4 of the 6 dev cells.
  R2   v11-full against each ablation (-chromatin, -targets, -CCLE), the same rule; for -CCLE the cell condition is over
       the 4 dev cells with CCLE (LNCAP, T47D, U2OS, U937): >= 3 of 4 (102.1 C3).
  Each slot uses its first completed attempt (100.8); a slot without one -> no claim for every R it enters.
  Every v11 record and prediction file must match the v11 ledger (100.11's rule).

Reported, never read (102 item 5, 102.1 C5):
  strata of every R's pair (strata.py, 100.13 / 101.1 C3): per dev cell with its CCLE availability, per mechanism class
  (compound_classes.json, with the compound-cluster CI), per compound (pooled-SE t on 4 df, BH q = 0.10, the 9-relabelling
  calibration), and per target status (known / probe / duplicate of an unseen compound / unannotated);
  delta r; each run's permutations; the known-target setting for unseen annotated compounds; attempt and collapse counts.

    python model/v11/v11_read.py --runs_dir external/kaggle_out/v11 --out model/results/v11/v11_reading.json
"""
import argparse, glob, hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
import v11_ledger  # noqa: E402
import strata  # noqa: E402
import tb_read  # noqa: E402

ARMS, SEEDS = ('full', 'no_chrom', 'no_targets', 'no_ccle'), (0, 1, 2)
DEV = ('HUVEC', 'LNCAP', 'NPC', 'T47D', 'U2OS', 'U937')
CCLE_DEV = ('LNCAP', 'T47D', 'U2OS', 'U937')
STATUS = ('known', 'probe', 'unseen_annotated', 'unannotated')
SUBSTRATE = 'external/v10c_full/v10c_full_substrate.npz'
CLASSES = 'model/results/v10c/compound_classes.json'
TB_RUNS = 'external/kaggle_out/v10c_tb'


def slot_name(arm, seed):
    return f'v11_{arm}_s{seed}'


def slot_kernel_dirs(runs_dir, slot):
    return sorted(glob.glob(os.path.join(runs_dir, slot)) + glob.glob(os.path.join(runs_dir, f'{slot}_r[12]')) +
                  glob.glob(os.path.join(runs_dir, f'{slot}_d[1-9]')))


def slot_attempts(runs_dir, slot, ledger):
    """Every attempt of a slot over its kernels, ordered: (attempt, returncode, record path or None). A kernel without
    V11_KERNEL_COMPLETE.json is accepted only if the ledger logs it as dead (review 092 C5's rule)."""
    out = []
    for kd in slot_kernel_dirs(runs_dir, slot):
        kc = os.path.join(kd, v11_ledger.KERNEL_DONE)
        if not os.path.exists(kc):
            dead = v11_ledger.dead_kernel_attempts(slot, os.path.basename(kd), ledger)
            if dead is None:
                raise SystemExit(f'FATAL: {kc} missing (download the kernel output first, or log its death)')
            out.extend(dead)
            continue
        for a in json.load(open(kc, encoding='utf-8'))['attempts'][slot]:
            sub = os.path.join(kd, os.path.basename(a['out'].rstrip('/')))
            recs = glob.glob(os.path.join(sub, v11_ledger.COMPLETE))
            out.append((a['attempt'], a['returncode'], recs[0] if a['returncode'] == 0 and len(recs) == 1 else None))
    out.sort(key=lambda x: x[0])
    if [x[0] for x in out] != list(range(len(out))):
        raise SystemExit(f'FATAL: {slot} attempts are not 0..k: {[x[0] for x in out]}')
    return out


def resolved(attempts, max_attempts=3):
    """A slot is resolved when it has a completed attempt or every one of its 100.8 attempts failed the guard."""
    return any(rc == 0 for _, rc, _ in attempts) or (len(attempts) >= max_attempts and all(rc == 3 for _, rc, _ in attempts))


def first_completed(attempts):
    for att, rc, rec in attempts:
        if rc == 0:
            if rec is None:
                raise SystemExit('FATAL: a completed attempt has no single V11_COMPLETE record')
            return att, rec
    return None, None


def load_slot(rec_path, slot, attempt, sha):
    raw = open(rec_path, 'rb').read()
    want = sha.get((slot, attempt))
    if want is None or hashlib.sha1(raw).hexdigest() != want[0]:
        raise SystemExit(f'FATAL: {slot} attempt {attempt}: the record is not the one in the ledger')
    rec = json.loads(raw)
    pred = os.path.join(os.path.dirname(rec_path), f"v11_pred_{rec['tag']}.npz")
    if v11_ledger.sha1_file(pred) != rec['pred_sha1'] or rec['pred_sha1'] != want[1]:
        raise SystemExit(f'FATAL: {slot} attempt {attempt}: the prediction file is not the one in the ledger')
    return rec, np.load(pred)


def load_reference(tb_runs=TB_RUNS, tb_ledger=tb_read.LEDGER):
    """T-B's off arm: tb_read's ledger-checked records, and each record's own prediction file (sha1 = its pred_sha1)."""
    runs = tb_read.load_runs(tb_runs, tb_ledger)
    out = {}
    for s in SEEDS:
        rec = runs[('off', s)]
        t = f'tb_off_s{s}'
        p = os.path.join(tb_runs, t, t, f"v10c_pred_{tb_read.tag_of('off', s)}.npz")
        if v11_ledger.sha1_file(p) != rec['pred_sha1']:
            raise SystemExit(f'FATAL: {p} is not the prediction file of T-B off s{s}')
        out[s] = (rec, np.load(p))
    return out


def per_cell(rec):
    return {c: v['contrast_r'] for c, v in rec['scores']['per_cell'].items()}


def read_rule(a_recs, b_recs, ccle_cells=False):
    a = [r['scores']['contrast_r_row_mean'] for r in a_recs]
    b = [r['scores']['contrast_r_row_mean'] for r in b_recs]
    res = tb_read.rule(a, b, [per_cell(r) for r in a_recs], [per_cell(r) for r in b_recs], min_cells=4)
    if sorted(res['per_cell_on']) != sorted(DEV):
        raise SystemExit(f"FATAL: the runs' cells {sorted(res['per_cell_on'])} are not the dev carve {DEV}")
    if ccle_cells:                                                   # 102.1 C3: the cell condition over the CCLE cells
        fav = [c for c in CCLE_DEV if res['per_cell_on'][c] > res['per_cell_off'][c]]
        res.update({'cell_condition': f'>= 3 of the 4 CCLE dev cells {list(CCLE_DEV)}', 'ccle_cells_favouring_full': fav,
                    'passes': bool(res['delta_clears_threshold'] and res['cell_mean_delta'] > 0 and len(fav) >= 3)})
    else:
        res['cell_condition'] = '>= 4 of the 6 dev cells'
    res['full_is'] = 'on'                                            # tb_read's "on" side is v11-full here
    return res


def aligned_r(preds, key='r_contrast'):
    rows = preds[0]['sub_row']
    for p in preds[1:]:
        if not np.array_equal(p['sub_row'], rows):
            raise SystemExit('FATAL: prediction files are not on the same rows')
    return rows, np.stack([np.asarray(p[key], float) for p in preds])


def pair_strata(pa, pb, meta, n_boot=strata.N_BOOT):
    rows, Ra = aligned_r(pa)
    rows_b, Rb = aligned_r(pb)
    if not np.array_equal(rows, rows_b):
        raise SystemExit('FATAL: the two arms are not on the same rows')
    cell, pert, cls, status = meta['cell'][rows], meta['pert'][rows], meta['cls'][rows], meta['status']
    ok = strata.finite_rows(Ra, Rb)
    dr = Ra.mean(0) - Rb.mean(0)
    cells = strata.by_group(Ra, Rb, cell)
    for c, v in cells.items():
        v['ccle'] = c in CCLE_DEV
    lab = {}
    for p in np.unique(pert[ok]):
        m = ok & (pert == p)
        lab[str(p)] = {'class': str(cls[m][0]), 'cells': sorted({str(x) for x in cell[m]})}
    out = {'n_rows': int(ok.sum()), 'per_cell': cells,
           'ccle_vs_no_ccle_cells': {k: strata.group_stats(Ra, Rb, ok & sel) for k, sel in
                                     (('ccle', np.isin(cell, CCLE_DEV)), ('no_ccle', ~np.isin(cell, CCLE_DEV)))
                                     if (ok & sel).any()},
           'per_class': strata.by_class(Ra, Rb, cls, pert, n_boot),
           'per_compound': strata.by_compound(Ra, Rb, pert, labels=lab),
           'row_mean_delta': float(dr[ok].mean())}
    if status is not None:
        st = np.array([STATUS[int(k)] for k in status])
        out['per_target_status'] = strata.by_group(Ra, Rb, st)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ledger', default=v11_ledger.LEDGER)
    ap.add_argument('--tb_runs', default=TB_RUNS)
    ap.add_argument('--tb_ledger', default=tb_read.LEDGER)
    ap.add_argument('--substrate', default=SUBSTRATE)
    ap.add_argument('--classes', default=CLASSES)
    ap.add_argument('--n_boot', type=int, default=strata.N_BOOT)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'FATAL: {a.out} exists -- v11 is read once')
    sha = v11_ledger.completed_sha1s(a.ledger)
    slots, counts, unresolved = {}, {}, []
    for arm in ARMS:
        counts[arm] = {}
        for s in SEEDS:
            name = slot_name(arm, s)
            att = slot_attempts(a.runs_dir, name, a.ledger)
            if not resolved(att):
                unresolved.append(name)
                continue
            k, rec = first_completed(att)
            counts[arm][str(s)] = {'attempts': len(att), 'failed_guard': sum(rc == 3 for _, rc, _ in att), 'used_attempt': k}
            slots[(arm, s)] = None if rec is None else load_slot(rec, name, k, sha)
    if unresolved:
        raise SystemExit(f'FATAL: not every slot is resolved {unresolved} -- nothing is read until every arm\'s runs exist '
                         f'(102.1 Ask 6)')
    ref = load_reference(a.tb_runs, a.tb_ledger)
    missing = sorted(slot_name(*k) for k, v in slots.items() if v is None)
    rec = {'rule': 'RESULTS 102 item 5 / 102.1 C2, C3, Ask 6 (100.4 rule; 100.8 first completed attempt)',
           'attempt_counts': counts, 'missing_slots': missing, 'readings': {}}
    z = np.load(a.substrate, allow_pickle=True)
    cl = json.load(open(a.classes, encoding='utf-8'))['class']
    meta = {'cell': z['meta_cell_iname'], 'pert': z['meta_pert_id']}
    meta['cls'] = np.array([cl.get(p, 'unannotated') for p in meta['pert']])
    pairs = {'R1': ('full', 'reference'), 'R2_chromatin': ('full', 'no_chrom'), 'R2_targets': ('full', 'no_targets'),
             'R2_ccle': ('full', 'no_ccle')}

    def arm_runs(arm):
        if arm == 'reference':
            return [ref[s] for s in SEEDS]
        return [slots[(arm, s)] for s in SEEDS]

    for name, (x, y) in pairs.items():
        if any(slots.get((arm, s)) is None for arm in (x, y) if arm != 'reference' for s in SEEDS):
            rec['readings'][name] = {'verdict': 'no claim: a slot has no completed attempt (100.8)'}
            continue
        A, B = arm_runs(x), arm_runs(y)
        res = read_rule([r for r, _ in A], [r for r, _ in B], ccle_cells=name == 'R2_ccle')
        if name == 'R1':
            res['verdict'] = ('v11\'s full-graph module (with its auxiliary target supervision) improves dev contrast r '
                              'over T-B\'s off arm' if res['passes'] else 'no claim')
        else:
            what = {'R2_chromatin': 'cell-specific chromatin', 'R2_targets': 'drug-target knowledge (input and head)',
                    'R2_ccle': 'the CCLE baseline'}[name]
            res['verdict'] = f'{what} contributes to v11-full on dev cells' if res['passes'] else 'no claim'
        res['comparison'] = f'{x} vs {y}'
        meta['status'] = A[0][1]['target_status'] if 'target_status' in A[0][1].files else None
        res['reported_strata'] = pair_strata([p for _, p in A], [p for _, p in B], meta, a.n_boot)
        res['reported_delta_r'] = {x: [r['scores']['delta_r_row_mean'] for r, _ in A],
                                   y: [r['scores']['delta_r_row_mean'] for r, _ in B]}
        rec['readings'][name] = res
    rec['reported'] = {
        'permutations': {slot_name(arm, s): slots[(arm, s)][0]['reported_permutations'] for arm in ARMS for s in SEEDS
                         if slots[(arm, s)] is not None},
        'known_target_setting': {slot_name(arm, s): slots[(arm, s)][0]['reported_known_target_setting'] for arm in ARMS
                                 for s in SEEDS if slots[(arm, s)] is not None},
        'seeds_used': {slot_name(arm, s): slots[(arm, s)][0]['args']['seed'] for arm in ARMS for s in SEEDS
                       if slots[(arm, s)] is not None},
        'end_of_training': {slot_name(arm, s): {k: slots[(arm, s)][0]['hist'][-1].get(k) for k in
                                                ('dose_scale_batch_median_mean', 'fg_gate_mean', 'target_bce_mean',
                                                 'fg_readout_rms_last_batch')}
                            for arm in ARMS for s in SEEDS if slots[(arm, s)] is not None},
        'reference': {f'tb_off_s{s}': ref[s][0]['pred_sha1'] for s in SEEDS},
        'note': 'the reference ran without the 100.8 guard (all three trained, 100.9); every v11 run is guarded'}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: {kk: v.get(kk) for kk in ('verdict', 'delta', 'threshold', 'n_cells_favouring_on')}
                      for k, v in rec['readings'].items()}, indent=1))
    return rec


if __name__ == '__main__':
    main()
