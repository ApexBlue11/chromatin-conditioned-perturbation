"""Tests for v11_read.py and v11_ledger.py on synthetic run directories (RESULTS 102 item 5 / 102.1)."""
import hashlib, json, os, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V10C = os.path.join(os.path.dirname(HERE), 'v10c')
for p in (HERE, V10C):
    if p not in sys.path:
        sys.path.insert(0, p)
import v11_read as vr  # noqa: E402
import v11_ledger as vl  # noqa: E402
import tb_read  # noqa: E402

CELLS = vr.DEV
N_PER = 40


def scores_of(r, cell):
    per = {}
    for c in CELLS:
        m = cell == c
        per[c] = {'n': int(m.sum()), 'contrast_r': float(np.nanmean(r[m])), 'delta_r': 0.4}
    return {'contrast_r_row_mean': float(np.nanmean(r)), 'delta_r_row_mean': 0.4, 'per_cell': per}


def make_world(tmp, effects, fail=(), n_boot=200):
    """effects: arm -> {cell: added r}. fail: slots whose 3 attempts all fail the guard."""
    rng = np.random.default_rng(0)
    n = N_PER * len(CELLS)
    cell = np.repeat(np.array(CELLS), N_PER)
    pert = np.array([f'p{k % 25}' for k in range(n)])
    sub_row = np.arange(100, 100 + n)
    meta_cell = np.array(['X'] * (100 + n), dtype=object)
    meta_pert = np.array(['q'] * (100 + n), dtype=object)
    meta_cell[sub_row], meta_pert[sub_row] = cell, pert
    np.savez(tmp / 'sub.npz', meta_cell_iname=meta_cell.astype(str), meta_pert_id=meta_pert.astype(str))
    json.dump({'class': {f'p{k}': ('kinase' if k < 12 else 'other') for k in range(25)}}, open(tmp / 'cls.json', 'w'))
    base = rng.normal(0.15, 0.05, n)
    base[::17] = np.nan
    # the reference: T-B's six records (tb_read.load_runs reads on and off)
    tb, led = tmp / 'tb', tmp / 'tb_ledger.jsonl'
    with open(led, 'w') as L:
        for arm in ('on', 'off'):
            for s in vr.SEEDS:
                t, tag = f'tb_{arm}_s{s}', tb_read.tag_of(arm, s)
                d = tb / t / t
                d.mkdir(parents=True)
                r = base + rng.normal(0, 0.003, n) + effects.get('reference', {}).get('_all', 0.0)
                pp = d / f'v10c_pred_{tag}.npz'
                np.savez(pp, sub_row=sub_row, r_contrast=r)
                rec = {'tag': tag, 'scores': scores_of(r, cell), 'pred_sha1': hashlib.sha1(pp.read_bytes()).hexdigest()}
                (d / f'V10C_COMPLETE_{tag}.json').write_text(json.dumps(rec))
                L.write(json.dumps({'run': tag, 'record_sha1': hashlib.sha1((d / f'V10C_COMPLETE_{tag}.json').read_bytes())
                                    .hexdigest()}) + '\n')
    runs, ledger = tmp / 'v11', tmp / 'v11_ledger.jsonl'
    for arm in vr.ARMS:
        for s in vr.SEEDS:
            slot = vr.slot_name(arm, s)
            kd = runs / slot
            kd.mkdir(parents=True)
            atts = []
            if slot in fail:                                         # three guard failures: the slot is exhausted
                for k in range(3):
                    out = slot if k == 0 else f'{slot}_retry{k}'
                    (kd / out).mkdir()
                    atts.append({'attempt': k, 'returncode': 3, 'out': f'/kaggle/working/{out}',
                                 'args': ['--seed', str(s + 10 * k)]})
                    (kd / out / f'V11_FAILED_{slot}.json').write_text(json.dumps({'step': 1000, 'guard_level': 0.0}))
            else:
                (kd / slot).mkdir()
                eff = np.array([effects.get(arm, {}).get(c, 0.0) for c in cell])
                r = base + rng.normal(0, 0.003, n) + eff
                tag = f'cold_cell_1_dev_v11{arm}_d256_s{s}'
                pp = kd / slot / f'v11_pred_{tag}.npz'
                np.savez(pp, sub_row=sub_row, r_contrast=r, target_status=(np.arange(n) % 4).astype(np.int8))
                rec = {'tag': tag, 'args': {'seed': s}, 'scores': scores_of(r, cell),
                       'pred_sha1': hashlib.sha1(pp.read_bytes()).hexdigest(), 'reported_permutations': {},
                       'reported_known_target_setting': {'n_rows': 0}, 'hist': [{'fg_gate_mean': 0.95}]}
                (kd / slot / f'V11_COMPLETE_{tag}.json').write_text(json.dumps(rec))
                atts.append({'attempt': 0, 'returncode': 0, 'out': f'/kaggle/working/{slot}', 'args': ['--seed', str(s)]})
            (kd / 'V11_KERNEL_COMPLETE.json').write_text(json.dumps({'attempts': {slot: atts}}))
            vl.add(str(kd), str(ledger))
    args = ['--runs_dir', str(runs), '--ledger', str(ledger), '--tb_runs', str(tb), '--tb_ledger', str(led),
            '--substrate', str(tmp / 'sub.npz'), '--classes', str(tmp / 'cls.json'), '--n_boot', str(n_boot)]
    return args


def test_r1_and_r2_follow_tb_reads_rule(tmp_path):
    up = {c: 0.02 for c in CELLS}
    args = make_world(tmp_path, {'full': up})
    rec = vr.main(args + ['--out', str(tmp_path / 'o.json')])
    R = rec['readings']
    assert R['R1']['passes'] and R['R2_chromatin']['passes'] and R['R2_targets']['passes'] and R['R2_ccle']['passes']
    assert R['R1']['verdict'].startswith("v11's full-graph module with its auxiliary target supervision")
    assert 'among runs whose drug path trained (100.8)' in R['R1']['verdict'] and 'without the guard' in R['R1']['verdict']
    assert R['R2_ccle']['verdict'].startswith('the CCLE') and R['R2_ccle']['verdict'].endswith('trained (100.8)')
    assert R['R1']['n_cells_favouring_on'] == 6 and abs(R['R1']['delta'] - 0.02) < 0.003
    st = R['R1']['reported_strata']
    assert set(st['per_cell']) == set(CELLS) and st['per_cell']['T47D']['ccle'] and not st['per_cell']['NPC']['ccle']
    assert set(st['per_class']) == {'kinase', 'other'} and set(st['per_target_status']) == set(vr.STATUS)
    assert len(st['per_compound']['relabelling_calibration']) == 9
    with pytest.raises(SystemExit, match='read once'):
        vr.main(args + ['--out', str(tmp_path / 'o.json')])


def test_no_effect_is_no_claim(tmp_path):
    rec = vr.main(make_world(tmp_path, {}) + ['--out', str(tmp_path / 'o.json')])
    assert all(v['verdict'] == 'no claim' for v in rec['readings'].values())


def test_r2_ccle_counts_only_the_ccle_cells(tmp_path):
    # full beats -CCLE on the two non-CCLE cells and on 2 of 4 CCLE cells: 4 of 6 overall, but 2 of 4 CCLE cells
    eff = {'HUVEC': 0.05, 'NPC': 0.05, 'LNCAP': 0.05, 'T47D': 0.05, 'U2OS': -0.01, 'U937': -0.01}
    rec = vr.main(make_world(tmp_path, {'full': eff, 'no_chrom': eff, 'no_targets': eff}) +
                  ['--out', str(tmp_path / 'o.json')])
    c = rec['readings']['R2_ccle']
    assert c['n_cells_favouring_on'] == 4 and c['ccle_cells_favouring_full'] == ['LNCAP', 'T47D'] and not c['passes']
    assert c['cell_condition'].startswith('>= 3 of the 4')


def test_a_slot_that_exhausts_its_retries_gives_no_claim_where_it_enters(tmp_path):
    args = make_world(tmp_path, {'full': {c: 0.02 for c in CELLS}}, fail=('v11_no_ccle_s1',))
    rec = vr.main(args + ['--out', str(tmp_path / 'o.json')])
    assert rec['readings']['R2_ccle']['verdict'].startswith('no claim: a slot') and rec['missing_slots'] == ['v11_no_ccle_s1']
    assert rec['readings']['R1']['passes'] and rec['attempt_counts']['no_ccle']['1']['failed_guard'] == 3


def test_nothing_is_read_while_a_slot_is_unresolved(tmp_path):
    args = make_world(tmp_path, {}, fail=('v11_full_s2',))
    kc = tmp_path / 'v11' / 'v11_full_s2' / 'V11_KERNEL_COMPLETE.json'
    k = json.loads(kc.read_text())
    k['attempts']['v11_full_s2'] = k['attempts']['v11_full_s2'][:1]       # one guard failure, retries pending
    kc.write_text(json.dumps(k))
    with pytest.raises(SystemExit, match='not every slot is resolved'):
        vr.main(args + ['--out', str(tmp_path / 'o.json')])


def test_a_changed_record_or_prediction_is_refused(tmp_path):
    args = make_world(tmp_path, {})
    d = tmp_path / 'v11' / 'v11_no_chrom_s0' / 'v11_no_chrom_s0'
    pred = next(d.glob('v11_pred_*.npz'))
    z = dict(np.load(pred))
    z['r_contrast'] = z['r_contrast'] + 0.1
    np.savez(pred, **z)
    with pytest.raises(SystemExit, match='prediction file is not the one'):
        vr.main(args + ['--out', str(tmp_path / 'o.json')])
    rec = next(d.glob('V11_COMPLETE_*.json'))
    rec.write_text(rec.read_text().replace('"seed": 0', '"seed": 5'))
    with pytest.raises(SystemExit, match='record is not the one'):
        vr.main(args + ['--out', str(tmp_path / 'o2.json')])


def test_a_changed_reference_prediction_is_refused(tmp_path):
    args = make_world(tmp_path, {})
    p = tmp_path / 'tb' / 'tb_off_s1' / 'tb_off_s1' / f"v10c_pred_{tb_read.tag_of('off', 1)}.npz"
    z = dict(np.load(p))
    z['r_contrast'] = z['r_contrast'] * 2
    np.savez(p, **z)
    with pytest.raises(SystemExit, match='not the prediction file of T-B off'):
        vr.main(args + ['--out', str(tmp_path / 'o.json')])


def test_ledger_add_is_idempotent_and_death_logs_earlier_guard_failures(tmp_path):
    args = make_world(tmp_path, {}, fail=('v11_full_s0',))
    led = tmp_path / 'v11_ledger.jsonl'
    n0 = len(led.read_text().splitlines())
    assert vl.add(str(tmp_path / 'v11' / 'v11_full_s1'), str(led)) == [] and len(led.read_text().splitlines()) == n0
    kd = tmp_path / 'v11' / 'v11_full_s0'
    vl.death('v11_full_s0', 2, 'OOM', kernel_dir=str(kd), ledger=str(tmp_path / 'L2.jsonl'))
    rows = [json.loads(x) for x in open(tmp_path / 'L2.jsonl')]
    assert [r['kind'] for r in rows] == ['attempt', 'attempt', 'non_guard_death'] and rows[1]['attempt'] == 1
    assert vl.dead_kernel_attempts('v11_full_s0', 'v11_full_s0', str(tmp_path / 'L2.jsonl')) == [(0, 3, None), (1, 3, None)]
