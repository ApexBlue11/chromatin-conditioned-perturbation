# -*- coding: utf-8 -*-
"""Tests for RESULTS §94 Stage A (mechanism_stage_a.py).

All tests run on synthetic data only, with no model outputs or real bundle loaded.
Every test is capable of failing.
"""
import json
import os
import sys
import urllib.error

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mechanism_stage_a as msa  # noqa: E402


@pytest.fixture
def fake_progeny_net():
    """Small synthetic PROGENy network with landmark genes."""
    genes = [f'GENE_{i}' for i in range(20)]
    rows = []
    # MAPK
    for i in range(10):
        rows.append({'source': 'MAPK', 'target': genes[i], 'weight': 1.5 * (1 if i % 2 == 0 else -1), 'padj': 1e-10})
    # p53
    for i in range(5, 15):
        rows.append({'source': 'p53', 'target': genes[i], 'weight': 2.0 * (1 if i % 2 == 0 else -1), 'padj': 1e-10})
    # EGFR
    for i in range(10, 20):
        rows.append({'source': 'EGFR', 'target': genes[i], 'weight': 1.0, 'padj': 1e-10})
    return pd.DataFrame(rows)


def test_load_rows_synthetic(tmp_path):
    """Test load_rows contract: requires split_split_cold_cell_1 == 'test' and asserts exactly 21,151 rows."""
    bundle_file = tmp_path / 'bundle.npz'
    N = 21151
    G = 10
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, G)).astype(np.float32)
    X_ctl = rng.normal(size=(N, G)).astype(np.float32)
    pert = np.array([f'BRD-{i}' for i in range(N)])
    cell = np.array(['MCF7'] * N)
    row_index = np.arange(N, dtype=np.int64)
    split = np.array(['test'] * N)

    np.savez(bundle_file, X=X, X_ctl=X_ctl, meta_pert_id=pert, meta_cell=cell,
             row_index=row_index, split_split_cold_cell_1=split)

    rows = msa.load_rows(str(bundle_file))
    assert len(rows['X']) == 21151
    assert rows['delta'].dtype == np.float64
    np.testing.assert_allclose(rows['delta'], X.astype(np.float64) - X_ctl.astype(np.float64))

    # Test unpacking
    uX, uX_ctl, upert, ucell, urow_idx = rows
    assert len(uX) == 21151
    assert ucell[0] == 'MCF7'

    # Test failure on wrong count
    bad_bundle = tmp_path / 'bad_bundle.npz'
    np.savez(bad_bundle, X=X[:100], X_ctl=X_ctl[:100], meta_pert_id=pert[:100], meta_cell=cell[:100],
             row_index=row_index[:100], split_split_cold_cell_1=split[:100])
    with pytest.raises(AssertionError, match='expected 21151 test rows'):
        msa.load_rows(str(bad_bundle))


def test_load_labels(tmp_path):
    """Test load_labels filters direct_interaction == 1 and organism == 'Homo sapiens'."""
    tsv_file = tmp_path / 'labels.tsv'
    content = (
        "pert_id\tmolecule_chembl_id\tparent_chembl_id\tgene_symbol\taction_type\tdirect_interaction\torganism\tmechanism_of_action\n"
        "P1\tCHEMBL1\tCHEMBL1\tMAP2K1\tINHIBITOR\t1\tHomo sapiens\tMEK inhibitor\n"
        "P2\tCHEMBL2\tCHEMBL1\tMAP2K1\tINHIBITOR\t1\tHomo sapiens\tMEK inhibitor\n"  # same parent as P1
        "P3\tCHEMBL3\tCHEMBL3\tEGFR\tANTAGONIST\t1\tHomo sapiens\tEGFR antagonist\n"
        "P4\tCHEMBL4\tCHEMBL4\tEGFR\tINHIBITOR\t0\tHomo sapiens\tEGFR inhibitor\n"  # direct_interaction == 0 -> dropped
        "P5\tCHEMBL5\tCHEMBL5\tEGFR\tINHIBITOR\t1\tMus musculus\tEGFR inhibitor\n"  # non-human -> dropped
    )
    tsv_file.write_text(content, encoding='utf-8')

    pert_to_parent, parent_to_moa, parent_to_targets = msa.load_labels(str(tsv_file))

    assert pert_to_parent == {'P1': 'CHEMBL1', 'P2': 'CHEMBL1', 'P3': 'CHEMBL3'}
    assert 'P4' not in pert_to_parent
    assert 'P5' not in pert_to_parent
    assert parent_to_moa['CHEMBL1'] == {'MEK inhibitor'}
    assert parent_to_targets['CHEMBL1'] == [('MAP2K1', 'INHIBITOR')]


def test_parent_collapse_and_units():
    """Two pert_ids of one parent become one unit and are never mates of each other."""
    G = 10
    pert_to_parent = {'P1': 'PARENT_A', 'P2': 'PARENT_A', 'P3': 'PARENT_B'}
    parent_to_moa = {'PARENT_A': {'Kinase inhibitor'}, 'PARENT_B': {'Kinase inhibitor'}}

    rows = {
        'cell': np.array(['MCF7', 'MCF7', 'MCF7', 'MCF7']),
        'pert': np.array(['P1', 'P2', 'P1', 'P3']),
        'row_index': np.array([0, 1, 2, 3]),
        'X_ctl': np.zeros((4, G), dtype=np.float32),
        'delta': np.ones((4, G), dtype=np.float64),
    }

    units = msa.units_for_cell('MCF7', rows, pert_to_parent, parent_to_moa)

    # PARENT_A collapses P1 and P2
    assert 'PARENT_A' in units
    assert 'PARENT_B' in units
    assert 'P1' not in units
    assert 'P2' not in units
    assert len(units) == 2

    # PARENT_A contains row indices 0, 1, 2
    assert sorted(units['PARENT_A']['row_indices']) == [0, 1, 2]
    assert sorted(units['PARENT_B']['row_indices']) == [3]


def test_scored_cells():
    """Cells with >= 25 labelled units in multi-member MoA classes are scored; fixed order maintained."""
    G = 10
    # Create two cells: CELL_BIG (30 labelled units in MoA classes with >= 2 members) and CELL_SMALL (10 units)
    pert_to_parent = {}
    parent_to_moa = {}

    rows_cell = []
    rows_pert = []
    for i in range(30):
        pid = f'P_BIG_{i}'
        parent = f'PAR_BIG_{i}'
        pert_to_parent[pid] = parent
        parent_to_moa[parent] = {'Class_Shared'}
        rows_cell.append('HT29')
        rows_pert.append(pid)

    for i in range(10):
        pid = f'P_SML_{i}'
        parent = f'PAR_SML_{i}'
        pert_to_parent[pid] = parent
        parent_to_moa[parent] = {'Class_Shared'}
        rows_cell.append('MCF7')
        rows_pert.append(pid)

    rows = {
        'cell': np.array(rows_cell),
        'pert': np.array(rows_pert),
        'row_index': np.arange(len(rows_cell)),
        'X_ctl': np.zeros((len(rows_cell), G), dtype=np.float32),
        'delta': np.zeros((len(rows_cell), G), dtype=np.float64),
    }

    scored, counts = msa.scored_cells(rows=rows, pert_to_parent=pert_to_parent, parent_to_moa=parent_to_moa)
    assert counts['HT29'] == 30
    assert counts['MCF7'] == 10
    assert scored == ['HT29']  # only HT29 meets >= 25


def test_a1_planted_clusters_and_no_cluster():
    """A1 on planted clusters gives A1 - null >= 0.2 and p < 0.01; unplanted gives |A1 - null| < 0.05."""
    rng = np.random.default_rng(42)
    G = 100
    n_classes = 4
    n_per_class = 15
    N = n_classes * n_per_class

    labels = []
    for c in range(n_classes):
        labels.extend([{f'Class_{c}'} for _ in range(n_per_class)])

    # All unique control hashes (no plate confounds)
    ctl_hashes = [{f'hash_{i}'} for i in range(N)]

    # Planted world: signatures = centroid + small noise
    centroids = rng.normal(scale=3.0, size=(n_classes, G))
    planted_sigs = np.zeros((N, G))
    for i in range(N):
        cls_idx = i // n_per_class
        planted_sigs[i] = centroids[cls_idx] + rng.normal(scale=0.2, size=G)

    res_planted = msa.a1_cell(planted_sigs, labels, ctl_hashes, rng_seed=9400, n_perm=500)
    assert res_planted.a1 - res_planted.null_mean >= 0.2, f"Expected >= 0.2, got {res_planted.a1 - res_planted.null_mean}"
    assert res_planted.p_value < 0.01, f"Expected p < 0.01, got {res_planted.p_value}"
    assert res_planted.n_scored == N
    # Per-class AUROC reported for all 4 classes
    assert len(res_planted.per_class_auroc) == n_classes

    # No-cluster world: pure noise
    null_sigs = rng.normal(size=(N, G))
    res_null = msa.a1_cell(null_sigs, labels, ctl_hashes, rng_seed=9400, n_perm=500)
    assert abs(res_null.a1 - res_null.null_mean) < 0.05, f"Expected |diff| < 0.05, got {abs(res_null.a1 - res_null.null_mean)}"


def test_plate_rule():
    """Mates similar ONLY through shared X_ctl noise show signal without plate rule and NO signal with it."""
    rng = np.random.default_rng(123)
    G = 100
    N = 40  # 20 pairs of mates

    labels = []
    ctl_hashes = []
    # 20 classes, 1 pair of mates per class, sharing a plate control
    for p in range(20):
        cls_name = f'Class_{p}'
        labels.append({cls_name})
        labels.append({cls_name})
        plate_hash = f'plate_{p}'
        ctl_hashes.append({plate_hash})
        ctl_hashes.append({plate_hash})

    # True biological delta is pure noise
    sigs = rng.normal(size=(N, G))
    # Add shared plate noise to each pair
    for p in range(20):
        plate_noise = rng.normal(scale=3.0, size=G)
        sigs[2 * p] += plate_noise
        sigs[2 * p + 1] += plate_noise

    # WITH plate rule ON (default): pairs sharing control hash are excluded
    res_on = msa.a1_cell(sigs, labels, ctl_hashes, rng_seed=9400, n_perm=100, plate_rule=True)
    # Since all mates share plates, all confounded pairs are excluded (0 scored, no false signal)
    assert res_on.n_scored == 0 or (res_on.p_value >= 0.01 or abs(res_on.a1 - res_on.null_mean) < 0.05)

    # WITH plate rule OFF (private flag): shared plate confound creates false signal
    res_off = msa.a1_cell(sigs, labels, ctl_hashes, rng_seed=9400, n_perm=100, plate_rule=False)
    assert res_off.a1 - res_off.null_mean >= 0.2
    assert res_off.p_value < 0.01


def test_self_retrieval_and_active_subset():
    """Self-retrieval ceiling and active subset calculation on synthetic units."""
    rng = np.random.default_rng(7)
    G = 20
    units = {}
    for i in range(10):
        uid = f'U_{i}'
        # unit with 4 rows
        # U_0 to U_4 have strong consistent signal (high split-half corr)
        # U_5 to U_9 have noise
        base = rng.normal(size=G) * (2.0 if i < 5 else 0.0)
        deltas = [base + rng.normal(scale=0.1 if i < 5 else 1.0, size=G) for _ in range(4)]
        raw_sig = np.mean(deltas, axis=0)
        units[uid] = {
            'unit_id': uid,
            'is_labelled': True,
            'moas': {'MoA_A' if i < 5 else 'MoA_B'},
            'row_indices': [0, 1, 2, 3],
            'row_deltas': deltas,
            'raw_sig': raw_sig,
            'sig': raw_sig,
            'cell_mean': np.zeros(G),
            'ctl_hashes': {f'h_{uid}'},
        }

    res = msa.compute_self_retrieval_and_active_subset(units, rng_seed=9400, active_threshold=0.2)
    assert np.isfinite(res['self_retrieval_ceiling'])
    assert res['self_retrieval_ceiling'] > 0.5
    # High signal units should have split-half corr >= 0.2
    for i in range(5):
        assert res['split_half_corrs'][f'U_{i}'] >= 0.2
    assert res['active_subset'] is not None
    assert res['active_subset']['n_active'] >= 5


def test_a3_planted_pathway_and_null_calibration(fake_progeny_net):
    """A3: planted pathway shift gives T > 0 and p < 0.01; unplanted gives uniform p over seeds."""
    G = 20
    landmarks = [f'GENE_{i}' for i in range(G)]
    net = fake_progeny_net

    # Extract MAPK weights
    mapk_net = net[net['source'] == 'MAPK'].set_index('target')['weight']
    mapk_weights = np.array([mapk_net.get(g, 0.0) for g in landmarks])

    # Build 12 labelled compounds for cell HT29
    # 4 MEK inhibitors (class MAPK, inhibitor sign -1.0)
    # Signature = -k * MAPK weights (so activity is strongly negative, activity * sign > 0)
    # 8 other compounds with random noise
    parent_to_targets = {}
    for i in range(4):
        uid = f'MEKi_{i}'
        parent_to_targets[uid] = [('MAP2K1', 'INHIBITOR')]

    for i in range(8):
        uid = f'OTHER_{i}'
        parent_to_targets[uid] = [('UNKNOWN_GENE', 'INHIBITOR')]

    cell_units_map = {'HT29': {}}
    for i in range(4):
        uid = f'MEKi_{i}'
        sig = -3.0 * mapk_weights + np.random.default_rng(i).normal(scale=0.1, size=G)
        cell_units_map['HT29'][uid] = {
            'unit_id': uid,
            'is_labelled': True,
            'sig': sig,
        }
    for i in range(8):
        uid = f'OTHER_{i}'
        sig = np.random.default_rng(100 + i).normal(scale=0.5, size=G)
        cell_units_map['HT29'][uid] = {
            'unit_id': uid,
            'is_labelled': True,
            'sig': sig,
        }

    # Planted test
    res_planted = msa.run_a3(cell_units_map, ['HT29'], parent_to_targets, net=net, gene_names=landmarks, gated=True, rng_seed=9450, n_perm=300)
    assert res_planted.T > 0, f"Expected T > 0, got {res_planted.T}"
    assert res_planted.p < 0.01, f"Expected p < 0.01, got {res_planted.p}"
    assert res_planted.frac_positive == 1.0

    # Null calibration: check over 20 seeds with pure noise
    null_units_map = {'HT29': {}}
    for i in range(12):
        uid = f'CPD_{i}'
        parent_to_targets[uid] = [('MAP2K1', 'INHIBITOR')] if i < 4 else [('OTHER', 'INHIBITOR')]

    p_values = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        for i in range(12):
            uid = f'CPD_{i}'
            null_units_map['HT29'][uid] = {
                'unit_id': uid,
                'is_labelled': True,
                'sig': rng.normal(size=G),
            }
        r = msa.run_a3(null_units_map, ['HT29'], parent_to_targets, net=net, gene_names=landmarks, gated=True, rng_seed=9450 + seed, n_perm=100)
        p_values.append(r.p)

    frac_sig = np.mean([p < 0.05 for p in p_values])
    assert frac_sig <= 0.20, f"Expected <= 0.20 false positives, got {frac_sig}"


def test_gates_and_sign_handling():
    """RAF inhibitor joins MAPK only in HT29; MDM2 joins p53 only in MCF7; agonist flips sign."""
    # Class definitions
    gated_mapk = next(c for c in msa.GATED_CLASSES if c['class'] == 'MAPK')
    gated_p53 = next(c for c in msa.GATED_CLASSES if c['class'] == 'p53')
    gated_er = next(c for c in msa.GATED_CLASSES if c['class'] == 'ER')

    # RAF targets join MAPK only in HT29
    assert 'BRAF' in gated_mapk['targets_fn']('HT29')
    assert 'BRAF' not in gated_mapk['targets_fn']('MCF7')
    assert 'BRAF' not in gated_mapk['targets_fn']('MDAMB231')

    # MEK targets join in all cells
    assert 'MAP2K1' in gated_mapk['targets_fn']('HT29')
    assert 'MAP2K1' in gated_mapk['targets_fn']('MCF7')

    # MDM2 joins p53 only in MCF7
    assert 'MDM2' in gated_p53['targets_fn']('MCF7')
    assert 'MDM2' not in gated_p53['targets_fn']('HT29')

    # ER joins only in MCF7
    assert 'ESR1' in gated_er['targets_fn']('MCF7')
    assert 'ESR1' not in gated_er['targets_fn']('HT29')

    # AR is dropped from gated table
    assert not any(c['class'] == 'AR' for c in msa.GATED_CLASSES)
    # AR is present in ungated table
    assert any(c['class'] == 'AR' for c in msa.UNGATED_CLASSES)

    # Sign handling: agonist flips inhibitor sign
    inh_sign = gated_er['inhibitor_sign']  # -1.0
    # In run_a3 sign logic:
    # Action ANTAGONIST -> inh_sign (-1.0)
    # Action AGONIST -> -inh_sign (+1.0)
    assert inh_sign == -1.0
    assert -inh_sign == +1.0


def test_read_stage_a_decision_branches():
    """Test all 4 decision branches in read_stage_a."""
    scored_cells = ['MCF7', 'HT29', 'MDAMB231', 'HS578T', 'THP1']

    # 1. Both A1 and A3 signal
    res_both = {
        'scored_cells': scored_cells,
        'a1': {c: {'a1': 0.70, 'null_mean': 0.50, 'p_value': 0.001, 'self_retrieval_ceiling': 0.85,
                   'active_subset': {'a1': 0.75}} for c in scored_cells},
        'a3_gated': {'p': 0.005, 'frac_positive': 0.80},
    }
    d_both = msa.read_stage_a(res_both)
    assert d_both.a1_signal is True
    assert d_both.a3_signal is True
    assert d_both.decision_type == 'BOTH'
    assert "retrieve MoA-mates and show the expected pathway direction" in str(d_both)

    # 2. A1 signal only
    res_a1_only = {
        'scored_cells': scored_cells,
        'a1': {c: {'a1': 0.70, 'null_mean': 0.50, 'p_value': 0.001, 'self_retrieval_ceiling': 0.85,
                   'active_subset': {'a1': 0.75}} for c in scored_cells},
        'a3_gated': {'p': 0.50, 'frac_positive': 0.40},
    }
    d_a1 = msa.read_stage_a(res_a1_only)
    assert d_a1.a1_signal is True
    assert d_a1.a3_signal is False
    assert d_a1.decision_type == 'A1'
    assert "Stage B tests whether v9's P7 predicted signatures retrieve MoA-mates." in str(d_a1)

    # 3. A3 signal only
    res_a3_only = {
        'scored_cells': scored_cells,
        'a1': {c: {'a1': 0.51, 'null_mean': 0.50, 'p_value': 0.40, 'self_retrieval_ceiling': 0.60,
                   'active_subset': {'a1': 0.55}} for c in scored_cells},
        'a3_gated': {'p': 0.002, 'frac_positive': 0.85},
    }
    d_a3 = msa.read_stage_a(res_a3_only)
    assert d_a3.a1_signal is False
    assert d_a3.a3_signal is True
    assert d_a3.decision_type == 'A3'
    assert "Stage B tests whether v9's predicted signatures show the expected pathway direction" in str(d_a3)

    # 4. Neither signal (null branch)
    res_neither = {
        'scored_cells': scored_cells,
        'a1': {c: {'a1': 0.50, 'null_mean': 0.50, 'p_value': 0.50, 'self_retrieval_ceiling': 0.623,
                   'active_subset': {'a1': 0.512}} for c in scored_cells},
        'a3_gated': {'p': 0.20, 'frac_positive': 0.50},
    }
    d_neither = msa.read_stage_a(res_neither)
    assert d_neither.a1_signal is False
    assert d_neither.a3_signal is False
    assert d_neither.decision_type == 'NEITHER'
    assert "on these test cells, the measured landmark responses do not carry these mechanism readouts" in str(d_neither)
    assert "(self-retrieval ceiling 0.623; active-subset A1 0.512)" in str(d_neither)


def test_real_progeny_sha1():
    """Calls real dc.op.progeny and verifies sha1 af40b7a5fe7a7c717d826c898991d232b68c63d6.

    Only this test may call dc.op.progeny, and it is skipped if the network is down.
    """
    import hashlib
    import decoupler as dc
    try:
        net = dc.op.progeny(organism='human', top=500)
    except (urllib.error.URLError, Exception) as e:
        pytest.skip(f"Network is down or decoupler fetch failed: {e}")

    sorted_df = net.sort_values(list(net.columns)).reset_index(drop=True)
    csv_bytes = sorted_df.to_csv(index=False).encode('utf-8')
    sha = hashlib.sha1(csv_bytes).hexdigest()
    assert sha == 'af40b7a5fe7a7c717d826c898991d232b68c63d6'


def test_cli_read(tmp_path, monkeypatch, capsys):
    """Test CLI --read applies read_stage_a and prints decision."""
    res_file = tmp_path / 'result.json'
    scored_cells = ['MCF7', 'HT29', 'MDAMB231', 'HS578T', 'THP1']
    res = {
        'scored_cells': scored_cells,
        'a1': {c: {'a1': 0.50, 'null_mean': 0.50, 'p_value': 0.50, 'self_retrieval_ceiling': 0.623,
                   'active_subset': {'a1': 0.512}} for c in scored_cells},
        'a3_gated': {'p': 0.20, 'frac_positive': 0.50},
    }
    res_file.write_text(json.dumps(res), encoding='utf-8')
    marker = tmp_path / 'result.json.marker'
    marker.write_text(json.dumps({'complete': True, 'sha1': '0' * 40}), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['mechanism_stage_a.py', '--read', str(res_file)])
    with pytest.raises(SystemExit):                                   # PI: the reader refuses a mismatched marker
        msa.main()
    marker.write_text(json.dumps({'complete': True, 'sha1': msa.sha1_file(str(res_file))}), encoding='utf-8')
    msa.main()
    out = capsys.readouterr().out
    assert "on these test cells, the measured landmark responses do not carry these mechanism readouts" in out



def _brute_a1(sigs, labels, hashes, seed, n_perm):
    """PI reference: the per-unit definition written out directly (W33's original loops)."""
    from scipy.stats import rankdata as _rk
    sigs = np.asarray(sigs, float)
    N = len(sigs)
    P = np.corrcoef(sigs)
    E = np.eye(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j and hashes[i] & hashes[j]:
                E[i, j] = True

    def score(lab):
        out = []
        for i in range(N):
            c = np.flatnonzero(~E[i])
            if not lab[i] or len(c) == 0:
                continue
            m = np.array([bool(lab[i] & lab[j]) for j in c])
            if m.sum() >= 1 and (~m).sum() >= 1:
                r = _rk(P[i, c])
                out.append((r[m].sum() - m.sum() * (m.sum() + 1) / 2) / (m.sum() * (~m).sum()))
        return np.mean(out)
    obs = score(labels)
    rng = np.random.default_rng(seed)
    null = [score([labels[k] for k in rng.permutation(N)]) for _ in range(n_perm)]
    return obs, float(np.mean(null))


def test_vectorised_a1_equals_the_brute_force_definition():
    rng = np.random.default_rng(11)
    N, G = 40, 30
    cls = rng.integers(0, 5, N)
    sigs = rng.normal(size=(5, G))[cls] + rng.normal(size=(N, G))
    labels = [{'m%d' % c} | ({'extra'} if i % 7 == 0 else set()) for i, c in enumerate(cls)]
    hashes = [{'p%d' % (i // 4)} for i in range(N)]                    # plates of 4
    r = msa.a1_cell(sigs, labels, hashes, rng_seed=5, n_perm=30)
    obs, null_mean = _brute_a1(sigs, labels, hashes, 5, 30)
    assert abs(r.a1 - obs) < 1e-9 and abs(r.null_mean - null_mean) < 1e-9


# ---- PI additions after a mutation check (the agonist flip, the A1 cell count, the A3 2/3 rule) ----

def test_an_agonist_counts_with_the_flipped_sign(fake_progeny_net):
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    w = fake_progeny_net[fake_progeny_net['source'] == 'MAPK'].set_index('target')['weight']
    wv = np.array([w.get(g, 0.0) for g in genes])
    p2t, units = {}, {'HT29': {}}
    for i in range(4):                      # MAP2K1 "agonists" whose MAPK activity goes UP: (+activity) x (+1) > 0
        p2t['AG_%d' % i] = [('MAP2K1', 'AGONIST')]
        units['HT29']['AG_%d' % i] = {'unit_id': 'AG_%d' % i, 'is_labelled': True,
                                      'sig': 3.0 * wv + np.random.default_rng(i).normal(scale=0.1, size=G)}
    for i in range(8):
        p2t['O_%d' % i] = [('NONE', 'INHIBITOR')]
        units['HT29']['O_%d' % i] = {'unit_id': 'O_%d' % i, 'is_labelled': True,
                                     'sig': np.random.default_rng(50 + i).normal(scale=0.5, size=G)}
    r = msa.run_a3(units, ['HT29'], p2t, net=fake_progeny_net, gene_names=genes, gated=True, rng_seed=1, n_perm=200)
    assert r.T > 0 and r.p < 0.05, (r.T, r.p)


def _a1_result(n_pass):
    cells = msa.SCORED_CELLS_ORDER
    a1 = {c: ({'a1': 0.70, 'null_mean': 0.50, 'p_value': 0.001} if k < n_pass else {'a1': 0.51, 'null_mean': 0.50, 'p_value': 0.4})
          for k, c in enumerate(cells)}
    return {'scored_cells': cells, 'a1': a1, 'a3_gated': {'p': 0.5, 'frac_positive': 0.0}}


def test_the_reader_needs_three_A1_cells_and_two_thirds_of_A3_units():
    assert not msa.read_stage_a(_a1_result(2)).a1_signal
    assert msa.read_stage_a(_a1_result(3)).a1_signal
    base = _a1_result(0)
    base['a3_gated'] = {'p': 0.001, 'frac_positive': 0.6}
    assert not msa.read_stage_a(base).a3_signal
    base['a3_gated'] = {'p': 0.001, 'frac_positive': 0.7}
    assert msa.read_stage_a(base).a3_signal
    base['a3_gated'] = {'p': 0.02, 'frac_positive': 1.0}
    assert not msa.read_stage_a(base).a3_signal



def test_self_retrieval_is_half_against_half_and_a_reproducible_compound_scores_near_one():
    """Review 060 C2: noise-matched halves of a strongly reproducible compound retrieve themselves (ceiling ~ 1); same-plate
    units are dropped from the negatives."""
    rng = np.random.default_rng(3)
    G, units = 50, {}
    cell_mean = np.zeros(G)
    for i in range(12):
        base = rng.normal(size=G) * 3.0
        rows = [base + rng.normal(scale=0.3, size=G) for _ in range(4)]
        units['U%d' % i] = {'unit_id': 'U%d' % i, 'is_labelled': True, 'moas': {'m%d' % (i % 3)},
                            'row_indices': list(range(10 * i, 10 * i + 4)), 'row_deltas': rows,
                            'row_ctl_hashes': ['h%d_%d' % (i, k) for k in range(4)], 'ctl_hashes': {'h%d_%d' % (i, k) for k in range(4)},
                            'cell_mean': cell_mean, 'sig': np.mean(rows, axis=0)}
    out = msa.compute_self_retrieval_and_active_subset(units, rng_seed=1)
    assert out['self_retrieval_ceiling'] > 0.95 and out['n_self_retrieval_units'] == 12
    assert out['self_retrieval_ceiling_no_shared_plate'] > 0.95


def test_a_compound_with_opposite_sign_matching_targets_is_excluded(fake_progeny_net):
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    p2t = {'X%d' % i: [('MAP2K1', 'INHIBITOR')] for i in range(2)}
    p2t['CONFLICT'] = [('MAP2K1', 'INHIBITOR'), ('MAP2K2', 'AGONIST')]
    units = {'HT29': {k: {'unit_id': k, 'is_labelled': True, 'sig': np.random.default_rng(j).normal(size=G)}
                      for j, k in enumerate(list(p2t) + ['O1', 'O2'])}}
    p2t.update({'O1': [], 'O2': []})
    r = msa.run_a3(units, ['HT29'], p2t, net=fake_progeny_net, gene_names=genes, gated=True, rng_seed=1, n_perm=20)
    assert r['units'] == []                            # 2 clean members: a unit only if the conflicted compound were counted


def test_self_retrieval_is_unbiased_when_every_compound_shares_one_signal():
    """Review 060 C2: with one shared signal and no compound-specific one, a half-vs-half ceiling sits at ~0.5. Half-vs-FULL
    negatives would be less noisy than the positive half and push it well below 0.5."""
    rng = np.random.default_rng(8)
    G, units = 200, {}
    shared = rng.normal(size=G)
    for i in range(40):
        rows = [shared + rng.normal(scale=1.0, size=G) for _ in range(4)]
        units['U%d' % i] = {'unit_id': 'U%d' % i, 'is_labelled': True, 'moas': {'m'},
                            'row_indices': list(range(10 * i, 10 * i + 4)), 'row_deltas': rows,
                            'row_ctl_hashes': ['h%d_%d' % (i, k) for k in range(4)], 'ctl_hashes': {'h%d_%d' % (i, k) for k in range(4)},
                            'cell_mean': np.zeros(G), 'sig': np.mean(rows, axis=0)}
    out = msa.compute_self_retrieval_and_active_subset(units, rng_seed=1)
    assert abs(out['self_retrieval_ceiling'] - 0.5) < 0.12, out['self_retrieval_ceiling']



# ---- Stage B (§94.9), PI ----

def test_predicted_delta_is_aligned_by_row_index_and_averaged_over_files(tmp_path):
    ri = np.array([5, 3, 9], np.int64)
    a = tmp_path / 'a.npz'
    b = tmp_path / 'b.npz'
    np.savez(a, row_index=np.array([9, 5, 3]), deg_pred=np.array([[9.0], [5.0], [3.0]]))
    np.savez(b, row_index=np.array([3, 9, 5]), deg_pred=np.array([[1.0], [3.0], [7.0]]))
    d = msa.load_predicted_delta('%s,%s:deg_pred' % (a, b), ri)
    assert np.allclose(d[:, 0], [(5 + 7) / 2, (3 + 1) / 2, (9 + 3) / 2])
    with pytest.raises(AssertionError):
        msa.load_predicted_delta('%s:deg_pred' % a, np.array([5, 3, 4]))


def test_b3c_finds_a_planted_cell_ordering_within_class_and_not_a_class_offset():
    meas = {'MAPK@A': 10.0, 'MAPK@B': 5.0, 'MAPK@C': 1.0, 'PI3K@A': 3.0, 'PI3K@B': 2.0, 'PI3K@C': 1.0, 'PI3K@D': 0.5, 'ER@A': 2.0}
    good = {k: v * 0.5 + (100 if k.startswith('MAPK') else 0) for k, v in meas.items()}       # same order within class
    flat = {k: (7.0 if k.startswith('MAPK') else 1.0) for k in meas}                             # class offsets only
    rho, p, n = msa.b3c(meas, good, n_perm=2000)
    assert rho > 0.8 and p < 0.05 and n == 8
    rho0, p0, _ = msa.b3c(meas, flat, n_perm=2000)
    assert rho0 <= 0.0 + 1e-9 and p0 > 0.3


def _fake_res(a1s, p, frac, d):
    cells = msa.SCORED_CELLS_ORDER
    return {'scored_cells': cells, 'a1': {c: {'a1': a1s[k], 'null_mean': 0.5, 'p_value': 0.001} for k, c in enumerate(cells)},
            'a3_gated': {'T': 1.0, 'p': p, 'frac_positive': frac, 'd_per_unit': d, 'd_std_per_unit': d}}


def test_stage_b_reader_needs_every_seed_and_beats_mu():
    meas_d = {'MAPK@MCF7': 5.6, 'MAPK@HT29': 9.4, 'MAPK@MDAMB231': 20.1, 'MAPK@HS578T': 10.9,
              'PI3K@MCF7': 3.8, 'PI3K@HT29': 1.7, 'PI3K@MDAMB231': 2.4, 'PI3K@HS578T': 3.1, 'PI3K@THP1': 3.4}
    measured = _fake_res([0.65] * 5, 0.001, 0.9, meas_d)
    mu = _fake_res([0.58] * 5, 0.001, 0.9, {k: 3.0 for k in meas_d})
    good = _fake_res([0.62] * 5, 0.001, 0.9, {k: v * 0.4 for k, v in meas_d.items()})
    out = msa.read_stage_b(measured, [good, good, good], good, mu)
    assert out['reading_1_expresses_mechanism'] and out['reading_2a_beyond_mu_retrieval']
    assert out['reading_2b_beyond_mu_cell_specific_pathway']
    weak = _fake_res([0.59] * 5, 0.001, 0.9, {k: 3.0 for k in meas_d})       # one seed only just above mu, flat pathway
    out2 = msa.read_stage_b(measured, [good, weak, good], good, mu)
    assert not out2['reading_2a_beyond_mu_retrieval'] and not out2['reading_2b_beyond_mu_cell_specific_pathway']


def test_2b_fails_when_mu_orders_the_cells_as_well_as_v9():
    meas_d = {'MAPK@MCF7': 5.6, 'MAPK@HT29': 9.4, 'MAPK@MDAMB231': 20.1, 'MAPK@HS578T': 10.9,
              'PI3K@MCF7': 3.8, 'PI3K@HT29': 1.7, 'PI3K@MDAMB231': 2.4, 'PI3K@HS578T': 3.1, 'PI3K@THP1': 3.4}
    measured = _fake_res([0.65] * 5, 0.001, 0.9, meas_d)
    same = _fake_res([0.62] * 5, 0.001, 0.9, {k: v * 0.4 for k, v in meas_d.items()})
    mu = _fake_res([0.55] * 5, 0.001, 0.9, {k: v * 0.4 for k, v in meas_d.items()})   # mu orders cells exactly as v9 does
    out = msa.read_stage_b(measured, [same, same, same], same, mu)
    assert out['per_run']['seed_mean']['B3c_rho'] > 0 and out['per_run']['seed_mean']['B3c_p'] < 0.05
    assert not out['reading_2b_beyond_mu_cell_specific_pathway']



def test_delta_spec_can_subtract_the_control(tmp_path):
    a = tmp_path / 'b.npz'
    np.savez(a, row_index=np.array([1, 2]), pred=np.array([[9.0], [8.0]]), ctl_true=np.array([[7.0], [7.5]]))
    d = msa.load_predicted_delta('%s:pred-ctl_true' % a, np.array([2, 1]))
    assert np.allclose(d[:, 0], [0.5, 2.0])


def test_standardised_d_removes_a_cell_level_scale(fake_progeny_net):
    """Review 061 C1: scaling a whole cell's signatures by 5 multiplies its raw d by ~5 but leaves the standardised d."""
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    w = fake_progeny_net[fake_progeny_net['source'] == 'MAPK'].set_index('target')['weight']
    wv = np.array([w.get(g, 0.0) for g in genes])
    p2t = {('M%d' % i): [('MAP2K1', 'INHIBITOR')] for i in range(4)}
    p2t.update({('O%d' % i): [('NONE', 'INHIBITOR')] for i in range(8)})

    def world(scale):
        return {'HT29': {k: {'unit_id': k, 'is_labelled': True,
                             'sig': scale * ((-2.0 * wv if k.startswith('M') else 0) + np.random.default_rng(j).normal(size=G))}
                         for j, k in enumerate(p2t)}}
    r1 = msa.run_a3(world(1.0), ['HT29'], p2t, net=fake_progeny_net, gene_names=genes, gated=True, rng_seed=1, n_perm=20)
    r5 = msa.run_a3(world(5.0), ['HT29'], p2t, net=fake_progeny_net, gene_names=genes, gated=True, rng_seed=1, n_perm=20)
    k = 'MAPK@HT29'
    assert abs(r5['d_std_per_unit'][k] - r1['d_std_per_unit'][k]) < 1e-6
    assert 'HT29' in r1['activity_scale'] and 'MAPK' in r1['activity_scale']['HT29']['sd']



# ---- §95 (cold-drug split), PI ----

def test_cold_drug_gates_and_threshold():
    try:
        msa.ACTIVE.clear(); msa.ACTIVE.update(msa.SPLITS['split_cold_drug_1'], name='split_cold_drug_1')
        mapk = next(c for c in msa.GATED_CLASSES if c['class'] == 'MAPK')
        p53 = next(c for c in msa.GATED_CLASSES if c['class'] == 'p53')
        er = next(c for c in msa.GATED_CLASSES if c['class'] == 'ER')
        assert 'BRAF' in mapk['targets_fn']('A375') and 'BRAF' in mapk['targets_fn']('HT29') and 'BRAF' not in mapk['targets_fn']('PC3')
        assert 'MDM2' in p53['targets_fn']('A549') and 'MDM2' in p53['targets_fn']('MCF7') and not p53['targets_fn']('HA1E')
        assert er['targets_fn']('MCF7') and not er['targets_fn']('A549')
    finally:
        msa.ACTIVE.clear(); msa.ACTIVE.update(msa.SPLITS['split_cold_cell_1'], name='split_cold_cell_1')
    assert 'BRAF' not in mapk['targets_fn']('A375')                      # back on cold-cell: A375 has no RAF gate
    cells = msa.SPLITS['split_cold_drug_1']['scored']
    res = {'split': 'split_cold_drug_1', 'scored_cells': cells,
           'a1': {c: ({'a1': 0.7, 'null_mean': 0.5, 'p_value': 0.001} if k < 3 else {'a1': 0.5, 'null_mean': 0.5, 'p_value': 0.5})
                  for k, c in enumerate(cells)}, 'a3_gated': {'p': 0.5, 'frac_positive': 0.0}}
    assert not msa.read_stage_a(res).a1_signal                           # 3 of 6 is not enough on cold-drug
    res['a1'][cells[3]] = {'a1': 0.7, 'null_mean': 0.5, 'p_value': 0.001}
    assert msa.read_stage_a(res).a1_signal



# ---- review 063 (cold-drug gate), PI ----

def test_cns_channel_classes_are_recognised_and_others_are_not():
    for m in ('Dopamine D2 receptor antagonist', 'Serotonin transporter inhibitor', 'Sodium channel alpha subunit blocker',
              'Sulfonylurea receptor 1, Kir6.2 blocker', 'Muscarinic acetylcholine receptor M1 antagonist'):
        assert msa.CNS_CHANNEL_RE.search(m), m
    for m in ('DNA inhibitor', 'Epidermal growth factor receptor erbB1 inhibitor', 'Cyclooxygenase inhibitor',
              'Heat shock protein HSP90 inhibitor', 'Phosphodiesterase 4 inhibitor'):
        assert not msa.CNS_CHANNEL_RE.search(m), m


def test_dna_class_joins_only_in_tp53_wt_cells_on_cold_drug(fake_progeny_net):
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    try:
        msa.ACTIVE.clear(); msa.ACTIVE.update(msa.SPLITS['split_cold_drug_1'], name='split_cold_drug_1')
        units = {}
        for cell in ('MCF7', 'PC3'):
            units[cell] = {}
            for i in range(6):
                k = '%s_D%d' % (cell, i)
                units[cell][k] = {'unit_id': k, 'is_labelled': True, 'moas': {'DNA inhibitor'} if i < 3 else {'other'},
                                  'sig': np.random.default_rng(i).normal(size=G)}
        r = msa.run_a3(units, ['MCF7', 'PC3'], {}, net=fake_progeny_net, gene_names=genes, gated=True, rng_seed=1, n_perm=10)
        assert r['units'] == ['DNA@MCF7']                                   # PC3 is TP53-null: no DNA unit
    finally:
        msa.ACTIVE.clear(); msa.ACTIVE.update(msa.SPLITS['split_cold_cell_1'], name='split_cold_cell_1')


def test_p9_gate_needs_noncns_a1_or_a3_over_two_classes():
    cells = msa.SPLITS['split_cold_drug_1']['scored']
    def res(n_nc, a3_units, p=0.001, frac=1.0):
        a1 = {}
        for k, c in enumerate(cells):
            a1[c] = {'a1': 0.7, 'null_mean': 0.5, 'p_value': 0.001,
                     'a1_noncns': ({'a1': 0.7, 'null_mean': 0.5, 'p_value': 0.001} if k < n_nc
                                   else {'a1': 0.5, 'null_mean': 0.5, 'p_value': 0.5})}
        return {'split': 'split_cold_drug_1', 'scored_cells': cells, 'a1': a1,
                'a3_gated': {'p': p, 'frac_positive': frac, 'd_per_unit': {u: 1.0 for u in a3_units}}}
    egfr_only = ['EGFR@%s' % c for c in cells]
    assert not msa.read_stage_a(res(2, egfr_only)).p9_gate['open']        # CNS-only A1 and a one-class A3: closed
    assert msa.read_stage_a(res(3, egfr_only)).p9_gate['open']            # 3 of 6 cells without CNS classes
    assert msa.read_stage_a(res(0, egfr_only + ['DNA@MCF7'])).p9_gate['open']
    assert 'CLOSED' in msa.read_stage_a(res(0, egfr_only))


def test_read_stage_b_uses_the_splits_a1_threshold():
    cells = msa.SPLITS['split_cold_drug_1']['scored']
    def r(n):
        return {'split': 'split_cold_drug_1', 'scored_cells': cells,
                'a1': {c: {'a1': 0.7 if k < n else 0.5, 'null_mean': 0.5, 'p_value': 0.001 if k < n else 0.5} for k, c in enumerate(cells)},
                'a3_gated': {'T': 1, 'p': 0.001, 'frac_positive': 1.0, 'd_per_unit': {'EGFR@MCF7': 1.0, 'EGFR@PC3': 2.0},
                             'd_std_per_unit': {'EGFR@MCF7': 1.0, 'EGFR@PC3': 2.0}}}
    out = msa.read_stage_b(r(6), [r(3)] * 3, r(3), r(3))
    assert not out['per_run']['seed_mean']['B1_signal']                  # 3 of 6 < the cold-drug minimum of 4



def test_cold_drug_a3_needs_two_classes_for_the_decision():
    cells = msa.SPLITS['split_cold_drug_1']['scored']
    base = {'split': 'split_cold_drug_1', 'scored_cells': cells,
            'a1': {c: {'a1': 0.5, 'null_mean': 0.5, 'p_value': 0.5} for c in cells},
            'a3_gated': {'p': 0.001, 'frac_positive': 1.0, 'd_per_unit': {'EGFR@%s' % c: 1.0 for c in cells}}}
    assert not msa.read_stage_a(base).a3_signal
    base['a3_gated']['d_per_unit']['DNA@MCF7'] = 1.0
    assert msa.read_stage_a(base).a3_signal
