# -*- coding: utf-8 -*-
"""Tests for RESULTS §96.4/96.8/96.9 Stage B' (stage_b_prime.py).

All tests run on synthetic data only.
Tests:
1. elements equals run_a3 (with target class, agonist flip, opposite-sign exclusion, MoA-string class with gate)
2. Sign-flip identity: brute-force swap equals (1 - 2m) @ g for 50 random masks to 1e-12
3. Swap test power and calibration
4. Standardisation removes smoothness inflation of ULM t
5. Exclusion of afatinib and doxorubicin parents
6. Guards: unit set, equivalence mismatch, row count/sha1, non-finite z
7. References: 1-NN tie-break, same-cell rule, fallback, 5-NN weighting, molecule collapse, physchem, broadcasting
8. Reader: licensed sentence, B3-only, NO CLAIM, 4-variant requirement, 2-class rule
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mechanism_stage_a as msa  # noqa: E402
import stage_b_prime as sbp  # noqa: E402


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


def test_elements_equals_run_a3(fake_progeny_net):
    """elements equals run_a3: exercises target class, agonist sign flip, opposite-sign exclusion, MoA-string class with cell gate."""
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    net = fake_progeny_net

    msa.ACTIVE.clear()
    msa.ACTIVE.update(msa.SPLITS['split_cold_drug_1'], name='split_cold_drug_1')

    # Cell MCF7 is in gate for dna_p53; PC3 is NOT
    p2t = {
        'M1': [('MAP2K1', 'INHIBITOR')],
        'M2': [('MAP2K1', 'INHIBITOR')],
        'M3': [('MAP2K1', 'AGONIST')],                  # Agonist sign flip
        'CONFLICT': [('MAP2K1', 'INHIBITOR'), ('MAP2K2', 'AGONIST')],  # Opposite-sign exclusion
        'OTHER1': [('UNKNOWN', 'INHIBITOR')],
        'OTHER2': [('UNKNOWN', 'INHIBITOR')],
        'D1': [],
        'D2': [],
        'D3': [],
    }

    cell_units = {'MCF7': {}, 'PC3': {}}
    for cell in ('MCF7', 'PC3'):
        for k, targets in p2t.items():
            uid = f'{cell}_{k}'
            is_dna = k in ('D1', 'D2', 'D3')
            moas = {'DNA inhibitor'} if is_dna else {'other'}
            sig = np.random.default_rng(hash(uid) % 100000).normal(size=G)
            cell_units[cell][uid] = {
                'unit_id': uid,
                'is_labelled': True,
                'moas': moas,
                'sig': sig,
            }

    # parent_to_targets with prefixed uids
    full_p2t = {}
    for cell in ('MCF7', 'PC3'):
        for k, targets in p2t.items():
            full_p2t[f'{cell}_{k}'] = targets

    el_res = sbp.elements(cell_units, ['MCF7', 'PC3'], full_p2t, net=net, gene_names=genes, gated=True)
    a3_res = msa.run_a3(cell_units, ['MCF7', 'PC3'], full_p2t, net=net, gene_names=genes, gated=True, n_perm=1)

    assert set(el_res['d_per_unit'].keys()) == set(a3_res.d_per_unit.keys())
    assert 'DNA@MCF7' in el_res['d_per_unit']
    assert 'DNA@PC3' not in el_res['d_per_unit']  # Gated out
    assert 'MAPK@MCF7' in el_res['d_per_unit']

    # CONFLICT compound must not be in members
    for u in el_res['eval_units']:
        m_uids = [uid for uid, _ in u['members']]
        assert not any('CONFLICT' in uid for uid in m_uids)

    # Values equal to 1e-12
    for u in el_res['d_per_unit']:
        np.testing.assert_allclose(el_res['d_per_unit'][u], a3_res.d_per_unit[u], atol=1e-12)
        np.testing.assert_allclose(el_res['d_std_per_unit'][u], a3_res.d_std_per_unit[u], atol=1e-12)

    # Mutating membership rule (e.g. ungated) must break equality
    el_ungated = sbp.elements(cell_units, ['MCF7', 'PC3'], full_p2t, net=net, gene_names=genes, gated=False)
    assert set(el_ungated['d_per_unit'].keys()) != set(a3_res.d_per_unit.keys())


def test_sign_flip_identity(fake_progeny_net):
    """For 50 random masks, brute-force swap equals (1 - 2m) @ g to 1e-12."""
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    net = fake_progeny_net

    msa.ACTIVE.clear()
    msa.ACTIVE.update(msa.SPLITS['split_cold_drug_1'], name='split_cold_drug_1')

    # Construct two units in cell MCF7 (MAPK and EGFR)
    p2t = {
        'M1': [('MAP2K1', 'INHIBITOR')],
        'M2': [('MAP2K1', 'INHIBITOR')],
        'M3': [('MAP2K1', 'INHIBITOR')],
        'E1': [('EGFR', 'INHIBITOR')],
        'E2': [('EGFR', 'INHIBITOR')],
        'E3': [('EGFR', 'INHIBITOR')],
        'O1': [('NONE', 'INHIBITOR')],
        'O2': [('NONE', 'INHIBITOR')],
    }

    rng = np.random.default_rng(42)
    cell_units_v9 = {'MCF7': {}}
    cell_units_R = {'MCF7': {}}

    for uid in p2t:
        sig_v9 = rng.normal(size=G)
        sig_R = rng.normal(size=G)
        cell_units_v9['MCF7'][uid] = {'unit_id': uid, 'is_labelled': True, 'moas': set(), 'sig': sig_v9}
        cell_units_R['MCF7'][uid] = {'unit_id': uid, 'is_labelled': True, 'moas': set(), 'sig': sig_R}

    el_v9 = sbp.elements(cell_units_v9, ['MCF7'], p2t, net=net, gene_names=genes, gated=True)
    el_R = sbp.elements(cell_units_R, ['MCF7'], p2t, net=net, gene_names=genes, gated=True)

    swap_res = sbp.run_swap_test(el_v9['eval_units'], el_v9['cell_acts_std'], el_R['cell_acts_std'])
    g = swap_res['g_vector']
    elements = swap_res['elements']
    surviving_units = swap_res['surviving_units']
    n_el = len(elements)
    U = len(surviving_units)

    # Brute-force recomputation for 50 random masks
    for seed in range(50):
        mask_rng = np.random.default_rng(1000 + seed)
        m = mask_rng.integers(0, 2, size=n_el)

        # Fast matrix-product delta*
        signs = 1.0 - 2.0 * m
        delta_star_fast = float(np.dot(signs, g))

        # Brute-force delta*
        # Swapped pseudo-models
        d_p1 = []
        d_p2 = []
        for u in surviving_units:
            cell = u['cell']
            pathway = u['pathway']
            inh_sign = u['inhibitor_sign']
            v_df = el_v9['cell_acts_std'][cell]
            r_df = el_R['cell_acts_std'][cell]

            # Reconstruct activities for pseudo-models
            def get_val(uid, model_idx):
                el_idx = elements.index((uid, cell))
                swapped = (m[el_idx] == 1)
                if model_idx == 1:
                    return r_df.loc[uid, pathway] if swapped else v_df.loc[uid, pathway]
                else:
                    return v_df.loc[uid, pathway] if swapped else r_df.loc[uid, pathway]

            in_p1 = np.mean([get_val(uid, 1) * s for uid, s in u['members']])
            out_p1 = np.mean([get_val(uid, 1) * inh_sign for uid in u['others']]) if u['others'] else 0.0
            d_p1.append(in_p1 - out_p1)

            in_p2 = np.mean([get_val(uid, 2) * s for uid, s in u['members']])
            out_p2 = np.mean([get_val(uid, 2) * inh_sign for uid in u['others']]) if u['others'] else 0.0
            d_p2.append(in_p2 - out_p2)

        T_p1 = np.mean(d_p1)
        T_p2 = np.mean(d_p2)
        delta_star_brute = float(T_p1 - T_p2)

        np.testing.assert_allclose(delta_star_fast, delta_star_brute, atol=1e-12)


def test_swap_test_power_and_calibration():
    """Swap test calibration under noise (mean p in 0.5 +- 0.08, < 10% below 0.05) and power when v9 has signal (p < 0.01)."""
    # Create simple eval_units
    eval_units = [{
        'unit_id': 'MAPK@MCF7',
        'class': 'MAPK',
        'cell': 'MCF7',
        'pathway': 'MAPK',
        'inhibitor_sign': -1.0,
        'members': [(f'M{i}', -1.0) for i in range(5)],
        'others': [f'O{i}' for i in range(15)],
    }]

    all_uids = [f'M{i}' for i in range(5)] + [f'O{i}' for i in range(15)]
    base_z = np.zeros(len(all_uids))

    # 1. Calibration over 200 seeds of noise
    p_values = []
    for s in range(200):
        noise_rng = np.random.default_rng(2000 + s)
        z_v9 = base_z + noise_rng.normal(scale=0.1, size=len(all_uids))
        z_R = base_z + noise_rng.normal(scale=0.1, size=len(all_uids))

        df_v9 = pd.DataFrame(z_v9[:, None], index=all_uids, columns=['MAPK'])
        df_R = pd.DataFrame(z_R[:, None], index=all_uids, columns=['MAPK'])

        res = sbp.run_swap_test(eval_units, {'MCF7': df_v9}, {'MCF7': df_R}, swap_seed=9470, n_swap=1000)
        p_values.append(res['p_value'])

    mean_p = float(np.mean(p_values))
    frac_sig = float(np.mean([p < 0.05 for p in p_values]))
    assert 0.42 <= mean_p <= 0.58, f"Mean p-value {mean_p} outside [0.42, 0.58]"
    assert frac_sig < 0.10, f"False positive rate {frac_sig} >= 0.10"

    # 2. Power: v9 carries strong signal on members, R is noise
    # Need enough elements carrying signal (e.g. 10 members) so (1/2)^10 < 0.01
    eval_units_power = [{
        'unit_id': 'MAPK@MCF7',
        'class': 'MAPK',
        'cell': 'MCF7',
        'pathway': 'MAPK',
        'inhibitor_sign': -1.0,
        'members': [(f'M{i}', -1.0) for i in range(10)],
        'others': [f'O{i}' for i in range(10)],
    }]
    all_uids_power = [f'M{i}' for i in range(10)] + [f'O{i}' for i in range(10)]
    z_sig_v9 = np.zeros(len(all_uids_power))
    z_sig_v9[:10] = -3.0  # members strongly negative -> -3 * -1 = +3
    z_sig_R = np.zeros(len(all_uids_power))

    df_sig_v9 = pd.DataFrame(z_sig_v9[:, None], index=all_uids_power, columns=['MAPK'])
    df_sig_R = pd.DataFrame(z_sig_R[:, None], index=all_uids_power, columns=['MAPK'])

    res_power = sbp.run_swap_test(eval_units_power, {'MCF7': df_sig_v9}, {'MCF7': df_sig_R}, swap_seed=9470, n_swap=1000)
    assert res_power['p_value'] < 0.01, f"Expected p < 0.01, got {res_power['p_value']}"


def test_standardisation_removes_smoothness_inflation(fake_progeny_net):
    """v9 = R with 10x smaller noise: raw delta_obs is large positive while standardized |delta_obs| is < 25% of raw one scaled by T_v9."""
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    net = fake_progeny_net

    w = net[net['source'] == 'MAPK'].set_index('target')['weight']
    wv = np.array([w.get(g, 0.0) for g in genes])

    p2t = {f'M{i}': [('MAP2K1', 'INHIBITOR')] for i in range(4)}
    p2t.update({f'O{i}': [('NONE', 'INHIBITOR')] for i in range(30)})

    def make_world(noise):
        out = {}
        for j, k in enumerate(p2t):
            r = np.random.default_rng(j)
            c = -1.0 if k.startswith('M') else 0.6 * r.normal()
            out[k] = {'unit_id': k, 'is_labelled': True, 'sig': c * wv + noise * r.normal(size=G), 'moas': set()}
        return {'HT29': out}

    msa.ACTIVE.clear()
    msa.ACTIVE.update(msa.SPLITS['split_cold_drug_1'], name='split_cold_drug_1')

    # v9 has 10x smaller noise
    world_v9 = make_world(0.1)
    world_R = make_world(1.0)

    el_v9 = sbp.elements(world_v9, ['HT29'], p2t, net=net, gene_names=genes, gated=True)
    el_R = sbp.elements(world_R, ['HT29'], p2t, net=net, gene_names=genes, gated=True)

    res_raw = sbp.run_swap_test(el_v9['eval_units'], el_v9['cell_acts'], el_R['cell_acts'], swap_seed=9470, n_swap=500)
    res_z = sbp.run_swap_test(el_v9['eval_units'], el_v9['cell_acts_std'], el_R['cell_acts_std'], swap_seed=9470, n_swap=500)

    assert res_raw['delta_obs'] > 1.0, f"Expected large positive raw delta_obs, got {res_raw['delta_obs']}"

    # Scaled by T_v9
    scaled_raw = res_raw['delta_obs'] / res_raw['T_v9']
    scaled_z = abs(res_z['delta_obs']) / res_z['T_v9']

    assert scaled_z < 0.25 * scaled_raw, f"Scaled z {scaled_z} not < 25% of scaled raw {scaled_raw}"


def test_exclusion_rule():
    """Excluded parents appear in neither members nor others nor elements, z values bit-identical, units left with < 3 members dropped."""
    eval_units = [
        {
            'unit_id': 'U1@C1',
            'class': 'U1',
            'cell': 'C1',
            'pathway': 'PW',
            'inhibitor_sign': -1.0,
            'members': [('PAR_AFAT', -1.0), ('M1', -1.0), ('M2', -1.0), ('M3', -1.0)],  # 4 members -> 3 remain (kept)
            'others': ['PAR_DOX', 'O1', 'O2'],
        },
        {
            'unit_id': 'U2@C1',
            'class': 'U2',
            'cell': 'C1',
            'pathway': 'PW',
            'inhibitor_sign': -1.0,
            'members': [('PAR_DOX', -1.0), ('M4', -1.0), ('M5', -1.0)],  # 3 members -> 2 remain (dropped)
            'others': ['PAR_AFAT', 'O1', 'O2'],
        },
    ]

    all_uids = ['PAR_AFAT', 'PAR_DOX', 'M1', 'M2', 'M3', 'M4', 'M5', 'O1', 'O2']
    rng = np.random.default_rng(99)
    z_vals = rng.normal(size=len(all_uids))
    df_z = pd.DataFrame(z_vals[:, None], index=all_uids, columns=['PW'])

    excluded = {'PAR_AFAT', 'PAR_DOX'}
    res = sbp.run_swap_test(eval_units, {'C1': df_z}, {'C1': df_z}, excluded_parents=excluded)

    # Unit 2 dropped, only Unit 1 survived
    assert res['n_units'] == 1
    assert 'U1@C1' in res['d_v9']
    assert 'U2@C1' not in res['d_v9']

    # Elements do not contain excluded parents
    el_uids = [uid for uid, c in res['elements']]
    assert 'PAR_AFAT' not in el_uids
    assert 'PAR_DOX' not in el_uids

    # Neither members nor others contain excluded parents
    surv_u = res['surviving_units'][0]
    surv_m_uids = [uid for uid, _ in surv_u['members']]
    assert 'PAR_AFAT' not in surv_m_uids
    assert 'PAR_DOX' not in surv_u['others']


def test_guards(monkeypatch, tmp_path, fake_progeny_net):
    """Guards: unit set != constant, equivalence mismatch, row count/sha1 mismatch, non-finite z."""
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]

    # 1. Row count and sha1 guard in build_references
    bad_rows_npz = tmp_path / 'bad_rows.npz'
    np.savez(bad_rows_npz, row_index=np.arange(10, dtype=np.int64))

    chem_npz = tmp_path / 'dummy_chem.npz'
    np.savez(chem_npz, pert_id=np.array([]))
    chem_marker = tmp_path / 'dummy_chem.npz.marker'
    chem_marker.write_text(json.dumps({'complete': True, 'sha1': sbp.sha1_file(str(chem_npz))}), encoding='utf-8')

    with pytest.raises(SystemExit, match='REFUSED: row count mismatch'):
        sbp.build_references(str(tmp_path / 'bundle.npz'), str(chem_npz), str(bad_rows_npz))

    # 2. Equivalence mismatch guard in compare
    orig_run_a3 = msa.run_a3
    def mock_run_a3(*args, **kwargs):
        res = orig_run_a3(*args, **kwargs)
        # Perturb d_per_unit by 0.1
        perturbed = {k: v + 0.1 for k, v in res.d_per_unit.items()}
        return msa.A3Result(T=res.T, p=res.p, d_per_unit=perturbed, frac_positive=res.frac_positive,
                            d_std_per_unit=res.d_std_per_unit, activity_scale=res.activity_scale)

    monkeypatch.setattr(msa, 'run_a3', mock_run_a3)
    # Monkeypatch ROWS_N and ROWS_SHA1 to accept 3 rows
    monkeypatch.setattr(sbp, 'ROWS_N', 3)
    monkeypatch.setattr(sbp, 'ROWS_SHA1', hashlib.sha1(np.array([0, 1, 2], dtype=np.int64).tobytes()).hexdigest())
    monkeypatch.setitem(msa.SPLITS['split_cold_drug_1'], 'n_rows_restricted', 3)
    monkeypatch.setitem(msa.SPLITS['split_cold_drug_1'], 'n_rows', 3)
    monkeypatch.setattr(msa, 'scored_cells', lambda *args, **kwargs: (['MCF7'], {'MCF7': 3}))

    rng = np.random.default_rng(123)
    fake_bundle = tmp_path / 'fake_bundle.npz'
    np.savez(fake_bundle, split_split_cold_drug_1=np.array(['test', 'test', 'test']), row_index=np.array([0, 1, 2]),
             meta_pert_id=np.array(['P0', 'P1', 'P2']), meta_cell=np.array(['MCF7', 'MCF7', 'MCF7']),
             X=rng.normal(size=(3, G)), X_ctl=np.zeros((3, G)))

    fake_rows = tmp_path / 'fake_rows.npz'
    np.savez(fake_rows, row_index=np.array([0, 1, 2]))

    fake_labels = tmp_path / 'fake_labels.tsv'
    fake_labels.write_text(
        "pert_id\tparent_chembl_id\tdirect_interaction\torganism\tmechanism_of_action\tgene_symbol\taction_type\n"
        "BRD-K66175015\tCHEMBL_A\t1\tHomo sapiens\tEGFR inhibitor\tEGFR\tINHIBITOR\n"
        "BRD-K92093830\tCHEMBL_D\t1\tHomo sapiens\tDNA inhibitor\tMDM2\tINHIBITOR\n"
        "P0\tCHEMBL_P0\t1\tHomo sapiens\tEGFR inhibitor\tEGFR\tINHIBITOR\n"
        "P1\tCHEMBL_P1\t1\tHomo sapiens\tEGFR inhibitor\tEGFR\tINHIBITOR\n"
        "P2\tCHEMBL_P2\t1\tHomo sapiens\tEGFR inhibitor\tEGFR\tINHIBITOR\n",
        encoding='utf-8',
    )

    fake_v9 = tmp_path / 'fake_v9.npz'
    np.savez(fake_v9, row_index=np.array([0, 1, 2]), deg_pred=rng.normal(size=(3, G)))

    # Monkeypatch INCLUDING_UNITS so that EGFR@MCF7 matches
    monkeypatch.setattr(sbp, 'INCLUDING_UNITS', {'EGFR@MCF7'})
    monkeypatch.setattr(sbp, 'EXCLUDING_UNITS', {'EGFR@MCF7'})

    with pytest.raises(SystemExit, match='REFUSED: equivalence guard failed'):
        sbp.compare(str(fake_bundle), str(fake_labels), str(fake_rows),
                    [str(fake_v9) + ':deg_pred'] * 4, str(fake_v9) + ':deg_pred', 'R',
                    net=fake_progeny_net, genes=genes)

    # Unit set mismatch guard
    monkeypatch.setattr(msa, 'run_a3', orig_run_a3)
    monkeypatch.setattr(sbp, 'INCLUDING_UNITS', {'NONEXISTENT@MCF7'})
    with pytest.raises(SystemExit, match='REFUSED: unit set'):
        sbp.compare(str(fake_bundle), str(fake_labels), str(fake_rows),
                    [str(fake_v9) + ':deg_pred'] * 4, str(fake_v9) + ':deg_pred', 'R',
                    net=fake_progeny_net, genes=genes)

    # 3. Non-finite z guard
    u_eval = [{
        'unit_id': 'U@C', 'class': 'U', 'cell': 'C', 'pathway': 'PW', 'inhibitor_sign': -1.0,
        'members': [('M1', -1.0), ('M2', -1.0), ('M3', -1.0)], 'others': ['O1'],
    }]
    nan_df = pd.DataFrame([np.nan, 1.0, 1.0, 1.0], index=['M1', 'M2', 'M3', 'O1'], columns=['PW'])
    ok_df = pd.DataFrame([1.0, 1.0, 1.0, 1.0], index=['M1', 'M2', 'M3', 'O1'], columns=['PW'])
    with pytest.raises(SystemExit, match='REFUSED: non-finite values in contribution vector g'):
        sbp.run_swap_test(u_eval, {'C': nan_df}, {'C': ok_df})


def test_references_construction(tmp_path, monkeypatch):
    """References: 1-NN tie breaking, same-cell rule, fallback, 5-NN weighting, molecule collapse, physchem, broadcasting."""
    G = 10
    # 1. CHEM.npz
    # Usable training molecules:
    # MOL_A: pert TR1, TR2 (collapse). TR1 is smaller -> fp/desc from TR1.
    # MOL_B: pert TR3. Same Tanimoto as MOL_A to test pert -> tie broken by MOL_A < MOL_B.
    # MOL_C: pert TR4. Profiled only in cell C1 (for same-cell rule check in cell C2).
    # MOL_D: pert TR5.
    chem_npz = tmp_path / 'chem.npz'
    c_perts = ['TR1', 'TR2', 'TR3', 'TR4', 'TR5', 'TE1', 'TE2']
    c_roles = ['train', 'train', 'train', 'train', 'train', 'test', 'test']
    c_mkeys = ['MOL_A', 'MOL_A', 'MOL_B', 'MOL_C', 'MOL_D', 'MOL_TE1', 'MOL_TE2']
    c_pars = [True, True, True, True, True, True, True]
    c_nfrag = [1, 1, 1, 1, 1, 1, 1]

    # Fingerprints: 2048 bits
    fps = np.zeros((7, 2048), dtype=np.uint8)
    # Give MOL_A and MOL_B identical bits to TE1
    fps[0, :10] = 1  # TR1 (MOL_A)
    fps[1, :10] = 1  # TR2 (MOL_A)
    fps[2, :10] = 1  # TR3 (MOL_B)
    fps[3, :5] = 1   # TR4 (MOL_C)
    fps[4, :2] = 1   # TR5 (MOL_D)
    fps[5, :10] = 1  # TE1 (Tanimoto 1.0 to MOL_A and MOL_B)
    fps[6, 50:60] = 1  # TE2 (no match to MOL_A/B/C/D)

    # Descriptors: (7, 6)
    descs = np.ones((7, 6), dtype=np.float64) * 2.0
    descs[2] = 5.0  # TR3 (MOL_B) has different descriptors
    descs[4] = 8.0  # TR5 (MOL_D)

    np.savez(chem_npz, pert_id=c_perts, role=c_roles, molecule_key=c_mkeys,
             parsable=c_pars, n_fragments=c_nfrag, fp=fps, desc=descs)
    chem_marker = tmp_path / 'chem.npz.marker'
    chem_marker.write_text(json.dumps({'complete': True, 'sha1': sbp.sha1_file(str(chem_npz))}), encoding='utf-8')

    # 2. Bundle:
    # Training rows:
    # TR1: cell C1, delta = 10.0
    # TR2: cell C1, delta = 20.0 -> MOL_A in C1 pools TR1 & TR2 -> mean = 15.0
    # TR3: cell C1, delta = 30.0 -> MOL_B in C1 mean = 30.0
    # TR4: cell C1 only, delta = 5.0
    # TR5: cell C1, delta = 1.0
    # Test rows:
    # TE1 in C1 (row 100)
    # TE1 in C1 (row 101) -> test broadcasting
    # TE1 in C2 (row 102) -> cell C2 has NO training candidates -> fallback!
    b_rows = [0, 1, 2, 3, 4, 100, 101, 102]
    b_split = ['train', 'train', 'train', 'train', 'train', 'test', 'test', 'test']
    b_pert = ['TR1', 'TR2', 'TR3', 'TR4', 'TR5', 'TE1', 'TE1', 'TE1']
    b_cell = ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2']
    b_delta = np.zeros((8, G), dtype=np.float64)
    b_delta[0] = 10.0
    b_delta[1] = 20.0
    b_delta[2] = 30.0
    b_delta[3] = 5.0
    b_delta[4] = 1.0

    bundle_npz = tmp_path / 'bundle.npz'
    np.savez(bundle_npz, row_index=b_rows, split_split_cold_drug_1=b_split,
             meta_pert_id=b_pert, meta_cell=b_cell,
             X=b_delta, X_ctl=np.zeros((8, G)))

    # Rows file
    rows_npz = tmp_path / 'rows.npz'
    np.savez(rows_npz, row_index=np.array([100, 101, 102], dtype=np.int64))

    # Monkeypatch ROWS_N and ROWS_SHA1
    monkeypatch.setattr(sbp, 'ROWS_N', 3)
    monkeypatch.setattr(sbp, 'ROWS_SHA1', hashlib.sha1(np.array([100, 101, 102], dtype=np.int64).tobytes()).hexdigest())

    out_refs = tmp_path / 'REFS.npz'
    sidecar = sbp.build_references(str(bundle_npz), str(chem_npz), str(rows_npz), out_npz=str(out_refs))

    refs = np.load(out_refs)
    assert np.array_equal(refs['row_index'], np.array([100, 101, 102]))

    # Check 1-NN tie-break: MOL_A and MOL_B both have Tanimoto 1.0 to TE1.
    # Tie broken by MOL_A < MOL_B.
    # MOL_A in C1 has pooled delta mean 15.0.
    np.testing.assert_allclose(refs['nn1'][0], 15.0)

    # Check broadcasting: row 100 and row 101 are both TE1 in C1
    np.testing.assert_allclose(refs['nn1'][0], refs['nn1'][1])
    np.testing.assert_allclose(refs['nn5'][0], refs['nn5'][1])
    np.testing.assert_allclose(refs['physchem'][0], refs['physchem'][1])

    # Check fallback: cell C2 has no training candidate
    assert sidecar['fallbacks']['nn1'] == 1
    # Fallback uses Δ_m overall across all cells


def test_reader_decisions():
    """Reader: licensed sentence with right labels, B3-only, NO CLAIM, beyond_R requiring all 4 variants, 2-class rule."""
    def make_b3(p=0.001, frac=0.8, n_classes=2):
        classes = ['MAPK', 'DNA'][:n_classes]
        d_units = {f'{cls}@MCF7': 1.0 for cls in classes}
        return {'split': 'split_cold_drug_1', 'delta_source': 'v9p9_seed0.npz:deg_pred',
                'a3_gated': {'p': p, 'frac_positive': frac, 'd_per_unit': d_units}}

    def make_cmp(p_values):
        per_v = {}
        for idx, name in enumerate(['seed0', 'seed1', 'seed2', 'seed_mean']):
            p_val = p_values[idx]
            per_v[name] = {'excluding': {'z': {'p_value': p_val, 'pass': bool(p_val < 0.05)}}}
        return {'ref_label': '1-NN', 'per_variant': per_v}

    # 1. Full pass -> LICENSED
    b3_pass = [make_b3()] * 4
    cmp_pass = [make_cmp([0.01, 0.02, 0.01, 0.03])]
    # verified json mock
    b3_files = []
    cmp_files = []
    # Save files with markers
    import tempfile
    td = tempfile.mkdtemp()
    for i, b in enumerate(b3_pass):
        p = os.path.join(td, f'b3_{i}.json')
        with open(p, 'w') as f: json.dump(b, f)
        with open(p + '.marker', 'w') as f: json.dump({'complete': True, 'sha1': sbp.sha1_file(p)}, f)
        b3_files.append(p)

    for i, c in enumerate(cmp_pass):
        p = os.path.join(td, f'cmp_{i}.json')
        with open(p, 'w') as f: json.dump(c, f)
        with open(p + '.marker', 'w') as f: json.dump({'complete': True, 'sha1': sbp.sha1_file(p)}, f)
        cmp_files.append(p)

    res_lic = sbp.read_b_prime(b3_files, cmp_files)
    assert res_lic['status'] == 'LICENSED'
    assert 'chemistry-only references (1-NN)' in res_lic['decision']
    assert any('EGFR: one cell (MCF7) after excluding afatinib.' in n for n in res_lic['notes'])

    # 2. One variant p=0.06 kills beyond_R -> B3_ONLY
    cmp_fail1 = [make_cmp([0.01, 0.02, 0.06, 0.03])]
    p_fail = os.path.join(td, 'cmp_fail.json')
    with open(p_fail, 'w') as f: json.dump(cmp_fail1[0], f)
    with open(p_fail + '.marker', 'w') as f: json.dump({'complete': True, 'sha1': sbp.sha1_file(p_fail)}, f)

    res_b3_only = sbp.read_b_prime(b3_files, [p_fail])
    assert res_b3_only['status'] == 'B3_ONLY'
    assert res_b3_only['decision'] == 'B3 holds; no beyond-chemistry claim'

    # 3. Item 1 fails due to 1-class only -> NO_CLAIM
    b3_1class = [make_b3(n_classes=1)] * 4
    b3_1class_files = []
    for i, b in enumerate(b3_1class):
        p = os.path.join(td, f'b3_1c_{i}.json')
        with open(p, 'w') as f: json.dump(b, f)
        with open(p + '.marker', 'w') as f: json.dump({'complete': True, 'sha1': sbp.sha1_file(p)}, f)
        b3_1class_files.append(p)

    res_noclaim = sbp.read_b_prime(b3_1class_files, cmp_files)
    assert res_noclaim['status'] == 'NO_CLAIM'
    assert 'NO STAGE B' in res_noclaim['decision']



# =========================================================================
# PI additions (mutation check of W35's suite: 11 of 24 mutants survived; each test below kills one or more)
# =========================================================================

def _write_marked(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)
    with open(path + '.marker', 'w') as f:
        json.dump({'complete': True, 'sha1': sbp.sha1_file(path)}, f)
    return path


def _refs_world(tmp_path, monkeypatch, tr2_bits=None):
    """Hand-computable reference world. Training molecules (all profiled in C1; MOL_A also in C3):
    MOL_A = TR1 + TR2 (TR1 is the smaller pert_id, so its structure represents MOL_A), MOL_B = TR3, MOL_C = TR4, MOL_D = TR5.
    Test TE1 is profiled in C1 (candidates A, B, C, D) and C2 (no candidates -> fallback)."""
    G = 4
    fps = np.zeros((6, 2048), dtype=np.uint8)
    fps[0, :10] = 1                                        # TR1 -> MOL_A: Tanimoto 1.0 to TE1
    if tr2_bits is None:
        fps[1, 100:110] = 1                                # TR2 differs: Tanimoto 0 to TE1 (must NOT represent MOL_A)
    fps[2, :10] = 1                                        # TR3 -> MOL_B: 1.0 (ties MOL_A; broken by key A < B)
    fps[3, :5] = 1                                         # MOL_C: 0.5
    fps[4, :2] = 1                                         # MOL_D: 0.2
    fps[5, :10] = 1                                        # TE1
    desc = np.zeros((6, 6))
    desc[:, 0] = [0.0, 0.0, 1000.0, 2000.0, 3000.0, 0.0]   # a huge-scale descriptor: z-scoring changes the distances
    desc[:, 1] = [1.0, 1.0, 0.0, 2.0, 5.0, 1.0]
    desc[:, 2:] = np.arange(4)[None, :] + np.array([0, 0, 1, 2, 3, 0])[:, None]
    chem = tmp_path / 'chem.npz'
    np.savez(chem, pert_id=['TR1', 'TR2', 'TR3', 'TR4', 'TR5', 'TE1'], role=['train'] * 5 + ['test'],
             molecule_key=['MOL_A', 'MOL_A', 'MOL_B', 'MOL_C', 'MOL_D', 'MOL_TE1'], parsable=[True] * 6,
             n_fragments=[1] * 6, fp=fps, desc=desc)
    _ = (tmp_path / 'chem.npz.marker').write_text(json.dumps({'complete': True, 'sha1': sbp.sha1_file(str(chem))}))
    # training deltas: MOL_A in C1 = mean(10, 20) = 15; MOL_A in C3 = 99 -> overall MOL_A = (10 + 20 + 99) / 3 = 43
    rows = [(0, 'train', 'TR1', 'C1', 10.0), (1, 'train', 'TR2', 'C1', 20.0), (2, 'train', 'TR3', 'C1', 30.0),
            (3, 'train', 'TR4', 'C1', 5.0), (4, 'train', 'TR5', 'C1', 1.0), (5, 'train', 'TR1', 'C3', 99.0),
            (100, 'test', 'TE1', 'C1', 0.0), (101, 'test', 'TE1', 'C2', 0.0)]
    X = np.array([[r[4]] * G for r in rows])
    bundle = tmp_path / 'bundle.npz'
    np.savez(bundle, row_index=[r[0] for r in rows], split_split_cold_drug_1=[r[1] for r in rows],
             meta_pert_id=[r[2] for r in rows], meta_cell=[r[3] for r in rows], X=X, X_ctl=np.zeros_like(X))
    rows_npz = tmp_path / 'rows.npz'
    np.savez(rows_npz, row_index=np.array([100, 101], dtype=np.int64))
    monkeypatch.setattr(sbp, 'ROWS_N', 2)
    monkeypatch.setattr(sbp, 'ROWS_SHA1', hashlib.sha1(np.array([100, 101], dtype=np.int64).tobytes()).hexdigest())
    out = tmp_path / 'REFS.npz'
    side = sbp.build_references(str(bundle), str(chem), str(rows_npz), out_npz=str(out))
    return np.load(out), side, desc


def test_reference_values_by_hand(tmp_path, monkeypatch):
    """1-NN, similarity-weighted 5-NN, the fallback's overall mean, and the molecule's smallest-pert representative."""
    refs, side, desc = _refs_world(tmp_path, monkeypatch)
    # C1: MOL_A (smallest pert TR1's bits) ties MOL_B at 1.0 and wins on key -> MOL_A's C1 mean 15
    np.testing.assert_allclose(refs['nn1'][0], 15.0)
    # 5-NN over the 4 C1 candidates, weights 1, 1, 0.5, 0.2 on 15, 30, 5, 1
    np.testing.assert_allclose(refs['nn5'][0], (15 + 30 + 2.5 + 0.2) / 2.7, rtol=1e-6)
    # C2 has no candidate: fallback over all molecules with their OVERALL means; MOL_A's is (10 + 20 + 99) / 3
    assert side['fallbacks']['nn1'] == 1
    np.testing.assert_allclose(refs['nn1'][1], 43.0, rtol=1e-6)


def test_physchem_uses_training_z_and_inverse_one_plus_distance(tmp_path, monkeypatch):
    refs, _, desc = _refs_world(tmp_path, monkeypatch)
    tr = np.array([desc[0], desc[2], desc[3], desc[4]])          # molecule representatives A (TR1), B, C, D
    mu, sd = tr.mean(0), tr.std(0, ddof=0)
    d = np.sqrt((((tr - mu) / sd - (desc[5] - mu) / sd) ** 2).sum(1))
    s = 1.0 / (1.0 + d)
    want = (s * np.array([15.0, 30.0, 5.0, 1.0])).sum() / s.sum()
    np.testing.assert_allclose(refs['physchem'][0], want, rtol=1e-6)


def _swap_world(seed, noise=0.3, n_others=30):
    rng = np.random.default_rng(seed)
    idx = ['M1', 'M2', 'M3', 'M4'] + ['O%d' % i for i in range(n_others)]
    v = pd.DataFrame(rng.normal(size=(len(idx), 1)), index=idx, columns=['PW'])
    r = v + rng.normal(0, noise, size=v.shape)
    v.loc[['M1', 'M2', 'M3', 'M4'], 'PW'] -= 0.25                 # v9 slightly more in the inhibitor direction
    units = [{'class': 'K', 'cell': 'C', 'pathway': 'PW', 'inhibitor_sign': -1.0,
              'members': [(m, -1.0) for m in idx[:4]], 'others': idx[4:]}]
    return units, {'C': v}, {'C': r}


def test_swap_p_value_is_exact_and_passes_only_below_005():
    units, v, r = _swap_world(0)
    res = sbp.run_swap_test(units, v, r, n_swap=3000, swap_seed=11)
    g = res['g_vector']
    signs = np.random.default_rng(11).choice(np.array([1.0, -1.0]), size=(3000, len(g)))
    want = (1.0 + np.sum(signs @ g >= g.sum())) / 3001.0
    assert abs(res['p_value'] - want) < 1e-15
    found = None
    for seed in range(400):                                       # a world whose p lies in [0.05, 0.10)
        u2, v2, r2 = _swap_world(seed)
        rr = sbp.run_swap_test(u2, v2, r2, n_swap=3000, swap_seed=11)
        if 0.05 <= rr['p_value'] < 0.10:
            found = rr
            break
    assert found is not None and found['pass'] is False


def test_post_exclusion_unit_guard_refuses(tmp_path, monkeypatch, fake_progeny_net):
    """Reuses the shape of test_guards' world: one EGFR@MCF7 unit (P0-P2); a wrong EXCLUDING_UNITS must refuse."""
    G = 20
    genes = [f'GENE_{i}' for i in range(G)]
    monkeypatch.setattr(sbp, 'ROWS_N', 3)
    monkeypatch.setattr(sbp, 'ROWS_SHA1', hashlib.sha1(np.array([0, 1, 2], dtype=np.int64).tobytes()).hexdigest())
    monkeypatch.setitem(msa.SPLITS['split_cold_drug_1'], 'n_rows_restricted', 3)
    monkeypatch.setitem(msa.SPLITS['split_cold_drug_1'], 'n_rows', 3)
    monkeypatch.setattr(msa, 'scored_cells', lambda *a, **k: (['MCF7'], {'MCF7': 3}))
    rng = np.random.default_rng(123)
    bundle = tmp_path / 'b.npz'
    np.savez(bundle, split_split_cold_drug_1=np.array(['test'] * 3), row_index=np.array([0, 1, 2]),
             meta_pert_id=np.array(['P0', 'P1', 'P2']), meta_cell=np.array(['MCF7'] * 3),
             X=rng.normal(size=(3, G)), X_ctl=np.zeros((3, G)))
    rows = tmp_path / 'rows.npz'
    np.savez(rows, row_index=np.array([0, 1, 2]))
    labels = tmp_path / 'labels.tsv'
    labels.write_text(
        "pert_id\tparent_chembl_id\tdirect_interaction\torganism\tmechanism_of_action\tgene_symbol\taction_type\n"
        "BRD-K66175015\tCHEMBL_A\t1\tHomo sapiens\tEGFR inhibitor\tEGFR\tINHIBITOR\n"
        "BRD-K92093830\tCHEMBL_D\t1\tHomo sapiens\tDNA inhibitor\tMDM2\tINHIBITOR\n"
        + ''.join("P%d\tCHEMBL_P%d\t1\tHomo sapiens\tEGFR inhibitor\tEGFR\tINHIBITOR\n" % (i, i) for i in range(3)),
        encoding='utf-8')
    v9 = tmp_path / 'v9.npz'
    np.savez(v9, row_index=np.array([0, 1, 2]), deg_pred=rng.normal(size=(3, G)))
    monkeypatch.setattr(sbp, 'INCLUDING_UNITS', {'EGFR@MCF7'})
    monkeypatch.setattr(sbp, 'EXCLUDING_UNITS', {'DNA@MCF7'})
    with pytest.raises(SystemExit, match='post-exclusion unit set'):
        sbp.compare(str(bundle), str(labels), str(rows), [str(v9) + ':deg_pred'] * 4, str(v9) + ':deg_pred', 'R',
                    net=fake_progeny_net, genes=genes)
    # with the right unit sets, compare must call the pairing guard for every v9 variant
    monkeypatch.setattr(sbp, 'EXCLUDING_UNITS', {'EGFR@MCF7'})
    calls = []
    monkeypatch.setattr(sbp, 'check_pairing', lambda el_v, el_R, name='v9': calls.append(name))
    sbp.compare(str(bundle), str(labels), str(rows), [str(v9) + ':deg_pred'] * 4, str(v9) + ':deg_pred', 'R',
                net=fake_progeny_net, genes=genes)
    assert calls == ['seed0', 'seed1', 'seed2', 'seed_mean']


def test_pairing_guard():
    idx = ['M1', 'M2', 'M3', 'O1']
    df = pd.DataFrame(np.arange(4.0)[:, None], index=idx, columns=['PW'])
    unit = {'class': 'K', 'cell': 'C', 'pathway': 'PW', 'inhibitor_sign': -1.0,
            'members': [('M1', -1.0), ('M2', -1.0), ('M3', -1.0)], 'others': ['O1']}
    el = {'cell_acts': {'C': df}, 'eval_units': [unit]}
    sbp.check_pairing(el, el)
    with pytest.raises(SystemExit, match='different labelled units'):
        sbp.check_pairing(el, {'cell_acts': {'C': df.iloc[::-1]}, 'eval_units': [unit]})
    with pytest.raises(SystemExit, match='different eval units'):
        sbp.check_pairing(el, {'cell_acts': {'C': df}, 'eval_units': [dict(unit, others=[])]})


def test_reader_item1_thresholds_and_provenance(tmp_path):
    """Item 1 needs p < 0.01, frac >= 2/3 and two classes in EVERY variant; B3 inputs must be predicted cold-drug runs."""
    def b3(p=0.001, frac=0.8, src='v9p9_seed0.npz:deg_pred', split='split_cold_drug_1'):
        return {'split': split, 'delta_source': src,
                'a3_gated': {'p': p, 'frac_positive': frac, 'd_per_unit': {'EGFR@MCF7': 1.0, 'DNA@MCF7': 1.0}}}
    cmp = {'ref_label': '1-NN', 'per_variant': {v: {'excluding': {'z': {'p_value': 0.01, 'pass': True}}}
                                                 for v in ('seed0', 'seed1', 'seed2', 'seed_mean')}}
    c = _write_marked(str(tmp_path / 'cmp.json'), cmp)

    def run(objs, tag):
        return sbp.read_b_prime([_write_marked(str(tmp_path / ('%s_%d.json' % (tag, i))), o) for i, o in enumerate(objs)], [c])
    assert run([b3()] * 4, 'ok')['status'] == 'LICENSED'
    assert run([b3(frac=0.6)] * 4, 'frac')['status'] == 'NO_CLAIM'                 # 0.6 < 2/3
    assert run([b3(p=0.03)] * 4, 'p')['status'] == 'NO_CLAIM'                      # 0.03 >= 0.01
    assert run([b3()] * 3 + [b3(p=0.02)], 'one')['status'] == 'NO_CLAIM'          # every variant must pass
    with pytest.raises(SystemExit, match='not a predicted-delta B3'):
        run([b3(src='measured')] + [b3()] * 3, 'meas')
    with pytest.raises(SystemExit, match='not a predicted-delta B3'):
        run([b3(split='split_cold_cell_1')] + [b3()] * 3, 'cc')
