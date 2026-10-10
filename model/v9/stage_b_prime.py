# -*- coding: utf-8 -*-
"""RESULTS §96.4, §96.8, §96.9: Stage B' cold-drug chemistry-reference evaluation.

Compares v9 predictions on cold-drug test compounds against chemistry-only
references (1-NN, 5-NN, physchem, ridge) using paired sign-flip swap tests:
- Reference construction: ECFP4 Tanimoto 1-NN, 5-NN, physchem descriptors
- Elements extraction mirroring mechanism_stage_a.run_a3
- Exclusion of afatinib and doxorubicin (BRD-K66175015, BRD-K92093830)
- Paired swap null as sign-flip test on standardized pathway activities
- Reader for Stage B' decisions

Runs under .venv-cuda. Does NOT import RDKit.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanism_stage_a as msa  # noqa: E402

ROWS_N = 13364
ROWS_SHA1 = '5f85ef0b5bec3b82bd1b0823c2f4d507f2e094a9'     # sha1(np.sort(row_index.astype(np.int64)).tobytes())
SPLIT = 'split_cold_drug_1'
EXCLUDED_PERTS = ('BRD-K66175015', 'BRD-K92093830')          # afatinib, doxorubicin (96.6 item 2, 96.9 item 6)
INCLUDING_UNITS = {'EGFR@MCF7', 'EGFR@PC3', 'EGFR@A375', 'EGFR@HA1E', 'EGFR@HT29', 'EGFR@A549', 'DNA@MCF7', 'DNA@A375', 'DNA@A549'}
EXCLUDING_UNITS = {'DNA@MCF7', 'DNA@A375', 'DNA@A549', 'EGFR@MCF7'}
N_SWAP = 10000
SWAP_SEED = 9470
SWAP_P_MAX = 0.05

B_PRIME_MEMBER_PERTS = {
    'BRD-K66175015',  # afatinib
    'BRD-K92093830',  # doxorubicin
    'BRD-K08799216',  # pelitinib
    'BRD-K70401845',  # erlotinib
    'BRD-K76908866',  # CP-724714
    'BRD-K79254416',  # decitabine
    'BRD-K93034159',  # cladribine
    'BRD-K67043667',  # altretamine
}


def sha1_file(path):
    """Compute sha1 hash of file contents."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _verified_json(path):
    """Load JSON file verifying its marker exists, is complete, and sha1 matches."""
    marker_path = path + '.marker'
    if not os.path.exists(marker_path):
        raise SystemExit(f'REFUSED: marker missing for {path}')
    with open(marker_path, 'r', encoding='utf-8') as f:
        mk = json.load(f)
    if not mk.get('complete') or mk.get('sha1') != sha1_file(path):
        raise SystemExit(f'REFUSED: {path} does not match its marker')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_references(bundle_path, chem_npz_path, rows_npz_path, out_npz=None, labels_path=None, member_perts=None):
    """Construct chemistry-only references (1-NN, 5-NN, physchem) for cold-drug test rows.

    Parameters:
    - bundle_path: path to xpert_mdmt_splits.npz
    - chem_npz_path: path to CHEM.npz written by stage_b_prime_chem.py
    - rows_npz_path: path to P9 seed-0 rows npz (only 'row_index' key read)
    - out_npz: path to output REFS.npz
    - labels_path: optional path to chembl_dti_edges.tsv
    - member_perts: optional iterable of member pert_ids for sidecar reporting
    """
    # 1. Verify chem npz against its marker
    chem_marker_path = chem_npz_path + '.marker'
    if not os.path.exists(chem_marker_path):
        raise SystemExit(f'REFUSED: chem npz marker missing for {chem_npz_path}')
    with open(chem_marker_path, 'r', encoding='utf-8') as f:
        mk = json.load(f)
    if not mk.get('complete') or mk.get('sha1') != sha1_file(chem_npz_path):
        raise SystemExit(f'REFUSED: {chem_npz_path} does not match its marker')

    # 2. Read only 'row_index' from rows_npz and verify count and sha1
    z_rows = np.load(rows_npz_path)
    if 'row_index' not in z_rows.files:
        raise SystemExit(f"REFUSED: 'row_index' key missing in {rows_npz_path}")
    row_index = np.asarray(z_rows['row_index']).astype(np.int64)

    if len(row_index) != ROWS_N:
        raise SystemExit(f"REFUSED: row count mismatch: expected {ROWS_N}, got {len(row_index)}")
    row_sha1 = hashlib.sha1(np.sort(row_index).tobytes()).hexdigest()
    if row_sha1 != ROWS_SHA1:
        raise SystemExit(f"REFUSED: row_index sha1 mismatch: expected {ROWS_SHA1}, got {row_sha1}")

    # 3. Load bundle and map test rows
    bundle = np.load(bundle_path, allow_pickle=True)
    split_key = f'split_{SPLIT}' if f'split_{SPLIT}' in bundle else SPLIT
    bundle_split = np.asarray(bundle[split_key]).astype(str)
    bundle_row_index = np.asarray(bundle['row_index']).astype(np.int64)
    bundle_pert = np.asarray(bundle['meta_pert_id']).astype(str)
    bundle_cell = np.asarray(bundle['meta_cell']).astype(str)

    b_idx_map = {int(r): i for i, r in enumerate(bundle_row_index)}
    test_bundle_indices = []
    for r in row_index:
        idx = b_idx_map.get(int(r))
        if idx is None:
            raise SystemExit(f"REFUSED: row_index {r} not found in bundle")
        if bundle_split[idx] != 'test':
            raise SystemExit(f"REFUSED: row_index {r} is not 'test' in bundle ({bundle_split[idx]})")
        test_bundle_indices.append(idx)
    test_bundle_indices = np.array(test_bundle_indices, dtype=np.int64)

    test_rows_pert = bundle_pert[test_bundle_indices]
    test_rows_cell = bundle_cell[test_bundle_indices]

    # 4. Load CHEM.npz
    chem_data = np.load(chem_npz_path, allow_pickle=True)
    c_pert = np.asarray(chem_data['pert_id']).astype(str)
    c_role = np.asarray(chem_data['role']).astype(str)
    c_mkey = np.asarray(chem_data['molecule_key']).astype(str)
    c_pars = np.asarray(chem_data['parsable']).astype(bool)
    c_fp = np.asarray(chem_data['fp']).astype(np.uint8)
    c_desc = np.asarray(chem_data['desc']).astype(np.float64)

    chem_map = {}
    for i, pid in enumerate(c_pert):
        chem_map[pid] = {
            'role': c_role[i],
            'molecule_key': c_mkey[i],
            'parsable': c_pars[i],
            'fp': c_fp[i],
            'desc': c_desc[i],
        }

    # Verify every test row's pert is parsable
    unique_test_perts = sorted(set(test_rows_pert))
    unparsable_test = [pid for pid in unique_test_perts if (pid not in chem_map or not chem_map[pid]['parsable'])]
    if unparsable_test:
        raise SystemExit(f"REFUSED: test pert_ids are unparsable or missing in CHEM.npz: {unparsable_test}")

    # 5. Group training compounds by molecule_key (96.9 item 2)
    train_perts = sorted(set(c_pert[c_role == 'train']))
    train_mols_to_perts = {}
    for pid in train_perts:
        mkey = chem_map[pid]['molecule_key']
        train_mols_to_perts.setdefault(mkey, []).append(pid)

    usable_mols = {}
    for mkey, plist in train_mols_to_perts.items():
        parsable_plist = [p for p in plist if chem_map[p]['parsable']]
        if len(parsable_plist) >= 1:
            smallest_p = min(parsable_plist)
            usable_mols[mkey] = {
                'perts': plist,
                'representative_pert': smallest_p,
                'fp': chem_map[smallest_p]['fp'],
                'desc': chem_map[smallest_p]['desc'],
            }

    n_usable_training_mols = len(usable_mols)
    n_multi_pert_training_mols = sum(1 for mkey, data in usable_mols.items() if len(data['perts']) > 1)

    # 6. Extract bundle training responses and pool by molecule
    train_mask = (bundle_split == 'train')
    train_indices = np.flatnonzero(train_mask)
    train_b_perts = bundle_pert[train_indices]
    train_b_cells = bundle_cell[train_indices]

    pert_to_mkey = {p: chem_map[p]['molecule_key'] for p in train_perts if p in chem_map}

    train_X = bundle['X'][train_indices].astype(np.float64)
    train_X_ctl = bundle['X_ctl'][train_indices].astype(np.float64)
    train_delta = train_X - train_X_ctl

    mol_overall_sums = {}
    mol_overall_counts = {}
    mol_cell_sums = {}
    mol_cell_counts = {}

    for idx, (p, c) in enumerate(zip(train_b_perts, train_b_cells)):
        mkey = pert_to_mkey.get(p)
        if mkey is None or mkey not in usable_mols:
            continue
        d = train_delta[idx]

        mol_overall_sums[mkey] = mol_overall_sums.get(mkey, 0.0) + d
        mol_overall_counts[mkey] = mol_overall_counts.get(mkey, 0) + 1

        cell_dict_s = mol_cell_sums.setdefault(c, {})
        cell_dict_c = mol_cell_counts.setdefault(c, {})
        cell_dict_s[mkey] = cell_dict_s.get(mkey, 0.0) + d
        cell_dict_c[mkey] = cell_dict_c.get(mkey, 0) + 1

    mol_delta_mean = {m: (mol_overall_sums[m] / mol_overall_counts[m]) for m in mol_overall_sums}
    mol_cell_delta_mean = {}
    for c, c_dict in mol_cell_sums.items():
        mol_cell_delta_mean[c] = {m: (c_dict[m] / mol_cell_counts[c][m]) for m in c_dict}

    # 7. Usable training molecules arrays
    usable_mkeys = sorted(usable_mols.keys())
    train_fps = np.stack([usable_mols[m]['fp'] for m in usable_mkeys], axis=0)  # (M, 2048) uint8
    train_descs = np.stack([usable_mols[m]['desc'] for m in usable_mkeys], axis=0)  # (M, 6) float64
    train_delta_means = np.stack([mol_delta_mean[m] for m in usable_mkeys], axis=0)  # (M, 978) float64

    mkey_to_train_idx = {m: i for i, m in enumerate(usable_mkeys)}

    # Physchem stats
    desc_mean = np.mean(train_descs, axis=0)
    desc_sd = np.std(train_descs, axis=0, ddof=0)
    if np.any(np.isnan(desc_sd)) or np.any(desc_sd == 0):
        raise SystemExit("REFUSED: descriptor standard deviation is 0 or NaN")

    train_descs_z = (train_descs - desc_mean) / desc_sd

    # Candidates per cell
    cell_candidates = {}
    unique_cells_in_test = sorted(set(test_rows_cell))
    for c in unique_cells_in_test:
        c_mols = sorted(mol_cell_delta_mean.get(c, {}).keys())
        cell_candidates[c] = c_mols

    # 8. Unique (test_pert, cell) pairs
    unique_pairs = []
    pair_map = {}
    for p, c in zip(test_rows_pert, test_rows_cell):
        pair = (p, c)
        if pair not in pair_map:
            pair_map[pair] = len(unique_pairs)
            unique_pairs.append(pair)

    fallbacks = {'nn1': 0, 'nn5': 0, 'physchem': 0}
    pert_1nn_per_cell = {}
    max_tanimoto_per_pert = {}

    pair_nn1 = []
    pair_nn5 = []
    pair_physchem = []

    train_fps_f = train_fps.astype(np.float32)
    train_fps_sum = train_fps.sum(axis=1).astype(np.float32)

    for p, c in unique_pairs:
        fp_k = chem_map[p]['fp'].astype(np.float32)
        desc_k = chem_map[p]['desc']
        desc_k_z = (desc_k - desc_mean) / desc_sd

        # Max Tanimoto across all usable training molecules (pooled)
        inter_all = np.dot(train_fps_f, fp_k)
        denom_all = train_fps_sum + float(np.sum(fp_k)) - inter_all
        t_all = np.where(denom_all > 0, inter_all / denom_all, 0.0)
        max_t_pooled = float(np.max(t_all))
        if p not in max_tanimoto_per_pert or max_t_pooled > max_tanimoto_per_pert[p]:
            max_tanimoto_per_pert[p] = max_t_pooled

        cands = cell_candidates.get(c, [])
        is_fallback = (len(cands) == 0)

        if is_fallback:
            fallbacks['nn1'] += 1
            fallbacks['nn5'] += 1
            fallbacks['physchem'] += 1
            c_mkeys = usable_mkeys
            c_indices = np.arange(len(usable_mkeys))
            c_deltas = train_delta_means
            c_fps = train_fps_f
            c_fps_sum = train_fps_sum
            c_descs_z = train_descs_z
        else:
            c_mkeys = cands
            c_indices = np.array([mkey_to_train_idx[m] for m in cands], dtype=np.int64)
            c_deltas = np.stack([mol_cell_delta_mean[c][m] for m in cands], axis=0)
            c_fps = train_fps_f[c_indices]
            c_fps_sum = train_fps_sum[c_indices]
            c_descs_z = train_descs_z[c_indices]

        # Tanimoto similarity to candidates
        inter = np.dot(c_fps, fp_k)
        denom = c_fps_sum + float(np.sum(fp_k)) - inter
        s_tan = np.where(denom > 0, inter / denom, 0.0)

        # Stable sort (-s_tan, molecule_key)
        c_keys_arr = np.array(c_mkeys, dtype=object)
        order_tan = np.lexsort((c_keys_arr, -s_tan))

        # 1-NN
        best_tan_idx = order_tan[0]
        pred_1nn = c_deltas[best_tan_idx]
        pair_nn1.append(pred_1nn)

        pert_1nn_per_cell.setdefault(p, {})[c] = {
            'molecule_key': str(c_mkeys[best_tan_idx]),
            'tanimoto': float(s_tan[best_tan_idx]),
            'fallback': is_fallback,
        }

        # 5-NN
        k_count = min(5, len(c_mkeys))
        top_tan_idx = order_tan[:k_count]
        w_tan = s_tan[top_tan_idx]
        w_tan_sum = float(np.sum(w_tan))
        if w_tan_sum > 0:
            pred_5nn = np.sum(w_tan[:, None] * c_deltas[top_tan_idx], axis=0) / w_tan_sum
        else:
            pred_5nn = np.mean(c_deltas[top_tan_idx], axis=0)
        pair_nn5.append(pred_5nn)

        # physchem Euclidean distance
        diff_pc = c_descs_z - desc_k_z
        d_pc = np.sqrt(np.sum(diff_pc ** 2, axis=1))
        s_pc = 1.0 / (1.0 + d_pc)

        order_pc = np.lexsort((c_keys_arr, -s_pc))
        top_pc_idx = order_pc[:k_count]
        w_pc = s_pc[top_pc_idx]
        w_pc_sum = float(np.sum(w_pc))
        if w_pc_sum > 0:
            pred_pc = np.sum(w_pc[:, None] * c_deltas[top_pc_idx], axis=0) / w_pc_sum
        else:
            pred_pc = np.mean(c_deltas[top_pc_idx], axis=0)
        pair_physchem.append(pred_pc)

    pair_nn1 = np.stack(pair_nn1, axis=0)
    pair_nn5 = np.stack(pair_nn5, axis=0)
    pair_physchem = np.stack(pair_physchem, axis=0)

    # 9. Broadcast to all rows
    row_pair_indices = np.array([pair_map[(p, c)] for p, c in zip(test_rows_pert, test_rows_cell)], dtype=np.int64)
    rows_nn1 = pair_nn1[row_pair_indices].astype(np.float32)
    rows_nn5 = pair_nn5[row_pair_indices].astype(np.float32)
    rows_physchem = pair_physchem[row_pair_indices].astype(np.float32)

    # 10. Write output REFS.npz
    out_npz = out_npz or 'REFS.npz'
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or '.', exist_ok=True)
    np.savez(
        out_npz,
        row_index=row_index,
        nn1=rows_nn1,
        nn5=rows_nn5,
        physchem=rows_physchem,
    )

    out_sha = sha1_file(out_npz)
    marker = {'complete': True, 'file': os.path.basename(out_npz), 'sha1': out_sha}
    with open(out_npz + '.marker', 'w', encoding='utf-8') as f:
        json.dump(marker, f, indent=2)

    # Sidecar JSON
    target_members = set(member_perts) if member_perts is not None else (
        (set(unique_test_perts) & B_PRIME_MEMBER_PERTS) if (set(unique_test_perts) & B_PRIME_MEMBER_PERTS)
        else set(unique_test_perts)
    )

    b_prime_1nn = {pid: pert_1nn_per_cell[pid] for pid in target_members if pid in pert_1nn_per_cell}

    sidecar = {
        'counts': {
            'training_molecules': n_usable_training_mols,
            'multi_pert_molecules': n_multi_pert_training_mols,
            'fallbacks': fallbacks,
        },
        'fallbacks': fallbacks,
        'b_prime_members_1nn': b_prime_1nn,
        'max_tanimoto_per_test_pert': max_tanimoto_per_pert,
    }
    with open(out_npz + '.json', 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2)

    return sidecar


def elements(cell_units_map, scored_cells_list, parent_to_targets, net, gene_names=None, tmin=None, gated=True):
    """Mirror run_a3 line by line, returning raw activities, standardized z, and unit evaluation structures."""
    import decoupler as dc

    landmarks = gene_names
    class_table = (msa.GATED_CLASSES + list(msa.ACTIVE.get('extra_classes', []))) if gated else msa.UNGATED_CLASSES
    unit_moas = {u['unit_id']: u.get('moas', set()) for cu in cell_units_map.values() for u in cu.values()}

    if tmin is None:
        source_counts = net['source'].value_counts()
        min_targets = int(source_counts.min()) if len(source_counts) > 0 else 1
        tmin = min(15, min_targets)

    cell_acts = {}
    cell_acts_std = {}
    cell_labelled_list = {}

    for cell in scored_cells_list:
        units = cell_units_map.get(cell, {})
        labelled = [u for u in units.values() if u['is_labelled']]
        if len(labelled) == 0:
            continue

        u_ids = [u['unit_id'] for u in labelled]
        sigs = np.stack([u['sig'] for u in labelled], axis=0)
        cols = gene_names if (gene_names is not None and len(gene_names) == sigs.shape[1]) else [f'g{i}' for i in range(sigs.shape[1])]

        mat = pd.DataFrame(sigs, index=u_ids, columns=cols)
        acts, _ = dc.mt.ulm(mat, net, tmin=tmin)
        cell_acts[cell] = acts
        cell_labelled_list[cell] = u_ids

        sd = acts.std(axis=0, ddof=1).replace(0, np.nan)
        cell_acts_std[cell] = (acts - acts.mean(axis=0)) / sd

    eval_units = []
    for cell in scored_cells_list:
        if cell not in cell_acts:
            continue
        acts = cell_acts[cell]
        u_ids = cell_labelled_list[cell]

        for cls_def in class_table:
            cls_name = cls_def['class']
            pathway = cls_def['pathway']
            if pathway not in acts.columns:
                continue

            if 'moa_match' in cls_def:
                if cell not in msa.ACTIVE['gates'].get(cls_def['gate'], set()):
                    continue
                members = [(uid, cls_def['inhibitor_sign']) for uid in u_ids if cls_def['moa_match'] in unit_moas.get(uid, set())]
                if len(members) >= 3:
                    m_set = {uid for uid, _ in members}
                    others = [uid for uid in u_ids if uid not in m_set]
                    eval_units.append({
                        'class': cls_name,
                        'cell': cell,
                        'pathway': pathway,
                        'inhibitor_sign': cls_def['inhibitor_sign'],
                        'members': members,
                        'others': others,
                        'all_units': u_ids,
                    })
                continue

            target_set = cls_def['targets_fn'](cell)
            if not target_set:
                continue

            inh_sign = cls_def['inhibitor_sign']
            members = []
            for uid in u_ids:
                targets = parent_to_targets.get(uid, [])
                signs = set()
                for gene, action in targets:
                    if gene in target_set:
                        act_upper = action.upper()
                        if act_upper in msa.INHIBITOR_ACTIONS:
                            signs.add(inh_sign)
                        elif act_upper in msa.AGONIST_ACTIONS:
                            signs.add(-inh_sign)
                if len(signs) == 1:
                    members.append((uid, signs.pop()))

            if len(members) >= 3:
                m_set = {uid for uid, _ in members}
                others = [uid for uid in u_ids if uid not in m_set]
                eval_units.append({
                    'class': cls_name,
                    'cell': cell,
                    'pathway': pathway,
                    'inhibitor_sign': inh_sign,
                    'members': members,
                    'others': others,
                    'all_units': u_ids,
                })

    d_per_unit = {}
    d_std_per_unit = {}
    for u in eval_units:
        unit_id = f"{u['class']}@{u['cell']}"
        acts = cell_acts[u['cell']]
        z = cell_acts_std[u['cell']]
        pathway = u['pathway']
        inh_sign = u['inhibitor_sign']

        # Raw d
        in_scores = [acts.loc[uid, pathway] * s for uid, s in u['members']]
        mu_in = np.mean(in_scores)
        out_scores = [acts.loc[uid, pathway] * inh_sign for uid in u['others']]
        mu_out = np.mean(out_scores) if len(u['others']) > 0 else 0.0
        d_per_unit[unit_id] = float(mu_in - mu_out)

        # Standardised d
        in_z = [z.loc[uid, pathway] * s for uid, s in u['members']]
        mu_in_z = np.mean(in_z)
        out_z = [z.loc[uid, pathway] * inh_sign for uid in u['others']]
        mu_out_z = np.mean(out_z) if len(u['others']) > 0 else 0.0
        d_std_per_unit[unit_id] = float(mu_in_z - mu_out_z)

    return {
        'd_per_unit': d_per_unit,
        'd_std_per_unit': d_std_per_unit,
        'eval_units': eval_units,
        'cell_acts': cell_acts,
        'cell_acts_std': cell_acts_std,
        'cell_labelled_list': cell_labelled_list,
    }


def run_swap_test(eval_units, cell_acts_v9, cell_acts_R, excluded_parents=None, n_swap=N_SWAP, swap_seed=SWAP_SEED):
    """Execute the exact sign-flip swap test between v9 and reference R.

    Parameters:
    - eval_units: list of unit dicts from elements()
    - cell_acts_v9: dict mapping cell -> DataFrame of activities (z or raw) for v9
    - cell_acts_R: dict mapping cell -> DataFrame of activities (z or raw) for R
    - excluded_parents: optional set of parent IDs to exclude from members and others
    """
    surviving_units = []
    for u in eval_units:
        if excluded_parents is not None:
            members = [(uid, s) for uid, s in u['members'] if uid not in excluded_parents]
            others = [uid for uid in u['others'] if uid not in excluded_parents]
        else:
            members = list(u['members'])
            others = list(u['others'])

        if len(members) >= 3:
            surviving_units.append({
                'unit_id': f"{u['class']}@{u['cell']}",
                'class': u['class'],
                'cell': u['cell'],
                'pathway': u['pathway'],
                'inhibitor_sign': u['inhibitor_sign'],
                'members': members,
                'others': others,
            })

    U = len(surviving_units)
    if U == 0:
        return {
            'delta_obs': 0.0,
            'p_value': 1.0,
            'pass': False,
            'T_v9': 0.0,
            'T_R': 0.0,
            'n_units': 0,
            'n_elements': 0,
            'd_v9': {},
            'd_R': {},
            'per_class': {},
            'per_class_same_direction': False,
        }

    unit_names = [u['unit_id'] for u in surviving_units]
    reading_cells = sorted(set(u['cell'] for u in surviving_units))

    # Indexed globally as (unit_id, cell)
    elements = []
    for cell in reading_cells:
        all_uids = list(cell_acts_v9[cell].index)
        for uid in all_uids:
            if excluded_parents is not None and uid in excluded_parents:
                continue
            elements.append((uid, cell))

    n_el = len(elements)
    el_map = {el: i for i, el in enumerate(elements)}

    g = np.zeros(n_el, dtype=np.float64)
    d_v9 = {}
    d_R = {}

    for u in surviving_units:
        cell = u['cell']
        pathway = u['pathway']
        inh_sign = u['inhibitor_sign']
        v9_df = cell_acts_v9[cell]
        R_df = cell_acts_R[cell]

        n_m = len(u['members'])
        n_o = len(u['others'])

        in_v9 = np.mean([v9_df.loc[uid, pathway] * s for uid, s in u['members']])
        out_v9 = np.mean([v9_df.loc[uid, pathway] * inh_sign for uid in u['others']]) if n_o > 0 else 0.0
        d_v9[u['unit_id']] = float(in_v9 - out_v9)

        in_R = np.mean([R_df.loc[uid, pathway] * s for uid, s in u['members']])
        out_R = np.mean([R_df.loc[uid, pathway] * inh_sign for uid in u['others']]) if n_o > 0 else 0.0
        d_R[u['unit_id']] = float(in_R - out_R)

        for uid, s in u['members']:
            idx = el_map[(uid, cell)]
            a_ui = s / n_m
            diff = v9_df.loc[uid, pathway] - R_df.loc[uid, pathway]
            g[idx] += (1.0 / U) * a_ui * diff

        for uid in u['others']:
            idx = el_map[(uid, cell)]
            a_ui = -inh_sign / n_o
            diff = v9_df.loc[uid, pathway] - R_df.loc[uid, pathway]
            g[idx] += (1.0 / U) * a_ui * diff

    if not np.all(np.isfinite(g)):
        raise SystemExit("REFUSED: non-finite values in contribution vector g")

    delta_obs = float(np.sum(g))

    rng = np.random.default_rng(swap_seed)
    signs = rng.choice(np.array([1.0, -1.0], dtype=np.float64), size=(n_swap, n_el))
    delta_star = signs @ g

    p_val = float((1.0 + np.sum(delta_star >= delta_obs)) / (1.0 + n_swap))
    passed = bool(p_val < SWAP_P_MAX)

    T_v9 = float(np.mean([d_v9[u] for u in unit_names]))
    T_R = float(np.mean([d_R[u] for u in unit_names]))

    classes = sorted(set(u.split('@')[0] for u in unit_names))
    per_class = {}
    class_signs = []
    for cls in classes:
        diffs = [d_v9[u] - d_R[u] for u in unit_names if u.startswith(cls + '@')]
        m_diff = float(np.mean(diffs))
        sgn = int(np.sign(m_diff))
        per_class[cls] = {'mean_diff': m_diff, 'sign': sgn}
        class_signs.append(sgn)

    same_dir = bool(len(set(class_signs)) == 1)

    return {
        'delta_obs': delta_obs,
        'p_value': p_val,
        'pass': passed,
        'T_v9': T_v9,
        'T_R': T_R,
        'n_units': U,
        'n_elements': n_el,
        'd_v9': d_v9,
        'd_R': d_R,
        'per_class': per_class,
        'per_class_same_direction': same_dir,
        'g_vector': g,
        'elements': elements,
        'surviving_units': surviving_units,
    }


def check_pairing(el_v, el_R, name='v9'):
    """PI guard: the swap pairs element (unit, cell) of v9 with the SAME element of R, so both models must have identical
    labelled units per cell (in order) and identical eval units (class, cell, pathway, members with signs, others)."""
    if sorted(el_v['cell_acts']) != sorted(el_R['cell_acts']):
        raise SystemExit('REFUSED: %s and R score different cells' % name)
    for c in el_v['cell_acts']:
        if list(el_v['cell_acts'][c].index) != list(el_R['cell_acts'][c].index):
            raise SystemExit('REFUSED: %s and R have different labelled units in %s' % (name, c))

    def key(u):
        return (u['class'], u['cell'], u['pathway'], u['inhibitor_sign'], tuple(u['members']), tuple(u['others']))
    if [key(u) for u in el_v['eval_units']] != [key(u) for u in el_R['eval_units']]:
        raise SystemExit('REFUSED: %s and R have different eval units' % name)


def compare(bundle_path, labels_path, rows_npz_path, v9_specs, ref_spec, ref_label, out_json=None, net=None, genes=None):
    """Execute complete Stage B' comparison across 4 v9 variants and one reference R."""
    msa.ACTIVE.clear()
    msa.ACTIVE.update(msa.SPLITS[SPLIT], name=SPLIT)

    # 1. Check rows_npz
    z_rows = np.load(rows_npz_path)
    row_index = np.asarray(z_rows['row_index']).astype(np.int64)
    if len(row_index) != ROWS_N:
        raise SystemExit(f"REFUSED: expected {ROWS_N} rows, got {len(row_index)}")
    row_sha = hashlib.sha1(np.sort(row_index).tobytes()).hexdigest()
    if row_sha != ROWS_SHA1:
        raise SystemExit(f"REFUSED: row_index sha1 mismatch: expected {ROWS_SHA1}, got {row_sha}")

    # 2. Labels and excluded parents mapping
    pert_to_parent, parent_to_moa, parent_to_targets = msa.load_labels(labels_path)
    for p in EXCLUDED_PERTS:
        if p not in pert_to_parent:
            raise SystemExit(f"REFUSED: excluded pert {p} not found in labels pert_to_parent")
    excluded_parents = {pert_to_parent[p] for p in EXCLUDED_PERTS}

    if net is None:
        net = msa.load_progeny_network()
    if genes is None:
        genes = msa.load_landmark_genes()

    if isinstance(v9_specs, str):
        v9_specs = [s.strip() for s in v9_specs.split() if s.strip()]
    if len(v9_specs) != 4:
        raise SystemExit(f"REFUSED: expected exactly 4 v9 specs, got {len(v9_specs)}")

    variant_names = ['seed0', 'seed1', 'seed2', 'seed_mean']
    all_specs = list(v9_specs) + [ref_spec]
    all_names = variant_names + ['R']

    model_elements = {}
    for name, spec in zip(all_names, all_specs):
        rows = msa.load_rows(bundle_path, rows_npz_path, spec, split=SPLIT)
        unique_cells = np.unique(rows['cell'])
        cell_units = {c: msa.units_for_cell(c, rows, pert_to_parent, parent_to_moa, parent_to_targets) for c in unique_cells}
        scored, _ = msa.scored_cells(cell_units)

        el_res = elements(cell_units, scored, parent_to_targets, net, gene_names=genes, gated=True)

        # Check non-finite
        for c, df in el_res['cell_acts'].items():
            if not np.all(np.isfinite(df.values)):
                raise SystemExit(f"REFUSED: non-finite activities found in model {name} cell {c}")
        for c, df in el_res['cell_acts_std'].items():
            if not np.all(np.isfinite(df.values)):
                raise SystemExit(f"REFUSED: non-finite standardized activities found in model {name} cell {c}")

        # Equivalence guard against run_a3
        a3_res = msa.run_a3(cell_units, scored, parent_to_targets, net=net, gene_names=genes, gated=True, n_perm=1)
        if set(el_res['d_per_unit'].keys()) != set(a3_res.d_per_unit.keys()):
            raise SystemExit(f"REFUSED: equivalence guard failed on unit keys for model {name}")
        for u, d_val in el_res['d_per_unit'].items():
            if abs(d_val - a3_res.d_per_unit[u]) > 1e-9:
                raise SystemExit(f"REFUSED: equivalence guard failed on raw d for {u}: {d_val} vs {a3_res.d_per_unit[u]}")
        for u, d_std_val in el_res['d_std_per_unit'].items():
            if abs(d_std_val - a3_res.d_std_per_unit[u]) > 1e-9:
                raise SystemExit(f"REFUSED: equivalence guard failed on d_std for {u}: {d_std_val} vs {a3_res.d_std_per_unit[u]}")

        # Unit-set guard
        if set(el_res['d_per_unit'].keys()) != INCLUDING_UNITS:
            raise SystemExit(f"REFUSED: unit set for {name} ({set(el_res['d_per_unit'].keys())}) != INCLUDING_UNITS")

        # Exclusion unit set guard
        excl_units = set()
        for u in el_res['eval_units']:
            m_excl = [uid for uid, _ in u['members'] if uid not in excluded_parents]
            if len(m_excl) >= 3:
                excl_units.add(f"{u['class']}@{u['cell']}")
        if excl_units != EXCLUDING_UNITS:
            raise SystemExit(f"REFUSED: post-exclusion unit set for {name} ({excl_units}) != EXCLUDING_UNITS")

        model_elements[name] = el_res

    # Paired sign-flip swap comparison per v9 variant against R
    el_R = model_elements['R']
    per_variant = {}

    for v_name in variant_names:
        el_v = model_elements[v_name]
        check_pairing(el_v, el_R, v_name)

        # Excluding z (of record)
        res_excl_z = run_swap_test(el_v['eval_units'], el_v['cell_acts_std'], el_R['cell_acts_std'], excluded_parents=excluded_parents)
        res_excl_z['label'] = 'of record'

        # Excluding raw (reported, not a reading)
        res_excl_raw = run_swap_test(el_v['eval_units'], el_v['cell_acts'], el_R['cell_acts'], excluded_parents=excluded_parents)
        res_excl_raw['label'] = 'reported, not a reading'

        # Including z (reported; the excluding reading is of record)
        res_incl_z = run_swap_test(el_v['eval_units'], el_v['cell_acts_std'], el_R['cell_acts_std'], excluded_parents=None)
        res_incl_z['label'] = 'reported; the excluding reading is of record'

        # Including raw (reported; the excluding reading is of record, reported, not a reading)
        res_incl_raw = run_swap_test(el_v['eval_units'], el_v['cell_acts'], el_R['cell_acts'], excluded_parents=None)
        res_incl_raw['label'] = 'reported; the excluding reading is of record, reported, not a reading'

        # Remove internal structures from JSON output
        for r in (res_excl_z, res_excl_raw, res_incl_z, res_incl_raw):
            r.pop('g_vector', None)
            r.pop('elements', None)
            r.pop('surviving_units', None)

        per_variant[v_name] = {
            'excluding': {'z': res_excl_z, 'raw': res_excl_raw},
            'including': {'z': res_incl_z, 'raw': res_incl_raw},
        }

    # Input sha1s
    input_sha1s = {
        'bundle': sha1_file(bundle_path),
        'labels': sha1_file(labels_path),
        'rows': sha1_file(rows_npz_path),
        'ref_spec': ref_spec,
    }
    for v_name, spec in zip(variant_names, v9_specs):
        paths = spec.rsplit(':', 1)[0].split(',')
        for p in paths:
            if os.path.exists(p):
                input_sha1s[os.path.basename(p)] = sha1_file(p)

    # Fallback counts if R is a REFS reference
    fallbacks = None
    ref_file = ref_spec.rsplit(':', 1)[0].split(',')[0]
    if os.path.exists(ref_file + '.json'):
        try:
            with open(ref_file + '.json', 'r', encoding='utf-8') as f:
                r_sidecar = json.load(f)
            fallbacks = r_sidecar.get('fallbacks', r_sidecar.get('counts', {}).get('fallbacks'))
        except Exception:
            pass

    out_data = {
        'ref_spec': ref_spec,
        'ref_label': ref_label,
        'input_sha1s': input_sha1s,
        'fallbacks': fallbacks,
        'per_variant': per_variant,
        'variants': per_variant,
    }

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or '.', exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)

        out_sha = sha1_file(out_json)
        marker = {'complete': True, 'file': os.path.basename(out_json), 'sha1': out_sha}
        with open(out_json + '.marker', 'w', encoding='utf-8') as f:
            json.dump(marker, f, indent=2)

    return out_data


def read_b_prime(b3_jsons, cmp_jsons):
    """Mechanical reading of Stage B' cold-drug benchmark."""
    # 1. Verification of all files
    b3_data = [_verified_json(p) for p in b3_jsons]
    cmp_data = [_verified_json(p) for p in cmp_jsons]

    if len(b3_data) != 4:
        raise SystemExit(f"REFUSED: expected 4 B3 JSONs, got {len(b3_data)}")
    for p, d in zip(b3_jsons, b3_data):     # PI guard: item 1 reads v9's PREDICTED delta on the cold-drug split only
        if d.get('split') != SPLIT or d.get('delta_source', 'measured') == 'measured':
            raise SystemExit(f"REFUSED: {p} is not a predicted-delta B3 on {SPLIT} "
                             f"(split={d.get('split')!r}, delta_source={d.get('delta_source')!r})")

    # 2. Item 1 per v9 variant
    item1_variants_pass = []
    variant_details = []
    for idx, d in enumerate(b3_data):
        a3_g = d.get('a3_gated', {})
        p_val = a3_g.get('p', 1.0)
        frac_pos = a3_g.get('frac_positive', 0.0)
        d_units = a3_g.get('d_per_unit', {})
        classes = sorted(set(u.split('@')[0] for u in d_units))
        passes = (p_val < 0.01) and (frac_pos >= (2.0 / 3.0)) and (len(classes) >= 2)
        item1_variants_pass.append(passes)
        variant_details.append({
            'index': idx,
            'p': p_val,
            'frac_positive': frac_pos,
            'classes': classes,
            'pass': passes,
        })

    item1_pass = bool(all(item1_variants_pass))

    # 3. Item 2 per R
    ref_results = []
    for d in cmp_data:
        r_label = d.get('ref_label', d.get('ref_spec', 'Unknown'))
        per_var = d.get('per_variant', d.get('variants', {}))
        var_pass = {}
        var_p = {}
        for v_name, v_data in per_var.items():
            excl_z = v_data.get('excluding', {}).get('z', {})
            var_pass[v_name] = bool(excl_z.get('pass', False))
            var_p[v_name] = excl_z.get('p_value', 1.0)

        beyond_R = bool(len(var_pass) == 4 and all(var_pass.values()))
        ref_results.append({
            'label': r_label,
            'beyond_R': beyond_R,
            'variant_p': var_p,
            'variant_pass': var_pass,
        })

    passed_refs = [r['label'] for r in ref_results if r['beyond_R']]

    # 4. Formulate decision
    notes = [
        "EGFR: one cell (MCF7) after excluding afatinib.",
        "A3's signal is carried by the DNA \u2192 p53 row (4 compounds \u00d7 3 TP53-wild-type cells).",
    ]

    if item1_pass and len(passed_refs) > 0:
        labels_str = ", ".join(passed_refs)
        statement = (
            f"for DNA-damaging and EGFR-inhibiting compounds unseen in training, "
            f"v9's predictions show the expected pathway direction more strongly than "
            f"chemistry-only references ({labels_str})"
        )
        status = 'LICENSED'
    elif item1_pass:
        statement = 'B3 holds; no beyond-chemistry claim'
        status = 'B3_ONLY'
    else:
        statement = 'NO STAGE B\u2032 CLAIM'
        status = 'NO_CLAIM'

    # 5. Build table
    table_lines = []
    table_lines.append("=" * 60)
    table_lines.append(f"Stage B' Reading Decision: {statement}")
    table_lines.append(f"Status: {status} (Item 1 pass: {item1_pass})")
    if status == 'LICENSED':
        for n in notes:
            table_lines.append(f"  * {n}")
    table_lines.append("=" * 60)
    table_lines.append("Per-Reference Comparison (Excluding z-block, p < 0.05 on all 4 variants):")
    table_lines.append(f"{'Reference':<18} {'seed0 p':<10} {'seed1 p':<10} {'seed2 p':<10} {'mean p':<10} {'beyond_R':<8}")
    table_lines.append("-" * 66)
    for r in ref_results:
        p_s0 = f"{r['variant_p'].get('seed0', 1.0):.4f}"
        p_s1 = f"{r['variant_p'].get('seed1', 1.0):.4f}"
        p_s2 = f"{r['variant_p'].get('seed2', 1.0):.4f}"
        p_sm = f"{r['variant_p'].get('seed_mean', 1.0):.4f}"
        table_lines.append(f"{r['label']:<18} {p_s0:<10} {p_s1:<10} {p_s2:<10} {p_sm:<10} {str(r['beyond_R']):<8}")
    table_lines.append("=" * 60)

    table_text = "\n".join(table_lines)
    print(table_text)

    return {
        'status': status,
        'decision': statement,
        'item1_pass': item1_pass,
        'passed_references': passed_refs,
        'ref_results': ref_results,
        'notes': notes,
        'table': table_text,
    }


def main():
    parser = argparse.ArgumentParser(description="RESULTS §96.4/96.8/96.9: Stage B' cold-drug chemistry-reference evaluation")
    parser.add_argument('--build_refs', action='store_true', help="Build chemistry-only references (1-NN, 5-NN, physchem)")
    parser.add_argument('--bundle', default=None, help="Path to bundle npz")
    parser.add_argument('--chem', default=None, help="Path to CHEM.npz")
    parser.add_argument('--rows', default=None, help="Path to rows npz")
    parser.add_argument('--labels', default=None, help="Path to labels TSV")
    parser.add_argument('--out', default=None, help="Path to output file")

    parser.add_argument('--compare', action='store_true', help="Execute comparison between v9 and reference")
    parser.add_argument('--v9_specs', nargs='+', default=None, help="Four v9 specs (seed0 seed1 seed2 seed_mean)")
    parser.add_argument('--ref_spec', default=None, help="Reference spec (e.g. REFS.npz:nn1)")
    parser.add_argument('--ref_label', default=None, help="Reference label (e.g. 1-NN)")

    parser.add_argument('--read', action='store_true', help="Execute mechanical reading of Stage B'")
    parser.add_argument('--b3', nargs='+', default=None, help="Four B3 result JSONs (seed0 seed1 seed2 seed_mean)")
    parser.add_argument('--cmp', nargs='+', default=None, help="Comparison JSONs (e.g. CMP_ridge CMP_nn1 CMP_nn5 CMP_physchem)")

    args = parser.parse_args()

    if args.build_refs:
        if not args.bundle or not args.chem or not args.rows or not args.out:
            parser.error("--build_refs requires --bundle, --chem, --rows, and --out")
        sidecar = build_references(args.bundle, args.chem, args.rows, out_npz=args.out, labels_path=args.labels)
        print(f"Wrote references to {args.out} (fallbacks: {sidecar['fallbacks']})")
        return

    if args.compare:
        if not args.bundle or not args.labels or not args.rows or not args.v9_specs or not args.ref_spec or not args.ref_label or not args.out:
            parser.error("--compare requires --bundle, --labels, --rows, --v9_specs, --ref_spec, --ref_label, and --out")
        out = compare(args.bundle, args.labels, args.rows, args.v9_specs, args.ref_spec, args.ref_label, out_json=args.out)
        print(f"Wrote comparison to {args.out}")
        return

    if args.read:
        if not args.b3 or not args.cmp:
            parser.error("--read requires --b3 (4 JSONs) and --cmp (>= 1 JSON)")
        read_b_prime(args.b3, args.cmp)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
