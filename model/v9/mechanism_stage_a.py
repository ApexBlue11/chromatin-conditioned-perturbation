# -*- coding: utf-8 -*-
"""RESULTS §94 Stage A: data-first mechanism ceilings.

Pre-registered analysis of measured test-cell responses in the XPert bundle
(split_split_cold_cell_1 == 'test') and ChEMBL DTI labels.

Functions:
- load_rows: load bundle test rows and compute delta = X - X_ctl
- load_labels: load curated human direct interaction labels
- units_for_cell: collapse rows to parent units and compute cell-centered signatures
- scored_cells: determine cells qualifying for mechanism analysis
- a1_cell: CMap-style MoA-mate retrieval AUROC with plate-rule exclusion and null
- self_retrieval_ceiling / active_subset_a1: C4 diagnostic metrics
- run_a3: PROGENy pathway activity analysis with gated/ungated tables
- read_stage_a: mechanical decision reader
- main: CLI runner
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANDMARKS_FILE_DEFAULT = os.path.join(REPO_ROOT, 'baseline', 'Network Data', 'pathway_landmark_genes.txt')
PROGENY_SHA1 = 'af40b7a5fe7a7c717d826c898991d232b68c63d6'
SCORED_CELLS_ORDER = ['MCF7', 'HT29', 'MDAMB231', 'HS578T', 'THP1']

INHIBITOR_ACTIONS = {'INHIBITOR', 'ANTAGONIST', 'NEGATIVE MODULATOR', 'BLOCKER'}
AGONIST_ACTIONS = {'AGONIST', 'POSITIVE MODULATOR'}

# §94.7 item 2: Gated class table
GATED_CLASSES = [
    {
        'class': 'MAPK',
        'pathway': 'MAPK',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: (
            {'MAP2K1', 'MAP2K2', 'BRAF', 'RAF1', 'ARAF', 'MAPK1', 'MAPK3'}
            if cell == 'HT29'
            else {'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3'}
        ),
    },
    {
        'class': 'EGFR',
        'pathway': 'EGFR',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'EGFR', 'ERBB2'},
    },
    {
        'class': 'PI3K',
        'pathway': 'PI3K',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'AKT1', 'AKT2', 'AKT3', 'MTOR'},
    },
    {
        'class': 'JAK',
        'pathway': 'JAK-STAT',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'JAK1', 'JAK2', 'JAK3', 'TYK2'},
    },
    {
        'class': 'p53',
        'pathway': 'p53',
        'inhibitor_sign': +1.0,  # MDM2 inhibitor activates p53
        'cells': ['MCF7'],  # TP53-wild-type only
        'targets_fn': lambda cell: {'MDM2'} if cell == 'MCF7' else set(),
    },
    {
        'class': 'ER',
        'pathway': 'Estrogen',
        'inhibitor_sign': -1.0,  # agonist +1.0
        'cells': ['MCF7'],  # ER-positive only
        'targets_fn': lambda cell: {'ESR1', 'ESR2'} if cell == 'MCF7' else set(),
    },
    {
        'class': 'NFkB',
        'pathway': 'NFkB',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'IKBKB', 'CHUK'},
    },
    {
        'class': 'Hypoxia',
        'pathway': 'Hypoxia',
        'inhibitor_sign': +1.0,  # EGLN inhibitors stabilise HIF
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'EGLN1', 'EGLN2', 'EGLN3'},
    },
]

# §94.4: Ungated class table (AR included, no cell genotype restrictions)
UNGATED_CLASSES = [
    {
        'class': 'MAPK',
        'pathway': 'MAPK',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'MAP2K1', 'MAP2K2', 'BRAF', 'RAF1', 'ARAF', 'MAPK1', 'MAPK3'},
    },
    {
        'class': 'EGFR',
        'pathway': 'EGFR',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'EGFR', 'ERBB2'},
    },
    {
        'class': 'PI3K',
        'pathway': 'PI3K',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'AKT1', 'AKT2', 'AKT3', 'MTOR'},
    },
    {
        'class': 'JAK',
        'pathway': 'JAK-STAT',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'JAK1', 'JAK2', 'JAK3', 'TYK2'},
    },
    {
        'class': 'AR',
        'pathway': 'Androgen',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'AR'},
    },
    {
        'class': 'ER',
        'pathway': 'Estrogen',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'ESR1', 'ESR2'},
    },
    {
        'class': 'p53',
        'pathway': 'p53',
        'inhibitor_sign': +1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'MDM2'},
    },
    {
        'class': 'NFkB',
        'pathway': 'NFkB',
        'inhibitor_sign': -1.0,
        'cells': SCORED_CELLS_ORDER,
        'targets_fn': lambda cell: {'IKBKB', 'CHUK'},
    },
]


class RowsDict(dict):
    """Dictionary holding bundle row data with attribute access and unpacking support."""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __iter__(self):
        # Allow unpacking as (X, X_ctl, pert, cell, row_index)
        return iter((self['X'], self['X_ctl'], self['pert'], self['cell'], self['row_index']))


class A1Result(dict):
    """Result of a1_cell supporting dict indexing, attribute access, and tuple unpacking."""

    def __init__(self, a1, null_mean, null_sd, p_value, n_scored, per_class_auroc, n_with_mate=0, **kwargs):
        super().__init__(
            a1=a1,
            null_mean=null_mean,
            null_sd=null_sd,
            p_value=p_value,
            n_scored=n_scored,
            per_class_auroc=per_class_auroc,
            n_with_mate=n_with_mate,
            **kwargs,
        )
        self.a1 = a1
        self.null_mean = null_mean
        self.null_sd = null_sd
        self.p_value = p_value
        self.n_scored = n_scored
        self.per_class_auroc = per_class_auroc
        self.n_with_mate = n_with_mate

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __iter__(self):
        return iter((self.a1, self.null_mean, self.null_sd, self.p_value, self.n_scored, self.per_class_auroc))


class A3Result(dict):
    """Result of A3 analysis supporting dict indexing, attribute access, and tuple unpacking."""

    def __init__(self, T, p, d_per_unit, frac_positive, **kwargs):
        super().__init__(
            T=T,
            p=p,
            d_per_unit=d_per_unit,
            frac_positive=frac_positive,
            **kwargs,
        )
        self.T = T
        self.p = p
        self.d_per_unit = d_per_unit
        self.frac_positive = frac_positive

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __iter__(self):
        return iter((self.T, self.p, self.d_per_unit, self.frac_positive))


class Decision(str):
    """String subclass representing Stage A decision with metadata."""

    def __new__(cls, text, a1_signal=False, a3_signal=False, decision_type='NEITHER', **kwargs):
        obj = str.__new__(cls, text)
        obj.decision = text
        obj.a1_signal = a1_signal
        obj.a3_signal = a3_signal
        obj.decision_type = decision_type
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj


def load_landmark_genes(path=None):
    """Load the 978 landmark gene symbols in order."""
    path = path or LANDMARKS_FILE_DEFAULT
    with open(path, 'r', encoding='utf-8') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_rows(bundle_path):
    """Load test rows from the XPert bundle (split_split_cold_cell_1 == 'test').

    Asserts exactly 21,151 rows.
    Returns RowsDict with keys: X, X_ctl, pert, cell, row_index, delta (float64).
    """
    if os.path.isdir(bundle_path):
        bundle_path = os.path.join(bundle_path, 'xpert_mdmt_splits.npz')

    z = np.load(bundle_path, allow_pickle=True)
    lab = z['split_split_cold_cell_1']
    test_idx = np.flatnonzero(lab == 'test')
    assert len(test_idx) == 21151, f'expected 21,151 test rows, got {len(test_idx)}'

    X = np.asarray(z['X'][test_idx], dtype=np.float32)
    X_ctl = np.asarray(z['X_ctl'][test_idx], dtype=np.float32)
    pert = np.asarray(z['meta_pert_id'][test_idx]).astype(str)
    cell = np.asarray(z['meta_cell'][test_idx]).astype(str)
    row_index = np.asarray(z['row_index'][test_idx]).astype(np.int64)
    delta = X.astype(np.float64) - X_ctl.astype(np.float64)

    return RowsDict({
        'X': X,
        'X_ctl': X_ctl,
        'pert': pert,
        'cell': cell,
        'row_index': row_index,
        'delta': delta,
    })


def load_labels(tsv_path):
    """Load mechanism labels from ChEMBL DTI table.

    Filters for direct_interaction == 1 and organism == 'Homo sapiens', with any target_type.
    Returns:
    - pert_to_parent: pert_id -> parent_chembl_id
    - parent_to_moa: parent_chembl_id -> set(mechanism_of_action)
    - parent_to_targets: parent_chembl_id -> list of (gene_symbol, action_type)
    """
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    # direct_interaction == 1 and organism == 'Homo sapiens'
    mask = (df['direct_interaction'].astype(str) == '1') & (df['organism'] == 'Homo sapiens')
    sub = df[mask]

    pert_to_parent = {}
    parent_to_moa = defaultdict(set)
    parent_to_targets = defaultdict(list)

    for _, row in sub.iterrows():
        p_id = str(row['pert_id']).strip()
        parent = str(row['parent_chembl_id']).strip()
        if not p_id or not parent or parent == 'nan':
            continue

        pert_to_parent[p_id] = parent

        moa = str(row['mechanism_of_action']).strip()
        if moa and moa != 'nan':
            parent_to_moa[parent].add(moa)

        gene = str(row['gene_symbol']).strip()
        action = str(row['action_type']).strip()
        if gene and gene != 'nan':
            pair = (gene, action)
            if pair not in parent_to_targets[parent]:
                parent_to_targets[parent].append(pair)

    # ensure defaultdicts become normal dicts
    return dict(pert_to_parent), dict(parent_to_moa), dict(parent_to_targets)


def units_for_cell(cell, rows, pert_to_parent=None, parent_to_moa=None, parent_to_targets=None):
    """Group rows for a given cell into compound units (one unit per parent for labelled compounds,

    one unit per pert_id for unlabelled ones).

    Signature: mean Δ over the unit's rows in the cell minus the mean of all units' signatures.
    Also returns each unit's set of X_ctl row hashes and row_index list.
    """
    if isinstance(rows, str) and isinstance(cell, (dict, object)):
        cell, rows = rows, cell

    pert_to_parent = pert_to_parent or {}
    parent_to_moa = parent_to_moa or {}
    parent_to_targets = parent_to_targets or {}

    cell_mask = (rows['cell'] == cell)
    cell_indices = np.flatnonzero(cell_mask)

    if len(cell_indices) == 0:
        return {}

    unit_groups = defaultdict(list)
    for idx in cell_indices:
        pert = rows['pert'][idx]
        if pert in pert_to_parent:
            uid = pert_to_parent[pert]
            is_labelled = True
        else:
            uid = pert
            is_labelled = False
        unit_groups[uid].append(idx)

    # Compute raw signatures (mean delta per unit) and gather row information
    raw_sigs = {}
    unit_info = {}

    for uid, idxs in unit_groups.items():
        deltas = rows['delta'][idxs]
        raw_sig = np.mean(deltas, axis=0)
        raw_sigs[uid] = raw_sig

        ctl_hashes = set()
        for i in idxs:
            row_ctl = rows['X_ctl'][i].astype(np.float32).tobytes()
            ctl_hashes.add(hashlib.sha1(row_ctl).hexdigest())

        row_indices = [int(rows['row_index'][i]) for i in idxs]
        # Store deltas and row indices aligned for split-half self-retrieval
        row_deltas = [rows['delta'][i] for i in idxs]

        unit_info[uid] = {
            'unit_id': uid,
            'is_labelled': (uid in parent_to_moa or uid in set(pert_to_parent.values())),
            'raw_sig': raw_sig,
            'ctl_hashes': ctl_hashes,
            'row_indices': row_indices,
            'row_deltas': row_deltas,
            'moas': parent_to_moa.get(uid, set()),
            'targets': parent_to_targets.get(uid, []),
        }

    # Mean of all units' signatures in this cell
    all_raw = np.stack(list(raw_sigs.values()), axis=0)
    cell_mean = np.mean(all_raw, axis=0)

    for uid, info in unit_info.items():
        info['sig'] = info['raw_sig'] - cell_mean
        info['cell_mean'] = cell_mean

    return unit_info


def scored_cells(cell_units_map=None, rows=None, pert_to_parent=None, parent_to_moa=None):
    """Identify scored cells: cells with >= 25 labelled units in MoA classes that have >= 2 labelled units there.

    Returns (scored_list, counts_dict).
    """
    if cell_units_map is None:
        if rows is None:
            raise ValueError('Either cell_units_map or rows must be provided')
        pert_to_parent = pert_to_parent or {}
        parent_to_moa = parent_to_moa or {}
        unique_cells = np.unique(rows['cell'])
        cell_units_map = {c: units_for_cell(c, rows, pert_to_parent, parent_to_moa) for c in unique_cells}

    counts = {}
    for cell, units in cell_units_map.items():
        labelled_units = [u for u in units.values() if u['is_labelled'] and len(u['moas']) > 0]
        # Count labelled units per MoA class in this cell
        moa_counts = defaultdict(int)
        for u in labelled_units:
            for m in u['moas']:
                moa_counts[m] += 1

        multi_classes = {m for m, count in moa_counts.items() if count >= 2}
        n_qualifying = sum(1 for u in labelled_units if any(m in multi_classes for m in u['moas']))
        counts[cell] = n_qualifying

    # Filter >= 25
    qualifying_set = {c for c, n in counts.items() if n >= 25}

    # PI: §94.7 item 1 fixed the five scored cells; any other qualifying cell is reported in counts, never scored
    scored = [c for c in SCORED_CELLS_ORDER if c in qualifying_set]

    return scored, counts


def calc_auroc(pos, neg):
    """Compute Mann-Whitney AUROC with average ranks (ties count 1/2)."""
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    scores = np.concatenate([pos, neg])
    ranks = rankdata(scores, method='average')
    sum_pos_ranks = np.sum(ranks[:n_pos])
    u = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def a1_cell(sigs, labels, ctl_hashes, rng_seed, n_perm=1000, plate_rule=True):
    """A1 readout for a single cell: MoA-mate retrieval AUROC.

    Parameters:
    - sigs: array of shape (N, G) or list of signatures for labelled units
    - labels: list of sets of MoA strings for each unit
    - ctl_hashes: list of sets of X_ctl sha1 strings for each unit
    - rng_seed: int
    - n_perm: int, default 1000
    - plate_rule: bool, default True (exclude pairs sharing control hashes)

    Returns: A1Result
    """
    sigs = np.asarray(sigs, dtype=np.float64)
    N = len(sigs)
    if N < 2:
        return A1Result(np.nan, np.nan, np.nan, 1.0, 0, {})

    # Compute pairwise Pearson correlations
    # Centered and normalized
    centered = sigs - np.mean(sigs, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = centered / norms
    P = np.matmul(normalized, normalized.T)
    np.clip(P, -1.0, 1.0, out=P)

    # Plate exclusion matrix
    E = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(E, True)
    if plate_rule:
        for i in range(N):
            hi = ctl_hashes[i]
            for j in range(i + 1, N):
                if bool(hi & ctl_hashes[j]):
                    E[i, j] = True
                    E[j, i] = True

    # PI (vectorised; identical to W33's loops): R[i, j] = rank of P[i, j] among i's candidates (average ranks), 0 elsewhere.
    C = ~E
    R = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        cands = np.flatnonzero(C[i])
        if len(cands) > 0:
            R[i, cands] = rankdata(P[i, cands], method='average')
    n_cand = C.sum(1)
    classes = sorted(set().union(*[set(l) for l in labels])) if N else []
    cidx = {m: k for k, m in enumerate(classes)}
    L = np.zeros((N, max(len(classes), 1)), dtype=np.float64)
    for i, li in enumerate(labels):
        for m in li:
            L[i, cidx[m]] = 1.0
    has_label = L.sum(1) > 0
    Mate = (L @ L.T) > 0

    def unit_scores(mate):
        mp = mate & C
        n_pos = mp.sum(1)
        n_neg = n_cand - n_pos
        ok = has_label & (n_pos >= 1) & (n_neg >= 1)
        u = (R * mp).sum(1) - n_pos * (n_pos + 1) / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            auc = u / (n_pos * n_neg)
        return auc, ok, has_label & (n_pos >= 1)

    auc, ok, with_mate = unit_scores(Mate)
    unit_aurocs = {int(i): float(auc[i]) for i in np.flatnonzero(ok)}
    n_scored = len(unit_aurocs)
    n_with_mate = int(with_mate.sum())
    A1_c = float(np.mean(list(unit_aurocs.values()))) if n_scored > 0 else np.nan

    # Per-class AUROC for classes with >= 3 members
    class_members = defaultdict(list)
    for i in range(N):
        for m in labels[i]:
            class_members[m].append(i)

    per_class_auroc = {}
    for cls_name, mems in class_members.items():
        if len(mems) >= 3:
            cls_aurocs = [unit_aurocs[m] for m in mems if m in unit_aurocs]
            if len(cls_aurocs) > 0:
                per_class_auroc[cls_name] = float(np.mean(cls_aurocs))

    if n_scored == 0 or np.isnan(A1_c):
        return A1Result(np.nan, np.nan, np.nan, 1.0, 0, per_class_auroc, n_with_mate=n_with_mate)

    # Null permutation distribution: unit i takes the labels of perm[i], so the mate matrix becomes Mate[perm][:, perm]
    rng = np.random.default_rng(rng_seed)
    null_scores = np.empty(n_perm, dtype=np.float64)
    hl0 = has_label
    for draw in range(n_perm):
        perm = rng.permutation(N)
        has_label = hl0[perm]
        a_d, ok_d, _ = unit_scores(Mate[np.ix_(perm, perm)])
        null_scores[draw] = float(np.mean(a_d[ok_d])) if ok_d.any() else np.nan
    has_label = hl0

    valid_null = null_scores[np.isfinite(null_scores)]
    if len(valid_null) == 0:
        null_mean = np.nan
        null_sd = np.nan
        p_val = 1.0
    else:
        null_mean = float(np.mean(valid_null))
        null_sd = float(np.std(valid_null, ddof=1)) if len(valid_null) > 1 else 0.0
        p_val = float((1.0 + np.sum(valid_null >= A1_c)) / (1.0 + len(valid_null)))

    return A1Result(
        a1=A1_c,
        null_mean=null_mean,
        null_sd=null_sd,
        p_value=p_val,
        n_scored=n_scored,
        per_class_auroc=per_class_auroc,
        n_with_mate=n_with_mate,
    )


def compute_self_retrieval_and_active_subset(cell_units, rng_seed, active_threshold=0.2, plate_rule=True):
    """Compute C4 diagnostics:

    1. Self-retrieval ceiling: split each unit's rows by parity of sorted row_index.
       AUROC of corr(sig_A_i, sig_B_i) against corr(sig_A_i, sig_j) over other units j (units with >= 2 rows).
    2. Active subset: units whose split-half self-correlation is >= active_threshold.
       Report A1 on the subset with its own label-permutation null.
    """
    labelled = [u for u in cell_units.values() if u['is_labelled'] and len(u['moas']) > 0]
    if len(labelled) == 0:
        return {
            'self_retrieval_ceiling': np.nan,
            'split_half_corrs': {},
            'active_subset': None,
        }

    # Compute split-halves
    split_half_corrs = {}
    sig_A_map = {}
    sig_B_map = {}

    for u in labelled:
        n_rows = len(u['row_indices'])
        if n_rows < 2:
            continue
        # sort rows by row_index
        sorted_pairs = sorted(zip(u['row_indices'], u['row_deltas']), key=lambda x: x[0])
        even_deltas = [p[1] for idx, p in enumerate(sorted_pairs) if idx % 2 == 0]
        odd_deltas = [p[1] for idx, p in enumerate(sorted_pairs) if idx % 2 == 1]

        sig_A = np.mean(even_deltas, axis=0) - u['cell_mean']
        sig_B = np.mean(odd_deltas, axis=0) - u['cell_mean']

        # Pearson correlation
        cA = sig_A - np.mean(sig_A)
        cB = sig_B - np.mean(sig_B)
        denom = (np.linalg.norm(cA) * np.linalg.norm(cB))
        r_self = float(np.dot(cA, cB) / denom) if denom > 0 else 0.0

        split_half_corrs[u['unit_id']] = r_self
        sig_A_map[u['unit_id']] = sig_A
        sig_B_map[u['unit_id']] = sig_B

    # Self-retrieval AUROC per qualifying unit
    self_aurocs = []
    qualifying_units = list(sig_A_map.keys())

    for uid in qualifying_units:
        sig_A = sig_A_map[uid]
        r_self = split_half_corrs[uid]

        # Negative scores: corr(sig_A, sig_j) for other units
        neg_corrs = []
        cA = sig_A - np.mean(sig_A)
        normA = np.linalg.norm(cA)

        for other_u in labelled:
            if other_u['unit_id'] == uid:
                continue
            sig_j = other_u['sig']
            cj = sig_j - np.mean(sig_j)
            denom = normA * np.linalg.norm(cj)
            r_other = float(np.dot(cA, cj) / denom) if denom > 0 else 0.0
            neg_corrs.append(r_other)

        if len(neg_corrs) > 0:
            # AUROC of 1 positive against len(neg_corrs) negatives
            auroc = float(np.mean([1.0 if r_self > neg else (0.5 if r_self == neg else 0.0) for neg in neg_corrs]))
            self_aurocs.append(auroc)

    self_retrieval_ceiling = float(np.mean(self_aurocs)) if len(self_aurocs) > 0 else np.nan

    # Active subset A1
    active_units = [u for u in labelled if split_half_corrs.get(u['unit_id'], -1.0) >= active_threshold]
    active_result = None

    if len(active_units) >= 2:
        active_sigs = [u['sig'] for u in active_units]
        active_labels = [u['moas'] for u in active_units]
        active_hashes = [u['ctl_hashes'] for u in active_units]
        res = a1_cell(active_sigs, active_labels, active_hashes, rng_seed=rng_seed + 100, plate_rule=plate_rule)
        active_result = dict(res)
        active_result['n_active'] = len(active_units)

    return {
        'self_retrieval_ceiling': self_retrieval_ceiling,
        'split_half_corrs': split_half_corrs,
        'active_subset': active_result,
    }


def load_progeny_network(top=500, landmarks_file=None):
    """Retrieve PROGENy footprints from decoupler and verify sha1.

    Restricts to the 978 landmark genes and pathways with >= 15 landmark genes.
    """
    import decoupler as dc

    net = dc.op.progeny(organism='human', top=top)
    sorted_df = net.sort_values(list(net.columns)).reset_index(drop=True)
    csv_bytes = sorted_df.to_csv(index=False).encode('utf-8')
    sha = hashlib.sha1(csv_bytes).hexdigest()
    if sha != PROGENY_SHA1:
        raise ValueError(f'PROGENy table sha1 mismatch: expected {PROGENY_SHA1}, got {sha}')

    landmarks = load_landmark_genes(landmarks_file)
    net_lm = net[net['target'].isin(landmarks)].copy()

    # Minimum 15 landmark genes per pathway (§94.7 item 8)
    counts = net_lm['source'].value_counts()
    keep_sources = counts[counts >= 15].index
    net_lm = net_lm[net_lm['source'].isin(keep_sources)].copy()
    return net_lm


def run_a3(cell_units_map, scored_cells_list, parent_to_targets, net=None, landmarks_file=None, gene_names=None, tmin=None, gated=True, rng_seed=9450, n_perm=1000):
    """Run A3 pathway activity readout (PROGENy ULM).

    Parameters:
    - cell_units_map: dict mapping cell -> dict of units
    - scored_cells_list: list of scored cells
    - parent_to_targets: dict mapping parent -> list of (gene_symbol, action_type)
    - net: PROGENy network DataFrame (if None, loaded via load_progeny_network)
    - gene_names: optional list of gene symbols matching signature dimensions
    - tmin: optional minimum number of targets per pathway for ULM (defaults to 15 or net minimum)
    - gated: bool (True for §94.7 item 2 gated table; False for §94.4 ungated table)
    - rng_seed: int, default 9450
    - n_perm: int, default 1000

    Returns: A3Result
    """
    import decoupler as dc

    if net is None:
        net = load_progeny_network(top=500, landmarks_file=landmarks_file)

    landmarks = load_landmark_genes(landmarks_file) if (landmarks_file or os.path.exists(LANDMARKS_FILE_DEFAULT)) else None
    class_table = GATED_CLASSES if gated else UNGATED_CLASSES

    if tmin is None:
        source_counts = net['source'].value_counts()
        min_targets = int(source_counts.min()) if len(source_counts) > 0 else 1
        tmin = min(15, min_targets)

    # Compute ULM activities per scored cell
    cell_acts = {}
    cell_labelled_list = {}

    for cell in scored_cells_list:
        units = cell_units_map.get(cell, {})
        labelled = [u for u in units.values() if u['is_labelled']]
        if len(labelled) == 0:
            continue

        u_ids = [u['unit_id'] for u in labelled]
        sigs = np.stack([u['sig'] for u in labelled], axis=0)

        # Build DataFrame with gene symbols
        if gene_names is not None and len(gene_names) == sigs.shape[1]:
            cols = gene_names
        elif landmarks is not None and len(landmarks) == sigs.shape[1]:
            cols = landmarks
        else:
            cols = [f'g{i}' for i in range(sigs.shape[1])]

        mat = pd.DataFrame(sigs, index=u_ids, columns=cols)
        assert len(set(cols) & set(net['target'])) > 0, 'signature genes and PROGENy targets do not overlap'

        # dc.mt.ulm
        acts, _ = dc.mt.ulm(mat, net, tmin=tmin)
        cell_acts[cell] = acts
        cell_labelled_list[cell] = u_ids

    # Find compounds for each class in each scored cell
    # A unit is a (class_def, cell) with >= 3 compounds
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

            target_set = cls_def['targets_fn'](cell)
            if not target_set:
                continue

            inh_sign = cls_def['inhibitor_sign']

            # Identify member compounds and their sign
            members = []
            for uid in u_ids:
                targets = parent_to_targets.get(uid, [])
                # Check matching targets
                matched_sign = None
                for gene, action in targets:
                    if gene in target_set:
                        act_upper = action.upper()
                        if act_upper in INHIBITOR_ACTIONS:
                            matched_sign = inh_sign
                            break
                        elif act_upper in AGONIST_ACTIONS:
                            matched_sign = -inh_sign
                            break
                if matched_sign is not None:
                    members.append((uid, matched_sign))

            if len(members) >= 3:
                eval_units.append({
                    'class': cls_name,
                    'cell': cell,
                    'pathway': pathway,
                    'inhibitor_sign': inh_sign,
                    'members': members,
                    'all_units': u_ids,
                })

    if len(eval_units) == 0:
        return A3Result(T=0.0, p=1.0, d_per_unit={}, frac_positive=0.0, units=[])

    # Compute observed d per unit
    d_per_unit = {}
    for u in eval_units:
        cell = u['cell']
        acts = cell_acts[cell]
        pathway = u['pathway']
        inh_sign = u['inhibitor_sign']

        # mean of (activity * sign) over class compounds
        in_scores = [acts.loc[uid, pathway] * s for uid, s in u['members']]
        mu_in = np.mean(in_scores)

        # other labelled compounds in cell
        member_set = {uid for uid, _ in u['members']}
        others = [uid for uid in u['all_units'] if uid not in member_set]
        if len(others) > 0:
            out_scores = [acts.loc[uid, pathway] * inh_sign for uid in others]
            mu_out = np.mean(out_scores)
        else:
            mu_out = 0.0

        d = float(mu_in - mu_out)
        d_per_unit[f"{u['class']}@{u['cell']}"] = d
        u['d_obs'] = d

    T_obs = float(np.mean(list(d_per_unit.values())))
    frac_positive = float(np.mean([d > 0 for d in d_per_unit.values()]))

    # Permutation null: permute labelled compounds within each cell
    rng = np.random.default_rng(rng_seed)
    null_T = np.empty(n_perm, dtype=np.float64)

    for draw in range(n_perm):
        # Permute within each cell
        cell_perm_map = {}
        for cell, u_ids in cell_labelled_list.items():
            perm = rng.permutation(len(u_ids))
            cell_perm_map[cell] = [u_ids[idx] for idx in perm]

        draw_ds = []
        for u in eval_units:
            cell = u['cell']
            acts = cell_acts[cell]
            pathway = u['pathway']
            inh_sign = u['inhibitor_sign']
            orig_u_ids = u['all_units']
            perm_u_ids = cell_perm_map[cell]

            # map original member positions to permuted units
            orig_indices = [orig_u_ids.index(uid) for uid, _ in u['members']]
            perm_members = [(perm_u_ids[idx], sign) for idx, (_, sign) in zip(orig_indices, u['members'])]

            in_scores = [acts.loc[uid, pathway] * s for uid, s in perm_members]
            mu_in = np.mean(in_scores)

            perm_member_set = {uid for uid, _ in perm_members}
            others = [uid for uid in perm_u_ids if uid not in perm_member_set]
            if len(others) > 0:
                out_scores = [acts.loc[uid, pathway] * inh_sign for uid in others]
                mu_out = np.mean(out_scores)
            else:
                mu_out = 0.0

            draw_ds.append(mu_in - mu_out)

        null_T[draw] = np.mean(draw_ds)

    p_val = float((1.0 + np.sum(null_T >= T_obs)) / (1.0 + n_perm))

    return A3Result(
        T=T_obs,
        p=p_val,
        d_per_unit=d_per_unit,
        frac_positive=frac_positive,
        units=[f"{u['class']}@{u['cell']}" for u in eval_units],
    )


def read_stage_a(result):
    """Mechanical reading of Stage A results per RESULTS §94.3, §94.4, §94.5 and §94.7.

    Decision rules:
    - A1 signal: >= 3 of the 5 scored cells with A1_c - null_mean >= 0.05 and p < 0.01
    - A3 signal: p < 0.01 and d > 0 in >= 2/3 of units (on gated table)
    """
    a1_data = result.get('a1', {})
    scored_cells_list = result.get('scored_cells', SCORED_CELLS_ORDER)

    # Evaluate A1 signal
    n_a1_sig_cells = 0
    self_ceilings = []
    active_a1s = []

    for cell in scored_cells_list:
        if cell in a1_data:
            c_res = a1_data[cell]
            a1_val = c_res.get('a1', np.nan)
            n_mean = c_res.get('null_mean', np.nan)
            p_val = c_res.get('p_value', 1.0)
            if np.isfinite(a1_val) and np.isfinite(n_mean):
                if (a1_val - n_mean >= 0.05) and (p_val < 0.01):
                    n_a1_sig_cells += 1

            sc = c_res.get('self_retrieval_ceiling')
            if sc is not None and np.isfinite(sc):
                self_ceilings.append(sc)

            act = c_res.get('active_subset')
            if act and act.get('a1') is not None and np.isfinite(act.get('a1')):
                active_a1s.append(act['a1'])

    a1_signal = (n_a1_sig_cells >= 3)

    # Evaluate A3 signal (on gated table)
    a3_gated = result.get('a3_gated', {})
    a3_p = a3_gated.get('p', 1.0)
    a3_frac = a3_gated.get('frac_positive', 0.0)
    a3_signal = (a3_p < 0.01) and (a3_frac >= (2.0 / 3.0))

    # Format scoping metrics for null sentence (§94.7 item 5)
    mean_sc = np.mean(self_ceilings) if len(self_ceilings) > 0 else np.nan
    mean_act = np.mean(active_a1s) if len(active_a1s) > 0 else np.nan
    X = f'{mean_sc:.3f}' if np.isfinite(mean_sc) else 'N/A'
    Y = f'{mean_act:.3f}' if np.isfinite(mean_act) else 'N/A'

    # Formulate decision string
    if a1_signal and a3_signal:
        text = (
            "Stage B tests whether v9's P7 predicted signatures retrieve MoA-mates and show "
            "the expected pathway direction. Its references are the drug-mean prediction \u03bc "
            "(no cell-specific information) and the measured ceiling, both registered then."
        )
        dtype = 'BOTH'
    elif a1_signal and not a3_signal:
        text = (
            "Stage B tests whether v9's P7 predicted signatures retrieve MoA-mates. "
            "Its references are the drug-mean prediction \u03bc (no cell-specific information) "
            "and the measured ceiling, both registered then."
        )
        dtype = 'A1'
    elif not a1_signal and a3_signal:
        text = (
            "Stage B tests whether v9's predicted signatures show the expected pathway direction, "
            "against the same references."
        )
        dtype = 'A3'
    else:
        text = (
            f"on these test cells, the measured landmark responses do not carry these mechanism "
            f"readouts (self-retrieval ceiling {X}; active-subset A1 {Y}), so model readouts of them "
            f"cannot be informative"
        )
        dtype = 'NEITHER'

    return Decision(
        text,
        a1_signal=a1_signal,
        a3_signal=a3_signal,
        decision_type=dtype,
        n_a1_sig_cells=n_a1_sig_cells,
        scoped_ceiling=X,
        scoped_active_a1=Y,
    )


def sha1_file(path):
    """Compute sha1 hash of file contents."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(bundle_path, labels_tsv, landmarks_file=None):
    """Execute complete Stage A analysis pipeline."""
    rows = load_rows(bundle_path)
    pert_to_parent, parent_to_moa, parent_to_targets = load_labels(labels_tsv)

    # Group into units for every cell
    unique_cells = np.unique(rows['cell'])
    cell_units = {c: units_for_cell(c, rows, pert_to_parent, parent_to_moa, parent_to_targets) for c in unique_cells}

    # Scored cells
    scored, cell_counts = scored_cells(cell_units)

    # A1 analysis per scored cell
    a1_results = {}
    for idx, cell in enumerate(scored):
        units = cell_units[cell]
        labelled = [u for u in units.values() if u['is_labelled'] and len(u['moas']) > 0]
        if len(labelled) == 0:
            continue

        sigs = [u['sig'] for u in labelled]
        labels = [u['moas'] for u in labelled]
        hashes = [u['ctl_hashes'] for u in labelled]

        # Seed per RESULTS §94.3: 9400 + k
        cell_seed = 9400 + idx
        res = a1_cell(sigs, labels, hashes, rng_seed=cell_seed)

        # C4 diagnostics
        c4 = compute_self_retrieval_and_active_subset(units, rng_seed=cell_seed)
        c_dict = dict(res)
        c_dict['self_retrieval_ceiling'] = c4['self_retrieval_ceiling']
        c_dict['active_subset'] = c4['active_subset']
        a1_results[cell] = c_dict

    # A3 analysis (PROGENy)
    net = load_progeny_network(landmarks_file=landmarks_file)
    genes = load_landmark_genes(landmarks_file)        # the bundle's gene axis (xpert_mdmt_extract asserted it identical)
    assert len(genes) == rows['delta'].shape[1] == 978, (len(genes), rows['delta'].shape)
    a3_gated = run_a3(cell_units, scored, parent_to_targets, net=net, gene_names=genes, gated=True)
    a3_ungated = run_a3(cell_units, scored, parent_to_targets, net=net, gene_names=genes, gated=False)

    return {
        'bundle_path': bundle_path,
        'labels_tsv': labels_tsv,
        'scored_cells': scored,
        'cell_counts': cell_counts,
        'a1': a1_results,
        'a3_gated': dict(a3_gated),
        'a3_ungated': dict(a3_ungated),
    }


def main():
    parser = argparse.ArgumentParser(description="RESULTS §94 Stage A: data-first mechanism ceilings")
    parser.add_argument('--bundle', help="Path to XPert split bundle (xpert_mdmt_splits.npz)")
    parser.add_argument('--labels', help="Path to ChEMBL DTI labels (chembl_dti_edges.tsv)")
    parser.add_argument('--landmarks', default=None, help="Path to 978 landmark genes txt")
    parser.add_argument('--out', help="Output JSON path")
    parser.add_argument('--read', help="Apply read_stage_a to an existing result JSON and print decision")
    args = parser.parse_args()

    if args.read:
        mk = json.load(open(args.read + '.marker', encoding='utf-8'))
        if not mk.get('complete') or mk.get('sha1') != sha1_file(args.read):
            raise SystemExit('REFUSED: %s does not match its marker' % args.read)
        with open(args.read, 'r', encoding='utf-8') as f:
            res = json.load(f)
        decision = read_stage_a(res)
        print(decision)
        return

    if not args.bundle or not args.labels or not args.out:
        parser.error("--bundle, --labels, and --out are required unless --read is specified.")

    result = run_pipeline(args.bundle, args.labels, landmarks_file=args.landmarks)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    out_sha1 = sha1_file(args.out)

    marker = {
        'complete': True,
        'file': os.path.basename(args.out),
        'sha1': out_sha1,
    }
    marker_path = args.out + '.marker'
    with open(marker_path, 'w', encoding='utf-8') as f:
        json.dump(marker, f, indent=2)

    # Also STAGE_A_COMPLETE.json in the same directory
    stage_a_marker = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'STAGE_A_COMPLETE.json')
    with open(stage_a_marker, 'w', encoding='utf-8') as f:
        json.dump(marker, f, indent=2)

    print(f"Wrote {args.out} (sha1: {out_sha1})")


if __name__ == '__main__':
    main()
