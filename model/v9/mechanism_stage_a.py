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
import re
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
# §95: per-split constants (cold-cell = §94, unchanged)
SPLITS = {
    'split_cold_cell_1': {'scored': SCORED_CELLS_ORDER, 'n_rows': 21151, 'a1_min': 3,
                          'gates': {'raf': {'HT29'}, 'mdm2': {'MCF7'}, 'er': {'MCF7'}}},
    'split_cold_drug_1': {'scored': ['MCF7', 'PC3', 'A375', 'HA1E', 'HT29', 'A549'], 'n_rows': 13445, 'a1_min': 4,
                          'gates': {'raf': {'HT29', 'A375'}, 'mdm2': {'MCF7', 'A549', 'A375'}, 'er': {'MCF7'},
                                    'dna_p53': {'MCF7', 'A549', 'A375'}},
                          # review 063 C1: DNA inhibitors -> p53 (+) in TP53-wild-type cells (membership by MoA string)
                          'extra_classes': [{'class': 'DNA', 'pathway': 'p53', 'inhibitor_sign': +1.0,
                                             'moa_match': 'DNA inhibitor', 'gate': 'dna_p53'}],
                          'noncns_a1_min': 3},
}
# review 063 C1(ii)-(iii): CNS / neurotransmitter / ion-channel classes, removed for the cold-drug A1 gate (fixed before data)
CNS_CHANNEL_RE = re.compile(r'dopamine|serotonin|5-ht|histamine|adrenergic|adrenoceptor|muscarinic|acetylcholine|gaba|'
                            r'opioid|glutamate|nmda|transporter|channel|cholinesterase|melatonin|sigma|cannabinoid|'
                            r'vasopressin|oxytocin|neurokinin|orexin|kir6', re.I)

ACTIVE = dict(SPLITS['split_cold_cell_1'], name='split_cold_cell_1')   # set by run_pipeline(split=...)

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
            if cell in ACTIVE['gates']['raf']
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
        'targets_fn': lambda cell: {'MDM2'} if cell in ACTIVE['gates']['mdm2'] else set(),
    },
    {
        'class': 'ER',
        'pathway': 'Estrogen',
        'inhibitor_sign': -1.0,  # agonist +1.0
        'cells': ['MCF7'],  # ER-positive only
        'targets_fn': lambda cell: {'ESR1', 'ESR2'} if cell in ACTIVE['gates']['er'] else set(),
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
        # review 060 C3: not a class-specific retrieval
        self['per_class_auroc_meaning'] = ('mean, over the class members, of each member\'s AUROC against all its mates (any '
                                           'shared MoA string); reported only')

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


def load_rows(bundle_path, row_index_npz=None, delta_spec=None, split='split_cold_cell_1'):
    """Load test rows from the XPert bundle (split_split_cold_cell_1 == 'test').

    Asserts exactly 21,151 rows.
    Returns RowsDict with keys: X, X_ctl, pert, cell, row_index, delta (float64).
    """
    if os.path.isdir(bundle_path):
        bundle_path = os.path.join(bundle_path, 'xpert_mdmt_splits.npz')

    z = np.load(bundle_path, allow_pickle=True)
    lab = z['split_' + split]
    test_idx = np.flatnonzero(lab == 'test')
    if row_index_npz is not None:
        # §94.2: the rows are the 21,151 row_index of v9p7_seed0.npz (P7's scored rows; the split's test set has 21,321).
        # Only the 'row_index' key is read from that file -- never a prediction.
        keep = set(np.load(row_index_npz)['row_index'].astype(np.int64).tolist())
        test_idx = test_idx[np.isin(np.asarray(z['row_index'][test_idx]).astype(np.int64), list(keep))]
    want_n = SPLITS[split]['n_rows']
    assert len(test_idx) == want_n, f'expected {want_n} test rows for {split}, got {len(test_idx)}'

    X = np.asarray(z['X'][test_idx], dtype=np.float32)
    X_ctl = np.asarray(z['X_ctl'][test_idx], dtype=np.float32)
    pert = np.asarray(z['meta_pert_id'][test_idx]).astype(str)
    cell = np.asarray(z['meta_cell'][test_idx]).astype(str)
    row_index = np.asarray(z['row_index'][test_idx]).astype(np.int64)
    delta = X.astype(np.float64) - X_ctl.astype(np.float64)
    if delta_spec:
        delta = load_predicted_delta(delta_spec, row_index)

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

        row_ctl_hashes = [hashlib.sha1(rows['X_ctl'][i].astype(np.float32).tobytes()).hexdigest() for i in idxs]
        ctl_hashes = set(row_ctl_hashes)

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
            'row_ctl_hashes': row_ctl_hashes,
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
    scored = [c for c in ACTIVE['scored'] if c in qualifying_set]

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

    # Compute split-halves (review 060 C2: negatives are half B of other units, same-plate units excluded)
    split_half_corrs = {}
    sig_A_map, sig_B_map, hashes_map, halves_share_plate = {}, {}, {}, {}

    def _pearson(x, y):
        cx, cy = x - np.mean(x), y - np.mean(y)
        den = np.linalg.norm(cx) * np.linalg.norm(cy)
        return float(np.dot(cx, cy) / den) if den > 0 else 0.0

    for u in labelled:
        n_rows = len(u['row_indices'])
        if n_rows < 2:
            continue
        hashes = u.get('row_ctl_hashes', [None] * n_rows)
        trip = sorted(zip(u['row_indices'], u['row_deltas'], hashes), key=lambda x: x[0])
        sig_A = np.mean([t[1] for k, t in enumerate(trip) if k % 2 == 0], axis=0) - u['cell_mean']
        sig_B = np.mean([t[1] for k, t in enumerate(trip) if k % 2 == 1], axis=0) - u['cell_mean']
        hA = {t[2] for k, t in enumerate(trip) if k % 2 == 0}
        hB = {t[2] for k, t in enumerate(trip) if k % 2 == 1}
        split_half_corrs[u['unit_id']] = _pearson(sig_A, sig_B)
        sig_A_map[u['unit_id']], sig_B_map[u['unit_id']] = sig_A, sig_B
        hashes_map[u['unit_id']] = u.get('ctl_hashes', set())
        halves_share_plate[u['unit_id']] = bool((hA & hB) - {None})

    self_aurocs, self_aurocs_clean = [], []
    qualifying_units = list(sig_A_map.keys())
    for uid in qualifying_units:
        r_self = split_half_corrs[uid]
        neg = [_pearson(sig_A_map[uid], sig_B_map[j]) for j in qualifying_units
               if j != uid and not (plate_rule and (hashes_map[uid] & hashes_map[j]))]
        if neg:
            auroc = float(np.mean([1.0 if r_self > x else (0.5 if r_self == x else 0.0) for x in neg]))
            self_aurocs.append(auroc)
            if not halves_share_plate[uid]:
                self_aurocs_clean.append(auroc)

    self_retrieval_ceiling = float(np.mean(self_aurocs)) if len(self_aurocs) > 0 else np.nan
    self_retrieval_ceiling_no_shared_plate = float(np.mean(self_aurocs_clean)) if self_aurocs_clean else np.nan

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
        'self_retrieval_ceiling_no_shared_plate': self_retrieval_ceiling_no_shared_plate,
        'n_self_retrieval_units': len(self_aurocs),
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
    class_table = (GATED_CLASSES + list(ACTIVE.get('extra_classes', []))) if gated else UNGATED_CLASSES
    unit_moas = {u['unit_id']: u.get('moas', set()) for cu in cell_units_map.values() for u in cu.values()}

    if tmin is None:
        source_counts = net['source'].value_counts()
        min_targets = int(source_counts.min()) if len(source_counts) > 0 else 1
        tmin = min(15, min_targets)

    # Compute ULM activities per scored cell
    cell_acts = {}
    cell_acts_std = {}
    activity_scale = {}
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
        # review 061 C1: each pathway's activity z-scored within the cell over its labelled compounds (removes cell scale)
        sd = acts.std(axis=0, ddof=1).replace(0, np.nan)
        cell_acts_std[cell] = (acts - acts.mean(axis=0)) / sd
        activity_scale[cell] = {'mean': {k: float(v) for k, v in acts.mean(axis=0).items()},
                                'sd': {k: float(v) for k, v in acts.std(axis=0, ddof=1).items()}}

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

            if 'moa_match' in cls_def:                       # review 063: membership by MoA string, gated by cell
                if cell not in ACTIVE['gates'].get(cls_def['gate'], set()):
                    continue
                members = [(uid, cls_def['inhibitor_sign']) for uid in u_ids if cls_def['moa_match'] in unit_moas.get(uid, set())]
                if len(members) >= 3:
                    eval_units.append({'class': cls_name, 'cell': cell, 'pathway': pathway,
                                       'inhibitor_sign': cls_def['inhibitor_sign'], 'members': members, 'all_units': u_ids})
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
                signs = set()
                for gene, action in targets:
                    if gene in target_set:
                        act_upper = action.upper()
                        if act_upper in INHIBITOR_ACTIONS:
                            signs.add(inh_sign)
                        elif act_upper in AGONIST_ACTIONS:
                            signs.add(-inh_sign)
                if len(signs) == 1:                  # review 060 C4: opposite-sign matching targets -> excluded
                    members.append((uid, signs.pop()))

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
        return A3Result(T=0.0, p=1.0, d_per_unit={}, frac_positive=0.0, units=[], d_std_per_unit={},
                        activity_scale=activity_scale)

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

    # review 061 C1: the same d on standardised activities (B3c of record uses it; A3's own reading stays on raw d)
    d_std_per_unit = {}
    for u in eval_units:
        z = cell_acts_std[u['cell']]
        member_set = {uid for uid, _ in u['members']}
        others = [uid for uid in u['all_units'] if uid not in member_set]
        mu_in = np.mean([z.loc[uid, u['pathway']] * sgn for uid, sgn in u['members']])
        mu_out = np.mean([z.loc[uid, u['pathway']] * u['inhibitor_sign'] for uid in others]) if others else 0.0
        d_std_per_unit[f"{u['class']}@{u['cell']}"] = float(mu_in - mu_out)

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
        d_std_per_unit=d_std_per_unit,
        activity_scale=activity_scale,
    )


def read_stage_a(result):
    """Mechanical reading of Stage A results per RESULTS §94.3, §94.4, §94.5 and §94.7.

    Decision rules:
    - A1 signal: >= 3 of the 5 scored cells with A1_c - null_mean >= 0.05 and p < 0.01
    - A3 signal: p < 0.01 and d > 0 in >= 2/3 of units (on gated table)
    """
    a1_data = result.get('a1', {})
    scored_cells_list = result.get('scored_cells', SPLITS[result.get('split', 'split_cold_cell_1')]['scored'])

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

    a1_signal = (n_a1_sig_cells >= SPLITS[result.get('split', 'split_cold_cell_1')]['a1_min'])

    # Evaluate A3 signal (on gated table)
    a3_gated = result.get('a3_gated', {})
    a3_p = a3_gated.get('p', 1.0)
    a3_frac = a3_gated.get('frac_positive', 0.0)
    a3_signal = (a3_p < 0.01) and (a3_frac >= (2.0 / 3.0))
    a3_classes = sorted({u.split('@')[0] for u in a3_gated.get('d_per_unit', {})})
    split_cfg = SPLITS[result.get('split', 'split_cold_cell_1')]
    p9_gate = None
    if 'noncns_a1_min' in split_cfg:                     # review 063 C1(iii), cold-drug only
        n_nc = sum(1 for c in scored_cells_list
                   if (a1_data.get(c, {}).get('a1_noncns') or {}).get('a1') is not None
                   and a1_data[c]['a1_noncns']['a1'] - a1_data[c]['a1_noncns']['null_mean'] >= 0.05
                   and a1_data[c]['a1_noncns']['p_value'] < 0.01)
        a1_nc_signal = n_nc >= split_cfg['noncns_a1_min']
        a3_multi = a3_signal and len(a3_classes) >= 2
        p9_gate = {'a1_noncns_cells': n_nc, 'a1_noncns_signal': a1_nc_signal, 'a3_classes': a3_classes,
                   'a3_signal_multiclass': a3_multi, 'open': bool(a1_nc_signal or a3_multi)}

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

    if p9_gate is not None and not p9_gate['open']:
        text += (" P9 mechanism gate CLOSED: the cold-drug split's held-out compounds cannot test mechanism-from-chemistry at "
                 "these readouts; P9 may be justified only on accuracy grounds, registered as such.")
    elif p9_gate is not None:
        text += " P9 mechanism gate OPEN (review 063 C1(iii))."
    return Decision(
        text,
        p9_gate=p9_gate,
        a3_classes=a3_classes,
        a1_signal=a1_signal,
        a3_signal=a3_signal,
        decision_type=dtype,
        n_a1_sig_cells=n_a1_sig_cells,
        scoped_ceiling=X,
        scoped_active_a1=Y,
    )


def load_predicted_delta(spec, row_index):
    """§94.9: 'a.npz[,b.npz,...]:key' -> the mean over the files of key, aligned to row_index by each file's own 'row_index'.
    Every file must hold exactly the Stage A rows."""
    paths, key = spec.rsplit(':', 1)
    minus = None
    if '-' in key:
        key, minus = key.split('-', 1)
    mats = []
    want = np.asarray(row_index, np.int64)
    for path in paths.split(','):
        z = np.load(path, allow_pickle=False)
        ri = np.asarray(z['row_index'], np.int64)
        pos = {int(r): i for i, r in enumerate(ri)}
        assert set(want.tolist()) <= set(pos), '%s lacks some Stage A rows' % path
        sel = [pos[int(r)] for r in want]
        m = np.asarray(z[key], np.float64)[sel]
        if minus:
            m = m - np.asarray(z[minus], np.float64)[sel]
        mats.append(m)
    return np.mean(mats, axis=0)


def b3c(d_meas, d_pred, n_perm=10000, rng_seed=9460):
    """§94.9 B3c: Spearman over the gated units of within-class-centred d (measured vs predicted); null = predicted d permuted
    across cells within each class. Returns (rho, p, n_units)."""
    from scipy.stats import spearmanr
    units = sorted(d_meas)
    assert set(units) == set(d_pred), 'measured and predicted units differ'
    cls = np.array([u.split('@')[0] for u in units])
    m = np.array([d_meas[u] for u in units], float)
    q = np.array([d_pred[u] for u in units], float)

    def centre(v):
        out = v.copy()
        for k in set(cls):
            out[cls == k] -= v[cls == k].mean()
        return out
    mc = centre(m)

    def _rho(a, b):          # a prediction with no within-class variation has no ordering: rho = 0 (Spearman undefined)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return 0.0
        return float(spearmanr(a, b).correlation)
    rho = _rho(mc, centre(q))
    rng = np.random.default_rng(rng_seed)
    groups = [np.flatnonzero(cls == k) for k in sorted(set(cls))]
    null = np.empty(n_perm)
    for t in range(n_perm):
        qp = q.copy()
        for g in groups:
            qp[g] = q[rng.permutation(g)]
        null[t] = _rho(mc, centre(qp))
    null = np.nan_to_num(null, nan=0.0)
    return rho, float((1 + np.sum(null >= rho)) / (1 + n_perm)), len(units)


def _verified_json(path):
    mk = json.load(open(path + '.marker', encoding='utf-8'))
    if not mk.get('complete') or mk.get('sha1') != sha1_file(path):
        raise SystemExit('REFUSED: %s does not match its marker' % path)
    return json.load(open(path, encoding='utf-8'))


def read_stage_b(measured, v9_seeds, v9_mean, mu, ridge=None):
    """§94.9's mechanical readings. Arguments are Stage-A-format result dicts (measured = Stage A's own)."""
    cells = list(measured['scored_cells'])
    dm = measured['a3_gated']['d_std_per_unit']           # review 061 C1: of record, standardised
    dm_raw = measured['a3_gated']['d_per_unit']

    def one(res):
        a1 = {c: res['a1'][c]['a1'] - res['a1'][c]['null_mean'] for c in cells}
        b1 = sum(1 for c in cells if a1[c] >= 0.05 and res['a1'][c]['p_value'] < 0.01) >= \
            SPLITS[measured.get('split', 'split_cold_cell_1')]['a1_min']
        b3 = res['a3_gated']['p'] < 0.01 and res['a3_gated']['frac_positive'] >= 2.0 / 3.0
        rho, p, n = b3c(dm, res['a3_gated']['d_std_per_unit'])
        rho_raw, p_raw, _ = b3c(dm_raw, res['a3_gated']['d_per_unit'])
        return {'B1_signal': bool(b1), 'B3_signal': bool(b3), 'B3c_rho': rho, 'B3c_p': p, 'B3c_units': n,
                'reported_B3c_raw_rho': rho_raw, 'reported_B3c_raw_p': p_raw,
                'A1': {c: res['a1'][c]['a1'] for c in cells}, 'B3_T': res['a3_gated']['T']}
    mu_r = one(mu)
    runs = {'seed_mean': v9_mean}
    runs.update({'seed%d' % k: r for k, r in enumerate(v9_seeds)})
    per = {}
    for name, res in runs.items():
        r = one(res)
        diff = {c: r['A1'][c] - mu_r['A1'][c] for c in cells}
        r['A1_minus_mu'] = diff
        r['expresses'] = r['B1_signal'] and r['B3_signal']
        r['beyond_mu_2a'] = (sum(v > 0 for v in diff.values()) >= 4) and (float(np.mean(list(diff.values()))) >= 0.02)
        r['beyond_mu_2b'] = (r['B3c_rho'] > 0) and (r['B3c_p'] < 0.05) and (r['B3c_rho'] > mu_r['B3c_rho'])
        r['fraction_of_measured_reference'] = {c: (r['A1'][c] - 0.5) / (measured['a1'][c]['a1'] - 0.5) for c in cells}
        per[name] = r
    allrun = lambda k: all(per[n][k] for n in per)                     # noqa: E731 -- seed-mean AND every seed
    mu_expresses = bool(mu_r['B1_signal'] and mu_r['B3_signal'])
    out = {'reading_1_expresses_mechanism': allrun('expresses'),
           'reading_1_mu_also_expresses': mu_expresses,           # review 061 C4: stated beside v9's
           'reading_1_sentence': ("v9's predictions express mechanism" + (' (as does mu)' if mu_expresses else ''))
           if allrun('expresses') else "v9's predictions do not express mechanism by Stage A's rules", 'reading_2a_beyond_mu_retrieval': allrun('beyond_mu_2a'),
           'reading_2b_beyond_mu_cell_specific_pathway': allrun('beyond_mu_2b'), 'per_run': per, 'mu': mu_r,
           'mu_fraction_of_measured_reference': {c: (mu_r['A1'][c] - 0.5) / (measured['a1'][c]['a1'] - 0.5) for c in cells},
           'measured': {'A1': {c: measured['a1'][c]['a1'] for c in cells}, 'B3_T': measured['a3_gated']['T']},
           'reported_ridge': one(ridge) if ridge else None,
           'scope': 'model predictions, not internal attributions; compounds seen in training (cold-cell split); A3 has no '
                    'positive-control unit (94.8); the measured values are a reference, not a ceiling for noise-free '
                    'predictions (fractions > 1 possible); C4 self-retrieval diagnostics are not meaningful on prediction runs'}
    return out


def sha1_file(path):
    """Compute sha1 hash of file contents."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(bundle_path, labels_tsv, landmarks_file=None, row_index_npz=None, delta_spec=None, split='split_cold_cell_1'):
    """Execute complete Stage A analysis pipeline (Stage B: delta_spec gives the predicted delta, §94.9)."""
    ACTIVE.clear()
    ACTIVE.update(SPLITS[split], name=split)
    rows = load_rows(bundle_path, row_index_npz, delta_spec, split=split)
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
        noncns = None
        if 'noncns_a1_min' in ACTIVE:
            lab_nc = [{m for m in l if not CNS_CHANNEL_RE.search(m)} for l in labels]
            keep = [i for i, l in enumerate(lab_nc) if l]
            if len(keep) >= 2:
                noncns = dict(a1_cell([sigs[i] for i in keep], [lab_nc[i] for i in keep], [hashes[i] for i in keep],
                                      rng_seed=cell_seed + 50))

        # C4 diagnostics
        c4 = compute_self_retrieval_and_active_subset(units, rng_seed=cell_seed)
        c_dict = dict(res)
        if noncns is not None:
            c_dict['a1_noncns'] = noncns
        c_dict['self_retrieval_ceiling'] = c4['self_retrieval_ceiling']
        c_dict['self_retrieval_ceiling_no_shared_plate'] = c4['self_retrieval_ceiling_no_shared_plate']
        c_dict['n_self_retrieval_units'] = c4['n_self_retrieval_units']
        c_dict['active_subset'] = c4['active_subset']
        a1_results[cell] = c_dict

    # A3 analysis (PROGENy)
    net = load_progeny_network(landmarks_file=landmarks_file)
    genes = load_landmark_genes(landmarks_file)        # the bundle's gene axis (xpert_mdmt_extract asserted it identical)
    assert len(genes) == rows['delta'].shape[1] == 978, (len(genes), rows['delta'].shape)
    a3_gated = run_a3(cell_units, scored, parent_to_targets, net=net, gene_names=genes, gated=True)
    a3_ungated = run_a3(cell_units, scored, parent_to_targets, net=net, gene_names=genes, gated=False)

    return {
        'split': split,
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
    parser.add_argument('--rows', default=None, help="npz whose 'row_index' key lists the rows (v9p7_seed0.npz; §94.2)")
    parser.add_argument('--split', default='split_cold_cell_1', choices=sorted(SPLITS))
    parser.add_argument('--delta', default=None, help="§94.9 Stage B: 'a.npz[,b.npz]:key', predicted delta in place of measured")
    parser.add_argument('--read_b', nargs='+', default=None,
                        help="§94.9: MEASURED V9_SEED0 V9_SEED1 V9_SEED2 V9_MEAN MU [RIDGE] (result JSONs with markers)")
    parser.add_argument('--out', help="Output JSON path")
    parser.add_argument('--read', help="Apply read_stage_a to an existing result JSON and print decision")
    args = parser.parse_args()

    if args.read_b:
        f = [_verified_json(x) for x in args.read_b]
        out = read_stage_b(f[0], f[1:4], f[4], f[5], f[6] if len(f) > 6 else None)
        print(json.dumps(out, indent=1, default=float))
        return

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

    result = run_pipeline(args.bundle, args.labels, landmarks_file=args.landmarks, row_index_npz=args.rows,
                          delta_spec=args.delta, split=args.split)
    result['delta_source'] = args.delta or 'measured'

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
