"""v11's DTI probe split (RESULTS 102 / 102.1; review 093 C5, 098 C6). Outcome-free.

Pool: compounds annotated in chembl_dti_edges.tsv or dti_reference.tsv, in molecule groups (InChIKey first block; a compound
without one is its own group). A group is excluded from the pool if ANY member is a test compound of either split
(cold_drug_1 test or excluded_test_compound; cold_cell_1 test rows), so no probe compound shares a molecule with a compound
whose annotations v11 may see. 20 % of the pool's compounds are drawn as whole groups by np.random.default_rng(0).
Probe compounds' targets never enter v11's inputs, graph or auxiliary labels; the probe is scored only on them.

    python model/v11/dti_probe_split.py --out model/results/v11/dti_probe_split.json
"""
import argparse, collections, csv, hashlib, json, os

import numpy as np

CHEMBL = 'drug/outputs/dti/chembl_dti_edges.tsv'
REF = 'drug/outputs/dti/dti_reference.tsv'
INCHI = 'drug/outputs/dti/drug_inchikeys.tsv'
SUB = 'external/v10c_full/v10c_full_substrate.npz'
FRACTION = 0.20


def groups_of(compounds, block1):
    g = collections.defaultdict(set)
    for p in compounds:
        g[block1.get(p) or f'self:{p}'].add(p)
    return g


def split(pool_groups, fraction=FRACTION, seed=0):
    keys = sorted(pool_groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    n_pool = sum(len(v) for v in pool_groups.values())
    probe, n = [], 0
    for i in order:
        if n >= fraction * n_pool:
            break
        probe.append(keys[i])
        n += len(pool_groups[keys[i]])
    return sorted(probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ann = {r['pert_id'] for r in csv.DictReader(open(CHEMBL, encoding='utf-8'), delimiter='\t')}
    ref_rows = list(csv.DictReader(open(REF, encoding='utf-8'), delimiter='\t'))
    ref = {r['pert_id'] for r in ref_rows}
    block1 = {r['pert_id']: r['block1'] for r in csv.DictReader(open(INCHI, encoding='utf-8'), delimiter='\t') if r.get('block1')}
    z = np.load(SUB, allow_pickle=True)
    pert = z['meta_pert_id']
    test = (set(pert[z['split_cold_drug_1'] == 'test']) | set(pert[z['split_cold_drug_1'] == 'excluded_test_compound'])
            | set(pert[z['split_cold_cell_1'] == 'test']))
    all_groups = groups_of(ann | ref | test, block1)
    pool = {k: v & (ann | ref) for k, v in all_groups.items() if not (v & test) and (v & (ann | ref))}
    probe_groups = split(pool)
    probe = sorted(set().union(*(pool[k] for k in probe_groups)))
    rec = {'rule': 'RESULTS 102 / 102.1: 20 % of the annotated, test-free molecule groups, default_rng(0)',
           'pool_compounds': int(sum(len(v) for v in pool.values())), 'pool_groups': len(pool),
           'probe_compounds': len(probe), 'probe_groups': len(probe_groups),
           'probe_in_dti_reference': len(set(probe) & ref),
           'probe_reference_pairs': sum(1 for r in ref_rows if r['pert_id'] in set(probe)),
           'probe_chembl_annotated': len(set(probe) & ann),
           'inputs': {os.path.basename(p): hashlib.sha1(open(p, 'rb').read()).hexdigest() for p in (CHEMBL, REF, INCHI)},
           'probe': probe}
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != 'probe'}, indent=1))


if __name__ == '__main__':
    main()
