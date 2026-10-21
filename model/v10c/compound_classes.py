"""Mechanism classes, one per compound, fixed before any T-C output exists (RESULTS 100.13 / 101). Outcome-free: built from
ChEMBL mechanism text, ChEMBL target symbols and the epigenetic name lists only.

Priority (a compound with several mechanisms takes the first that matches):
  epigenetic > nuclear_receptor > kinase > dna_damage > cytoskeleton > proteostasis > receptor_channel_transporter >
  other_annotated ; no ChEMBL mechanism and no name match -> unannotated
Sub-labels kept for 101's tests: dnmt_inhibitor, bet_inhibitor, bet_inactive_enantiomer, hdac_inhibitor.

    python model/v10c/compound_classes.py --out model/results/v10c/compound_classes.json
"""
import argparse, csv, json, re, collections

CHEMBL = 'drug/outputs/dti/chembl_dti_edges.tsv'
DRUGS = 'drug/outputs/drug_list.tsv'
EPI = 'drug/outputs/dti/epi_drug_pert_ids.json'

SUB_BY_NAME = {
    'dnmt_inhibitor': {'azacitidine', 'decitabine', 'zebularine'},
    'bet_inhibitor': {'jq1-(+)', 'i-bet-762', 'i-bet-151', 'pfi-1'},
    'bet_inactive_enantiomer': {'jq1-(-)'},
    'other_epigenetic_by_name': {'tranylcypromine', 'bix-01294', 'gsk-j4', 'chaetocin', 'gsk126', 'epz-005687', 'epz-5676',
                                 'epz004777'},
}
SIRTUIN_OR_WEAK = {'ex-527', 'nicotinamide', 'resveratrol', 'sirtinol', 'salermide', 'tenovin-1', 'curcumin'}
RULES = [
    ('epigenetic', re.compile(r'histone|bromodomain|DNA \(cytosine-5\)-methyltransferase|methyltransferase|demethylase', re.I)),
    ('nuclear_receptor', re.compile(r'(glucocorticoid|estrogen|androgen|progesterone|mineralocorticoid|retinoic acid|retinoid X|'
                                    r'peroxisome proliferator|vitamin D|thyroid hormone)[^;]* receptor', re.I)),
    ('kinase', re.compile(r'kinase|growth factor receptor[^;]*inhibitor|mTOR', re.I)),
    ('dna_damage', re.compile(r'^(?!.*(bacterial|virus|viral)).*(DNA inhibitor|topoisomerase|DNA polymerase|ribonucleotide '
                              r'reductase|thymidylate synthase|dihydrofolate reductase|DNA alkylat)', re.I)),
    ('cytoskeleton', re.compile(r'tubulin|kinesin', re.I)),
    ('proteostasis', re.compile(r'HSP90|proteasome|ubiquitin', re.I)),
    ('receptor_channel_transporter', re.compile(r'receptor|channel|transporter|cotransporter', re.I)),
]
ORDER = [r[0] for r in RULES] + ['other_annotated', 'unannotated']


def classify(names, moa_by_pert, epi):
    out, sub = {}, {}
    inv = collections.defaultdict(set)
    for p, n in names.items():
        inv[(n or '').lower()].add(p)
    for label, ns in SUB_BY_NAME.items():
        for n in ns:
            for p in inv[n]:
                sub[p] = label
    for p in epi:
        if p not in sub and (names.get(p) or '').lower() not in SIRTUIN_OR_WEAK:
            sub.setdefault(p, 'hdac_inhibitor')
    for p in names:
        moas = moa_by_pert.get(p, set())
        if p in sub or p in epi:
            out[p] = 'epigenetic'
            continue
        cls = None
        for label, rx in RULES:
            if any(rx.search(m or '') for m in moas):
                cls = label
                break
        out[p] = cls or ('other_annotated' if moas else 'unannotated')
    for p, lab in list(sub.items()):
        if lab == 'other_epigenetic_by_name':
            del sub[p]
    return out, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    names = {r['pert_id']: r['pert_iname'] for r in csv.DictReader(open(DRUGS, encoding='utf-8'), delimiter='\t')}
    moa = collections.defaultdict(set)
    for r in csv.DictReader(open(CHEMBL, encoding='utf-8'), delimiter='\t'):
        moa[r['pert_id']].add(r['mechanism_of_action'])
    epi = set(json.load(open(EPI, encoding='utf-8')))
    cls, sub = classify(names, moa, epi)
    rec = {'rule': 'RESULTS 100.13 / 101: one class per compound, priority ' + ' > '.join(ORDER),
           'counts': dict(collections.Counter(cls.values())), 'sub_counts': dict(collections.Counter(sub.values())),
           'class': cls, 'sub': sub}
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rec[k] for k in ('counts', 'sub_counts')}, indent=1))


if __name__ == '__main__':
    main()
