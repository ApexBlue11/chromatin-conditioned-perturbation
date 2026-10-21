"""Mechanism classes, one per compound, fixed before any T-C output exists (RESULTS 100.13 / 101; re-frozen per review 093
C2). Outcome-free: built from ChEMBL mechanism text and name lists only.

Priority (a compound takes the first that matches):
  by name: bet_inactive_control (JQ1(-)), sirtuin_or_natural_product (outside "epigenetic")
  epigenetic > nuclear_receptor > kinase > dna_damage > cytoskeleton > proteostasis > receptor_channel_transporter >
  other_annotated ; no ChEMBL mechanism and no name match -> unannotated
"epigenetic" = the HDAC / DNMT / BET / histone-lysine methylation compounds only (the regex no longer matches COMT or CYP51).
Sub-labels for 101: dnmt_inhibitor, bet_inhibitor, bet_inactive_enantiomer, hdac_inhibitor; sensitivity sets:
  hdac_core = hdac_inhibitor minus HDAC6-selective compounds and valproate; bet_with_apabetalone = bet_inhibitor + apabetalone.

    python model/v10c/compound_classes.py --out model/results/v10c/compound_classes.json [--previous OLD.json --diff DIFF.json]
"""
import argparse, csv, json, re, collections

CHEMBL = 'drug/outputs/dti/chembl_dti_edges.tsv'
DRUGS = 'drug/outputs/drug_list.tsv'
EPI = 'drug/outputs/dti/epi_drug_pert_ids.json'

SUB_BY_NAME = {
    'dnmt_inhibitor': {'azacitidine', 'decitabine', 'zebularine'},
    'bet_inhibitor': {'jq1-(+)', 'i-bet-762', 'i-bet-151', 'pfi-1'},
    'bet_inactive_enantiomer': {'jq1-(-)'},
}
EPIGENETIC_BY_NAME = {'tranylcypromine', 'bix-01294', 'gsk-j4', 'chaetocin', 'gsk126', 'epz-005687', 'epz-5676', 'epz004777'}
SIRTUIN_OR_WEAK = {'ex-527', 'nicotinamide', 'resveratrol', 'sirtinol', 'salermide', 'tenovin-1', 'curcumin'}
HDAC_SENSITIVITY_OUT = {'tubastatin-a', 'acy-1215', 'valproic-acid'}       # HDAC6-selective; mM-range valproate
BET_SENSITIVITY_IN = {'apabetalone'}                                       # BD2-selective BET inhibitor
EPI_RX = re.compile(r'histone|bromodomain|DNA \(cytosine-5\)-methyltransferase|lysine[^;]*(demethylase|methyltransferase)', re.I)
HDAC_RX = re.compile(r'histone deacetylase', re.I)
RULES = [
    ('epigenetic', EPI_RX),
    ('nuclear_receptor', re.compile(r'(glucocorticoid|estrogen|androgen|progesterone|mineralocorticoid|retinoic acid|retinoid X|'
                                    r'peroxisome proliferator|vitamin D|thyroid hormone)[^;]* receptor', re.I)),
    ('kinase', re.compile(r'kinase|growth factor receptor[^;]*inhibitor|mTOR', re.I)),
    ('dna_damage', re.compile(r'^(?!.*(bacterial|virus|viral)).*(DNA inhibitor|topoisomerase|DNA polymerase|ribonucleotide '
                              r'reductase|thymidylate synthase|dihydrofolate reductase|DNA alkylat)', re.I)),
    ('cytoskeleton', re.compile(r'tubulin|kinesin', re.I)),
    ('proteostasis', re.compile(r'HSP90|proteasome|ubiquitin', re.I)),
    ('receptor_channel_transporter', re.compile(r'receptor|channel|transporter|cotransporter', re.I)),
]
ORDER = (['bet_inactive_control', 'sirtuin_or_natural_product'] + [r[0] for r in RULES] + ['other_annotated', 'unannotated'])


def classify(names, moa_by_pert, epi):
    inv = collections.defaultdict(set)
    for p, n in names.items():
        inv[(n or '').lower()].add(p)
    low = {p: (n or '').lower() for p, n in names.items()}
    sub = {}
    for label, ns in SUB_BY_NAME.items():
        for n in ns:
            for p in inv[n]:
                sub[p] = label
    for p in names:
        if p in sub:
            continue
        if low[p] in SIRTUIN_OR_WEAK:
            continue
        if (p in epi and low[p] not in SIRTUIN_OR_WEAK) or any(HDAC_RX.search(m or '') for m in moa_by_pert.get(p, ())):
            sub[p] = 'hdac_inhibitor'
    out = {}
    for p in names:
        moas = moa_by_pert.get(p, set())
        if sub.get(p) == 'bet_inactive_enantiomer':
            out[p] = 'bet_inactive_control'
        elif low[p] in SIRTUIN_OR_WEAK:
            out[p] = 'sirtuin_or_natural_product'
        elif p in sub or low[p] in EPIGENETIC_BY_NAME or low[p] in BET_SENSITIVITY_IN:
            out[p] = 'epigenetic'
        else:
            cls = None
            for label, rx in RULES:
                if any(rx.search(m or '') for m in moas):
                    cls = label
                    break
            out[p] = cls or ('other_annotated' if moas else 'unannotated')
    sens = {'hdac_core': sorted(p for p, s in sub.items() if s == 'hdac_inhibitor' and low[p] not in HDAC_SENSITIVITY_OUT),
            'bet_with_apabetalone': sorted([p for p, s in sub.items() if s == 'bet_inhibitor'] +
                                           [p for p in names if low[p] in BET_SENSITIVITY_IN])}
    return out, sub, sens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--previous')
    ap.add_argument('--diff')
    a = ap.parse_args()
    names = {r['pert_id']: r['pert_iname'] for r in csv.DictReader(open(DRUGS, encoding='utf-8'), delimiter='\t')}
    moa = collections.defaultdict(set)
    for r in csv.DictReader(open(CHEMBL, encoding='utf-8'), delimiter='\t'):
        moa[r['pert_id']].add(r['mechanism_of_action'])
    epi = set(json.load(open(EPI, encoding='utf-8')))
    cls, sub, sens = classify(names, moa, epi)
    rec = {'rule': 'RESULTS 100.13 / 101 (re-frozen per review 093 C2): one class per compound, priority ' + ' > '.join(ORDER),
           'counts': dict(collections.Counter(cls.values())), 'sub_counts': dict(collections.Counter(sub.values())),
           'class': cls, 'sub': sub, 'sensitivity_sets': sens}
    json.dump(rec, open(a.out, 'w', encoding='utf-8'), indent=1)
    if a.previous and a.diff:
        old = json.load(open(a.previous, encoding='utf-8'))
        d = {'class_changes': {p: [old['class'].get(p), c, names.get(p)] for p, c in cls.items() if old['class'].get(p) != c},
             'sub_changes': {p: [old['sub'].get(p), sub.get(p), names.get(p)] for p in set(old['sub']) | set(sub)
                             if old['sub'].get(p) != sub.get(p)}}
        json.dump(d, open(a.diff, 'w', encoding='utf-8'), indent=1)
        print('class changes', len(d['class_changes']), 'sub changes', len(d['sub_changes']))
    print(json.dumps({k: rec[k] for k in ('counts', 'sub_counts')}, indent=1))


if __name__ == '__main__':
    main()
