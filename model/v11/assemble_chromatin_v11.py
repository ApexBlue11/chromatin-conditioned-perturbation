"""v11 genome-wide chromatin features (outcome-free): v10-C's encoding (chromatin_features.build: per-track rank over genes in
[0, 1], K-track mean per axis, 5 windows x ACC / ACT / PCG + CCLE promoter methylation, 4-channel mask), now over the STRING
node order (19,496 genes; the 18,658 with a MANE TSS carry data, the rest are masked).

    python model/v11/assemble_chromatin_v11.py --out external/v11_data/derived/v11_chromatin.npz
"""
import argparse, csv, glob, hashlib, json, os, sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'model', 'v10c'))
import chromatin_features as cf  # noqa: E402

SIGNAL = 'external/kaggle_out/v11_signal/v11_signal_s*.npz'
METHYL = 'external/kaggle_out/v11_signal/v11_methyl.npz'
SELECTION = 'model/results/v10c/chipatlas_selection.json'
CELLS = 'model/results/v10c/full_cells.json'
TSS = 'model/results/v11/tss_hg38_string.tsv'
GRAPH = 'network/outputs/v9/string_graph_v9.npz'


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.chdir(REPO)
    paths = sorted(glob.glob(SIGNAL))
    if len(paths) != 4:
        raise SystemExit(f'FATAL: expected 4 signal shards, found {paths}')
    for p in paths:
        k = p.split('_s')[-1].split('.')[0]
        rec = json.load(open(f'external/kaggle_out/v11_signal/V11_SIGNAL_COMPLETE_s{k}.json', encoding='utf-8'))
        if rec['npz_sha1'] != sha1(p) or rec['n_ok'] != rec['n_assigned']:
            raise SystemExit(f'FATAL: shard {k} is incomplete or its npz differs from its record')
    mrec = json.load(open('external/kaggle_out/v11_signal/V11_METHYL_COMPLETE.json', encoding='utf-8'))
    if mrec['npz_sha1'] != sha1(METHYL):
        raise SystemExit('FATAL: the methylation npz differs from its record')
    nodes = [str(x) for x in np.load(GRAPH, allow_pickle=True)['nodes']]
    sym2ent = {r['symbol']: r['entrez'] for r in csv.DictReader(open(TSS, encoding='utf-8'), delimiter='\t')}
    model_entrez = [sym2ent.get(g, f'none:{g}') for g in nodes]
    sel = json.load(open(SELECTION, encoding='utf-8'))['selection']
    cells = json.load(open(CELLS, encoding='utf-8'))['cells']
    meth = dict(np.load(METHYL, allow_pickle=True))
    F, M, names, used = cf.build(paths, sel, list(cells), TSS, model_entrez, methyl=meth)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, F=F.astype(np.float16), M=M, cells=np.array(list(cells)), genes=np.array(nodes),
                        entrez=np.array(model_entrez), feature_names=np.array(names))
    rec = {'rule': "v10-C's encoding over the STRING nodes (model/v11/assemble_chromatin_v11.py)",
           'cells': len(cells), 'genes': len(nodes), 'genes_with_tss': int(sum(not e.startswith('none:') for e in model_entrez)),
           'features': names, 'cells_with_axis': {ax: int(M[:, :, k].any(1).sum()) for k, ax in enumerate(('ACC', 'ACT', 'PCG', 'METH'))},
           'mean_gene_coverage_when_present': {ax: float(M[:, :, k][M[:, :, k].any(1)].mean()) for k, ax in enumerate(('ACC', 'ACT', 'PCG', 'METH'))},
           'inputs': {os.path.basename(p): sha1(p) for p in paths + [METHYL, SELECTION, CELLS, TSS, GRAPH]},
           'out_sha1': sha1(a.out)}
    json.dump(rec, open(a.out.replace('.npz', '_record.json'), 'w'), indent=1)
    print(json.dumps({k: rec[k] for k in ('cells', 'genes', 'genes_with_tss', 'cells_with_axis', 'mean_gene_coverage_when_present')}, indent=1))


if __name__ == '__main__':
    main()
