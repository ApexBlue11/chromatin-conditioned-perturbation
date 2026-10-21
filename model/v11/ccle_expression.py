"""CCLE baseline expression for the LINCS cells (outcome-free; v11 and T-E's expression control).
CCLE RSEM gene TPM (2018-09-29) -> log2(TPM + 1), genes mapped Ensembl gene -> Entrez through MANE Select (v1.5), placed on
v9's STRING node order (19,496; by Entrez via model/results/v11/tss_hg38_string.tsv) and on the 978 landmarks (by Entrez).
Cells: the LINCS -> CCLE match of model/results/v11/tss_ccle_coverage.json (131 of 217).

    python -I model/v11/ccle_expression.py --out external/v11_data/derived/ccle_expr_v11.npz
"""
import argparse, csv, gzip, hashlib, io, json, os

import numpy as np

TPM = 'external/v11_data/ccle/CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz'
MANE = 'external/v11_data/mane/MANE.GRCh38.v1.5.summary.txt.gz'
COV = 'model/results/v11/tss_ccle_coverage.json'
TSS = 'model/results/v11/tss_hg38_string.tsv'
GRAPH = 'network/outputs/v9/string_graph_v9.npz'
CHROM = 'external/v10c_chromatin.npz'


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
    txt = gzip.decompress(open(MANE, 'rb').read()).decode('utf-8', 'replace')
    ens2ent = {}
    for r in csv.DictReader(io.StringIO(txt), delimiter='\t'):
        if r.get('MANE_status') == 'MANE Select':
            ens2ent[r['Ensembl_Gene'].split('.')[0]] = r['#NCBI_GeneID'].replace('GeneID:', '')
    match = json.load(open(COV, encoding='utf-8'))['ccle']['match']
    cells = sorted(c for c, v in match.items() if v)
    with gzip.open(TPM, 'rt', encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        col = {c: head.index(match[c]) for c in cells}
        by_ent = {}
        for line in f:
            p = line.rstrip('\n').split('\t')
            ent = ens2ent.get(p[0].split('.')[0])
            if ent is None:
                continue
            by_ent[ent] = np.array([float(p[col[c]]) for c in cells], np.float64)
    nodes = [str(x) for x in np.load(GRAPH, allow_pickle=True)['nodes']]
    sym2ent = {r['symbol']: r['entrez'] for r in csv.DictReader(open(TSS, encoding='utf-8'), delimiter='\t')}
    lm_ent = [str(x) for x in np.load(CHROM, allow_pickle=True)['entrez']]

    def place(ents):
        V = np.full((len(cells), len(ents)), np.nan, np.float32)
        for j, e in enumerate(ents):
            if e is not None and e in by_ent:
                V[:, j] = np.log2(by_ent[e] + 1.0)
        return V

    Vn = place([sym2ent.get(g) for g in nodes])
    Vl = place(lm_ent)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, cells=np.array(cells), nodes=np.array(nodes), expr_nodes=Vn, landmark_entrez=np.array(lm_ent),
                        expr_landmarks=Vl)
    rec = {'cells': len(cells), 'nodes': len(nodes), 'nodes_with_expr': int(np.isfinite(Vn[0]).sum()),
           'landmarks_with_expr': int(np.isfinite(Vl[0]).sum()),
           'inputs': {os.path.basename(p): sha1(p) for p in (TPM, MANE, COV, TSS, GRAPH, CHROM)}, 'out_sha1': sha1(a.out)}
    json.dump(rec, open(a.out.replace('.npz', '_record.json'), 'w'), indent=1)
    print(json.dumps({k: rec[k] for k in ('cells', 'nodes', 'nodes_with_expr', 'landmarks_with_expr')}))


if __name__ == '__main__':
    main()
