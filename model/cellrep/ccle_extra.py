"""CCLE inputs for the cell-representation programme, stage 2 (outcome-free; IDEAS E6 / E7; downloads approved 2026-11-02):
global chromatin profiling (42 histone-mark levels per line), miRNA profiling (734 miRNAs), and all-genes RPKM (GENCODE 19,
56,202 genes incl. non-coding), on the LINCS cells of v11's LINCS -> CCLE match (model/results/v11/tss_ccle_coverage.json).

    python -I model/cellrep/ccle_extra.py --out external/cellrep_data/ccle_extra.npz
"""
import argparse, csv, gzip, hashlib, json, os

import numpy as np

GLOBAL = 'external/dl_ccle_globalchrom/CCLE_GlobalChromatinProfiling_20181130.csv'
MIRNA = 'external/dl_ccle_mirna/CCLE_miRNA_20181103.gct'
RPKM = 'external/dl_ccle_rpkm/CCLE_RNAseq_genes_rpkm_20180929.gct.gz'
COV = 'model/results/v11/tss_ccle_coverage.json'


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def read_gct(lines, cols):
    """A GCT body (after the 2 header lines): returns (names, descriptions, matrix [genes, len(cols)]) for the CCLE columns."""
    head = next(lines).rstrip('\n').split('\t')
    pos = [head.index(c) if c in head else -1 for c in cols]
    names, desc, rows = [], [], []
    for line in lines:
        p = line.rstrip('\n').split('\t')
        names.append(p[0])
        desc.append(p[1])
        rows.append([float(p[k]) if k >= 0 and p[k] not in ('', 'NA', 'nan') else np.nan for k in pos])
    return np.array(names), np.array(desc), np.array(rows, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    match = json.load(open(COV, encoding='utf-8'))['ccle']['match']
    cells = sorted(c for c, v in match.items() if v)
    ccle = [match[c] for c in cells]

    with open(GLOBAL, encoding='utf-8') as f:
        r = csv.reader(f)
        head = next(r)
        marks = head[2:]
        by = {row[0]: row[2:] for row in r}
    G = np.array([[float(x) if x not in ('', 'NA') else np.nan for x in by[n]] if n in by else [np.nan] * len(marks)
                  for n in ccle], np.float32)

    with open(MIRNA, encoding='utf-8') as f:
        f.readline(), f.readline()
        mir_names, _, Mi = read_gct(iter(f), ccle)
    with gzip.open(RPKM, 'rt', encoding='utf-8') as f:
        f.readline(), f.readline()
        ens, sym, Rk = read_gct(iter(f), ccle)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, cells=np.array(cells), ccle_names=np.array(ccle), global_marks=np.array(marks),
                        global_chromatin=G, mirna_names=mir_names, mirna=Mi.T, rpkm_ensembl=ens, rpkm_symbol=sym,
                        log2_rpkm=np.log2(np.clip(Rk.T, 0, None) + 1.0).astype(np.float32))
    rec = {'cells': len(cells),
           'global_chromatin_cells': int(np.isfinite(G).any(1).sum()), 'global_marks': len(marks),
           'mirna_cells': int(np.isfinite(Mi.T).any(1).sum()), 'mirnas': len(mir_names),
           'rpkm_cells': int(np.isfinite(Rk.T).any(1).sum()), 'rpkm_genes': len(ens),
           'inputs': {os.path.basename(p): sha1(p) for p in (GLOBAL, MIRNA, RPKM, COV)}, 'out_sha1': sha1(a.out)}
    json.dump(rec, open(a.out.replace('.npz', '_record.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != 'inputs'}))


if __name__ == '__main__':
    main()
