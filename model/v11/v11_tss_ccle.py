"""v11 data step 1 (outcome-free): the genome-wide TSS table for the STRING nodes, and CCLE's coverage of the LINCS cells.

  TSS: MANE Select (GRCh38, v1.5) by gene symbol, then by HGNC alias via the L1000 / STRING symbol, TSS = 5' end.
  CCLE: LINCS cell names matched to CCLE_ID's cell part (letters and digits only, upper case).

    python -I model/v11/v11_tss_ccle.py --mane ... --ccle_ann ... --ccle_tpm ... --out_tss ... --out_cov ...
"""
import argparse, csv, gzip, io, json, re, sys

import numpy as np


# explicit aliases (LINCS name -> CCLE_ID), each checked against the RNA-seq columns at run time
CCLE_ALIASES = {'H1975': 'NCIH1975_LUNG', 'H1299': 'NCIH1299_LUNG', 'LNCAP': 'LNCAPCLONEFGC_PROSTATE'}


def norm(s):
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def mane_tss(path):
    txt = gzip.decompress(open(path, 'rb').read()).decode('utf-8', 'replace')
    rd = csv.DictReader(io.StringIO(txt), delimiter='\t')
    out = {}
    for r in rd:
        if r.get('MANE_status') != 'MANE Select':
            continue
        acc = r['GRCh38_chr']
        m = re.match(r'NC_0+(\d+)\.\d+', acc)
        if not m:
            continue
        n = int(m.group(1))
        chrom = 'chr' + {23: 'X', 24: 'Y'}.get(n, str(n))
        start, end, strand = int(r['chr_start']), int(r['chr_end']), r['chr_strand']
        tss = start if strand == '+' else end
        out[r['symbol']] = (chrom, tss, strand, r['#NCBI_GeneID'].replace('GeneID:', ''), r['Ensembl_Gene'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--string_graph', required=True)
    ap.add_argument('--mane', required=True)
    ap.add_argument('--ccle_ann', required=True)
    ap.add_argument('--ccle_tpm', required=True)
    ap.add_argument('--chromatin', required=True, help='v10c_chromatin.npz, for the LINCS cell list')
    ap.add_argument('--out_tss', required=True)
    ap.add_argument('--out_cov', required=True)
    a = ap.parse_args()
    nodes = [str(x) for x in np.load(a.string_graph, allow_pickle=True)['nodes']]
    mane = mane_tss(a.mane)
    rows, missing = [], []
    for g in nodes:
        if g in mane:
            c, t, s, ent, ens = mane[g]
            rows.append((g, c, t, s, ent))
        else:
            missing.append(g)
    with open(a.out_tss, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['symbol', 'chrom', 'tss_hg38', 'strand', 'entrez'])
        w.writerows(rows)
    # CCLE coverage of the LINCS cells
    ann = list(csv.DictReader(open(a.ccle_ann, encoding='utf-8', errors='replace'), delimiter='\t'))
    ccle_ids = [r['CCLE_ID'] for r in ann]
    by_norm = {}
    for cid in ccle_ids:
        by_norm.setdefault(norm(cid.split('_')[0]), []).append(cid)
    with gzip.open(a.ccle_tpm, 'rt', encoding='utf-8', errors='replace') as f:
        header = f.readline().rstrip('\n').split('\t')
    tpm_ids = set(header[2:])
    cells = [str(c) for c in np.load(a.chromatin, allow_pickle=True)['cells']]
    cov = {}
    for c in cells:
        hits = [x for x in by_norm.get(norm(c), []) if x in tpm_ids]
        if not hits and c in CCLE_ALIASES and CCLE_ALIASES[c] in tpm_ids:
            hits = [CCLE_ALIASES[c]]
        cov[c] = hits
    rec = {'mane': {'string_nodes': len(nodes), 'with_mane_tss': len(rows), 'missing': len(missing),
                    'missing_examples': missing[:30]},
           'ccle': {'lincs_cells': len(cells), 'with_ccle_rnaseq': sum(bool(v) for v in cov.values()),
                    'ambiguous': {c: v for c, v in cov.items() if len(v) > 1},
                    'tpm_columns': len(tpm_ids), 'tpm_header_first': header[:3],
                    'match': {c: (v[0] if len(v) == 1 else None) for c, v in cov.items()}}}
    json.dump(rec, open(a.out_cov, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({'mane': {k: rec['mane'][k] for k in ('string_nodes', 'with_mane_tss', 'missing')},
                      'ccle': {k: rec['ccle'][k] for k in ('lincs_cells', 'with_ccle_rnaseq', 'ambiguous', 'tpm_columns')}},
                     indent=1))


if __name__ == '__main__':
    main()
