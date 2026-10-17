"""v10-C L3: per-gene window signal from a bigWig, and CCLE promoter methylation per gene (RESULTS 100.1).

Pure functions: the Kaggle data kernels call them with an open pyBigWig handle or the CCLE text; the tests call them with
fakes. Nothing here reads expression data.
"""
import csv, gzip, io, math, re

# (name, start offset, end offset) relative to the TSS, in the gene's own orientation (upstream negative).
WINDOWS = (('prom_1kb', -1_000, 1_000), ('tss_10kb', -10_000, 10_000), ('down_1_20kb', 1_000, 20_000),
           ('up_2_50kb', -50_000, -2_000), ('tss_50kb', -50_000, 50_000))


def load_tss(path):
    """[(symbol, chrom, tss, strand)] in file order (the 978 landmark genes, MANE Select, hg38)."""
    out = []
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            out.append((row['symbol'], row['chrom'], int(row['tss_hg38']), row['strand']))
    return out


def window_bounds(tss, strand, lo, hi, chrom_len):
    """Genomic [start, end) of a TSS-relative window; strand '-' mirrors it. None if empty after clamping."""
    if strand == '-':
        lo, hi = -hi, -lo
    s, e = max(0, tss + lo), min(chrom_len, tss + hi)
    return (s, e) if e > s else None


def window_means(bw, genes, windows=WINDOWS):
    """Mean signal per (gene, window). Within a chromosome the bigWig covers, no data means 0 (ChIP-Atlas omits zero
    runs); a chromosome absent from the file, or a window empty after clamping, is NaN."""
    chroms = bw.chroms()
    vals = [[math.nan] * len(windows) for _ in genes]
    absent = set()
    for i, (_sym, chrom, tss, strand) in enumerate(genes):
        if chrom not in chroms:
            absent.add(chrom)
            continue
        for j, (_n, lo, hi) in enumerate(windows):
            b = window_bounds(tss, strand, lo, hi, chroms[chrom])
            if b is None:
                continue
            v = bw.stats(chrom, b[0], b[1], type='mean', exact=True)[0]
            vals[i][j] = 0.0 if v is None else float(v)
    return vals, {'absent_chroms': sorted(absent), 'n_nan': sum(math.isnan(x) for r in vals for x in r)}


def norm_name(s):
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def name_keys(cell):
    """A LINCS name and its NCI-H twin (H1975 <-> NCIH1975), as in chromatin_select.cell_keys."""
    k = {norm_name(cell)}
    m = re.match(r'^NCIH(\d+)$', cell)
    if m:
        k.add('H' + m.group(1))
    m = re.match(r'^H(\d{3,4})$', cell)
    if m:
        k.add('NCIH' + m.group(1))
    return k


CCLE_META = ('tss_id', 'gene', 'chr', 'fpos', 'tpos', 'strand', 'avg_coverage')


def ccle_methyl(text, symbols, cells):
    """CCLE RRBS TSS +/- 1 kb table -> {cell: [mean methylation per symbol (NaN if absent)]} for the cells whose CCLE name
    (the part before the first '_', normalised) equals a cell's normalised name. Multiple TSS rows of one gene are
    averaged; 'NA' / empty / non-numeric entries are skipped."""
    rd = csv.reader(io.StringIO(text), delimiter='\t')
    header = next(rd)
    low = [h.strip().lower() for h in header]
    if 'gene' not in low:
        raise ValueError(f'no gene column in {header[:10]}')
    gcol = low.index('gene')
    meta_idx = {i for i, h in enumerate(low) if h in CCLE_META}
    want = {}
    for c in cells:
        for key in name_keys(c):
            want.setdefault(key, []).append(c)
    colcell, taken = {}, set()
    for i, h in enumerate(header):
        if i in meta_idx:
            continue
        cs = [c for c in want.get(norm_name(h.split('_', 1)[0]), []) if c not in taken]
        if cs:
            colcell[i] = cs
            taken.update(cs)
    sym_idx = {s: k for k, s in enumerate(symbols)}
    acc = {c: [[0.0, 0] for _ in symbols] for cs in colcell.values() for c in cs}
    for row in rd:
        if len(row) <= gcol:
            continue
        k = sym_idx.get(row[gcol].strip())
        if k is None:
            continue
        for i, cs in colcell.items():
            if i >= len(row):
                continue
            try:
                v = float(row[i])
            except ValueError:
                continue
            if math.isnan(v):
                continue
            for c in cs:
                acc[c][k][0] += v
                acc[c][k][1] += 1
    out = {c: [s / n if n else math.nan for s, n in a] for c, a in acc.items()}
    return out, {'n_cell_columns': len(header) - len(meta_idx), 'matched': sorted(out)}


def read_gz_text(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        return f.read()
