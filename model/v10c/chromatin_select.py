"""v10-C L2: choose ChIP-Atlas experiments per (cell, axis) by the rule registered in RESULTS 100.1.

Input: the hg38 rows of ChIP-Atlas experimentList.tab (columns: SRX, genome, antigen class, antigen, cell type class,
cell type, cell type description, processing log 'reads,mapped%,dup%,peaks', title, then key=value metadata).
Output: {cell: {axis: [srx, ...]}} plus one provenance record per chosen experiment. Outcome-free: reads metadata only.
"""
import json, re, sys

AXES = ('ACC', 'ACT', 'PCG')
DIRECT = {'ACC': {'ATAC-Seq', 'DNase-seq'}, 'ACT': {'H3K27ac'}, 'PCG': {'H3K27me3'}}
SUBST = {'ACC': set(), 'ACT': {'H3K9ac', 'EP300', 'CREBBP'},
         'PCG': {'EZH2', 'SUZ12', 'EED', 'H2AK119ub', 'H2AK119ub1', 'RNF2'}}
K = 2
MIN_READS, MIN_MAPPED, MIN_PEAKS = 5_000_000, 50.0, 500

# Same-line names ChIP-Atlas uses (fixed before any signal was read; RESULTS 100.1). Primary cells match by cell type.
ALIASES = {'HEK293': ['293'], 'WA09': ['hESC H9', 'hESC WA09'], 'HME1': ['hTERT-HME1'], 'U266B1': ['U-266'],
           'PHH': ['Hepatocytes'], 'NPC': ['Neural progenitor cells']}
PRIMARY = {'PHH', 'NPC', 'HUVEC', 'CD34'}

CONTROL = re.compile(r'\b(dmso|vehicle|untreated|non-?treated|control|ctrl|ctl|mock|parental|wild[- ]?type|wt|etoh|'
                     r'ethanol|nodox|no dox|minus dox|scrambled?|non-?target(?:ing)?|ntc|sint|shnt|sictrl|shctrl|'
                     r'sgnt|sgctrl|sicon|shcon|empty vector|ev|none|normal|baseline|unstimulated|naive)\b', re.I)
PERTURB = re.compile(r'(treat|knock-?down|knock-?out|knock-?in|\bko\b|\bkd\b|\bki\b|\bsh[a-z0-9]|\bsi[a-z0-9]|\bsg[a-z0-9]|'
                     r'crispr|cas9|dcas|overexpress|\boe\b|inhibit|agonist|antagon|infect|virus|viral|transfect|'
                     r'transduc|mutant|mutation|\bdox\b|\+dox|doxycyclin|tamoxifen|4-?oht|fulvestrant|estrogen|estradiol|'
                     r'\be2\b|dht|androgen|enzalutamide|dexameth|\bdex\b|tnf|lps|ifn|interferon|tgf|\begf\b|stimulat|'
                     r'hypoxi|irradiat|radiation|heat shock|starv|serum[- ]free|resistan|\btamr\b|degrad|dtag|auxin|'
                     r'\biaa\b|\bpma\b|ionomycin|differentiat|senescen|\bdrug|compound|\benz(?:\b|\d)|enzalut|\bdel[a-z0-9]|'
                     r'deletion|\bdelta\b)', re.I)
PERTURB_KEYS = re.compile(r'^(treatment|agent|drug|compound|stimulated with|stimulus|stimulation|condition|transfection|'
                          r'infection|transduction|shrna|sirna|sgrna|sgguide|guide|genotype|genotype/variation|'
                          r'variation|knockdown|knockout|perturbation|treatment time|dose)$', re.I)
CELL_KEYS = {'source_name', 'cell line', 'cell type', 'cell types', 'cell_line', 'cell lines', 'sample type'}
ENCODE = re.compile(r'\b(ENCLB|ENCDO|ENCBS|ENCSR)|\bENCODE\b')
STANDARD_LIB = re.compile(r';\s*(ChIP-Seq|ATAC-seq|DNase-Hypersensitivity)\s*$', re.I)
UNINFORMATIVE = {'missing', 'not applicable', 'n/a', 'na', 'not collected', 'unknown', ''}


def norm(s):
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def cell_keys(cell):
    """The normalised ChIP-Atlas names that identify `cell` (exact, the NCI-H rule, and the alias table)."""
    k = {norm(cell)}
    m = re.match(r'^NCIH(\d+)$', cell)
    if m:
        k.add('H' + m.group(1))
    m = re.match(r'^H(\d{3,4})$', cell)
    if m:
        k.add('NCIH' + m.group(1))
    for a in ALIASES.get(cell, []):
        k.add(norm(a))
    return k


def axis_of(row):
    acl, ant = row[2], row[3]
    for ax in AXES:
        if acl in DIRECT[ax] or ant in DIRECT[ax]:
            return ax, 'direct'
        if ant in SUBST[ax]:
            return ax, 'substitute'
    return None, None


def qc(row):
    try:
        reads, mapped, _dup, peaks = row[7].split(',')
        ok = int(reads) >= MIN_READS and float(mapped) >= MIN_MAPPED and int(peaks) >= MIN_PEAKS
        return ok, int(peaks)
    except (ValueError, IndexError):
        return False, 0


def _perturbed_text(text):
    """A perturbation term left after the control vocabulary is removed (so 'untreated' or 'siCtrl' do not count)."""
    return bool(PERTURB.search(CONTROL.sub(' ', text)))


def baseline_tier(row):
    """'E' ENCODE reference, 'C' explicit control, 'N' neutral, or None (perturbed: excluded)."""
    title, meta = row[8], row[9:]
    perturbed = _perturbed_text(title)
    has_control = bool(CONTROL.search(title))
    for kv in meta:
        if '=' not in kv:
            continue
        key, val = (x.strip() for x in kv.split('=', 1))
        if PERTURB_KEYS.match(key):
            if val.lower() in UNINFORMATIVE:
                continue
            if CONTROL.search(val) and not _perturbed_text(val):
                has_control = True
            else:
                perturbed = True
        elif key.lower() in CELL_KEYS and _perturbed_text(val):
            perturbed = True
    if perturbed:
        return None
    if ENCODE.search(title) or any(ENCODE.search(kv) for kv in meta):
        return 'E'
    return 'C' if has_control else 'N'


TIER_RANK = {'E': 0, 'C': 1, 'N': 2}


def select(rows, cells):
    """rows: split hg38 rows; cells: iterable of LINCS cell names. Returns (selection, records)."""
    key2cell = {}
    for c in cells:
        for k in cell_keys(c):
            key2cell.setdefault(k, []).append(c)
    cand = {}
    for r in rows:
        if len(r) < 9 or r[1] != 'hg38':
            continue
        ax, kind = axis_of(r)
        if ax is None:
            continue
        cs = key2cell.get(norm(r[5]))
        if not cs:
            continue
        ok, peaks = qc(r)
        if not ok:
            continue
        tier = baseline_tier(r)
        if tier is None:
            continue
        rank = (0 if kind == 'direct' else 1, TIER_RANK[tier], 0 if STANDARD_LIB.search(r[8]) else 1, -peaks, r[0])
        rec = {'srx': r[0], 'antigen': r[3], 'antigen_class': r[2], 'kind': kind, 'tier': tier, 'peaks': peaks,
               'chipatlas_cell': r[5], 'title': r[8]}
        for c in cs:
            cand.setdefault((c, ax), []).append((rank, rec))
    selection, records = {}, []
    for (c, ax), lst in sorted(cand.items()):
        lst.sort(key=lambda t: t[0])
        chosen, seen = [], set()
        for _, rec in lst:
            if rec['srx'] not in seen:
                chosen.append(rec)
                seen.add(rec['srx'])
            if len(chosen) == K:
                break
        selection.setdefault(c, {})[ax] = [x['srx'] for x in chosen]
        for x in chosen:
            records.append(dict(x, cell=c, axis=ax, primary_celltype=c in PRIMARY, n_eligible=len(lst)))
    return selection, records


def main():
    tab, cells_json, out = sys.argv[1:4]
    cells = list(json.load(open(cells_json, encoding='utf-8'))['cells'])
    rows = [l.rstrip('\n').split('\t') for l in open(tab, encoding='utf-8')]
    selection, records = select(rows, cells)
    json.dump({'rule': 'RESULTS 100.1', 'K': K, 'selection': selection, 'records': records},
              open(out, 'w', encoding='utf-8'), indent=1)
    n = {ax: sum(ax in v for v in selection.values()) for ax in AXES}
    print('cells with any axis', len(selection), 'per axis', n, 'tracks', len(records))


if __name__ == '__main__':
    main()
