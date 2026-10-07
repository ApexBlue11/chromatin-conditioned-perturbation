"""chromatin_features93  –  §93 chromatin accessibility features for LINCS v9.

Rewritten against verified package APIs (PROBE FACTS, 2026-10-07).
"""

import sys
import os
import json
import re
import csv
import gzip
import argparse
import hashlib
import math
import datetime
import numpy as np
import time
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)        # chromatin_funnel (rank_normal) is imported from beside this file, here and on Kaggle
PACKAGES = ('MOODS-python', 'py2bit', 'pyjaspar', 'pyranges', 'decoupler', 'numpy', 'pandas', 'scipy')

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def fetch_to_file(url, cache_dir):
    """Download *url* to ``cache_dir/<sha1(url)>`` in 1 MB chunks.

    Reuses an existing file.  Writes to a ``.part`` file before renaming.
    Returns the path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, h)
    if os.path.exists(cache_path):
        return cache_path
    part_path = cache_path + '.part'
    req = urllib.request.Request(url, headers={"User-Agent": "lincs"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part_path, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
    os.replace(part_path, cache_path)
    return cache_path


def fetch(url, cache_dir):
    """Download *url* (via :func:`fetch_to_file`) and return its bytes."""
    path = fetch_to_file(url, cache_dir)
    with open(path, 'rb') as f:
        return f.read()


def sha1_file(path):
    """Return the hex SHA-1 of a file, read in 1 MB chunks."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ENCODE file lookup (network)
# ---------------------------------------------------------------------------

def get_encode_files(acc, fetch_fn, cache_dir):
    q = (f"https://www.encodeproject.org/search/?type=File&dataset=/experiments/{acc}/"
         f"&file_format=bed&file_format_type=narrowPeak&assembly=GRCh38&status=released"
         f"&output_type=replicated+peaks&output_type=peaks&output_type=pseudoreplicated+peaks"
         f"&format=json&field=accession&limit=3")
    try:
        raw = fetch_fn(q, cache_dir)
        d = json.loads(raw.decode("utf-8"))
        accs = [g["accession"] for g in d.get("@graph", [])][:1]
        return [f"https://www.encodeproject.org/files/{a}/@@download/{a}.bed.gz" for a in accs]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sample selection  (unchanged – matches step10)
# ---------------------------------------------------------------------------

def select_atac_samples(cov_tsv, cistrome_json, gsm2srx, encode_files_for, log_cells, cache_dir):
    import pandas as pd
    cov = pd.read_csv(cov_tsv, sep="\t", dtype=str, keep_default_na=False)
    with open(cistrome_json) as f:
        S = json.load(f)
    id2gsm = {str(s["id"]): s.get("external_id") for s in S if s.get("external_id_type") == "GEO"}

    out = {}
    resolved = cov[(cov.status == "resolved") & (cov.assay_target == "ATAC-seq")]
    for _, r in resolved.iterrows():
        cell = r.lincs_cell_id
        if cell not in log_cells:
            continue
        src = r.source_used
        urls = []
        if src == "encode":
            for acc in re.findall(r"ENCSR\w+", r.notes)[:4]:
                urls.extend(encode_files_for(acc, cache_dir))
        elif src == "cistrome":
            sids = [x for x in re.split(r"[,\s]+", r.sample_ids_used) if x.isdigit()]
            srxs = [gsm2srx.get(id2gsm.get(s)) for s in sids]
            srxs = [x for x in srxs if x][:6]
            for srx in srxs:
                urls.append(f"https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/{srx}.05.bed")
        out[cell] = urls
    return out


# ---------------------------------------------------------------------------
# BED parsing  (unchanged – matches step10)
# ---------------------------------------------------------------------------

def parse_bed(raw):
    """narrowPeak/broadPeak (ENCODE + ChIP-Atlas): signalValue = col 7 (index 6), fallback col 5.
    return {chrom: (starts_array, records[(start,end,sig)])} sorted by start."""
    txt = gzip.decompress(raw).decode("utf-8", "replace") if raw[:2] == b"\x1f\x8b" else raw.decode("utf-8", "replace")
    by = defaultdict(list)
    for ln in txt.splitlines():
        if not ln or ln[0] == "#": continue
        f = ln.split("\t")
        if len(f) < 5: continue
        try:
            st, en = int(f[1]), int(f[2])
            sig = float(f[6]) if (len(f) > 6 and f[6] not in ("", ".")) else float(f[4])
        except:
            continue
        by[f[0]].append((st, en, sig))
    out = {}
    for c, ivs in by.items():
        ivs.sort()
        out[c] = ([x[0] for x in ivs], ivs)
    return out


# ---------------------------------------------------------------------------
# Interval merging
# ---------------------------------------------------------------------------

def merge_intervals_reference(df):
    """Pure-Python merge: overlapping or book-ended intervals merge
    (``next.Start <= cur.End``), a 1-bp gap does not.  This is bedtools
    ``merge -d 0``.

    Takes no slack argument.
    """
    import pandas as pd
    records = []
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Chromosome", "Start", "End"])
    for chrom, group in df.groupby("Chromosome"):
        ivs = sorted(zip(group["Start"].tolist(), group["End"].tolist()))
        if not ivs:
            continue
        merged = [list(ivs[0])]
        for st, en in ivs[1:]:
            if st <= merged[-1][1]:          # overlapping or book-ended
                merged[-1][1] = max(merged[-1][1], en)
            else:
                merged.append([st, en])
        for st, en in merged:
            records.append({"Chromosome": chrom, "Start": st, "End": en})
    if not records:
        return pd.DataFrame(columns=["Chromosome", "Start", "End"])
    out = pd.DataFrame(records)
    out["Start"] = out["Start"].astype(np.int64)
    out["End"] = out["End"].astype(np.int64)
    return out.sort_values(["Chromosome", "Start"]).reset_index(drop=True)


def merge_peaks(df):
    """Merge intervals using pyranges (slack=0, the default).

    Returns a DataFrame with Chromosome, Start, End as int64, sorted.
    **No try/except fallback.**
    """
    import pyranges as pr
    merged = pr.PyRanges(df).merge().df
    merged["Start"] = merged["Start"].astype(np.int64)
    merged["End"] = merged["End"].astype(np.int64)
    return merged.sort_values(["Chromosome", "Start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Peak windows  (unchanged)
# ---------------------------------------------------------------------------

def compute_windows(merged_df, tss_dict, canon_genes):
    peaks_by_chrom = {}
    if merged_df is not None and len(merged_df) > 0:
        for chrom, group in merged_df.groupby("Chromosome"):
            peaks_by_chrom[chrom] = group[["Start", "End"]].values

    F_prom = np.zeros(len(canon_genes), dtype=np.float32)
    F_enh = np.zeros(len(canon_genes), dtype=np.float32)

    for gi, g in enumerate(canon_genes):
        if g not in tss_dict:
            continue
        chrom, tss = tss_dict[g]
        if chrom not in peaks_by_chrom:
            continue
        ivs = peaks_by_chrom[chrom]

        p_start, p_end = tss - 1000, tss + 1000
        prom_overlap_mask = (ivs[:, 0] < p_end) & (ivs[:, 1] > p_start)
        prom_ivs = ivs[prom_overlap_mask]
        if len(prom_ivs) > 0:
            overlaps = np.minimum(prom_ivs[:, 1], p_end) - np.maximum(prom_ivs[:, 0], p_start)
            F_prom[gi] = overlaps.sum() / 2000.0

        midpoints = (ivs[:, 0] + ivs[:, 1]) / 2.0
        d = np.abs(midpoints - tss)

        enh_mask = (d >= 1000) & (d <= 50000) & (~prom_overlap_mask)
        enh_ivs = ivs[enh_mask]
        d_enh = d[enh_mask]
        if len(enh_ivs) > 0:
            lengths = enh_ivs[:, 1] - enh_ivs[:, 0]
            val = (lengths / 1000.0) * np.exp(-d_enh / 10000.0)
            F_enh[gi] = val.sum()

    return F_prom, F_enh


# ---------------------------------------------------------------------------
# Near-landmark filtering  (unchanged)
# ---------------------------------------------------------------------------

def near_landmark_peaks(merged_df, tss_dict, max_d=50000):
    if merged_df is None or len(merged_df) == 0:
        return []
    tss_by_chrom = {}
    for c, t in tss_dict.values():
        tss_by_chrom.setdefault(c, []).append(t)

    filtered_peaks = []
    for chrom, group in merged_df.groupby("Chromosome"):
        if chrom not in tss_by_chrom: continue
        ivs = group[["Start", "End"]].values
        tsses = np.array(tss_by_chrom[chrom])
        tsses.sort()
        midpoints = (ivs[:, 0] + ivs[:, 1]) / 2.0
        idx = np.searchsorted(tsses, midpoints)
        idx = np.clip(idx, 0, len(tsses)-1)
        idx_prev = np.clip(idx-1, 0, len(tsses)-1)
        d = np.minimum(np.abs(tsses[idx] - midpoints), np.abs(tsses[idx_prev] - midpoints))
        mask = d <= max_d
        for st, en in ivs[mask]:
            filtered_peaks.append((chrom, int(st), int(en)))
    return filtered_peaks


# ---------------------------------------------------------------------------
# Motif scanning  (MOODS, verified API)
# ---------------------------------------------------------------------------

def build_matrices(motifs, bg, ps=0.01):
    """Build log-odds matrices from pyjaspar motif objects.

    Returns ``(fw_list, rc_list)`` where each is a list of log-odds matrices
    (one per motif, in *motifs* order).
    """
    import MOODS.tools
    fw_list, rc_list = [], []
    for m in motifs:
        c = m.counts
        counts = [list(map(float, c['A'])), list(map(float, c['C'])),
                  list(map(float, c['G'])), list(map(float, c['T']))]
        fw = MOODS.tools.log_odds(counts, bg, ps)
        rc = MOODS.tools.reverse_complement(fw, 4)
        fw_list.append(fw)
        rc_list.append(rc)
    return fw_list, rc_list


def make_scanner(fw_list, rc_list, bg, p=1e-4):
    """Build a ``MOODS.scan.Scanner`` for the given matrices.

    The scanner is built **once** and reused for every sequence.
    """
    import MOODS.tools
    import MOODS.scan
    matrices = fw_list + rc_list
    thresholds = [MOODS.tools.threshold_from_p(m, bg, p) for m in matrices]
    scanner = MOODS.scan.Scanner(7)
    scanner.set_motifs(matrices, bg, thresholds)
    return scanner


def motif_hits(scanner, n_motifs, seqs):
    """Scan *seqs* and return ``bool[n_seqs, n_motifs]``.

    Motif *t* is hit iff ``results[t]`` (forward) or ``results[n_motifs + t]``
    (reverse complement) is non-empty.
    """
    out = np.zeros((len(seqs), n_motifs), dtype=bool)
    for si, seq in enumerate(seqs):
        results = scanner.scan(seq.upper())
        for t in range(n_motifs):
            if len(results[t]) > 0 or len(results[n_motifs + t]) > 0:
                out[si, t] = True
    return out


_WORKER = {}


def _scan_init(fw_list, rc_list, bg):
    _WORKER['scanner'] = make_scanner(fw_list, rc_list, bg, p=1e-4)
    _WORKER['n'] = len(fw_list)


def _scan_chunk(seqs):
    return motif_hits(_WORKER['scanner'], _WORKER['n'], seqs)


def make_parallel_scan_fn(motifs, bg, threads):
    """scan_fn(seqs) -> bool[n_seqs, n_motifs] with MOODS. threads > 1: a Pool whose workers each build their own scanner
    (Pool initializer); threads == 1: in-process."""
    fw_list, rc_list = build_matrices(motifs, bg, ps=0.01)
    n = len(fw_list)
    if threads <= 1:
        scanner = make_scanner(fw_list, rc_list, bg, p=1e-4)
        return lambda seqs: motif_hits(scanner, n, seqs)
    import multiprocessing as mp
    fw_py = [[list(r) for r in m] for m in fw_list]
    rc_py = [[list(r) for r in m] for m in rc_list]
    pool = mp.get_context('fork').Pool(threads, initializer=_scan_init, initargs=(fw_py, rc_py, tuple(bg)))

    def scan(seqs):
        if not seqs:
            return np.zeros((0, n), bool)
        k = max(1, int(np.ceil(len(seqs) / (threads * 4))))
        parts = pool.map(_scan_chunk, [seqs[i:i + k] for i in range(0, len(seqs), k)])
        return np.concatenate(parts, axis=0)
    return scan


def bench_scan(motifs, bg, n_bp=200000, seed=0):
    """Seconds to scan n_bp of random sequence with all motifs in one process (printed before the real run)."""
    fw_list, rc_list = build_matrices(motifs, bg, ps=0.01)
    scanner = make_scanner(fw_list, rc_list, bg, p=1e-4)
    rng = np.random.default_rng(seed)
    seqs = [''.join(rng.choice(list('ACGT'), size=500)) for _ in range(n_bp // 500)]
    t = time.time()
    motif_hits(scanner, len(fw_list), seqs)
    return time.time() - t


def motif_accessibility(hits):
    """Return ``hits.mean(0)`` (share of peaks carrying each motif), or
    zeros if there are no sequences."""
    if hits.shape[0] == 0:
        return np.zeros(hits.shape[1] if hits.ndim > 1 else 0, dtype=np.float32)
    return hits.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Standardisation  (unchanged)
# ---------------------------------------------------------------------------

def standardise_across_cells(raw_MA_mat, has_mat):
    MA_std = np.zeros_like(raw_MA_mat)
    n_motifs = raw_MA_mat.shape[1]
    for t in range(n_motifs):
        col = raw_MA_mat[has_mat, t]
        if len(col) > 1:
            mean = np.mean(col)
            sd = np.std(col, ddof=1)
            if sd > 0:
                MA_std[has_mat, t] = (col - mean) / sd
    return MA_std


# ---------------------------------------------------------------------------
# Regulon scores  (unchanged)
# ---------------------------------------------------------------------------

def regulon_scores(net_df, MA_std_row, motif_map, canon_genes):
    F_reg = np.zeros(len(canon_genes), dtype=np.float32)
    reg_grouped = net_df.groupby("target")
    for gi, g in enumerate(canon_genes):
        if g not in reg_grouped.groups:
            continue
        g_regs = reg_grouped.get_group(g)
        vals = []
        for _, row in g_regs.iterrows():
            tf = row["source"].upper()
            w = np.sign(row["weight"])
            if tf in motif_map:
                tf_ma = np.mean([MA_std_row[m_i] for m_i in motif_map[tf]])
                vals.append(w * tf_ma)
        if vals:
            F_reg[gi] = np.mean(vals)
    return F_reg


# ---------------------------------------------------------------------------
# Per-cell feature computation
# ---------------------------------------------------------------------------

def cell_features(beds, tss_dict, canon, genome, scan_fn, merge_fn):
    """Compute features for one cell.

    Parameters
    ----------
    beds : list of parsed-bed dicts (from :func:`parse_bed`)
    tss_dict : {gene: (chrom, tss_pos)}
    canon : list of 978 gene symbols
    genome : object with ``.chroms()``, ``.sequence(chrom, start, end)``
    scan_fn : callable(seqs) → bool[n_seqs, n_motifs]
    merge_fn : callable(df) → merged DataFrame

    Returns
    -------
    dict with F_prom, F_enh, MA_raw, has, counts, and per-file gene sets.
    """
    import pandas as pd

    result = {
        'F_prom': np.zeros(len(canon), dtype=np.float32),
        'F_enh': np.zeros(len(canon), dtype=np.float32),
        'MA_raw': None,   # set below
        'has': False,
        'raw_intervals': 0,
        'merged': 0,
        'near_landmarks': 0,
        'dropped_chrom': 0,
        'genes_with_prom': 0,
        'per_file_genes': [],
    }

    if not beds:
        # No records at all – need n_motifs from scan_fn on empty list
        dummy = scan_fn([])
        n_motifs = dummy.shape[1] if dummy.ndim > 1 else 0
        result['MA_raw'] = np.zeros(n_motifs, dtype=np.float32)
        return result

    # Collect intervals from all beds; also compute per-file gene sets
    records = []
    for bed in beds:
        # Build a small merged df for this file alone, compute windows, record
        # which genes have F_prom > 0
        file_records = []
        for chrom, (starts, ivs) in bed.items():
            for st, en, sig in ivs:
                file_records.append({"Chromosome": chrom, "Start": st, "End": en})
                records.append({"Chromosome": chrom, "Start": st, "End": en})
        if file_records:
            fdf = pd.DataFrame(file_records)
            fdf["Start"] = fdf["Start"].astype(np.int64)
            fdf["End"] = fdf["End"].astype(np.int64)
            f_merged = merge_fn(fdf)
            fp, _ = compute_windows(f_merged, tss_dict, canon)
            genes_in_file = set(np.array(canon)[fp > 0])
        else:
            genes_in_file = set()
        result['per_file_genes'].append(genes_in_file)

    result['raw_intervals'] = len(records)

    if not records:
        dummy = scan_fn([])
        n_motifs = dummy.shape[1] if dummy.ndim > 1 else 0
        result['MA_raw'] = np.zeros(n_motifs, dtype=np.float32)
        return result

    df = pd.DataFrame(records)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    merged = merge_fn(df)
    result['merged'] = len(merged)

    F_prom, F_enh = compute_windows(merged, tss_dict, canon)
    result['F_prom'] = F_prom
    result['F_enh'] = F_enh
    result['genes_with_prom'] = int((F_prom > 0).sum())

    if not (F_prom > 0).any():
        dummy = scan_fn([])
        n_motifs = dummy.shape[1] if dummy.ndim > 1 else 0
        result['MA_raw'] = np.zeros(n_motifs, dtype=np.float32)
        return result

    result['has'] = True

    # Filter to near-landmark peaks
    filtered_peaks = near_landmark_peaks(merged, tss_dict, max_d=50000)

    # Extract sequences from the genome
    chroms_in_genome = genome.chroms()
    seqs = []
    dropped = 0
    for chrom, st, en in filtered_peaks:
        if chrom not in chroms_in_genome:
            dropped += 1
            continue
        L = chroms_in_genome[chrom]
        sc = max(0, st)
        ec = min(L, en)
        if ec <= sc:
            continue
        seqs.append(genome.sequence(chrom, sc, ec).upper())

    result['near_landmarks'] = len(seqs)
    result['dropped_chrom'] = dropped

    # Scan for motif hits
    hits = scan_fn(seqs)
    result['MA_raw'] = motif_accessibility(hits)
    return result


# ---------------------------------------------------------------------------
# Genome background
# ---------------------------------------------------------------------------

def genome_background(tb):
    """Compute the genome-wide base composition as 4 fractions (A, C, G, T).

    Sums ``tb.bases(chrom, 0, length, False)`` over every chromosome.
    """
    totals = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for chrom, length in tb.chroms().items():
        counts = tb.bases(chrom, 0, length, False)
        for b in 'ACGT':
            totals[b] += counts[b]
    total = sum(totals.values())
    return tuple(totals[b] / total for b in 'ACGT')


# ---------------------------------------------------------------------------
# TF → motif map
# ---------------------------------------------------------------------------

def build_motif_map(motifs):
    """Build TF → [motif indices] from JASPAR motif names.

    Names are split on ``::`` (case-insensitively), so a dimer counts for
    both partners.
    """
    motif_map = {}
    for i, m in enumerate(motifs):
        for part in re.split(r"::", m.name):
            motif_map.setdefault(part.strip().upper(), []).append(i)
    return motif_map


# ---------------------------------------------------------------------------
# Split-half report
# ---------------------------------------------------------------------------

def _corr_pair(a, b):
    from scipy.stats import pearsonr, spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan'), float('nan')
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def split_half_report(arrays, halves):
    """93.8: split-half reliability of each cell's deviation, reported against the mismatched-cell baseline. No gate.

    For F in (F_prom, F_enh, F_reg): rank_normal per (cell, feature) over all genes; dev_h[c] = F_h[c] - mean over the other
    kept cells with has of rank_normal(F_full[c']); correlations over genes with a TSS (arrays['has_tss']).
    For MA: raw MA halves, deviation from the other has-cells' full-union MA_raw, correlations over motifs.
    Baseline for cell c: the median over c' != c (cells with halves) of corr(dev_A[c], dev_B[c']). Excess = own - baseline."""
    from chromatin_funnel import rank_normal
    cells = [str(c) for c in arrays['cells']]
    has = np.asarray(arrays['has'], bool)
    idx = {c: i for i, c in enumerate(cells)}
    hc = [c for c in halves['cells_with_halves'] if has[idx[c]]]
    report = {'cells_with_halves': len(hc), 'per_cell': {c: {} for c in hc}, 'medians': {},
              'rule': '93.8 with the mismatched-cell baseline; reported, never read'}
    if not hc:
        return report
    tss = np.asarray(arrays['has_tss'], bool)

    def other_mean(F_full, c, transform):
        rows = [transform(F_full[j]) for j, c2 in enumerate(cells) if has[j] and c2 != c]
        return np.mean(rows, axis=0) if rows else np.zeros(F_full.shape[1])

    blocks = [(f, tss, lambda v: rank_normal(np.asarray(v, np.float64))) for f in ('F_prom', 'F_enh', 'F_reg')]
    blocks.append(('MA', np.ones(arrays['MA_raw'].shape[1], bool), lambda v: np.asarray(v, np.float64)))
    for fname, mask, tf in blocks:
        full = arrays['MA_raw'] if fname == 'MA' else arrays[fname]
        key = 'MA_raw' if fname == 'MA' else fname
        dev = {}
        for c in hc:
            m = other_mean(full, c, tf)
            dev[c] = {h: (tf(halves[c]['%s_%s' % (key, h)]) - m)[mask] for h in 'AB'}
        stats = {k: [] for k in ('pearson_own', 'spearman_own', 'pearson_baseline', 'spearman_baseline', 'pearson_excess',
                                 'spearman_excess')}
        for c in hc:
            own = _corr_pair(dev[c]['A'], dev[c]['B'])
            mm = [_corr_pair(dev[c]['A'], dev[c2]['B']) for c2 in hc if c2 != c]
            base = (float(np.nanmedian([x[0] for x in mm])) if mm else float('nan'),
                    float(np.nanmedian([x[1] for x in mm])) if mm else float('nan'))
            row = {'pearson_own': own[0], 'spearman_own': own[1], 'pearson_baseline': base[0], 'spearman_baseline': base[1],
                   'pearson_excess': own[0] - base[0], 'spearman_excess': own[1] - base[1], 'n': int(mask.sum())}
            report['per_cell'][c][fname] = row
            for k in stats:
                stats[k].append(row[k])
        report['medians'][fname] = {k: float(np.nanmedian(v)) for k, v in stats.items()}
    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(cfg, fetch_fn=None, open_genome=None, load_motifs=None,
                   load_net=None, scan_factory=None, merge_fn=None):
    """Build all §93 chromatin features.

    The ``None`` defaults resolve (inside this function) to the real
    py2bit.open, pyjaspar, decoupler, a MOODS scanner and merge_peaks,
    with lazy imports.

    Tests inject fakes for all of these.

    Returns ``(arrays, manifest, halves)``.
    """
    import pandas as pd

    if fetch_fn is None:
        fetch_fn = fetch
    if merge_fn is None:
        merge_fn = merge_peaks

    # ---- Load cell index ----
    with open(cfg['cell_index']) as f:
        cell_index_data = json.load(f)
        cell_index = cell_index_data.get("cell_id_to_row", cell_index_data)
    cells_order = [c for c, _ in sorted(cell_index.items(), key=lambda kv: kv[1])]

    # ---- Parse peaks log ----
    log_cells = {}
    with open(cfg['peaks_log'], encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^([^/]+)/ATAC-seq src=[a-z]+: (\d+) bed track", line)
            if m:
                c = m.group(1)
                t = int(m.group(2))
                if t > 0:
                    log_cells[c] = t

    # ---- Build GSM→SRX ----
    gsm2srx = {}
    with open(cfg['chipatlas_list'], encoding="utf-8") as f:
        for l in f:
            p = l.rstrip("\n").split("\t")
            if len(p) >= 8:
                m = re.search(r"(GSM\d+)", p[7])
                if m:
                    gsm2srx.setdefault(m.group(1), p[0])

    # ---- Select samples ----
    def encode_fn(acc, cache_dir):
        return get_encode_files(acc, fetch_fn, cache_dir)

    cell_urls = select_atac_samples(
        cfg['cov_tsv'], cfg['cistrome_json'], gsm2srx,
        encode_fn, log_cells, cfg['cache'])

    # ---- Load genes and TSS ----
    canon = [l.strip() for l in open(cfg['genes'], encoding="utf-8") if l.strip()]
    tss_dict = {}
    with open(cfg['tss'], encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["symbol"] in canon:
                tss_dict[r["symbol"]] = (r["chrom"], int(r["tss_hg38"]))

    # ---- Open genome ----
    if open_genome is None:
        import py2bit
        open_genome = py2bit.open

    genome = open_genome(cfg['twobit'])

    # ---- Compute genome background ----
    bg = genome_background(genome)

    # ---- Genome sha1 ----
    manifest = {}
    if os.path.isfile(cfg['twobit']):
        manifest['genome_sha1'] = sha1_file(cfg['twobit'])
    manifest['genome_bg'] = list(bg)

    # ---- Load motifs ----
    if load_motifs is None:
        from pyjaspar import jaspardb
        jdb = jaspardb(release='JASPAR2024')
        motifs = jdb.fetch_motifs(collection='CORE', tax_group=['vertebrates'],
                                  all_versions=False)
    else:
        motifs = load_motifs()

    n_motifs = len(motifs)
    motif_map = build_motif_map(motifs)

    manifest['motifs'] = {
        'release': 'JASPAR2024',
        'count': n_motifs,
        'ids': [m.matrix_id for m in motifs],
        'rule': 'latest version per ID (all_versions=False)',
        'ps': 0.01,
        'p': 1e-4,
        'tf_motif_rule': 'JASPAR names split on ::, case-insensitive; a TF with several motifs takes the mean of their standardised MA',
    }

    # ---- Build scanner / scan function ----
    if scan_factory is None:
        scan_fn = make_parallel_scan_fn(motifs, bg, int(cfg.get('threads', 1)))
    else:
        scan_fn = scan_factory(motifs, bg)

    # ---- Load network ----
    if load_net is None:
        import decoupler as dc
        net = dc.op.collectri(organism='human')
        dc_version = dc.__version__
    else:
        net, dc_version = load_net()

    sorted_net = net.sort_values(by=net.columns.tolist()).reset_index(drop=True)
    dc_date = datetime.datetime.now().strftime("%Y-%m-%d")
    h_collec = hashlib.sha1(sorted_net.to_csv(index=False).encode('utf-8')).hexdigest()
    manifest['collectri'] = {
        'call': 'dc.op.collectri',
        'version': dc_version,
        'date': dc_date,
        'rows': len(sorted_net),
        'sha1': h_collec,
    }

    # ---- Package versions ----
    import importlib.metadata as md
    manifest['packages'] = {}
    for pkg_name in PACKAGES:
        try:
            manifest['packages'][pkg_name] = md.version(pkg_name)
        except md.PackageNotFoundError:
            manifest['packages'][pkg_name] = None          # a fake was injected for it (tests only)
    manifest['python'] = sys.version

    # ---- Process cells ----
    cells_kept = [c for c in cells_order if c in cell_urls]
    NC = len(cells_kept)
    NG = len(canon)
    F_prom_mat = np.zeros((NC, NG), dtype=np.float32)
    F_enh_mat = np.zeros((NC, NG), dtype=np.float32)
    F_reg_mat = np.zeros((NC, NG), dtype=np.float32)
    raw_MA_mat = np.zeros((NC, n_motifs), dtype=np.float32)
    has_mat = np.zeros(NC, dtype=bool)

    manifest['n_tracks_v9'] = dict(log_cells)
    manifest['n_tracks_now'] = {}
    manifest['mismatches'] = {}
    manifest['files'] = {}
    manifest['per_cell'] = {}

    # For split halves: store per-cell info about usable samples
    cell_usable_beds = {}  # cell -> list of (url_index, bed) for usable samples

    for ci, cell in enumerate(cells_kept):
        t_cell = time.time()
        urls = cell_urls[cell]
        manifest['per_cell'][cell] = {'urls': urls}
        parsed_beds = []
        usable = []   # (url_index, bed) for usable samples
        n_now = 0
        for ui, u in enumerate(urls):
            try:
                raw = fetch_fn(u, cfg['cache'])
                file_sha1 = hashlib.sha1(raw).hexdigest()
                bed = parse_bed(raw)
                n_ivs = sum(len(x[0]) for x in bed.values())
                manifest['files'][u] = {
                    'bytes': len(raw), 'sha1': file_sha1,
                    'n_intervals': n_ivs,
                }
                parsed_beds.append(bed)
                n_now += 1
                if n_ivs > 0:
                    usable.append((ui, bed))
            except Exception as e:
                manifest['files'][u] = {'error': str(e)}

        manifest['n_tracks_now'][cell] = n_now
        if cell in log_cells and n_now != log_cells[cell]:
            manifest['mismatches'][cell] = {
                'v9': log_cells[cell], 'now': n_now}

        cell_usable_beds[cell] = usable

        cf = cell_features(parsed_beds, tss_dict, canon, genome, scan_fn, merge_fn)
        parsed_urls = [u for u in urls if 'error' not in manifest['files'][u]]
        for u, g in zip(parsed_urls, cf['per_file_genes']):
            manifest['files'][u]['genes_with_peak'] = len(g)
        print('cell %d/%d %s: %d files, %d usable, has=%s, %d near-landmark peaks, %.0f s'
              % (ci + 1, NC, cell, n_now, len(usable), cf['has'], cf['near_landmarks'], time.time() - t_cell), flush=True)
        F_prom_mat[ci] = cf['F_prom']
        F_enh_mat[ci] = cf['F_enh']
        raw_MA_mat[ci] = cf['MA_raw']
        has_mat[ci] = cf['has']

        manifest['per_cell'][cell].update({
            'raw_intervals': cf['raw_intervals'],
            'merged': cf['merged'],
            'near_landmarks': cf['near_landmarks'],
            'dropped_chrom': cf['dropped_chrom'],
            'genes_with_prom': cf['genes_with_prom'],
        })
        if cf['per_file_genes']:
            manifest['per_cell'][cell]['per_file_genes'] = [
                len(g) for g in cf['per_file_genes']]

    # ---- Standardise MA and compute F_reg ----
    MA_std = standardise_across_cells(raw_MA_mat, has_mat)
    for ci in range(NC):
        if has_mat[ci]:
            F_reg_mat[ci] = regulon_scores(sorted_net, MA_std[ci], motif_map, canon)

    # ---- Build arrays dict ----
    arrays = {
        'cells': np.array(cells_kept),
        'genes': np.array(canon),
        'F_prom': F_prom_mat,
        'F_enh': F_enh_mat,
        'F_reg': F_reg_mat,
        'has': has_mat,
        'has_tss': np.array([g in tss_dict for g in canon]),
        'MA_raw': raw_MA_mat,
        'MA_std': MA_std,
        'motif_ids': np.array([m.matrix_id for m in motifs]),
        'motif_names': np.array([m.name for m in motifs]),
    }

    # ---- Split halves (93.8), while the genome is still open ----
    halves = _compute_halves(cells_kept, has_mat, cell_usable_beds, tss_dict, canon, genome, scan_fn, merge_fn, raw_MA_mat,
                             n_motifs, motif_map, sorted_net)
    if hasattr(genome, 'close'):
        genome.close()
    manifest['halves'] = {'rule': '93.8: usable samples (parsed, >= 1 interval) in URL order; A = positions 0, 2, 4 of the '
                                  'usable list, B = 1, 3, 5; cells with has only; F_reg from MA standardised with the '
                                  'full-union mean and sd',
                          'cells': list(halves['cells_with_halves'])}

    return arrays, manifest, halves


def _compute_halves(cells_kept, has_mat, cell_usable_beds, tss_dict, canon, genome, scan_fn, merge_fn, raw_MA_mat,
                    n_motifs, motif_map, sorted_net):
    """93.8: per-half features for every kept cell with has and >= 2 usable samples."""
    col = raw_MA_mat[has_mat]
    mean = col.mean(0) if len(col) else np.zeros(n_motifs)
    sd = col.std(0, ddof=1) if len(col) > 1 else np.zeros(n_motifs)
    halves = {'cells_with_halves': []}
    for ci, cell in enumerate(cells_kept):
        usable = cell_usable_beds.get(cell, [])
        if not has_mat[ci] or len(usable) < 2:
            continue
        out = {}
        for h, beds in (('A', [bed for i, (_, bed) in enumerate(usable) if i % 2 == 0]),
                        ('B', [bed for i, (_, bed) in enumerate(usable) if i % 2 == 1])):
            cf = cell_features(beds, tss_dict, canon, genome, scan_fn, merge_fn)
            ma_std = np.where(sd > 0, (cf['MA_raw'] - mean) / np.where(sd > 0, sd, 1.0), 0.0).astype(np.float32)
            out.update({'F_prom_' + h: cf['F_prom'], 'F_enh_' + h: cf['F_enh'], 'MA_raw_' + h: cf['MA_raw'],
                        'F_reg_' + h: regulon_scores(sorted_net, ma_std, motif_map, canon), 'has_' + h: cf['has'],
                        'n_samples_' + h: len(beds)})
        halves[cell] = out
        halves['cells_with_halves'].append(cell)
    return halves


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest():
    """Raises AssertionError on any failure.  Needs no network.

    (a) pyranges against the reference on fixtures and random data.
    (b) Real MOODS with a real JASPAR motif (CTCF MA0139.2).
    (c) Equivalence: build_features on a full test fixture.
    """
    import MOODS.tools
    import MOODS.scan
    import pandas as pd
    import pyranges as pr
    from pyjaspar import jaspardb

    # ================================================================
    # (a) pyranges against the reference
    # ================================================================
    df = pd.DataFrame([
        {"Chromosome": "chr1", "Start": 10, "End": 20},
        {"Chromosome": "chr1", "Start": 15, "End": 25},
        {"Chromosome": "chr1", "Start": 25, "End": 30},
        {"Chromosome": "chr1", "Start": 31, "End": 35},
        {"Chromosome": "chr1", "Start": 40, "End": 50},
        {"Chromosome": "chr2", "Start": 100, "End": 200},
        {"Chromosome": "chr2", "Start": 150, "End": 250},
    ])
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)

    ref = merge_intervals_reference(df)
    pr_merged = merge_peaks(df)

    assert len(ref) == len(pr_merged), (
        f"pyranges and reference differ in length: {len(ref)} vs {len(pr_merged)}")
    assert list(ref.Start) == list(pr_merged.Start), "Start mismatch"
    assert list(ref.End) == list(pr_merged.End), "End mismatch"
    assert list(ref.Chromosome) == list(pr_merged.Chromosome), "Chromosome mismatch"

    # Expected: [10,30], [31,35], [40,50] on chr1; [100,250] on chr2
    assert len(ref) == 4, f"Expected 4 merged intervals, got {len(ref)}"

    # Random intervals
    rng = np.random.default_rng(42)
    chroms = ['chr1', 'chr2', 'chr3']
    rows = []
    for _ in range(2000):
        c = chroms[rng.integers(0, 3)]
        s = rng.integers(0, 1_000_000)
        e = s + rng.integers(1, 10000)
        rows.append({"Chromosome": c, "Start": s, "End": e})
    df2 = pd.DataFrame(rows)
    df2["Start"] = df2["Start"].astype(np.int64)
    df2["End"] = df2["End"].astype(np.int64)

    ref2 = merge_intervals_reference(df2)
    pr2 = merge_peaks(df2)
    assert len(ref2) == len(pr2), f"Random: length mismatch {len(ref2)} vs {len(pr2)}"
    assert list(ref2.Start) == list(pr2.Start), "Random: Start mismatch"
    assert list(ref2.End) == list(pr2.End), "Random: End mismatch"

    # ================================================================
    # (b) Real MOODS with CTCF MA0139.2
    # ================================================================
    jdb = jaspardb(release='JASPAR2024')
    all_motifs = jdb.fetch_motifs(collection='CORE', tax_group=['vertebrates'],
                                  all_versions=False)
    ctcf_list = [m for m in all_motifs if m.matrix_id == 'MA0139.2']
    assert len(ctcf_list) == 1, "CTCF MA0139.2 not found"
    ctcf = ctcf_list[0]
    assert ctcf.name == 'CTCF', f"Expected CTCF, got {ctcf.name}"

    # Build consensus
    c = ctcf.counts
    counts = [list(map(float, c['A'])), list(map(float, c['C'])),
              list(map(float, c['G'])), list(map(float, c['T']))]
    ncols = len(counts[0])
    consensus = ''
    for j in range(ncols):
        col = [counts[b][j] for b in range(4)]
        consensus += 'ACGT'[col.index(max(col))]

    # Reverse complement of the consensus
    comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    rc_consensus = ''.join(comp[b] for b in reversed(consensus))

    bg = (0.25, 0.25, 0.25, 0.25)
    fw = MOODS.tools.log_odds(counts, bg, 0.01)
    rc = MOODS.tools.reverse_complement(fw, 4)
    max_fw_score = MOODS.tools.max_score(fw)

    scanner = make_scanner([fw], [rc], bg, p=1e-4)

    # Plant peaks: 40 random 300-bp peaks
    rng = np.random.default_rng(123)
    n_peaks = 40
    n_fw = 10
    n_rc = 10
    seqs = []
    planted_positions = []
    for i in range(n_peaks):
        base_seq = ''.join(rng.choice(list('ACGT'), size=300))
        pos = rng.integers(10, 300 - ncols - 10)
        if i < n_fw:
            # Plant forward consensus
            base_seq = base_seq[:pos] + consensus + base_seq[pos + ncols:]
            planted_positions.append(('fw', pos))
        elif i < n_fw + n_rc:
            # Plant RC consensus only
            base_seq = base_seq[:pos] + rc_consensus + base_seq[pos + ncols:]
            planted_positions.append(('rc', pos))
        else:
            planted_positions.append(('none', -1))
        seqs.append(base_seq)

    hits = motif_hits(scanner, 1, seqs)
    assert hits.shape == (40, 1), f"Wrong shape: {hits.shape}"

    # Every planted peak is hit
    for i in range(n_fw + n_rc):
        assert hits[i, 0], f"Planted peak {i} not hit"

    # (b) On each RC-only peak: the RC matrix's best hit score equals max_score(fw)
    # at the planted position
    for i in range(n_fw, n_fw + n_rc):
        results = scanner.scan(seqs[i].upper())
        # results[1] is the RC matrix hits
        rc_hits = results[1]
        assert len(rc_hits) > 0, f"RC-only peak {i} has no RC hit"
        best_rc = max(rc_hits, key=lambda h: h.score)
        assert abs(best_rc.score - max_fw_score) < 1e-6, (
            f"RC hit score {best_rc.score} != max_score {max_fw_score}")
        _, planted_pos = planted_positions[i]
        assert best_rc.pos == planted_pos, (
            f"RC hit position {best_rc.pos} != planted {planted_pos}")

    # MA equals per-peak hit mean exactly
    ma = motif_accessibility(hits)
    assert ma[0] == hits[:, 0].mean(), "MA != mean of hits"

    # MA >= 0.5
    assert ma[0] >= 0.5, f"MA {ma[0]} < 0.5"

    # MA = (planted + chance-hit unplanted) / all: the 20 planted peaks give exactly 0.5, chance hits add k / 40
    k_chance = int(hits[n_fw + n_rc:, 0].sum())
    assert abs(ma[0] - (n_fw + n_rc + k_chance) / n_peaks) < 1e-9, (ma[0], k_chance)

    # ================================================================
    # (c) build_features end to end: real MOODS + real pyranges against the reference merge; F_reg by hand
    # ================================================================
    import tempfile
    import shutil
    motif2 = next(m for m in all_motifs if m.matrix_id != 'MA0139.2' and '::' not in m.name)
    test_motifs = [ctcf, motif2]
    tmpdir = tempfile.mkdtemp()
    try:
        fx = selftest_fixture(tmpdir, consensus)
        net_df = pd.DataFrame([{'source': 'CTCF', 'target': 'DDR1', 'weight': 1.0},
                               {'source': motif2.name, 'target': 'PAX8', 'weight': -1.0}])
        runs = {}
        for name, mf in (('pyranges', merge_peaks), ('reference', merge_intervals_reference)):
            runs[name] = build_features(fx['cfg'], fetch_fn=fx['fetch'], open_genome=lambda path: fx['genome'],
                                        load_motifs=lambda: test_motifs, load_net=lambda: (net_df, 'selftest'),
                                        scan_factory=None, merge_fn=mf)
        A, R = runs['pyranges'][0], runs['reference'][0]
        for k in ('F_prom', 'F_enh', 'F_reg', 'MA_raw', 'has'):
            assert np.array_equal(A[k], R[k]), 'pyranges and reference differ in %s' % k
        assert list(A['has']) == [True, True, False], A['has']          # CellC: peaks give no promoter coverage
        assert A['MA_raw'][0, 0] >= 2 / 3 - 1e-9, A['MA_raw']            # CTCF planted in 2 of CellA's 3 merged peaks
        ms = standardise_across_cells(A['MA_raw'], A['has'])
        g = list(A['genes'])
        assert abs(A['F_reg'][0, g.index('DDR1')] - ms[0, 0]) < 1e-6        # +1 x CTCF
        assert abs(A['F_reg'][0, g.index('PAX8')] + ms[0, 1]) < 1e-6        # -1 x motif2
        assert np.all(A['F_reg'][2] == 0)
        assert runs['pyranges'][2]['cells_with_halves'] == ['CellA'], runs['pyranges'][2]['cells_with_halves']
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class DictGenome:
    """A dict-backed genome with the four py2bit methods the module uses (selftest and tests only)."""

    def __init__(self, seqs):
        self._seqs = seqs

    def chroms(self):
        return {c: len(v) for c, v in self._seqs.items()}

    def sequence(self, chrom, start, end):
        return self._seqs[chrom][start:end]

    def bases(self, chrom, start, end, fraction=True):
        sub = self._seqs[chrom][start:end].upper()
        return {b: sub.count(b) for b in 'ACGT'}

    def close(self):
        pass


def selftest_fixture(tmpdir, consensus):
    """Three cistrome-source cells over a 2-chromosome random genome. CellA: 2 usable samples, CTCF consensus planted in
    its peaks; CellB: 1 sample; CellC: peaks far from every TSS (has = False). Returns cfg, fetch and genome."""
    rng = np.random.default_rng(7)
    seqs = {ch: ''.join(rng.choice(list('ACGT'), size=300000)) for ch in ('chr1', 'chr6')}
    a = seqs['chr6']
    for pos in (49700, 40300):
        a = a[:pos] + consensus + a[pos + len(consensus):]
    seqs['chr6'] = a
    files = {'genes': 'DDR1\nPAX8\nNOTSS\n',
             'tss': 'symbol\tentrez\tchrom\ttss_hg38\tstrand\nDDR1\t780\tchr6\t50000\t+\nPAX8\t7849\tchr1\t100000\t-\n',
             'cell_index': json.dumps({'cell_id_to_row': {'CellA': 0, 'CellB': 1, 'CellC': 2}}),
             'peaks_log': ''.join('%s/ATAC-seq src=cistrome: %d bed track(s), nonzero_genes=2  (1s)\n' % (c, n)
                                  for c, n in (('CellA', 2), ('CellB', 1), ('CellC', 1))),
             'cov_tsv': 'lincs_cell_id\tstatus\tassay_target\tsource_used\tnotes\tsample_ids_used\n'
                        'CellA\tresolved\tATAC-seq\tcistrome\t\t1, 2\nCellB\tresolved\tATAC-seq\tcistrome\t\t3\n'
                        'CellC\tresolved\tATAC-seq\tcistrome\t\t4\n',
             'cistrome_json': json.dumps([{'id': i, 'external_id_type': 'GEO', 'external_id': 'GSM%d' % i} for i in (1, 2, 3, 4)]),
             'chipatlas_list': ''.join('SRX%d\thg38\tATAC-Seq\tx\tx\tx\tx\tGSM%d: sample\n' % (i, i) for i in (1, 2, 3, 4))}
    cfg = {'cache': os.path.join(tmpdir, 'cache'), 'twobit': 'fake.2bit', 'threads': 1}
    for k, v in files.items():
        cfg[k] = os.path.join(tmpdir, k)
        io_write(cfg[k], v)
    beds = {1: 'chr6\t49600\t50400\t.\t1\t.\t5\nchr6\t40200\t40800\t.\t1\t.\t5\n',
            2: 'chr6\t49900\t50600\t.\t1\t.\t5\nchr1\t99500\t100500\t.\t1\t.\t5\n',
            3: 'chr6\t49000\t49500\t.\t1\t.\t5\nchr1\t99800\t100300\t.\t1\t.\t5\nchrUn\t10\t20\t.\t1\t.\t5\n',
            4: 'chr1\t200000\t200500\t.\t1\t.\t5\n'}
    served = {'https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/SRX%d.05.bed' % i: b.encode() for i, b in beds.items()}

    def fetch_fn(url, cache_dir):
        if url not in served:
            raise IOError('not served: ' + url)
        return served[url]
    return {'cfg': cfg, 'fetch': fetch_fn, 'genome': DictGenome(seqs)}


def io_write(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def selftest_genome(tb):
    """Check the real genome after download.  Called by main()."""
    chroms = tb.chroms()
    assert chroms.get('chr1') == 248956422, (
        f"chr1 length: {chroms.get('chr1')} != 248956422")
    seq = tb.sequence('chr1', 0, 10)
    assert seq == 'NNNNNNNNNN', f"chr1[0:10]: {seq!r} != 'NNNNNNNNNN'"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="§93 chromatin features")
    parser.add_argument("--cov_tsv", required=True)
    parser.add_argument("--cistrome_json", required=True)
    parser.add_argument("--peaks_log", required=True)
    parser.add_argument("--tss", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--chipatlas_list", required=True)
    parser.add_argument("--twobit", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cell_index", required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    selftest()
    print('selftest passed (%.0f s)' % (time.time() - t0), flush=True)
    if args.twobit.startswith('http'):
        args.twobit = fetch_to_file(args.twobit, args.cache)           # ~0.8 GB, streamed to the cache
        print('genome downloaded (%.0f s)' % (time.time() - t0), flush=True)

    cfg = {
        'cov_tsv': args.cov_tsv,
        'cistrome_json': args.cistrome_json,
        'peaks_log': args.peaks_log,
        'tss': args.tss,
        'genes': args.genes,
        'chipatlas_list': args.chipatlas_list,
        'twobit': args.twobit,
        'cache': args.cache,
        'out_dir': args.out_dir,
        'cell_index': args.cell_index,
        'threads': args.threads,
    }

    import py2bit
    tb = py2bit.open(args.twobit)
    selftest_genome(tb)
    bg = genome_background(tb)
    tb.close()
    from pyjaspar import jaspardb
    motifs = jaspardb(release='JASPAR2024').fetch_motifs(collection='CORE', tax_group=['vertebrates'], all_versions=False)
    print('MOODS bench: %.1f s per 200 kb with %d motifs, both strands, one process' % (bench_scan(motifs, bg), len(motifs)),
          flush=True)

    arrays, manifest, halves = build_features(cfg)

    # Write outputs
    out = args.out_dir
    npz_path = os.path.join(out, "c93_features.npz")
    np.savez(npz_path,
             cells=arrays['cells'], genes=arrays['genes'],
             F_prom=arrays['F_prom'], F_enh=arrays['F_enh'],
             F_reg=arrays['F_reg'], has=arrays['has'], has_tss=arrays['has_tss'],
             MA_raw=arrays['MA_raw'], MA_std=arrays['MA_std'],
             motif_ids=arrays['motif_ids'], motif_names=arrays['motif_names'])

    halves_path = os.path.join(out, "c93_halves.npz")
    halves_save = {}
    halves_save['cells_with_halves'] = np.array(halves['cells_with_halves'])
    for c in halves['cells_with_halves']:
        for k, v in halves[c].items():
            halves_save[f'{c}_{k}'] = v
    np.savez(halves_path, **halves_save)

    report = split_half_report(arrays, halves)
    report_path = os.path.join(out, "c93_split_half.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    manifest_path = os.path.join(out, "c93_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Completion marker
    h_npz = sha1_file(npz_path)
    h_halves = sha1_file(halves_path)
    h_report = sha1_file(report_path)
    h_manifest = sha1_file(manifest_path)
    complete = {
        'c93_features.npz': h_npz,
        'c93_halves.npz': h_halves,
        'c93_split_half.json': h_report,
        'c93_manifest.json': h_manifest,
    }
    with open(os.path.join(out, "C93_FEATURES_COMPLETE.json"), 'w') as f:
        json.dump(complete, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
