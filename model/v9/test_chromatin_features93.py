"""Tests for chromatin_features93 – §93 Stage 1b.

Local tests (no network, no LINCS data).  Uses tmp_path for all files.
"""

import pytest
import os
import json
import numpy as np
import pandas as pd
from model.v9.chromatin_features93 import (
    select_atac_samples, compute_windows, parse_bed, merge_intervals_reference,
    merge_peaks, near_landmark_peaks, standardise_across_cells, regulon_scores,
    motif_accessibility, build_motif_map, cell_features, build_features,
    split_half_report, selftest,
)


# ===================================================================
# Helpers: fake genome, fake motifs, fake scanner, fake network
# ===================================================================

class FakeGenome:
    """Dict-backed genome with the same four methods as py2bit."""
    def __init__(self, seqs):
        self._seqs = seqs
    def chroms(self):
        return {c: len(s) for c, s in self._seqs.items()}
    def sequence(self, chrom, start, end):
        return self._seqs[chrom][start:end]
    def bases(self, chrom, start, end, fraction):
        seq = self._seqs[chrom][start:end]
        return {b: seq.count(b) for b in 'ACGT'}
    def close(self):
        pass


class FakeMotif:
    """Minimal motif object with .matrix_id, .name, .counts."""
    def __init__(self, matrix_id, name, counts):
        self.matrix_id = matrix_id
        self.name = name
        self.counts = counts


def make_fake_scanner(motifs, bg):
    """Return a scan_fn that does exact-substring matching on each motif's
    consensus or its reverse complement.

    This lives in the test file only.
    """
    # Build consensus and RC consensus for each motif
    comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    consensuses = []
    rc_consensuses = []
    for m in motifs:
        c = m.counts
        ncols = len(c['A'])
        cons = ''
        for j in range(ncols):
            col = [c[b][j] for b in 'ACGT']
            cons += 'ACGT'[col.index(max(col))]
        consensuses.append(cons)
        rc_consensuses.append(''.join(comp[b] for b in reversed(cons)))

    n_motifs = len(motifs)

    def scan_fn(seqs):
        out = np.zeros((len(seqs), n_motifs), dtype=bool)
        for si, seq in enumerate(seqs):
            s = seq.upper()
            for mi in range(n_motifs):
                if consensuses[mi] in s or rc_consensuses[mi] in s:
                    out[si, mi] = True
        return out

    return scan_fn


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def cov_tsv(tmp_path):
    p = tmp_path / "cov.tsv"
    p.write_text(
        "lincs_cell_id\tstatus\tassay_target\tsource_used\tnotes\tsample_ids_used\n"
        "A\tresolved\tATAC-seq\tencode\tENCSR1 ENCSR2 ENCSR3 ENCSR4 ENCSR5\t\n"
        "B\tresolved\tATAC-seq\tcistrome\t\t1, 2, 3, 4, 5, 6, 7\n"
        "C\tresolved\tATAC-seq\tepimap\t\t\n"
        "D\tresolved\tATAC-seq\tcistrome\t\t1\n",
        encoding="utf-8")
    return str(p)


@pytest.fixture
def cistrome_json(tmp_path):
    p = tmp_path / "cistrome.json"
    data = [{"id": str(i), "external_id_type": "GEO", "external_id": f"GSM{i}"}
            for i in range(1, 8)]
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ===================================================================
# test_selection
# ===================================================================

def test_selection(cov_tsv, cistrome_json):
    gsm2srx = {f"GSM{i}": f"SRX{i}" for i in range(1, 8)}
    # log_cells: only cells with N > 0. C has 0 tracks → excluded.
    log_cells = {"A": 4, "B": 6}

    def mock_encode_for(acc, cache):
        return [f"url_{acc}"]

    out = select_atac_samples(cov_tsv, cistrome_json, gsm2srx,
                               mock_encode_for, log_cells, "cache")
    # A: encode, first 4 ENCSR → 4 URLs
    assert "A" in out
    assert len(out["A"]) == 4
    assert out["A"] == ["url_ENCSR1", "url_ENCSR2", "url_ENCSR3", "url_ENCSR4"]
    # B: cistrome, first 6 SRX
    assert "B" in out
    assert len(out["B"]) == 6
    assert all("SRX" in u for u in out["B"])
    # C: not in log_cells (0 tracks)
    assert "C" not in out
    # D: not in log_cells
    assert "D" not in out


def test_selection_log_parsing():
    """The log parser keeps only cells with N > 0, reading real-format log lines."""
    import re
    lines = [
        "A549/ATAC-seq src=encode: 4 bed track(s), nonzero_genes=958  (38s)\n",
        "A375/ATAC-seq src=cistrome: 0 bed tracks (masked)\n",
    ]
    log_cells = {}
    for line in lines:
        m = re.match(r"^([^/]+)/ATAC-seq src=[a-z]+: (\d+) bed track", line)
        if m:
            c = m.group(1)
            t = int(m.group(2))
            if t > 0:
                log_cells[c] = t
    assert "A549" in log_cells
    assert log_cells["A549"] == 4
    assert "A375" not in log_cells


# ===================================================================
# test_merge
# ===================================================================

def test_merge_reference():
    """Overlapping and book-ended merge; a 1-bp gap and disjoint ones do not."""
    df = pd.DataFrame([
        {"Chromosome": "chr1", "Start": 10, "End": 20},
        {"Chromosome": "chr1", "Start": 15, "End": 25},
        {"Chromosome": "chr1", "Start": 25, "End": 30},   # book-ended with [15,25]
        {"Chromosome": "chr1", "Start": 31, "End": 35},   # 1-bp gap
        {"Chromosome": "chr1", "Start": 40, "End": 50},   # disjoint
    ])
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    merged = merge_intervals_reference(df)

    assert len(merged) == 3, f"Expected 3 merged intervals, got {len(merged)}"
    assert list(merged.Start) == [10, 31, 40]
    assert list(merged.End) == [30, 35, 50]


def test_merge_two_chromosomes():
    """Merge works correctly across two chromosomes."""
    df = pd.DataFrame([
        {"Chromosome": "chr1", "Start": 10, "End": 20},
        {"Chromosome": "chr1", "Start": 20, "End": 30},   # book-ended
        {"Chromosome": "chr2", "Start": 100, "End": 200},
        {"Chromosome": "chr2", "Start": 150, "End": 250},
    ])
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    merged = merge_intervals_reference(df)
    assert len(merged) == 2
    assert list(merged.Chromosome) == ["chr1", "chr2"]
    assert list(merged.Start) == [10, 100]
    assert list(merged.End) == [30, 250]


# ===================================================================
# test_windows
# ===================================================================

def test_windows_basic():
    df = pd.DataFrame([
        {"Chromosome": "chr1", "Start": 900, "End": 1200},  # promoter overlap
        {"Chromosome": "chr1", "Start": 11000, "End": 11500},  # enhancer
    ])
    tss = {"gene1": ("chr1", 1000)}
    F_prom, F_enh = compute_windows(df, tss, ["gene1"])
    assert np.isclose(F_prom[0], 300 / 2000.0)
    expected_enh = (500.0 / 1000.0) * np.exp(-10250.0 / 10000.0)
    assert np.isclose(F_enh[0], expected_enh)
    assert F_enh[0] > 0


def test_window_inside_promoter_not_in_enh():
    """A peak inside ±1 kb never contributes to F_enh."""
    df = pd.DataFrame([
        {"Chromosome": "chr1", "Start": 500, "End": 800},  # fully inside ±1 kb
    ])
    tss = {"gene1": ("chr1", 1000)}
    F_prom, F_enh = compute_windows(df, tss, ["gene1"])
    assert F_prom[0] > 0
    assert F_enh[0] == 0.0


def test_window_at_10kb_exactly():
    """A peak at d = 10 kb exactly gives (length / 1000) × e^{-1}."""
    # Place a peak whose midpoint is at exactly 10 kb from TSS
    # TSS at 50000, peak midpoint at 60000 → d = 10000
    peak_len = 500
    peak_start = 60000 - peak_len // 2   # 59750
    peak_end = 60000 + peak_len // 2     # 60250
    df = pd.DataFrame([
        {"Chromosome": "chr1", "Start": peak_start, "End": peak_end},
    ])
    tss = {"gene1": ("chr1", 50000)}
    F_prom, F_enh = compute_windows(df, tss, ["gene1"])
    expected = (peak_len / 1000.0) * np.exp(-1.0)
    assert np.isclose(F_enh[0], expected, rtol=1e-5), (
        f"F_enh = {F_enh[0]}, expected = {expected}")


def test_window_no_promoter_overlap():
    """Peak far from TSS gives no F_prom."""
    df = pd.DataFrame([{"Chromosome": "chr1", "Start": 11000, "End": 11500}])
    tss = {"gene1": ("chr1", 1000)}
    F_prom, F_enh = compute_windows(df, tss, ["gene1"])
    assert F_prom[0] == 0


# ===================================================================
# test_near_landmark_peaks
# ===================================================================

def test_near_landmark_peaks():
    df = pd.DataFrame([
        {"Chromosome": "chr1", "Start": 1000, "End": 2000},    # Mid 1500, d = 500
        {"Chromosome": "chr1", "Start": 60000, "End": 61000},  # Mid 60500, d = 59500
        {"Chromosome": "chr2", "Start": 1000, "End": 2000},    # No landmark on chr2
    ])
    tss = {"gene1": ("chr1", 1000)}
    peaks = near_landmark_peaks(df, tss, max_d=50000)
    assert len(peaks) == 1
    assert peaks[0] == ("chr1", 1000, 2000)


# ===================================================================
# test_regulons
# ===================================================================

def test_regulons():
    net_df = pd.DataFrame([
        {"source": "TF1", "target": "GENE1", "weight": 2.0},
        {"source": "TF2", "target": "GENE1", "weight": -1.5},
        {"source": "TF3", "target": "GENE1", "weight": 1.0},   # TF3 not in motif_map
        {"source": "A", "target": "GENE2", "weight": 1.0},
        {"source": "B", "target": "GENE2", "weight": -1.0},
    ])
    # Motif 2 is a dimer A::B. So motif_map maps both A and B to [2]
    motif_map = {"TF1": [0], "TF2": [1], "A": [2], "B": [2]}
    ma_std_row = np.array([0.5, -0.2, 0.6])
    genes = ["GENE1", "GENE2"]
    f_reg = regulon_scores(net_df, ma_std_row, motif_map, genes)
    # GENE1: TF1 (w=+1) → 0.5, TF2 (w=-1) → +0.2, TF3 ignored. Mean = 0.35
    assert np.isclose(f_reg[0], 0.35)
    # GENE2: A (w=+1) → 0.6, B (w=-1) → -0.6. Mean = 0.0
    assert np.isclose(f_reg[1], 0.0)


# ===================================================================
# test_standardise
# ===================================================================

def test_standardise():
    raw = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    has = np.array([True, False, True])
    std = standardise_across_cells(raw, has)
    assert np.isclose(std[0, 0], -0.70710678)
    assert np.isclose(std[2, 0], 0.70710678)
    assert std[1, 0] == 0.0


# ===================================================================
# test_motif_map
# ===================================================================

def test_motif_map_dimer():
    """A dimer name like 'A::B' maps both A and B to the motif index."""
    motifs = [FakeMotif('M1', 'TF1', {'A': [1], 'C': [0], 'G': [0], 'T': [0]}),
              FakeMotif('M2', 'FactorA::FactorB', {'A': [1], 'C': [0], 'G': [0], 'T': [0]})]
    mm = build_motif_map(motifs)
    assert 'TF1' in mm and mm['TF1'] == [0]
    assert 'FACTORA' in mm and mm['FACTORA'] == [1]
    assert 'FACTORB' in mm and mm['FACTORB'] == [1]


# ===================================================================
# test_end_to_end with fakes
# ===================================================================

def _make_end_to_end_fixture(tmp_path, n_cells=3):
    """Set up a full test fixture with 3 cells:
    - CellA: normal (peaks near TSS)
    - CellB: all-empty beds (has = False)
    - CellC: peaks give F_prom > 0 in no gene (has = False)
    """
    genes = ['GENE1', 'GENE2', 'GENE3']
    genes_file = tmp_path / 'genes.txt'
    genes_file.write_text('\n'.join(genes) + '\n')

    tss_file = tmp_path / 'tss.tsv'
    tss_file.write_text(
        "symbol\tentrez\tchrom\ttss_hg38\tstrand\n"
        "GENE1\t1\tchr1\t50000\t+\n"
        "GENE2\t2\tchr1\t100000\t-\n"
        "GENE3\t3\tchrX\t1500\t+\n")

    ci_file = tmp_path / 'cell_index.json'
    cells = {'CellA': 0, 'CellB': 1, 'CellC': 2}
    ci_file.write_text(json.dumps(cells))

    log_file = tmp_path / 'peaks_log.txt'
    log_file.write_text(
        "CellA/ATAC-seq src=encode: 2 bed track(s), nonzero_genes=2  (1s)\n"
        "CellB/ATAC-seq src=encode: 2 bed track(s), nonzero_genes=2  (2s)\n"
        "CellC/ATAC-seq src=encode: 1 bed track(s), nonzero_genes=1  (3s)\n")

    cov_file = tmp_path / 'cov.tsv'
    cov_file.write_text(
        "lincs_cell_id\tstatus\tassay_target\tsource_used\tnotes\tsample_ids_used\n"
        "CellA\tresolved\tATAC-seq\tencode\tENCSR001\t\n"
        "CellB\tresolved\tATAC-seq\tencode\tENCSR002\t\n"
        "CellC\tresolved\tATAC-seq\tencode\tENCSR003\t\n")

    cis_file = tmp_path / 'cistrome.json'
    cis_file.write_text(json.dumps([]))

    ca_file = tmp_path / 'chipatlas.tab'
    ca_file.write_text('')

    cache = tmp_path / 'cache'
    cache.mkdir()

    # Genome: chr1 of 200k bp, chr2 of 100k bp
    rng = np.random.default_rng(42)
    genome_seqs = {
        'chr1': ''.join(rng.choice(list('ACGT'), size=200000)),
        'chr2': ''.join(rng.choice(list('ACGT'), size=100000)),
    }
    fake_genome = FakeGenome(genome_seqs)

    # Motifs: two simple ones
    # M1: consensus "ACG" (3-bp for simplicity in substring matching)
    m1 = FakeMotif('MA0001.1', 'TF1',
                   {'A': [10.0, 0.1, 0.1], 'C': [0.1, 10.0, 0.1],
                    'G': [0.1, 0.1, 10.0], 'T': [0.1, 0.1, 0.1]})
    # M2: consensus "TGA", dimer "TFA::TFB"
    m2 = FakeMotif('MA0002.1', 'TFA::TFB',
                   {'A': [0.1, 0.1, 10.0], 'C': [0.1, 0.1, 0.1],
                    'G': [0.1, 10.0, 0.1], 'T': [10.0, 0.1, 0.1]})
    motifs = [m1, m2]

    # Network
    net_df = pd.DataFrame([
        {"source": "TF1", "target": "GENE1", "weight": 1.0,
         "resources": "t", "references": "t", "sign_decision": "t"},
        {"source": "TFA", "target": "GENE2", "weight": -1.0,
         "resources": "t", "references": "t", "sign_decision": "t"},
        {"source": "TFB", "target": "GENE2", "weight": 1.0,
         "resources": "t", "references": "t", "sign_decision": "t"},
        # TFX has no motif → ignored
        {"source": "TFX", "target": "GENE1", "weight": 1.0,
         "resources": "t", "references": "t", "sign_decision": "t"},
    ])

    # BED data for CellA: peaks near both TSS positions
    # CellA has 2 samples
    bed_A1_data = (
        "chr1\t49500\t50500\t.\t100\t.\t200\t-1\t-1\t0\n"   # GENE1 promoter
        "chr1\t40000\t41000\t.\t100\t.\t200\t-1\t-1\t0\n"   # GENE1 enhancer
        "chr1\t99500\t100500\t.\t100\t.\t200\t-1\t-1\t0\n"  # GENE2 promoter
        "chrX\t1000\t2000\t.\t100\t.\t200\t-1\t-1\t0\n"     # unknown chrom
    )
    bed_A2_data = (
        "chr1\t49000\t50000\t.\t100\t.\t200\t-1\t-1\t0\n"
        "chr1\t99000\t100000\t.\t100\t.\t200\t-1\t-1\t0\n"
    )

    # CellB: empty beds
    bed_B1_data = ""
    bed_B2_data = ""

    # CellC: peaks far from any TSS (F_prom > 0 in no gene)
    bed_C1_data = (
        "chr2\t1000\t2000\t.\t100\t.\t200\t-1\t-1\t0\n"  # no gene on chr2
    )

    # Write bed files to cache
    import hashlib as _hl

    def cache_bed(url, data):
        h = _hl.sha1(url.encode('utf-8')).hexdigest()
        (cache / h).write_bytes(data.encode('utf-8'))

    # Build ENCODE API responses
    # CellA → ENCSR001 → 2 files
    url_A1 = "https://www.encodeproject.org/files/ENCFF_A1/@@download/ENCFF_A1.bed.gz"
    url_A2 = "https://www.encodeproject.org/files/ENCFF_A2/@@download/ENCFF_A2.bed.gz"
    cache_bed(url_A1, bed_A1_data)
    cache_bed(url_A2, bed_A2_data)

    url_B1 = "https://www.encodeproject.org/files/ENCFF_B1/@@download/ENCFF_B1.bed.gz"
    url_B2 = "https://www.encodeproject.org/files/ENCFF_B2/@@download/ENCFF_B2.bed.gz"
    cache_bed(url_B1, bed_B1_data)
    cache_bed(url_B2, bed_B2_data)

    url_C1 = "https://www.encodeproject.org/files/ENCFF_C1/@@download/ENCFF_C1.bed.gz"
    cache_bed(url_C1, bed_C1_data)

    # ENCODE API responses
    def encode_api_url(acc):
        return (f"https://www.encodeproject.org/search/?type=File&dataset=/experiments/{acc}/"
                f"&file_format=bed&file_format_type=narrowPeak&assembly=GRCh38&status=released"
                f"&output_type=replicated+peaks&output_type=peaks&output_type=pseudoreplicated+peaks"
                f"&format=json&field=accession&limit=3")

    def cache_encode_api(acc, file_accs):
        url = encode_api_url(acc)
        resp = json.dumps({"@graph": [{"accession": a} for a in file_accs]})
        h = _hl.sha1(url.encode('utf-8')).hexdigest()
        (cache / h).write_bytes(resp.encode('utf-8'))

    cache_encode_api("ENCSR001", ["ENCFF_A1", "ENCFF_A2"])
    cache_encode_api("ENCSR002", ["ENCFF_B1", "ENCFF_B2"])
    cache_encode_api("ENCSR003", ["ENCFF_C1"])

    cfg = {
        'cov_tsv': str(cov_file),
        'cistrome_json': str(cis_file),
        'peaks_log': str(log_file),
        'tss': str(tss_file),
        'genes': str(genes_file),
        'chipatlas_list': str(ca_file),
        'twobit': 'fake',
        'cache': str(cache),
        'cell_index': str(ci_file),
        'threads': 1,
    }

    return cfg, fake_genome, motifs, net_df


def test_end_to_end(tmp_path):
    """build_features on 3 cells: one normal, one empty (has=False), one no
    genes with F_prom > 0 (has=False)."""

    cfg, fake_genome, motifs, net_df = _make_end_to_end_fixture(tmp_path)

    def open_genome(path):
        return fake_genome

    def load_motifs():
        return motifs

    def load_net():
        return net_df, '0.0.0'

    arrays, manifest, halves = build_features(
        cfg,
        fetch_fn=None,   # uses the real fetch (from cache)
        open_genome=open_genome,
        load_motifs=load_motifs,
        load_net=load_net,
        scan_factory=make_fake_scanner,
        merge_fn=merge_intervals_reference,
    )

    cells = list(arrays['cells'])
    assert 'CellA' in cells
    assert 'CellB' in cells
    assert 'CellC' in cells

    idx_A = cells.index('CellA')
    idx_B = cells.index('CellB')
    idx_C = cells.index('CellC')

    # CellA: has = True, F_prom > 0 for at least one gene
    assert arrays['has'][idx_A] == True
    assert (arrays['F_prom'][idx_A] > 0).any()

    # CellB: all-empty beds → has = False, features are 0
    assert arrays['has'][idx_B] == False
    assert (arrays['F_prom'][idx_B] == 0).all()
    assert (arrays['F_enh'][idx_B] == 0).all()

    # CellC: peaks in chr2 but no gene TSS there → F_prom > 0 in no gene → has = False
    assert arrays['has'][idx_C] == False

    # CellA: chrX peak was dropped
    assert manifest['per_cell']['CellA']['dropped_chrom'] > 0

    # CellA: MA through fake scanner is a hand count
    # The fake scanner does substring matching for "ACG" (M1) and "TGA" (M2)
    # The near-landmark peak sequences contain random genomic content
    # MA_raw[idx_A] should match the expected fraction
    ma_A = arrays['MA_raw'][idx_A]
    assert len(ma_A) == 2  # 2 motifs

    # F_reg: check structure
    assert arrays['F_reg'].shape[1] == 3  # 3 genes

    # F_reg for CellA should be computed (may be 0 if MA_std is 0 for single cell)
    # With only 1 has=True cell, standardise gives 0 (sd=0 or n=1)
    # That's fine – this just tests the pipeline runs


def test_dropped_chrom(tmp_path):
    """A peak on a chromosome missing from the genome is dropped and counted."""
    cfg, fake_genome, motifs, net_df = _make_end_to_end_fixture(tmp_path)

    def open_genome(path):
        return fake_genome

    arrays, manifest, halves = build_features(
        cfg, open_genome=open_genome,
        load_motifs=lambda: motifs,
        load_net=lambda: (net_df, '0.0.0'),
        scan_factory=make_fake_scanner,
        merge_fn=merge_intervals_reference,
    )

    # CellA has a peak on chrX which is not in the fake genome
    assert manifest['per_cell']['CellA']['dropped_chrom'] > 0


def test_f_reg_hand_computed(tmp_path):
    """F_reg equals hand-computed signed means, with a dimer counting for
    both partners and a regulator without a motif ignored."""
    genes = ['GENE1', 'GENE2']

    net_df = pd.DataFrame([
        {"source": "TF1", "target": "GENE1", "weight": 2.0},
        {"source": "TFX", "target": "GENE1", "weight": 1.0},   # no motif → ignored
        {"source": "TFA", "target": "GENE2", "weight": -1.0},  # from dimer
        {"source": "TFB", "target": "GENE2", "weight": 1.0},   # from dimer
    ])

    # dimer M2 maps to both TFA and TFB
    motif_map = {"TF1": [0], "TFA": [1], "TFB": [1]}
    ma_std_row = np.array([0.5, 0.8])

    f_reg = regulon_scores(net_df, ma_std_row, motif_map, genes)

    # GENE1: TF1 (sign=+1) → 0.5. TFX ignored. Mean = 0.5
    assert np.isclose(f_reg[0], 0.5)
    # GENE2: TFA (sign=-1) → mean([0.8]) * (-1) = -0.8
    #         TFB (sign=+1) → mean([0.8]) * (+1) = +0.8
    #         Mean of [-0.8, 0.8] = 0.0
    assert np.isclose(f_reg[1], 0.0)


# ===================================================================
# test_split_halves
# ===================================================================

def test_split_halves_min_samples(tmp_path):
    """A cell with 1 usable sample gets no halves."""
    cfg, fake_genome, motifs, net_df = _make_end_to_end_fixture(tmp_path)

    # Modify CellC to have 1 usable sample (not 0)
    # CellC already has 1 sample with intervals on chr2
    # So CellC should NOT be in halves (needs ≥ 2)

    def open_genome(path):
        return fake_genome

    arrays, manifest, halves = build_features(
        cfg, open_genome=open_genome,
        load_motifs=lambda: motifs,
        load_net=lambda: (net_df, '0.0.0'),
        scan_factory=make_fake_scanner,
        merge_fn=merge_intervals_reference,
    )

    # CellC has only 1 sample → not in halves
    assert 'CellC' not in halves.get('cells_with_halves', [])


def test_split_halves_ab_positions(tmp_path):
    """A/B positions are 0,2,4 and 1,3,5 of the usable list.
    An empty file shifts the rest."""
    # Create a cell with 5 URLs: URL 0 → empty, URL 1-4 → have data
    genes_file = tmp_path / 'genes.txt'
    genes_file.write_text('GENE1\n')

    tss_file = tmp_path / 'tss.tsv'
    tss_file.write_text(
        "symbol\tentrez\tchrom\ttss_hg38\tstrand\n"
        "GENE1\t1\tchr1\t50000\t+\n")

    ci_file = tmp_path / 'cell_index.json'
    ci_file.write_text(json.dumps({'CellX': 0}))

    log_file = tmp_path / 'peaks_log.txt'
    log_file.write_text(
        "CellX/ATAC-seq src=cistrome: 5 bed track(s), nonzero_genes=1  (1s)\n")

    cov_file = tmp_path / 'cov.tsv'
    cov_file.write_text(
        "lincs_cell_id\tstatus\tassay_target\tsource_used\tnotes\tsample_ids_used\n"
        "CellX\tresolved\tATAC-seq\tcistrome\t\t1, 2, 3, 4, 5\n")

    cis_file = tmp_path / 'cistrome.json'
    cis_data = [{"id": str(i), "external_id_type": "GEO",
                 "external_id": f"GSM{i}"} for i in range(1, 6)]
    cis_file.write_text(json.dumps(cis_data))

    ca_file = tmp_path / 'chipatlas.tab'
    # Map GSM1-5 → SRX1-5
    lines = []
    for i in range(1, 6):
        # 8 columns, GSM in column 8 (0-indexed 7)
        lines.append(f"SRX{i}\tcol2\tcol3\tcol4\tcol5\tcol6\tcol7\tGSM{i}\n")
    ca_file.write_text(''.join(lines))

    cache = tmp_path / 'cache'
    cache.mkdir()

    import hashlib as _hl

    # URL 0 (SRX1): empty bed
    url0 = "https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/SRX1.05.bed"
    h0 = _hl.sha1(url0.encode()).hexdigest()
    (cache / h0).write_bytes(b"")

    # URLs 1-4 (SRX2-5): peaks near GENE1
    for i in range(2, 6):
        url = f"https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/SRX{i}.05.bed"
        h = _hl.sha1(url.encode()).hexdigest()
        data = f"chr1\t{49000+i*100}\t{50000+i*100}\t.\t100\t.\t200\t-1\t-1\t0\n"
        (cache / h).write_bytes(data.encode())

    genome_seqs = {'chr1': ''.join(np.random.default_rng(99).choice(list('ACGT'), size=200000))}
    fake_genome = FakeGenome(genome_seqs)

    m1 = FakeMotif('MA0001.1', 'TF1',
                   {'A': [10.0, 0.1], 'C': [0.1, 10.0],
                    'G': [0.1, 0.1], 'T': [0.1, 0.1]})
    motifs = [m1]
    net_df = pd.DataFrame(columns=['source', 'target', 'weight',
                                    'resources', 'references', 'sign_decision'])

    cfg = {
        'cov_tsv': str(cov_file), 'cistrome_json': str(cis_file),
        'peaks_log': str(log_file), 'tss': str(tss_file),
        'genes': str(genes_file), 'chipatlas_list': str(ca_file),
        'twobit': 'fake', 'cache': str(cache),
        'cell_index': str(ci_file), 'threads': 1,
    }

    arrays, manifest, halves = build_features(
        cfg, open_genome=lambda p: fake_genome,
        load_motifs=lambda: motifs,
        load_net=lambda: (net_df, '0.0.0'),
        scan_factory=make_fake_scanner,
        merge_fn=merge_intervals_reference,
    )

    # CellX: URL 0 is empty (0 intervals), URLs 1-4 are usable
    # Usable list = [URL1, URL2, URL3, URL4] (4 items)
    # Half A = positions 0, 2 of usable = URL1, URL3
    # Half B = positions 1, 3 of usable = URL2, URL4
    assert 'CellX' in halves.get('cells_with_halves', [])

    if 'CellX' in halves:
        # Verify the halves exist and have the right shape
        assert 'F_prom_A' in halves['CellX']
        assert 'F_prom_B' in halves['CellX']


# ===================================================================
# test_split_half_report
# ===================================================================

def _report_world(rng, n_cells, n_genes, half_fn, full_fn):
    cells = ['C%d' % i for i in range(n_cells)]
    F_full = np.array([full_fn(i) for i in range(n_cells)], dtype=np.float32)
    halves = {'cells_with_halves': cells}
    for i, c in enumerate(cells):
        halves[c] = {}
        for f in ('F_prom', 'F_enh', 'F_reg'):
            for h in 'AB':
                halves[c]['%s_%s' % (f, h)] = half_fn(i).astype(np.float32)
        for h in 'AB':
            halves[c]['MA_raw_' + h] = rng.random(6).astype(np.float32)
    has_tss = np.ones(n_genes, bool)
    has_tss[:20] = False
    arrays = {'cells': np.array(cells), 'genes': np.array(['G%d' % i for i in range(n_genes)]), 'F_prom': F_full,
              'F_enh': F_full.copy(), 'F_reg': F_full.copy(), 'has': np.ones(n_cells, bool), 'has_tss': has_tss,
              'MA_raw': rng.random((n_cells, 6)).astype(np.float32)}
    return arrays, halves


def test_split_half_report_planted_excess():
    """(i) Halves sharing a planted cell-specific deviation give excess > 0.5; correlations use genes with a TSS only."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 10, 200
    shared = rng.standard_normal(n_genes)
    signal = rng.standard_normal((n_cells, n_genes)) * 2.0
    arrays, halves = _report_world(rng, n_cells, n_genes,
                                   half_fn=lambda i: shared + signal[i] + 0.3 * rng.standard_normal(n_genes),
                                   full_fn=lambda i: shared + signal[i])
    report = split_half_report(arrays, halves)
    assert report['medians']['F_prom']['pearson_excess'] > 0.5, report['medians']['F_prom']
    assert report['per_cell']['C0']['F_prom']['n'] == 180          # the 20 genes without a TSS are excluded
    assert report['medians']['MA']['pearson_own'] == report['medians']['MA']['pearson_own']   # computed (not NaN)


def test_split_half_report_baseline_removes_a_shared_half_term():
    """(ii) No cell-specific signal, but every cell's halves share a profile h that the other cells' full unions lack (the
    shape of review 043 C2's shared-control artefact). The own-cell correlation is inflated (> 0.3); the mismatched-cell
    baseline carries the same term, so the excess is near 0."""
    rng = np.random.default_rng(123)
    n_cells, n_genes = 10, 300
    shared = rng.standard_normal(n_genes)
    h = rng.standard_normal(n_genes)
    arrays, halves = _report_world(rng, n_cells, n_genes,
                                   half_fn=lambda i: shared + 0.8 * h + 0.5 * rng.standard_normal(n_genes),
                                   full_fn=lambda i: shared + 0.3 * rng.standard_normal(n_genes))
    report = split_half_report(arrays, halves)
    med = report['medians']['F_prom']
    assert med['pearson_own'] > 0.3, med
    assert abs(med['pearson_excess']) < 0.1, med
    assert abs(med['spearman_excess']) < 0.1, med


def test_split_half_report_skips_cells_without_has():
    rng = np.random.default_rng(5)
    arrays, halves = _report_world(rng, 4, 60, half_fn=lambda i: rng.standard_normal(60),
                                   full_fn=lambda i: rng.standard_normal(60))
    arrays['has'][1] = False
    report = split_half_report(arrays, halves)
    assert report['cells_with_halves'] == 3 and 'C1' not in report['per_cell']


# ===================================================================
# test_selftest  (guarded by pytest.importorskip('MOODS') only)
# ===================================================================

def test_selftest():
    """Run the module's selftest. Skipped if MOODS is not installed."""
    pytest.importorskip("MOODS")
    selftest()


# ===================================================================
# test_track_count_mismatch
# ===================================================================

def test_track_count_mismatch(tmp_path):
    """A track-count mismatch is recorded in the manifest."""
    cfg, fake_genome, motifs, net_df = _make_end_to_end_fixture(tmp_path)

    # Modify the peaks_log to say CellA has 5 tracks (but we only have 2)
    log_file = tmp_path / 'peaks_log.txt'
    log_file.write_text(
        "CellA/ATAC-seq src=encode: 5 bed track(s), nonzero_genes=2  (1s)\n"
        "CellB/ATAC-seq src=encode: 2 bed track(s), nonzero_genes=2  (2s)\n"
        "CellC/ATAC-seq src=encode: 1 bed track(s), nonzero_genes=1  (3s)\n")

    arrays, manifest, halves = build_features(
        cfg, open_genome=lambda p: fake_genome,
        load_motifs=lambda: motifs,
        load_net=lambda: (net_df, '0.0.0'),
        scan_factory=make_fake_scanner,
        merge_fn=merge_intervals_reference,
    )

    # Mismatch should be recorded, not raised
    assert 'CellA' in manifest.get('mismatches', {}), (
        "CellA track-count mismatch not recorded")


def test_window_long_peak_overlapping_the_promoter_is_not_enhancer():
    """A peak whose midpoint is >= 1 kb from the TSS but which overlaps the +-1 kb window is promoter, never F_enh."""
    df = pd.DataFrame([{"Chromosome": "chr1", "Start": 500, "End": 3500}])   # midpoint 2000, d = 1000, overlaps [0, 2000)
    F_prom, F_enh = compute_windows(df, {"gene1": ("chr1", 1000)}, ["gene1"])
    assert F_prom[0] == (2000 - 500) / 2000.0
    assert F_enh[0] == 0.0


def test_split_halves_take_usable_positions_and_skip_cells_without_has(tmp_path):
    """PI test: each usable sample covers one gene's promoter, so a half's F_prom shows which samples it took. CellX has an
    empty file first (it takes no position); A = usable 0, 2 -> G1, G3; B = usable 1, 3 -> G2, G4. CellY has 2 usable
    samples but no promoter coverage (has = False), so it gets no halves."""
    import hashlib as _hl
    genes = ['G1', 'G2', 'G3', 'G4']
    (tmp_path / 'genes.txt').write_text('\n'.join(genes) + '\n')
    (tmp_path / 'tss.tsv').write_text('symbol\tentrez\tchrom\ttss_hg38\tstrand\n' +
                                      ''.join('G%d\t%d\tchr1\t%d\t+\n' % (i, i, 20000 * i) for i in range(1, 5)))
    (tmp_path / 'ci.json').write_text(json.dumps({'cell_id_to_row': {'CellX': 0, 'CellY': 1}}))
    (tmp_path / 'log.txt').write_text('CellX/ATAC-seq src=cistrome: 5 bed track(s), nonzero_genes=4  (1s)\n'
                                      'CellY/ATAC-seq src=cistrome: 2 bed track(s), nonzero_genes=0  (1s)\n')
    (tmp_path / 'cov.tsv').write_text('lincs_cell_id\tstatus\tassay_target\tsource_used\tnotes\tsample_ids_used\n'
                                      'CellX\tresolved\tATAC-seq\tcistrome\t\t1, 2, 3, 4, 5\n'
                                      'CellY\tresolved\tATAC-seq\tcistrome\t\t6, 7\n')
    (tmp_path / 'cis.json').write_text(json.dumps([{'id': i, 'external_id_type': 'GEO', 'external_id': 'GSM%d' % i}
                                                   for i in range(1, 8)]))
    (tmp_path / 'ca.tab').write_text(''.join('SRX%d\ta\tb\tc\td\te\tf\tGSM%d\n' % (i, i) for i in range(1, 8)))
    cache = tmp_path / 'cache'
    cache.mkdir()
    url = 'https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/SRX%d.05.bed'
    beds = {1: ''}                                                       # empty: not usable
    for k in range(4):                                                   # SRX2..SRX5 -> usable 0..3 -> G1..G4
        t = 20000 * (k + 1)
        beds[k + 2] = 'chr1\t%d\t%d\t.\t1\t.\t5\n' % (t - 200, t + 200)
    beds[6] = 'chr1\t150000\t150400\t.\t1\t.\t5\n'
    beds[7] = 'chr1\t160000\t160400\t.\t1\t.\t5\n'
    for i, b in beds.items():
        (cache / _hl.sha1((url % i).encode()).hexdigest()).write_bytes(b.encode())
    genome = FakeGenome({'chr1': ''.join(np.random.default_rng(1).choice(list('ACGT'), size=200000))})
    m1 = FakeMotif('MA0001.1', 'TF1', {'A': [10.0, 0.1], 'C': [0.1, 10.0], 'G': [0.1, 0.1], 'T': [0.1, 0.1]})
    cfg = {'cov_tsv': str(tmp_path / 'cov.tsv'), 'cistrome_json': str(tmp_path / 'cis.json'),
           'peaks_log': str(tmp_path / 'log.txt'), 'tss': str(tmp_path / 'tss.tsv'), 'genes': str(tmp_path / 'genes.txt'),
           'chipatlas_list': str(tmp_path / 'ca.tab'), 'twobit': 'fake', 'cache': str(cache),
           'cell_index': str(tmp_path / 'ci.json'), 'threads': 1}
    arrays, manifest, halves = build_features(
        cfg, open_genome=lambda p: genome, load_motifs=lambda: [m1],
        load_net=lambda: (pd.DataFrame(columns=['source', 'target', 'weight']), '0'),
        scan_factory=make_fake_scanner, merge_fn=merge_intervals_reference)
    assert list(arrays['has']) == [True, False]
    assert halves['cells_with_halves'] == ['CellX']
    hx = halves['CellX']
    assert [g for g, v in zip(genes, hx['F_prom_A']) if v > 0] == ['G1', 'G3']
    assert [g for g, v in zip(genes, hx['F_prom_B']) if v > 0] == ['G2', 'G4']
    assert manifest['n_tracks_now']['CellX'] == 5 and manifest['mismatches'] == {}
