"""Tests for v10-C L2/L3 data: the ChIP-Atlas selection rule, the window signal, CCLE methylation and the kernel builder."""
import gzip, json, math, os, py_compile, sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'orchestration'))
import chromatin_select as sel  # noqa: E402
import chromatin_signal as sig  # noqa: E402
import make_v10c_data_kernels as mk  # noqa: E402


def row(srx, antigen, cell, title='ChIP of x; Homo sapiens; ChIP-Seq', log='20000000,95.0,10.0,20000', meta=(),
        acl=None, genome='hg38'):
    acl = acl or ('ATAC-Seq' if antigen == 'ATAC-Seq' else 'DNase-seq' if antigen == 'DNase-seq' else 'Histone')
    return [srx, genome, acl, antigen, 'Breast', cell, 'Primary Tissue=Breast', log, title] + list(meta)


# --- the cell match -------------------------------------------------------------------------------------------------
def test_exact_name_match_not_substring():
    rows = [row('SRX1', 'H3K27ac', 'C4-2'), row('SRX2', 'H3K27ac', 'SNU-C4')]
    s, _ = sel.select(rows, ['SNUC4', 'C42'])
    assert s['SNUC4']['ACT'] == ['SRX2'] and s['C42']['ACT'] == ['SRX1']


def test_aliases_and_ncih_rule():
    rows = [row('SRX1', 'H3K27ac', '293'), row('SRX2', 'H3K27ac', 'NCI-H1975'), row('SRX3', 'H3K27ac', 'hESC H9')]
    s, _ = sel.select(rows, ['HEK293', 'H1975', 'NCIH1975', 'WA09', 'HEK293T'])
    assert s['HEK293']['ACT'] == ['SRX1'] and 'HEK293T' not in s
    assert s['H1975']['ACT'] == ['SRX2'] and s['NCIH1975']['ACT'] == ['SRX2']
    assert s['WA09']['ACT'] == ['SRX3']


def test_hg19_rows_ignored():
    s, _ = sel.select([row('SRX1', 'H3K27ac', 'MCF-7', genome='hg19')], ['MCF7'])
    assert s == {}


# --- the axes ----------------------------------------------------------------------------------------------------------
def test_axes_and_tier2_proxies_unused():
    rows = [row('SRX1', 'ATAC-Seq', 'MCF-7'), row('SRX2', 'DNase-seq', 'MCF-7'), row('SRX3', 'EZH2', 'MCF-7'),
            row('SRX4', 'H3K4me3', 'MCF-7'), row('SRX5', 'H3K9me3', 'MCF-7')]
    s, recs = sel.select(rows, ['MCF7'])
    assert set(s['MCF7']) == {'ACC', 'PCG'}
    assert sorted(s['MCF7']['ACC']) == ['SRX1', 'SRX2'] and s['MCF7']['PCG'] == ['SRX3']
    assert {r['srx']: r['kind'] for r in recs}['SRX3'] == 'substitute'


def test_direct_beats_substitute_even_at_a_worse_tier():
    rows = [row('SRX1', 'EZH2', 'MCF-7', title='ChIP (ENCLB1); Homo sapiens; ChIP-Seq'),
            row('SRX2', 'H3K27me3', 'MCF-7'), row('SRX3', 'SUZ12', 'MCF-7', title='DMSO ChIP; Homo sapiens; ChIP-Seq')]
    s, _ = sel.select(rows, ['MCF7'])
    assert s['MCF7']['PCG'] == ['SRX2', 'SRX1']


# --- QC and the baseline tiers -----------------------------------------------------------------------------------------
@pytest.mark.parametrize('log', ['4000000,95,1,20000', '20000000,45,1,20000', '20000000,95,1,400', 'bad'])
def test_qc_floor(log):
    s, _ = sel.select([row('SRX1', 'H3K27ac', 'MCF-7', log=log)], ['MCF7'])
    assert s == {}


@pytest.mark.parametrize('title,meta', [
    ('MCF7 shGATA3 K27ac; Homo sapiens; ChIP-Seq', ()),
    ('H3K27ac E2 r1; Homo sapiens; ChIP-Seq', ()),
    ('ATAC 5FU; Homo sapiens; ATAC-seq', ('agent=100uM 5FU',)),
    ('rep1; Homo sapiens; ChIP-Seq', ('treatment=MG132 (24H)',)),
    ('rep1; Homo sapiens; ChIP-Seq', ('genotype=ARID1A KO',)),
    ('K562, delEX1, H3K27me3; Homo sapiens; ChIP-Seq', ()),
    ('rep1; Homo sapiens; ChIP-Seq', ('source_name=MCF7-tet-on-shCTR9',)),
    ('VCaP ATAC ENZ3w EtOH rep2; Homo sapiens; ATAC-seq', ()),
])
def test_perturbed_samples_excluded(title, meta):
    assert sel.baseline_tier(row('SRX1', 'H3K27ac', 'MCF-7', title=title, meta=meta)) is None


@pytest.mark.parametrize('title,meta,tier', [
    ('ChIP-seq from MCF-7 (ENCLB632NJV); Homo sapiens; ChIP-Seq', (), 'E'),
    ('H3K27ac DMSO; Homo sapiens; ChIP-Seq', (), 'C'),
    ('rep1; Homo sapiens; ChIP-Seq', ('treatment=Control',), 'C'),
    ('ATAC siCtrl rep1; Homo sapiens; ATAC-seq', (), 'C'),
    ('MCF-7 untreated; Homo sapiens; ATAC-seq', (), 'C'),
    ('MCF7.H3K27ac.rep1; Homo sapiens; ChIP-Seq', ('treatment=missing',), 'N'),
])
def test_baseline_tiers(title, meta, tier):
    assert sel.baseline_tier(row('SRX1', 'H3K27ac', 'MCF-7', title=title, meta=meta)) == tier


def test_rank_tier_then_library_then_peaks_and_k_cap():
    rows = [row('SRX1', 'H3K27ac', 'MCF-7', log='20000000,95,1,90000'),                                    # N
            row('SRX2', 'H3K27ac', 'MCF-7', title='DMSO; Homo sapiens; OTHER', log='20000000,95,1,99000'),  # C, OTHER
            row('SRX3', 'H3K27ac', 'MCF-7', title='DMSO; Homo sapiens; ChIP-Seq', log='20000000,95,1,1000'),  # C
            row('SRX4', 'H3K27ac', 'MCF-7', title='x (ENCLB9); Homo sapiens; ChIP-Seq', log='20000000,95,1,600')]  # E
    s, recs = sel.select(rows, ['MCF7'])
    assert s['MCF7']['ACT'] == ['SRX4', 'SRX3']
    assert all(r['n_eligible'] == 4 for r in recs)


def test_duplicate_rows_counted_once():
    rows = [row('SRX1', 'H3K27ac', 'MCF-7'), row('SRX1', 'H3K27ac', 'MCF-7'), row('SRX2', 'H3K27ac', 'MCF-7')]
    s, _ = sel.select(rows, ['MCF7'])
    assert s['MCF7']['ACT'] == ['SRX1', 'SRX2']


# --- the windows -------------------------------------------------------------------------------------------------------
def test_window_bounds_strand_and_clamp():
    assert sig.window_bounds(100_000, '+', 1_000, 20_000, 10**9) == (101_000, 120_000)
    assert sig.window_bounds(100_000, '-', 1_000, 20_000, 10**9) == (80_000, 99_000)
    assert sig.window_bounds(100_000, '+', -50_000, -2_000, 10**9) == (50_000, 98_000)
    assert sig.window_bounds(100_000, '-', -50_000, -2_000, 10**9) == (102_000, 150_000)
    assert sig.window_bounds(1_000, '+', -10_000, 10_000, 5_000) == (0, 5_000)
    assert sig.window_bounds(1_000, '+', -50_000, -2_000, 10**9) is None


class FakeBW:
    """Signal = 1.0 on [s1, e1) of chr1, nothing elsewhere; stats is the exact mean over the queried span."""
    def __init__(self, s1, e1, chroms=None):
        self.s1, self.e1, self.calls = s1, e1, []
        self._chroms = chroms or {'chr1': 10**6}

    def chroms(self):
        return self._chroms

    def stats(self, chrom, s, e, type='mean', exact=False):
        assert type == 'mean' and exact
        self.calls.append((chrom, s, e))
        ov = max(0, min(e, self.e1) - max(s, self.s1))
        return [None] if ov == 0 else [ov / (e - s)]


def test_window_means_values_zero_and_absent_chrom():
    genes = [('A', 'chr1', 500_000, '+'), ('B', 'chr1', 500_000, '-'), ('C', 'chrY', 10_000, '+')]
    bw = FakeBW(501_000, 520_000)                     # exactly A's downstream window
    vals, info = sig.window_means(bw, genes)
    names = [w[0] for w in sig.WINDOWS]
    a, b, c = vals
    assert a[names.index('down_1_20kb')] == pytest.approx(1.0)
    assert b[names.index('down_1_20kb')] == 0.0       # B's downstream is upstream of the TSS on the + axis
    assert b[names.index('up_2_50kb')] == pytest.approx(18_000 / 48_000)
    assert a[names.index('prom_1kb')] == 0.0
    assert all(math.isnan(x) for x in c) and info['absent_chroms'] == ['chrY'] and info['n_nan'] == len(names)


def test_load_tss_real_file():
    g = sig.load_tss(os.path.join(REPO, 'phase2_assembly', 'outputs', 'tss_hg38.tsv'))
    assert len(g) == 977 and 'MUC1' not in {x[0] for x in g} and g[0] == ('DDR1', 'chr6', 30884519, '+') and {x[3] for x in g} == {'+', '-'}


# --- CCLE methylation --------------------------------------------------------------------------------------------------
CCLE = ('TSS_id\tgene\tchr\tfpos\ttpos\tstrand\tavg_coverage\tMCF7_BREAST\tNCIH1975_LUNG\tDMS53_LUNG\tHT29_LARGE_INTESTINE\n'
        'DDR1_1\tDDR1\t6\t1\t2\t+\t30\t0.5\t0.1\tNA\t0.9\n'
        'DDR1_2\tDDR1\t6\t1\t2\t+\t30\t0.7\t\t0.2\tNA\n'
        'PAX8_1\tPAX8\t2\t1\t2\t-\t30\tNA\t0.3\t0.4\t0.0\n'
        'XYZ_1\tXYZ\t2\t1\t2\t-\t30\t1\t1\t1\t1\n')


def test_ccle_methyl_average_match_and_missing():
    m, info = sig.ccle_methyl(CCLE, ['DDR1', 'PAX8', 'RPS5'], ['MCF7', 'H1975', 'NCIH1975', 'HT29', 'A549'])
    assert set(m) == {'MCF7', 'H1975', 'NCIH1975', 'HT29'} and info['n_cell_columns'] == 4
    assert m['MCF7'][0] == pytest.approx(0.6) and math.isnan(m['MCF7'][1]) and math.isnan(m['MCF7'][2])
    assert m['H1975'] == m['NCIH1975'] and m['H1975'][0] == pytest.approx(0.1) and m['H1975'][1] == pytest.approx(0.3)
    assert m['HT29'][0] == pytest.approx(0.9) and m['HT29'][1] == 0.0


def test_ccle_methyl_requires_gene_column():
    with pytest.raises(ValueError, match='no gene or locus_id'):
        sig.ccle_methyl('a\tb\n1\t2\n', ['DDR1'], ['MCF7'])


# --- the kernel builder ------------------------------------------------------------------------------------------------
def test_builder_pins_outcome_free_and_shards(tmp_path):
    recs = [{'srx': f'SRX{i}'} for i in range(10)] + [{'srx': 'SRX3'}]
    sp = tmp_path / 'selection.json'
    sp.write_text(json.dumps({'records': recs, 'selection': {}}), encoding='utf-8')
    cp = tmp_path / 'cells.json'
    cp.write_text(json.dumps({'cells': {'MCF7': 1}}), encoding='utf-8')
    pins, built = mk.build(str(sp), str(cp), src_dir=str(tmp_path / 'src'), kern_dir=str(tmp_path / 'k'))
    assert set(pins) == {'selection.json', 'full_cells.json', 'tss_hg38.tsv', 'chromatin_signal.py'}
    for n, p in pins.items():
        assert mk.sha1(str(tmp_path / 'src' / n)) == p
    assert len(built) == mk.N_SHARD + 1
    for name, d in built.items():
        meta = json.load(open(os.path.join(d, 'kernel-metadata.json')))
        assert meta['enable_internet'] and not meta['enable_gpu'] and meta['dataset_sources'] == [mk.DATASET]
        assert meta['is_private'] and 'machine_shape' not in meta
        code = open(os.path.join(d, name + '.py'), encoding='utf-8').read()
        py_compile.compile(os.path.join(d, name + '.py'), doraise=True)
        assert repr(pins) in code and 'expression data mounted' in code
    # the shards partition the distinct tracks
    tracks = sorted({r['srx'] for r in recs})
    got = sorted(t for k in range(mk.N_SHARD) for t in tracks[k::mk.N_SHARD])
    assert got == tracks
    assert "SHARD, N_SHARD" in open(os.path.join(built['lincs-v10c-signal-s2'], 'lincs-v10c-signal-s2.py')).read()
    assert '= 2, 4,' in open(os.path.join(built['lincs-v10c-signal-s2'], 'lincs-v10c-signal-s2.py')).read()


# --- added after the first mutation pass (6 of 25 survived) ------------------------------------------------------------
def test_derivative_line_names_do_not_match():
    rows = [row('SRX1', 'H3K27ac', 'PC-3M'), row('SRX2', 'H3K27ac', 'MDA-MB-231-BoM'), row('SRX3', 'H3K27ac', 'PC-3')]
    s, _ = sel.select(rows, ['PC3', 'MDAMB231'])
    assert s == {'PC3': {'ACT': ['SRX3']}}


def test_ncih_rule_both_directions():
    rows = [row('SRX1', 'H3K27ac', 'H2009'), row('SRX2', 'H3K27ac', 'NCI-H1299')]
    s, _ = sel.select(rows, ['NCIH2009', 'H1299'])
    assert s['NCIH2009']['ACT'] == ['SRX1'] and s['H1299']['ACT'] == ['SRX2']


def test_peaks_break_ties_within_tier_and_library():
    rows = [row('SRX1', 'H3K27ac', 'MCF-7', log='20000000,95,1,1000'),
            row('SRX2', 'H3K27ac', 'MCF-7', log='20000000,95,1,30000'),
            row('SRX3', 'H3K27ac', 'MCF-7', log='20000000,95,1,9000')]
    s, _ = sel.select(rows, ['MCF7'])
    assert s['MCF7']['ACT'] == ['SRX2', 'SRX3']


def test_select_drops_perturbed_rows():
    rows = [row('SRX1', 'H3K27ac', 'MCF-7', title='MCF7 shGATA3 K27ac; Homo sapiens; ChIP-Seq', log='20000000,95,1,90000'),
            row('SRX2', 'H3K27ac', 'MCF-7', log='20000000,95,1,1000')]
    s, recs = sel.select(rows, ['MCF7'])
    assert s['MCF7']['ACT'] == ['SRX2'] and recs[0]['n_eligible'] == 1


def test_ccle_methyl_h_prefixed_column():
    text = 'gene\tavg_coverage\tH2009_LUNG\nDDR1\t30\t0.25\n'
    m, _ = sig.ccle_methyl(text, ['DDR1'], ['NCIH2009'])
    assert m['NCIH2009'][0] == pytest.approx(0.25)


def test_outcome_guard_raises():
    assert "raise SystemExit(f'FATAL: expression data mounted in an outcome-free kernel: {p}')" in mk.COMMON


def test_ccle_methyl_locus_id_format_of_the_real_file():
    """The 20181022 header and row shapes, as read from the file on 2026-10-17 (values padded with spaces, NaN)."""
    rows = [['locus_id', 'CpG_sites_hg19', 'avg_coverage', 'DMS53_LUNG', 'MCF7_BREAST'],
            ['DDR1_6_30884000_30885000', '6:30884100;6:30884200', '  25.00', '0.00000', '0.75000'],
            ['DDR1_6_30885000_30886000', '6:30885100', '   8.27', '    NaN', '0.25000'],
            ['HLA-DRB1_6_1_2', '6:1', ' 3.0', '0.5', '    NaN'],
            ['SGIP1_1_66998638_66999638', '1:66998970', ' 25.0', '1.0', '1.0']]
    text = ''.join(chr(9).join(r) + chr(10) for r in rows)
    m, info = sig.ccle_methyl(text, ['DDR1', 'HLA-DRB1', 'PAX8'], ['MCF7', 'DMS53'])
    assert info['n_cell_columns'] == 2 and set(m) == {'MCF7', 'DMS53'}
    assert m['MCF7'][0] == pytest.approx(0.5) and math.isnan(m['MCF7'][1]) and math.isnan(m['MCF7'][2])
    assert m['DMS53'][0] == 0.0 and m['DMS53'][1] == pytest.approx(0.5)
