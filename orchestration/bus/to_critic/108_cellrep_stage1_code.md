# PACKET 108 — CODE REVIEW: stage 1 (RESULTS §103 / §103.1 / §103.2), before its one run
packet_id: 108
created: 2026-11-02
type: **CODE REVIEW** (no genetic signature value read; dry runs used fake values only)

**The code:** `model/cellrep/stage1.py`, with 12 tests in `model/cellrep/test_stage1.py` (12 / 12 pass). The tests cover:
- the leak test: the 103 definition scores −1 and the fixed one < 0.02;
- no training target changes when only held-out signatures change;
- r = 0 for a constant prediction;
- the percentile directions;
- two-way shares on known components;
- disjoint, alternating halves;
- a partial Spearman that removes a confound;
- the within-cell statistic ignoring cell shifts;
- dose parsing and the dose / time rule;
- the own-coordinate skip;
- a reproducible cluster bootstrap.

**New inputs since §103.2** (the principal approved them 2026-11-02):
- `GSE92742_Broad_LINCS_sig_metrics.txt.gz` (12.5 MB, sha1 `cdf16e53…`): reported strata. P0 and B1 by the `tas` tertile of the
  knockdown / drug signature; it covers trt_sh.cgs and phase-1 trt_cp.
- **Hairpin-level `trt_sh`** (review 107 C3): re-extracted from the same approved GCTX (sha1 `28d1e910…`; integrity diff 0.0).
  - 154,993 signatures (`l1000_trt_sh_hairpin_lm978.npz`, sha1 `46a1234d…`);
  - D1 is also computed with s_kd from hairpins targeting the 2,100 genes: about 13,000–15,000 rows per core cell at 96 h, own
    target excluded per landmark.
- Three CCLE files (global chromatin, all-genes RPKM, miRNA) are downloaded for stage 2. Stage 1 doesn't use them.

**Implementation choices to check:**
1. **B1 reads Level 5 drug signatures** (`Y_target_level5_978.npy` with `signatures.tsv`, the same MODZ z normalisation as the
   knockdowns), not the Level-3 substrate. The substrate supplies only the split membership:
   - the cells are PC3, A375, A549, HEPG2, every row `split_cold_cell_1 == train`;
   - the compounds are those with no `split_cold_drug_1 != train` row anywhere;
   - Level 5 compounds missing from the substrate are excluded.
2. **B1's ranking** uses one coordinate set per pair: t's own coordinate is excluded from every correlation in that pair's
   ranking (the other knockdowns keep theirs).
3. **The hub control** draws, with `default_rng(0)`, one gene from the 10 knocked-down genes in c nearest in STRING degree to t.
4. **D1's mask:** a cell's chromatin feature is used only if its axis is available (e.g. HT29 has no PCG).
5. **The noise ceiling** uses the 120 h signature when both 120 h and 144 h exist.

**Dry runs on fake values** (real metadata and inputs, random Y), with the P0 / D1 gates forced open:
- the leak test: fixed |r| ≤ 0.0008, leaky −1.0000;
- VD: MS_gxc / noise bound = 1.0009, exactly the pure-noise value;
- ceiling ≈ 0;
- B1: median 0.511, CI [0.495, 0.528] (no pass); top-5 % fraction 0.046; hub 0.508;
- B1's n: 3,312 pairs and 502 drugs; 0 unparseable doses.

## ASKS
1. **Is the code faithful to §103 / §103.1 / §103.2,** including the gates (P0 failing stops everything; D1 is computed only
   if its positive control passes)?
2. **Should the hairpin-level s_kd become D1's primary** (the matched construction, review 107 C3's "best"), with consensus
   signatures reported? Or the reverse? This is decided now, before the run.
3. **Are choices 1–5 acceptable?**
