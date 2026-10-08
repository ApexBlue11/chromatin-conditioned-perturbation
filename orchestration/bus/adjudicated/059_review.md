# REVIEW OF PACKET 059
verdict: SOUND-WITH-CAVEATS
reviewed_commit: cdcccd8 (§94), 5ca47f6 (§92.11); packet 196ce9b

**§94 Stage A is the right kind of first step.** It measures the ceiling on the measured responses before asking the model, with
label-permutation nulls and a reading fixed before data.

**Two design choices bias it toward a null,** and a null here would be recorded as *"the measured responses do not carry these
readouts"*:
- **C1:** the single-protein label filter drops the strongest transcriptional classes, and most PI3K inhibitors.
- **C2:** A3's expected signs ignore cell genotype and receptor status, in cells where the biology says the sign is wrong or the
  effect is absent.

Both are identity-level fixes, so they can be made now without reading a response. C3–C7 are MINOR.

**§92.11 (E3) is sound,** with one MINOR (C8: fully tied channels should become missing, not constant). One side result
strengthens E3: **H3K27me3's Z = 22,778 is now confirmed exactly for 19 of its 26 cells** by `E_peaks_log.txt`.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | label selection | **`target_type == SINGLE PROTEIN` keeps 903 of the 1,363 labelled compounds in `chembl_dti_edges.tsv`.** The 460 dropped are labelled only through PROTEIN FAMILY, COMPLEX or COMPLEX GROUP targets. Among the dropped strings are **26S proteasome, tubulin, histone deacetylase, cyclin-dependent kinase, topoisomerase and DNA inhibitors**: the classes with the most reproducible L1000 signatures, CMap's usual positives.<br>**A3's PI3K row is hit directly:** of the 104 "PI3-kinase" rows, **96 are PROTEIN COMPLEX GROUP** and only 8 SINGLE PROTEIN. Pan-class-I PI3K inhibitors would mostly be missing from the class.<br>**What the filter doesn't do** is exclude non-human targets, which is the filter that matters (bacterial ribosome and penicillin-binding protein "mechanisms" are meaningless in human cells). | **For both readouts, fixed now:**<br>- keep direct interactions with `organism == Homo sapiens` and any protein target type (SINGLE PROTEIN, PROTEIN COMPLEX, PROTEIN COMPLEX GROUP, PROTEIN FAMILY, SELECTIVITY GROUP, CHIMERIC PROTEIN);<br>- **A1:** the MoA string defines the class;<br>- **A3:** class membership uses the rows' `gene_symbol` (complex-group rows list PIK3CA, PIK3CB, …).<br>Recount the scored cells under that rule (identities only) and record the counts. |
| 2 | MAJOR | expected sign | **Three of A3's rows have expected signs that the scored cells' known genotypes contradict or void.**<br>**MAPK (RAF targets):** first-generation RAF inhibitors *activate* ERK output in BRAF-wild-type and RAS-mutant cells (paradoxical activation). Of the five scored cells only **HT29 is BRAF V600E**. MDAMB231 (KRAS G13D), HS578T (HRAS G12D) and THP1 (NRAS G12D) are RAS-mutant, and MCF7 is RAS/RAF wild-type.<br>**p53 (MDM2):** an MDM2 inhibitor activates p53 only in TP53-wild-type cells. Only **MCF7** is; HT29 (R273H), MDAMB231 (R280K) and HS578T (V157F) are mutant, and THP1 is null.<br>**ER / AR:** receptor-negative cells (ER: all but MCF7) can't respond.<br>**Why it biases toward a null:** units in non-responsive cells give d ≈ 0 ± noise, about half of them positive. That breaks A3's *"d > 0 in ≥ 2/3 of units"* even when responsive units are strongly positive, and dilutes T. | **Gate the table before the data,** on genotype and receptor status (not outcome):<br>- **MAPK:** MEK/ERK targets (MAP2K1/2, MAPK1/3) in every cell; RAF targets only in BRAF-V600 cells (HT29);<br>- **p53:** TP53-wild-type cells only (MCF7);<br>- **ER:** ER-positive cells (MCF7);<br>- **AR:** AR-expressing cells, by a fixed basal-expression rule, otherwise dropped.<br>Record the genotype sources. Reported, not read: the same statistics on the ungated table. |
| 3 | MINOR | confound | **Duplicate compounds become MoA-mates of themselves.** Among single-protein compounds, **20 names map to more than one `pert_id`** (BI-2536, clomifene, cimetidine, …). Two IDs of one molecule share every MoA string and their signatures correlate as replicates. That inflates A1 through identity rather than mechanism, and the label-permutation null can't remove it. | Collapse compounds by `parent_chembl_id` before scoring: one signature per parent per cell, from all its rows. Mates must be distinct parents. |
| 4 | MINOR | interpretability | **A null A1 or A3 can't separate "no mechanism signal" from "signatures too noisy or inert to show any".** CNS and other ligands that are inert in these cells (your concern), more of them under C1, push A1_c toward the null as a mean over compounds. The two small cells (HS578T 32, THP1 28) have few compounds with mates. | **Reported, never read, per scored cell:**<br>- a **self-retrieval ceiling** (each compound's rows split into fixed halves; the AUROC of its own other half against other compounds);<br>- A1 on a **label-blind active subset** (compounds whose split-half self-correlation exceeds a threshold fixed now; labels permuted within the subset);<br>- the number of compounds with ≥ 1 mate, and the null sd.<br>Scope §94.5's null sentence by them: *"… do not carry these readouts (self-retrieval ceiling X; active-subset A1 Y)"*. |
| 5 | MINOR | decision logic | **"Neither has signal → move to the cold-drug split" skips a step.** Whether the measured responses carry mechanism depends on the cells and compounds, not on the split. If Stage A is null on the cold-cell test cells, a v9 training run on `split_cold_drug_1` (GPU) may meet the same ceiling. | Make the fallback **Stage A on the cold-drug split's test rows first** (measured data only, local CPU). Register the cold-drug v9 run only for a readout that carries signal there. |
| 6 | MINOR | selection | **Stage A reports the classes that carry A1. If Stage B then evaluates v9 on those classes, the classes are chosen on the test cells' measured responses.** There's no model output involved, but the selection still makes Stage B's ceiling optimistic. | State in §94.5 that Stage B uses Stage A's readout as registered (all scored cells and classes). Any class restriction is registered as such, with its source. |
| 7 | MINOR | power | **PROGENy top-500 restricted to 978 landmarks leaves an unknown, possibly small, footprint per pathway.** A pathway with very few landmark genes gives ULM activities dominated by noise. | Before the data, record each pathway's landmark footprint size. Drop any pathway below a fixed minimum (for example 15 genes) and its class row. |
| 8 | MINOR | E3 design | **Tying a channel with no non-tied entries makes it a present constant, not a missing one.** SKBR3's ATAC has 0 nonzero genes (`E_peaks_log`), so under `tie` it's 100 % one value. After average-rank rank-normal that's an **all-zero vector with the mask still on**. HME1's ATAC (9 nonzero) is close. E1 already treats the H3K27me3 tracks with < 10 peaks as **missing**, and the same logic applies here. SKBR3 is one of E2/T4's "help" cells, so how E3 encodes it matters for the per-cell report. | In `--chromatin_encoding tie`, mask as missing every (cell, mark) with fewer than 10 non-tied genes, mirroring the failed-ChIP rule. List what it masks (SKBR3 ATAC; HME1 ATAC) in 92.11 before the code. The decision table and everything else are unchanged. |

## Answers to the asks

**Ask 1 — A1 and A3 are the right first readouts, once C1–C4 are in.**
- **Seen compounds** (cold-cell split): irrelevant to Stage A, since only measured data is read. For Stage B, μ already carries
  each compound's training-cell response, so μ as the reference is exactly right. v9 must beat μ.
- **Cell-centring** is standard and removes the cell's baseline shift. It also removes part of a class's own effect where that
  class is a large share of the cell's compounds. That's conservative, and fine.
- **Inert CNS classes:** C4's active subset and ceiling, rather than dropping classes by judgement.
- **Multi-target compounds:** "a mate shares any MoA string" is fine for A1. In A3, a compound in two classes enters both, which
  is fine with C2's gating.
- **The bars:** +0.05 AUROC with p < 0.01 in ≥ 3 of 5 cells is reasonable. Whether the third cell is in reach depends on the
  small cells' power, hence C4's counts.
- **The A3 table:** defensible after C1 and C2. If you want a robust positive control row, EGLN1/2/3 inhibitors → Hypoxia (+)
  act in essentially every cell. It's optional, and must be fixed now if used.

**Ask 2 — §94.5 is sound in structure.**
- **Signal → Stage B against μ and the measured ceiling:** right.
- **The fallback** needs C5, and the null sentence needs C4's scope.

**Ask 3 — E3: no defect beyond C8.**
- **The decision table** is consistent with 92.9 item 4's pattern.
- **The comparator** reuses 92.9 item 3: Δ ≥ max(0.003, 2√(s²/3 + s²/3)) and per-cell > 0 in ≥ 4 of 6.
- **The freeze order** (registration before E1's seeds 1–2; code after E1's read) is right.
- **On TIE's H3K27me3 Z:** for the **19 cells** whose H3K27me3 appears in `E_peaks_log.txt`, each cell's zero count (covered −
  nonzero_genes) **equals exactly** its count inside the estimated block (19 of 19; 16,830 entries). The remaining 5,948 entries
  are the 7 cells absent from that log (A549, A673, HCT116, HEPG2, HL60, MCF7, PC3). So Z = 22,778 is exact for 19 of 26 cells
  and an estimate only for those 7.
- **A provenance side-note for 93.16:** step13's docstring says channel 2 is bigWig coverage (step12) throughout. But for those
  19 cells, its zero pattern matches step10's narrowPeak counts exactly. So the channel looks like a narrowPeak / coverage
  hybrid. Worth one line in 93.16.

## What I checked and found sound

- **The registrations:** §94.1–94.6 and §92.11 in full, and §92.9 items 1–4, for the comparator E3 reuses.
- **`chembl_dti_edges.tsv`** (identities only):
  - target-type counts; compounds kept and dropped by the single-protein filter, and the dropped MoA strings;
  - the PI3K rows' target types and gene symbols;
  - duplicate names across `pert_id`s.
- **`E_peaks_log.txt`:** (cell, mark) channels with < 60 nonzero genes (C8), and the per-cell H3K27me3 zero counts against
  `E_final`'s estimated block.
- **Cell genotypes used in C2** (standard cell-line annotations): HT29 BRAF V600E and TP53 R273H; MDAMB231 KRAS G13D and TP53
  R280K; HS578T HRAS G12D and TP53 V157F; THP1 NRAS G12D and TP53-null; MCF7 TP53-wild-type and ER-positive.

## What I could not assess, and why

- **PROGENy's landmark footprint sizes.** That needs the decoupler download, which I didn't do. Hence C7 asks for it before data.
- **The scored-cell counts under C1's broader label rule.** That needs the P7 rows' compound list per cell. It's an identity-only
  recount for the PI.
- **Whether `X_ctl` is plate-matched.** If it isn't, plate or batch effects shared by same-class compounds profiled together could
  inflate A1. Worth one line in 94.2 stating how `X_ctl` is formed.
