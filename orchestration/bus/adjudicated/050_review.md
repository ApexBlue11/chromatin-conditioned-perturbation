# REVIEW OF PACKET 050
verdict: SOUND-WITH-CAVEATS
reviewed_commit: df321c5

**§93's structure is right.**
- **Stage 1a** reuses §91's machinery, rows of record, bars and calibration.
- **Stage 1b** freezes its data by sha1 before any test.
- **H1** uses a same-data promoter reference.
- **The scope** is dev and training cells only, and writing it after §91 and §92 is disclosed.

**Three things must change before anything runs:**
- **C1:** the packet's "new fact" about VCAP is wrong about v9's input, and the H1 manifest is built from the coverage report,
  not from what v9 assembled.
- **C2:** the N1 conjunct in H3 and H1 is a count with no magnitude, so a gene-generic effect can pass.
- **C3:** H3's calibration lacks §91's fault checks and an MDE rule.

C4–C6 are MINOR.

**88.7 item 2 (review 049): checked.** `train_v9_gpu.py:145` seeds and `:198` builds the model. In between are config, CUDA-only
reseeding, data loading and `build_splits`, and I find no torch-CPU random draw in `data_v9.py`. The statement "from the code
path, not verified against weights" is accurate.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | provenance | **VCAP has no accessibility track in v9 at all, so the "new fact" (VCAP's ATAC "is an EpiMap imputed track") is false about v9's input.** Four sources agree: `E_final_mask` has VCAP's ATAC channel empty; `E_peaks_log.txt:129` reads *"VCAP/ATAC-seq src=epimap: 0 bed tracks (masked)"*; the funnel's `T0.v9.marks_dev` gives VCAP `[H3K27ac, H3K27me3]`; and F9 panel c already labels VCAP *"K27ac / K27me3 (failed)"*. The same is true of AGS, NPC and PHH: their H3K4me3 "proxies" gave 0 tracks and were masked. In fact **every** EpiMap entry gave 0 tracks. The coverage report lists sources, but `step13_merge_hybrid_tensor.py` takes channel 0 from `E_peaks`, and those entries never got there. As a result `atac_sources_v9_93.json` (25 cells, "Cistrome 18, ENCODE 6, EpiMap 1") includes **6 cells whose accessibility never entered v9**: A375 (Cistrome ATAC, 0 tracks), AGS, NOMO1 (ENCODE DNase), NPC, PHH and VCAP. For these 25 cells, v9's ATAC channel is **19 cells: 14 Cistrome + 5 ENCODE**. Separately, SKBR3's listed ATAC samples gave `nonzero_genes=0` at assembly (`E_peaks_log`), yet its channel is present. So "the same samples v9 used" needs checking per cell even where the channel exists. | **§93.1 H5:** *"VCAP has no accessibility track in v9 (its EpiMap entry yielded no tracks); its chromatin is measured H3K27ac plus a failed H3K27me3. AGS, NPC and PHH likewise have no accessibility track."* **H1's sample list:** build it from what produced `E_peaks` channel 0 (the assembly log and provenance), not from `coverage_report_final.tsv`. Decide now whether A375 and NOMO1, which ChIP-Atlas or ENCODE may now serve, are excluded or included and labelled *"accessibility new to v9"*. In the second case, their F_prom is not "the same data as v9". |
| 2 | MAJOR | stats | **The N1 conjunct in H3 and H1 is a count with no magnitude, so a gene-generic effect can advance on a coin flip.** H3 advances if S(C) − S(B) ≥ 0.004 **and** C > N1 in ≥ 4 of 6 cells. §91.12 found that chromatin's transferable content is gene-generic, which N1 keeps. Trees can use gene-level chromatin as a gene identifier and fit gene-generic residual structure, so S(C) − S(B) could clear the bar on content N1 has equally. The per-cell C − N1 signs would then be noise: P(≥ 4 of 6) = 0.34, and P(≥ 4 of 5) = 0.19 for H1. A pass would buy a GPU screen of what v9's gene embedding can already represent. This is review 041 C1's defect in the N1 conjunct. (§91's T1 had the same count-only N1 conjunct. Its real Δ was below the raw bar, so no reading changes.) | Add a magnitude: **S(C) − S(N1) ≥ half the raw bar** (+0.002 on the drug-known rows), together with the cell count, in H3 and in H1, with N1 built from that test's own features. Add one test: a planted gene-generic effect (the same f\* for every cell) at a size clearing the raw bar must fail. |
| 3 | MAJOR | stats | **H3's calibration is under-specified for the most flexible learner in the programme.** It has no π = 0 null draws and no P3 (drug-independent) case, so neither of §91.9's VOID rules is carried over. With 3 draws there is no MDE rule: "misses P1 at 2 %" doesn't say 3 of 3 or 2 of 3. Nor does it say whether each calibration draw re-runs the LOCO choice of min-child. It must, so that the test is calibrated as run (review 041 C2). | Carry §91.9 over: π = 0 in each draw (any pass → VOID); P3 at 5 % (any pass → VOID); **MDE = the smallest π passing in 3 of 3 draws**; LOCO min-child chosen inside every calibration draw; C2's gene-generic planted case. State the fault and MDE readings in `read_*`, committed before the run, as for §91. |
| 4 | MINOR | wrong-quantity | **H2's "rising" read and its licensed sentence don't say which Δ, and only FBC − N1 answers H2.** FBC − FB is mostly gene-generic (N1 keeps +0.0012 of +0.0014). Its gene-generic part also improves with k, through better coefficient estimates, so a rise in FBC − FB says nothing about learning a cell-specific rule. Two more weaknesses: k = all is a single fit with no spread, and 8-of-11 subsets overlap heavily. | Read "rising" on **FBC − N1** only, with FBC − FB reported. Draw the planted-P1 curve on the **same** subsets. Note that k = all has no spread. |
| 5 | MINOR | feasibility | **Stage 1a's "about 1 h" doesn't cover H3, and several of H1's free choices aren't fixed.** H3 fits per (row, gene): the covered dev-train cells' drug-known rows × 978 genes is tens of millions of samples. Then there's LOCO × 3 min-child values × 3 arms, plus a calibration of 2 forms × 3 π × 3 draws (plus C3's π = 0 and P3 cases), each with its own LOCO. H1 says "e.g. `pychromvar` / MOODS" and leaves the motif threshold, background, CollecTRI version, TSS source and peak-merging rule open. | Fix now, outcome-free: H3's row and gene subsampling (seeded) if any, applied identically in calibration; threads; the session plan, so the run fits Kaggle's 12 h session or splits by stage. For H1: one motif tool with its p-value and background; the CollecTRI release; TSS from `tss_hg38.tsv`; merged-union peaks by a stated `bedtools merge` rule. |
| 6 | MINOR | design | **H1 compares replacement, but what a pass buys is an addition.** H1 tests FB + {F_enh, F_reg} against FB + F_prom. What a pass buys is a v9 input **added** to v9's existing promoter chromatin. Peaks cluster, so the windows are correlated. Replacing one with the other can miss a real increment, or show a difference that isn't additive value. | Make the incremental comparison binding: **FB + F_prom + {F_enh, F_reg} against FB + F_prom**. The replacement comparison can be reported. |

## Answers to the asks

**Ask 1 — H3 is decisive if C2 and C3 are in. H2 can't be decisive, and it shouldn't gate investment.**
- **H2 at k ≤ 11:** a flat curve doesn't bound what 30 cells would give. A rising FBC − N1 curve is weak positive evidence,
  given that the cells aren't a random sample of cell space. "Reported only" is right.

**Ask 2 — the fixed settings are tight enough against tuning on dev.** Fixed trees, leaves and learning rate, with one
LOCO-chosen parameter over 3 values, chosen without dev rows, is fine. Its compute and calibration need C3 and C5.

**Ask 3 — answers on H1's reference, the cell rule and LOCO.**
- **F_prom:** it's the right reference: same peaks, promoter window. Make the comparison incremental (C6).
- **≥ 4 of 5:** right. Its chance rate (0.19) is stricter than 4 of 6 (0.34).
- **LOCO:** not as co-primary. The same LOCO folds choose λ, so evaluating on them is optimistic unless nested. And "either
  passes" would give two chances at a pass. Keep it secondary and nested, and report it.

**Ask 4 — no change. VCAP has no ATAC in v9 (C1).**
- **§91.11:** VCAP's T4 gain is its failed H3K27me3 (T4b), and that stands.
- **§92's VCAP numbers:** they need no rewording.

**Ask 5 — H5 as drafted screens only H3K27me3. The assembly log shows sparse tracks on other marks, two of them in dev cells.**
- **HEK293T's ATAC:** `E_peaks_log` gives **63** genes with a peak (the failed K27me3 tracks have 4–10). In `E_final`, its
  z-scored max is **7.8**, against 2.2–2.9 for the other dev cells' ATAC. Its reliability is 1.00. HEK293T is a T4 harm cell
  whose harm sits in its good marks (T4 against T4b), and this is the heavy-tailed one.
- **SKBR3's ATAC:** 0 genes with a peak from 3 tracks, yet present.
- **Other cells:** AGS K27ac 73, HUH7 ATAC 33, HME1 ATAC 9, SKMEL1 K27ac 19.
- **What H5 should add:** a sparsity and tail screen of every (cell, mark) from the assembly log, with these cases named. Any
  test that follows from it (e.g. ablating HEK293T's ATAC at inference, as T4b did for K27me3) would need its own packet.
- **The principal's two other directions** (chromatin gating TF-regulon or pathway edges, and pretraining) belong after Stage 1,
  in their own packets, as you propose. C7 already tested chromatin gating on union-graph edges (+0.0020, below its bar), so a
  gating packet should say how its gate differs.

## What I checked and found sound

- **Accessibility provenance:**
  - `E_final_mask` against `coverage_report_final.tsv` and `atac_sources_v9_93.json`, cell by cell;
  - `E_peaks_log.txt`, all tracks and their `nonzero_genes`;
  - `step13_merge_hybrid_tensor.py` (channel 0 = `E_peaks` channel 0);
  - the funnel's `T0` marks;
  - `E_final` per dev cell (non-zero counts, max |z|).
- **The 88.7 code path:** `train_v9_gpu.py:145-198` and `probe_moa_88.py:171-172`; no CPU RNG draw in `data_v9.py`.
- **§93 against §91.2–91.12, `BARS['known']` and review 041:** the bars, the rows of record and the calibration forms.

## What I could not assess, and why

- **How `E_peaks`' dense features relate to the log's `nonzero_genes`:** SKBR3's ATAC is dense and has a normal tail despite 0
  genes with a peak. I didn't trace the `E_peaks` builder. This is part of C1's per-cell check.
- **The feasibility sampling of ChIP-Atlas files (4 of 7 usable):** I relied on your report.
