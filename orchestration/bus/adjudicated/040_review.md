# REVIEW OF PACKET 040
verdict: SOUND-WITH-CAVEATS
reviewed_commit: b81cf8f

**The funnel's structure is right.**
- **Closed form and cheap:** the tests are closed-form and CPU-only.
- **Dev rows only:** the funnel uses only dev rows, and no dev row enters any fit.
- **Clear gates:** each test has one estimand and a capacity-matched null (N1).
- **A pass only earns a screen:** it buys a separately priced GPU screen, never an adoption.

**I reproduce the row sets and coverage exactly.**
- **Rows:** `split_cold_cell_1` has 47,509 train rows. The six dev cells' train rows are exactly 4,043, with sorted sha1
  `51e7e4ab`, so no dev-cell row can enter the 43,466 dev-train rows.
- **Coverage:** 18 of 32 training cells have chromatin, 12 of the 26 dev-train cells, and the dev cells' marks are as
  stated.

**Two things need fixing before the script is written.**
- **C1 — the inputs:** the H3K27me3 tracks flagged as failed ChIP enter at full weight. That covers two of the six dev
  cells and is the *only* mark of one of the "12 covered" training cells.
- **C2 — the estimand:** T1 and T2 can pass on a drug-independent cell-mean offset, the very thing a drug-conditioned
  head would not add.

Also: the diagnosis's "nothing lets the chromatin effect depend on the drug" is false for v9 (C3), and N1 and T2's
similarity are under-specified (C4).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | provenance | **Failed-ChIP H3K27me3 tracks enter at full weight, and `PHH` is "covered" only by one.** `E_final_provenance.json` flags failed ChIPs (raw peaks < 10), and `E_reliability.tsv` gives their H3K27me3 a reliability of **0.30**. Four are in this funnel: **HEK293T** and **VCAP** (dev) and **HELA** and **PHH** (dev-train). §91.2 encodes chromatin "exactly as `xpert_arm.py` 190–194", but that z-scores per (cell, mark), which undoes any down-weighting, and `xpert_arm.py:209` sets `r` from the mask, so the reliability file is never read. **Measured:** max \|z\| is **11.4** (HEK293T), **10.1** (VCAP), **20.7** (HELA) and **28.5** (PHH), against 2.1–2.5 for good tracks (A549, HL60). Failed tracks correlate with good ones at a median across-gene r of **0.106**, against **0.270** good–good, so they're noise, not duplicates. **PHH's only mark is its failed H3K27me3**, so the "12 covered dev-train cells" are effectively 11. **What this does to each test:** in T1, a few genes at \|z\| 10–28 dominate the H3K27me3 coefficients through leverage. In T2, s_C for HEK293T and VCAP, and with neighbours HELA and PHH, averages a noise Pearson into the similarity. T3 is affected the same way. | Pre-register in §91.2: **H3K27me3 tracks with reliability 0.30 (the failed-ChIP flag) are missing in every funnel test.** That leaves PHH uncovered (11 covered dev-train cells), HEK293T with ATAC + H3K27ac, and VCAP with H3K27ac only. The as-v9 encoding becomes a reported secondary. Also record in §91.1 that v9's own input has the same property, so §55's null was measured with these tracks at full weight. P7 is unchanged. |
| 2 | MAJOR | code-vs-intent | **T1 and T2 can pass on the cell-mean delta profile, which is drug-independent, so a pass wouldn't support a *drug-conditioned* head.** T1's additive term `w·f[c,g]` is the same for every row of a cell. It shifts every prediction by one gene pattern, and the raw per-row Pearson rewards that whenever the cell's mean response correlates with `f`. That's a per-(cell, gene) offset, which v9's `SignedChromatinHead` already provides (`epi(E)·r`, `modules_v9.py:385-390`). It's exactly how C6 behaved: raw +0.0048, centred +0.0002 (§85.13). T2 can likewise gain by retrieving neighbours with a similar cell-mean profile. Yet C9 and C10 are both drug-dependent mechanisms. | **Add Δ_centred > 0 in ≥ 4 of 6 dev cells** (`score_dev.centred_r`) as a conjunct of the T1 and T2 advance rules, mirroring §85.11. Centring removes any per-cell constant exactly, so what survives is the drug-dependent part: T1's gain term `μ·v·f`, and δ_d and ε_d. Report T1's additive-only rule (v = δ = ε = 0) beside it so the decomposition is visible. |
| 3 | MINOR | wrong-quantity | **§91.1's "Nothing lets the chromatin effect depend on the drug" is false for v9.** Chromatin is summed into each gene token (`GeneRepresentation`: `epi_proj(E·r)`, `modules_v9.py:149-157`). The gene tokens then pass through the perturb blocks together with the drug tokens (`blk(h, D, …)`, `model_v9.py:197-200`), so v9 *can* express drug × chromatin interactions. Only the additive `SignedChromatinHead` is drug-independent. What v9 lacks is an *explicit* drug-conditioned chromatin term. That changes what a pass means: information exists that v9 could represent but evidently doesn't use (§55). So C9's case is "an explicit term may be learned where an implicit one wasn't", not "v9 cannot express it". | Reword §91.1 and IDEAS A12: *"v9's only explicit chromatin terms are drug-independent (the gene-token embedding and the additive head); drug × chromatin interactions are expressible only implicitly, through the gene tokens' cross-attention to the drug, and §55 shows they are not used."* Keep this sentence out of the paper in its current form. |
| 4 | MINOR | stats | **N1 and T2's s_C are under-specified in ways that change what they measure.** (a) **N1, refit or apply:** is the rule *refitted* with N1's features, or is FBC's fit *applied* to them? Only refitting tests "this cell's chromatin beats a gene-generic chromatin feature of the same capacity". Applying FBC's fit is an ablation. (b) **N1 and missing marks:** if N1 fills marks a cell doesn't have (U937 has ATAC only), N1 gets inputs FBC doesn't. (c) **N1 in LOCO:** during tuning, N1's gene mean includes the held-out cell's chromatin. (d) **T2 with no shared mark:** s_C is undefined for neighbours sharing no mark with the target (U937 against A375 and AGS, and PHH under C1; VCAP against HME1), yet "all variants use the same neighbour set". (e) **T3 has no N1 gate,** so it can pass on a gene-generic chromatin prior beyond `v̄_g`. | Pre-register: (a) N1 is **refitted**; (b) N1 replaces only the marks a cell has, using the mean over covered dev-train cells **that have that mark**; (c) inside LOCO the mean excludes the held-out cell; (d) a rule for s_C when no mark is shared, e.g. "those neighbours take the target's median s_C over its other neighbours", fixed now; (e) add ρ(FBC) − ρ(N1) > 0 in ≥ 4 of 6 cells to T3's advance rule, as T1 has. |

## Answers to the asks

**Ask 1 — N1 is the right null for T1 once C4 specifies it, and the basal comparator is the right bar for T2.**
- **N1 matches capacity:** it has the same features and parameters as FBC, so a pass over N1 can't be capacity alone.
- **T2's comparator:** v9 sees `x_ctl`, so "chromatin beyond basal similarity" is the right question.
- **T2's limit:** T2 is a *cell-level* method over 11–12 neighbours, the same small-n regime §91.1 holds against §44.
  Its power is low and a null from it says little. It's cheap, so keep it, but weigh its outcome accordingly.

**Ask 2 — I found no leakage in the row sets or the targets.**
- **Rows:** as above, the dev cells' train rows are exactly the pinned 4,043.
- **The design choices are right:**
  - μ^(−c) for fit targets;
  - LOCO over covered dev-train cells only;
  - the dev cells' own `X_ctl` as an input, since it's available at test time;
  - z-scoring within the cell, which carries no cross-cell information;
  - T3's dev `v` as an evaluation target only.
- **Two additions:**
  - Count δ_d's cell floor **within each LOCO fold**, excluding the held-out cell, so tuning mimics deployment.
  - N1's mean excludes the held-out cell (C4c).
- **The floor:** with the shrinkage chosen by LOCO, a floor of 3 cells is adequate. A stronger floor isn't needed,
  since the penalty already governs how much a three-cell deviation is trusted.

**Ask 3 — yes, as a necessary condition, with C2's centred conjunct.**
- **Noise:** the tests are deterministic, so there's no seed noise. Cell-sampling noise is handled by ≥ 4 of 6 cells.
- **Leniency:** gains over a closed-form base without v9's representation usually shrink once added to v9, so +0.003
  is lenient. That suits a funnel, since the GPU screen at +0.0034 is the real test.
- **Two routes:** "all rows or top tercile" is a mild multiplicity, acceptable here.
- **Headroom:** report S(B0) and S(FB) beside P2's 0.43693 so the headroom is visible.

**Ask 4 — two things are missing from the brainstorm.**
- **Does v9 use the chromatin it has?** The cheapest direct check is inference-time N2 on the network itself: swap or
  mean-ablate chromatin on P2's three dev checkpoints. It separates "information absent" (what the funnel measures)
  from "v9 doesn't read it" (what C9 would address). §55 was one training-time run per arm on the test cells, which is a
  different question.
- **The encoding itself is a candidate:** reliability weighting with no z-score re-inflation (C1).

**Ask 5 — none of T1–T3 duplicates earlier work.**
- **§44** is a linear cell-level block (24 SVD components over 40 cells), measured on the cold-cell **test** cells.
  T1's gene-local, drug-conditioned rule is a new functional form. T2 shares §44's cell-level, small-n regime but not
  its form.
- **§55** is a training-time ablation of v9. It tests use, not information.
- **§82.7** is diffusion of landmark targets over a graph, unrelated to T1–T3.
- **Test cells:** the funnel, unlike §44, never reads the test cells.

## What I checked and found sound

- **Rows:** 47,509 train rows and 32 training cells; the dev rows are all train rows of the six dev cells, with sha1
  matching §85's pin.
- **Coverage, from `E_final_mask.npy`** (identical to the v9 payload's copy):
  - 18 covered training cells, of which 12 are dev-train;
  - among those 12: ATAC 9, H3K27ac 10, H3K27me3 8, of which HELA and PHH are failed tracks;
  - dev marks: HEK293T, HL60, LNCAP and SKBR3 have all three; VCAP has H3K27ac + H3K27me3; U937 has ATAC only.
- **Code:** `score_dev.row_pearson` and `centred_r` exist (`score_dev.py:22`, `:58`). The chromatin paths are in
  `modules_v9.py` and `model_v9.py`, as cited in C3.

## What I could not assess, and why

- **The funnel script.** It doesn't exist yet, so this review covers the contract only. The script needs its own
  review before it runs.
- **Whether the ChEMBL target lists for the epigenetic-drug stratum are on disk.** That stratum is reported, never a
  gate.
