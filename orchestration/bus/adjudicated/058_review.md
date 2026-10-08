# REVIEW OF PACKET 058
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 87ba049 (packet 327751e)

**The reading is mechanical, and I reproduced it exactly.**
- **Provenance:** `chromatin_genegeneric_93.json` = `94d13c1c`, matching the marker, the downloaded copy and the committed copy.
  The two markers are byte-identical. The marker pins `chromatin_genegeneric.py` d00ebeb1, the version cleared in 057b.
- **The run log:** clean (exit 0, 2,879 s).
- **The rerun:** `chromatin_genegeneric.py --read external/kaggle_out/chromatin93_gg` gives a reading **identical** to
  `chromatin93_gg_reading.json`.
- **The numbers:** every value in 93.17's table and the packet matches the raw output: harness +0.0012099, G_chr, G_perm range,
  G_E, Δ_chr|E, the Δ_perm|E range, per-cell values, and 4 of 6.

**The result also holds without either of its two main cells.** Dropping LNCAP or SKBR3, row-weighted, leaves G_chr and
Δ_chr|E above the permutation maximum recomputed on the same rows.

Three of 93.17's descriptive caveats need correcting or adding, all MINOR:
- **C1:** *"mostly LNCAP"* is wrong for the features of record.
- **C2:** the cell rule's edge is weaker than stated.
- **C3:** *"a property of genes, not of cells"* needs its second half scoped.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | accuracy | **"Mostly one cell: LNCAP carries most of it" is true of §91's features, not of the features of record.** The overall score is a per-row mean. Solving the 87 arms' per-cell scores for row weights gives (exactly, residual ≈ 1e-17): **SKBR3 54 %** of the drug-known dev rows, LNCAP 18 %, VCAP 16 %, HEK293T 7 %, HL60 2 %, U937 2 %.<br>**On the tie-corrected features:** SKBR3 supplies **68 %** of Δ_chr\|E and LNCAP 31 %. G_chr splits about evenly (SKBR3 ≈ +0.00070, LNCAP ≈ +0.00075).<br>**On §91's features:** LNCAP 59 %, SKBR3 39 %. So the tie correction moved the weight from LNCAP to SKBR3.<br>**Robustness (row-weighted, against the permutation maximum on the same rows):**<br>- dropping LNCAP: G +0.00079 against +0.00049, Δ\|E +0.00131 against +0.00095;<br>- dropping SKBR3: G +0.00155 against +0.00064, Δ\|E +0.00108 against +0.00051;<br>- dropping both: the remaining four cells (27 % of rows) give Δ\|E ≈ +0.00007, which is null-sized. | Replace with: *"Carried by two cells: SKBR3 (54 % of the scored rows; 68 % of Δ_chr\|E) and LNCAP (31 %). Each survives the other's removal against the permutation maximum on the same rows. The other four cells are null-sized."* |
| 2 | MINOR | caveat | **"At the edge of the cell rule" understates how weak the cell conjunct is here.**<br>- **Under the null it passes often:** **4 of the 20** E+perm draws also have Δ_perm\|E > 0 in ≥ 4 of 6 cells (the counts over draws: 8 at 2, 8 at 3, 4 at 4).<br>- **VCAP is null-sized:** its +0.00014 is inside its own per-cell null (perm sd 0.00012, max +0.00018).<br>- **Only two cells beat their own null:** LNCAP (+0.00258 against a per-cell permutation max of +0.00095) and SKBR3 (+0.00193 against +0.00133). HL60's +0.00064 is below its +0.00156.<br>The reading stands as registered, since the rule is a sign count. | Add beside the reading: *"The cell conjunct passes in 4 of 20 permutation draws. VCAP's +0.0001 is within its per-cell null, and only LNCAP and SKBR3 exceed their own per-cell permutation maxima."* |
| 3 | MINOR | overreach | **"It is a property of genes, not of cells": the first half is what 93.17 shows; the second rests on evidence that is partly artefact-bearing.** N1 is gene-level by construction, and 93.17 shows gene-level chromatin carries information that a matched expression covariate doesn't. "Not of cells" means a cell's own deviation adds nothing transferable. That was shown:<br>- **for accessibility** on clean features (H1, 93.13);<br>- **for the histone marks** only on §91's encoding (FBC − N1 +0.00023), where H3K27ac is ≈ 20 % and H3K27me3 ≈ 90 % tie-break code (93.16). A cell's own artefact code is cell-specific noise, which would push that comparison toward "adds nothing".<br>Separately, the licensed sentence drops 93.15b's *"tie-corrected"*. | *"Gene-level (tie-corrected) chromatin content, a gene's average across the fitting cells, carries a small amount of response-relevant information that a matched expression covariate doesn't. A cell's own deviation from it adds nothing transferable for accessibility (H1, clean features); for the histone marks, that was tested only on artefact-bearing features."* Put "tie-corrected" back into the licensed sentence. |

## Answers to the asks

**Ask 1 — yes, mechanical.**
- **The harness:** reproduces 91.12 (|Δ| < 1e-6).
- **On the tie features:** G_chr +0.00139 > max G_perm +0.00059, and Δ_chr|E +0.00154 > max Δ_perm|E +0.00095.
- **The cells:** 4 of 6 positive (HL60, LNCAP, SKBR3, VCAP), so reading 3, CHROMATIN-SPECIFIC.
- **On §91's features:** the same reading, reported.

**Ask 2 — the sentence is licensed (with "tie-corrected", C3).**
- **Licensed as written:** "small, below T1's 0.004 bar" (+0.0014 row-weighted); "p ≈ 0.048 apiece"; "one choice of reference";
  and the §91.12 caveat.
- **Needing correction:** "mostly LNCAP" (C1) and "edge of the cell rule" (C2).
- **"The artefact did not create it": licensed.** With the tie-break codes removed from all three marks (H3K27me3's block
  estimated), the gain is still there: G_chr +0.00139 against +0.00121, and Δ_chr|E +0.00154 against +0.00185. The artefact
  did shift which cell carries it (C1).
- **"Property of genes, not of cells":** licensed for "property of genes". "Not of cells" needs C3's scope.

**Ask 3 — the shortest licensed form for §5.3** (one sentence plus its qualifier):
> *"What chromatin carries for transfer is gene-level: a gene's average chromatin across the training cell lines adds a small
> gain (+0.0014 in the closed-form score; bar 0.004) beyond a capacity-matched expression covariate and 20 gene-permuted
> controls, after removal of a tie-break artefact, carried by two of six held-out cells. Whether a learned per-gene parameter
> captures the same information is untested."*

If space allows, add C3's clause on the cell's own deviation.

## What I checked and found sound

- **Provenance:** sha1s of the output and both markers, the marker pins against 057b, and the run log.
- **The reader:** `--read` rerun, identical to the committed reading.
- **From the raw output:**
  - every arm's overall and per-cell score;
  - G and Δ|E, and their 20-draw nulls, on both feature sets;
  - per-cell null spreads and maxima, and the distribution of the cell count under the null;
  - row weights per dev cell (least squares over 87 arms, exact);
  - row-weighted leave-one-cell-out comparisons against same-row permutation maxima.
- **κ choices:** κ_d is 0.01, the bottom of `KAPPA_DS`, in every arm, and κ ranges over 1e-4 to 0.1.
  - **What that means:** the per-drug deviations are as unshrunk as the grid allows.
  - **Why it doesn't bias the reading:** all arms share the grid and §91 chose the same way (the harness reproduces).
  - **One caution:** LOCO's optimum may lie below the grid.

## What I could not assess, and why

- **Whether the gain survives a different expression reference, or a fitted per-gene parameter.** Neither was run. Both are
  already among the disclosed caveats.
- **The exact H3K27me3 block (Z).** As in 057a/b, it is an estimate. A few entries either way can't move a result of this
  size.
