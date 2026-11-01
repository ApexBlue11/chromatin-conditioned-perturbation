# REVIEW OF PACKET 106 (cell-representation programme, stage 1 design; §103, before any code or genetic value)
verdict: NOT-SUPPORTED
reviewed_commit: 36347c2

**Not supported as registered.** It has one blocking and three major design errors, all fixable in the registration text
before any value is read.

**The programme's logic is right.**
- Fixing the harness first, gating every rung on a positive control, and assigning each null to its first failed rung is the
  right order.
- The data provenance is solid. I confirmed every count in §103 from the extract's **metadata arrays only**, without reading
  `Y`:
  - 20,696 core `trt_sh.cgs` at 96 h, at most one per (gene, cell);
  - 3,927 genes, of which 3,271 are in ≥ 5 core cells and 2,100 in all six;
  - per-cell counts of 3,287–3,669;
  - VCAP at 96 h: 28;
  - 2,770 pairs with a 120 / 144 h partner;
  - record and npz sha1s as stated.

**But the harness, which every later stage inherits, has an exact leak in LOCO** (C1). Its noise ceiling is biased upward
(C2). D1's quantity is largely set by how L1000 normalises z-scores (C3). And B1's stated meaning needs a condition it
doesn't test (C4). Each is a change to the registration text, not new data.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | harness: LOCO leak (Ask 1) | **The target d sums to zero over a gene's cells, so in LOCO the held-out target is exactly minus the sum of the training targets.**<br>• With d(g, c) = y(g, c) − mean over g's other core cells, for any gene: **Σ_c d(g, c) = 0** over its core cells. This holds for genes in 5 cells and in all 6.<br>• When c is held out, the training rows' targets d(g, c′) are computed with ȳ₋c′(g), which **includes y(g, c)**.<br>• So **d(g, c) = −Σ_{c′≠c} d(g, c′)**.<br>**Any model with gene-specific inputs can recover or anti-predict the held-out target from the training targets.** That covers gene embeddings, propagation from the knocked-down gene, and per-gene parameters, which is every stage-2 candidate. A per-gene intercept alone predicts −d / (n−1), so per-row r is −1.<br>LOCO is invalid as registered. | **Recompute the reference per fold.** In the fold holding out c, every row's ȳ uses training cells only: ȳ₋{c, c′}(g) for training rows, and ȳ₋c(g) for the held-out row.<br>**Fix the minimum count for that**, e.g. ≥ 3 training core cells. Alternatively, take the reference from all 17 knockdown cells minus c, which keeps more genes.<br>**Add a test:** with synthetic y where d is pure noise, a per-gene model must score ≈ 0 in LOCO, not ±1. |
| 2 | MAJOR | harness: the noise ceiling (Ask 1) | **The ceiling subtracts the same ȳ₋c(g) from both the 96 h and the 120 / 144 h signature.**<br>• Both d's contain the reference's own estimation noise, −e, with the same sign. Their agreement is inflated by Var(e) ≈ σ² / (n − 1).<br>• **For scale:** with noise σ² = 1, Var(e) = 0.2 and a true cell-specific variance of 0.1, the "ceiling" r is ≈ 0.23, against a true 0.09.<br>• So it isn't "a lower bound". Time difference pushes it down, the shared reference pushes it up, and the net is unknown.<br>• **It also exists only in PC3 (1,355 pairs) and MCF7 (1,415).** No other core cell has 120 / 144 h knockdowns, so it can't be reported per cell for A375, A549, HEPG2 or HT29. | **Remove the reference rather than add it to both.** Regress each of y96 and y120 on ȳ₋c(g) per row (or project it out), then correlate the residuals. Or use a reference built from disjoint signatures.<br>**State that the ceiling covers 2 of the 6 cells.** |
| 3 | MAJOR | D1's quantity, centring and controls (Ask 3) | **(a) s(c, j) is computed on plate-normalised z.**<br>• Level 5 MODZ averages level-4 robust z, each gene standardised by its median and MAD over the plate's wells. A plate is one cell, mostly knockdown wells.<br>• So the gene's bulk variability in that cell is **divided out by construction**.<br>• What remains in the SD is tail heaviness and replicate reproducibility relative to the bulk. That isn't "capacity to move".<br>• Expression relates to L1000 measurement quality, so D1's positive control (expression ↔ s̃) can pass for technical reasons.<br>**(b) Composition:** each cell's s uses its own knockdown set, and only 2,100 of the 3,927 genes are knocked down in all six. Part of s̃ is then which genes were knocked down.<br>**(c) The pooled statistic** is open to cell-level shifts. If one cell has globally higher s̃ (better plates) and globally higher centred ACT (deeper tracks), the pooled Spearman is driven by 6 cell means. A landmark-cluster bootstrap can't see that. The ≥ 5 of 6 per-cell sign condition is immune to it; the pooled CI isn't. | **(a) A technical negative control,** registered with D1's positive control: the same s from the **`ctl_vector` signatures** of each core cell (503–592 per cell at 96 h, already in the extract).<br>D1's associations must be **specific to knockdowns**: use s_kd / s_ctl, or require the control's association to be null. State plainly what s measures.<br>**(b)** Compute s over the **2,100 genes knocked down in all six cells.**<br>**(c)** Make the pooled statistic the **mean of within-cell Spearman** (or double-centre by landmark and cell), with the landmark bootstrap. Keep ≥ 5 of 6.<br>**State the scope:** 6 cells; the bootstrap is over landmarks, not cells. |
| 4 | MAJOR | B1's meaning (Ask 4) | **The meaning fixed now says a pass shows "a knockdown-trained cell representation has a path to drugs".**<br>• A pass of the registered rule only shows a **gene-level** correspondence: the drug resembles its target's knockdown more than other knockdowns.<br>• That needs no cell representation at all. The knockdown in any cell would do.<br>• The cell-matched against mismatched comparison is the test of the cell part, and it is **reported only**. | **Either register matched > mismatched** (paired over pairs, drug-cluster CI) as the condition for the cell-representation meaning,<br>**or reword the meaning:** *"a pass shows a gene-level drug ↔ knockdown correspondence; whether it is cell-specific is the matched / mismatched comparison."* |
| 5 | MINOR | B1's details (Ask 4) | **Four details:**<br>• **The direction of "rank percentile" isn't stated.** It must be 0 = most correlated for "< 0.45" to mean what's intended.<br>• **When t is a landmark,** its own coordinate (the knockdown's deep dip) should be excluded from corr(y_cp, y_kd), as the harness already excludes it from knockdown scores.<br>• **Averaging all doses and times** dilutes active signatures with inactive low doses. 96 h can't be matched (compounds are 6 / 24 h).<br>• **The row filter** should be stated at row level: `split_cold_cell_1 == train` **and** `split_cold_drug_1 == train`. | **State the percentile direction.** Exclude t's own coordinate.<br>**Fix one dose / time rule in advance,** e.g. each drug's highest dose at 24 h per cell, with the all-signature average reported.<br>**State the row-level filter.** |
| 6 | MINOR | missing baselines and diagnostics (Asks 2, 5) | **Four things are missing for "find out why it works or doesn't":**<br>• **A variance decomposition of the core knockdown data** (gene, cell, gene × cell, noise; noise from the 96 / 120 / 144 h pairs). It says up front how much cell-specific signal exists to learn. If gene × cell is near noise, no representation can win on d.<br>• **A cell-offset baseline:**<br>  – in cold-gene, the mean d over the cell's training genes;<br>  – in LOCO, from the held-out cell's own `ctl_vector` signatures, which are control measurements, not outcomes.<br>  Per-plate z-scoring should make it small. If it isn't, every candidate must beat it.<br>• **Reproducibility strata:** sig_metrics' distil_cc_q75 / pct_self_rank_q25 / distil_nsample aren't in the extract.<br>• **A seed (off-target) diagnostic:** residual hairpin seed effects depend on the cell's transcriptome, so they can masquerade as an "expression-explained cell-specific response". `trt_sh.css` is in the extract, but the hairpin → seed map needs the pert metadata. | **Register the variance decomposition and the cell-offset baseline in stage 1.**<br>**The strata and the seed diagnostic** need GEO's small `sig_info` / `sig_metrics` / `pert_info` files. They are a new download, so the principal's approval is needed. If approved, register them as strata or diagnostics; if not, say they're unavailable. |

## Answers to the asks

**Ask 1 — d is the right kind of target, but only with per-fold references** (C1).
- **With 6 core cells,** LOCO gives 6 folds. Any stage-2 gain rule has 6 cells and will be coarse; register it with that in
  mind (per-cell signs plus a coarse cell CI).
- **Non-core cells entering training with chromatin masked is fine.** It adds gene coverage. But a chromatin model still
  learns its chromatin → d map from only the 5 training core cells per fold.
- **Non-core cells as reference cells** is the cleanest fix for C1's minimum-count problem.

**Ask 2 — the three baselines are a good core. Add the cell-offset baseline** (C6).
- **Optionally, a k-nearest-cell average** (k = 2–3, CCLE-weighted). It is more stable than 1-NN with 16 cells.

**Ask 3:**
- **P0's threshold** (median own-gene percentile ≤ 0.10) is a sensible data gate. Save the per-gene own-gene percentile; it
  is the knockdown-efficacy stratum later, for the 946 landmark genes.
- **"SD over knockdowns" as capacity to move:** not as such, because of the normalisation (C3a).
- **The centring:** it is right for a descriptive association. It isn't an unseen-cell quantity, since it uses all six
  cells. Say so.
- **The positive control as the gate:** keep it, but pair it with the `ctl_vector` negative control (C3a). An expression
  association alone can pass for measurement reasons. This is a different failure from §91 T3's.
- **Landmark-cluster bootstrapping:** right for landmark dependence. It is silent on cell-level confounding and on
  generalisation beyond the 6 cells (C3c).

**Ask 4:**
- **The restriction:** with the row-level filter of C5, no test row of either split is read. That matches B / H-DD's use of
  the excluded rows.
- **Dose and time:** fix one rule (C5).
- **The 0.45 threshold:** a modest bar. CMap reported target-knockdown connectivity as enriched but far from universal. The
  drug-cluster CI is the operative part, so 0.45 is reasonable.
- **Also report the fraction of pairs in the top 5 %** (CMap-style), descriptively.

**Ask 5 — the variance decomposition first** (C6). Then the D1 negative control (C3a). Then the strata and seed diagnostic,
if their metadata is approved. With C1's per-fold references, the harness is then fit for stage 2.

## What I checked and found sound

- **§103 in full:** the data, the harness, P0, D1, B1, the order, stage 2's outline.
- **The extract's file list,** and its record (counts by `pert_type`, no NaN, GCTX and npz sha1s, the integrity diff 0.0).
- **From metadata arrays only** (no `Y` read):
  - the time distribution of `trt_sh.cgs`;
  - the core counts per cell;
  - one signature per (gene, cell);
  - genes by number of core cells;
  - VCAP;
  - the 120 / 144 h pairs by cell (PC3 and MCF7 only);
  - `ctl_vector` / `ctl_untrt` counts per core cell and time;
  - the knockdown cell list (17).
- **The algebra behind C1:** Σ_c d(g, c) = 0 for genes in 5 and in 6 core cells.

## What I could not assess, and why

- **Every quantity that needs signature values:** P0's percentiles, the size of the cell offset, the variance decomposition,
  D1, B1. I didn't read `Y`, to keep stage 1 blind.
- **CMap's published connectivity rates** for target-knockdown pairs. I didn't look them up, so 0.45 is judged on its
  structure, not against a number.
