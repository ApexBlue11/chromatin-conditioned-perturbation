# TASK W24 — Paper figures: polish F1–F5, update F3 with every screen, add F6 (seed ensemble) and F7 (pathway alignment)

## Read first
`model/figures/make_figures.py`, `model/figures/test_make_figures.py`, `model/figures/POLISH_TODO.md` (binding polish list),
`model/results/RESULTS.md` sections `### 85.8`, `### 85.9`, `### 85.10` (what F3/F6/F7 show and the licensed wording).

## Design (binding, all figures)
- Palette (validated, colour-blind safe): **v9 `#2a78d6`** (blue), **XPert / comparator `#eb6834`** (orange), third role (untrained,
  reference, null) **`#1baf7a`** (aqua) — aqua is below 3:1 contrast so anything drawn in aqua **must carry a direct text label**.
  Neutrals: text `#0b0b0b`, secondary text `#52514e`, grid `#e6e6e3`, reference lines `#8a8985` dashed. Never cycle matplotlib's
  default colours; never a fourth hue (fold into neutral gray + label).
- Thin marks: lines 1.5–2 pt, markers ≥ 6 pt, thin error bars (capsize 0–2); light horizontal grid only; no top/right spines.
- **No titles that make a claim**; axis labels with units; the reading, if any, as a small annotation. One y-axis per panel — never a
  twin/dual axis. A legend whenever a panel has ≥ 2 series, plus direct labels on at most a few points.
- Text in neutral ink, never in a series colour. PNG at 300 dpi + SVG for every figure; sized for a journal column (≤ 7.2 in wide).

## Figures (sources binding; print every plotted number to stdout)
**F1–F2, F4, F5:** apply `POLISH_TODO.md` exactly; data sources unchanged.
**F3 `f3_dev_screens`** — forest plot of the §85 screens, one row each, label = human name + seeds:
| row | source JSON (`model/results/`) | label | decision (RESULTS 85.8) |
|---|---|---|---|
| C1 | `v9_dev_score_C1_noatoms.json` | "C1 no atom tokens (1 seed)" | dropped |
| C2 | `v9_dev_score_C2_lctl0.json` | "C2 no control encoder (1 seed)" | dropped |
| C3 | `v9_dev_score_C3_listnet.json` | "C3 ListNet ranking loss (1 seed)" | dropped |
| C4 | `v9_dev_score_C4_degk50.json` | "C4 DEG-reweighted loss (1 seed)" | dropped |
| C6 | `v9_dev_score_C6_signhead_3seed.json` | "C6 sign head (3 seeds)" | **accepted** |
| C7 | `v9_dev_score_C7_chromedges_3seed.json` | "C7 chromatin-gated edges (3 seeds)" | not accepted |
| C8b | `v9_dev_score_C8b_postpath.json` | "C8b post-drug pathway layer (3 seeds)" | not accepted |
Point = `vs_baseline.delta_per_row_mean`; for 3-seed rows draw a bar of ± 2·√(s0²/3 + sd²/3) (s0 = `baseline_sd` of the JSON, sd =
`sd`). Accepted row in blue, others neutral gray, decision as a small right-hand text column. Vertical references: 0 (solid), 0.00169
(s0, "drop"), 0.0034 ("advance", 1 seed), 0.003 ("accept floor", dashed). Panel B (same rows, shared y): `vs_baseline.delta_centred`
where present (rows without it left blank and labelled "n/a"). x label "Δ vs baseline, per-row delta Pearson (dev cells)".
**F6 `f6_seed_ensemble` (new)** — from `external/kaggle_out/v9dev_base2/v9dev_base_dev6s0_seed{0,1,2}.npz` (P2) and
`external/kaggle_out/v9dev_c8b/v9dev_c8b_dev6s0_seed{0,1,2}.npz` (C8b): mean per-row delta Pearson of the prediction average over K
seeds for K = 1 (each seed), K = 2 (each of the 3 pairs), K = 3; raw (left panel) and cell-centred (right panel). **Import and use**
`row_pearson` and `centred_r` / `cell_of_rows` from `model/v9/score_dev.py` — do not re-implement them. Points per K, a line through the
means, P2 blue, C8b orange. **Assert** P2 K = 3 raw = 0.4662 and centred = 0.5018 (± 5e-5; RESULTS 85.9) before plotting.
**F7 `f7_pathway_alignment` (new)** — Panel A from `v9_dev_align_P2_baseline_aux.json`: the aux-readout alignment per P2 seed (blue
dots), horizontal references for `references.training_prior` ("cell-agnostic training-row prior", 0.2292) and `references.loco_prior`
("leave-one-cell-out prior"), the per-seed `cell_shuffle_mean` ± `cell_shuffle_sd` (aqua, labelled "another cell's readout") and the
column-permutation `null_mean` (gray, labelled). Panel B from `v9_dev_align_P2_baseline_incell_aux.json` and
`v9_dev_align_C6_signhead_aux.json`: `in_cell.per_cell_3seed_mean` for each dev cell, P2 blue and C6 orange, a line at 0; y label
"own readout − other cells' mean readout (ρ)". Annotation: "in this cell: licensed (6 / 6 cells)" from `in_cell.licensed` /
`cells_positive`.

## Tests — extend `model/figures/test_make_figures.py`
For F3, F6, F7: every printed number equals its source field / recomputation (tolerance 1e-9 for copied fields, 5e-5 for F6's
recomputed means); every output file exists (PNG + SVG).

## Rules
Edit only `model/figures/make_figures.py`, `model/figures/test_make_figures.py`; outputs only under `model/figures/out/`. **Never delete
existing docstrings or comments.** Interpreter **only** `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` by full path. Run
`make_figures.py` **once** and `test_make_figures.py` **once**, sequentially — run NO other test suite. No installs, no git, no GPU,
**no scratch files anywhere in the repository**. Paste both outputs. Report anything ambiguous and what you chose.
