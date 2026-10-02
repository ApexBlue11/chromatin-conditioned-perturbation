# TASK W25 — Paper figures: F3 gains the V2 and P6 rows; F6 becomes the variance figure (seed vs snapshot vs MC)

## Read first
`model/figures/make_figures.py` (functions `make_f3`, `make_f6`, `p_val`, the palette constants at the top),
`model/figures/test_make_figures.py`, `model/v9/score_dev.py` (functions `row_pearson`, `centred_r`, `cell_of_rows`),
`model/results/RESULTS.md` sections `### 85.13`, `### 90.7`, `### 90.8` (what the new rows mean).

## Design (binding; unchanged from W24)
- Palette: v9 `#2a78d6` (`COLOR_V9`), comparator `#eb6834` (`COLOR_COMP`), third role `#1baf7a` (`COLOR_THIRD`; anything drawn in it
  **must carry a direct text label**), neutrals as defined. Never matplotlib's default cycle; never a fourth hue.
- Thin marks, light grid, no top/right spines (use the existing `setup_axes`). No titles that make a claim. One y-axis per panel —
  never twin axes. Legend whenever a panel has ≥ 2 series. Text in neutral ink, never in a series colour. PNG 300 dpi + SVG; ≤ 7.2 in
  wide per column (F6 may be 7.2 in wide × ≤ 3.4 in tall for three panels, or two rows — your choice, say which).

## F3 `f3_dev_screens` — two rows added, one layout bug fixed
1. Append these rows to `rows` in `make_f3`, after C8b, in this order (key, label, decision, file):
   - `("V2_snap3", "V2 snapshot ensemble, 3 cycles (3 seeds)", "accepted", "v9_dev_score_V2_snap3.json")`
   - `("P6_c6_v2", "P6 stack: C6 + V2 (3 seeds)", "confirmed (stack)", "v9_dev_score_P6_c6_v2.json")`
   Both are 3-seed rows scored against the same P2 baseline as the others, so the existing point / error-bar / centred-panel code applies.
   Colour: `COLOR_V9` for decision `"accepted"` **and** `"confirmed (stack)"`; `COLOR_GRAY` otherwise.
2. Draw a thin neutral horizontal separator (`COLOR_GRID`, lw 1) in both panels between the C8b row and the V2 row, and put two small
   secondary-text (`COLOR_SEC_TEXT`, fontsize 8) group labels at the left edge of panel A just above each group:
   `"architecture / loss screens"` above C1 and `"variance and stack"` above V2.
3. **Bug:** the right-hand decision text is placed at `ax1.get_xlim()[1]` *inside* the loop, so its x drifts as later points widen the
   axis. Move the decision-text drawing to **after** the loop (after every point and error bar is drawn), using the final xlim.
   The decision text's colour stays neutral ink (`COLOR_SEC_TEXT`), not the series colour (design rule: text never in series colour).
4. Keep every existing `p_val` call; the two new rows print through the same calls.

## F6 `f6_seed_ensemble` — rebuilt as the variance figure (keep the function name and output file names)
Three panels, sharing nothing (each its own y-axis):
**Panel A (raw) and Panel B (cell-centred):** mean per-row delta Pearson of the prediction average over K seeds, K = 1 (each seed),
K = 2 (each of the 3 pairs), K = 3 — exactly the existing `evaluate_ensemble` computation — for two series:
- **P2** (`COLOR_V9`, label "P2 (one 12-epoch model per seed)"): `external/kaggle_out/v9dev_base2/v9dev_base_dev6s0_seed{0,1,2}.npz`
- **V2** (`COLOR_COMP`, label "V2 (each seed = mean of 3 snapshots)"): `external/kaggle_out/v9dev_v2/v9dev_v2_dev6s0_seed{0,1,2}.npz`
**C8b leaves this figure** (its curve stays in RESULTS 85.9; do not delete the code path — keep `c8b_files` computed and its values
printed with `p_val`, just not plotted, so the §85.9 numbers remain checkable). x label "seeds averaged (K)"; y labels as now.
**Panel C — gain over one model, and what it costs** (horizontal dot plot, one row each, all `COLOR_V9`, single series → no legend;
each row's value printed with `p_val` and written as a small neutral-ink number to the right of its dot):
| row label | value | source |
|---|---|---|
| "MC dropout + stoch. depth, 8 passes (1 run)" | `vs_baseline.delta_per_row_mean` | `model/results/v9_dev_score_V1_full.json` |
| "snapshot ensemble, 3 cycles (1 run)" | `vs_baseline.delta_per_row_mean` | `model/results/v9_dev_score_V2_snap3.json` |
| "seed ensemble, 3 seeds (3 runs)" | P2's K = 3 raw value from panel A minus `baseline_mean` of `v9_dev_score_V2_snap3.json` (= μ0) | computed |
x label "Δ per-row delta Pearson over a single model (dev cells)"; a solid line at 0. Small secondary-text note under the panel:
`"MC row: paired vs the same checkpoint's deterministic pass; other rows: vs P2's single-model mean"`.

**Asserts before plotting** (keep the existing two; add these):
- P2 K = 3 raw = 0.4662 and centred = 0.5018 (± 5e-5) — existing.
- V2 K = 1: the mean of the three single-seed raw values equals `mean` of `v9_dev_score_V2_snap3.json` (± 1e-6).
- V2 K = 1: the mean of the three single-seed centred values equals `mean_centred` of `v9_dev_score_V2_snap3.json` (± 1e-6).
- Panel C row 3 is within 5e-5 of 0.4662 − 0.43693.
Print every plotted number with `p_val("F6", ...)`, using keys `V2_raw_K{k}_{i}`, `V2_centred_K{k}_{i}` (new) alongside the existing
`P2_*` and `C8b_*` keys, plus `gain_mc`, `gain_snapshot`, `gain_seed3`.

## Tests — extend `model/figures/test_make_figures.py`
Tests must read the source files themselves and compare with what `make_figures.py` printed (they must be able to fail):
- F3: for V2_snap3 and P6_c6_v2, the printed `{key}_delta_per_row_mean` and `{key}_delta_centred` equal the JSON fields (1e-9).
- F6: printed `gain_mc` and `gain_snapshot` equal the JSON fields (1e-9); printed `gain_seed3` is within 5e-5 of 0.4662 − 0.43693;
  the mean of printed `V2_raw_K1_{0,1,2}` equals the JSON `mean` (1e-6).
- The output files still exist (the existing list covers them).

## Rules
Edit only `model/figures/make_figures.py` and `model/figures/test_make_figures.py`; outputs only under `model/figures/out/`.
**Never delete existing docstrings or comments; do not reorder or restyle code you were not asked to change.** Interpreter **only**
`C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` by full path. Run `make_figures.py` **once** and then
`-m pytest model/figures/test_make_figures.py -q` **once**, sequentially, from `C:\Projects\LINCS` — run NO other test suite, never two
python processes at once. No installs, no git, no GPU, **no scratch files anywhere in the repository**. Write your report to
`C:\Users\Surya\AppData\Local\Temp\claude\C--Projects-LINCS-LINCS-project-scope-review-1b6bd9\1a2a87ef-e957-4ed6-bb10-ec2e7f9e1823\scratchpad\W25_report.md`
with both outputs pasted and a mandatory `## What I was unsure about` section.
