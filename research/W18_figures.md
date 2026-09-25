# TASK W18 — Paper figures from committed result files (no numbers from memory, ever)

## Deliverable
`model/figures/make_figures.py` — one function per figure, each reading ONLY the file(s) named below, writing
`model/figures/out/<name>.png` (300 dpi) and `.svg`, and printing every number it plots to stdout so the PI can check
them against the source. Colour-blind-safe palette (e.g. Okabe–Ito); clean, journal style (no titles inside panels that
make claims; axis labels with units; light grids). Run all figures; paste the stdout.

## Figures (sources are binding; if a field is missing, stop and report — do not substitute)
**F1 `f1_coldcell_h2h`** — `model/results/coldcell_h2h_split_cold_cell_1_O2.json`. Left panel: per held-out cell line,
`d_c_median` with its `d_c_median_ci95` as a horizontal error bar, cells ordered by `n_scored`, labels "CELL (n=…)";
a vertical line at 0; a shaded band for the cluster mean's `cluster_ci95` with the mean marked. Right panel: the two
models' per-cell mean scores (`ours_mean`, `theirs_mean`) as paired dots joined by a line per cell. Caption text in a
separate `f1_coldcell_h2h.txt`: "v9 minus XPert (trained to its published recipe), per-row delta Pearson; 5 of 8 cells
favour v9; the pre-registered criterion (≥ 7 of 8 and a cluster CI excluding 0) was not met." — take "5 of 8" from
`cluster.cells_favouring_ours`, never hard-code it.
**F2 `f2_xpert_reproduction`** — bar chart: XPert's published cold-cell mean 0.383 with sd 0.027 (Nature Machine
Intelligence 2025, Supplementary Table R8 — hard-code these two values with that citation in a comment), our trained
XPert on the test rows (`reproduction.theirs_all_rows_mean` from the F1 file), and the band `reproduction.band`.
**F3 `f3_dev_screens`** — forest plot of every `model/results/v9_dev_score_*.json` that has a `vs_baseline` block:
`delta_per_row_mean` per candidate (label from `label`), with vertical reference lines at 0, at the drop threshold
+0.00169 (s0) and the advance threshold +0.0034, taken from `model/results/v9_dev_score_baseline_kaggle.json` (`sd`)
— compute s0 from that file and 0.0034 as max(2·s0, 0.003).
**F4 `f4_moa_probe`** — `model/results/probe_moa_v9_reading_86.json`: for seeds 0–2, the trained gradient readout's
`diff` beside the untrained `diff`, with a reference line at −0.02 (the pre-registered floor); annotate the reading
(`reading` field).
**F5 `f5_input_coverage`** — `model/results/cc1_input_coverage.json` `summary`: grouped bars for train / dev / test of the
row fraction with a direct CCLE match and with each chromatin track.

## Rules
- Create only `model/figures/make_figures.py`, `model/figures/test_make_figures.py` (asserts each output file exists and
  each printed number equals its source field), and the outputs under `model/figures/out/`. Touch nothing else.
- Interpreter `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe` (matplotlib 3.11 is installed). No installs.
- Paste the full stdout of `make_figures.py` and of the test in your final message.
