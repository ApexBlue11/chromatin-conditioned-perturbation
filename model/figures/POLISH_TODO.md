# Figure polish list (PI review of W18 drafts, 2026-09-25) — apply before the manuscript, after the screens finish

All numbers verified against their source JSONs (make_figures.py prints them; test_make_figures.py asserts them).
Presentation only:
- F1: left-panel x ticks overlap (fewer ticks / 2 decimals); x label "v9 − XPert: per-row Δ Pearson, median per cell";
  legend "v9" / "XPert (trained to its published recipe)"; legend entry for the cluster-mean band and its CI.
- F3: human candidate names (C1 no atom tokens, C2 raw control / no learned encoder, C4 DEG-reweighted loss, ...);
  x label "dev Δ per-row delta Pearson vs baseline (seed 0)"; shade ±s0 around 0 for the baseline seed spread;
  regenerate as screens complete.
- F4: y label "S − label-permutation null (rank percentile; lower = better)"; annotate each bar with its p.
- All: no claim-making titles; keep the reading as a small annotation, not a title.
