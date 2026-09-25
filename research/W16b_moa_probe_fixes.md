# TASK W16b — Fix `model/v9/probe_moa_v9.py` to match its pre-registration (RESULTS §86) exactly

The PI reviewed `model/v9/probe_moa_v9.py` (from W16) against `model/results/RESULTS.md` section `## 86.` and found
the deviations below. Fix every one. Do not change anything else. Read §86 again before starting.

F1. **Per-row importance.** `importance()` currently returns `.mean(0)` over the rows of a compound. Return the per-row
    vectors `[n_rows, P]`. Per compound: `raw imp` = median over rows of the per-row `imp`; `Δimp` = median over rows of
    (`imp_row − imp0_row`), where `imp0_row` is the same row under the mean-drug baseline.
F2. **Reference readouts, projection order.** §86 is `|M_norm @ (Ŷ_d − Ŷ_mean)|` — project the SIGNED difference, then
    take the absolute value. The code takes `abs` first. Fix both:
    - output projection per row: `abs(M_norm @ (yhat_row − yhat0_row))`; per compound the median over rows;
    - data projection per row: `abs(M_norm @ (y_delta_row − cellmean))`, where `y_delta` is the batch's `y_delta`
      (Level-3 delta — NOT `ds.Y`, which is the Level-5 z-score), and `cellmean` = the mean `y_delta` over the
      SCORED rows (`selected_rows`) of that row's cell; per compound the median over rows.
F3. **One scoring function, applied identically everywhere.** Write `score(scores, allpos, sizes, rng_seed, n_perm,
    n_size) -> dict(S, null1_mean, null1_sd, diff, p, null2_mean, n)` (rank percentile 0 = top; Null 1 = label
    permutation among the compounds passed in; Null 2 = size-matched as now) and apply it, with the same seed each
    time, to: the gradient readout (`|Δimp|`), the output projection, the data projection, every stratum (F4; the
    permutation is WITHIN the stratum), and the landmark-only tier (F5). Record every result.
F4. **Strata:** seen vs unseen as now. **Responsiveness:** a compound is `target_responsive` iff at least one of its
    ChEMBL targets is a landmark gene among the 50 genes with the largest mean `|y_delta|` over its scored rows;
    otherwise `target_not_responsive`. Map symbols to landmark indices with `network/outputs/v9/landmark_symbols_v9.tsv`
    (its `hgnc_symbol` column, in landmark order), not `dti_reference.tsv`. Record n per stratum.
F5. **Landmark-only tier:** positives = nodes whose row of `M_pathway_v9.npy` has ≥ 1 of the compound's targets (via the
    same landmark mapping). Scored with F3 on the gradient readout.
F6. **Record** in the JSON: the checkpoint file name or `"untrained seed S"`, n compounds scored, n per stratum, the
    number of gate pairs used, `k_med`, and a positive-set size summary (min / median / max) beside the list.
    `--out` default becomes `model/results/probe_moa_v9_<ckptstem or untrained_sS>.json`.
F7. **Tests** in `model/v9/test_probe_moa_v9.py`, added to the existing four: (e) a planted case where
    project-then-abs differs from abs-then-project, asserting the function uses project-then-abs; (f) `score()` gives
    `diff < 0` on a planted signal and |diff| small on a shuffled one; (g) a stratum's permutation only draws positive
    sets from compounds inside that stratum.

## Rules
- Edit only `model/v9/probe_moa_v9.py` and `model/v9/test_probe_moa_v9.py`. Delete the stray `probe_moa_v9_out.json`
  in the repository root.
- Interpreter `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`. No installs. No training.
- Run the tests, then ONE smoke run: `--untrained --seed 0 --n_perm 50 --n_size 20`. **Do not run any trained
  checkpoint.** Paste the full test output and the smoke JSON's top-level fields in your final message.
