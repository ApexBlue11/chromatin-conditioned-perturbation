# TASK W16 — Port the gradient MoA probe to v9: `model/v9/probe_moa_v9.py` (RESULTS §86, pre-registered)

## Read first, in full
1. `model/results/RESULTS.md`, section `## 86.` — the pre-registration. **It is binding: implement exactly what it says.**
2. `model/v6/probe_moa_v6.py` — the probe being ported. Keep its five safeguards.
3. `model/v9/probe_v9.py` — how to load a v9 checkpoint, build an UNTRAINED v9 correctly (the quantiser must be fitted
   on training rows, lines ~66-100), the dataset, and the fold-0 splits.
4. `model/v9/model_v9.py` (`LincsV9.forward`, `return_interp=True` gives `aux['pathway_activations']`, shape [B,P,dp]),
   `model/v9/modules_v9.py` (`NamedPathwayReadout`, which holds `M` and `M_norm`).

## Deliverable: `model/v9/probe_moa_v9.py`
CLI: `--ckpt PATH` or `--untrained --seed S`; `--max_rows_per_drug 4`; `--n_perm 1000`; `--n_size 200`; `--out JSON`.
Computes, for ONE checkpoint (or one untrained model), everything in §86.1–86.2 and writes one JSON:
- **Rows:** the three fold-0 TEST splits (`test_coldcell`, `test_colddrug`, `test_coldboth`), `strength >=
  dc.eval_min_strength`, `has_l3`, at most `--max_rows_per_drug` rows per compound chosen deterministically (lowest row
  indices), identical for every checkpoint. Record the row list's sha1.
- **Positive sets:** ChEMBL mechanism targets from `drug/outputs/dti/chembl_dti_edges.tsv` (`organism == 'Homo sapiens'`,
  `target_type == 'SINGLE PROTEIN'`, `direct_interaction == '1'`), keyed by `pert_id`. A node's FULL gene set comes from
  `network/data/ReactomePathways.gmt` or `network/data/GO_Biological_Process_2023.gmt`, matched by the `term_id` column of
  `network/outputs/v9/pathway_info_v9.tsv` (inspect the GMT name/id format and match robustly; REPORT the number of the
  800 nodes matched and list any unmatched). Positive set of a compound = nodes whose full set contains >= 1 target.
  Secondary tier: landmark-only membership (`M_pathway_v9.npy` rows). Report positive-set size distributions.
- **Mean-drug baseline (one fixed input for every row):** `u_feats` = mean over the scored compounds; atom tokens = `k`
  copies of the mean valid atom vector over those compounds, `k` = the median atom count, mask valid for exactly `k`.
  Every other input is the row's own.
- **Importance:** `O = (out['delta'] ** 2).sum()`; `imp[p] = (a * dO/da).sum(-1)` per row (a = pathway activations);
  `dimp = imp_drug - imp_mean`; per compound the median over its rows (vectors of length 800).
- **Checks in order, all recorded:** degeneracy (max |imp| > 1e-12); gate `rho_del` = median over (a fixed seeded sample
  of up to 20,000) distinct compound pairs of Spearman(|dimp_d|, |dimp_d'|), and `rho_raw` the same on raw `imp`;
  `S` = median over compounds of the median rank percentile (0 = top) of its positives by |dimp|; Null 1 label
  permutation (each compound gets another's positive set; `--n_perm`), `diff`, one-sided p; Null 2 size-matched (each
  positive node swapped for a node of similar full-set size; `--n_size`).
- **Reference readouts** (same compounds, positives and nulls; S, diff, p for each): output projection
  `|M_norm @ (Yhat_d - Yhat_mean)|` per row, median per compound; data projection `|M_norm @ (Delta_measured_d -
  mean Delta of that cell's rows in the scored set)|`.
- **Strata** (S, diff, p within each): compounds seen vs unseen in fold-0 training; compounds whose targets are vs are
  not among the landmark genes with the largest |measured delta| (top 50) in their rows.
- **No reading, no verdict:** the script computes and records; it does NOT print SIGNAL/NULL. The PI applies §86.3.

## Tests: `model/v9/test_probe_moa_v9.py`
Small, fast, CPU-capable: (a) positive-set construction on a toy GMT + toy DTI; (b) rank percentile and label-permutation
on a planted example where the answer is known (planted signal gives diff < 0, no signal gives diff ~ 0);
(c) the mean-drug baseline is byte-identical for two different rows; (d) `rho_del` is 1.0 when all compounds have the
same |dimp| vector.

## Rules
- Create only the two files above. Do not edit any other file.
- Interpreter: `C:\Projects\LINCS\.venv-cuda\Scripts\python.exe`. Do not install packages.
- Run the tests. Then run the script ONCE on `--untrained --seed 0` with `--n_perm 50 --n_size 20` as a smoke test
  (small laptop GPU or CPU; inference and backward only, no training) and paste its summary. **Do not run it on any
  trained checkpoint** — the PI does that.
- Paste the full test output and the smoke-test summary in your final message.
