# REVIEW OF PACKET 001
verdict: NOT-SUPPORTED
reviewed_commit: 4e0bc24

Scoped, because the packet carries two independent results and they do not fare the same:

- **v9 > ridge on unseen cell lines (+0.1775)** — I could not break this. It survives every stratification
  and every metric in the artefact, and the margin is ~40x the seed spread the packet quotes. Sound,
  subject to C4 and C6.
- **chromatin contributes on unseen cell lines (+0.0042)** — not supported by what is here. The interval
  quoted is the wrong uncertainty for the question (C1), the number cannot be recomputed from anything in
  the repo (C2), and a closed-form model with no seed noise gains the same amount at matched lambda (C3).

The verdict is the lower of the two.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | BLOCKING | stats | The +0.0042 CI [0.0036, 0.0049] bootstraps **rows**, but the claim generalises over **cell lines**, and the test set has 8 of them — 78.8 % of rows from two (MCF7 10,969 = 51.4 %, HT29 5,837 = 27.4 %; then MDAMB231 2,188, HS578T 1,074, THP1 820, CD34 295, BJAB 84, H1975 54). Rows within a cell line share the chromatin vector exactly, so they are not independent draws. The effective n for this claim is nearer 5 than 21,151. Separately the two arms are **two training runs at one seed each** — a between-run comparison, not a within-run ablation — so training noise is entirely unmeasured and the row CI conditions it away. | Cluster-bootstrap over the 8 test cell lines, and report the paired delta **per cell line**. Then >=3 seeds per arm, with the between-seed spread of the ON arm alone reported next to the +0.0042. If the ON-arm seed spread on this split exceeds ~0.004, the effect is unmeasurable at n=1 seed. |
| 2 | BLOCKING | provenance | The central number has no reproducible source. (a) **No script in the repo writes `v9_chromatin_ablation_cold_cell_1.json`** — `grep -rn chromatin_ablation` over all `.py` returns only the packet. Its 258 bytes carry no seed, no metric string, no strata, no `row_index` evidence, and none of the keys `head_to_head_mdmt.py` emits, so the packet's statement that that script "scores arms pairwise on identical rows" is not what produced this number, and its alignment guards (`row_index` intersection, `y_true`/`ctl_true` assertions) are unevidenced here. (b) **The chromatin-ON arm has no arm-level artefact at all.** The only one present, `v9_xpert_arm_split_cold_cell_1_seed0.json`, contains `"ablate_epi": true` — i.e. it is the ablated run sitting in the un-suffixed filename. `git log -S"_noepi"` shows the `_noepi` suffix was introduced in the *same* commit (`5e80c5d`) that added these artefacts, so the overwrite the script's own comment warns about ("the ablated run overwrote the full run's json once") happened again here and the fix landed after the run. (c) No `--save_pred` `.npz` exists for either cold-cell arm, so nothing can be re-scored without retraining. | Commit the script that produced the comparison. Re-run both arms with `--save_pred`, regenerate the comparison through `head_to_head_mdmt.py` so the row-alignment guards actually execute, and keep both arm JSONs under distinct names. |
| 3 | MAJOR | leakage | **Ridge lambda is selected on the test rows.** `xpert_mdmt_baselines.py` scores each lambda with `per_row_pearson(pr - C[te], (X-C)[te])` and keeps the argmax; there is no validation split. The two arms therefore select *different* lambdas (no-chromatin `lam_best`=1e4, chromatin `lam_best`=1e3), and the quoted +0.0029 gain is `max_lam(chrom) - max_lam(no chrom)`. **At matched lambda the gain is +0.0038 (1e3), +0.0018 (1e4), +0.0017 (1e2), +0.0013 (1e5).** At lambda=1e3 the ridge gains **+0.0038** from chromatin — statistically indistinguishable from v9's +0.0042, and it is closed-form, so it carries *no seed noise*. That inverts the reading the juxtaposition invites: the evidence does not separate "v9 extracts chromatin signal" from "chromatin is a weak linear cell covariate that any model picks up equally". It also means the cleanest chromatin measurement in this packet is the ridge's, not v9's. | Select lambda on a held-out fold of the training cells, or fix lambda across arms and report that. Then state the chromatin gain as the matched-lambda ridge value with its own CI, and say explicitly whether v9's gain exceeds it. |
| 4 | MAJOR | leakage | Direct answer to ask 3: **yes, on the v9 side.** `xpert_arm.py` builds a per-cell aggregate control under the comment "from TRAINING rows only", but the fallback is `self.cell_mean[c] = self.C[sel].mean(0) if sel.any() else self.C[m].mean(0)`. For a **cold** cell `sel.any()` is always False, so `x_cell` for all 21,321 test rows is the mean control over that cell's **test rows**. `x_cell` is a live model input (`model_v9.forward` -> `self.cell_enc(x_cell, ...)`). It is an unsupervised aggregate of an input, not of the target, so it is the mild form — but it is computed over the evaluation set, in exactly the regime under study, and the code comment asserts the opposite. The ridge's `feats()` uses only per-row `C[idx]`, so **v9 has a cell-level input the comparator does not**. It affects both v9 arms equally, so C1's +0.0042 is untouched; the +0.1775 and the XPert comparison are not. | Re-run v9 cold-cell with `x_cell` set to the global *training* control mean for unseen cells, and report the change in 0.4734. Or give the ridge the identical per-cell aggregate and re-fit. Either makes the comparison like-for-like; the first also removes the transduction. |
| 5 | MAJOR | wrong-quantity | **There is a free placebo stratum and it was not reported.** Of the 8 test cell lines, 3 have no chromatin track — HS578T is in the cell index but `E_final_mask` is empty for it (0 tracks), and BJAB/H1975 are absent entirely. Those 1,212 rows (5.7 %) receive identically-zero chromatin in the ON arm. Neither arm has cell-specific chromatin there, so the paired delta on that stratum is a **direct estimate of the training-noise floor** for this comparison. If it is also ~+0.004, C1 is settled negatively and nothing else is needed. The packet reports only the pooled 94.3 %/5.7 % coverage figure, not the contrast. | Report the paired delta split three ways: rows with >=2 tracks (MCF7/MDAMB231/CD34 = 3 tracks, HT29/THP1 = 2), rows with 0 tracks (HS578T/BJAB/H1975), and pooled. Needs only saved predictions, no retraining. |
| 6 | MAJOR | provenance | Row sets are mixed inside one table. "Rows using compounds that cannot be featurised are dropped from **both** sides" is false for the ridge: `xpert_mdmt_baselines.py` does `r = np.clip(rows[idx], 0, None)`, i.e. unfeaturisable compounds get **drug index 0's** fingerprint and descriptors plus a `have` flag, and its JSON scores **n_test = 21,321** / trains on 47,509. v9 drops them: **n_test = 21,151** / trains on 46,931. So in the packet's table the ridge row's `0.2951` is a 21,321-row number while its `0.9584` and every v9 number are 21,151-row numbers, all presented under "n = 21,151". The correctly paired ridge value is **0.2959** (`v9_vs_ridge_cold_cell_1.json`), not 0.2951. The two models are also trained on different row counts. | Quote 0.2959 for the paired table and carry n per row. Either drop the 170 rows from the ridge too, or state that the ridge imputes them and v9 does not. |
| 7 | MAJOR | provenance | `v9_vs_ridge_cold_cell_1.json` stores the ridge's scores under the JSON key **`XPert_released_ckpt`** (`"XPert_released_ckpt": {"delta": {"mean": 0.2959}}`), with `"theirs_label": "ridge"` recorded separately. The script's docstring flags precisely this trap and `--theirs_label` was added to fix it — but only the *printout* was fixed; the JSON key is hard-coded. Any downstream read of this file states that XPert scores 0.2959 cold-cell. Given the project has already been burned once by a baseline named after their model, this is a live landmine, not a cosmetic issue. | Key the JSON off `--theirs_label`. Then grep every results JSON for `XPert_released_ckpt` and check which ones are actually XPert. |
| 8 | MINOR | overreach | Ask 2: the citation itself is clean (I verified it, see below), but v9 has **fold 1 only** against a **five-fold mean** 0.383 +/- 0.027, and fold-1 difficulty is uncalibrated. The 170-row difference in scored rows applies here too. | The ridge is closed-form — run it on all five `split_cold_cell_*` folds (minutes of CPU). If ridge sits flat near 0.295 across folds, fold 1 is not anomalous and "above the published five-fold mean by >3 published sd" becomes defensible. For an actual head-to-head, `external/xpert/code/XPert/train_xpert.py` is present, so XPert can be retrained on `split_cold_cell_1` and scored on the identical rows — the same thing that was already done at warm. |
| 9 | MINOR | leakage | The chromatin SVD in the ridge is fitted over `uniq` = **all 40 cell lines**, test included, and centred on `mu_c` computed over all 40. The reconstruction is exact, so no information is added — but ridge L2 is **not rotation-invariant**, so the basis and the singular-value scaling (both chosen with the held-out cells present) change the fit. Small, but it is in the arm the chromatin claim rests on. | Fit the SVD on the 32 training cell lines and project the 8 test cells into that basis. |
| 10 | MINOR | provenance | The arm JSON records no `batch`, `epochs` or `lr`. The packet's "12 epochs, seed 0, batch 8" and "both arms share schedule and batch size" cannot be checked against any artefact; the script default is `--batch 48`. | Add the parsed args to the JSON the arm writes. |
| 11 | MINOR | code-vs-intent | `known_cell_frac` (0.9721 in the artefact) counts **train+test** rows whose cell is in the index, but both print statements describe it as a property of the test rows ("`{n}`% of rows use a cell line we have no chromatin or lineage for"). The packet's 94.3 % is correct but comes from somewhere else. I measured it directly: 20,109/21,321 = **94.32 %**. Test-only index coverage is 99.35 %, which is a third distinct number. | Compute and store the figure over `D.te` only, and store the track-count breakdown alongside it. |
| 12 | MINOR | code-vs-intent | `their_pearson` uses `np.nanmean`; XPert's `metrics.py:pearson` uses `np.mean` over `pearsonr` outputs. A single degenerate row yields NaN for their whole score and a silently dropped row for ours. Immaterial on these data, but the conventions are not identical and the packet calls them the same. | Assert zero non-finite per-row correlations rather than absorbing them, so the two agree by construction. |

## What I checked and found sound

- **The split is genuinely cold.** Loaded `xpert_mdmt_splits.npz` directly rather than trusting the name:
  `split_split_cold_cell_1` gives 47,509 train / 21,321 test, 32 train cell lines vs 8 test, and the
  intersection is **empty**. `strata_delta` independently confirms it — "pair NOT in train" covers all
  21,151 rows and the "pair in train" stratum is absent because it is empty. No cell and no (cell,
  compound) pair leaks. This is a real cold-cell evaluation.
- **The metric matches theirs.** `external/xpert/code/XPert/metrics.py:74` is the mean of per-row
  `pearsonr`. `their_pearson` and `per_row_pearson` reproduce that (modulo C12). Median is emitted
  alongside and the mean is used for comparison, which is the conservative choice.
- **The published numbers are cited correctly.** Table R8, `supplementary_text.txt:1018`, L1000_mdmt,
  five-fold CV. Columns are MLP_Morgan | MLP_UniMol | DeepCE | CIGER | PRnet | TranSiGen | Xpert | Mean;
  the cold-cell PCC row reads 0.227 | 0.229 | 0.089 | 0.236 | 0.195 | **0.293** | **0.383** | **0.224**.
  Mean 0.224, CIGER 0.236, TranSiGen 0.293+/-0.017, XPert 0.383+/-0.027 all check out, and the models
  omitted from the packet's table are the *weaker* ones. Not reverse-engineered; read from the SI.
- **The warm-split reproduction is the strongest thing in this packet and is doing more work than it is
  credited with.** Running XPert's released checkpoint here gave 0.6933 against their published
  0.688+/-0.011. That is what establishes R8's "PCC" is `Pearson_deg` and not absolute — the alignment
  ask 2 depends on — and it is empirical rather than assumed. It also validates the harness end to end.
- **The ablation is to the mean, on both channels.** `self.E[:] = self.E[m_tr].mean(0)` and
  `self.r[:] = self.r[m_tr].mean(0)`, training rows only, applied at training time with lineage
  untouched. Retraction 2 is not repeated: no ablate-to-0/1 anywhere in this path.
- **The quantiser guard is genuinely repaired.** `bins_fitted` now calls `discriminates()`
  (`modules_v9.py:83`), which checks finite edges and a non-degenerate range rather than
  `fitted == 1.0`. Retraction 7 is fixed. One note: `xpert_arm.py` invokes it with `x=None`, which
  skips the strongest check (`unique(bucket(x)) > 1`); passing a real batch would close it completely.
- **Fitting hygiene elsewhere is clean.** Quantiser bins fit on `D.C[D.tr]`; `dose_n`/`time_n` normalised
  on training statistics; per-cell chromatin z-scoring is within-cell so no information crosses rows;
  ridge feature standardisation `mu, sd` from `Ftr` only. Apart from C3, C4 and C9 I found nothing
  fitted on scored rows.
- **The pairing machinery in `head_to_head_mdmt.py` is well built** — `row_index` set intersection, a
  95 % overlap floor that aborts, and hard assertions that both sides agree on `y_true` and `ctl_true`.
  My complaint in C2 is that the chromatin comparison does not appear to have gone through it.
- **The +0.1775 v9-over-ridge result is robust.** It holds on all 8 metrics in `metric_robustness`
  including MSE, MAE, Spearman and Precision@100 up/down; on both dose strata; and across all four
  true-effect-size quartiles — 0.4127/0.4692/0.4832/0.5287 against 0.2124/0.2717/0.3091/0.3903, with
  the *largest* margin in Q1, the weakest-signal quartile. That last point matters: retraction 6 (inert
  signatures diluting or inverting an effect) is addressed here, and it does not save C1, which has no
  stratification at all.

## What I could not assess, and why

- **Whether +0.0042 is above the training-noise floor.** This is the whole of ask 1 and it is
  unanswerable from the artefacts: one seed per arm, no saved predictions, no per-cell-line breakdown.
  C1 and C5 are the two cheapest routes to an answer, and C5 needs no GPU.
- **Whether the comparison was row-aligned.** No generating script and no `row_index` in the output
  (C2). The n=21,151 matching v9's test count is consistent with correct alignment but does not
  demonstrate it.
- **Ask 4, as posed.** I cannot answer "is n=21,151 enough for 4 decimal places" because 21,151 is not
  the relevant n. For a cell-line-generalisation claim the denominator is 8 clusters dominated by two;
  for a between-run architectural claim the denominator is seeds, of which there is one. The interval is
  narrow because it answers "would this differ if I resampled rows from these two fixed models" — which
  is not the question asked. Four decimals are defensible for the *point estimate*; the interval is not.
- **XPert on cold-cell.** Not run, so ask 2 has no head-to-head arm at all. `train_xpert.py` is present,
  so this is a compute decision rather than a blocked one — your call, not mine.
- **The absolute-Pearson drop in the chromatin ridge (0.9585 -> 0.9445, n=21,321 both).** Adding
  chromatin improves delta PCC by ~0.003 while degrading absolute PCC by ~0.014. I have no account of
  this and did not have the budget to chase it, but a feature block that helps one and hurts the other
  by 5x as much is worth a look before chromatin is described as informative.
- **The noise ceiling.** Nothing in the packet says how much of the gap from 0.4734 to 1.0 is reachable.
  `model/v9/replicate_noise_mdmt.py` exists and is not cited; if it estimates replicate agreement on
  these rows, 0.4734 and +0.0042 both want to be read against it.
- **Deliberate omission:** I did not read `RESULTS.md`, `MANUSCRIPT.md`, `V9_HANDOFF.md` or any
  narrative, to stay blinded. One leak to disclose: tracing C2's provenance put `git log` output on
  screen and I saw the subject line of commit `5e80c5d`, which states a conclusion. My assessment of
  C1/C3 was already written at that point and I have not revised it; flagging it so you can discount
  this review if you disagree.
