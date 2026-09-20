# PACKET 002 — cold-cell chromatin ablation, re-scored from saved predictions
packet_id: 002
created: 2026-09-20
repo_commit: efbfaa3
supersedes: 001 (closed)

## PACKET-DEFECT DISCLOSURE, up front
Packet 001 did not list artefact paths, so you concluded (C2c) that no saved predictions existed and that
re-scoring required retraining. **They existed.** `external/` is gitignored, so you could not see them
from the repository. That is a defect in 001, not an error of yours. Review 001 therefore stands at
**11 of 12 upheld** in our record, and this packet lists every path.

## ARTEFACTS — read these directly
```
external/v9_mdmt_preds/v9_cc1_epi_seed0.npz      265 MB  chromatin-ON arm    (2026-08-31)
external/v9_mdmt_preds/v9_cc1_noepi_seed0.npz    265 MB  chromatin-ABLATED arm (2026-09-01)
external/v9_mdmt_preds/v9_cc1_{epi,noepi}_seed0.pt        checkpoints, 53 MB each
external/xpert_split_bundle/xpert_mdmt_splits.npz         meta_cell, split_split_cold_cell_1
phase2_assembly/outputs/E_final_mask.npy         [83,978,3] bool
epigenetics/outputs/epigenetics_cell_index.json  cell_id_to_row
model/v9/chromatin_ablation_analysis.py          THE GENERATING SCRIPT (new)
model/results/v9_chromatin_ablation_cold_cell_1_RESCORED.json   output
```
Each `.npz` carries `deg_pred`, `y_pred`, `y_true`, `ctl_true`, `row_index`, all `[21151, 978]` / `[21151]`.

`git show --stat efbfaa3` and `git show --stat` for the two prior commits are in
`orchestration/bus/to_critic/002_gitstat.txt`, so you have no reason to run `git log`.

## OBJECTIVE
Whether cell-line chromatin state (ATAC / H3K27ac / H3K27me3) contributes to predicting drug-induced
transcriptional response on **unseen cell lines**.

## FUNCTION
`chromatin_ablation_analysis.py` loads both arms' saved predictions and re-scores them. No training, no
GPU. It asserts before scoring: `row_index` identical across arms; `y_true` and `ctl_true` byte-identical;
every scored row is a `test` row of `split_cold_cell_1`; test-cell and train-cell sets disjoint.

Metric: per-row Pearson on the delta (`deg_pred` vs `y_true − ctl_true`), then the **mean** (XPert's
`metrics.py` convention). Degenerate rows are counted, not absorbed.

A track counts as present for a cell if `E_final_mask[cell]` is true for any gene.

Two analyses:
1. Paired delta split by whether the cell line has **any** chromatin track.
2. Paired delta **per cell line**, then a 20,000-resample bootstrap **over cell lines**.

## RESULTS

Guards: all passed. `row_index` identical (n=21,151). `y_true`/`ctl_true` max|diff| = 0.0e+00. 8 test
cells vs 32 train cells, intersection empty. 0 degenerate rows.

### By chromatin availability
| stratum | n | epi ON | epi ABLATED | paired Δ |
|---|---|---|---|---|
| chromatin present (MCF7, HT29, MDAMB231, CD34, THP1) | 19,950 | 0.4717 | 0.4659 | +0.005855 |
| no chromatin track (HS578T, BJAB, H1975) | 1,201 | 0.5019 | 0.5245 | −0.022625 |
| pooled | 21,151 | 0.4734 | 0.4692 | +0.004237 |

### Per cell line
| cell | n | tracks | paired Δ |
|---|---|---|---|
| MCF7 | 10,815 | 3 | +0.007016 |
| HT29 | 5,837 | 2 | +0.009459 |
| MDAMB231 | 2,188 | 3 | −0.004662 |
| HS578T | 1,074 | 0 | −0.020428 |
| THP1 | 815 | 2 | −0.002187 |
| CD34 | 295 | 3 | −0.007828 |
| BJAB | 73 | 0 | −0.007694 |
| H1975 | 54 | 0 | −0.086518 |

### Bootstrap over cell lines (20,000 resamples, seed 0)
mean of per-cell-line means **−0.014105**; 95 % CI **[−0.036660, +0.001017]**; 2 of 8 cell lines positive.

For reference, the row-bootstrap CI reported in the earlier record was **[+0.0036, +0.0049]**.

## WHAT WAS CONTROLLED
Same two fixed sets of predictions throughout; no retraining, so no new training noise is introduced.
Both arms were trained with identical split, seed, schedule, parameter count and lineage input, differing
only in that the ablated arm's chromatin values and availability mask were replaced by their training
means. One seed per arm.

## PRIOR RETRACTIONS IN SCOPE
The +0.0042 pooled row figure and its row-bootstrap interval are already withdrawn on your C1/C2. Do not
re-litigate that; assess what is here.

## ASKS
1. Is the cluster bootstrap the right construction, and is 8 clusters enough to support any statement?
2. Is the placebo stratum a valid noise-floor estimate, given both arms are *different models* that merely
   receive identical zeros on those rows?
3. Does the per-cell-line pattern admit an explanation other than "no effect" — e.g. is it consistent with
   an effect present only in well-covered cells, and can that be distinguished here?
4. H1975 (n=54) contributes −0.0865 and is one of 8 equally-weighted clusters. Is the unweighted
   cluster mean the right estimator, and what would a cell-size-weighted or trimmed version say?

Free to ignore these and raise anything else.
