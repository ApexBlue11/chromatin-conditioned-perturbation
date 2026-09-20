# PACKET 001 — cold-cell performance and the chromatin ablation
packet_id: 001
created: 2026-09-20
repo_commit: 4e0bc24

## OBJECTIVE
The project predicts drug-induced transcriptional response over the 978 LINCS L1000 landmark genes. Two
questions bear on the paper:

1. How does the model (`v9`) perform on **unseen cell lines** relative to (a) a linear baseline and
   (b) published numbers from an external model (XPert, Nat Mach Intell 2026)?
2. Does the **chromatin input** (ATAC-seq, H3K27ac, H3K27me3 per cell line) contribute accuracy in that
   regime?

## FUNCTION

**Benchmark.** `external/xpert/code/XPert/processed_data/l1000_mdmt_68830_subset.h5ad` — XPert's own
released benchmark. 68,830 conditions, 40 cell lines, 1,977 compounds. Carries their split definitions
`split_1..5` (warm), `split_cold_cell_1..5`, `split_cold_drug_1..5`. Bundled by
`model/v9/xpert_mdmt_extract.py` → `external/xpert_split_bundle/xpert_mdmt_splits.npz`.

**Split used.** `split_cold_cell_1`: 8 held-out cell lines, stated as zero overlap with the 32 training
cell lines. We have chromatin tracks for 5 of the 8, covering a stated 94.3 % of its test rows.

**Target.** `xdeg` = trt − ctl, the delta. Metric is per-row Pearson, then the **mean** across rows
(XPert's `metrics.py` convention). The median is also emitted.

**Arms.**
- `model/v9/xpert_arm.py` trains v9 on the split's training rows. 12 epochs, seed 0, **batch 8**.
- `model/v9/xpert_arm.py --ablate_epi` replaces the chromatin values **and** the track-availability mask
  with their training means, leaving architecture, parameter count and the lineage input untouched.
- `model/v9/xpert_mdmt_baselines.py` fits a closed-form ridge on [control, ECFP4, descriptors, log dose,
  time], with `--with_chromatin --lam_sweep` adding chromatin as 24 exact SVD components.
- `model/v9/head_to_head_mdmt.py` scores arms pairwise on identical rows.

## RESULTS

**Artefacts:** `model/results/v9_vs_ridge_cold_cell_1.json`,
`model/results/v9_chromatin_ablation_cold_cell_1.json`,
`model/results/xpert_mdmt_baselines_split_cold_cell_1*.json`.

`split_cold_cell_1`, n = 21,151 paired rows, delta Pearson (mean of per-row):

| arm | delta | absolute |
|---|---|---|
| copy the control | 0 by construction | 0.9570 |
| mean drug delta | 0.1101 | — |
| ridge, no chromatin | 0.2951 | 0.9584 |
| ridge, with chromatin (best λ per arm) | 0.2980 | — |
| **v9, chromatin ABLATED** (12 ep, seed 0, batch 8) | **0.4692** | — |
| **v9, chromatin ON** (12 ep, seed 0, batch 8) | **0.4734** | 0.9665 |

- v9 (chromatin on) − ridge, paired: **+0.1775 [0.1760, 0.1791]**, v9 better on 93.3 % of rows, Wilcoxon p ~ 0.
- v9 chromatin on − ablated, paired: **+0.0042 [+0.0036, +0.0049]**, chromatin better on 55.8 % of rows,
  Wilcoxon p ~ 0.
- Ridge chromatin gain, other splits: `split_2` (warm) +0.0002; `split_cold_drug_1` +0.0003.
- Seed spread measured for this arm on a *different* split (`split_2`, 3 seeds): 0.0004.

**External published numbers** (XPert Supplementary Table R8, `external/xpert/supplementary/`, fivefold CV
mean ± sd, PCC on xdeg, same L1000_mdmt benchmark):

| model | warm-start | cold-drug | cold-cell |
|---|---|---|---|
| Mean baseline | 0.236 | 0.236 | 0.224 |
| CIGER | 0.525 | 0.397 | 0.236 |
| TranSiGen | 0.635 | 0.609 | **0.293 ± 0.017** |
| XPert | 0.688 ± 0.011 | 0.645 ± 0.008 | **0.383 ± 0.027** |

Separately, on the **warm** `split_2` fold, XPert's released checkpoint was executed here and scored
**0.6933**; v9 trained on the same fold scored **0.7053** over 3 seeds (spread 0.0004), paired
+0.0120 [0.0113, 0.0127] on 13,615 identical rows.

## CODE
Commit `4e0bc24`. `model/v9/xpert_arm.py`, `model/v9/head_to_head_mdmt.py`,
`model/v9/xpert_mdmt_baselines.py`, `model/v9/xpert_mdmt_extract.py`, `model/v9/model_v9.py`,
`model/v9/modules_v9.py`, `model/v9/data_v9.py`. Design tests: `model/v9/test_v9.py`,
`model/v9/test_xpert_compare.py`.

## WHAT WAS CONTROLLED
Both v9 arms share split, seed, schedule, batch size, parameter count and the lineage input; only the
chromatin values and mask differ. Before the run, E's standard deviation across rows was recorded falling
0.5722 → 1.4e-04 under ablation while lineage still varied and targets were byte-identical. The ridge
chromatin arm uses 24 exact SVD components (reconstruction verified to 1e-3) with λ swept per arm, best λ
taken per arm. Rows using compounds that cannot be featurised are dropped from both sides; the paired
comparison refuses to run below 95 % row overlap.

## PRIOR RETRACTIONS IN SCOPE
Do not spend effort re-finding these; they are already recorded:
- An earlier chromatin test was run on the **warm** split, where 99.8 % of test rows share a (cell,
  compound) pair with training. Recorded as near-worthless by construction.
- An earlier ablation-to-1 of a different multiplicative component produced a 30× inflated effect;
  ablate-to-mean is now the standing rule.
- An earlier comparison set our number on our rows against their number on their rows and was reversed
  when both were run on identical rows.

## ASKS
1. Does the +0.0042 chromatin result support a claim that chromatin contributes on unseen cell lines?
2. Is the v9 0.4734 vs XPert-published 0.383 comparison admissible, and if not, what exactly is missing?
3. Is anything in either arm fitted on rows it is scored on?
4. Is `n = 21,151` paired rows sufficient for a 4-decimal-place interval, given one seed?

You are free to ignore these and raise anything else.
