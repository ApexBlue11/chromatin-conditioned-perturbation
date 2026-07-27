# Phase 2 — Final Dataset Assembly

Joins the branch outputs into the model's training data: the **target** `Y` (LINCS Level 5 COMPZ,
978 landmark genes), the per-signature **metadata**, and the **epigenetic feature tensor** `E`.
Baseline (`X_base`) and network matrices are joined at train time by key (not materialized here).

## Data-loading design (DECIDED — normalized, not denormalized)
There are **312,438 trt_cp signatures** but only **83 cell lines** and **21,299 drugs** — each cell
repeats **~3,764×**. So baseline/epigenetics/drug are **NOT** copied per row. Layout:
| file | keyed by | role |
|---|---|---|
| `Y_target_level5_978.npy` (N×978) | row index | Level 5 z-score target |
| `signatures_usable.tsv` | sig_id | cell_id, pert_id, dose, time, phase |
| `../baseline/…/X_base_lincs.npy` | **cell_id** | baseline expression (1/cell) |
| `E_final.npy` (83×978×3) | **cell_id** | epigenetic gate input (1/cell) |
| drug features (Phase 3) | **pert_id** | 1/drug |
The `Dataset.__getitem__` gathers `Y[i]`, `X_base[cell]`, `E[cell]`, `drug[pert]`, dose, time by key.

## Task A — Level 5 target (DONE)
`step6_extract_level5_target.py`: both-phases `trt_cp` × **978 landmark genes** (canonical order,
all 978 mapped to Entrez), pulled from the two Level 5 COMPZ GCTX.
- **`Y_target_level5_978.npy` = (312,438 × 978)**, float32, **0 NaN**. `signatures.tsv` = per-row metadata.
- **SMILES exclusion** (`step8`): the doc's "252 / 2 proprietary drugs" did NOT match reality —
  **79 drugs / 2,324 sigs have NO `canonical_smiles`** (un-featurizable). Those are excluded →
  **`signatures_usable.tsv` = 310,114 usable** (+ `Y_keep_mask_smiles.npy`).
  The **2 proprietary drugs (legal exclusion) are STILL PENDING** — need the architecture doc to identify
  them (see `outputs/EXCLUSIONS_TODO.md`). Nothing beyond the no-SMILES set has been dropped.

## Checkpoint 3 — epigenetic coverage impact (DECIDED)
`step1_checkpoint3_impact.py`: of 312,438 sigs, **67% full / 15% partial / 18% zero** epigenetic
coverage; the designated cell-split test lines (MCF10A, NPC) are both fully covered (safe).
**Decision: keep ALL signatures** with a per-mark **availability mask** + **reliability weights**
(gate = 1 fallback when a cell has no epigenetics) — **no imputation**, no signature dropped for
missing epigenetics. `outputs/checkpoint3_per_cell_impact.tsv`.

## Recovery + revamp — the DATA CEILING (established empirically)
Attempts to raise epigenetic coverage before finalizing:
- `step2`–`step4` targeted recovery of high-signature cells (A375, HA1E…): **unrescuable** — A375 has
  only active-mark data (no Polycomb axis); HA1E/HCC515/YAPC/BT20/HS578T have ~no epi data anywhere.
- `step5` property-matched substitute sweep (all 121 unresolved): **+4 slots** only
  (NOMO1/H3K27ac←H3K9ac; RMGI/SKMEL28/U937←H3K4me3) → folded in by `step7` → **coverage_report_phase2.tsv
  = 132/249** (36 full / 16 partial / 31 zero).
- `step11` revamp (prefer-imputed rule + PRC2/PRC1 repression substitutes): **0 cells gained**
  (cells lacking H3K27me3 lack ALL repression assays; tissue-matched cells have no EpiMap imputation).
**Conclusion:** no fallback tier can add coverage — the missing data does not exist. The model handles
gaps via the mask; the revamp's real value is representation *quality* (below).

## Task B — epigenetic feature tensor `E` (hybrid)
Turns coverage into numbers at **978 gene TSS** (`step9`: MANE Select GRCh38, 977/978; MUC1 unmapped;
lifted to hg19 for EpiMap). Environment/access were the hard part:
- **bigWig on Windows/py3.14 is impossible** → solved with `uv venv --python 3.12 .venv-epi` + **`pybigtools`**
  (remote reads). Cistrome exposes no per-sample bigWig → routed via **GEO GSM → ChIP-Atlas SRX** bigWig/bed.
  Access verified for ENCODE (direct) + Cistrome-via-ChIP-Atlas (73/98) + EpiMap (imputed). **25 Cistrome
  slots (19%) unsourceable → masked.**
- **`step10` peak tensor** (`E_peaks`, 83×978×3): ATAC 32 cells, H3K27ac 35 — validated biology
  (corr(ATAC,H3K27ac)=**+0.37**, corr(H3K27ac,H3K27me3)=**−0.11**, correct signs).
- **H3K27me3 finalized as narrowPeak at TSS** (`step14`): bigWig coverage was intractable on this
  connection (~3–5 h, bandwidth-bound) and **broadPeak H3K27me3 does not exist** in any source (all
  narrowPeak). narrowPeak at the ±10 kb TSS window is biologically sound — Polycomb is enriched at
  *repressed promoters*, which is what we score. Signal is genuinely **sparse** (the 978 landmark genes
  are mostly *active*, not Polycomb targets) — that is correct biology, not low quality.
- **`E_final` = [ATAC peaks, H3K27ac peaks, H3K27me3 narrowPeak]** (83×978×3) + `E_final_mask.npy` +
  `E_reliability.tsv`. Coverage: **ATAC 32, H3K27ac 35, H3K27me3 26 cells; 45/83 cells have ≥1 mark**
  (the rest fall back to gate=1). Windows: ATAC/H3K27ac ±2 kb, H3K27me3 ±10 kb; per-mark rank-normalized.
- **Biology validation:** corr(ATAC,H3K27ac)=**+0.37**, corr(H3K27ac,H3K27me3)=**−0.185** (correct signs).
- **Reliability** (`step15`): base = resolution tier; **9 failed-ChIP H3K27me3 cells** (raw peaks <10:
  H1299, HEK293T, HELA, MDAMB231, NKDBA, PHH, SKMEL1, SKMEL28, VCAP) hard-down-weighted ×0.3.

## Scripts (`scripts/`, execution order)
`step1` checkpoint-3 impact · `step2`–`step4` high-impact recovery (ChIP-Atlas/Cistrome/ENCODE) ·
`step5` substitute sweep · `step6` Level 5 target · `step7` fold substitutes · `step8` SMILES exclusion ·
`step9` TSS coords · `step10` peak tensor · `step11` repression revamp analysis · `step12` H3K27me3
coverage · `step13` merge hybrid tensor. (`step10`/`step12`/`step13` use the `.venv-epi` Python 3.12.)

## Key outputs (`outputs/`)
`Y_target_level5_978.npy`, `signatures_usable.tsv` (+ `Y_keep_mask_smiles.npy`), `coverage_report_phase2.tsv`,
`E_final.npy` + `E_final_mask.npy` + `E_reliability.tsv`, `tss_hg38.tsv`/`tss_hg19.tsv`, `EXCLUSIONS_TODO.md`,
per-step provenance/logs. (`E_peaks*` and `E_h3k27me3_coverage*` are the pre-merge components.)

## Current state / open items
- ✅ `Y` target + usable-signature table; ✅ coverage 132/249 + mask + reliability;
  ✅ **`E_final` (83×978×3) complete** — ATAC/H3K27ac peaks + H3K27me3 narrowPeak, biology-validated,
  reliability-calibrated (45/83 cells have ≥1 mark; the rest gate=1).
- **Deferred (optional):** bigWig-coverage H3K27me3 (marginal gain over narrowPeak-at-TSS; ~3–5 h on
  this connection) — `step12` is ready+resumable if ever wanted on a faster link.
- **Drug branch (Phase 3) — DONE** in `../drug/` (see its `report.md`): descriptors + ECFP4 + UniMol 3D
  + ChemBERTa (Kaggle T4×2) + a ChEMBL/STITCH DTI validation reference. 2 proprietary drugs already in the
  no-SMILES drop (resolved).
- **Pending:** wire the drug branch into the cross-attention model; the final train/val/test split and the
  `Dataset` loader that gathers by key.
- Cross-branch keys: `cell_id` (83) + the 978 canonical gene order — identical across all branches.
