# Baseline Branch — CCLE Baseline Expression (`x_base`)

## Purpose
Produce **`X_base`** for the Multi-Encoder Cross-Attention VAE: a matrix of
**83 LINCS cell lines × 978 landmark genes** giving each cell line's baseline
(unperturbed) transcriptional state. This is the tensor the epigenetic gate
modulates: `x_mod[cell,gene] = x_base[cell,gene] * gate[cell,gene]`.

## Cross-branch join keys (shared with epigenetics + network)
- **Cell axis:** 83 cell lines = union of `pert_type=="trt_cp"` cell_ids across LINCS
  Phase I (GSE92742) + Phase II (GSE70138). Join to other branches by `cell_id`
  string via `outputs/ccle_baseline_lincs_v5/lincs_cell_index.json`.
- **Gene axis:** 978 landmark genes in the **frozen canonical order** from
  `Network Data/pathway_landmark_genes.txt` (first `DDR1` … last `NPEPL1`) — the
  same order used by the network branch's matrices.

## What was explored → what was done
1. **Phase reconciliation** (`scripts/step1_analyze_phases.py`): confirmed Phase I and
   Phase II are genuinely distinct files (P1 sig=473,647 / inst=1,319,138 rows;
   P2 sig=118,050 / inst=345,976). Cell breakdown: 53 P1-only, 12 P2-only, 18 both → **83 union**.
2. **CCLE ingest** (`scripts/step2_load_ccle.py`): parse DepMap
   `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` (log-TPM) + `Model.csv` metadata.
3. **Canonical build** (`scripts/step3_build_x_base_lincs.py` — the ONE script that
   reproduces the final output; self-contained, SHA-tracked):
   - Match each LINCS cell to CCLE by **exact normalized name** (uppercase, strip
     non-alphanumeric) against `StrippedCellLineName`, `CellLineName`, `CCLEName`.
   - **Resolution:** 57 direct CCLE matches; **26 DMSO fallbacks** (own-line
     `ctl_vehicle` profiles: 15 from Phase II, 11 from Phase I) for cells absent from
     the CCLE expression matrix; 0 unresolved.
   - **Engineered-derivative policy:** derivatives (e.g. NPC.CAS9) never inherit a
     parental CCLE baseline — own-line DMSO only.
   - **Gene axis:** matched by **Entrez ID** (not symbol) against HGNC
     `hgnc_complete_set.txt`; all 978 landmark genes resolved, 0 excluded.
   - **Scaling:** `StandardScaler` fit on the **full CCLE background population**
     (`ccle_full_background.npy`), applied to the 83×978 matrix.
   - **No imputation** anywhere (verified in self-audit: no `fillna`/`SimpleImputer`/mean-fill).

## Outputs — `outputs/ccle_baseline_lincs_v5/` (canonical, validated)
| file | what |
|---|---|
| `X_base_lincs.npy` | **(83, 978) float32**, StandardScaler-normalized — the model input |
| `X_base_lincs_raw.npy` | (83, 978) pre-scaling log-TPM |
| `ccle_full_background.npy` | full CCLE background used to fit the scaler |
| `lincs_cell_index.json` | **cell_id → row index** (the cross-branch join key) |
| `ccle_resolution_lincs.tsv` | per-cell resolution (direct / DMSO-fallback + source) |
| `gene_resolution_report.tsv` | per-gene Entrez resolution (978 rows) |
| `ccle_baseline_provenance.json` | full input manifest, hashes, resolution breakdown |
| `validation_report.txt`, `methodology_summary.txt`, `self_audit_checklist.json`, `script_sha256.txt` | validation + audit |

Validation: shape (83,978), 0 NaN (raw & scaled), gene-mean range [-1.85, 4.09].

## Inputs
- `CCLE/` — DepMap expression CSV (305 MB) + `Model.csv` (in this folder).
- **LINCS metadata** (sig/inst/gene/cell info, GSE92742 + GSE70138) — the branch's own duplicate
  copy was removed as redundant; use the **shared top-level `../Data Info/`** folder instead
  (update `METADATA_DIR` in `scripts/step3_build_x_base_lincs.py` accordingly).
- `Network Data/pathway_landmark_genes.txt` — the frozen 978-gene canonical order (in this folder).
- `hgnc_complete_set.txt` — HGNC gene-symbol/Entrez authority (in this folder).

## Cleanup performed (this reorganization)
- **Removed** superseded output versions `ccle_baseline`, `..._v3`, `..._v4` (kept only v5).
- **Removed** `lincs_env/` (a virtualenv bound to a non-existent base interpreter
  `C:\Users\apexb\...Python311` — non-functional on this machine). Replaced by
  **`requirements.txt`** (Python 3.11.9; 11 packages incl. numpy/pandas/scikit-learn/h5py).
- **Removed** `GSE70138_Level3.gctx-002.h5` (17 GB). Verified it was the **genuine
  Phase-2 Level 3** matrix (345,976 instances × 12,328 genes) — used only for the
  (completed) DMSO fallback and **not needed for Phase 2** (which uses Level **5** COMPZ).
  Re-obtainable from GEO GSE70138 if ever required.
- **Removed** `__pycache__/`.
- **Removed** the branch's duplicate `Lincs metadata/` copy (identical to the shared top-level
  `Data Info/`) — ~311 MB de-duplicated.
- Scripts split into `scripts/` (step1–3), `scripts/superseded/` (04–06 early
  x_base iterations that produced the removed `ccle_baseline` output), `scripts/checks/`
  (diagnostics: `check_*`, `inspect_*`, `validate_xbase.py`, `compare_phases.py`, …).

## Notes / caveats
- The canonical builder's paths reference the original authoring machine
  (`C:\Users\apexb\Downloads\LINCS Project`); update `BASE_DIR` before re-running here.
- `03_build_x_base.py` was dead scratch (crashed; not imported by step3) — already absent.
- Status: **CLOSED.** `X_base_lincs.npy` is final and feeds Phase-2 assembly by `cell_id`.
