# LINCS — Multi-Encoder Cross-Attention VAE (data-preparation branches)

Predicts drug-induced transcriptional response (LINCS L1000 Level 5 COMPZ z-scores,
978 landmark genes) from a drug (SMILES), cell line, dose, and timepoint.
This repo holds the **input-preparation branches**. Each branch is self-contained with
`scripts/` (execution-ordered), `outputs/`, `data/`, and a `report.md`.
Model-architecture design + related-work notes: **`design_notes.md`**.

## Branches
| folder | produces | shape / key | status |
|---|---|---|---|
| **`baseline/`** | `X_base_lincs.npy` — CCLE baseline expression | (83 cells × 978 genes), keyed by `cell_id` | CLOSED |
| **`epigenetics/`** | `coverage_report_final.tsv` — ATAC/H3K27ac/H3K27me3 per cell | 249 (cell,mark) rows, keyed by `cell_id` | CLOSED |
| **`network/`** | Reactome membership + co-pathway + STRING PPI matrices | (360/978 × 978), gene-only | CLOSED |
| **`phase2_assembly/`** | `Y` target + signature table + epigenetic tensor `E` | 312k sigs; `E` = (83×978×3) | IN PROGRESS — see its `report.md` |
| **drug branch** | UniMol + MolBERTa + descriptors + MoA, keyed by `pert_id` | — | NOT STARTED — see `design_notes.md` |
| `Data Info/` | shared LINCS Phase I/II metadata (sig/inst/gene/cell/pert info) | — | reference |
| `level5 data/` | LINCS Level 5 COMPZ GCTX (both phases) — the target `y` source | — | present |

## Cross-branch joins
- **Cell axis (83):** union of `pert_type=="trt_cp"` cell_ids across GSE92742 + GSE70138.
  Join by **`cell_id`** string (baseline `lincs_cell_index.json`, epigenetics `epigenetics_cell_index.json`).
- **Gene axis (978):** frozen canonical order from `pathway_landmark_genes.txt` (`DDR1`…`NPEPL1`),
  shared by baseline columns and all network matrices.

## Status / next
- **Phase 2 (final assembly)** is UNDERWAY in `phase2_assembly/` (Level 5 files are present). Done so far:
  the `Y` target matrix (both phases, 978 genes), the SMILES exclusion, Checkpoint-3 (kept all signatures
  with a per-mark availability mask + reliability weights — no imputation), and the epigenetic feature
  tensor `E` (peaks for ATAC/H3K27ac, bigWig coverage for H3K27me3). Full detail: `phase2_assembly/report.md`.
- **Data-loading design (decided):** NORMALIZED — a narrow signature table (sig_id, cell_id, pert_id,
  dose, time) + `Y` (N×978), with per-key lookups for baseline/epigenetics (by `cell_id`) and drug
  (by `pert_id`). Never denormalize (each cell repeats ~3,764×).
- **Open:** the drug branch (Phase 3), and the 2 proprietary-SMILES drugs (need the architecture doc to
  identify them). The 31 zero-epigenetic-coverage cells are kept (masked), not excluded.
