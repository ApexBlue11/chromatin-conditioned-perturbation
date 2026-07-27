# Network Branch — Pathway & PPI Priors over the 978 Landmark Genes

## Purpose
Build the biological-network priors the model uses for pathway projection / graph
attention over the **978 landmark genes** (canonical order `pathway_landmark_genes.txt`,
`DDR1` … `NPEPL1` — the same axis as the baseline `X_base`). Two complementary priors:
Reactome pathway membership and the STRING protein–protein interaction graph.

## Script
`scripts/network-data.ipynb` (pipeline `v4_corrected_local`). Config: pathway
`min_size=10`, root-umbrella exclusion `> 70` landmark genes, co-pathway normalization at
the 99th percentile, STRING score threshold `400`.

## Inputs → `data/`
- `ReactomePathways.gmt` — Reactome pathway → gene-set memberships.
- `9606.protein.info.v12.0.txt`, `9606.protein.links.v12.0.txt` — STRING v12.0 human PPI.

## What was done
1. **Symbol → landmark alias resolution** (`outputs/network_data_provenance.json`): GMT gene
   symbols mapped to the 978 landmarks — 747 direct, 1 rescued via mygene (`TRM3→TARBP1`);
   several **rejected** as Entrez-collision false matches (e.g. `CHD5→WRB`, `HK2→HOOK2`) rather
   than accepted blindly.
2. **Reactome pathway selection:** 2,855 pathways in the GMT → 369 candidates after `min_size` →
   **9 root "umbrella" pathways excluded** (Signal Transduction, Immune System, Metabolism, …) →
   **360 final pathways**, covering **735 / 978** landmark genes (243 have no qualifying pathway).
3. **STRING graph:** 25,330 edges at score ≥ 400.

## Outputs → `outputs/`
| file | shape | what |
|---|---|---|
| `M_reactome.npy` | (360, 978) int8 | pathway **membership** (1 = gene in pathway); 2.28% nonzero |
| `M_norm_reactome.npy` | (360, 978) f32 | membership normalized for projection |
| `A_copathway.npy` | (978, 978) f32 | gene–gene **co-pathway** adjacency, p99-capped; 12.8% nonzero |
| `STRING_adj_978.npy` | (978, 978) f32 | STRING **PPI** adjacency (threshold 400); 2.65% nonzero |
| `network_data_provenance.json` | — | full config, alias resolution, pathway selection, missing-gene list |

## Notes / caveats
- The provenance flags `STRING_adj_978.npy` as **"NOT used in V2 training"** — retained as an
  alternative/ablation prior; the Reactome matrices are the primary network input.
- Gene order is **frozen** to `pathway_landmark_genes.txt` (978), identical to the baseline branch —
  so `M_*`/`A`/`STRING` columns align 1:1 with `X_base` columns. No `cell_id` axis here (gene-only priors).
- 243 landmark genes are absent from all selected pathways (listed in the provenance) — expected, not a defect.

## Folder layout
- `scripts/` — `network-data.ipynb`.
- `outputs/` — the 4 matrices + provenance.
- `data/` — raw Reactome GMT + STRING v12.0 files (folder/file nesting flattened during cleanup).
- Consolidated from the former `Network Matrix Creation/`, `Reactome/`, `String/` folders.
- Status: **CLOSED.** Matrices are final and align to the 978-gene canonical order.
