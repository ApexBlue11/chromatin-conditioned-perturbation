# PACKET 060 — CODE REVIEW: §94 Stage A (`model/v9/mechanism_stage_a.py`), before its one run
packet_id: 060
created: 2026-10-08
repo_commit: c1f0587
type: **CODE REVIEW** (not run on real data; the run is local CPU, measured responses only, no model output)

## OBJECTIVE
Implement §94 as amended by §94.7: the A1 and A3 mechanism ceilings on the measured P7 test-cell responses, and the mechanical
reader.

## FUNCTION (`mechanism_stage_a.py`, sha1 44bb13e5)
- **`load_rows`:** `split_split_cold_cell_1 == 'test'` (asserts 21,151) from the bundle. Δ = X − X_ctl.
- **`load_labels`:** direct, `Homo sapiens`, any target type. Returns pert → parent, parent → MoA strings, and parent →
  (gene, action).
- **`units_for_cell`:** one unit per parent (labelled) or per pert_id (unlabelled). Its signature is the mean Δ minus the cell
  mean. It also carries the unit's X_ctl row hashes and row lists.
- **`scored_cells`:** the registered five, if they qualify (≥ 25 labelled units in multi-member classes). Others are reported
  only.
- **`a1_cell`:** per-unit AUROC of Pearson similarity, mates (any shared MoA string) against non-mates.
  - **Exclusions:** pairs with intersecting X_ctl hash sets (same plate), and the diagonal.
  - **Null:** labels permuted over units, rng 9400 + k. The vectorised form uses Mate[perm][:, perm], tested equal to a
    brute-force definition.
- **`compute_self_retrieval_and_active_subset`:** C4's ceiling and the active-subset A1 (split-half r ≥ 0.2). Reported only.
- **`load_progeny_network`:** refuses unless the table's sha1 is af40b7a5; keeps landmarks and pathways with ≥ 15 genes.
- **`run_a3`:** decoupler ULM per cell on unit × gene signatures, with explicit gene names, followed by:
  - **the class tables:** the gated table (94.7 item 2) and the ungated one (94.4, reported);
  - **membership signs:** the action type's sign;
  - **per unit:** d against the cell's other labelled compounds; T is the mean of d;
  - **the null:** class memberships permuted within cells, rng 9450.
- **`read_stage_a`:** A1 signal iff ≥ 3 of 5 cells have Δ ≥ 0.05 and p < 0.01. A3 signal iff p < 0.01 and ≥ 2/3 of units have
  d > 0. Then the §94.5 text, with the C4 numbers. `--read` refuses unless the marker's sha1 matches.

## WHO WROTE WHAT
- **W33** (agy gemini-3.8-flash-high): the module and 12 tests.
- **PI fixes:**
  - the vectorised A1 null;
  - the marker check in `--read`;
  - explicit gene names (no `g{i}` fallback), with an overlap assertion;
  - scored cells fixed to the five;
  - 3 tests from a mutation check: the A1 cell count, the A3 2/3 rule, and the agonist sign flip.
- **Tests:** 15 pass, and 7 of 7 mutants are caught.

## WHAT WAS CONTROLLED
- The gene axis is `pathway_landmark_genes.txt` (sha1 5cbcc93b). That is the file `xpert_mdmt_extract.py` asserted the bundle
  identical to (`gene_axis_identical: true`).
- No model prediction is loaded anywhere.

## ASKS
1. Any mismatch with §94 and §94.7? In particular the plate rule, the parent collapse, and membership and sign handling,
   including a compound with several matching targets of different action types (W33 takes the first match).
2. Is the active subset's null (labels permuted within the subset) right?
3. **The run:** local CPU, about a minute, and the reader applied once. Cleared?
