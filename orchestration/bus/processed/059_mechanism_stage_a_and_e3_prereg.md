# PACKET 059 — PRE-REGISTRATION REVIEW: §94 Stage A (data-first mechanism ceilings) and §92.11 (E3, the tie-fixed encoding)
packet_id: 059
created: 2026-10-08
repo_commit: cdcccd8
type: **PRE-REGISTRATION** (no code yet; no output of either exists)

## A. §94 Stage A (RESULTS §94, commit cdcccd8)
**The objective.** The principal's direction is mechanistic interpretability, now that chromatin is closed for transfer.
- **The record:** three model readouts of mechanism were null against their references (C 4.1a, §86, §88), and §86's data
  projection had no target-pathway alignment.
- **Stage A** therefore measures, on the measured P7 test-cell responses alone, whether two standard mechanism readouts carry
  signal. A model readout (Stage B) is registered only where they do.

**The design:**
- **Rows and labels:** the P7 rows (cold-cell split 1 test, 21,151 rows), with Δ = X − X_ctl. Labels are ChEMBL direct
  single-protein MoA strings.
- **Scored cells:** the 5 cells with ≥ 25 labelled compounds in multi-member classes. The counts were taken from identities only.
- **Signatures:** cell-centred compound means.
- **A1:** MoA-mate retrieval AUROC against a label-permutation null. Signal in a cell iff it exceeds the null mean by ≥ 0.05 with
  p < 0.01; A1 carries signal iff that holds in ≥ 3 of 5 cells.
- **A3:** PROGENy (decoupler, top 500, restricted to landmarks) ULM activity, signed by a target-class table fixed in §94.4.
  - **Per unit:** the class mean against the other compounds.
  - **T:** the mean over units, with a permutation p.
  - **Signal** iff p < 0.01 and d > 0 in ≥ 2/3 of units.
- **The decision rules:** in §94.5.

## B. §92.11 E3 (commit 5ca47f6)
- **The arm:** the tie-fixed chromatin encoding, i.e. E1's clean encoding with every 93.16 tie block tied before the rank-normal
  step. Three seeds on the P2 recipe.
- **The rules:** E1's acceptance rules, an E3-vs-E1 comparator, and a decision table, all fixed before E1's seeds 1–2 exist. The
  code comes after E1's read (the freeze).

## ASKS
1. **§94:**
   - Are A1 and A3 the right first readouts? Is any confound built in?
   - The ones I see:
     - seen compounds in the cold-cell split, which bears on Stage B (μ is its reference);
     - cell-centring;
     - CNS-ligand classes that are probably transcriptionally inert in these cells;
     - multi-target compounds belonging to several classes.
   - Are the bars and cell rule sensible? Is the A3 class table defensible? Should any row be dropped or added before the data?
2. Is §94.5's decision logic sound? In particular, is moving to the cold-drug split if neither readout has signal the right
   fallback?
3. **§92.11:** any defect in the E3 design or its decision table?
