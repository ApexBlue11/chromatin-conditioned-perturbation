# PACKET 016 — A1 (chromatin-gated edges) zero-parameter pre-tests: both null, one after a disclosed fix
packet_id: 016
created: 2026-09-24
repo_commit: e637591
type: **RESULT.** 0 GPU-hours. O2 session 1 is running on Kaggle; nothing here touches it.

## Background
IDEAS A1: chromatin should decide *which graph edges conduct in a cell*, not how much a gene moves (the per-gene form
measured null). Its trained arm costs ~5.8 GPU-h. Before spending, two zero-parameter pre-tests, each pre-registered
before any code existed.

## Test 1 (RESULTS 82, committed 602c66f): landmark targets on the landmark union graph
Random walk with restart (alpha 0.5 fixed) from a compound's DTI targets that are themselves landmarks, over the
binary union of STRING / Reactome / GO:BP on the 978 landmarks (81,846 edges); edges gated by `sqrt(a_i a_j)` of ATAC.
Score: Spearman(s, |z|) at non-target landmarks, Level 5. Conditions OWN (own ATAC), MISMATCH (5 other cells, averaged;
the null key), MEAN, NONE (ungated), DEGREE. Unit = cell line; cluster bootstrap. Pre-committed first rule:
*median(NONE) <= 0.01 or NONE−DEGREE CI includes 0 ⇒ UNINFORMATIVE.*
```
ATAC:    32 cells, 52,496 rows. median NONE -0.0006; NONE-DEGREE +0.0010 [-0.0001, +0.0023]   -> UNINFORMATIVE
         OWN-MISMATCH +0.0002 [-0.0004, +0.0009], 16/32   (reported, not read)
H3K27ac: 35 cells, 60,406 rows. median NONE -0.0007; NONE-DEGREE +0.0010 [+0.0000, +0.0021]  -> UNINFORMATIVE
```
Recorded limitation: the DTI table lists only landmark targets. My first write-up said a non-landmark table was needed;
one already existed (`chembl_dti_edges.tsv`) — corrected the same day (method rule 20).

## Test 2 (RESULTS 83, committed 00e5316): the prerequisite on the full graph
Same propagation from ALL ChEMBL mechanism targets over full STRING (19,496 nodes, 929,472 weighted edges), read out
at the landmarks, no gating. Primary null: the same number of RANDOM targets (5 draws). Pre-committed readings:
*SIGNAL* needs median(NONE) > 0.01 AND NONE−RANDOM mean > 0, CI excluding 0, ≥ 75 % of cells; else
*SPECIFIC_BUT_NEGLIGIBLE* if only the NONE−RANDOM part passes; else *NO_SIGNAL*.

Worker defects I fixed before any number was read: the brief's landmark-order assertion was wrong (34 of the canonical
L1000 symbols are older aliases of the graph's HGNC names; the assertion now checks the l1000→HGNC chain via
`landmark_symbols_v9.tsv`); the secondary contrast was omitted; the tests exercised a re-implementation, so the real
script now asserts W's symmetry/no-doubling and the fixed-point residual (< 1e-8) on the real graph.

**Disclosed:** the first full run's primary contrast was NaN — a random draw on isolated nodes gives a constant vector,
so its Spearman is undefined, and one NaN poisoned the row mean and then the cell median. Fixed (mean over defined
draws; per-contrast paired-complete rows), estimand unchanged, committed (0506102) before the rerun. I had seen the
first run's medians (NONE −0.00016), so the fix could not produce SIGNAL.
```
1,149 compounds, 73,983 rows, 78 cells; 125 ChEMBL symbols unmapped; 36 iterations
median NONE -0.00016 | RANDOM -0.0029 | DEGREE -0.0025
NONE-RANDOM  +0.0019 [+0.0001, +0.0032]  57/78 cells (73 %)  sign p 5.6e-5   -> NO_SIGNAL (75 % bar missed)
NONE-DEGREE  +0.0021 [+0.0006, +0.0033]  56/78               p 1.5e-4
78 of 369,915 random-draw scores undefined
```
Recorded: real but tiny drug specificity, ~0 absolute; the reading does not hinge on the 2-point miss (the median floor
alone rules out SIGNAL). The diffusion route to A1 is closed; A1 needs a trained model, queued behind O2.

## ASKS
1. Are both readings correct as pre-committed? Is anything reported-not-read being used?
2. Was the NaN fix legitimate after the first run was seen — estimand unchanged, reading provably unaffected — or
   should the first run's NaN-contaminated output have been the record?
3. Is "the diffusion route to A1 is closed" the right strength, given the landmark-only chromatin (the full-graph
   gating could act only on landmark–landmark edges) and the 125 unmapped symbols?
4. Anything else, including whether A1's trained arm is worth pricing at all after two nulls.
