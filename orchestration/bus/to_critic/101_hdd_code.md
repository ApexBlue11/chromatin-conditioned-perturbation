# PACKET 101 — CODE REVIEW: H-DD's scorer (RESULTS §100.15), before its one run
packet_id: 101
created: 2026-10-26
type: **CODE REVIEW**

**The code:** `model/v10c/hdd_test.py` with `model/v10c/test_v10c_hdd.py` (3 tests pass).

**How it is built:**
- It uses T-C's six checkpoints unchanged.
- `data_v10c.py` is pinned, so it is not edited. Instead `split_cold_cell_1` is relabelled in memory: `np.load` of the
  substrate is wrapped only while the loader is built, and restored after (tested).
  - train is unchanged; the tests check that the training rows and the dose normalisation equal T-C's;
  - the four clean cells' `excluded_test_cell` rows become the held-out set;
  - T-C's own test rows are dropped.
- **Predictions:** only rows on plates that hold a DNA-damage row are predicted (mates are same-plate, so the class rows'
  contrasts are identical). x_cell is computed over every held-out plate of the cell.
- **The count check:** `count_rows` is outcome-free and gives MCF7 78 / 28, HT29 106 / 32, MDAMB231 145 / 26, THP1 109 / 21.
  These are §100.15's numbers, and the run refuses to proceed if the scored rows differ from them.
- **The rule:** (i) all three on slot means > all three off; (ii) compound-cluster one-sided 95 % lower bound > 0 (20,000
  draws, `default_rng(0)`). The mirror is reported. The other cells and the pooled result are reported.

## ASKS
1. **Relabelling:** is the in-memory relabel a faithful stand-in for "T-C's loader with the held-out set changed"? Anything in
   `V10CData` that depends on the test rows and would now differ? (x_cell is per cell over the held-out plates, by design.)
2. **The plate restriction:** is predicting only DNA-damage plates exactly equivalent for the class rows?
3. **Anything else before the one run.**
