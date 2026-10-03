# PACKET 046 — RESULT: P7 scored once (RESULTS 85.14), the test-cell pathway readout, and the manuscript's P7 wording
packet_id: 046
created: 2026-10-03
repo_commit: 09ed5ea
type: **RESULT + WORDING REVIEW** (no GPU; the one scoring under 85.12 item 5 has happened and is not repeated)

## A. What ran, in order
1. **The kernel:** `lincs-v9p7` v1 was pushed by hand at 08:44 IST, byte-identical to 1cd614c, and was COMPLETE by 14:36. The
   kernel printed no test metric (blinded).
2. **The chain:** at 14:36 it saw P7 COMPLETE with `P7_COMPLETE.json` and pushed E2 s0, which is now RUNNING. t1 is still running,
   so t2 has not started and the E1 upload waits (review 044 C1).
3. **The download:** to `external/kaggle_out/v9p7`, 20 files.
4. **`score_p7.py --dry_check`:** passed. All 20 manifest sha1s match; `_last` = `_snap2` exactly; main = mean of snapshots with
   max |d| 0; O2 is `69484323…`.
5. **`score_p7.py` once:** wrote `model/results/coldcell_h2h_split_cold_cell_1_P7.json`.
6. **The readout pin:** `align_dev.py` `P7_SHA1` was pinned from `P7_COMPLETE.json` (the three `.pt` sha1s were recomputed and
   matched).
7. **The readout:** `align_dev.py --ckpts <3 .pt> --label P7 --readout aux --rows test --no_nulls`, once (P6's rule-8 command, test
   rows), on the local GPU. It wrote `model/results/v9_dev_align_P7_aux_test.json`.
8. **The figure:** F8 rendered from the scored JSON.

## B. The reading (mechanical, by `coldcell_h2h.py`)
- **Cluster:** +0.0503 [+0.0107, +0.0883], width 0.078; **6 of 8** cells favour v9.
- **Verdict:** **NO CELL-LEVEL CLAIM [71.3]**, because ≥ 7 of 8 is not met.
- **The two cells against v9:** CD34 −0.0335 [−0.0417, −0.0262] and H1975 −0.0302 [−0.0402, −0.0174]. BJAB is +0.0128
  [−0.0003, +0.0265].
- **Against §87:** +0.0465, 5 of 8.
- **Secondary:**
  - per seed +0.054 / +0.048 / +0.049 (each 6 of 8);
  - row-pooled +0.0944;
  - centred +0.0303 [+0.0186, +0.0436], 8 of 8;
  - P7-last +0.0418 [+0.0026, +0.0799], 5 of 8;
  - `ensemble_own` 0.498;
  - reproduction 0.38618, admissible.

## C. The test-cell pathway readout (85.12 item 8)
- **Alignment:** 0.2916 / 0.3039 / 0.2895, against a training-row prior of 0.2406 (all 32 training cells; LOCO 0.2307). It beats
  the prior on every seed, so *"beats a cell-agnostic training-row prior"* is **licensed**.
- **In-cell increment:** > 0 in **8 of 8** test cells (≥ 7 needed), with seed means +0.082 / +0.075 / +0.073, so *"in this cell"*
  is **licensed**.

## D. The manuscript (`MANUSCRIPT_v2_DRAFT.md` at 09ed5ea): proposed wording
- **Abstract (ii):** an added sentence on P7, giving 6 of 8, the two cells, the 7 needed, and the cluster mean with its CI.
- **Abstract (iv):** the pathway readout's numbers are now the test-cell ones (0.295 against 0.241, every seed; 8 of 8 cells).
- **§5.1b:** the proposed wording in bold; the cluster mean against §87; the secondaries; one sentence on the centred score
  (all 8 cells), labelled as licensing no claim.
- **§5.4:** the test-cell readout, without nulls.
- **§7:** the run-variance sentence and the readout limitation are updated.

## ASKS
1. Is anything in 85.14 wrong, or read other than mechanically?
2. Is §5.1b's bold sentence the right permitted wording? In particular, "beyond row-level noise" for CD34 and H1975: their per-cell
   CIs are row bootstraps and exclude 0.
3. **The centred sentence** ("consistent with v9's deficit in those two cells lying in each cell's mean response profile"): is it
   licensed as a secondary, or should it go?
4. **The abstract (iv) change.** It replaces dev numbers with test numbers. Is "ranks which pathways move in unseen test cell
   lines" licensed with no permutation nulls run on the test rows? (85.12 item 8 required none; the P6 rule-8 command ran none.)
5. Any objection to the E1 upload and push proceeding mechanically under review 044 once t2 has started?
