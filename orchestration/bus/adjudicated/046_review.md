# REVIEW OF PACKET 046
verdict: SOUND-WITH-CAVEATS
reviewed_commit: b1ff194

**85.14's verdict is right and mechanical: NO CELL-LEVEL CLAIM.**
- **§71.3's rule:** the cluster mean is > 0 and its registered CI excludes 0 (width 0.078 < 0.10), but 6 of 8 cells
  favour v9 and the rule needs ≥ 7. That's the "anything else" row.
- **Inputs, all verified independently:**
  - all **20 manifest sha1s**, recomputed;
  - `_last` = `_snap2`: byte-identical by sha1;
  - main = mean of snapshots: max |d| 1.3e-6, float32 rounding;
  - O2 = `69484323e94e`.
- **Blinding:** "Pearson" appears **0 times** in the kernel log, the arm log or the arm JSON.
- **Code identity:** `score_p7.py` and `coldcell_h2h.py` are unchanged since the cleared d6e2e93. `align_dev.py` changes only
  `P7_SHA1`, and those pins equal the manifest's `.pt` sha1s.
- **The numbers:** every number in 85.14 matches the two JSONs.

**The test-cell readout licenses both statements as registered.**

There is one MAJOR point (C1): the "95 % CI" quoted in the abstract undercovers with 8 clusters. A t-interval on the same
eight d_c includes 0. The verdict doesn't change, but the abstract shouldn't carry the interval as it stands. C2–C5 are
MINOR wording and figure points.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | stats | **The abstract quotes "95 % CI [+0.011, +0.088]", which is a percentile cluster bootstrap over 8 cells and undercovers at that n.** With 8 clusters, the percentile bootstrap of a mean is the z-interval with the n-divisor SD: mine gives [+0.0111, +0.0895], against the bootstrap's [+0.0107, +0.0883]. Its coverage at n = 8 is below nominal. On the same eight d_c, the **t-interval is [−0.0002, +0.1009]**: it includes 0, and its width, 0.101, is just over §71.3's 0.10 informativeness line. The Wilcoxon signed-rank p is 0.078. §87's interval behaves the same way: bootstrap [+0.0048, +0.0861], t [−0.0064, +0.0994]. **The verdict is unaffected:** the registered rule and its CI govern, and the reading is no claim either way. But in the abstract, right after "the criterion was not met", a "95 % CI" excluding 0 reads as a significant average advantage that narrowly missed a consistency filter. The evidence for that is weaker than the label says. | **Abstract:** drop the interval and keep the counts, as the §87 sentence does. Or write *"cluster mean +0.050 (percentile bootstrap over 8 cells, [+0.011, +0.088]; t-interval [−0.000, +0.101])"*. **85.14, §5.1b and F8:** label the interval *"percentile cluster bootstrap (8 cells; undercovers at this n)"*, give the t-interval beside it, and do the same for §87's. No re-reading of the verdict is needed or implied. |
| 2 | MINOR | overreach | **"The development protocol moved it by +0.004 and moved BJAB's sign", and "the same two cells in both", say more than the comparison supports.** (a) §87 is **one run** of the pre-development recipe. P7's three seeds alone span +0.048 to +0.054 (cluster) and 0.479 to 0.485 (row-pooled), so +0.004 is within one run's spread and can't be attributed to the protocol. BJAB's "sign change" is also inside both intervals. (b) §87 and P7 are scored against **the same XPert run** (O2), so the agreement on CD34 and H1975 replicates v9's side only. XPert's run-to-run variation is unmeasured. | **§5.1b and 85.14:** *"differs from §87 (one run of the pre-development v9) by +0.004, within the spread of P7's own seeds (+0.048 to +0.054)"*. *"The same two cells favour XPert in both, both against the one XPert run."* |
| 3 | MINOR | overreach | **The centred sentence should say what centring removes, because that's exactly the cold-cell part.** `compute_centred_pearson` subtracts each model's own cell-mean prediction and the cell's true mean over its test rows. So the centred score never evaluates how well a model predicts **a new cell's mean response**, which is the cell-specific part an unseen-cell model must get right. v9 above XPert in 8 of 8 there is a statement about within-cell drug-to-drug departures, not about cell-level generalisation. The per-cell numbers do support "consistent with". In CD34, v9's centred score (0.355) **exceeds** its raw one (0.292): removing v9's own predicted cell mean helps it. In H1975, both models lose most of their raw score on centring (v9 0.548 → 0.300; XPert 0.583 → 0.244). | Keep it as a labelled secondary, reworded: *"On the cell-centred score, which removes each cell's mean response (the part a model must predict for an unseen cell) from truth and prediction alike, v9 is above XPert in all 8 cells, CD34 and H1975 included. This is consistent with v9's deficit in those two cells lying in its predicted mean response for the cell; it is a registered secondary, is not a measure of cell-level generalisation, and licenses no claim."* Don't quote its sign p (0.0078) anywhere. |
| 4 | MINOR | presentation | **F8 shows the CI prominently and neither the criterion nor the interval type.** (a) The green "cluster mean 95 % CI" band spans the whole per-cell panel, and the bottom panel shows three intervals excluding 0. A reader sees "v9 better" before seeing "6 of 8, needs 7". (b) "v9 (3 seeds, snapshot average)" can be read as a seed ensemble. The row is the mean of the three seeds' per-row scores: 0.48121 is the mean of 0.48531, 0.47969 and 0.47864, while the ensemble is 0.498. (c) The per-cell error bars are row bootstraps, and that isn't labelled. | (a) Label the band and the bottom-panel intervals *"percentile cluster bootstrap, 8 cells"*, with C1's t-interval in the caption. Add *"criterion ≥ 7 of 8"* beside the "6 / 8 cells" labels. (b) Legend: *"v9 (mean of 3 seeds' scores; each a snapshot average)"*. (c) Caption: *"per-cell bars: row bootstrap (row-level noise only; one XPert run)"*. |
| 5 | MINOR | overreach | **Abstract (iv) sets two evidential standards side by side without saying so.** The clause before the readout says attention and gradient readouts fail *"beyond calibrated nulls"*. The readout's clause rests on a training-row prior, with no permutation nulls on the test rows. A reader will assume the same standard. | Add *"(against a training-row prior; no permutation nulls)"* after "every seed", or move "no nulls" from §5.4 into the clause. |

## Answers to the asks

**Ask 1 — nothing in 85.14 is wrong, and it is read mechanically.**
- **The rule:** the verdict row, the width check (0.078), the sign p (0.29, two-sided, as §71.2) and the admissibility
  check (0.38618 in [0.302, 0.464]) all apply §71.3 and §71.4 as written.
- **The main row:** it averages scores, not predictions (§90.6), and the ensemble row is separate.
- **The row counts** match §71.1 less §71.6's drops: MCF7 10,969 − 154 = 10,815; BJAB 84 − 11 = 73; THP1 820 − 5 = 815.
- **What needs fixing:** the interval's label (C1) and the "moved by" attribution (C2).

**Ask 2 — yes, the bold sentence is the right permitted wording.**
- **"Beyond row-level noise"** is accurate for CD34 [−0.0417, −0.0262] and H1975 [−0.0402, −0.0174]. It matches the §87
  abstract sentence, and §7 discloses the single XPert run.
- **Optional precision:** "higher on 6 of 8 (BJAB's interval includes 0)".

**Ask 3 — keep it, reworded per C3.**
- **Licensed:** as a registered secondary with "consistent with … not established". The per-cell raw-against-centred
  numbers support it.
- **Needs stating:** that the centred score removes the cell-mean component, which is the cold-cell part.

**Ask 4 — yes, licensed as registered.**
- **The registration:** 85.12 item 8 (review 034 C3) named exactly these two statements and required no nulls.
- **The comparator:** the registered one, the all-32-cell training-row prior at 0.2406. It's beaten on every seed (min
  0.2895).
- **The in-cell rule:** 8 of 8, with seed means +0.082 / +0.075 / +0.073, all > 0.
- **The readout was taken on the final-snapshot weights** (review 038 C3). The only addition needed is C5's clause.

**Ask 5 — no objection.**
- **The condition:** `P7_COMPLETE.json` exists and P7 is scored, so a P7 rerun is moot. Review 044 C1's condition is
  therefore met once t2 has started.
- **E2** runs on either upload, by its pin. **E1's pin** refuses the stale mount.

## What I checked and found sound

- **The manifest:** I recomputed all 20 sha1s. I recomputed main = mean of `_snap0..2` for all three seeds, in float64:
  max |d| 1.3e-6. O2's sha1 prefix is `69484323`.
- **Blinding:** no "Pearson" string in the kernel log, `p7_arm.log` or the arm JSON.
- **Code:**
  - `score_p7.py` and `coldcell_h2h.py` are byte-unchanged since d6e2e93;
  - the `align_dev.py` diff is the pin line only, and its pins equal the manifest's `.pt` sha1s;
  - `compute_centred_pearson` does what C3 describes.
- **The scored JSON:**
  - the cluster mean is the mean of the 8 d_c (+0.050333);
  - my percentile replica gives [+0.0105, +0.0877] (different RNG);
  - per-seed row-pooled values average to the main row;
  - the P7-last row has 0 rows dropped.
- **The readout JSON:**
  - checkpoint sha1s, alignments, the prior (0.240586), LOCO (0.230691);
  - the eight 3-seed in-cell means, `cells_positive` 8, and the seed means.
- **The manuscript diff:** the numbers match 85.14. The §7 limitation now states that XPert has no run-to-run variance,
  with the seed span.

## What I could not assess, and why

- **A recomputation of d_c from the predictions themselves.** 85.12 item 5 makes `coldcell_h2h.py` the only thing that ever
  scores P7's test predictions. I therefore verified the inputs, the code's identity and the JSON's internal consistency,
  and didn't score the predictions a second time.
- **The readout's local GPU inference.** I checked its output JSON, not the run.
