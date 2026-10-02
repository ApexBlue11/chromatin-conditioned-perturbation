# REVIEW OF PACKET 042
verdict: SOUND-WITH-CAVEATS
reviewed_commit: ea6c49f

**C1–C5 are implemented as I intended.**
- **Tests:** I reran both test files and got **22 passed**.
- **Pins:** the kernel's new pins (`3de8a1a3`, `eb209adc`) equal the repo and the staged `external/kaggle_chromatin_src`.

**T4 reproduces exactly from the prediction files.**
- **Overall:** ablated − intact = **+0.00596**, per seed +0.0087 / +0.0042 / +0.0050.
- **Cell-centred:** **+0.00563**.
- **By row set:** drug-known rows +0.0071 (3,074 rows), drug-unknown rows +0.0024; top tercile +0.0122.
- **Identity:** the local intact pass equals P2's saved predictions (min per-row r 1.0000000).
- **The ablation is clean:** the dev carve precedes the mean (`xpert_arm.py:130-167` before `:214`), so the ablated value
  is the dev-train mean.

**Keeping P7 unchanged is right, and so is calling the inference-time ablation "not a candidate".**

There are four MINOR points:
- C1: T4's attribution to the failed-ChIP tracks overreaches.
- C2: M1(b) still shares one pool term, a residue of my own C4 settle.
- C3: T3 has no π = 0 fault check.
- C4: §91.10 now says something T4 contradicts.

**The push is cleared once C3 is in.** C2 is optional.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **The failed-ChIP tracks don't explain T4's per-cell pattern.** The largest harm is **LNCAP** (+0.025 to +0.027, by either per-cell convention), whose three tracks are all good. **SKBR3** has the heaviest-tailed *good* track (H3K27me3, reliability 1.00, max \|z\| 20.9; §91.8) yet is slightly *helped* by its chromatin (−0.003 to −0.004). Only 2 of the 3 harmed cells carry a failed track, out of n = 6 cells. "Concentrated in HEK293T and VCAP, the two dev cells whose H3K27me3 is a failed-ChIP track" reads as a cause the data don't single out. | State the per-cell pattern without attributing it: *"two of the three cells harmed (HEK293T, VCAP) carry a failed H3K27me3 track; the most harmed (LNCAP) has three good tracks"*. If the failed-track hypothesis matters for S12, the direct check is cheap: ablate **only** the two failed H3K27me3 tracks at inference on the same three checkpoints. Also state the per-cell convention. My medians of seed-averaged paired row differences are LNCAP +0.0252, VCAP +0.0161, SKBR3 −0.0029, U937 +0.0012, HL60 −0.0130 and HEK293T +0.0225, close to but not equal to §91.11's. |
| 2 | MINOR | confound | **M1(b) still shares a pool term between the two sides. My C4 settle ("form every e with μ^B") left it in.** μ^B for any row contains the mean drug-independent offset of the B-half cells that have its key. The row and its within-cell neighbours mostly see the same B cells, so both residuals carry the same −(B-offset) term, a positive artefact on real data. Random-value data have no cell offsets, so the smoke (+0.0005) can't show it. M1(a) is clean, since its two sides use disjoint halves. | Form the **row's** e with μ^B and the **neighbours'** e with μ^A, keeping selection by μ^A. The row's residual then shares nothing with the neighbours' but the row's own cell offset, which is the signal M1(b) is meant to measure. Since M1 is descriptive, the alternative is to label M1(b) *"an upper bound (shares the B-pool offset)"*. |
| 3 | MINOR | code-vs-intent | **T3's calibration has no π = 0 fault check, though §91.9's fault ("a pass at π = 0 in any draw") applies to every calibrated test.** `one_draw` runs `null_T1` only, and `read_chromatin91.py` can VOID only T1. T3's reading is binding through its MDE label, so its null check belongs in the same run. | In `one_draw`, add `t3_case` on the unplanted y for each draw (≈ 1 s each). Add `T3_pass_at_pi_0_draws` to `instrument_faults`, and have the reader VOID T3 on any pass. Add one test: an unplanted synthetic world gives no T3 pass. |
| 4 | MINOR | wrong-quantity | **§91.10's reading says the earlier work "shows that the trained models did not read chromatin". T4 shows v9 does, and that reading it costs on unseen cells.** The "did not read" rested on §45's inference ablation of **v6** (−0.0001). | Scope §91.10 to *"v6 did not read its chromatin (§45); v9 does, and on dev cells reading it costs +0.006 (T4, §91.11)"*. Carry the same correction into any manuscript sentence about chromatin. |

## Answers to the asks

**Ask 1 — yes. C1–C5 match my intent.**
- **C1:**
  - The `centred_all_ge_bar` conjunct sits beside the cell count in both `t1_reading` and `t2_reading`.
  - The bars are half the raw bar, in `BARS`.
  - Your 10-case P3 test gives 0 passes, and the old sign-only test now fails, as it should.
- **C2:**
  - `install_calibration_tracks` gives FBC\* the real availability, with f\* in the primary slot and the other marks
    permuted, one permutation per mark.
  - The funnel's own `feature_builder` builds FBC\* and N1\*, and T3 uses the same set.
  - Per-draw installation is safe under `fork`: each builder captures its own `enc` dict.
  - One tidy-up: `_CTX`'s comment "never mutated after" is now false, though harmless.
- **C3:** `ROW_SET_OF_RECORD = 'known'`, with the scaled bars used by both the funnel and the calibration.
- **C4:** split pools, apart from C2 above.
- **C5:** the marker carries both output hashes, and the reader refuses without it.
- **What remains:** C3, and C2 if you adopt it.

**Ask 2 — the T4 reading is right on its main points.**
- **Overstated:** the failed-ChIP attribution (C1).
- **Understated: what centred ≈ raw means.**
  - Δ_centred is +0.0056 against a raw +0.0060, so v9's chromatin cost sits almost entirely in the **drug-specific**
    component, not in a per-cell offset.
  - So v9 uses chromatin through the drug-interacting gene-token path. That's direct evidence for review 040 C3's
    correction of §91.1, and worth one sentence.
- **Keeping P7 unchanged is right.**
  - P7's recipe was fixed by §85.11 and §85.12 before T4 existed.
  - T4 is a post-hoc, off-distribution inference ablation.
  - Changing P7 now would be choosing the final model on a dev outcome after the protocol closed.
- **"Not a candidate" is right** for the inference-time ablation. A training-time no-chromatin arm is the legitimate
  candidate, as you say.
- **Two consequences to record:**
  - **The paper:** P7 carries a chromatin input that cost +0.006 on dev cells at inference. The paper should report T4
    beside P7's chromatin description and make no claim that chromatin contributes to P7's accuracy.
  - **Test cells:** any test-cell use of T4 (for example, P7 with chromatin ablated, as a reported row) must be
    registered and reviewed **before** `score_p7.py` runs. Otherwise no test-cell T4 statement can be made.

**Ask 3 — cleared, once C3 is implemented.**
- **What to push:** `apexblue/lincs-chromatin-funnel` and `kern_chromatin91`, regenerated with new pins, after a test
  rerun. C2 is optional.
- **No further packet:** if the diff is confined to C2–C3 and the regenerated pins, I'll check it from the commit.

## What I checked and found sound

- **T4, from the prediction files:**
  - all six reported quantities;
  - the per-seed signs, which agree on all three seeds except U937;
  - the drug-known row count, 3,074.
- **The reader:** the order of its verdicts (VOID, then NOT INTERPRETED, then ADVANCES, then the MDE label) follows
  §91.9.
- **The kernel:** the only change is the output hashes in the marker.

## What I could not assess, and why

- **Whether ablating only the failed tracks reproduces T4's HEK293T and VCAP gains.** That's C1's suggested check, which
  needs a GPU inference pass, and I didn't run it.
