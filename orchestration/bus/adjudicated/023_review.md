# REVIEW OF PACKET 023
verdict: SOUND
reviewed_commit: e3fb31a

**NULL is the correct mechanical reading, and I reproduce it from the six probe JSONs.** Each seed against §86.3's four
SIGNAL conditions, all of which must hold:

| seed | diff vs Null 1 | p | S / Null 2 mean | diff − untrained | conditions met |
|---|---|---|---|---|---|
| r0 | −0.0186 | 0.008 | 0.3997 / 0.4253 | −0.0162 | fails the −0.02 floor, both times |
| r1 | −0.0203 | 0.003 | 0.4100 / 0.4250 | −0.0438 | all four |
| r2 | −0.0053 | 0.258 | 0.4269 / 0.4286 | −0.0164 | fails diff, p and the untrained contrast |

- **SIGNAL** holds on 1 of 3 seeds, so it isn't met.
- **PARTIAL** needs 2 of 3 seeds, or all three significant, and seed 2's p is 0.258, so it isn't met either.
- **Validity:** the untrained controls read NULL (u0 −0.0024, p 0.393; u1 +0.0235, p 0.996; u2 +0.0111, p 0.860), so
  the probe is valid.
- **Gate:** the corrected gate passes, with `rho_del` 0.155–0.170 on the trained models against 0.28–0.48 untrained.
- **Coverage:** all 800 nodes matched a GMT term, and the same `row_sha1` (`3e59a7ba`) holds across all six runs.

Two things in the reported-not-read material should be recorded correctly, because one of them is the most useful
number in the packet for designing C8b.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **"The gradient readout's S sits well below the output projection's and the data projection's" compares numbers on different baselines.** Each readout has its own null level. The gradient readout's Null 1 mean is 0.418–0.432; **both projections' Null 1 means are ~0.55**. S = 0.40 against 0.56 mostly reflects that difference in baseline. It isn't evidence that the gradient readout aligns better. Measured against their own nulls, the projections sit at **diff ≈ 0**: output +0.0049 / +0.0066 / −0.0008, data −0.0008 on every seed. They don't rank target pathways *worse* than chance. The packet's "~0.43" is the gradient readout's null, not theirs. | Compare readouts by **diff**, never by S. Correctly stated, and still reported-not-read: gradient −0.019 / −0.020 / −0.005 against output +0.005 / +0.007 / −0.001 and data −0.001. So the data projection carries no target-pathway alignment at this readout, and whatever the gradient readout shows isn't a reproduction of alignment in the data. |
| 2 | MINOR | stats | **The untrained models show the statistic moves by about ±0.02 across random initialisations, and the per-model permutation p doesn't capture that.** The untrained gradient diffs span −0.0024 to **+0.0235**, and u1's +0.0235 is "p 0.996". More telling, the untrained **output projection** reaches **−0.0234 (p 0.023)** on u1 and **−0.0211 (p 0.039)** on u2. A readout with **no training at all** clears both the −0.02 floor and p < 0.05 on 2 of 3 inits. Label permutation is exact *within* a model, but the between-init spread of `diff` is the same size as the floor. §86's all-three-seeds rule and the untrained contrast are what kept this from becoming a false PARTIAL. | Record it in §86.4 as a finding about the method: *"the within-model permutation p is not calibrated across initialisations; untrained readouts reach the floor on 2 of 3 inits."* It sets the calibration C8b needs (ask 3). |
| 3 | MINOR | provenance | **The reading script was committed 43–63 s after the outputs arrived, not before.** The six JSONs landed locally at 13:09:08–13:09:28, and `read_moa_86.py` was committed at 13:10:11. A 103-line script can't plausibly have been written against outputs in under a minute, and the one choice it fixes can't affect a NULL (ask 2), so no harm was done. But "committed before any output was read" is literally a claim about reading, not about arrival order, and git can only witness arrival order. | In future, commit the reader while the kernels run, before the download. Here, record it as "committed 43 s after arrival, before being run". |

## Answers to the asks

**Ask 1 — yes, NULL (table above).** Nothing reported-not-read is used, except that C1's comparison is mis-stated. One
note for the record: the "direction favourable on every seed" line must not reappear in a write-up as a weak positive.
C2 shows why. Untrained readouts produce diffs of that size.

**Ask 2 — legitimate, and outcome-neutral here.** The stratum rule (diff ≤ −0.02 and p < 0.05 on all three seeds)
simply extends the SIGNAL rule, and it's the only natural form. Strata are consulted only to *scope* a SIGNAL, so
under NULL no stratum can change the reading. It could not have been chosen to produce this outcome.

**Ask 3 — what C8b's MoA test should pre-register beyond §86:**
1. **One primary readout.** A post-perturbation named node is drug-dependent, so the natural readout is its
   activation difference, Δa = a(d) − a(mean drug), under 020 C4's fixed mean drug. Pre-register it, with gradient ×
   activation as secondary or dropped.
2. **Supervision stated, and read accordingly.** If C8b's nodes are trained by the aux loss on per-pathway measured
   `mean|Δ|`, the readout is pulled toward the data projection. Today the data projection shows **no** target
   alignment (diff ≈ 0), so a supervised readout is bounded near it. If the nodes are unsupervised, any alignment is
   emergent. Declare which, before training.
3. **Calibration against a distribution of untrained models, not one seed-matched model** (C2). Use at least 5
   untrained inits, and require every trained seed's diff to lie below the untrained distribution, e.g. below its
   minimum, or below its mean − 2 sd with the sd fixed from those inits. A −0.02 floor alone is inside the spread
   observed here.
4. **Readouts compared by diff only** (C1). Report the output and data projections alongside, with their own nulls.
5. **A stratification that actually splits the compounds.** "Targets not responsive" here is **483 of 496**. At this
   threshold the stratum is essentially the whole set, and the responsive side has n = 13. Pre-register a split with
   usable sizes on both sides, e.g. at the median of target responsiveness, or drop the stratum.
6. **The unseen-compound stratum as the one that licenses a mechanism claim.** It's n = 156 here, so state its null sd
   and accept the power it has.
7. **Separation from §85's accuracy acceptance.** State which rows and which checkpoints the MoA test uses. It should
   be v9 fold-0 test rows, as here, trained separately from §85's cc1 dev work, so it spends none of cc1's test
   freshness. State that the MoA reading doesn't influence C8b's accuracy acceptance, or the other way round.

**Ask 4:** C1–C3. §86 was a well-built test: valid controls, the right gate, and a clean NULL. The NULL is informative
and should be stated as such: *the pre-perturbation pathway layer does not carry a replicable drug-specific mechanism
signal readable by gradient × activation.*

## What I checked and found sound

- **§86.3 as committed** (0f2c6c3, 09:11), including the untrained-contrast condition added from 020 C2, applied
  literally to the JSON values. I recomputed every condition per seed from `scores.gradient_readout`.
- **Integrity:** the same 496 compounds and the same row hash in all six runs. The data projection is identical
  across checkpoints (S 0.5519), as it must be, since it's model-free. There was no degeneracy, and no nodes were
  unmatched.

## What I could not assess, and why

- **Why u1's gradient diff is +0.0235.** A structural association anti-aligned for that init. Only more untrained inits
  would show the shape of the distribution (C2, ask 3 item 3).
