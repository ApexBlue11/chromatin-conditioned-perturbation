# REVIEW OF PACKET 085 (Phase 1 local results, §99.6), with C3 (§99.7) and C2b (§99.8), read after this packet was filed
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 33d7eac (§99.7 at 51c5b04, §99.8 at 1a9e5ce)

**Every number I checked reproduces with my own code** (own plate hashing, own vectorised contrast, own molecule reading):

| | record | my rescore |
|---|---|---|
| C1 v9 − M+dt (contrast) | +0.0756 [+0.0616, +0.0901], 263 / 337 | **+0.0756 [+0.0616, +0.0904], 263 / 337**; row means 0.109 / 0.032 |
| C1c (original target) | lookup 0.662, v9 0.654, M 0.661 | **0.662 / 0.654 / 0.661**; v9 − lookup −0.0184 [−0.0221, −0.0146]; M − lookup −0.0041 [−0.0059, −0.0026]; 79.0 % of clean rows |
| C3 intact − swapped (contrast) | +0.0700 [+0.0540, +0.0865], 247 / 337 | **+0.0700 [+0.0541, +0.0868], 247 / 337** |
| C3 intact − swapped (original target) | +0.0171 [+0.0123, +0.0223] | **+0.0171 [+0.0122, +0.0222]**, 256 / 346 |

**The findings stand:**
- **C1:** v9 beats M+dt on the within-plate contrast.
- **C3:** v9's predictions depend on the right compound.
- **C1c:** a plate lookup matches or beats both models on the original target.

**One interpretation in §99.7 says something the data contradict** (C1). The rest is wording.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | interpretation (§99.7) | **"Our drug-use tests on the standard target could not see a real effect" contradicts C3.**<br>• **C3 itself shows the effect on the standard target:** the right compound beats a dissimilar one by **+0.0171 [+0.0122, +0.0222]**, 256 / 346 molecules, a CI well clear of 0.<br>• **What the standard target hides is the *benchmark margin* over a drug-free model,** because that model recovers the plate through the exact control (M2; C1c: M 0.661 against the lookup's 0.662).<br>• So "M3 confirmed in the sense that matters" should read: *on the standard target, compound use is visible to a compound-swap test but not as a margin over a drug-free model.*<br>• **"That is why … Bai et al.'s benchmark read 'no drug use'" extrapolates beyond the data.** C3 concerns v9 only, on this bundle. Bai et al. tested seven *other* models, on single-dose data, by retrain ablation. The mechanism (the plate absorbed by control-matched baselines) is a hypothesis for their result, not a finding about it. | **Reword §99.7's reading:**<br>*"v9 uses the compound (C3): +0.070 per molecule on the contrast, and +0.017 on the standard target (both CIs exclude 0). On the standard target that use doesn't appear as a margin over a drug-free model, which recovers the plate through the control (C1c). The same mechanism may explain Bai et al.'s 'no drug use' on their benchmark. That is untested here."*<br>Apply the same to M3's status. |
| 2 | MINOR | wording / scope (§99.6, C1c) | **"The field's cold-compound benchmark is won by a plate lookup" and "beats every model"** go beyond the comparison made.<br>• **The scope:** 9,466 clean rows with ≥ 2 training plate-mates (79 %).<br>• **The comparison:** two models (v9, M) on this split. XPert / O9 and PertDiT weren't scored on C1c. | *"On the 79 % of clean test rows with ≥ 2 other-compound training wells on their plate, the mean training response of those plate-mates scores 0.662, at or above v9 (0.654, −0.018 per molecule) and M (0.661, −0.004)."* Add O9 to C1c when its predictions exist (reported). |
| 3 | MINOR | wording (§99.6) | **"R1's tie hid a real advantage"** needs its estimand. The +0.076 is on the within-plate contrast. On R1's target there is no advantage, and that is R1's result. | *"R1's tie on the standard target coexists with a +0.076 per-molecule v9 lead on the within-plate contrast."* With C3, attribute the lead to compound use. |
| 4 | MINOR | label bug (C1c JSON) | **The reported `m_vs_training_plate_predictor` verdict reads "…predicts unseen compounds better than v9"** when the comparison is M against the lookup. `score_p9.verdict` hard-codes "v9" as the left-hand model. | Pass the left-hand label, or rewrite the string for non-v9 comparisons. It is reported only, so a note in §99.6 is enough if the JSON stays as run. |
| 5 | MINOR | interpretation (C4) | **"M7 is weaker than stated" rests on v9's mean ρ (0.36 against the truth's 0.47), but M+dt's group-level agreement with the truth is 0.20, close to v9's 0.25.** M+dt has no compound input, and its mean ρ is −0.02.<br>Both magnitudes subtract the same measured P_meas, so part of corr(ρ_truth, ρ_pred) is shared plate structure.<br>**What C4 shows:**<br>• v9 scales response size with dose on average;<br>• compound-specific dose coupling is barely above a compound-blind model's. | Report M+dt's corr (0.20) beside v9's. Word M7 as *"v9 couples response size to dose on average; compound-specific dose coupling is weak (0.25 against M+dt's 0.20)"*. Keep F5 as a fix, not only a cleanup. |
| 6 | MINOR | wording (§99.8, C2b) | **"So the collapse is fully explained by input shift" overstates the bound.** The seen-cell drop (−0.252) is input shift **plus** the legitimate plate-deviation signal (review 084 C2): an upper bound on pure shift, not a measure of it.<br>The correct conclusion, which §99.8's own heading states, is that the test can't attribute the drop to transduction. | *"The unseen-cell drop (−0.223) is no larger than the same replacement costs in seen cells (−0.252), where nothing transductive is at stake. So this test can't attribute any of it to transduction."* Drop "fully explained". |

## Answers to the asks

**Ask 1 — mostly.** C1's statement is right. §99.6's "hid a real advantage" needs its estimand (challenge 3). §99.7's
reading needs challenge 1's correction, and that is the important one.

**Ask 2 — not as stated.** Use challenge 2's scoped sentence. The finding itself is strong and reproduces.

**Ask 3 — no objection to reporting H1975 as is.** F_tr is exactly 0 there by definition: one shared control means zero
deviation. Report the cluster mean both with and without H1975, labelled.
- **Note:** for H1975, v9's own `x_cell` (the mean of its test controls) also equals `X_ctl`. So C2b's replacement changes
  nothing for that cell, and its Δ is 0 by construction too.

**Ask 4 — C3 and C2b have been read.** Their reproduction and wording are covered above: C3 by challenge 1, C2b by challenge 6.

## What I checked and found sound

- **Independent rescoring:**
  - C1 (primary, row means);
  - C1c (lookup, v9, M, per-molecule contrasts, coverage);
  - C3 (contrast and original target, of-record variant) from the downloaded `swap_seed{k}.npz`, aligned to P9 rows, with the
    targets checked equal.
- **The 99.5 fixes in code:**
  - `floor_contrast` over floor-bearing mates;
  - the seen-cell bound in `read_c2b`;
  - the donor equality across seeds;
  - `free` on the tani rows.
- **The output JSONs for C1, C1c, C2a and C4,** against the §99.6 text.

## What I could not assess, and why

- **C1b, C2a, C2c and C2b's values.** I didn't rescore them independently. I read their JSONs, and checked C2b's reading
  against its rule and the bound.
- **The Kaggle runs' logs beyond the manifests** the reader verifies.
