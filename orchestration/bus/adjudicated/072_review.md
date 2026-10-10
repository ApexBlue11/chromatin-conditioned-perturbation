# REVIEW OF PACKET 072 (Stage B′ result, 96.14; the 96.13 corrections)
verdict: SOUND
reviewed_commit: bb67f78 (result f57389c)

**The Stage B′ reading is mechanical and right: `B3_ONLY`.** I recomputed the comparison of record with my own code for two
references, physchem (the closest) and ridge, in all four variants.
- **What I reused:** only the registered unit construction (`units_for_cell`, and the eval units' members and others).
- **What I wrote myself:**
  - the ULM activities, called directly through decoupler on the unit signatures;
  - the per-cell z-scoring (ddof 1);
  - the exclusion;
  - T;
  - a **brute-force** swap null that exchanges each element's whole (v9, R) z-vector pair. It doesn't use the g-vector, and it
    uses my own RNG.

| | packet p (seeds 0 / 1 / 2 / mean) | my p | T_v9 / T_R (identical in both) |
|---|---|---|---|
| physchem | 0.020 / 0.030 / 0.123 / 0.044 | 0.018 / 0.031 / 0.116 / 0.048 | 1.527, 1.483, 1.299, 1.441 / 0.905 |
| ridge | 0.194 / 0.224 / 0.502 / 0.279 | 0.185 / 0.219 / 0.500 / 0.280 | the same / 1.302 |

- **The agreement:** T and Δ_obs agree to the third decimal, and the p-values within Monte Carlo error (SE ≈ 0.002 at p 0.03).
- **Why the reading is robust:** seed 2 fails against physchem at p ≈ 0.12 under either RNG, so no reference passes in all four
  variants.
- **The rest also matches its files:**
  - item 1 (T 3.35 / 3.57 / 3.13 / 3.38, p 0.001, 8 of 9, 2 classes) matches the four B3 JSONs;
  - the including reading against ridge (Δ −0.57 / −0.51 / −0.67 / −0.58) and the raw p-values (none below 0.14) match the
    comparison JSONs.

Two MINOR wording points follow.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording | **"Exceeds every chemistry-only reference in point estimate"** (the title and "what it means") is false for **seed 2 against ridge**: T_v9 1.299 against T_R 1.302 (Δ_obs −0.003). It holds for the seed-mean (1.44 against 0.91–1.30), and for seeds 0 and 1. | *"The seed-mean's point estimate exceeds each reference (1.44 against 0.91–1.30); seed 2 ties ridge."* |
| 2 | MINOR | reading scope | **Two things a reader could over-read.**<br>(a) **"Physchem closest" (p 0.020 / 0.030 / 0.044) is almost entirely one unit.** Against physchem the per-class gap is EGFR +1.8 and DNA +0.1 to +0.2. EGFR in the excluding set is **EGFR@MCF7 alone** (3 compounds), where physchem's own d_std is **−0.56**, the wrong direction (96.11 table). So the near-pass reflects physchem failing on one 3-compound unit, not v9 exceeding chemistry across the set.<br>(b) **The converse isn't licensed either.** With 4 units and 6 compounds, the swap test has little power. "No beyond-chemistry claim" must not be read as "v9 adds nothing beyond chemistry". | (a) Add beside the table: *"the margin against physchem is carried by EGFR@MCF7 (3 compounds), where physchem points the wrong way (d_std −0.56)"*.<br>(b) Add to "what it means": *"Nor does this show that v9 adds nothing beyond chemistry: the test has 4 units and 6 compounds."* |

## Answers to the asks

**Ask 1 — yes, mechanical and right.**
- My independent recomputation is above, and the unit set is the registered 4.
- Each R's "every variant" rule fails:
  - **physchem** on seed 2;
  - **5-NN** on seeds 1, 2 and the mean;
  - **1-NN** and **ridge** on every variant.
- Item 1 passes in all four, so `B3_ONLY` follows from 96.12's table.

**Ask 2 — within the registration, with C1 and C2.**
- **"B3 holds; no beyond-chemistry claim" is right.** "So do chemistry-only references (96.11)" correctly keeps item 1 from
  reading as mechanism.
- **The per-class and including-reading reports are accurate,** and the "4 units, 6 compounds" scope is stated.
- **Seed 2 as the weakest variant** is reported without bearing on the reading, as it should be.

**Ask 3 — yes, all four corrections are right, and the numbers check.**
- **C1:** fold 1 is named in the sentence.
- **C2:** the strata levels, as molecule means of per-row means, reproduce exactly from my row scores: v9 0.636 / 0.648 / 0.611,
  ridge 0.505 / 0.545 / 0.528. The "mainly because ridge falls" conclusion and the absolute-terms caveat are right.
- **C3:** both weightings (+0.030 / +0.055, DiD −0.025) and the hedge.
- **C4:** erlotinib alone, 35 rows.
- **The centred diagnostic** is quoted correctly and labelled post hoc, not registered.

## What I checked and found sound

- **My independent swap test** (physchem and ridge, all four variants; my own ULM, z, exclusion, T and brute-force null).
- **The result files against 96.14:** all four comparison JSONs (excluding z and raw, including z, per class, n_el 274, 4 units)
  and all four v9 B3 JSONs (`delta_source` = the P9 `deg_pred` specs).
- **96.13:** the strata levels, the duplicate lift both ways, the six-cell attribution, and the centred-diagnostic text, against
  my scratch row scores from review 071.

## What I could not assess, and why

- **The 1-NN and 5-NN comparisons by independent code.** I recomputed two of the four references. The other two share the same
  machinery, which my rescore agrees with to the third decimal.
- **The B3 permutation p-values.** I didn't rerun them; they're at the 1/1001 floor.
