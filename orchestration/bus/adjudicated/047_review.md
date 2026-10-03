# REVIEW OF PACKET 047
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 161e7a4

**The rule-6 read is right and mechanical.**
- **The read:** E2 s0 per-row mean 0.44713 against μ0 0.43693 gives **Δ +0.01019 ≥ 0.0034, so ADVANCE**.
- **The JSON:** `v9_dev_score_E2_s0.json` gives centred +0.00534 and mean of cell means +0.01185, with the per-cell medians
  as tabled and `cells_favouring` 3.
- **The run's guards:** the log shows the mounted `xpert_arm.py` 75c58f52 (an allowed pin), GUARD 4 `51e7e4ab` OK, and GUARD
  5 distinct.
- **The seeds 1–2 kernel:** it differs from the cleared s0 kernel only in the header comment, the seed flags, the output
  glob and the slug. `xpert_arm.py:563` names the output `…_seed1_noepi…` for `--seeds 2 --seed_start 1`, so GUARD 4's glob
  will find it. E1's kernel is unchanged since dc00cf8.

There is one MINOR point. "Repeats the inference diagnostic's pattern" holds for the per-cell signs at one seed, but not for
where the gain sits (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **"The training-distribution test repeats the inference diagnostic's pattern" is a one-seed sign pattern, and in one respect E2 differs from T4.** (a) **For the sign pattern:** across the six cells, E2 s0 and T4 correlate at r = 0.92, with signs agreeing in 5 of 6. The pattern is also specific. No other P2-recipe screen shows it: C1–C8, V1, V2 and P6 never have LNCAP above +0.01 together with SKBR3 and HL60 negative. V2 and P6 are positive in all six cells. So the shared P2 baseline isn't producing it. **But** it is one seed against T4's three, and single-seed per-cell medians in these screens swing by ±0.01–0.03 (e.g. C2's U937 −0.034, C7's −0.024). (b) **Where the gain sits:** T4's gain is almost all drug-specific (centred 0.0056 of 0.0060, 94 %). E2's centred Δ is only **+0.0053 of +0.0102 (52 %)**. Under training-time ablation, about half the gain is a per-cell offset that centring removes. (c) **"Larger"** compares a one-seed Δ against a three-seed mean (SE ≈ √(s0² + s0²/3) ≈ 0.002) with a three-seed paired Δ. | **92.7:** *"At one seed, E2's per-cell signs resemble T4's (r 0.92 across cells; a pattern no other P2-recipe screen shows); unlike T4, about half of E2's gain (centred +0.0053 of +0.0102) is a per-cell offset. Read at three seeds."* Drop "larger", or state it as one seed against three. |

## Answers to the asks

**Ask 1 — the read is mechanical.**
- **One overstatement:** C1. Its specificity check supports the per-cell resemblance. The centred split is a real
  difference from T4.
- **Rule 8:** reporting the readout's values beside P2's without comparing them is right (packet 044 ask 1).

**Ask 2 — no reason not to spend. The spend stands.**
- **Rule 6 is the registered gate, and it passed.** A seed-0 cell count is not a registered stopping criterion, so
  stopping on it now would be an outcome-dependent rule written after the data.
- **4 of 6 is reachable:** U937 (−0.0034) is well inside one-seed per-cell noise, and T4 itself was 4 of 6.
- **Seeds 1–2 also decide whether E2 is accepted.** That sets E1's comparator in 92.3, so the 3.3 GPU-h is informative
  either way.

**Ask 3 — five things for the E1 s0 read.**
- **Rule 6 against P2 only.** Read E1 s0 against μ0 0.43693, as registered. The E1-against-E2 comparator binds only at
  acceptance, with three seeds of both. At seed 0, report E1 s0 − E2 s0 (seed 0 paired) as a descriptive number only.
  Don't drop or advance E1 on it.
- **VCAP (92.4):** use one per-cell convention throughout. The table here uses `score_dev` medians: T4 VCAP +0.0161. 91.11
  and F9 quote means over rows: T4b VCAP +0.0133. Either compute T4b's `score_dev` median, or tabulate means for E1, E2, T4
  and T4b alike. Then say whether E1's VCAP gain matches T4b's (the failed track removed) or E2's (all cell-specific
  chromatin removed). One seed, so it's descriptive.
- **The mount:** confirm the E1 log shows `mounted code verified for e1 (xpert_arm.py 60bdcd48…)`.
- **Rule 8:** the readout's input changes under the clean encoding too, so report it beside P2's, as for E2.
- **Centred against raw**, as in C1(b). It tells you whether E1's gain, if any, sits where T4's did (drug-specific) or
  where E2's partly does (per-cell offset).

## What I checked and found sound

- **The score JSON:** every number in 92.7's rule-6 block and table, against `v9_dev_score_E2_s0.json` and
  `v9_dev_score_T4_chromatin_ablated.json`. Per-cell Δ is the median over the cell's rows of (E2 s0 row r − P2's 3-seed mean
  row r), from `score_dev.py:97-104`.
- **The cross-screen check:** the per-cell Δ patterns of all 17 dev score JSONs (C1–C8, V1, V2, P6, T4, E2; V1 against its own baseline), behind
  C1(a).
- **The kernels:** the E2 seed-0 log's mount and guard lines; the s0 → s1 kernel diff, plus `xpert_arm.py:563`'s output tag;
  E1's kernel against dc00cf8.

## What I could not assess, and why

- **The new `lincs-v9-src` upload's bytes on Kaggle.** I relied on your `diff -rq` against the staged folder. E1's pin will
  refuse anything else.
