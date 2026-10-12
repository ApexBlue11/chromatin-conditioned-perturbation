# REVIEW OF PACKET 077 (R1: v9 against Bai et al.'s drug-free MLP, 98.11)
verdict: SOUND
reviewed_commit: 61ddbde (scorer d9843dcc at 23d1f75)

**R1 is mechanical and right, and my own code reproduces it.** I scored it from the downloaded M files. The six sha1s match
`P10_M_COMPLETE.json`, and each record carries the pinned input sha1s and `n_test` 13,445.

| | packet | my rescore (own code, own bootstrap seed) |
|---|---|---|
| clean, per molecule | −0.0038 [−0.0086, +0.0016] | −0.0038 [−0.0087, +0.0015] |
| molecules favouring v9 | 96 / 346 | 96 / 346 (sign p 6e-17, toward M) |
| full | +0.0045 [−0.0016, +0.0112] | +0.0045 [−0.0016, +0.0112] |
| row-pooled (clean) | v9 0.642, M 0.635 | 0.6417 / 0.6354 |
| distinct-control stratum | +0.044 [0.033, 0.056], 208 / 267 | +0.0444 [0.0331, 0.0558], 208 / 267; my own hashing gives the same 2,091 rows, `def96fc0` |

**Verdict:** NO COMPOUND-LEVEL CLAIM. With mean ≤ 0, the reading is "does not beat".

There are three MINOR points below, about wording and the precision of the post hoc description, plus one suggestion within the
freeze.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | wording (98.11, §5.5) | **"Not a margin over the strongest simple baseline" ranks baselines in a way R1 doesn't.** R1 compares v9 with one baseline. M is "simple" in its input (the control only) but is a 3 × 2,048 MLP, so not simple in capacity. Untested baselines could be stronger. What R1 licenses is that v9's margin over ridge is not a margin over this drug-free model. | *"So the +0.119 over ridge is not a margin over a drug-free baseline: Bai et al.'s control-only MLP matches v9 per molecule."* |
| 2 | MINOR | precision of the reported pattern | **"M's parity with v9 sits on the rows whose X_ctl also occurs in training" understates it.** On those rows **M is ahead**: clean shared-control rows give −0.0111 [−0.0151, −0.0068], with only 64 of 345 molecules favouring v9. On distinct-control rows v9 is ahead (+0.044). The aggregate parity is the mix of the two. | *"On clean rows whose control also occurs in training, M is ahead (−0.011 [−0.015, −0.007]); on rows with a new control, v9 is ahead (+0.044). The aggregate parity mixes the two."* Reported, never read. |
| 3 | MINOR | bounds of the post hoc reading | **The post hoc reading is within bounds** as labelled (untested, non-randomised stratum). One check narrows it and one alternative stays open.<br>• **Checked:** it isn't cell composition. The split replicates **within each of the six largest cells**: v9 − M on distinct rows +0.039 to +0.068, on shared rows −0.003 to −0.013 (MCF7, PC3, A549, A375, HA1E, HT29).<br>• **Still open:** within-cell **plate or time composition**. New controls may come from particular plates or batches where v9's other inputs help. | Add the within-cell replication and the remaining plate/time alternative to the post hoc note. On the confirmatory design (Ask 2), see below. |

## Answers to the asks

**Ask 1 — yes, mechanical and correctly worded,** with C1.
- **"Does not beat", not "M beats v9":** right. The cluster CI includes 0, and `verdict` needs the CI to exclude it, whatever the
  sign test's 250 / 346.
- **"The row-pooled sign is v9's":** correct, and "neither summary licenses a winner" is the right gloss. The two summaries weight
  molecules differently.
- **"Not a margin over the strongest simple baseline":** licensed only in C1's narrower form.

**Ask 2 — within bounds** (C2's precision, C3's additions).
- **On the confirmatory design:** another cold-drug fold with distinct-control rows as the unit replicates the pattern, but the
  stratum is non-randomised in every fold, so it can't separate "M recalls plate responses" from plate/time composition.
- **Two designs that would separate them:**
  - **(i) A control-substitution M:** retrain M with each row's `X_ctl` replaced by another control of the same cell, which
    breaks the exact-control link. Its shared-row advantage should vanish if the recall mechanism is right. It's CPU only.
  - **(ii) A plate-disjoint split:** test controls never seen in training, by construction.
- Either needs its own registration. This lists what would settle it, not a choice.

**Ask 3 — yes.** y′ = X − x̄_ctl(c) removes the input control's own noise, but keeps X's plate component, which X_ctl shares. M's
recall of plate-level responses still pays on it. So "M better on the noise-free target" doesn't separate the shared-plate
route from compound use.

**Ask 4 — within the record, with C1's wording.** The bullet keeps:
- "does not beat" (not "loses");
- the CI and the count;
- the deferral of compound use to B1;
- the distinct-control result as reported and unconfirmed.

**Ask 5 — nothing changes R2's registered reading. Two reported-only additions** would make R2 more interpretable without
touching the freeze.
- **(a) R2 by control stratum.** `score_p10 --ref v9_drugblind` already writes the distinct-control block. Place v9 − B1 per
  stratum beside v9 − M, and add a post hoc **B1 − M** per stratum, labelled post hoc. If B1 also leads M on distinct-control
  rows, that lead is not the compound.
- **(b) A positive control on the duplicates** (post hoc, labelled). On the 38 duplicate molecules (full split), v9 − M is
  **+0.099** row-pooled (v9 0.690, M 0.592).
  - **Why it's informative:** v9 has seen those molecules in training under another identifier. So v9 − B1 on them shows
    whether v9's compound pathway carries information when the compound is known.
  - **How it reads:** if v9 ≈ B1 even there, v9 isn't using the compound at all. If v9 ≫ B1 there but ≈ B1 on the clean
    molecules, v9 uses compound identity but doesn't generalise it.
  - **The caveat:** composition (which compounds are duplicates) applies.

## What I checked and found sound

- **My own R1 scorer:** P10_M_COMPLETE sha1s; row alignment to P9's 13,364 (13,445 per seed file); the per-seed Pearson mean;
  the per-molecule medians; the bootstrap; the sign test.
- **The strata:** the distinct-control stratum from my own control hashing (2,091 / 267 / `def96fc0`); the shared-control
  complement; the per-cell split within the six largest cells; the duplicate-molecule contrast.
- **The records:** the three seeds' records (pins, best / last epochs, `n_test`); the `raw_s*` contents (logs, npz, record,
  `resource_log`, `best_model.pt`; no test truth).
- **The text:** 98.11, and the §5.5 diff.

## What I could not assess, and why

- **The F0, noise-free and cell-centred blocks** beyond their consistency with my earlier measurements: F0 0.391 equals my
  073 figure. I didn't rerun those three blocks.
- **Plate identities.** The bundle carries no plate key that I know of, so C3's plate/time alternative can't be checked from
  these files.
