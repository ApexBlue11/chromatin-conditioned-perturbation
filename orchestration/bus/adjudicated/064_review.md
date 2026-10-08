# REVIEW OF PACKET 064
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 2f37a36 (packet 6a02e42)

**95.6's reading is mechanical, and I reproduced it.**
- **Provenance:** `stage_a_prime.json` = 723235d5, matching its marker.
- **The rerun:** `--read` gives output **identical** to `stage_a_prime_reading.txt`.
- **The numbers:** every A1, A1_noncns and A3 number in 95.6 matches the JSON. A549's p is 10/1001 = 0.00999 < 0.01, so it
  counts, by the rule.
- **The code change after clearance (97c8616 → f7c58f51)** is exactly review 063a's MINOR: two lines enforcing the 2-class A3
  rule on cold-drug, plus a test.

**The scoping is licensed.** "Every Δ is positive; the misses are on p", "retrieval not established" and "thin substrate" are the
right words. One fact belongs beside the gate (C1).

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | disclosure | **The gate opened through the DNA row added in 95.5.**<br>- **The DNA units carry A3:** +5.3, +5.9 and +6.4 against EGFR's mean +1.15.<br>- **Without the DNA row, A3 wouldn't carry signal:** the ungated table, which is EGFR-only, gives T 1.15, p 0.012, and fails p < 0.01.<br>- **What that rests on:** the same 4 DNA-damaging compounds in 3 TP53-wild-type cells, showing a generic DNA-damage → p53 response. That's robust biology, and also the kind of response chemistry-only references may reproduce easily if those compounds have close structural relatives in training. | Add beside the gate: *"A3's signal is carried by the DNA → p53 row (4 compounds × 3 TP53-wild-type cells; added before data in 95.5). EGFR alone gives T 1.15, p 0.012."* |

## Answers to the asks

**Ask 1 — yes, mechanical, and the scoping is licensed (with C1).** Retrieval is "not established", not "absent": Δ is +0.069
to +0.120 in all six cells, between 1.5 and 2.8 null sd, with 16–53 compounds. The commit message's "no retrieval signal" is
looser than 95.6's text. Keep 95.6's wording.

**Ask 2 — my assessment: register P9 on accuracy grounds, with Stage B′ as a secondary, pre-scoped readout.** The mechanism
case is real but narrow:
- **What a Stage B′ pass could license:** at most *"for DNA-damaging and EGFR-inhibiting unseen compounds, v9's predictions show
  the expected pathway direction more strongly than chemistry-only references"*. That covers 2 classes and about 8 compounds.
- **What a fail would mean:** little, given the power (below).
- **The accuracy case:** the cold-drug head-to-head stands on its own.

Registering P9 as accuracy-primary avoids implying that the mechanism gate justifies a GPU run whose mechanism readout can
license only that two-class sentence. Which is primary is your call. The registration should state the B′ scope either way.

**Ask 3 — worth about 11 GPU-h only if the paper makes a cold-drug performance claim.** If it does, an unseen-compound number
against SOTA is the natural reviewer question.
- **The O2 safeguards to reuse:**
  - **termination:** §84.1, since XPert's stopper monitors test loss. Push sessions without reading the logged loss, and use the
    same stopping handling.
  - **hardening:** §84.2.
- **Fixed before launch:**
  - a **cold-drug estimand** analogous to §71.2, where the held-out unit is the compound, so per-compound or per-row is fixed
    then;
  - a §71.4-style reproduction check on XPert's own number for that split, if one is published;
  - the same split bundle and metric code as P9.
- **Admissibility:** I haven't re-read §71/§84 in full for this. Admissibility should be argued item by item in the comparator's
  own registration.

**Ask 4 — Stage B′'s A3-only reading. Five requirements, fixed in its registration:**
1. **B3 on v9's predictions by A3's rule:** p < 0.01, ≥ 2/3 of units, units spanning ≥ 2 classes. Seed-mean **and** every seed.
2. **"Beyond chemistry", per reference R** (ridge, 1-NN, 5-NN, physicochemical; the structure-only floor applies to retrieval,
   not A3):
   - v9's T exceeds R's, with a **paired swap null at the (compound, cell) level** (swap v9's and R's predicted signatures per
     compound-cell, recompute T), one-sided p < 0.05;
   - the per-class direction is the same in both classes, as a **descriptive** consistency check.
   - **Why not unit-level swaps:** the DNA class has 3 units, so only 2³ = 8 assignments, and its minimum p is 0.125. It can't
     reach 0.05. At the compound-cell level, DNA has 12 elements and EGFR about 19.
3. **B3c reported only.** Nine units in 2 classes leave about 7 df after centring, so it's powerless.
4. **Before the run, list the about 8 member compounds** with each one's max Tanimoto to training and its nearest training
   analogue (identity level). If a member has a near-duplicate in training (for example, max similarity > 0.8), say so. A "beyond
   chemistry" pass carried by it would be structural.
5. **The licensed sentence is fixed then:** *"for DNA-damaging and EGFR-inhibiting compounds unseen in training, …"*. Nothing
   wider.

## What I checked and found sound

- **Stage A′:** output and marker sha1s; `--read` reproduced; A1, A1_noncns, A3 and per-unit d against the JSON; per-cell
  member-mean AUROCs (for example, DNA inhibitor 0.81 in MCF7 and 0.45–0.61 elsewhere; EGFR 0.80 in MCF7).
- **The code:** the post-clearance diff (two lines plus a test).
- **The RESULTS text:** 95.6.

## What I could not assess, and why

- **The structural relationships between the about 8 member compounds and the training compounds.** That needs RDKit, which
  isn't installed, so I made it requirement 4 for the PI before Stage B′.
- **§71 and §84 in full.** I read only their headings and §84.1's gist for Ask 3.
