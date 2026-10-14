# REVIEW OF PACKET 083 (the architecture and target audit, M1–M10; Phase 1's C1–C4; holding B1)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 6eb7197

**The measurement is right, and my own code reproduces it.** On `split_cold_drug_1` training rows:
- the plate predictor P gives **0.662**;
- adding the other-plate drug term reaches **0.675**, an increment of **+0.013**;
- the contrast target's other-plate predictor gives **0.147**.

The target is plate-dominated (M1), and the shared control is a plate shortcut (M2). The code facts M4–M10 are as stated.

**Two conclusions go further than the numbers.**
- **(1) The ceiling is against an oracle.** "The most the drug can add is ≈ +0.013" is the increment over an **oracle** plate
  predictor, built from the plate-mates' *true* responses.
  - Against a drug-free predictor a model can actually have, F0, the same drug term adds **+0.070** on rows with plate-mates and
    **+0.152** on rows with none.
  - So M3's "CONFIRMED (arithmetic)", R1's new wording and the case for holding B1 all rest on the oracle baseline (C1).
- **(2) C1's pass sentence attributes to the compound what dose can partly earn** (C2). Within a plate, 61 % of plate-mates sit at
  another dose. A compound-blind, dose-aware predictor scores **0.027** on the contrast, so "beats M+dt" isn't "uses the
  compound".

**One factual label is wrong** (C3). The bundle has exactly one row per (compound, cell, time, dose), which §91.9 already records.
So the "compound's own replicate" at 0.147 is the same compound **at other doses** on other plates. No dose-matched comparison
exists.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | inference (M3, §99.1–99.2, the B1 hold) | **"The most the drug can add on this target is ≈ +0.013" holds only against the oracle P.**<br>**What P is:** the mean measured y of the other compounds' wells on the row's plate. No model has that for an unseen compound's plate. It can only approximate it through recall of the exact control from training wells on the same plate (M2): 78 % of test rows, by your count.<br>**The same drug term over a baseline a model can have** (training rows, mine):<br>• over F0: **+0.070** on the 31,722 rows with ≥ 2 plate-mates (0.404 → 0.474);<br>• **+0.152** on the 9,125 rows with **no** plate-mate (0.323 → 0.476).<br>**So:**<br>• the room for compound information depends on how much of the plate a model can recover;<br>• it is largest where the plate is unknown, which is the distinct-control stratum, where v9 led M by +0.044 (98.11);<br>• M3 shows the drug is worth little *once the plate is known*. It doesn't show the R1 / R2 / O9 / D tests can't detect drug use. | **Reword three places:**<br>• **M1 / M3 and §99.1:** *"over a plate predictor built from the plate-mates' measured responses, the compound's other-dose response adds +0.013. Over a drug-free predictor without plate information it adds +0.07 (rows with plate-mates) to +0.15 (rows without)."*<br>• **R1's interim wording (§99.2):** drop *"the drug can add at most ≈ 0.013"*. Keep R1 as measured, with M2's shared-control caveat.<br>• **B1:** re-state the hold as a quota and priority decision, not as "its target can't show it" (see Ask 5). |
| 2 | MAJOR | estimand / attribution (C1) | **C1's "v9 better" sentence, "v9 uses the compound", is earned partly by dose modelling.**<br>• Within a plate, rows differ in compound **and** dose: on average 61 % of a row's other-compound plate-mates are at a different dose.<br>• **A dose-only floor** (training rows; leave-own-control-out mean y at the same (cell, time, dose), contrasted the same way) scores **0.027** on the contrast. For comparison, the compound's other-dose response scores 0.147 on the same rows.<br>• M+dt adds only +0.0003 per molecule over M on the original target (98.13), so it may capture little of that dose structure.<br>• So v9 > M+dt on the contrast can be better dose modelling, with no compound use at all. This is 073 C1's confound again, at smaller scale.<br>• **The compound-specific test is C3:** the derangement keeps dose, time, control and cell fixed, and swaps only the compound. | **Choose one, before any code:**<br>(a) make C1's statements benchmark-level (*"v9 beats a drug-free model with dose and time on the within-plate contrast"*), and route "uses the compound" through **C3**; or<br>(b) keep C1's sentence, but require v9 to beat **both** M+dt and the zero-parameter dose-only floor (registered with its definition), or B1 on the contrast.<br>**Either way:** report the dose-only floor beside C1. |
| 3 | MINOR | factual label (§99.1, the commit, the plan, F1's ceiling) | **No replicates exist in this bundle:** 68,830 rows equal 68,830 distinct (compound, cell, time, dose), as §91.9 already says. So "the compound's own replicate on other plates" (0.147) is the compound **at other doses**, and so is A2's D.<br>**Dose-matching isn't possible, and wouldn't help anyway.** The nearest-dose version adds less: +0.008 over P, and +0.002 when within 0.3 log10. The averaged term's contrast r doesn't rise cleanly with the number averaged: 0.132 / 0.170 / 0.151 / 0.139 for k = 1 / 2 / 3–4 / 5–7.<br>**So 0.147 is not a noise ceiling.** F1 calls it one. | Relabel it everywhere *"the same compound at other doses on other plates"*. Drop "noise ceiling" from F1: a replicate ceiling needs replicate data (for example, the GSE Level 3 / 4 replicates), not this bundle. |
| 4 | MINOR | reading wording (C1, C3) | **The "no claim" statements read as stronger negatives than the CI allows.** *"Not detected even where only the compound can help"* (C1) and *"not shown"* (C3) carry no bound. Review 075 C4's lesson applies. | Give both no-claim sentences the bound: *"… not detected: its contrast margin is at most {ci_hi:+.3f} per molecule (95 % cluster CI)."* |
| 5 | MINOR | C2b attribution | **The registered sentence's attribution:**<br>• Replacing `x_cell` by the **global** training-mean control removes *all* cell information, not just the transduction. A drop could therefore come from losing cell identity, which an inductive per-row input would keep (each row's own X_ctl is a legitimate inductive input). So the sentence ("substantially carried by the *transductive* cell-average input") may misattribute.<br>• **"Half of the intact lead" is undefined** if the intact final-snapshot lead is ≈ 0 or negative. | **The replacement:** use the per-row control (`x_cell := X_ctl`) for the registered reading, and report the global mean beside it. Or keep the global mean and reword the sentence to *"… carried by the cell-level baseline input"*.<br>**A precondition:** the rule applies only if the intact final-snapshot cluster lead is > 0 with a cluster CI excluding 0. Otherwise it is "not assessable". |
| 6 | MINOR | C3 swap | **A random within-cell derangement can pair close analogues,** which understates sensitivity. MoA labels are incomplete, so they're a weaker filter than structure. | Make the derangement compound-level (one substitute per compound, consistent across its rows), constrained to ECFP4 Tanimoto < 0.4 (or a different Murcko scaffold). Report the unconstrained derangement beside it. |

## Answers to the asks

**Ask 1 — the proxy is sound enough, and A2's increment is about the right size, but against the wrong baseline for the
claim** (C1, C3).
- **The proxy:**
  - no control group spans two cells;
  - only 12 of 12,241 span two times.
  - So "byte-identical control" marks rows measured together: a plate, or a batch if the source pooled replicate plates. Either
    way it is the unit of both the plate effect and the shortcut.
- **The routes you named don't inflate the increment, and dose-matching lowers it** (my variants):
  - **Dose:** the nearest-dose D adds +0.008 over P, against the all-dose mean's +0.013.
  - **Plate noise:** the plate-contrast replicate Dc adds +0.011.
  - **The global w:** a per-tercile w gives a similar figure.
  - **Strata:** across the number-averaged strata, +0.009 to +0.018.
- **The bound I'd accept** is an increment stated against both baselines: the oracle P (+0.013), and a model-available drug-free
  predictor such as F0 (+0.07, or +0.15 without plate-mates).

**Ask 2 — the contrast is a good estimand, and power is adequate. The comparator needs C2's fix.**
- **The estimand:** it removes the control and the plate from truth and prediction alike. A compound-blind dose model gets only
  0.027 on it, against 0.147 for compound information.
- **The comparator:** M+dt is right for a benchmark statement, but not for "uses the compound" (C2).
- **Power,** from a training-row analogue with my own per-compound medians:
  - with 300 compounds, the paired 95 % cluster-CI width is about **0.036** (0.051 at 150);
  - so a margin of about ≥ 0.02 is detectable;
  - the 0.10 UNINFORMATIVE rule will never trigger at this scale, which is why C4's bound matters.
- **Scaffold stratification:** worth reporting, not reading. Use the mean Tanimoto of each row's plate-mates (above / below
  median). Plates of analogues shrink every model's contrast.

**Ask 3 — internally consistent, with C5's precondition and attribution.**
- Final snapshot against final snapshot is the right like-for-like basis.
- The record should say "final-snapshot basis" beside P7's three-snapshot headline.

**Ask 4 — a derangement within cell is right** (control, dose and time fixed, so only the compound moves). Add C6's analogue
constraint.
- **Prefer structure (Tanimoto, scaffold) over MoA.** Its coverage is complete.
- **C3 is the cleanest compound-use test in Phase 1** (C2).

**Ask 5 — the stated reason for holding B1 doesn't hold** (C1 and C2).
- **"C1 asks the same question for free" isn't right.** C1's comparator differs from v9 in dose modelling and architecture.
- **The free test of compound dependence is C3.**
- **B1 answers a different, matched question:** does training with the compound help, all else equal?
- **Its registered R2 reading is not hopeless on this target:**
  - the drug's room over model-available baselines is large where the plate is unknown;
  - 98.12(a) already reports R2 by control stratum.
- **Running B1 or not is a quota and priority call, which is yours.** If it is held, record it as *deferred*, not superseded.
  Note that B1 scored on the contrast would also be the exact C1 comparator.

**Ask 6 — I don't choose the target. Here are the trade-offs and the gaps I see.**
- **The contrast target:**
  - in-bundle, and comparable across all the saved predictions;
  - but relational: its reference set is whatever compounds share the plate, a property of the design, not of the compound;
  - in this sparse bundle, only 26,892 of 55,385 training rows have ≥ 2 other-compound plate-mates, so about half the rows carry
    no contrast loss.
- **Level-5 MODZ:**
  - a field-standard signature: robust z against the full plate population, consolidated over replicates;
  - addresses noise as well as the plate;
  - but needs new data and splits, and new runs for every baseline, so today's head-to-heads lose comparability.
- **Missing from M1–M10:**
  - (i) **No replicates in the bundle** (C3), so no noise ceiling is available here.
  - (ii) **The metric:** per-row Pearson over 978 genes rewards the shared direction (the plate, the generic response). That is
    distinct from the target, and it is why P scores so high.
  - (iii) **Dose structure inside the plate** (C2), which any plate-normalised target must account for in its comparators.

## What I checked and found sound

- **The audit script in full,** and its JSON. A0 is reproduced (12,241 controls; 85.5 % sharing), and so are P and the increment
  (0.662 → 0.675) and the contrast figure (0.147), with my own code.
- **My variants, all on training rows only:**
  - the plate-proxy structure (cells and times per control group);
  - the uniqueness of (compound, cell, time, dose);
  - nearest-dose D and Dc;
  - increments by the number averaged and by dose gap;
  - the dose-only floor on the contrast;
  - increments over F0, with and without plate-mates;
  - a power analogue for C1.
- **The plan's §5–§7, and the RESULTS §99 text.**
- **M4's code:** `xpert_arm.py` builds the cell mean from training rows. An unseen cell falls back to all of its own rows, its
  test controls.

## What I could not assess, and why

- **Any test-row quantity.** None was read, as registered, so C1–C4's own values are unknown.
- **What XPert's preprocessing makes X_ctl**: one plate's DMSO mean, or a pooled batch. The bundle has no plate key. It doesn't
  change any conclusion above.
- **M5–M10's mechanistic claims** beyond the code facts cited. They are Phase 2's subject, each to be registered and tested.
