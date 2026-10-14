# PACKET 083 — DESIGN: the architecture and target audit (M1–M10), and Phase 1's confirmation tests C1–C4 (RESULTS §99; `model/V10_AUDIT_AND_PLAN.md`)
packet_id: 083
created: 2026-10-13
repo_commit: (this commit)
type: **DESIGN REVIEW** (findings A0–A3 measured on training rows only. No Phase 1 code exists. No test row has been read.)

## 0. Why

The principal asked for an audit of why v9 "doesn't really seem to use the drug input or the chromatin input". The
request was to find and confirm every mistake before planning fixes. I read the code of seven reference models
(DeepCE, CIGER, MultiDCP, PRnet, TranSiGen at Bai et al.'s pinned commits; chemCPA and STATE at HEAD) beside v9's own.
I then measured what the target is made of, on training rows.

## 1. Findings (`model/v9/audit99_target_sources.py`, output `model/results/audit99/target_sources.json`)

**Disclosed:** computed before this packet, as desk diagnostics on training rows (scratch, then this script, which
reproduces the scratch numbers). No model, prediction or test row was read.

**The plate proxy** is the control group, i.e. byte-identical `X_ctl` (the bundle has no plate key).

- **A0** (`split_cold_drug_1` training rows):
  - 85.5 % of rows share their exact control with another row;
  - 12,241 controls over 55,385 rows;
  - every (drug, cell) has ≥ 2 rows.
- **A1** (44,894 rows) — per-row r of y = X − X_ctl with:

  | predictor | r |
  |---|---|
  | the cell mean delta | 0.162 |
  | F0 | 0.399 |
  | **P = the mean y of other-drug rows on the same control** | **0.651** |
  | D = the mean y of the same (drug, cell) on other controls | 0.238 |
  | P + D − cell mean | 0.571 |

  The pattern holds in every ‖y‖ tercile.
- **A2** (31,722 rows; D matched on time):
  - P = 0.662;
  - P + w(D − cell mean), w ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}: best 0.675 at w = 0.2;
  - **so the increment is +0.013.**
- **The plate-contrast target** y_c = X − the mean X of the other-drug wells on the plate (≥ 2 of them; 26,892 rows):
  - the same (drug, cell, time) on other plates predicts it at r 0.147 (terciles 0.121 / 0.132 / 0.189);
  - its drug-free predictor (the cell mean of y_c) scores 0.012.
- **A3** (`split_cold_cell_1` training rows): −(X_ctl − the cell mean control) scores 0.362 alone; F0 scores 0.397.
- **Test-set identity counts** (plate membership only):
  - unseen compounds: 10,524 of 13,445 rows have ≥ 2 other-compound training wells on their plate; 8,878 have ≥ 2
    other-compound test wells;
  - unseen cells: 16,326 of 21,321 have ≥ 2 other-compound test wells; **0** share a plate with a training row.

## 2. The mistakes register (`V10_AUDIT_AND_PLAN.md` §4)

| # | mistake | status |
|---|---|---|
| M1 | the target is plate-dominated | CONFIRMED (A1, A2) |
| M2 | a plate shortcut through the exact control | CONFIRMED structurally (A0, R1's strata) |
| M3 | the drug-use tests (R1, R2/B1, O9, D) are read on M1's target, where the drug can add ≈ 0.013 | CONFIRMED (arithmetic) |
| M4 | the transductive `x_cell` for unseen cells (`xpert_arm.py:311`; review 001 C4, never rerun) | CODE FACT; C2 |
| M5 | the pathway readout runs before the drug enters (`model_v9.py` forward) | CODE FACT |
| M6 | chromatin cannot interact with the drug: a gene-token term and an additive head with no drug input | CODE FACT + T4 / §91 / §44 |
| M7 | dose is not tied to the drug: FiLM on the gene tokens before the drug; the drug token is RMS-normalised; softmax-only entry | CODE FACT; C4 |
| M8 | nothing routes the response through the drug (no adversary, no bottleneck; two full control stacks) | CODE FACT |
| M9 | the drug encoder is `Linear(2580 → 256)` over 1,581 training compounds; atoms are not contextualised | CODE FACT + §37 |
| M10 | stochastic depth is highest on the drug blocks | CODE FACT, minor |

## 3. Phase 1, registered (`V10_AUDIT_AND_PLAN.md` §5, binding as committed)

**The within-plate contrast:**
- **Definition:** c_i = v_i − mean(v_j) over the *scored* rows j on the same plate with a different compound, with ≥ 2 such
  j. It is applied to truth and prediction alike.
- **Scoring:** per-row Pearson. Zero-variance predictions are scored 0 and counted.
- **Machinery:** P9's per-molecule machinery on the clean subset, and §71.3's cell machinery.

- **C1 (decisive):**
  - **Primary:** v9 (P9, 3 seeds) against M+dt on the contrast, on unseen-compound rows. The verdict is `score_p9.verdict`,
    with three statements fixed.
  - **Reported:**
    - per seed;
    - v9 against 0;
    - M, which is 0 by construction;
    - the magnitude ratio;
    - C1b, the oracle-plate version (truth − P_meas, prediction − P_meas; 10,650 rows);
    - C1c, the zero-parameter training-plate predictor on the original target.
- **C2 (unseen cells, M4):**
  - **C2a:** F_tr, the transductive term, against F_ind, its inductive version, beside v9 (P7) and XPert (O2), per cell.
  - **C2b:** P7's final-snapshot checkpoints, intact against `x_cell` = the global training-row mean control. It reports
    the per-cell Δ and redoes §71.3 against XPert. **Fixed rule:** if the replaced cluster mean is below half of the
    intact final-snapshot one, then *"the lead is substantially carried by the transductive input"*.
  - **C2c:** the contrast, v9 against XPert, reported.
- **C3:** P9's final-snapshot checkpoints, intact against the drug inputs swapped (a derangement within cell, seed 0). It
  reports Δr on the original target and on the contrast, and the sensitivity. The reading uses the contrast Δ per
  molecule, with a CI excluding 0.
- **C4:** per (compound, cell, time) with ≥ 3 doses, the Spearman of dose against ‖y − P_meas‖ and against ‖ŷ − P_meas‖,
  for v9 and M+dt. Reported.

## 4. The Saturday change

B1 is held: its registered reading is on M1's target, and C1 answers its question for free. O9 session 3 and D's
timing run go ahead.

## ASKS
1. **M1's measurement.** Is the plate proxy sound? Is there any route by which A2 *understates* the drug's increment, for
   example:
   - D averages other doses;
   - w is one global weight;
   - D carries its own plates' noise.

   If so, what bound would you accept instead?
2. **C1:**
   - Is the within-plate contrast the right estimand for "uses the compound"?
   - Is M+dt (dose and time known) the right comparator, rather than M?
   - Are 8,878 rows, before the clean filter, enough? The power is not computed.
   - Should rows whose plate-mates are mostly the *same* scaffold be stratified?
3. **C2's half-of-lead rule:** is it well-posed? The comparison is final-snapshot against final-snapshot, while P7's
   record is the three-snapshot mean.
4. **C3:**
   - Is a within-cell derangement the right swap?
   - Should the swap be restricted to compounds of a different MoA, to avoid near-duplicates?
5. **Holding B1:** do you agree, or should it run for R2's registered reading anyway?
6. **Phase 2's F1:** do you prefer the within-plate contrast or CMap Level-5 MODZ as v10's target? Anything missing from
   M1–M10?
