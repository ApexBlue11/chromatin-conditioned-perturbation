# PACKET 085 — RESULT: Phase 1's local tests, each read once (RESULTS §99.6; outputs `model/results/audit99/phase1_*.json`)
packet_id: 085
created: 2026-10-15
repo_commit: 8d9b62e
type: **RESULT REVIEW** (C1, C1b, C1c, C2a, C2c, C4 read once, after 99.5's fixes; C2b and C3 are running as 6 CPU kernels)

## Readings
- **C1 (of record, a benchmark statement per §8):**
  - v9 − M+dt on the within-plate contrast is **+0.0756 [+0.0616, +0.0901]** per molecule, with 263 of 337 molecules
    favouring v9 (sign p 6e-26);
  - `verdict`: v9 better;
  - statement: *"v9 beats a drug-free model with dose and time on the within-plate contrast (a benchmark statement; whether
    v9 uses the compound is C3)"*.
- **C1, reported:**
  - row means: v9 0.109, M+dt 0.032, M 0, the dose floor 0.023;
  - **M+dt − floor = +0.003, no claim;** v9 − floor = +0.079;
  - per seed: +0.075 / +0.075 / +0.077;
  - magnitude ratio: v9 0.13, M+dt 0.03;
  - Tanimoto strata: +0.081 (below the median) / +0.069 (at or above).
- **C1b:** v9 0.163, M+dt 0.104, M 0.088; v9 − M+dt +0.047 [+0.038, +0.057].
- **C1c:** on the original target, the zero-parameter training-plate lookup scores 0.662, against v9 0.654 and M 0.661.
  v9 − lookup is −0.018 [−0.022, −0.015] per molecule.
- **C2a:** F_tr 0.336 (cluster 0.277), F_ind 0.224, v9 0.481, XPert 0.387.
- **C2c:** v9 0.160, XPert 0.146; cluster +0.009 [−0.0002, +0.018], 6 of 8 cells.
- **C4:** mean ρ(dose, response size) is 0.47 for the truth, 0.36 for v9 and −0.02 for M+dt; corr(ρ_truth, ρ_v9) 0.25.

## Interpretation, as recorded in §99.6
- M3 is answered at the benchmark level: R1's tie hid a +0.076 per-molecule advantage once the plate is removed.
- M1 / M2 are confirmed on test rows: a plate lookup beats every model on the field's target.
- "Uses the compound" waits for C3.
- M7 is weaker than stated (C4).

## ASKS
1. Is §99.6's wording within what each test licenses? In particular, is "R1's tie hid a real advantage" within C1's
   benchmark-level statement?
2. **C1c:** is "the field's cold-compound benchmark is won by a plate lookup" right as stated? The scope is the 9,466 rows that
   have training plate-mates, 79 % of the clean rows.
3. **C2a:** H1975's F_tr is exactly 0 (one shared control), and that pulls F_tr's cluster mean down. Any objection to
   reporting it as is?
4. Anything to add before C2b and C3 are read?
