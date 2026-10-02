# PACKET 043 — RESULT: the chromatin funnel read once (RESULTS §91.12); proposed next steps
packet_id: 043
created: 2026-10-03
repo_commit: 58c6081
type: **RESULT** of the run you cleared (review 042). CPU only. The next steps proposed in E each need their own packet.

## A. The run
- `kern_chromatin91`, version 1, on a Kaggle CPU: inputs verified (9 pins + the split bundle); funnel 232 s; calibration 2,258 s.
- **Marker:** `CHROMATIN91_COMPLETE.json` carries the outputs' sha1s.
- **Read once** with `read_chromatin91.py` on the 3,074 drug-known rows of record.
- **Outputs, committed:** `model/results/chromatin_funnel_91.json`, `chromatin_power_91.json` and the marker.

## B. Mechanical verdicts
| test | rule | positive control | calibration | verdict |
|---|---|---|---|---|
| T1 | Δ +0.00144 (top +0.00255) against 0.004 / 0.008; centred +0.00101 against 0.002; cells 3/6, centred 3/6, vs N1 4/6 | passes (FB − B0 +0.00262, 6/6) | P1 gain: 5/5 passes at every π, so MDE **≤ 0.5 %**. P2 drug-specific shift: MDE 5 %. P3: never passes. π = 0: 0 passes | **does not advance; informative null** |
| T2 | Δ_T2 0.0000 (LOCO chose β = 0); s_C − uniform 0.0000 (LOCO chose τ = ∞) | passes on known rows (fails on all rows) | not calibrated | does not advance |
| T3 | ρ_T3 ≤ 0.0016 in every cell | **fails**: FB adds nothing over the gene prior, which scores r 0.42–0.73 | P4 5 %; π = 0: 0 | not interpreted |

The conclusions are the same on all 4,043 rows. T1's ρ is 0.238 in every draw.

## C. Reported, not read (as written in §91.12)
- **M4, information and redundancy:**
  - FC − B0 = +0.00208 (5/6 cells), about 80 % of basal's +0.00262.
  - FBC − FB = +0.00144.
  - **FBC − N1 ≈ +0.0002, and FBC − N2 = +0.0003.** Nearly all of chromatin's increment is gene-generic, which I read as
    "this cell's own chromatin adds ≈ nothing transferable".
- **Variants** (minus FB): FBC⊥ +0.0013; global rule +0.0011; **additive-only −0.0011**; epigenetic drugs (39 rows) +0.0005.
- **Encoding:** v9's encoding gives FBC − FB = +0.0003, against +0.0014 with the primary encoding.
- **M1 (split pools):** dose-neighbour agreement 0.255; within-cell drug-neighbour predictability 0.282. The cell-specific residual is
  structured, but linear promoter chromatin captures almost none of it.
- **Caveat:** the closed-form base is weak. B0 scores 0.142 against v9's 0.437, since it lacks the row's own control profile, so
  increments need not transfer to v9.

## D. Where this leaves the principal's question ("make chromatin matter"; "are we measuring it properly")
- **Measurement:** the instrument sees a gain-type chromatin effect at ≤ 0.5 % of the cell-specific residual, so its null is
  informative. The earlier nulls were not, because one training run per arm could not detect effects below ≈ 0.006–0.016 (§91.10).
- **Information:** in this form (promoter marks, 978 landmarks, linear, gene-local), chromatin's transferable content is gene-level,
  which gene identity already carries. Its cell-specific content transfers ≈ nothing.
- **Use:** trained v9 reads chromatin, and on dev cells reading it costs +0.006 (T4); VCAP's cost is its failed track (T4b).

## E. Proposed next steps (each to be pre-registered in its own packet; GPU only after review)
1. **S12 encoding fix in v9:** failed tracks missing, rank-normal transform, `r` from mask × reliability. Run as a dev screen on the
   P2 recipe (rule 6, one seed, ≈ 1.8 GPU-h), or on the P6 stack.
2. **A training-time chromatin ablation on the dev carve** (`--ablate_epi`, P2 recipe), one seed then rule 7: does v9 do better
   without chromatin at all?
3. Richer features (S9: enhancers, TF motifs) only as a data project, and not this week.

This week's quota after P7, t1 and t2 is ≈ 9 GPU-h.

## ASKS
1. Is the mechanical reading right, and is any "reported" sentence wrong? In particular:
   - the gene-generic reading from FBC − N1 ≈ +0.0002 and FBC − N2 +0.0003;
   - the reading of the M1 room.
2. Is the informative-negative wording right with MDE ≤ 0.5 % (the smallest π tested)? Should the P2 bound (5 %) sit beside it in
   every sentence?
3. Which of E1 / E2 is worth the ≈ 9 h left this week, and against which baseline (P2's μ0 or P6)?
