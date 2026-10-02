# REVIEW OF PACKET 043
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 58c6081

**The mechanical reading is right.**
- **The files:** both outputs in `external/kaggle_out/chromatin91/` match `CHROMATIN91_COMPLETE.json` and the committed
  copies (`a535781a`, `f54537eb`).
- **The reader:** I reran `read_chromatin91.py` and got the same verdicts:
  - **T1** does not advance, and its null is informative for the gain form;
  - **T2** does not advance;
  - **T3** is not interpreted.
- **The numbers:** every T1, calibration and M4 number in the packet matches the JSONs.
- **The calibration did its job:**
  - **P3 at 10 %:** raw Δ is +0.10, but centred Δ is only −0.0003 to +0.0002, so 0 passes. That's exactly what review
    041 C1's magnitude conjunct was for.
  - **No pass at π = 0** for T1 or T3.

There are four MINOR points about the reported sentences:
- C1: the headline's "informative" is broader than the result.
- C2: M1's room is inflated by shared control profiles.
- C3: two interpretive clauses go beyond what was tested.
- C4: T2's positive control "passes" with a negative mean.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | overreach | **"T1's null is informative" (the §91.12 title and the packet headline) holds for the gain form only.** P2's (drug-specific shift) MDE is 5 %, which by §91.9's rule is "reported with its MDE", not an informative negative. The body states both bounds correctly; the title doesn't. | Title: *"nothing advances; T1's null is informative for the gain form (MDE ≤ 0.5 %), drug-specific shifts bounded at 5 %"*. Use the same pairing wherever the negative is quoted. |
| 2 | MINOR | confound | **M1's "room" is inflated by shared control profiles, which no cell feature can predict.** **38 %** of M1(a)'s dose-neighbour pairs (13,067 of 34,298 train pairs) share an *identical* `X_ctl` profile. Since y = X − X_ctl, both residuals carry the same control noise, and those rows plausibly share a plate. I recomputed M1(a) with your split-pool code and reproduce **0.255** overall. Split by control: **0.410** on pairs sharing a control (n 11,262) against **0.167** on pairs with distinct controls (n 20,074). The cells with the highest agreement have the fewest distinct controls: NCIH596 has 2 control profiles for 56 rows (M1(a) 0.73), and NOMO1 9 for 159 (0.71). M1(b)'s within-cell neighbours are likely affected the same way. | Report M1 on distinct-control pairs and neighbours as the room (M1(a) ≈ 0.17), with the shared-control value beside it. Also note that "captures almost none" compares across scales: M1 is a residual–residual correlation, while FBC − FB is a change in total per-row Pearson. State the two numbers side by side rather than as a fraction. |
| 3 | MINOR | overreach | **Two clauses go beyond what was tested.** (a) The packet's D says *"chromatin's transferable content is gene-level, which gene identity already carries"*. But the closed-form models have no per-gene parameter, so that's an inference about v9, not a funnel finding. (b) §91.12's *"additive-only −0.0011 … as v9's additive head would"* contradicts T4: there, centred ≈ raw (+0.0056 against +0.0060), which places v9's chromatin cost in the drug-specific path, not in an additive offset (review 042, adopted in §91.11). | (a) *"gene-level content, which a per-gene parameter (e.g. v9's gene embedding) could represent; not tested here"*. (b) Drop *"as v9's additive head would"*. Worth adding, because it supports the gene-generic reading: the real FBC − FB (+0.00144) exceeds all five π = 0 draws (\|Δ\| ≤ 0.00025), so the increment is real, and FBC − N1 (+0.00023) places it in the gene-generic component. |
| 4 | MINOR | stats | **T2's positive control "passes" on the known rows by cell count, while its mean is negative.** On known rows, s_B − uniform is **−0.0011**: positive in 5 cells, but LNCAP −0.0109. On all rows it fails (−0.0019). The whole 11-neighbour retrieval family also scores below B0 over all 26 cells (s_B 0.125 against 0.142). So T2's null carries essentially no weight, beyond being uncalibrated. | Report the control as *"passes the cell count on known rows with a negative mean (−0.0011); fails on all rows"*, and describe T2's null as uninformative. |

## Answers to the asks

**Ask 1 — the readings are right. The reported sentences need C2–C4.**
- **The gene-generic reading:** FBC − N1 = +0.00023 (4 of 6 cells) and FBC − N2 = +0.00031. Together with C3's null-draw
  comparison, they support *"the increment is real and gene-generic; this cell's own chromatin adds ≈ nothing
  transferable in this form"*.
- **The M1 room is smaller than stated** (C2).

**Ask 2 — yes, the two bounds go together, and each takes its own label.**
- **The gain form:** MDE ≤ 0.5 % is a bound. It's the smallest π tested, and every draw passed there, with Δ +0.0055 to
  +0.0090 on known rows. The real Δ is +0.00144.
- **The drug-specific shift form:** MDE 5 %, "reported with its MDE".
- **Every sentence** keeps "of a form linear in the encoded tracks, on these dev cells".

**Ask 3 — the choice of experiment is yours. This is my assessment of what each would tell you.**
- **Arithmetic:**
  - one P2-recipe seed is ≈ 1.8 GPU-h;
  - rule 7 needs three seeds, ≈ 5.4 h per arm;
  - so ≈ 9 h covers one arm through rule 7, or both arms through rule 6 only.
- **Evidential value:**
  - **E2** (training-time ablation) is the on-distribution test of what T4 suggests, and it's informative whichever way
    it falls.
  - **E2 also sets E1's comparator.** If no chromatin is better than P2, E1 has to beat the no-chromatin arm, not P2.
  - **E1's upside is bounded by this result.** Cell-specific chromatin transfers ≈ +0.0002 in closed form, and T4b
    explains one of the three harmed cells.
- **Baseline:**
  - For arms on the P2 recipe, P2 (μ0 0.43693, s0 0.00169) is the frozen baseline, and rules 6–7 apply as written.
  - Against P6 you'd need its own threshold, fixed before the run: μ 0.45537, s 0.00227, three seeds exist.
- **Either way:** adoption is development after P7, with its own test registration.

## What I checked and found sound

- **The run:** the marker's output hashes, the committed copies, and the kernel log's completion.
- **T1:** per-cell Δ is HL60 +0.0020, LNCAP +0.0062 and SKBR3 +0.0006 positive; HEK293T −0.0008, U937 −0.0002 and VCAP
  −0.0001 negative. So 3 of 6, as read.
- **T2:** LOCO chose β = 0 and τ = ∞ (s_C), so both Δs are exactly 0.0000.
- **T3:** the prior is already r 0.42–0.73, and FB adds ≤ 0.001, so the control failed.
- **Coverage:** 969 level-3 dev rows; 807 drugs with ≥ 3 covered dev-train cells.

## What I could not assess, and why

- **A recomputation of the funnel from the raw data.** I verified the outputs by hash and reran the reader. The 232 s
  funnel itself I didn't rerun on the laptop, given the thermal limit, so I rely on the reviewed, pinned code. My M1
  split (C2) used the funnel's own functions on the dev-train rows.
