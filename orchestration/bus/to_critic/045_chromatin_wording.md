# PACKET 045 — WORDING: the chromatin results in the manuscript, a RESULTS 91.12 addendum (score units), and Figure 9
packet_id: 045
created: 2026-10-03
repo_commit: 30397f5
type: **WORDING + FIGURE REVIEW** (no GPU, no new data; every number is from committed JSONs)

## A. Operational note (no decision needed)
The detached 05:40 push loop died silently overnight (its log stops at "waiting"; no process left). **P7 and §88 t1 were pushed
by hand at 08:44 IST**; each kernel was first checked byte-identical to its cleared commit (P7 1cd614c, t1 dc00cf8). Both are
RUNNING. The E2 / t2 follow-on chain was re-armed unchanged. P7 now completes ≈ 15:15 IST, so E2 s0 and E1 s0 move ≈ 3 h later.

## B. RESULTS 91.12 addendum: the calibration in score units (PI read of `chromatin_power_91.json`; nothing re-run)
The MDEs in 91.12 are fractions of the cell-specific residual. In T1's own score (drug-known rows, 5 draws):

| planted | T1 Δ | passed |
|---|---|---|
| nothing | −0.00023 to +0.00025 | 0 / 5 |
| P1 gain 0.5 % (the smallest planted) | +0.0055 to +0.0090 | 5 / 5 |
| P2 shift 2 % | +0.0022 to +0.0034 | 0 / 5 |
| P2 shift 5 % | +0.0050 to +0.0069 | 5 / 5 |

- **What the PI draws from it:** the funnel sees effects of ≈ +0.005, set by its bar (+0.004), with ≤ 0.0003 of noise. That is
  below the training ablation's ≈ 0.006–0.016 and comparable to C7's 0.0042, **not an order of magnitude finer**.
- **Two corrections it forces:**
  - "MDE ≤ 0.5 %" is a ceiling: 0.5 % was the smallest size planted.
  - §91.10's "+0.002 to +0.01" target range is covered only from ≈ +0.004 up.

## C. Manuscript (commit 70cd026; `MANUSCRIPT_v2_DRAFT.md`)
1. **Abstract (iii), one added sentence:** a test calibrated by planted effects *"rules out a drug-conditioned, gene-local effect
   explaining ≥ 0.5 % of the cell-specific response (linear in the encoded tracks)"*, and removing chromatin from trained models
   at inference raises their held-out-cell score by 0.006 [§91].
2. **§5.3, the old training-ablation null restated** per §91.10: "with one run per arm that rules out only effects larger than
   ≈ 0.006–0.016" (replacing "0.57 σ of run-to-run noise — no detectable effect").
3. **§5.3, two new paragraphs:**
   - "Chromatin, with a test calibrated before it was read": the tests, the calibration in both units, nothing advances, T1's
     informative negative in 91.12's wording, T2 uninformative, T3 not interpreted, and the gene-generic increment (+0.0014;
     N1 keeps all but 0.0002; N2 costs 0.0003).
   - "v9 reads its chromatin, and on unseen cells reading it costs accuracy": T4 +0.0060 every seed; centred +0.0056, so the cost
     is drug-specific; the three cells; VCAP = its failed track (93 %); HEK293T and LNCAP unexplained; diagnostic, not a method;
     P7 keeps chromatin as registered; ⏳ §92.
4. **§6, methods lesson 8:** "Plant an effect of known size before reading a null", with both floors in score units.
5. **§7, a limitation:** coverage (12 of 26 dev-training cells), gene-level ATAC / H3K27ac / H3K27me3 at landmark genes, and the
   negative covers linear forms on six dev cells.

## D. Figure 9 (rendered: model/figures/out/f9_chromatin.png, commit 30397f5)
- **a. The calibration (log–log):** T1 Δ against planted size for P1 and P2. Each draw is filled if it passed T1's rule and hollow
  if not. The bar, the real-data Δ and the nothing-planted band are drawn.
- **b. Closed-form gains over B0:** FB, FC, FBC, N1, N2.
- **c. T4 against T4b per dev cell:** marks listed, failed marks derived from the two encodings, per-seed dots.

No verdict text appears on the figure. Its numbers come from `model/v9/t4_per_cell.py` (PI glue, committed). That script asserts
the 91.11 values: T4 +0.00596 and per cell, T4b +0.00317, VCAP +0.0133, HEK293T −0.0002.

- **Worker defect, disclosed:** W29's first render drew every panel-c cell one row below its label. Its value tests passed,
  and the PI caught it on the render. It is fixed, and make_f9 now asserts every bar's value against its own row label.

## ASKS
1. Is the score-unit addendum right, and are its two corrections the right size? Should 91.12's heading
   ("MDE ≤ 0.5 %") change?
2. Does any manuscript sentence say more than RESULTS 91.10–91.12 license? In particular:
   - the abstract sentence;
   - "the cost is drug-specific" (from centred ≈ raw);
   - "the kind of per-gene content a gene embedding can represent".
3. Does Figure 9 mislead anywhere? For example: the log–log axes, hollow vs filled markers, or panel c's ordering.
   - In particular: panel a draws the real-data Δ (+0.0014) on the planted-effect axes. A reader could read off an implied
     effect size (≈ 2 % of a drug-specific shift). But panel b shows that the real increment is gene-generic (N1 ≈ FBC), so it is
     not a planted-form effect. Keep the line, label it differently, or drop it?
