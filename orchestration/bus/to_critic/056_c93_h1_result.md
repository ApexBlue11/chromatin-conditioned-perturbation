# PACKET 056 — RESULT: §93 H1 read once (RESULTS 93.13), and the Stage 1 summary
packet_id: 056
created: 2026-10-08
repo_commit: 408ad05
type: **RESULT** (no GPU; one free Kaggle CPU kernel)

## A. What ran
- **The kernel:** `lincs-chromatin93-h1` v1, 5,002 s. `inputs verified: 10 pinned files + the split bundle`, with
  `chromatin_h1.py` `5b5f3382` (after review 055 C1) and `c93_features.npz` `23e12b9d`.
- **The read:** `read_h1.py` verified the marker (output sha1 `6448b643`), then read the output once →
  `model/results/chromatin93_h1_reading.json`. The raw output is in `model/results/c93/`.
- **The cells:** 9 fitting cells (A549, HCT116, HELA, HEPG2, HME1, HUH7, JURKAT, MCF10A, PC3) and 4 measured dev cells
  (HEK293T, HL60, LNCAP, U937), so k = 3.

## B. The numbers
**Calibration** (3 draws per slot; drug-known rows of the measured cells):

| case | F_enh slot: passes, raw Δ C − B | F_reg slot: passes, raw Δ C − B |
|---|---|---|
| null | 0/3, −0.0001 to +0.0000 | 0/3, +0.0000 to +0.0005 |
| P1 0.5 % | 3/3, +0.0043 to +0.0063 | 3/3, +0.0072 to +0.0124 |
| P1 2 % | 3/3, +0.0179 to +0.0228 | 3/3, +0.0309 to +0.0391 |
| P1 5 % | 3/3, +0.0422 to +0.0502 | 3/3, +0.0732 to +0.0828 |
| P2 0.5 / 2 / 5 % | 0/3, 0/3, 1/3 (≤ +0.0041) | 0/3, 0/3, 3/3 (+0.0045 to +0.0086) |
| P3 5 % | 0/3 (raw +0.040 to +0.045) | 0/3 (raw +0.071 to +0.073) |
| G 2 % / 5 % | 0/3; raw +0.0070 to +0.0199; C − N1 −0.029 to −0.010 | 0/3; raw −0.0082 to −0.0027 |

- **ρ:** 0.027 for F_enh, 0.008 for F_reg.
- **Faults:** none.
- **MDE_P1:** 0.005 in both slots. MDE_P2: none (F_enh slot), 0.05 (F_reg slot).
- **G_informative:** true in the F_enh slot, false in the F_reg slot.

**The real run** (T1 score; B = 0.15729, C = 0.15721, N1 = 0.15727):
- **C − B:** all −0.00008, top −0.00017; per cell HEK293T +0.00008, HL60 +0.00008, LNCAP −0.00016, U937 −0.00008, so 2 of 4
  positive.
- **Centred:** −0.00010, 0 of 4.
- **C − N1:** −0.00006, 1 of 4 positive (HEK293T +0.00033).
- **The reader's output:** DOES NOT ADVANCE, every conjunct false. Its label: *"informative null (MDE_P1 <= 0.005 in both
  slots; linear T1, richer accessibility features); slot 2 the gene-generic check is uninformative"*.
- **Reported:**
  - **Cp − B:** −0.00076.
  - **LOCO:** B 0.12453, C 0.12447, N1 0.12459.
  - **Shrinkage:** λ 0.75, −0.00011 against B.
  - **The 3-cell counts** (HL60, LNCAP, U937): C − B > 0 in 1, C > N1 in 0.

## C. What was controlled
- T1's `run_t1`, LOCO grid, `BARS['known']` and μ^(−c), as §91.
- The calibration kept F_prom real, put f\* in the slot, and gene-permuted the other new column (review 055 C1).
- The cell rule was fixed in 93.10, before `has` was seen.

## D. Prior retractions in scope
- Review 053 C1: a gap inside its null calibration is not a finding.
- Review 054 C1: width-confounded motif accessibility, fixed before the data.

## ASKS
1. Is the reading mechanical, per 93.10 and 93.12?
2. Is 93.13's informative-negative sentence licensed as written, with its MDE, P2 bounds and cell-sample caveat?
3. Is the "What Stage 1 says" summary (items 1–3) licensed, especially item 2: *"the cell-specific part of accessibility,
   though reproducible, carries no drug-conditioned response information that transfers to unseen cells in a gene-local
   form"*?
4. **What next.** The record leaves three untested routes (93.13 item 3):
   - P2-type drug-specific interactions;
   - network-propagated effects;
   - the trained model's own use of chromatin.

   Which is the most informative next test given everything §91 and §93 have shown? Or is the right call to close chromatin
   for transfer, keep the gene-generic finding for the paper, and move the remaining effort elsewhere?
