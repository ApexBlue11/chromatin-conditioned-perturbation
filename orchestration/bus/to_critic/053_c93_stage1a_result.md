# PACKET 053 — RESULT: §93 Stage 1a (H2, H3) read once (RESULTS 93.7)
packet_id: 053
created: 2026-10-06
repo_commit: 45496f1
type: **RESULT** (no GPU; two free Kaggle CPU kernels)

## A. What ran
- **The kernels:** `lincs-chromatin93-h2` (4,185 s) and `lincs-chromatin93-h3` (34,002 s).
  - Each was built by `orchestration/make_chromatin93_kernels.py` at d4b3a4e's pins and ran on the uploaded
    `lincs-chromatin-funnel` version.
  - Log line 1 of each: `inputs verified: 10 pinned files + the split bundle; lightgbm 4.6.0`. The local tests used 4.7.0.
  - The H3 log repeats one sklearn `UserWarning` ("X does not have valid feature names, but LGBMRegressor was fitted with
    feature names"). Fit and predict both take `make_dataset`'s ndarray (`chromatin_gbm.py:42–67`, `:151`, `:169`).
  - H3 took 9.4 h against 93.6's ≈ 1 h estimate. The fit count was as estimated: 31 runs × 39 fits = 1,209.
- **The read:** each marker's output sha1 was verified by `read_chromatin93.py` (`d90fa910…`, `faef65c3…`), then each output
  was read once.
  - **Readings:** `model/results/chromatin93_h2_reading.json`, `chromatin93_h3_reading.json`.
  - **Raw outputs:** `model/results/chromatin_h2_93.json`, `chromatin_gbm_93.json` (sha1s equal the markers').

## B. The numbers (drug-known dev rows unless stated)
**H2** (medians over 5 subsets per k; FBC − N1 is read, FBC − FB is reported):

| k | FBC − N1 | FBC − FB | planted P1 0.5 %, FBC − N1 (3 draws × 5 subsets) |
|---|---|---|---|
| 4 | −0.00048 | +0.00114 | +0.00422 |
| 6 | −0.00012 | +0.00117 | +0.00371 |
| 8 | +0.00013 | +0.00141 | +0.00403 |
| all | +0.00023 | +0.00144 | +0.00386 |

- **The k = 4 subsets:** sd 0.00115.
- **The rise:** Δ(all) − median(k = 4) = +0.00071; the curve is monotone.
- **The harness check:** k = all equals 91.12 (`k_all_reproduces_91` true).
- **The reader's output:** NOT RISING.

**H3 calibration** (3 draws; T1 bars plus the N1 magnitude conjunct):

| case | passes | raw Δ C − B, range over draws | C − N1, range |
|---|---|---|---|
| null | 0 / 3 | −0.0007 to +0.0005 | −0.0044 to −0.0021 |
| P1 0.5 % | 3 / 3 | +0.0217 to +0.0222 | +0.0094 to +0.0123 |
| P1 2 % | 3 / 3 | +0.0683 to +0.0711 | +0.0354 to +0.0471 |
| P1 5 % | 3 / 3 | +0.1239 to +0.1307 | +0.0695 to +0.0840 |
| P2 0.5 / 2 / 5 % | 0 / 3 at each | ≤ +0.0031 | −0.0060 to −0.0026 |
| P3 5 % | 0 / 3 | +0.1302 to +0.1434 | +0.0747 to +0.0823 |
| G 2 % | 0 / 3 | +0.0295 to +0.0336 | −0.0322 to −0.0259 |
| G 5 % | 0 / 3 | +0.0503 to +0.0558 | −0.0523 to −0.0431 |

- **Faults, drug-known rows:** VOID_null, VOID_P3 and VOID_G all false.
- **MDEs:** MDE_P1 = 0.005, the smallest π planted. MDE_P2 = none.
- **All rows (reported only):** **VOID_P3 is true**; MDE_P1 = 0.005.

**H3 real run:**
- **C − B:** S(C) − S(B) = +0.00188; top tercile +0.00042; > 0 in 3 of 6 cells.
  - Per cell: HEK293T +0.0037, HL60 −0.0005, LNCAP −0.0015, SKBR3 +0.0037, U937 +0.0033, VCAP −0.0012.
- **Centred:** +0.00300, > 0 in 4 of 6.
- **C − N1:** S(C) − S(N1) = −0.00339; C > N1 in 0 of 6 cells.
  - Per cell: HEK293T −0.0025, HL60 −0.0105, LNCAP −0.0045, SKBR3 −0.0029, U937 −0.0024, VCAP −0.0033.
  - Centred: −0.00464, 1 of 6.
- **min_child_samples chosen:** FB 50, FBC 50, N1 200.
- **The reader's output:** DOES NOT ADVANCE, labelled *"informative null for the gain form (MDE ≤ 0.005; non-linear learner, these
  features, these dev cells)"*, with `G_check_informative` true.
- **All rows:** does not pass.

## C. What was controlled
- **Shared with §91:** the same rows of record, encoding, μ^(−c), bars and pinned inputs.
- **H3:** the same 5 % (row, gene) sample (rng 9301) and the same folds (rng 9302) in every arm and case. N1's fold features
  came from the fold's training cells.
- **H2:** the same subsets for the real and planted curves.

## D. Prior retractions in scope
- §93.1's VCAP "imputed ATAC" (review 050 C1).
- The funnel's cross-scale comparison (review 045 C2).

## ASKS
1. Are both readings mechanical, per 93.2, 93.5 C2–C4 and 93.6?
2. Does the all-rows VOID_P3 need any action, given that the drug-known rows bind?
3. Are 93.7's interpretation items 1–4 licensed as written?
   - Item 2 is the sentence *"gene-generic chromatin beats the cell's own, in every dev cell"*, with its two readings: no
     transferable cell-specific content, or noise.
   - Item 4 concerns what this implies for H1 and the principal's direction.
4. Is the H3 runtime note (bench timed the fit alone) an adequate disclosure?
5. Before Stage 1b's test is pre-registered in code: should H1 add anything because of this result? For example, a
   denoising control (a cell's own features shrunk toward N1) to separate item 2's two readings.
