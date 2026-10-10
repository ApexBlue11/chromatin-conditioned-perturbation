# PACKET 072 — RESULT: Stage B′ (RESULTS 96.14), and review 071's corrections to 96.13
packet_id: 072
created: 2026-10-10
repo_commit: f57389c
type: **RESULT**

## A. Stage B′ (96.14), read mechanically by `read_b_prime`
- **The runs:**
  - B3 via `mechanism_stage_a.py --split split_cold_drug_1 --rows v9p9_seed0.npz --delta <spec>`, for the four `deg_pred`
    specs;
  - `stage_b_prime.py --compare` against each of the four references;
  - `--read --p9_score p9_accuracy_ridge.json`.
  - **The guards:** every input guard passed (key `deg_pred`, `deg_pred` = `y_pred − ctl_true`, the references' sha1s and
    markers, the seed sha1s across the P9 manifest, comparisons and disk, the unit sets, the equivalence with `run_a3`).
- **Item 1:** v9's T is 3.35 / 3.57 / 3.13 / 3.38, p 0.001, 8 of 9 units positive, 2 classes. It **passes** in all four variants.
- **Item 2,** the excluding set of record (z, sign-flip swap with 10,000 draws):
  - **v9 T:** 1.53 / 1.48 / 1.30 / 1.44;
  - **1-NN** (T_R 0.92): p 0.137 / 0.147 / 0.238 / 0.167;
  - **5-NN** (1.15): 0.030 / 0.078 / 0.243 / 0.094;
  - **physchem** (0.91): 0.020 / 0.030 / 0.123 / 0.044;
  - **ridge** (1.30): 0.194 / 0.224 / 0.502 / 0.279.
  - **No reference passes in all four variants.**
- **The reading:** **B3_ONLY**, *"B3 holds; no beyond-chemistry claim"*.
- **Reported:**
  - **Per class against ridge:** DNA favours v9 and EGFR@MCF7 favours ridge.
  - **The including reading:** below ridge (p > 0.98).
  - **Raw:** no pass.
  - **Seed 2** is the weakest variant throughout.

## B. 96.13, corrected per review 071 (all four verified by me before editing)
- **C1:** the sentence names fold 1.
- **C2:** the strata, with per-model levels: v9 0.636 / 0.648 / 0.611 and ridge 0.505 / 0.545 / 0.528, molecule means of per-row
  scores. Mine use a slightly different molecule statistic from yours, with the same pattern: the margin is mainly ridge
  falling.
- **C3:** the duplicate lift, both ways: molecule-weighted v9 +0.030 and ridge +0.055, DiD −0.025, as you found; with the hedge.
- **C4:** the six cells are erlotinib alone (verified from the bundle: the panel profiled afatinib + erlotinib only).
- **Your centred diagnostic** is recorded as a post hoc check, not registered.

## ASKS
1. Is the Stage B′ reading mechanical and right? A rescore of one or two comparisons with independent code would be welcome.
2. Is 96.14's "what it means" within the registration (4 units, 6 compounds), and does any wording overreach?
3. Are the 96.13 corrections right?
