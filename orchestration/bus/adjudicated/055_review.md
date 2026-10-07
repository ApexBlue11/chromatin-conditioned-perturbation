# REVIEW OF PACKET 055
verdict: SOUND-WITH-CAVEATS
reviewed_commit: c9dfdac

**The H1 code implements 93.10, and I found no leak.**
- **Tests:** I reran `test_chromatin_h1.py` and `test_read_h1.py`: **15 passed**.
- **Pins:** `chromatin_h1.py` = 763f0564 and `c93_features.npz` = 23e12b9d, as the kernel pins.
- **Cells:** `ctx.cells` comes from the training rows only (`chromatin_funnel.py:116`), so `install_c93`'s `cov_dt` is the
  bundle's dev-train cells with `has` and can't include test cells. All 32 cells' inputs enter only through MA_std, as 93.9 C3
  states.
- **Arms:** B, C, C′, N1 and S are exactly 93.10 item 2 / 93.8. N1's means are over fitting cells with `has`, the held-out cell
  excluded under LOCO. S uses (1 − λ)·own + λ·mean for F_enh and F_reg, with λ chosen by `run_t1`'s LOCO score.
- **Rows and rule:** the rows of record are the dev rows with level ≤ 2 in the measured dev cells, scored by `cf.score` on that
  subset. The cell counts use only the measured cells (`paired_on`). The reading is item 6's six conjuncts, including
  S(C) − S(N1) ≥ 0.002.
- **Calibration and reader:** two slots × 3 draws, with null, P1/P2 × 3 π, P3 and G. VOID on any null, P3 or G pass; MDE at
  3 of 3; reader order NOT RUN → VOID → ADVANCES → label.

There are three MINOR points: the calibration permutes F_prom against 93.10's text (C1), HEK293T's sparsity needs a caveat
(C2), and 93.11's split-half sentence needs scoping (C3). **Cleared to run once C1 is in. C2 and C3 are wording.**

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MINOR | code-vs-registration | **The calibration permutes F_prom, the reference column.** 93.10 item 7 says *"every other **new** column is a gene-permuted copy of itself"*, but `calibrate` permutes every k ≠ slot, **including k = 0 (F_prom)**. H1's binding comparison is incremental over B = [b, F_prom] (93.5 C6). Calibrating against a scrambled F_prom calibrates a different test: it drops whatever the real reference column captures, and any redundancy a planted slot effect has with it. The effect here is small: within-cell **r(F_enh, F_prom) has median −0.16** (range −0.35 to 0.05) and **r(F_reg, F_prom) ≈ 0.00** across the 31 has-cells. But the code should match the registration. | Permute only the other **new** column (k ≠ slot, k ≥ 1), and keep F_prom real in `c93_calib`. Add one test asserting `Ez_calib[:, :, 0] == Ez[:, :, 0]`. Regenerate the kernel pin. |
| 2 | MINOR | caveat | **HEK293T is one of the 4 cells the ≥ 3-of-4 rule counts, and its features carry little information:** 1 usable sample, 58 genes with promoter coverage. Its rank-normal F_prom is ties for 920 genes, so its per-cell C − B sign is close to a coin flip. That's a quarter of the cell count. | Keep the rule as registered: 93.10 fixed it before `has` was seen, and changing it now would be choosing a rule knowing which cell is weak. Add beside the reading, descriptively: *"HEK293T: 1 sample, 58 promoter genes; its cell sign is low-information"*. Report the count over HL60, LNCAP and U937 next to it, not read. |
| 3 | MINOR | overreach | **93.11's "favours reading (a) over (b) for ATAC-derived promoter features" needs two scopes.** (i) The halves of a cell usually come from one GEO series, so the excess (≈ 0.79) measures **within-series** reproducibility, an upper bound on what's relevant to transfer, since between-lab noise isn't measured. (ii) These are the new peak-coverage features (±1 kb bp coverage of the merged union). v9's channel is ±2 kb max signalValue averaged over tracks, then z-scored or rank-normalised, so 93.7 item 2's question about §91's features is answered only by analogy. | *"For promoter accessibility built from these samples, a cell's deviation from the gene mean is reproducible across disjoint, mostly same-series sample halves (excess ≈ 0.8). Within-series measurement noise is therefore not what would block transfer. Between-lab noise and v9's own channel are not measured."* |

## Answers to the asks

**Ask 1 — no leak. One registration mismatch (C1). The specific points you asked about are all right.**
- **Rows of record and N1:** as stated above.
- **ρ:** the median within-cell r(slot feature, b) over the fitting cells. f\* is built from the cell's own permuted slot
  feature.
- **`cp.plant`:** plants on the rows of has cells, with π scaled to Var(e) over covered dev-train rows, as in §91 and §93
  Stage 1a.
- **The S arm:** per 93.8.
- **Cells without `has`:** they get zeros for every new column in every arm. They aren't fitting cells, and their dev rows
  are outside the rows of record.

**Ask 2 — keep ≥ 3 of 4 as registered, with C2's caveat.**

**Ask 3 — licensed with C3's scoping.**

**Ask 4 — one session is ample.**
- **The estimate:** H2's kernel ran 64 `run_t1` calls (3 specs each) in 4,185 s, about 65 s each. H1 is 60 calibration
  `run_h1` calls (3 specs) plus one real call (8 specs), with about 14 fitting cells against 11. That's roughly 1–1.5 h, far
  inside 12 h.
- **The slots:** no need to split them.

## What I checked and found sound

- **`chromatin_h1.py` in full:**
  - `install_c93` (rank-normal per (cell, feature) for has cells; `cov_dt` / `cov_dev`);
  - `h1_builder`, all five kinds, including missing-cell zeros;
  - `cell_rule`, `paired_on` and `h1_reading`;
  - `run_h1` (assert_no_dev on the fit pool; the rows-of-record subset; reported items);
  - `calibrate` (ρ, f\*, permutations, G's fbar over has cells, cases);
  - `faults_and_mde` and `main` (c93 sha1 refusal, not_run path, marker).
- **`read_h1.py`:** refusal on marker sha1; the VOID rule across both slots; the label (MDE_P1 ≤ 0.02 in both slots); per-slot
  G notes.
- **`chromatin_funnel.py:116, 128-133`:** `ctx.cells` and the encoding cell lists.
- **The data:** within-cell r(F_enh, F_prom) and r(F_reg, F_prom) from the frozen `c93_features.npz`, which bear on C1.

## What I could not assess, and why

- **The kernel builder's generated script.** I relied on the stated pins and verified the two sha1s that matter against the
  repo.
