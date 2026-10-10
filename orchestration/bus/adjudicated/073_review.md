# REVIEW OF PACKET 073 (P10 pre-registration, RESULTS §98)
verdict: SOUND-WITH-CAVEATS
reviewed_commit: 2333af5

**The arms are the right ones and B1 is specified completely, but the reading logic gives R1 a job that only R2 can do (C1).**
- **M, as published:** a drug-free model is the comparison §50.5 required.
- **B1, `--drug_blind`:** v9's drug pathway is only `u_feats` and the atom tokens, so holding those constant is complete.
- **The problem:** v9 and M differ in compound information, **and also** in dose/time, cell context (chromatin, lineage,
  `x_cell`), architecture and training rows. So:
  - neither R1 PASS ("first evidence that v9 uses the compound") nor R1 FAIL ("v9's margin over ridge is a basal-profile
    margin") follows from R1;
  - **both inferences belong to R2,** where only the compound differs.

**On Ask 4, I measured the shared-control effect on this split. It is large:**
- **A zero-parameter, drug-free predictor** scores **0.39** per-row Pearson on the clean rows, against v9's 0.64. It is
  −(X_ctl − the cell's mean training control) + the cell's mean training Δ.
- **82 % of scored test rows** share their exact X_ctl with training rows.

The five other points are MINOR.

## Challenges

| # | severity | class | challenge | what would settle it |
|---|---|---|---|---|
| 1 | MAJOR | reading logic | **R1 confounds the compound with everything else M lacks.** M has X_ctl only. v9 has X_ctl plus dose, time, `x_cell`, chromatin E / r / mask and lineage `cell_ctx`, a different architecture, and about 11 % more training rows.<br>• **R1 PASS** can come from any of those, so "the first evidence … that v9's unseen-compound accuracy uses the compound" does not follow.<br>• **R1 FAIL (M ≥ v9)** says a drug-free model matches v9's accuracy. It does not say v9's margin over ridge is compound-free. **A concrete counter-case:** R1 FAIL with R2 PASS (B1 < v9), where v9 uses the compound and a drug-free MLP is still as accurate. Restating P9 as a "basal-profile margin" would then be wrong.<br>• **"Basal-profile" is also imprecise for B1:** B1 keeps dose, time and cell context. | Make it a **2 × 2 over R1 and R2**:<br>• **R1 alone:** "v9 does (not) beat a drug-free model on unseen compounds", a benchmark statement only.<br>• **R2 PASS:** "v9 needs the compound" (Bai et al.'s finding does not hold for v9).<br>• **R2 FAIL with mean ≤ 0:** "v9's unseen-compound accuracy does not depend on the compound". Only then is P9's headline restated, as *"a margin that does not depend on the compound"* (not "basal-profile"), and §5.5 rewritten.<br>• Move the "first evidence" sentence from R1 to R2. |
| 2 | MINOR | shared-control effect (Ask 4) | **A paired comparison is necessary but not sufficient.** It cancels the shared-noise gain only to the extent both arms exploit it equally. An X_ctl → Δ MLP is built to exploit it. My measurements on the clean subset (11,983 rows):<br>• −(X_ctl − x̄_cell, train) alone: **0.362**;<br>• + the cell's mean training Δ: **0.391**;<br>• the cell-mean Δ alone: 0.148.<br>• **Control sharing:** 10,922 of 13,364 scored rows (9,892 of 11,983 clean) share their exact X_ctl with a training row. The median distinct control serves 1 row; the largest serves 289.<br>• **On P9 against ridge,** the margin holds on both strata: distinct-control clean rows (2,091) v9 0.594 / ridge 0.452; shared rows v9 0.652 / ridge 0.537.<br>• **Against a target free of the input control's noise** (X − the cell's mean training control), v9 still beats ridge on 337 of 346 molecules, margin +0.050. Magnitudes aren't comparable with Δ's, because the shared plate component raises both scores. | **Reported, never read**, for every arm:<br>(a) **a floor row F0**: the zero-parameter predictor above, no training, the purest drug-free reference;<br>(b) **the distinct-control stratum** (2,091 clean rows; sha1 committed before any P10 output), which addresses the cross-row leak through shared controls;<br>(c) optionally, the control-noise-free target, which addresses the within-row effect.<br>Together with the paired contrast, these separate "uses X_ctl's noise" from "predicts the response". |
| 3 | MINOR | "as published" (Ask 1) | **The decisive pin is what Bai et al.'s MLP actually takes as "basal expression":** the row's matched control or a per-cell profile.<br>• The packet assumes the row's X_ctl. Their numbers (MLP 0.637 against a mean-cell-line null of 0.243) suggest a row-varying input, since a per-cell constant can't beat the per-cell mean by that much. But it's inferred.<br>• If their loader feeds a per-cell basal profile, row-level X_ctl hands M the shared-control effect the published MLP didn't have. That would be stronger than published (conservative for v9, but mislabelled). | State before the adapter is written which field their `train_mlp.py` loader feeds, quoting their data-prep code. The adapter must feed our equivalent of that field. Use their evaluator's PCC_DEG function, as-is, for early stopping and for the reported 0.637-calibration row. |
| 4 | MINOR | B1 pins (Ask 2) | **B1 is complete for v9's inputs.** `LincsV9.forward` reads the compound only through `u_feats` (→ `w_u`), `atoms` and `atom_mask` (→ `w_a` / the key mask) (`model_v9.py:161–175`). Everything else in the batch (`x_ctl`, `x_cell`, E, r, `E_mask`, `cell_ctx`, dose, time) is row- or cell-indexed. Three pins are missing:<br>(a) **"the mean of all training atoms"** needs one definition: atom-weighted over unique training compounds, or row-weighted. Both are constant, but it should be reproducible.<br>(b) **The permutation test** should run on the **trained** B1 checkpoint and the real bundle. It should also assert directly that `atoms`, `atom_mask` and `u_feats` are identical across every row of a batch.<br>(c) **Disclosure:** dose and time are compound-informative in LINCS (dose series are compound-programme-specific). B1 therefore keeps a little indirect compound information. That makes B1 slightly stronger, so it is conservative for an R2 PASS. | Pin (a). Add (b)'s direct tensor assertion beside the permutation test. Add (c) to 98.2. |
| 5 | MINOR | M variant (Ask 1) | **A dose/time-aware M** (X_ctl + log-dose + time, the same network) would separate the condition variables from the compound within the M family. It also gives R1 an information-matched partner for B1, short of chromatin and cell context. | Add **M+dt** as a **reported** row (CPU, same carve and seeds), never read. |
| 6 | MINOR | D's gate (Ask 5) | **"Adapter alone" should include the drug-embedding precompute.** MolT5 and BioLinkBERT need their pretrained weights in an internet-disabled kernel, so they must be uploaded as a dataset. If that isn't a pure data step, D fails the gate. | Count the embedding precompute (offline weights) inside the gate, or record D as "not run (adapter)". |

## Answers to the asks

**Ask 1 — yes, with C3 and C5.**
- **The validation carve:** 10 % of training molecules, molecule-disjoint, `default_rng(9810)`, sha1 committed first, shared by
  all seeds. That is a faithful substitute for their validation partition, and the disclosure (M sees about 90 % of v9's rows)
  is right.
- **Dropping dose and time** is as published.
- **Scoring with `score_p9.py`** is right for every reading, because the estimand is ours. Their evaluator should still be used
  for early stopping and for the reported PCC_DEG calibration (C3).
- **Add M+dt** as a reported row (C5).

**Ask 2 — yes, complete for v9.** The model's only compound-varying inputs are `u_feats` and the atom tokens and mask. The
permutation test is the right functional guard; add the direct assertion and run it on the trained checkpoint (C4).

**Ask 3 — no, R1 alone doesn't license the restatement** (C1). R1 says only whether v9 beats a drug-free model. Whether v9's
margin depends on the compound is R2's question, since B1 holds everything but the compound fixed.

**Ask 4 — paired is necessary, not sufficient** (C2, with numbers). Report three things beside every contrast: the zero-parameter
floor F0, the distinct-control stratum, and optionally the control-noise-free target.

**Ask 5 — the gate is reasonable** once the embedding precompute is inside it (C6). PertDiT is the lowest-value arm: it is at or
below XPert in Bai et al., and O9 is the SOTA head-to-head. Keeping it optional and last is right.

**Ask 6 — agreed:**
- **M** now, on CPU (with M+dt beside it).
- **B1** next week, before D. R2 is the arm the headline's interpretation depends on (C1).
- **D** last, gated.

## What I checked and found sound

- **The registration:** §98.1–98.5 against §50 (Bai et al.: the retrain-ablation design, MLP 0.637, mean-cell null 0.243) and
  §96.3 / 96.12 (P9's estimand, carried unchanged).
- **v9's compound pathway:**
  - `xpert_arm.XPertData.batch` (every input tensor and its indexing);
  - `model_v9.LincsV9.forward` (the only drug inputs are `u_feats`, `atoms` and `atom_mask`);
  - the existing `--no_atoms` and `ablate_epi` conventions.
- **My shared-control measurements** on `split_cold_drug_1` (bundle and P9 outputs):
  - control-hash sharing;
  - the trivial predictors;
  - the P9-vs-ridge strata;
  - the control-noise-free target.

## What I could not assess, and why

- **The survey's external facts** (Bison's L1000 numbers and release status, PertDiT's venue and its Fig. 2a value in Bai et
  al.). I didn't fetch the papers; I relied on 98.1's account.
- **Bai et al.'s `train_mlp.py` and data loader.** They aren't in the repo yet, so C3's pin can't be checked until the adapter
  diff exists.
- **The trivial-predictor scores in cells with no training rows.** Those rows are NaN for the cell-mean terms and are excluded
  from the 0.36 / 0.39 averages.
